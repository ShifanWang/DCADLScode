#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOWS2 beta MST test (MST-1 + MST-3) with progress bars.

Implements:
- MST-1: Fit power-law exponent beta from log-log regression of mean KL vs alpha.
- MST-3: Split images into 3 texture bins (low/mid/high) using residual energy and fit beta per bin.

Outputs:
- CSV table: table_alpha_kl_auc_beta_mst.csv
- Summary: summary_beta_mst.json, summary_beta_mst.txt
- Figures: kl_loglog_with_fit_region.png, rs_auc_logx.png, beta_by_texture_bins.png
"""

import os, glob, argparse, csv, json
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# -----------------------------
# IO
# -----------------------------
def list_images(root: str):
    exts = ("*.pgm", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root, "**", e), recursive=True))
    files = sorted(files)
    if len(files) == 0:
        raise FileNotFoundError(f"No images found under: {root}")
    return files

def read_grayscale_u8(path: str) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 3:
        img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).round().astype(np.uint8)
    img = img.astype(np.uint8)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape={img.shape} for {path}")
    return img


# -----------------------------
# Utility: conv2d same size, reflect padding
# -----------------------------
def conv2d_same_reflect(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    h = h.astype(np.float32)
    kh, kw = h.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += h[i, j] * xp[i:i + x.shape[0], j:j + x.shape[1]]
    return out

def shift_same(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    H, W = a.shape
    pad_y = abs(dy)
    pad_x = abs(dx)
    ap = np.pad(a, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    y0 = pad_y + dy
    x0 = pad_x + dx
    return ap[y0:y0 + H, x0:x0 + W]


# -----------------------------
# Embedding primitives
# -----------------------------
def uniform_lsb_flipping(img_u8: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    N = img_u8.size
    k = int(np.floor(alpha * N))
    if k <= 0:
        return img_u8.copy()
    flat = img_u8.reshape(-1).copy()
    idx = rng.choice(N, size=k, replace=False)
    flat[idx] = np.bitwise_xor(flat[idx], 1).astype(np.uint8)
    return flat.reshape(img_u8.shape)


# -----------------------------
# Structural cost rho_{i,j} based on residuals
# -----------------------------
def structural_cost_rho(img_u8: np.ndarray, h: np.ndarray) -> np.ndarray:
    X = img_u8.astype(np.float32)
    R = conv2d_same_reflect(X, h)

    # XOR1 delta: even->+1, odd->-1
    delta = (1.0 - 2.0 * (img_u8 & 1).astype(np.float32))

    kh, kw = h.shape
    ph, pw = kh // 2, kw // 2
    rho = np.zeros_like(R, dtype=np.float32)

    for iu in range(kh):
        for jv in range(kw):
            coef = float(h[iu, jv])
            if coef == 0.0:
                continue
            dy = iu - ph
            dx = jv - pw
            R_shift = shift_same(R, dy, dx)
            Rp = R_shift + delta * coef
            rho += np.abs(np.abs(Rp) - np.abs(R_shift))

    return rho

def structural_adaptive_lsb_flipping(img_u8: np.ndarray, alpha: float, h: np.ndarray,
                                    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Deterministic 'pick k smallest rho'.
    Optionally break ties with tiny noise using rng (for stability checks).
    """
    N = img_u8.size
    k = int(np.floor(alpha * N))
    if k <= 0:
        return img_u8.copy()

    rho = structural_cost_rho(img_u8, h).reshape(-1).astype(np.float64)

    if rng is not None:
        # tiny jitter to break ties without changing ordering materially
        rho = rho + 1e-12 * rng.standard_normal(rho.shape)

    chosen = np.argsort(rho)[:k]
    flat = img_u8.reshape(-1).copy()
    flat[chosen] = np.bitwise_xor(flat[chosen], 1).astype(np.uint8)
    return flat.reshape(img_u8.shape)


# -----------------------------
# KL on epsilon-smoothed histograms (plugin)
# -----------------------------
def hist_eps(img_u8: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    h = np.bincount(img_u8.reshape(-1), minlength=256).astype(np.float64)
    p = h / img_u8.size
    p = np.clip(p, eps, 1.0)
    p /= p.sum()
    return p

def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * (np.log(p) - np.log(q))))


# -----------------------------
# RS score (fast vectorized version)
# -----------------------------
def rs_score_fast(img_u8: np.ndarray, group_size: int = 4, mask=None) -> float:
    if mask is None:
        mask = np.array([1, 0, 1, 0], dtype=np.int8)
    mask = np.asarray(mask, dtype=np.int8)
    if mask.size != group_size:
        raise ValueError("mask length must equal group_size")

    flat = img_u8.reshape(-1)
    n_groups = flat.size // group_size
    data = flat[:n_groups * group_size].reshape(n_groups, group_size).astype(np.int16)

    def discrim(g):
        return np.sum(np.abs(g[:, 1:] - g[:, :-1]), axis=1)

    f0 = discrim(data)

    even = (data % 2 == 0)

    gF1 = data.copy()
    gF1[even] += 1
    gF1[~even] -= 1
    np.clip(gF1, 0, 255, out=gF1)

    gFm1 = data.copy()
    gFm1[even] -= 1
    gFm1[~even] += 1
    np.clip(gFm1, 0, 255, out=gFm1)

    gM = data.copy()
    gN = data.copy()

    pos = (mask == 1)
    neg = (mask == -1)

    if np.any(pos):
        gM[:, pos] = gF1[:, pos]
        gN[:, pos] = gFm1[:, pos]
    if np.any(neg):
        gM[:, neg] = gFm1[:, neg]
        gN[:, neg] = gF1[:, neg]

    fM = discrim(gM)
    fN = discrim(gN)

    Rm = np.mean(fM > f0); Sm = np.mean(fM < f0)
    Rn = np.mean(fN > f0); Sn = np.mean(fN < f0)

    return float(-(abs(Rm - Rn) + abs(Sm - Sn)))


# -----------------------------
# Beta estimation helpers
# -----------------------------
def fit_beta_loglog(alphas: np.ndarray, y: np.ndarray, y_se: np.ndarray | None = None):
    """
    Fit log(y) = a + beta log(alpha) via (weighted) least squares.
    Returns beta, intercept, R2 (in log domain).
    """
    x = np.log(alphas)
    z = np.log(y)

    if y_se is None:
        w = np.ones_like(y)
    else:
        sigma_log = np.clip(y_se / np.maximum(y, 1e-300), 1e-12, np.inf)
        w = 1.0 / (sigma_log ** 2)

    W = np.sum(w)
    xbar = np.sum(w * x) / W
    zbar = np.sum(w * z) / W
    Sxx = np.sum(w * (x - xbar) ** 2)
    Sxz = np.sum(w * (x - xbar) * (z - zbar))

    beta = Sxz / Sxx
    a = zbar - beta * xbar

    zhat = a + beta * x
    ss_tot = np.sum(w * (z - zbar) ** 2)
    ss_res = np.sum(w * (z - zhat) ** 2)
    R2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    return float(beta), float(a), float(R2)

def bootstrap_beta(alphas: np.ndarray, per_image_y: np.ndarray, fit_mask: np.ndarray,
                   B: int, rng: np.random.Generator):
    """
    Bootstrap beta by resampling images.
    per_image_y: shape [M_images, K_alphas]
    """
    M, K = per_image_y.shape
    betas = []
    for _ in range(B):
        idx = rng.integers(0, M, size=M)
        yb = per_image_y[idx].mean(axis=0)
        yy = np.maximum(yb[fit_mask], 1e-300)
        aa = alphas[fit_mask]
        beta, _, _ = fit_beta_loglog(aa, yy, None)
        betas.append(beta)
    betas = np.array(betas, dtype=np.float64)
    lo, hi = np.quantile(betas, [0.025, 0.975])
    return float(np.mean(betas)), float(lo), float(hi)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bows2_dir", type=str, required=True)
    ap.add_argument("--M", type=int, default=2000, help="number of images sampled from BOWS2")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", type=str, default="out_beta_mst")
    ap.add_argument("--filter", type=str, default="laplacian4", choices=["laplacian4", "hp8"])

    # alpha grid
    ap.add_argument("--alpha_min", type=float, default=5e-4)
    ap.add_argument("--alpha_max", type=float, default=1e-2)
    ap.add_argument("--alpha_num", type=int, default=12)
    ap.add_argument("--alpha_grid", type=str, default="log", choices=["log", "linear"])

    # fit range
    ap.add_argument("--fit_max_alpha", type=float, default=-1.0,
                    help="max alpha used to fit beta; if <0, use 1/sqrt(N)")
    ap.add_argument("--fit_min_alpha", type=float, default=0.0,
                    help="min alpha used to fit beta")

    # stability
    ap.add_argument("--uniform_reps", type=int, default=1, help="repeat uniform embedding reps per image/alpha")
    ap.add_argument("--break_ties", action="store_true",
                    help="add tiny jitter to rho to break ties (uses rng)")

    # bootstrap
    ap.add_argument("--bootstrap", type=int, default=400)

    # RS
    ap.add_argument("--rs_group", type=int, default=4)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # high-pass filter H
    if args.filter == "laplacian4":
        H = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]], dtype=np.float32)
    else:
        H = np.array([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1]], dtype=np.float32)

    # Load images
    files = list_images(args.bows2_dir)
    if args.M > len(files):
        raise ValueError(f"M={args.M} exceeds dataset size {len(files)}")

    sel = rng.choice(len(files), size=args.M, replace=False)
    sel_files = [files[i] for i in sel]

    print(f"[Info] Loading {len(sel_files)} images ...")
    covers = [read_grayscale_u8(p) for p in tqdm(sel_files, desc="Load images", unit="img")]

    N = covers[0].size
    alpha_crit = 1.0 / np.sqrt(N)

    # alpha grid
    if args.alpha_grid == "log":
        alphas = np.exp(np.linspace(np.log(args.alpha_min), np.log(args.alpha_max), args.alpha_num))
    else:
        alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_num)
    alphas = np.array(alphas, dtype=np.float64)

    fit_max = alpha_crit if args.fit_max_alpha < 0 else args.fit_max_alpha
    fit_min = args.fit_min_alpha
    fit_mask = (alphas <= fit_max) & (alphas >= fit_min)

    print(f"[Info] M={args.M}, image size={covers[0].shape}, N={N}, 1/sqrt(N)≈{alpha_crit:.6f}, filter={args.filter}")
    print(f"[Info] alpha grid: {alphas}")
    print(f"[Info] fit range: [{fit_min:g}, {fit_max:g}] -> using {fit_mask.sum()} points")
    if fit_mask.sum() < 3:
        raise ValueError("Need at least 3 alpha points in fit range for stable beta estimation.")

    # texture score for MST-3: mean |H*X|
    print("[Info] Computing texture scores for MST-3 ...")
    tex = []
    for img in tqdm(covers, desc="Texture score", unit="img"):
        R = conv2d_same_reflect(img.astype(np.float32), H)
        tex.append(float(np.mean(np.abs(R))))
    tex = np.array(tex, dtype=np.float64)

    order = np.argsort(tex)
    thirds = np.array_split(order, 3)
    bin_names = ["low_texture", "mid_texture", "high_texture"]

    # storage per-image KL for bootstrap
    M = len(covers)
    K = len(alphas)
    KL_u_img = np.zeros((M, K), dtype=np.float64)
    KL_s_img = np.zeros((M, K), dtype=np.float64)

    # mean + SE (across images)
    KL_u_mean = np.zeros(K); KL_u_se = np.zeros(K)
    KL_s_mean = np.zeros(K); KL_s_se = np.zeros(K)

    # RS AUC per alpha
    AUC_u = np.zeros(K); AUC_s = np.zeros(K)

    # main loop with progress bars
    for t, a in enumerate(tqdm(alphas, desc="Alpha grid", unit="alpha")):
        # per-alpha RNG streams
        rng_u = np.random.default_rng(args.seed + 1001 + int(a * 1e9) % 1000003)
        rng_s = np.random.default_rng(args.seed + 2002 + int(a * 1e9) % 1000003)

        scores_u = []; labels_u = []
        scores_s = []; labels_s = []

        img_pbar = tqdm(
            enumerate(covers),
            total=len(covers),
            desc=f"Images @ alpha={a:.2e}",
            unit="img",
            leave=False
        )

        for i, img in img_pbar:
            pX = hist_eps(img)

            # uniform: optionally average over reps for stability
            kls_u = []
            rs_u = []
            for _ in range(args.uniform_reps):
                stego_u = uniform_lsb_flipping(img, float(a), rng_u)
                kls_u.append(kl(hist_eps(stego_u), pX))
                rs_u.append(rs_score_fast(stego_u, group_size=args.rs_group))
            KL_u_img[i, t] = float(np.mean(kls_u))

            # structural: deterministic + optional tiny tie-breaking jitter
            stego_s = structural_adaptive_lsb_flipping(
                img, float(a), H, rng=(rng_s if args.break_ties else None)
            )
            KL_s_img[i, t] = kl(hist_eps(stego_s), pX)

            # RS scores (cover score shared)
            s0 = rs_score_fast(img, group_size=args.rs_group)
            scores_u.extend([s0, float(np.mean(rs_u))]); labels_u.extend([0, 1])
            scores_s.extend([s0, rs_score_fast(stego_s, group_size=args.rs_group)]); labels_s.extend([0, 1])

            # live status
            if (i + 1) % 50 == 0 or (i + 1) == len(covers):
                img_pbar.set_postfix({
                    "KL_u_mean": f"{KL_u_img[:i+1, t].mean():.2e}",
                    "KL_s_mean": f"{KL_s_img[:i+1, t].mean():.2e}",
                })

        # per-alpha summary
        KL_u_mean[t] = KL_u_img[:, t].mean()
        KL_u_se[t] = KL_u_img[:, t].std(ddof=1) / np.sqrt(M)
        KL_s_mean[t] = KL_s_img[:, t].mean()
        KL_s_se[t] = KL_s_img[:, t].std(ddof=1) / np.sqrt(M)

        auc0_u = roc_auc_score(np.array(labels_u), np.array(scores_u))
        auc0_s = roc_auc_score(np.array(labels_s), np.array(scores_s))
        AUC_u[t] = max(auc0_u, 1.0 - auc0_u)
        AUC_s[t] = max(auc0_s, 1.0 - auc0_s)

        tqdm.write(f"[alpha={a:.6g}] KL_u={KL_u_mean[t]:.3e} KL_s={KL_s_mean[t]:.3e} | AUC_u={AUC_u[t]:.4f} AUC_s={AUC_s[t]:.4f}")

    # -----------------------------
    # MST-1: global beta
    # -----------------------------
    beta_u, a_u, R2_u = fit_beta_loglog(alphas[fit_mask], np.maximum(KL_u_mean[fit_mask], 1e-300), KL_u_se[fit_mask])
    beta_s, a_s, R2_s = fit_beta_loglog(alphas[fit_mask], np.maximum(KL_s_mean[fit_mask], 1e-300), KL_s_se[fit_mask])

    bmean_u, blo_u, bhi_u = bootstrap_beta(alphas, KL_u_img, fit_mask, args.bootstrap, rng)
    bmean_s, blo_s, bhi_s = bootstrap_beta(alphas, KL_s_img, fit_mask, args.bootstrap, rng)

    summary = {
        "N": int(N),
        "alpha_crit_1_over_sqrtN": float(alpha_crit),
        "alphas": [float(x) for x in alphas],
        "fit_min_alpha": float(args.fit_min_alpha),
        "fit_max_alpha": float(fit_max),
        "MST1_global": {
            "uniform": {"beta_hat": beta_u, "R2_loglog": R2_u, "bootstrap_mean": bmean_u, "ci95": [blo_u, bhi_u]},
            "structural": {"beta_hat": beta_s, "R2_loglog": R2_s, "bootstrap_mean": bmean_s, "ci95": [blo_s, bhi_s]},
        },
        "MST3_by_texture_bins": {}
    }

    # -----------------------------
    # MST-3: 3 bins by texture
    # -----------------------------
    for name, idx in zip(bin_names, thirds):
        idx = np.array(idx, dtype=int)
        mu_u_bin = KL_u_img[idx].mean(axis=0)
        mu_s_bin = KL_s_img[idx].mean(axis=0)

        beta_u_bin, _, R2_u_bin = fit_beta_loglog(alphas[fit_mask], np.maximum(mu_u_bin[fit_mask], 1e-300), None)
        beta_s_bin, _, R2_s_bin = fit_beta_loglog(alphas[fit_mask], np.maximum(mu_s_bin[fit_mask], 1e-300), None)

        # bootstrap CI within bin
        rng_bin = np.random.default_rng(args.seed + 999 + (abs(hash(name)) % 100000))
        def boot_bin(per_image_y):
            Mbin = per_image_y.shape[0]
            betas = []
            for _ in range(args.bootstrap):
                bidx = rng_bin.integers(0, Mbin, size=Mbin)
                yb = per_image_y[bidx].mean(axis=0)
                beta, _, _ = fit_beta_loglog(alphas[fit_mask], np.maximum(yb[fit_mask], 1e-300), None)
                betas.append(beta)
            betas = np.array(betas, dtype=np.float64)
            lo, hi = np.quantile(betas, [0.025, 0.975])
            return float(np.mean(betas)), float(lo), float(hi)

        bmu_u, blo_u2, bhi_u2 = boot_bin(KL_u_img[idx])
        bmu_s, blo_s2, bhi_s2 = boot_bin(KL_s_img[idx])

        summary["MST3_by_texture_bins"][name] = {
            "count": int(len(idx)),
            "texture_mean": float(tex[idx].mean()),
            "uniform": {"beta_hat": float(beta_u_bin), "R2_loglog": float(R2_u_bin), "bootstrap_mean": bmu_u, "ci95": [blo_u2, bhi_u2]},
            "structural": {"beta_hat": float(beta_s_bin), "R2_loglog": float(R2_s_bin), "bootstrap_mean": bmu_s, "ci95": [blo_s2, bhi_s2]},
        }

    # -----------------------------
    # Save CSV table
    # -----------------------------
    csv_path = os.path.join(args.outdir, "table_alpha_kl_auc_beta_mst.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha",
                    "mean_KL_uniform", "se_KL_uniform",
                    "mean_KL_structural", "se_KL_structural",
                    "RS_AUC_uniform", "RS_AUC_structural"])
        for t, a in enumerate(alphas):
            w.writerow([float(a),
                        float(KL_u_mean[t]), float(KL_u_se[t]),
                        float(KL_s_mean[t]), float(KL_s_se[t]),
                        float(AUC_u[t]), float(AUC_s[t])])
    print(f"[Saved] {csv_path}")

    # Save summary JSON
    js_path = os.path.join(args.outdir, "summary_beta_mst.json")
    with open(js_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {js_path}")

    # -----------------------------
    # Plots
    # -----------------------------
    plt.figure()
    plt.errorbar(alphas, KL_u_mean, yerr=KL_u_se, fmt="o-", capsize=3, label="Uniform KL")
    plt.errorbar(alphas, KL_s_mean, yerr=KL_s_se, fmt="s--", capsize=3, label="Structural-adaptive KL")
    plt.axvline(alpha_crit, linestyle="--", label="1/sqrt(N)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("alpha (flip probability)")
    plt.ylabel("Average empirical KL (plugin)")
    plt.title("BOWS2: KL scaling (log-log)")
    plt.legend()
    plt.tight_layout()
    fig1 = os.path.join(args.outdir, "kl_loglog_with_fit_region.png")
    plt.savefig(fig1, dpi=260)
    print(f"[Saved] {fig1}")

    plt.figure()
    plt.plot(alphas, AUC_u, "o-", label="Uniform (RS-AUC)")
    plt.plot(alphas, AUC_s, "s--", label="Structural-adaptive (RS-AUC)")
    plt.axvline(alpha_crit, linestyle="--", label="1/sqrt(N)")
    plt.xscale("log")
    plt.ylim(0.48, 0.85)
    plt.xlabel("alpha (flip probability)")
    plt.ylabel("RS AUC (symmetric)")
    plt.title("BOWS2: RS detection AUC")
    plt.legend()
    plt.tight_layout()
    fig2 = os.path.join(args.outdir, "rs_auc_logx.png")
    plt.savefig(fig2, dpi=260)
    print(f"[Saved] {fig2}")

    bins = list(summary["MST3_by_texture_bins"].keys())
    bu = [summary["MST3_by_texture_bins"][b]["uniform"]["bootstrap_mean"] for b in bins]
    bs = [summary["MST3_by_texture_bins"][b]["structural"]["bootstrap_mean"] for b in bins]
    plt.figure()
    plt.plot(range(3), bu, "o-", label="Uniform beta (boot mean)")
    plt.plot(range(3), bs, "s--", label="Structural beta (boot mean)")
    plt.xticks(range(3), bins, rotation=15)
    plt.ylabel("beta estimate")
    plt.title("MST-3: beta by texture bins")
    plt.legend()
    plt.tight_layout()
    fig3 = os.path.join(args.outdir, "beta_by_texture_bins.png")
    plt.savefig(fig3, dpi=260)
    print(f"[Saved] {fig3}")

    txt_path = os.path.join(args.outdir, "summary_beta_mst.txt")
    with open(txt_path, "w") as f:
        f.write(f"N={N}, alpha_crit=1/sqrt(N)={alpha_crit:.6g}\n")
        f.write(f"fit range: [{args.fit_min_alpha:g}, {fit_max:g}] using {fit_mask.sum()} points\n\n")
        f.write("MST-1 (global log-log fit on mean KL):\n")
        f.write(f"  uniform:    beta_hat={beta_u:.4f}, R2={R2_u:.4f}, boot_mean={bmean_u:.4f}, CI95=[{blo_u:.4f},{bhi_u:.4f}]\n")
        f.write(f"  structural: beta_hat={beta_s:.4f}, R2={R2_s:.4f}, boot_mean={bmean_s:.4f}, CI95=[{blo_s:.4f},{bhi_s:.4f}]\n\n")
        f.write("MST-3 (by texture bins, bootstrap mean + CI95):\n")
        for b in bins:
            d = summary["MST3_by_texture_bins"][b]
            f.write(f"  [{b}] count={d['count']}, texture_mean={d['texture_mean']:.4f}\n")
            u = d["uniform"]; s = d["structural"]
            f.write(f"    uniform:    boot_mean={u['bootstrap_mean']:.4f}, CI95=[{u['ci95'][0]:.4f},{u['ci95'][1]:.4f}], R2={u['R2_loglog']:.4f}\n")
            f.write(f"    structural: boot_mean={s['bootstrap_mean']:.4f}, CI95=[{s['ci95'][0]:.4f},{s['ci95'][1]:.4f}], R2={s['R2_loglog']:.4f}\n")
    print(f"[Saved] {txt_path}")

    # Print concise summary to console
    print("\n==== MST Summary ====")
    print(f"Fit range: alpha in [{args.fit_min_alpha:g}, {fit_max:g}]")
    print(f"Uniform:    beta_hat={beta_u:.4f}, boot_mean={bmean_u:.4f}, CI95=[{blo_u:.4f},{bhi_u:.4f}]")
    print(f"Structural: beta_hat={beta_s:.4f}, boot_mean={bmean_s:.4f}, CI95=[{blo_s:.4f},{bhi_s:.4f}]")
    for b in bins:
        d = summary["MST3_by_texture_bins"][b]
        print(f"  [{b}] uniform beta={d['uniform']['bootstrap_mean']:.4f} (CI {d['uniform']['ci95'][0]:.4f},{d['uniform']['ci95'][1]:.4f}) | "
              f"struct beta={d['structural']['bootstrap_mean']:.4f} (CI {d['structural']['ci95'][0]:.4f},{d['structural']['ci95'][1]:.4f})")


if __name__ == "__main__":
    main()