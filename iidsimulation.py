#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================================================
# Fixed image size (per your earlier requirement)
# =========================================================
IMG_HW = 256  # 256x256


# =========================================================
# 生成单像素概率质量函数 p(x)
# =========================================================
def make_pmf(kind="sawtooth_pairs", seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(256)

    if kind == "random":
        w = rng.random(256) + 1e-6
        return w / w.sum()

    if kind == "smooth_mixture":
        g1 = np.exp(-0.5 * ((x - 70.0) / 18.0) ** 2)
        g2 = np.exp(-0.5 * ((x - 170.0) / 25.0) ** 2)
        w = 0.65 * g1 + 0.35 * g2 + 1e-6
        return w / w.sum()

    if kind == "sawtooth_pairs":
        w = np.ones(256)
        for k in range(128):
            w[2 * k] = 2.0 + 0.5 * np.sin(k / 7.0)
            w[2 * k + 1] = 1.0 + 0.2 * np.cos(k / 5.0)
        w = np.clip(w, 1e-6, None)
        return w / w.sum()

    raise ValueError(f"Unknown pmf kind: {kind}")


# =========================================================
# LSB flipping 后的诱导分布 (uniform chapter-3 model)
# p_alpha(x) = (1-alpha)p(x) + alpha p(x xor 1)
# =========================================================
def induced_p_after_flipping(p, alpha):
    idx = np.arange(256)
    p_xor = p[np.bitwise_xor(idx, 1)]
    return (1 - alpha) * p + alpha * p_xor


# =========================================================
# 离散 KL 散度：KL(p||q) = sum p log(p/q)
# =========================================================
def kl_discrete(p, q, eps=1e-300):
    q_safe = np.maximum(q, eps)
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q_safe[mask]))))


# =========================================================
# 样本直方图 → 概率质量函数
# =========================================================
def hist_pmf(samples_u8):
    h = np.bincount(samples_u8.astype(np.int64), minlength=256).astype(np.float64)
    s = h.sum()
    if s <= 0:
        raise ValueError("Empty histogram.")
    return h / s


# =========================================================
# log-log 拟合斜率：y ≈ C alpha^k
# =========================================================
def fit_loglog_slope(alphas, y, fit_max_alpha=0.004):
    a = np.asarray(alphas, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = (a > 0) & (y > 0) & (a <= fit_max_alpha)
    if np.count_nonzero(m) < 2:
        return float("nan"), float("nan")
    X = np.log(a[m])
    Y = np.log(y[m])
    s, c = np.polyfit(X, Y, 1)
    return float(s), float(c)


def parse_list_csv_floats(s: str) -> np.ndarray:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if not parts:
        raise ValueError("Empty list string.")
    arr = np.array([float(x) for x in parts], dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError("All values must be > 0.")
    return arr


# =========================================================
# =====================  TASK 1: UNIFORM KL LAW  =====================
# =========================================================
def run_uniform_task(args):
    os.makedirs(args.outdir, exist_ok=True)

    N = int(args.N)
    M = int(args.M)

    alphas = parse_list_csv_floats(args.alpha_list)

    p = make_pmf(args.pmf_kind, seed=args.pmf_seed)

    # True per-pixel KL
    true_kl = np.array([kl_discrete(induced_p_after_flipping(p, a), p) for a in alphas], dtype=np.float64)

    # trials buffers
    estA_trials = np.zeros((args.trials, len(alphas)), dtype=np.float64)
    estB_trials = np.zeros((args.trials, len(alphas)), dtype=np.float64)

    rng = np.random.default_rng(args.seed)

    trial_iter = range(args.trials)
    if not args.no_progress:
        trial_iter = tqdm(trial_iter, desc="Uniform: Monte Carlo trials", unit="trial")

    for t in trial_iter:
        X = rng.choice(256, size=N, p=p).astype(np.uint8)
        pX_hat = hist_pmf(X)

        for i, a in enumerate(alphas):
            p_a = induced_p_after_flipping(p, a)

            # Est A: sample Y~p_alpha
            Y_mc = rng.choice(256, size=M, p=p_a).astype(np.uint8)
            estA_trials[t, i] = float(np.mean(np.log(p_a[Y_mc] / p[Y_mc])))

            # Est B: flip X with prob a
            mask = rng.random(N) < a
            Y = X.copy()
            Y[mask] ^= np.uint8(1)
            pY_hat = hist_pmf(Y)
            estB_trials[t, i] = kl_discrete(pY_hat, pX_hat)

    estA_mean = estA_trials.mean(axis=0)
    estA_std = estA_trials.std(axis=0, ddof=1) if args.trials > 1 else np.zeros_like(estA_mean)
    estB_mean = estB_trials.mean(axis=0)
    estB_std = estB_trials.std(axis=0, ddof=1) if args.trials > 1 else np.zeros_like(estB_mean)

    true_total = N * true_kl
    estA_total_mean = N * estA_mean
    estA_total_std = N * estA_std
    estB_total_mean = N * estB_mean
    estB_total_std = N * estB_std

    denom = N * alphas**2
    ratio_true = true_total / denom
    ratio_A_mean = estA_total_mean / denom
    ratio_A_std = estA_total_std / denom
    ratio_B_mean = estB_total_mean / denom
    ratio_B_std = estB_total_std / denom

    s_true, _ = fit_loglog_slope(alphas, true_kl, args.fit_max_alpha)
    s_A, _ = fit_loglog_slope(alphas, estA_mean, args.fit_max_alpha)
    s_B, _ = fit_loglog_slope(alphas, estB_mean, args.fit_max_alpha)

    print("=== [UNIFORM] log-log slope on per-pixel KL (fit alpha <= {:.6g}) ===".format(args.fit_max_alpha))
    print("True slope:", s_true)
    print("Est A slope:", s_A)
    print("Est B slope:", s_B)

    # CSV
    csv_path = os.path.join(args.outdir, args.csv_name_uniform)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "alpha",
            "true_kl_per_pixel",
            "estA_mean_per_pixel", "estA_std_per_pixel",
            "estB_mean_per_pixel", "estB_std_per_pixel",
            "true_total",
            "estA_total_mean", "estA_total_std",
            "estB_total_mean", "estB_total_std",
            "ratio_true",
            "ratioA_mean", "ratioA_std",
            "ratioB_mean", "ratioB_std",
            "N",
            "M",
            "trials",
            "seed",
            "pmf_kind",
            "pmf_seed",
            "img_hw",
        ])
        for i in range(len(alphas)):
            w.writerow([
                float(alphas[i]),
                float(true_kl[i]),
                float(estA_mean[i]), float(estA_std[i]),
                float(estB_mean[i]), float(estB_std[i]),
                float(true_total[i]),
                float(estA_total_mean[i]), float(estA_total_std[i]),
                float(estB_total_mean[i]), float(estB_total_std[i]),
                float(ratio_true[i]),
                float(ratio_A_mean[i]), float(ratio_A_std[i]),
                float(ratio_B_mean[i]), float(ratio_B_std[i]),
                int(N),
                int(M),
                int(args.trials),
                int(args.seed),
                args.pmf_kind,
                int(args.pmf_seed),
                int(IMG_HW),
            ])

    # Plot 1
    fig1_path = os.path.join(args.outdir, "uniform_kl_total_scale.png")
    plt.figure()
    plt.loglog(alphas, true_total, "o-", label="True (pmf KL)")
    plt.errorbar(alphas, estA_total_mean, yerr=estA_total_std, fmt="o-", capsize=3, label="Est A (MC, mean±std)")
    plt.errorbar(alphas, estB_total_mean, yerr=estB_total_std, fmt="o-", capsize=3, label="Est B (plugin, mean±std)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$D_{\mathrm{KL}}(P_Y^{(N)}\|P_X^{(N)})$")
    plt.title("Uniform flipping: Synthetic KL vs alpha (total scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)

    # Plot 2
    fig2_path = os.path.join(args.outdir, "uniform_kl_normalized_ratio.png")
    plt.figure()
    plt.plot(alphas, ratio_true, "o-", label="True")
    plt.errorbar(alphas, ratio_A_mean, yerr=ratio_A_std, fmt="o-", capsize=3, label="Est A (mean±std)")
    plt.errorbar(alphas, ratio_B_mean, yerr=ratio_B_std, fmt="o-", capsize=3, label="Est B (mean±std)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$R(\alpha)=D_{\mathrm{KL}}/(N\alpha^2)$")
    plt.title("Uniform flipping: Normalized ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)

    print("Saved [UNIFORM] outputs to:", args.outdir)
    print(" Figure 1:", fig1_path)
    print(" Figure 2:", fig2_path)
    print(" CSV     :", csv_path)


# =========================================================
# =====================  TASK 2: ADAPTIVE vs UNIFORM  =====================
# =========================================================
def build_weights_from_histogram(h, eta=1e-12):
    idx = np.arange(256)
    idx_xor = np.bitwise_xor(idx, 1)
    num = np.abs(h - h[idx_xor])
    den = h + h[idx_xor] + eta
    r = num / den
    w = 1.0 - r
    w = np.clip(w, 0.0, 1.0)
    return w, r


def weighted_sample_without_replacement(rng, weights, m):
    p = np.asarray(weights, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        p = None
    else:
        p = p / s
    return rng.choice(len(weights), size=m, replace=False, p=p)


def embed_uniform_fixed_payload(rng, X_u8, rho):
    N = X_u8.size
    M = int(np.floor(rho * N))
    Y = X_u8.copy()
    if M <= 0:
        return Y
    idx = rng.choice(N, size=M, replace=False)
    Y[idx] ^= np.uint8(1)
    return Y


def embed_adaptive_fixed_payload(rng, X_u8, rho, eta=1e-12):
    N = X_u8.size
    M = int(np.floor(rho * N))
    Y = X_u8.copy()
    if M <= 0:
        return Y
    h = hist_pmf(X_u8)
    w_x, _ = build_weights_from_histogram(h, eta=eta)
    w_i = w_x[X_u8]
    idx = weighted_sample_without_replacement(rng, w_i, M)
    Y[idx] ^= np.uint8(1)
    return Y


def induced_q_from_p_and_alpha(p, alpha_x):
    idx = np.arange(256)
    idx_xor = np.bitwise_xor(idx, 1)
    q = p * (1.0 - alpha_x) + p[idx_xor] * alpha_x[idx_xor]
    return q


def alpha_from_weights_payload(p, w_x, rho):
    denom = float(np.sum(p * w_x))
    if denom <= 0:
        return np.full(256, rho, dtype=np.float64)
    alpha_x = rho * w_x / denom
    alpha_x = np.clip(alpha_x, 0.0, 1.0)
    return alpha_x


def save_csv_simple(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def parse_rhos(rhos_str: str) -> np.ndarray:
    s = rhos_str.strip()
    if s.startswith("linspace:"):
        _, a, b, n = s.split(":")
        return np.linspace(float(a), float(b), int(n), dtype=np.float64)
    return parse_list_csv_floats(s)


def run_adaptive_task(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    N = int(args.N)

    rhos = parse_rhos(args.rhos)
    rhos = np.array([r for r in rhos if r > 0], dtype=np.float64)
    rhos.sort()

    p = make_pmf(args.pmf_kind, seed=args.pmf_seed)
    rng = np.random.default_rng(args.seed)

    kl_uniform_plugin = np.zeros_like(rhos, dtype=np.float64)
    kl_adapt_plugin = np.zeros_like(rhos, dtype=np.float64)
    kl_uniform_analytic = np.zeros_like(rhos, dtype=np.float64)
    kl_adapt_analytic = np.zeros_like(rhos, dtype=np.float64)

    total_steps = args.trials * len(rhos)
    pbar = None
    if not args.no_progress:
        pbar = tqdm(total=total_steps, desc="Adaptive: Running", unit="step")

    for _ in range(args.trials):
        X = rng.choice(256, size=N, p=p).astype(np.uint8)
        pX_hat = hist_pmf(X)
        w_x, _ = build_weights_from_histogram(pX_hat, eta=args.eta)

        for i, rho in enumerate(rhos):
            Y_u = embed_uniform_fixed_payload(rng, X, rho)
            pY_u_hat = hist_pmf(Y_u)
            kl_uniform_plugin[i] += kl_discrete(pY_u_hat, pX_hat)

            Y_a = embed_adaptive_fixed_payload(rng, X, rho, eta=args.eta)
            pY_a_hat = hist_pmf(Y_a)
            kl_adapt_plugin[i] += kl_discrete(pY_a_hat, pX_hat)

            alpha_u = np.full(256, rho, dtype=np.float64)
            q_u = induced_q_from_p_and_alpha(p, alpha_u)
            kl_uniform_analytic[i] += kl_discrete(q_u, p)

            alpha_a = alpha_from_weights_payload(p, w_x, rho)
            q_a = induced_q_from_p_and_alpha(p, alpha_a)
            kl_adapt_analytic[i] += kl_discrete(q_a, p)

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    kl_uniform_plugin /= args.trials
    kl_adapt_plugin /= args.trials
    kl_uniform_analytic /= args.trials
    kl_adapt_analytic /= args.trials

    total_uniform_plugin = N * kl_uniform_plugin
    total_adapt_plugin = N * kl_adapt_plugin
    total_uniform_analytic = N * kl_uniform_analytic
    total_adapt_analytic = N * kl_adapt_analytic

    s_u, _ = fit_loglog_slope(rhos, kl_uniform_analytic, args.fit_max)
    s_a, _ = fit_loglog_slope(rhos, kl_adapt_analytic, args.fit_max)

    print(f"=== [ADAPTIVE] per-pixel analytic log-log slope (rho <= {args.fit_max}) ===")
    print("Uniform analytic slope:", s_u)
    print("Adaptive analytic slope:", s_a)

    csv_path = outdir / args.csv_name_adaptive
    header = [
        "rho",
        "kl_uniform_plugin_perpix",
        "kl_adapt_plugin_perpix",
        "kl_uniform_analytic_perpix",
        "kl_adapt_analytic_perpix",
        "total_uniform_plugin",
        "total_adapt_plugin",
        "total_uniform_analytic",
        "total_adapt_analytic",
        "R_uniform_plugin",
        "R_adapt_plugin",
        "R_uniform_analytic",
        "R_adapt_analytic",
        "N",
        "trials",
        "seed",
        "pmf_kind",
        "pmf_seed",
        "img_hw",
        "eta",
        "fit_max",
    ]

    rows = []
    for i, rho in enumerate(rhos):
        R_up = total_uniform_plugin[i] / (N * rho**2) if rho > 0 else np.nan
        R_ap = total_adapt_plugin[i] / (N * rho**2) if rho > 0 else np.nan
        R_ua = total_uniform_analytic[i] / (N * rho**2) if rho > 0 else np.nan
        R_aa = total_adapt_analytic[i] / (N * rho**2) if rho > 0 else np.nan

        rows.append([
            float(rho),
            float(kl_uniform_plugin[i]),
            float(kl_adapt_plugin[i]),
            float(kl_uniform_analytic[i]),
            float(kl_adapt_analytic[i]),
            float(total_uniform_plugin[i]),
            float(total_adapt_plugin[i]),
            float(total_uniform_analytic[i]),
            float(total_adapt_analytic[i]),
            float(R_up),
            float(R_ap),
            float(R_ua),
            float(R_aa),
            int(N),
            int(args.trials),
            int(args.seed),
            args.pmf_kind,
            int(args.pmf_seed),
            int(IMG_HW),
            float(args.eta),
            float(args.fit_max),
        ])

    save_csv_simple(csv_path, header, rows)
    print(f"[saved] {csv_path}")

    fig1_png = outdir / "adaptive_kl_vs_rho.png"
    plt.figure()
    plt.loglog(rhos, total_uniform_analytic, "o-", label="Uniform (analytic approx)")
    plt.loglog(rhos, total_adapt_analytic, "o-", label="Adaptive (analytic approx)")
    plt.loglog(rhos, total_uniform_plugin, "o--", label="Uniform (plugin hist/hist)")
    plt.loglog(rhos, total_adapt_plugin, "o--", label="Adaptive (plugin hist/hist)")
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$D_{\mathrm{KL}}(P_Y^{(N)}\|P_X^{(N)})$")
    plt.title("Uniform vs Adaptive (fixed payload)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1_png, dpi=200)
    print(f"[saved] {fig1_png}")

    fig2_png = outdir / "adaptive_normalized_ratio.png"
    plt.figure()
    plt.plot(rhos, total_uniform_analytic / (N * rhos**2), "o-", label="Uniform (analytic)")
    plt.plot(rhos, total_adapt_analytic / (N * rhos**2), "o-", label="Adaptive (analytic)")
    plt.plot(rhos, total_uniform_plugin / (N * rhos**2), "o--", label="Uniform (plugin)")
    plt.plot(rhos, total_adapt_plugin / (N * rhos**2), "o--", label="Adaptive (plugin)")
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$R(\rho)=D_{\mathrm{KL}}/(N\rho^2)$")
    plt.title("Normalized ratio (2nd-order constant comparison)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2_png, dpi=200)
    print(f"[saved] {fig2_png}")


# =========================================================
# CLI
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Merged KL-law experiments (uniform + adaptive). Image size fixed at 256x256."
    )

    ap.add_argument("--task", type=str, default="both",
                    choices=["uniform", "adaptive", "both"],
                    help="Which experiment to run")

    # unified seed parameter
    ap.add_argument("--seed", type=int, default=123, help="Random seed for the RNG stream")

    # NEW: N, M parameters (per your request)
    ap.add_argument("--N", type=int, default=6553500,
                    help="Total sample size N used for Est B and total KL scaling (default: 6553500)")
    ap.add_argument("--M", type=int, default=2000000,
                    help="MC sample size M for Est A (default: 2000000)")

    # common output
    ap.add_argument("--outdir", type=str, default="out_merged", help="Output directory")
    ap.add_argument("--trials", type=int, default=20, help="Number of trials")

    # pmf controls
    ap.add_argument("--pmf_kind", type=str, default="sawtooth_pairs",
                    choices=["random", "smooth_mixture", "sawtooth_pairs"],
                    help="Type of cover pmf p(x)")
    ap.add_argument("--pmf_seed", type=int, default=0,
                    help="Seed used to generate pmf when pmf_kind=random")

    # uniform-task specific
    ap.add_argument("--alpha_list", type=str,
                    default="0.0002,0.0005,0.001,0.002,0.003,0.004,0.006,0.01",
                    help="Comma-separated list of alphas (uniform task)")
    ap.add_argument("--fit_max_alpha", type=float, default=0.004,
                    help="Upper alpha bound for uniform log-log slope fit")
    ap.add_argument("--csv_name_uniform", type=str, default="uniform_kl_results.csv",
                    help="CSV filename for uniform task (inside outdir)")

    # adaptive-task specific
    ap.add_argument("--rhos", type=str,
                    default="0.0002,0.0005,0.001,0.002,0.003,0.004,0.006,0.01",
                    help='Payload list for adaptive task. Supports "a,b,c" or "linspace:a:b:n"')
    ap.add_argument("--fit_max", type=float, default=0.004,
                    help="Upper rho bound for adaptive analytic log-log slope fit")
    ap.add_argument("--eta", type=float, default=1e-12, help="Stabilizer eta for r(x)")
    ap.add_argument("--csv_name_adaptive", type=str, default="adaptive_ch4_results.csv",
                    help="CSV filename for adaptive task (inside outdir)")

    # UI controls
    ap.add_argument("--no_show", action="store_true", help="Do not show figures interactively")
    ap.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bar")

    return ap.parse_args()


def main():
    args = parse_args()

    if args.N <= 0 or args.M <= 0 or args.trials <= 0:
        raise ValueError("N, M, trials must be positive.")

    print("=== Settings ===")
    print("IMG_HW =", IMG_HW, "(fixed)")
    print("N      =", args.N)
    print("M      =", args.M)
    print("seed   =", args.seed)
    print("trials =", args.trials)
    print("task   =", args.task)
    print("================")

    if args.task in ("uniform", "both"):
        run_uniform_task(args)

    if args.task in ("adaptive", "both"):
        run_adaptive_task(args)

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()