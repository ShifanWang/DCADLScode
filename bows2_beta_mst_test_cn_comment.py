#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOWS2 beta MST test (MST-1 + MST-3) with progress bars.

功能概述（中文）：
本脚本用于在 BOWS2 自然图像数据集上做 beta 指数的 MST 测试，并带进度条显示。

实现的两类实验（对应你文档里的 MST-1 / MST-3）：
- MST-1：对“平均经验 KL ~ alpha^beta”的幂律关系做 log-log 线性回归，拟合得到 beta。
- MST-3：用高通残差能量把图像分成 3 个纹理强度桶（低/中/高），并分别拟合每个桶的 beta。

输出：
- CSV 表格：table_alpha_kl_auc_beta_mst.csv（每个 alpha 下的 KL 均值/标准误 + RS-AUC）
- 汇总：summary_beta_mst.json, summary_beta_mst.txt（beta 的估计、R2、bootstrap 置信区间等）
- 图片：kl_loglog_with_fit_region.png, rs_auc_logx.png, beta_by_texture_bins.png
"""

import os, glob, argparse, csv, json
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# -----------------------------
# IO：读数据集图片
# -----------------------------
def list_images(root: str):
    """
    递归枚举 root 目录下的图像文件，支持常见扩展名。
    返回排序后的文件路径列表。
    """
    exts = ("*.pgm", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root, "**", e), recursive=True))
    files = sorted(files)
    if len(files) == 0:
        raise FileNotFoundError(f"No images found under: {root}")
    return files

def read_grayscale_u8(path: str) -> np.ndarray:
    """
    读取图片并保证输出为 uint8 的二维灰度图（H×W）。
    - 如果读到 RGB（三通道），用标准亮度加权转灰度；
    - 如果不是二维灰度，报错。
    """
    img = imageio.imread(path)
    if img.ndim == 3:
        # RGB -> 灰度（近似 ITU-R BT.601 luma）
        img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).round().astype(np.uint8)
    img = img.astype(np.uint8)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape={img.shape} for {path}")
    return img


# -----------------------------
# 工具函数：二维卷积（same 尺寸）+ reflect padding
# -----------------------------
def conv2d_same_reflect(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    对图像 x 做二维卷积，输出与输入同尺寸（same）。
    - 使用 reflect padding（镜像填充）以减小边界伪影；
    - 这里用显式 for 循环实现卷积核滑动（可读性更强，但速度一般）。
    """
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
    """
    返回 a 的“反射边界平移版本”（尺寸不变）：
    - 先按 dy/dx 需要的量做 reflect pad；
    - 再取回与原图同尺寸的窗口。
    作用：配合 structural_cost_rho 中“卷积残差的局部影响传播”计算。
    """
    H, W = a.shape
    pad_y = abs(dy)
    pad_x = abs(dx)
    ap = np.pad(a, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    y0 = pad_y + dy
    x0 = pad_x + dx
    return ap[y0:y0 + H, x0:x0 + W]


# -----------------------------
# 嵌入原语：均匀 LSB flipping
# -----------------------------
def uniform_lsb_flipping(img_u8: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """
    均匀随机 LSB flipping：
    - N 为像素总数；
    - k = floor(alpha * N) 为翻转像素个数；
    - 随机无放回选 k 个位置；
    - 对这些像素做 XOR 1（最低位翻转）。
    """
    N = img_u8.size
    k = int(np.floor(alpha * N))
    if k <= 0:
        return img_u8.copy()
    flat = img_u8.reshape(-1).copy()
    idx = rng.choice(N, size=k, replace=False)

    # 关键操作：bitwise_xor(x,1) 等价于 x 的最低有效位 LSB 取反
    # 例如：偶数(LSB=0) -> +1 变奇数；奇数(LSB=1) -> -1 变偶数（在二进制层面是 LSB 翻转）
    # 这就是“LSB flipping”抽象模型对应的像素修改。
    flat[idx] = np.bitwise_xor(flat[idx], 1).astype(np.uint8)
    return flat.reshape(img_u8.shape)


# -----------------------------
# 结构代价 rho_{i,j}：基于残差的“翻转影响强度”
# -----------------------------
def structural_cost_rho(img_u8: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    计算每个像素的结构代价 rho：
    思想：先用高通滤波器 h 得到残差 R=H*X。
    再评估如果对像素做 XOR1 翻转，会如何改变局部残差幅度 |R|。
    rho 越小表示“翻转该像素对结构统计(残差幅度)扰动越小”，更适合嵌入。
    """
    X = img_u8.astype(np.float32)
    R = conv2d_same_reflect(X, h)

    # XOR1 的“符号增量”delta：
    # 若像素为偶数（LSB=0），XOR1 等价于 +1；
    # 若像素为奇数（LSB=1），XOR1 等价于 -1；
    # 用 delta = 1 - 2*(LSB) 统一写成 +1 或 -1。
    delta = (1.0 - 2.0 * (img_u8 & 1).astype(np.float32))

    kh, kw = h.shape
    ph, pw = kh // 2, kw // 2
    rho = np.zeros_like(R, dtype=np.float32)

    # 遍历卷积核的每个系数，累积“翻转导致的残差幅度变化量”
    for iu in range(kh):
        for jv in range(kw):
            coef = float(h[iu, jv])
            if coef == 0.0:
                continue
            dy = iu - ph
            dx = jv - pw

            # R_shift：把残差图按 (dy,dx) 取对齐版本
            R_shift = shift_same(R, dy, dx)

            # Rp：如果中心像素翻转，那么对应残差会变化 delta*coef
            Rp = R_shift + delta * coef

            # abs(abs(Rp)-abs(R_shift))：残差幅度的改变量（绝对值）
            # 再对所有核位置求和作为 rho
            rho += np.abs(np.abs(Rp) - np.abs(R_shift))

    return rho

def structural_adaptive_lsb_flipping(img_u8: np.ndarray, alpha: float, h: np.ndarray,
                                    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    结构自适应 LSB flipping：
    - 先计算每个像素代价 rho；
    - 选取 rho 最小的 k=floor(alpha*N) 个像素进行翻转（确定性策略）；
    - 可选：用极小噪声打破 rho 的并列（用于稳定性/可重复性检查）。
    """
    N = img_u8.size
    k = int(np.floor(alpha * N))
    if k <= 0:
        return img_u8.copy()

    rho = structural_cost_rho(img_u8, h).reshape(-1).astype(np.float64)

    if rng is not None:
        # 加极小抖动：只用于打破完全相同 rho 的 tie，不应实质改变排序
        rho = rho + 1e-12 * rng.standard_normal(rho.shape)

    chosen = np.argsort(rho)[:k]
    flat = img_u8.reshape(-1).copy()
    flat[chosen] = np.bitwise_xor(flat[chosen], 1).astype(np.uint8)
    return flat.reshape(img_u8.shape)


# -----------------------------
# 经验 KL：对直方图做 epsilon 平滑的 plugin 估计
# -----------------------------
def hist_eps(img_u8: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    计算 256 灰度直方图并转成概率分布 p(x)。
    - 使用 eps 进行截断/平滑，避免 KL 中 log(0)；
    - 最后重新归一化保证 sum p = 1。
    """
    h = np.bincount(img_u8.reshape(-1), minlength=256).astype(np.float64)
    p = h / img_u8.size
    p = np.clip(p, eps, 1.0)
    p /= p.sum()
    return p

def kl(p: np.ndarray, q: np.ndarray) -> float:
    """
    计算离散 KL 散度：D_KL(p||q) = sum p * (log p - log q)
    注意：这里使用自然对数 np.log（单位是 nats），但用于幂律拟合不影响 beta。
    """
    return float(np.sum(p * (np.log(p) - np.log(q))))


# -----------------------------
# RS score：快速向量化实现
# -----------------------------
def rs_score_fast(img_u8: np.ndarray, group_size: int = 4, mask=None) -> float:
    """
    计算 RS 分析中的一个“检测分数”（这里取一个对称差异的负值，越大通常越“像 cover”）。
    简要流程：
    - 将像素序列分成长度 group_size 的组；
    - discrim：用相邻差的绝对值和作为判别函数 f；
    - 构造 F1/F-1 翻转（偶数±1、奇数∓1，并裁剪到[0,255]）；
    - 根据 mask 构造 M/N 两种掩码翻转组；
    - 计算 Rm,Sm,Rn,Sn（f 增大/减小的比例）；
    - 返回 -( |Rm-Rn| + |Sm-Sn| ) 作为一个“对称差异指标”。
    """
    if mask is None:
        mask = np.array([1, 0, 1, 0], dtype=np.int8)
    mask = np.asarray(mask, dtype=np.int8)
    if mask.size != group_size:
        raise ValueError("mask length must equal group_size")

    flat = img_u8.reshape(-1)
    n_groups = flat.size // group_size
    data = flat[:n_groups * group_size].reshape(n_groups, group_size).astype(np.int16)

    def discrim(g):
        # 判别函数：组内相邻差的绝对值和
        return np.sum(np.abs(g[:, 1:] - g[:, :-1]), axis=1)

    f0 = discrim(data)

    even = (data % 2 == 0)

    # F1：偶数+1，奇数-1（并裁剪）
    gF1 = data.copy()
    gF1[even] += 1
    gF1[~even] -= 1
    np.clip(gF1, 0, 255, out=gF1)

    # F-1：偶数-1，奇数+1（并裁剪）
    gFm1 = data.copy()
    gFm1[even] -= 1
    gFm1[~even] += 1
    np.clip(gFm1, 0, 255, out=gFm1)

    # M/N：按 mask 在部分位置应用 F1/F-1
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

    # R/S：判别函数增大/减小的比例
    Rm = np.mean(fM > f0); Sm = np.mean(fM < f0)
    Rn = np.mean(fN > f0); Sn = np.mean(fN < f0)

    # 返回一个对称差异（负号使得“差异越小分数越大”）
    return float(-(abs(Rm - Rn) + abs(Sm - Sn)))


# -----------------------------
# beta 拟合：log-log 回归 + bootstrap
# -----------------------------
def fit_beta_loglog(alphas: np.ndarray, y: np.ndarray, y_se: np.ndarray | None = None):
    """
    在 log 域拟合幂律：y ≈ const * alpha^beta
    等价于：log(y) = a + beta * log(alpha)

    - 若提供 y_se：在 log 域做加权最小二乘（近似把 y 的标准误转为 log(y) 的标准误）
    - 返回：beta（斜率）、a（截距）、R2（log 域拟合优度）
    """
    x = np.log(alphas)
    z = np.log(y)

    if y_se is None:
        w = np.ones_like(y)
    else:
        # 误差传播：sigma_log ≈ se / y
        sigma_log = np.clip(y_se / np.maximum(y, 1e-300), 1e-12, np.inf)
        w = 1.0 / (sigma_log ** 2)

    # 加权线性回归的闭式解
    W = np.sum(w)
    xbar = np.sum(w * x) / W
    zbar = np.sum(w * z) / W
    Sxx = np.sum(w * (x - xbar) ** 2)
    Sxz = np.sum(w * (x - xbar) * (z - zbar))

    beta = Sxz / Sxx
    a = zbar - beta * xbar

    # log 域 R2
    zhat = a + beta * x
    ss_tot = np.sum(w * (z - zbar) ** 2)
    ss_res = np.sum(w * (z - zhat) ** 2)
    R2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    return float(beta), float(a), float(R2)

def bootstrap_beta(alphas: np.ndarray, per_image_y: np.ndarray, fit_mask: np.ndarray,
                   B: int, rng: np.random.Generator):
    """
    对 beta 做 bootstrap（按“图像”为单位重采样）：
    - per_image_y: [M_images, K_alphas]，例如每张图每个 alpha 的 KL
    - 每次 bootstrap 从 M 张图中有放回抽 M 张，求均值曲线，再拟合 beta
    - 输出 beta 的均值与 95% 置信区间
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
# 主程序入口
# -----------------------------
def main():
    # 解析命令行参数
    ap = argparse.ArgumentParser()
    ap.add_argument("--bows2_dir", type=str, required=True)
    ap.add_argument("--M", type=int, default=2000, help="number of images sampled from BOWS2")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", type=str, default="out_beta_mst")
    ap.add_argument("--filter", type=str, default="laplacian4", choices=["laplacian4", "hp8"])

    # alpha 网格参数
    ap.add_argument("--alpha_min", type=float, default=5e-4)
    ap.add_argument("--alpha_max", type=float, default=1e-2)
    ap.add_argument("--alpha_num", type=int, default=12)
    ap.add_argument("--alpha_grid", type=str, default="log", choices=["log", "linear"])

    # 拟合 beta 的 alpha 范围（避免过大 alpha 进入非渐近区）
    ap.add_argument("--fit_max_alpha", type=float, default=-1.0,
                    help="max alpha used to fit beta; if <0, use 1/sqrt(N)")
    ap.add_argument("--fit_min_alpha", type=float, default=0.0,
                    help="min alpha used to fit beta")

    # 稳定性选项
    ap.add_argument("--uniform_reps", type=int, default=1, help="repeat uniform embedding reps per image/alpha")
    ap.add_argument("--break_ties", action="store_true",
                    help="add tiny jitter to rho to break ties (uses rng)")

    # bootstrap 次数
    ap.add_argument("--bootstrap", type=int, default=400)

    # RS 分组长度
    ap.add_argument("--rs_group", type=int, default=4)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 根据参数选择高通滤波器 H
    # laplacian4：4 邻域拉普拉斯
    # hp8：8 邻域高通
    if args.filter == "laplacian4":
        H = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]], dtype=np.float32)
    else:
        H = np.array([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1]], dtype=np.float32)

    # 加载 BOWS2 图片路径并随机抽样 M 张
    files = list_images(args.bows2_dir)
    if args.M > len(files):
        raise ValueError(f"M={args.M} exceeds dataset size {len(files)}")

    sel = rng.choice(len(files), size=args.M, replace=False)
    sel_files = [files[i] for i in sel]

    # 读图为灰度 uint8
    print(f"[Info] Loading {len(sel_files)} images ...")
    covers = [read_grayscale_u8(p) for p in tqdm(sel_files, desc="Load images", unit="img")]

    # N：每张图像素数（假设所有图片同尺寸）
    N = covers[0].size
    alpha_crit = 1.0 / np.sqrt(N)  # 常用“临界/参考尺度”：1/sqrt(N)

    # 构造 alpha 网格（log 或线性）
    if args.alpha_grid == "log":
        alphas = np.exp(np.linspace(np.log(args.alpha_min), np.log(args.alpha_max), args.alpha_num))
    else:
        alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_num)
    alphas = np.array(alphas, dtype=np.float64)

    # 拟合区间：默认 fit_max = 1/sqrt(N)
    fit_max = alpha_crit if args.fit_max_alpha < 0 else args.fit_max_alpha
    fit_min = args.fit_min_alpha
    fit_mask = (alphas <= fit_max) & (alphas >= fit_min)

    print(f"[Info] M={args.M}, image size={covers[0].shape}, N={N}, 1/sqrt(N)≈{alpha_crit:.6f}, filter={args.filter}")
    print(f"[Info] alpha grid: {alphas}")
    print(f"[Info] fit range: [{fit_min:g}, {fit_max:g}] -> using {fit_mask.sum()} points")
    if fit_mask.sum() < 3:
        raise ValueError("Need at least 3 alpha points in fit range for stable beta estimation.")

    # MST-3 的纹理度量：mean(|H*X|)，即高通残差的平均绝对值
    print("[Info] Computing texture scores for MST-3 ...")
    tex = []
    for img in tqdm(covers, desc="Texture score", unit="img"):
        R = conv2d_same_reflect(img.astype(np.float32), H)
        tex.append(float(np.mean(np.abs(R))))
    tex = np.array(tex, dtype=np.float64)

    # 按纹理强度排序并三等分：低/中/高
    order = np.argsort(tex)
    thirds = np.array_split(order, 3)
    bin_names = ["low_texture", "mid_texture", "high_texture"]

    # 存每张图、每个 alpha 下的 KL（用于 bootstrap）
    M = len(covers)
    K = len(alphas)
    KL_u_img = np.zeros((M, K), dtype=np.float64)  # uniform
    KL_s_img = np.zeros((M, K), dtype=np.float64)  # structural

    # 统计量：均值 + 标准误（跨图像）
    KL_u_mean = np.zeros(K); KL_u_se = np.zeros(K)
    KL_s_mean = np.zeros(K); KL_s_se = np.zeros(K)

    # RS 检测 AUC（每个 alpha 一个）
    AUC_u = np.zeros(K); AUC_s = np.zeros(K)

    # -----------------------------
    # 主循环：遍历 alpha，再遍历图像
    # -----------------------------
    for t, a in enumerate(tqdm(alphas, desc="Alpha grid", unit="alpha")):
        # 为每个 alpha 派生独立 RNG，保证可重复并避免不同 alpha 之间强相关
        # 1000003 常见用作大素数模（让取模后的分布更均匀，减少碰撞）
        rng_u = np.random.default_rng(args.seed + 1001 + int(a * 1e9) % 1000003)
        rng_s = np.random.default_rng(args.seed + 2002 + int(a * 1e9) % 1000003)

        # 用于 RS AUC：scores 是检测分数，labels=0/1 表示 cover/stego
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
            # cover 的一阶分布（epsilon 平滑直方图）
            pX = hist_eps(img)

            # ---------- uniform embedding ----------
            # 可重复做 uniform_reps 次并平均，降低一次随机采样带来的方差
            kls_u = []
            rs_u = []
            for _ in range(args.uniform_reps):
                stego_u = uniform_lsb_flipping(img, float(a), rng_u)
                kls_u.append(kl(hist_eps(stego_u), pX))
                rs_u.append(rs_score_fast(stego_u, group_size=args.rs_group))
            KL_u_img[i, t] = float(np.mean(kls_u))

            # ---------- structural adaptive embedding ----------
            # 结构自适应：选 rho 最小的 k 个像素翻转；默认确定性
            stego_s = structural_adaptive_lsb_flipping(
                img, float(a), H, rng=(rng_s if args.break_ties else None)
            )
            KL_s_img[i, t] = kl(hist_eps(stego_s), pX)

            # ---------- RS 分数 + 构造二分类数据 ----------
            # 这里 cover 分数 s0 被同时用于 uniform/structural 的对照
            s0 = rs_score_fast(img, group_size=args.rs_group)

            # 对 uniform：加入 cover 与 stego(均值) 两个样本
            scores_u.extend([s0, float(np.mean(rs_u))]); labels_u.extend([0, 1])

            # 对 structural：加入 cover 与 stego 两个样本
            scores_s.extend([s0, rs_score_fast(stego_s, group_size=args.rs_group)]); labels_s.extend([0, 1])

            # 进度条实时显示当前累计均值（便于观察收敛趋势）
            if (i + 1) % 50 == 0 or (i + 1) == len(covers):
                img_pbar.set_postfix({
                    "KL_u_mean": f"{KL_u_img[:i+1, t].mean():.2e}",
                    "KL_s_mean": f"{KL_s_img[:i+1, t].mean():.2e}",
                })

        # ---------- 每个 alpha 的汇总统计 ----------
        KL_u_mean[t] = KL_u_img[:, t].mean()
        KL_u_se[t] = KL_u_img[:, t].std(ddof=1) / np.sqrt(M)
        KL_s_mean[t] = KL_s_img[:, t].mean()
        KL_s_se[t] = KL_s_img[:, t].std(ddof=1) / np.sqrt(M)

        # RS-AUC：先算 AUC，再对称化到 >=0.5（方向不重要）
        auc0_u = roc_auc_score(np.array(labels_u), np.array(scores_u))
        auc0_s = roc_auc_score(np.array(labels_s), np.array(scores_s))
        AUC_u[t] = max(auc0_u, 1.0 - auc0_u)
        AUC_s[t] = max(auc0_s, 1.0 - auc0_s)

        tqdm.write(f"[alpha={a:.6g}] KL_u={KL_u_mean[t]:.3e} KL_s={KL_s_mean[t]:.3e} | AUC_u={AUC_u[t]:.4f} AUC_s={AUC_s[t]:.4f}")

    # -----------------------------
    # MST-1：全局 beta 拟合（对 mean KL 曲线做 log-log 回归）
    # -----------------------------
    beta_u, a_u, R2_u = fit_beta_loglog(alphas[fit_mask], np.maximum(KL_u_mean[fit_mask], 1e-300), KL_u_se[fit_mask])
    beta_s, a_s, R2_s = fit_beta_loglog(alphas[fit_mask], np.maximum(KL_s_mean[fit_mask], 1e-300), KL_s_se[fit_mask])

    # 对 beta 进行 bootstrap（按图像重采样）得到置信区间
    bmean_u, blo_u, bhi_u = bootstrap_beta(alphas, KL_u_img, fit_mask, args.bootstrap, rng)
    bmean_s, blo_s, bhi_s = bootstrap_beta(alphas, KL_s_img, fit_mask, args.bootstrap, rng)

    # 汇总信息（保存为 JSON）
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
    # MST-3：按纹理桶分别拟合 beta
    # -----------------------------
    for name, idx in zip(bin_names, thirds):
        idx = np.array(idx, dtype=int)

        # 该桶内的均值 KL 曲线
        mu_u_bin = KL_u_img[idx].mean(axis=0)
        mu_s_bin = KL_s_img[idx].mean(axis=0)

        # 仅对 fit_mask 区间拟合
        beta_u_bin, _, R2_u_bin = fit_beta_loglog(alphas[fit_mask], np.maximum(mu_u_bin[fit_mask], 1e-300), None)
        beta_s_bin, _, R2_s_bin = fit_beta_loglog(alphas[fit_mask], np.maximum(mu_s_bin[fit_mask], 1e-300), None)

        # 桶内 bootstrap：重采样该桶的图像集合
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
    # 保存 CSV：每个 alpha 的 KL 均值/SE + RS-AUC
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

    # 保存 summary JSON
    js_path = os.path.join(args.outdir, "summary_beta_mst.json")
    with open(js_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {js_path}")

    # -----------------------------
    # 画图：KL-loglog / RS-AUC / beta-by-bin
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

    # 保存可读性更强的 txt 汇总
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

    # 终端打印简短总结
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