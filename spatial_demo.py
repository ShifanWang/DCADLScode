import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import argparse
import os

# --------------------------------------------------
# 1. 读取图像
# --------------------------------------------------
def load_grayscale(path):
    img = imageio.imread(path)
    if img.ndim == 3:
        # 转灰度
        img = np.mean(img, axis=2)
    return img.astype(np.float64)


# --------------------------------------------------
# 2. 随机打乱像素（保持直方图不变）
# --------------------------------------------------
def shuffle_image(img, seed=0):
    rng = np.random.default_rng(seed)
    flat = img.flatten().copy()
    rng.shuffle(flat)
    return flat.reshape(img.shape)


# --------------------------------------------------
# 3. 计算水平相邻像素相关系数
# --------------------------------------------------
def adjacent_correlation(img):
    x = img[:, :-1].flatten()
    y = img[:, 1:].flatten()
    return np.corrcoef(x, y)[0, 1]


def plot_demo(orig, shuffled, outdir):
    os.makedirs(outdir, exist_ok=True)

    corr_orig = adjacent_correlation(orig)
    corr_shuf = adjacent_correlation(shuffled)

    print(f"Original adjacent correlation: {corr_orig:.4f}")
    print(f"Shuffled adjacent correlation: {corr_shuf:.4f}")

    # 使用 constrained_layout 避免裁剪
    fig, axes = plt.subplots(
        1, 3,
        figsize=(13, 4.5),
        constrained_layout=True
    )

    # 原图
    axes[0].imshow(orig, cmap="gray")
    axes[0].set_title(f"Original\ncorr={corr_orig:.3f}", fontsize=12)
    axes[0].axis("off")

    # shuffle 图
    axes[1].imshow(shuffled, cmap="gray")
    axes[1].set_title(f"Shuffled\ncorr={corr_shuf:.3f}", fontsize=12)
    axes[1].axis("off")

    # 散点图
    x = orig[:, :-1].flatten()
    y = orig[:, 1:].flatten()
    idx = np.random.choice(len(x), size=min(20000, len(x)), replace=False)

    axes[2].scatter(x[idx], y[idx], s=1, alpha=0.3)
    axes[2].set_xlabel("Pixel value")
    axes[2].set_ylabel("Right neighbor")
    axes[2].set_title("Adjacent Scatter", fontsize=12)

    # 保存时防裁剪
    fig.savefig(
        os.path.join(outdir, "spatial_demo.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1
    )

    plt.close(fig)

    print(f"Saved to {os.path.join(outdir, 'spatial_demo.png')}")



# --------------------------------------------------
# 主函数
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to grayscale image")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    img = load_grayscale(args.input)
    shuffled = shuffle_image(img, seed=args.seed)

    plot_demo(img, shuffled, args.outdir)


if __name__ == "__main__":
    main()