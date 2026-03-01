# LSB 图像隐写中的可检测性、容量与自适应设计

本仓库包含以下研究工作的实验代码：

**Detectability, Capacity, and Adaptive Design in LSB Image Steganography**

本项目研究 LSB 嵌入的基本统计尺度律，
在 KL 安全约束下推导其渐近行为，
并在合成 i.i.d. 模型和真实自然图像（BOWS2）上对理论进行验证。

---

## 仓库结构

| 脚本                       | 功能                      | 模型类型            |
| ------------------------ | ----------------------- | --------------- |
| `iidsimulation.py`       | 合成 KL 尺度律验证             | i.i.d. 概率质量函数模型 |
| `bows2_beta_mst_test.py` | β 指数实证检验（MST-1 / MST-3） | 真实自然图像          |
| `spatial_demo.py`        | 空间相关性演示                 | 结构性示意           |

---

# iidsimulation.py

（合成 KL 尺度律验证）

该程序在 i.i.d. 像素分布模型下验证理论 KL 尺度律。

实现内容包括：

* 均匀 LSB flipping（第三章模型）
* 基于直方图的自适应 flipping（第四章模型）
* KL 的蒙特卡洛估计
* KL 的解析表达对比
* log-log 幂指数拟合

图像尺寸概念上固定为 256×256。
总样本规模 `N` 和蒙特卡洛样本规模 `M` 可通过参数配置。

---

## 使用示例

运行均匀与自适应实验：

```bash
python iidsimulation.py
```

仅运行均匀实验：

```bash
python iidsimulation.py --task uniform
```

仅运行自适应实验：

```bash
python iidsimulation.py --task adaptive
```

自定义实验参数：

```bash
python iidsimulation.py \
    --N 8000000 \
    --M 3000000 \
    --trials 30 \
    --seed 2026
```

---

## 输出文件

均匀实验输出：

* `uniform_kl_total_scale.png`
* `uniform_kl_normalized_ratio.png`
* `uniform_kl_results.csv`

自适应实验输出：

* `adaptive_kl_vs_rho.png`
* `adaptive_normalized_ratio.png`
* `adaptive_ch4_results.csv`

---

# bows2_beta_mst_test.py

（真实图像 β 指数验证）

该脚本在 BOWS2 数据集上进行实证检验。

实现内容包括：

* MST-1：全局幂指数 β 估计
* MST-3：基于纹理分组的指数分析
* RS 隐写分析 AUC 比较
* Bootstrap 置信区间估计

---

## 使用示例

```bash
python bows2_beta_mst_test.py \
    --bows2_dir /path/to/BOWS2 \
    --M 2000 \
    --alpha_min 0.0005 \
    --alpha_max 0.01 \
    --alpha_num 12 \
    --bootstrap 400
```

---

## 输出文件

* `table_alpha_kl_auc_beta_mst.csv`
* `summary_beta_mst.json`
* `summary_beta_mst.txt`
* `kl_loglog_with_fit_region.png`
* `rs_auc_logx.png`
* `beta_by_texture_bins.png`

---

# spatial_demo.py

（空间相关性演示）

该程序用于展示：

* 自然图像具有强邻接像素相关性
* 像素随机打乱保持直方图不变，但破坏空间结构
* 仅基于直方图的模型忽略了结构信息

---

## 使用方法

```bash
python spatial_demo.py \
    --input example.png \
    --outdir results \
    --seed 0
```

输出：

* `spatial_demo.png`

---

# 运行环境要求

Python ≥ 3.9

安装依赖：

```bash
pip install numpy matplotlib tqdm imageio scikit-learn
```

---

# 数据集说明

该脚本在 BOWS2 数据集上进行实证检验。

本仓库不包含 BOWS2 数据集。

你需要自行从官方来源或学术镜像下载 BOWS2 数据集。

下载后解压，并通过以下参数指定路径：

```
--bows2_dir /path/to/BOWS2
```

---

# 联系方式

Steve Shifan Wang
University of Bristol

```

