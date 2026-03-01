# Detectability, Capacity, and Adaptive Design in LSB Image Steganography

This repository contains the experimental code accompanying the study:

**Detectability, Capacity, and Adaptive Design in LSB Image Steganography**

The project investigates the fundamental statistical scaling law of LSB embedding,
derives its asymptotic behavior under KL-security constraints,
and validates the theory on both synthetic i.i.d. models and real natural images (BOWS2).


## Repository Structure

| Script | Purpose | Model Type |
|--------|----------|------------|
| `iidsimulation.py` | Synthetic KL scaling law verification | i.i.d. pmf model |
| `bows2_beta_mst_test.py` | Empirical β exponent test (MST-1 / MST-3) | Real natural images |
| `spatial_demo.py` | Spatial correlation demonstration | Structural illustration |


# iidsimulation.py  
(Synthetic KL Scaling Verification)

This program verifies the theoretical KL scaling law under an i.i.d. pixel distribution model.

It implements:

- Uniform LSB flipping (Chapter 3 model)
- Histogram-adaptive flipping (Chapter 4 model)
- Monte Carlo estimation of KL
- Analytic KL comparison
- Log-log exponent fitting

Image size is conceptually fixed at 256×256.  
Total sample size `N` and MC size `M` are configurable.

## Example Usage

Run both uniform and adaptive experiments:

```bash
python iidsimulation.py
````

Run only uniform experiment:

```bash
python iidsimulation.py --task uniform
```

Run only adaptive experiment:

```bash
python iidsimulation.py --task adaptive
```

Custom experiment:

```bash
python iidsimulation.py \
    --N 8000000 \
    --M 3000000 \
    --trials 30 \
    --seed 2026
```

## Output

Uniform experiment:

* `uniform_kl_total_scale.png`
* `uniform_kl_normalized_ratio.png`
* `uniform_kl_results.csv`

Adaptive experiment:

* `adaptive_kl_vs_rho.png`
* `adaptive_normalized_ratio.png`
* `adaptive_ch4_results.csv`

---

# bows2_beta_mst_test.py

(Real-Image β Exponent Validation)

This script performs empirical validation on the BOWS2 dataset.

It implements:

* MST-1: Global power-law exponent β estimation
* MST-3: Texture-bin dependent exponent analysis
* RS steganalysis AUC comparison
* Bootstrap confidence intervals

## Example Usage

```bash
python bows2_beta_mst_test.py \
    --bows2_dir /path/to/BOWS2 \
    --M 2000 \
    --alpha_min 0.0005 \
    --alpha_max 0.01 \
    --alpha_num 12 \
    --bootstrap 400
```

## Output

* `table_alpha_kl_auc_beta_mst.csv`
* `summary_beta_mst.json`
* `summary_beta_mst.txt`
* `kl_loglog_with_fit_region.png`
* `rs_auc_logx.png`
* `beta_by_texture_bins.png`

---

# spatial_demo.py

(Spatial Correlation Demonstration)

This program demonstrates that:

* Natural images exhibit strong adjacent pixel correlation
* Random pixel shuffling preserves histogram but destroys spatial structure
* Histogram-only models ignore structural information

## Usage

```bash
python spatial_demo.py \
    --input example.png \
    --outdir results \
    --seed 0
```

Output:

* `spatial_demo.png`

---

# Requirements

Python ≥ 3.9

Install dependencies:

```bash
pip install numpy matplotlib tqdm imageio scikit-learn
```

This script performs empirical validation on the BOWS2 dataset.

The BOWS2 dataset is not included in this repository.
You must download it manually from the official source or academic mirrors.

After downloading, extract the dataset and provide its directory path using:

--bows2_dir /path/to/BOWS2


---

# Contact

Steve Shifan Wang
University of Bristol

