# Wildfire Risk Portugal

Weekly wildfire occurrence prediction for Mainland Portugal on a 5 × 5 km grid, combining
historical fire records, ERA5-Land weather and Sentinel-2 imagery — from tabular baselines
to a convolutional neural network.

> **Main finding.** Wildfire occurrence is dominated by *where and when* fires usually
> happen. Weather adds a robust, moderate improvement. Satellite imagery adds a **small but
> genuine** signal — but only when a CNN reads the full image; tabular vegetation indices
> did not achieve it. The image contribution replicated on two unseen test years, including
> after controlling for the ensemble effect.

---

## The question

Given everything known on a Monday, what is the probability that each 5 km cell records
**at least one new wildfire in the following 7 days**? And does satellite imagery of
vegetation state add information beyond weather and fire history?

| | |
|---|---|
| Study area | Mainland Portugal, 3,822 grid cells (5 × 5 km) |
| Period | 2017–2025, fire season May–October |
| Unit | cell × week (Monday reference date) — 875,238 observations |
| Target | ≥ 1 new wildfire in the next 7 days (rekindles excluded) — 5.61% positive |
| Split | train 2017–2022 · validation 2023 · **test 2024–2025 (used once)** |

## Data

All sources are free.

| Source | Use |
|---|---|
| ICNF wildfire records (SGIF), 2011–2025 | target, fire persistence, historical climatology |
| CAOP 2025 (DGT) | Mainland Portugal boundary |
| ERA5-Land daily aggregates (Google Earth Engine) | 16 antecedent weather features (temperature, precipitation, soil moisture, VPD, wind, dryness) |
| Sentinel-2 surface reflectance + cloud probability (Google Earth Engine) | NDVI/NDMI features and monthly image composites |
| CORINE Land Cover 2018 | static vegetation fraction |

Raw and processed data are not included in this repository (several tens of GB).

## Method

```text
01  Fire records ──► grid assignment, weekly target, persistence features
02  ERA5-Land ─────► antecedent weather features (7–30 days before each Monday)
03  Sentinel-2 ────► NDVI/NDMI statistics and trends; static land-cover features
04  Tabular modelling: baselines → logistic regression → XGBoost ablation ladder
05  Monthly Sentinel-2 composites (B4, B8, B11; 40 m) → 125 × 125 patches per cell
06  CNN on Kaggle (free GPU): ResNet-18 image branch + tabular branch
07  Combination test, calibration and the single evaluation on the test period
```

**Design choices that matter for the conclusions**

- **Strict temporal split** and leakage-free features: every predictor uses only
  information available before the reference Monday; the image of a week is the composite
  of the *previous* month.
- **A real climatological baseline** (historical fire rate per cell and month, computed
  leave-one-year-out inside the training period).
- **Ablation by feature group** (weather / context / Sentinel), each model tuned
  separately with expanding-window temporal cross-validation.
- **Paired block bootstrap over weeks** for every comparison, since cells in the same week
  share weather and are not independent.
- **An ensemble-effect control**: the image network is compared with an identical network
  without the image, and both are combined with XGBoost, to separate the information in the
  image from the benefit of averaging two different models.
- **Decisions fixed before the test**: the primary model, the calibration procedure and the
  metrics were written down before 2024–2025 was used.

## Results

### Test period (2024–2025), Average Precision

| Model | AP |
|---|---:|
| Prevalence baseline (random) | 0.047 |
| Persistence baseline (fires in previous 4 weeks) | 0.144 |
| Climatological baseline | 0.193 |
| XGBoost — weather only | 0.104 |
| XGBoost — weather + Sentinel indices | 0.146 |
| XGBoost — context only | 0.230 |
| XGBoost — weather + context | 0.250 |
| XGBoost — weather + context + Sentinel indices | 0.253 |
| CNN — tabular branch only | 0.253 |
| CNN — image + tabular | 0.261 |
| **Primary model: XGBoost (weather + context) + CNN image, 50/50** | **0.259** |

*Context* = historical climatology, fire persistence, vegetation and land fraction.

### Does the image add information? (paired block bootstrap, test period)

| Comparison | ΔAP | 95% CI |
|---|---:|---|
| CNN tabular → CNN image | +0.0075 | [+0.0049, +0.0103] |
| XGB + CNN tabular → XGB + CNN image *(ensemble control)* | +0.0049 | [+0.0034, +0.0065] |
| XGB weather + context → primary model | +0.0085 | [+0.0056, +0.0120] |
| Climatology → XGB weather + context | +0.0568 | [+0.0430, +0.0715] |

### Operational view

Flagging the **top 5% of cells each week** (191 of 3,822):

| | Precision | Share of fire cell-weeks captured |
|---|---:|---:|
| Random | 4.7% | 5.0% |
| Climatological baseline | 24.1% | 25.5% |
| **Primary model** | **26.4%** | **28.0%** |

Share of individual fires occurring in a flagged cell, by fire size:

| | All | ≥ 1 ha | ≥ 10 ha | ≥ 100 ha |
|---|---:|---:|---:|---:|
| Climatological baseline | 33.2% | 25.1% | 23.8% | 19.8% |
| **Primary model** | **37.0%** | **28.7%** | **27.1%** | **23.5%** |

The model mainly anticipates where fires **start**; whether a fire becomes large depends on
spread conditions and suppression, which are not modelled.

### Calibration

Platt scaling fitted on out-of-sample predictions for 2020–2023. On the test period the
calibrated model predicts 4.69% average risk against 4.72% observed (Brier Skill Score
≈ 0.12 vs. the prevalence baseline). Calm years are slightly overestimated and active years
underestimated.

## Lessons learned

1. **Baselines first.** An early version reported a +62% gain from Sentinel-2 features. It
   was a proxy for *location*: those features identified cells that burn often. A proper
   climatological baseline showed the gain disappeared once fire history was known.
2. **Control for the ensemble effect.** Averaging two models often helps even without new
   information. Comparing against the same combination without the image isolated the
   image's contribution.
3. **Pre-register decisions.** On the test period, the CNN alone was marginally (not
   significantly) better than the pre-registered primary model. The primary model was kept,
   because switching after seeing the test would turn the test into a validation set.

## Repository structure

```text
notebooks/
├── 01_fire_data_exploration.ipynb   fire records, grid, weekly target, persistence
├── 02_weather_era5_land.ipynb       ERA5-Land weather features (Earth Engine)
├── 03_sentinel2_exploration.ipynb   Sentinel-2 NDVI/NDMI features, static features
├── 04_exploratory_analysis.ipynb    EDA, baselines, XGBoost ablation, calibration
├── 05_monthly_composites.ipynb      monthly composites, patches, Kaggle upload
├── 06_cnn_training.ipynb            CNN training (run on Kaggle)
└── 07_final_evaluation.ipynb        combination test, calibration, test evaluation
scripts/
└── extract_sentinel.py              standalone Sentinel-2 feature extraction (Notebook 03)
requirements.txt                     Python dependencies for the local notebooks
data/                                not versioned (raw, interim, processed)
```

## Reproducing the project

Everything runs on free tools.

- **Python** with the packages in `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```
- **Google Earth Engine** (free non-commercial project) for ERA5-Land and Sentinel-2.
  Weather and Sentinel feature extraction is slow (days); monthly composites are exported
  as national rasters and cut locally.
- **Kaggle** (free GPU tier, phone verification required) for notebook 06, with PyTorch and
  torchvision. The uint8 patches (~9.8 GB) are uploaded as a private Kaggle dataset with the
  Kaggle CLI.

Run the notebooks in order (01 → 07). Notebook 06 runs on Kaggle; its outputs (predictions
and weights) are downloaded and used by notebook 07.

## Limitations and future work

**Limitations.** The target counts all occurrences, dominated by small human-caused
ignitions. Weather features describe the week *before* the target week, and wind and
humidity are derived from daily means. Monthly composites can be up to about five weeks
old, with residual cloud artefacts in the weakest months (2017). Calibration cannot
anticipate unusually calm or active seasons.

**Future work.** Weather forecasts for the target week; a target based on fire size or
burned area; hourly weather variables; spatial cross-validation to test whether the image
signal generalises to regions without fire history.

---

*Author: Daniel Fonseca — Data Science, ISCTE-IUL.*
