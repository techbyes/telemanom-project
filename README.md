# Spacecraft Telemetry Anomaly Detection 

**Author:** Esra (Dataset & Preprocessing)

## Abstract

For this project I prepared the Telemanom dataset for anomaly detection on spacecraft telemetry. The data comes from HuggingFace ([appleparan/telemanom](https://huggingface.co/datasets/appleparan/telemanom)) and is based on real NASA measurements from the SMAP satellite and the MSL (Curiosity) rover. The goal is to tell normal behavior apart from anomalies in long multivariate time series. In this repo I load the channels, handle missing values, apply StandardScaler, build fixed-length windows with a sliding window, and split the data in an autoencoder-safe way: training and validation come only from normal train CSVs (chronological val split, no shuffling), while the labeled test CSV is kept separate for evaluation. I also ran exploratory analysis in a Jupyter notebook. Model training is not included here—that is handled by another team member.

## Dataset Description

**HuggingFace:** [appleparan/telemanom](https://huggingface.co/datasets/appleparan/telemanom)

**Origin:** Telemanom benchmark (Hundman et al., KDD 2018) — [paper](https://arxiv.org/abs/1802.04431), [GitHub](https://github.com/khundman/telemanom)

The dataset contains telemetry from two NASA missions:

- **SMAP** — Soil Moisture Active Passive (Earth satellite)
- **MSL** — Mars Science Laboratory (Curiosity rover)

Each channel is a multivariate time series: column `value` is the main signal, and `cmd_*` columns are one-hot command features. The HuggingFace version has train and test splits per channel; anomaly intervals are listed in `labeled_anomalies.csv`.

## Problem Definition

The task is **anomaly detection in spacecraft telemetry** — finding time windows where the signal does not look normal. Because measurements depend on previous timesteps, the data has strong **temporal structure**; windows must not be shuffled randomly. An autoencoder must train on **normal data only**, so train and test streams are never mixed. Validation is taken from the tail of each channel’s train windows in time order. The held-out test CSV (with labeled anomaly intervals) is used only for evaluation.

## Dataset Characteristics

| Property | Description |
|----------|-------------|
| Channels | 82 in total |
| Spacecraft | 55 SMAP, 27 MSL |
| Features | SMAP: 25 dims; MSL: 55 dims |
| Labels | Binary — normal (0) / anomaly (1) at window level |
| Imbalance | Most windows are normal |
| Padding | SMAP channels padded to 55 features to match MSL |

## Preprocessing Pipeline

Implemented in `preprocessing.py` and called from `data_loader.py`:

1. **Missing values** — column mean imputation, then `nan_to_num`
2. **Normalization** — `StandardScaler` fit on HF train splits only
3. **Padding** — SMAP (25 features) zero-padded to 55
4. **Sequences** — sliding window, length 50, stride 10
5. **Labels** — train windows → normal; test windows → anomaly if any timestep in window is labeled anomalous
6. **Split** — train windows → chronological train / val (default 85% / 15% of each channel’s train stream); test windows → separate `X_test` / `y_test` from HF test CSVs (no concatenation, no shuffle)

## Exploratory Data Analysis

In `dataset_analysis.ipynb` I included:

- Class distribution (normal vs anomaly, spacecraft breakdown)
- Example time-series plot with anomaly regions highlighted
- Sequence length statistics across channels
- Class imbalance on the test set; sanity check that train/val contain no anomaly labels

## Project Structure

```
.
├── preprocessing.py
├── data_loader.py
├── dataset_analysis.ipynb
├── requirements.txt
└── README.md
```

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook dataset_analysis.ipynb
```

First run downloads data from HuggingFace (needs internet). Run all cells in the notebook to see the plots.

## Role Clarification

**This repo is only my data-preparation part.** I do not train any models here. The output is preprocessed arrays and PyTorch DataLoaders that the modeling part of our group can use.

## Bonus Justification

- Loaded data directly from **HuggingFace** (no manual `.npy` download)
- Used **real NASA telemetry** (SMAP + MSL), not a toy dataset
- Full pipeline from raw channels to batched tensors
- EDA with imbalance analysis on the labeled test stream; leakage-safe splits for autoencoder training

## Reference

```bibtex
@inproceedings{hundman2018detecting,
  title={Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding},
  author={Hundman, Kyle and Constantinou, Valentino and Laporte, Christopher and Colwell, Ian and Soderstrom, Tom},
  booktitle={KDD}, year={2018}
}
```
