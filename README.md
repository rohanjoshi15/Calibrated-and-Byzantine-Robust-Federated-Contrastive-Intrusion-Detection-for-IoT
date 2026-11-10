# Calibrated and Byzantine-Robust Federated Contrastive Intrusion Detection for IoT

Code and results for a federated contrastive intrusion detector for IoT networks. The detector extends FeCo
(Wang et al., IEEE TDSC 2025) with:

- per-gateway split-conformal thresholds, which bound each client's false-positive rate at a chosen level α;
- multi-prototype benign modelling: spherical k-means and vMF-mixture prototypes over the embedding space,
  computed centrally or federated;
- Byzantine-robust aggregation (trimmed mean, coordinate-wise median, geometric median, Rob-FCP-style robust quantile)
  against three attacks: A1 whitelisting model poisoning, A2 prototype poisoning and A3 calibration poisoning.

The repository also contains a FeCo reimplementation, a leakage-free evaluation protocol, and baselines (Deep SVDD,
Deep SAD, autoencoder, Isolation Forest, CLAD-style scoring) on NSL-KDD, N-BaIoT and CICIoT2023. The FL loop is an
in-process simulation with FedAvg, FedProx and MOON.

## Repository layout

```
src/           library code: data loading and splits, models and losses, FL, calibration, Byzantine attacks, metrics
experiments/   entry points: run.py (one run), sweep.py (a grid), scripts that build tables and figures
configs/       experiment grids (one Python module per grid) and the validation-selected hyper-parameters (*_selected.json)
tests/         unit tests (pytest)
results/       config.yaml + metrics.json of every run, and the result tables and figures
```

## Install

Python 3.10 (tested with torch 2.6.0).

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the torch build that matches your CUDA driver if you want to run on a GPU.

## Datasets

The datasets are not redistributed. Download them from the official sources:

- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
- N-BaIoT: https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot
- CICIoT2023: https://www.unb.ca/cic/datasets/iotdataset-2023.html (per-pcap CSV files)

Expected layout:

```
data/raw/nslkdd/KDDTrain+.txt
data/raw/nslkdd/KDDTest+_full.txt              the official 43-column KDDTest+.txt, renamed
data/raw/nbaiot/<device>/benign_traffic.csv    for each of the 9 devices
data/processed/nbaiot_extracted/<device>/gafgyt_attacks/*.csv    gafgyt_attacks.rar extracted
data/processed/nbaiot_extracted/<device>/mirai_attacks/*.csv     mirai_attacks.rar extracted (7 devices)
data/raw/ciciot2023/<class>/*.pcap.csv         one folder per class
```

`data/raw/nslkdd/KDDTest+.txt` is only read with `data.test_file: mirror` (a 42-column binary-label copy used for
one CPU-vs-GPU equivalence check). For CICIoT2023, place only the classes listed in `FAMILY` in
`src/data/ciciot.py` under `data/raw/ciciot2023/`; the converter rejects any other class folder.

The default locations are `data/raw` and `data/processed` relative to the working directory. Set `DATA_RAW` and
`DATA_PROCESSED` to use other locations.

Convert the raw files (writes parquet files, split indices and audit files to `data/processed/`):

```
python -m src.data.nslkdd
python -m src.data.nbaiot
python -m src.data.ciciot
```

Then run the tests:

```
pytest -q
```

Some tests read the processed data.

## Reproducing results

Run all commands from the repository root. Seeds 0-4 are used throughout; the H1 paired study (`T_H1_*`) adds seeds 5-9.

A single run takes a YAML config and writes `results/<phase>/<run_id>/{config.yaml, metrics.json, stdout.log}`. It is
skipped if `metrics.json` already exists:

```
python experiments/run.py <config.yaml> [--force]
```

A grid module in `configs/` defines `configs()`; the `STAGE` environment variable picks the stages (comma-separated,
listed in each module's docstring). `sweep.py` runs a grid in parallel worker processes and skips finished runs:

```
STAGE=<stage> python experiments/sweep.py configs/<grid>.py [--workers N] [--device cpu|cuda]
```

Hyper-parameters were tuned on validation splits only. The selected values are committed in `configs/*_selected.json`,
so the `tune_*` stages and `experiments/select.py` do not need to be re-run.

The runs in `results/` were produced as follows. Most runs come from the GPU grid; every config in it sets
`device: cuda`:

```
STAGE=check,tune_cic,nsl,nba,cic,h1,fedcic,cut_b python experiments/sweep.py configs/phase8_grid.py
```

The N-BaIoT multi-prototype and conformal runs, and the tuning runs, are CPU runs (their `metrics.json` has no
`device` key). Run them after the GPU grid, so that runs already finished by the GPU grid are skipped:

```
STAGE=central,fed python experiments/sweep.py configs/phase3_grid.py --device cpu
python experiments/sweep.py configs/phase4_grid.py --device cpu
STAGE=tune_central,tune_fl python experiments/sweep.py configs/phase1_grid.py --device cpu
STAGE=tune_nba python experiments/sweep.py configs/phase3_grid.py --device cpu
STAGE=tune_mu python experiments/sweep.py configs/phase6_grid.py --device cpu
```

Tables and figures are built from `results/**/metrics.json`, in this order:

```
python experiments/tables_phase1.py     # T_repro.csv/.tex
python experiments/table_tau_sweep.py   # T_tau_sweep_val.csv
python experiments/tables_phase2.py     # T1_feco_reported_vs_corrected, T1b_leakage_selection, T1c_loco_zeroday
python experiments/table_T1d.py         # T1d_val_per_category_recall
python experiments/tables_phase3.py     # T3_central_K, T3_fed_K, T3_privacy_sigma; F_K_tpr1, F_privacy_utility
python experiments/tables_h1.py         # T_H1_valselected, T_H1_valselected_perseed, T_H1_fixedK_sensitivity
python experiments/tables_phase4.py     # T4_*; F_fpr_vs_alpha, F_violation_vs_heterogeneity
python experiments/tables_phase5.py     # T5_whitelist, T5_proto, T5_calib; F_byzantine
python experiments/tables_phase6.py     # T6_central, T6_fl, T6_lofo, T6_perdevice; F_zeroday_lofo
python experiments/latency.py           # T6_latency (trains one model, then times CPU inference)
python experiments/make_tex.py          # results/tables/*.tex
```

The `T8_*` tables (`T8_cpu_vs_gpu*`, `T8_gpu_determinism`, `T8_changes_v1_v2`) come from
`experiments/phase8_equiv.py` and `experiments/summary_numbers.py`. Both compare the GPU runs with an archive of
earlier CPU runs in `results/_v1/`, which is not part of this repository, so they cannot be regenerated from it.

## Results

`results/tables/` (CSV, and LaTeX for the main tables):

- `T_repro`: FeCo reported numbers vs our reproduction (NSL-KDD, central, FL and self-learning).
- `T_tau_sweep_val`: FeCo τ / momentum / batch-size / epoch tuning grid, validation metrics.
- `T1_feco_reported_vs_corrected`: FeCo protocol vs the leakage-free protocol.
- `T1b_leakage_selection`: optimism of selecting hyper-parameters on test instead of validation.
- `T1c_loco_zeroday`: leave-one-category-out zero-day recall, NSL-KDD.
- `T1d_val_per_category_recall`: per-category recall on the NSL-KDD validation split.
- `T3_central_K`, `T3_fed_K`: multi-prototype detectors by K, central and federated.
- `T3_privacy_sigma`: detection vs Gaussian noise on shared centroids.
- `T_H1_valselected`, `T_H1_valselected_perseed`, `T_H1_fixedK_sensitivity`: H1 paired test, K=1 vs the val-selected multi-prototype detector.
- `T4_fpr_control`, `T4_fpr_control_perseed`, `T4_fpr_control_K4`, `T4_fpr_perclient`: per-client FPR control of the threshold methods.
- `T4_calib_size`: FPR control vs calibration-set size.
- `T5_whitelist`, `T5_proto`, `T5_calib`: Byzantine attacks A1-A3 vs defences.
- `T6_central`, `T6_fl`: all methods and datasets, central and FedAvg/FedProx/MOON.
- `T6_lofo`: leave-one-family-out zero-day recall, CICIoT2023.
- `T6_perdevice`: FeCo protocol per N-BaIoT device.
- `T6_latency`: CPU inference latency, fp32 vs int8.
- `T8_cpu_vs_gpu`, `T8_cpu_vs_gpu_nbaiot_reruns`, `T8_gpu_determinism`, `T8_changes_v1_v2`: CPU-vs-GPU equivalence and determinism checks.

`results/figures/` (PNG and PDF):

- `F_fpr_vs_alpha`: empirical per-client FPR vs target α.
- `F_violation_vs_heterogeneity`: share of clients violating the FPR bound vs data heterogeneity.
- `F_K_tpr1`: TPR at 1% FPR vs number of prototypes K.
- `F_privacy_utility`: detection vs centroid noise σ.
- `F_byzantine`: Byzantine attacks and defences.
- `F_zeroday_lofo`: zero-day recall on held-out families.

`results/<phase>/<run_id>/` holds the `config.yaml` and `metrics.json` of every run behind these tables.

## Authors

Rohan Joshi, Pavan Kumar, Meera R, under the guidance of Prof. Pradeep Kumar Bhale (IIITDM Kancheepuram).

## Citation

To appear. Placeholder entry:

```
@misc{joshi_calibrated_byzantine_fedids,
  title  = {Calibrated and Byzantine-Robust Federated Contrastive Intrusion Detection for IoT},
  author = {Joshi, Rohan and Kumar, Pavan and R, Meera},
  note   = {To appear}
}
```
