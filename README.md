# TSAD Evaluation Transfer

Reproducible code for a controlled empirical study of whether conclusions from retrospective time-series anomaly-detection (TSAD) evaluation transfer to sequential, deployment-oriented evaluation on a multivariate fault-injection dataset.

## Study design

The detector implementation, detector parameters, training data, and input variables are held fixed across protocols. Every trainable detector is fit only on the designated labeled-normal reference batch. Both protocols treat each non-reference process/simulation batch as an independent evaluation run; evaluation batches are never concatenated before scoring. The frozen detector scores each batch independently using only information available through each timestamp. The same causal timestamp-score vector feeds both views: retrospective evaluation computes score-based metrics for each complete batch, while online evaluation applies a fixed threshold estimated exclusively from reference-batch scores to obtain operational metrics.

The two protocols use metrics appropriate to their evaluation setting rather than treating a retrospective metric computed after sequential scoring as an "online" metric. Retrospective evaluation reports AUC-PR and VUS-PR using the TSB-AD-derived evaluation implementation. Online evaluation uses the deployment metrics retained from the existing benchmark code, including event detection, point false-positive rate, contiguous false-positive events, and delay. Raw scores are archived so secondary sensitivity analyses can be performed without retraining.

## Detector set

Model selection is fixed a priori by methodological coverage rather than target-dataset performance. The study contains 15 detectors:

| Group | Detectors |
|---|---|
| Contemporary pretrained | Chronos2-FT |
| Reconstruction-based | AutoEncoder, USAD |
| Forecasting-based | LSTMAD, CNN |
| Encoding-based | PCA, RobustPCA |
| Tree-based | Isolation Forest, EIF |
| Distribution-based | MCD, OCSVM |
| Clustering-based | KMeansAD, CBLOF |
| Proximity-based | KNN, LOF |

The detector source files are based on the supplied TSB-AD implementations. The study layer provides configuration, protocol orchestration, metric computation, and analysis, while the detector adaptations required for causal evaluation are documented in `TSB_AD_PATCHES.md`.

## Data

The dataset is not included. Place the batch CSV/XLSX/XLS files directly in a local data directory. The configured reference batch is:

`BEACON_SimMode_IC_Batch11_ProcessNoise_ON_MeasNoise_ON_clean`

The loader expects the eight variables listed in `configs/study.yaml` and derives anomaly labels from the dataset's `*_active` fault columns. The reference batch is required to contain no labeled anomalies. All remaining batches are evaluation batches.

## Install

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[foundation,test]"
```

`numba` is a declared runtime dependency because it is required by the vendored TSB-AD support code. Foundation-model checkpoints must be available according to the model identifiers/paths in `configs/study.yaml`.

For the publication run, create the exact environment on the target CUDA machine and archive it with:

```bash
pip freeze > requirements-lock.txt
```

## Run

```bash
python scripts/run_study.py \
  --config configs/study.yaml \
  --data data \
  --output results/main
```

The full publication run evaluates all 15 configured detectors. `data/` and `results/` are gitignored by default.

## Outputs

Each model directory contains `retrospective_scores.csv`, `retrospective_metrics_overall.csv`, `retrospective_metrics_by_fault.csv`, `retrospective_metrics_by_batch.csv`, `sequential_scores.csv`, `sequential_metrics_by_batch.csv`, and `run.json`.

Retrospective evaluation is performed **per independent batch/run**. Each batch is passed to the frozen detector separately and AUC-PR and VUS-PR are computed for that batch using the TSB-AD-derived metric implementation. Fault-level and overall retrospective results are macro-averages of these batch-level metrics. Independent runs are not concatenated into an artificial continuous time series.

At the result root, `retrospective_by_batch.csv`, `retrospective_by_fault.csv`, and `retrospective_overall.csv` provide the retrospective levels. `sequential_by_batch.csv`, `sequential_by_fault.csv`, and `sequential_overall.csv` provide the deployment-oriented levels. Fault and overall tables are macro-aggregated from batch-level metrics. The sequential tables contain deployment-oriented threshold metrics rather than retrospective score metrics recomputed after sequential scoring.

## Reproducibility rules

The reference batch must contain no labeled anomalies. Model selection and parameters are fixed in YAML before target-data results are inspected. Evaluation labels are never used for sequential threshold selection. No detector is retrained between evaluation batches. Each evaluation batch is scored once by the same frozen detector, and the resulting scores are shared by the retrospective and sequential evaluation views. Raw scores and sequential binary predictions are archived. Any detector incompatibility discovered during execution should be documented rather than silently repaired or replaced after observing performance.

Sequential evaluation uses timestamp-causal scores: the score assigned to timestamp `t` is constructed without observations after `t`. Models requiring temporal context emit `NaN` until sufficient context is available; these timestamps are treated as unscorable.

## Provenance

Files under `src/TSB_AD/` are vendored from the TSB-AD model/support files supplied for this study and retain their upstream source comments. Preserve the upstream TSB-AD attribution and license when redistributing them. Study-specific orchestration is under `src/tsad_transfer/`. `basic_metrics.py` is adapted directly from the TSB-AD evaluation implementation and is marked accordingly in the source. `retrospective_metrics.py` is the study-specific wrapper around those metrics. `online_metrics.py` contains the deployment-oriented metrics retained from the existing benchmark code.


## Causal scoring

Evaluation-time preprocessing does not use future observations. Forecasting/reconstruction models that normalize inputs reuse statistics derived from the clean reference batch. Window-based detector scores are aligned to the end of each window; timestamps before sufficient context is available are represented as `NaN` and treated as unscorable. The VUS window is resolved once at run start as the median inclusive duration (in samples) of contiguous ground-truth anomaly events across the evaluation batches. The resolved integer is written to `resolved_config.json` before any model is run. The final study roster contains 15 detectors. TSPulse is not included because causality of its underlying full-batch scorer could not be established without a more invasive model-specific modification.
