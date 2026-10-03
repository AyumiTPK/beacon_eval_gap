# Time Series Evaluation Gap Study

A controlled empirical study of whether conclusions from retrospective time series anomaly detection evaluation transfer to sequential, deployment-oriented evaluation on a multivariate fault-injection dataset.

## Study design

The same trained detectors are used for both evaluation protocols.

- **Retrospective evaluation:** evaluates complete test batches using threshold-independent metrics (AUC-PR and VUS-PR).
- **Deployment-oriented evaluation:** converts scores into alarms using a fixed threshold calibrated exclusively on the normal reference batch, then evaluates event detection, false positives, and detection delay.
- **Controlled comparison:** models, parameters, training data, and test batches are identical.


## Detector set

Model selection is fixed a priori by methodological coverage rather than target-dataset performance. The study contains 16 detectors:

| Group | Detectors |
|---|---|
| Contemporary pretrained | Chronos2 (fine-tuned), TimesFM3 (zero-shot) |
| Reconstruction-based | AutoEncoder, USAD |
| Forecasting-based | LSTMAD, CNN |
| Encoding-based | PCA, RobustPCA |
| Tree-based | Isolation Forest, EIF |
| Distribution-based | MCD, OCSVM |
| Clustering-based | KMeansAD, CBLOF |
| Proximity-based | KNN, LOF |

The detector source files are based on the TSB-AD implementations. 

## Data

The dataset is not included. Place the batch CSV/XLSX/XLS files directly in a local data directory. The configured reference batch is:

`BEACON_SimMode_IC_Batch11_ProcessNoise_ON_MeasNoise_ON_clean`

The loader expects the eight variables listed in `configs/study.yaml` and derives anomaly labels from the dataset's `*_active` fault columns. The reference batch is required to contain no labeled anomalies. 

## Install

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[foundation,test]"
```

Foundation model checkpoints must be available according to the model identifiers/paths in `configs/study.yaml`.

## Run

```bash
python scripts/run_study.py \
  --config configs/study.yaml \
  --data data \
  --output results/main
```

The full publication run evaluates all 16 configured detectors. 


## TSB-AD

Detector implementations and evaluation components in this repository are derived from TSB-AD:

**Qinghua Liu and John Paparrizos. “The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark.” NeurIPS 2024.**

```bibtex
@inproceedings{liu2024elephant,
  title={The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark},
  author={Liu, Qinghua and Paparrizos, John},
  booktitle={NeurIPS 2024},
  year={2024}
}
```
Files under `src/TSB_AD/` are adapted from the TSB-AD files and retain their upstream source comments. Preserve the upstream TSB-AD attribution and license when redistributing them. Study-specific orchestration is under `src/tsad_transfer/`. `basic_metrics.py` is adapted directly from the TSB-AD evaluation implementation and is marked accordingly in the source. 
