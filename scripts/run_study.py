from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from tsad_transfer.analysis import analyze
from tsad_transfer.config import load_config
from tsad_transfer.data import load_fault_folder, select_reference
from tsad_transfer.protocols import resolve_vus_window, run_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/main")
    args = parser.parse_args()

    config, specs = load_config(args.config)
    features = tuple(config["dataset"]["features"])
    batches = load_fault_folder(args.data, features)
    reference_id = config["dataset"].get("reference_batch_id")
    reference, evaluation = select_reference(batches, reference_id)

    evaluation_config = config["evaluation"]
    evaluation_config["vus_window"] = resolve_vus_window(evaluation, evaluation_config.get("vus_window"))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2))

    manifest = pd.DataFrame([
        {
            "batch_order": i,
            "batch_id": batch.batch_id,
            "n_samples": len(batch.X),
            "n_anomaly_samples": int(batch.y.sum()),
            "fault_type": batch.metadata.get("fault_type"),
        }
        for i, batch in enumerate(evaluation)
    ])
    manifest.to_csv(output / "evaluation_manifest.csv", index=False)

    for spec in specs:
        print(f"=== {spec.name} ===")
        run_model(
            spec,
            reference,
            evaluation,
            output,
            evaluation_config["threshold_quantile"],
            evaluation_config["vus_window"],
        )

    analyze(output)


if __name__ == "__main__":
    main()