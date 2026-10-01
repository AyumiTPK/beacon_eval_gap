from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .types import Batch

DEFAULT_FEATURES = (
    "T_meas_degC", "pH_meas", "pO2_meas_pc", "F_air_NLh",
    "F_o2_NLh", "N_stirrer_rpm", "heating_pc", "cooling_pc",
)


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file: {path}")


def _fault_type(path: Path, active_col: str | None) -> str:
    if active_col:
        return re.sub(r"_active$", "", re.sub(r"^fault\d+_", "", active_col))
    return "clean"


def load_fault_folder(folder: str | Path, feature_columns: tuple[str, ...] = DEFAULT_FEATURES) -> list[Batch]:
    folder = Path(folder)
    paths = sorted([
        *folder.glob("*.csv"),
        *folder.glob("*.xlsx"),
        *folder.glob("*.xls"),
    ])
    if not paths:
        raise ValueError(f"No CSV/Excel files in {folder}")

    batches = []
    for path in paths:
        dataframe = _read(path)
        active_columns = [
            column
            for column in dataframe.columns
            if column.lower().endswith("_active")
        ]
        if len(active_columns) > 1:
            raise ValueError(f"Multiple *_active columns in {path.name}: {active_columns}")

        active_column = active_columns[0] if active_columns else None
        missing = [
            column
            for column in feature_columns
            if column not in dataframe.columns
        ]
        if missing:
            raise ValueError(f"Missing features in {path.name}: {missing}")

        if active_column:
            labels = (
                dataframe[active_column]
                .fillna(0)
                .astype(int)
                .to_numpy()
            )
        else:
            labels = np.zeros(len(dataframe), dtype=int)
            
        time = dataframe["time_h"].to_numpy() if "time_h" in dataframe.columns else None
        features = dataframe[list(feature_columns)].to_numpy(float)
        metadata = {"source": str(path), "fault_type": _fault_type(path, active_column)}
        
        batch = Batch(
            batch_id=path.stem,
            X=features,
            y=labels,
            time=time,
            feature_names=tuple(feature_columns),
            metadata=metadata,
        )
        batches.append(batch)

    return batches


def select_reference(batches: list[Batch], reference_id: str | None) -> tuple[Batch, list[Batch]]:
    if reference_id:
        matches = [batch for batch in batches if batch.batch_id == reference_id]
        if len(matches) != 1:
            raise ValueError(f"Reference batch not found uniquely: {reference_id}")
        reference = matches[0]
    else:
        normal_batches = [batch for batch in batches if batch.is_labeled_normal]
        if not normal_batches:
            raise ValueError("No labeled-normal reference batch")
        reference = normal_batches[0]

    if not reference.is_labeled_normal:
        raise ValueError("Reference batch must be labeled normal")
    return reference, [batch for batch in batches if batch.batch_id != reference.batch_id]