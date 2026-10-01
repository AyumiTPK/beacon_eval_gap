from __future__ import annotations
from pathlib import Path
import yaml
from .models import ModelSpec

def load_config(path: str | Path):
    config = yaml.safe_load(Path(path).read_text())
    specs = [
        ModelSpec(
            model["name"],
            model.get("params", {}),
            model.get("mode", "fit"),
            model.get("family", "unspecified"),
        )
        for model in config["models"]
        if model.get("enabled", True)
    ]
    return config, specs