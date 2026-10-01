"""
TimesFM 3.0 adapter for the evaluation protocol used in this study.
Original model: https://github.com/google-research/timesfm
"""

import numpy as np
import torch
from timesfm3 import TimesFM3Torch


class TimesFM3:
    def __init__(
        self,
        model_name="google/timesfm-3.0-pytorch",
        context_length=512,
        prediction_length=1,
        batch_size=32,
        device=None,
    ):
        self.model_name = model_name
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TimesFM3Torch.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.decision_scores_ = None

    def fit(self, data, y=None):
        """TimesFM3 is evaluated zero-shot; no target-data fitting is performed."""
        return self

    def decision_function(self, X):
        data = np.asarray(X, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D input (time, features), got shape {data.shape}")

        n_samples = len(data)
        scores = np.full(n_samples, np.nan, dtype=float)
        if n_samples <= self.context_length:
            self.decision_scores_ = scores
            return scores

        print(f"[DEBUG] TimesFM3 data shape: {data.shape}")
        print(f"[DEBUG] TimesFM3 context_length: {self.context_length}, prediction_length: {self.prediction_length}")

        windows = np.lib.stride_tricks.sliding_window_view(
            data, window_shape=self.context_length, axis=0
        )[:-1]
        targets = data[self.context_length:]
        print(f"[DEBUG] TimesFM3 scoring windows: {windows.shape}, targets: {targets.shape}")

        window_scores = []
        with torch.no_grad():
            for start in range(0, len(windows), self.batch_size):
                stop = min(start + self.batch_size, len(windows))
                contexts = torch.as_tensor(
                    np.ascontiguousarray(windows[start:stop]),
                    dtype=torch.float32,
                    device=self.device,
                )
                logits = self.model.decode(
                    target=contexts,
                    horizon=self.prediction_length,
                )
                predictions = logits[:, :, 0, 4]
                actual = torch.as_tensor(
                    targets[start:stop], dtype=torch.float32, device=self.device
                )
                errors = (actual - predictions) ** 2
                window_scores.append(errors.mean(dim=1).cpu().numpy())

        window_scores = np.concatenate(window_scores)
        scores[self.context_length:] = window_scores
        print(f"[DEBUG] TimesFM3 scores shape: {window_scores.shape}")
        print(
            f"[DEBUG] TimesFM3 scores min/max/mean: {window_scores.min():.4f} / "
            f"{window_scores.max():.4f} / {window_scores.mean():.4f}"
        )

        self.decision_scores_ = scores
        return self.decision_scores_
