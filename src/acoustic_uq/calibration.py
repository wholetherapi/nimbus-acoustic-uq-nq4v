"""Species-conditional conformal calibration."""
from __future__ import annotations

import numpy as np


class SpeciesConditionalConformal:
    """Split-conformal calibration with one threshold per species."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.thresholds: dict[str, float] = {}

    def fit(self, species: np.ndarray, probabilities: np.ndarray) -> "SpeciesConditionalConformal":
        for name in np.unique(species):
            mask = species == name
            scores = 1.0 - probabilities[mask].max(axis=1)
            level = min(1.0, np.ceil((len(scores) + 1) * (1 - self.alpha)) / len(scores))
            self.thresholds[str(name)] = float(np.quantile(scores, level))
        return self

    def predict_set(self, name: str, probabilities: np.ndarray) -> np.ndarray:
        threshold = self.thresholds[str(name)]
        return (1.0 - probabilities) <= threshold
