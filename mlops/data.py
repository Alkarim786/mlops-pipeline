"""Synthetic data stream with injectable covariate shift and concept drift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_FEATURES = 6
FEATURES = [f"x{i}" for i in range(N_FEATURES)]


def make_batch(
    n: int, rng: np.random.Generator, shift: float = 0.0, concept: float = 0.0, noise: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Binary classification batch.

    `shift` moves the mean of x0, x1 (covariate / data drift: P(X) changes).
    `concept` is an angle: at 0 the label depends on x0 + x1, at pi/2 on x2 + x3 only
    (concept drift: P(y|X) changes even when P(X) does not).
    """
    X = rng.normal(0.0, 1.0, size=(n, N_FEATURES))
    X[:, 0:2] += shift
    z = np.cos(concept) * (X[:, 0] + X[:, 1] - 2 * shift) + np.sin(concept) * (X[:, 2] + X[:, 3])
    y = (z + noise * rng.normal(size=n) > 0).astype(int)
    return X, y


@dataclass
class DriftSchedule:
    """Regime as a function of the time step."""

    drift_at: int | None = None
    shift: float = 0.0
    concept: float = 0.0

    def regime(self, step: int) -> tuple[float, float]:
        if self.drift_at is not None and step >= self.drift_at:
            return self.shift, self.concept
        return 0.0, 0.0


class Stream:
    def __init__(self, schedule: DriftSchedule, batch_size: int = 100, seed: int = 0):
        self.schedule = schedule
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def batch(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        shift, concept = self.schedule.regime(step)
        return make_batch(self.batch_size, self.rng, shift, concept)
