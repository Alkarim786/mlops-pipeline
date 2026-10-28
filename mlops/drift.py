"""Data-drift (PSI, KS) and concept-drift (DDM) detection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import ks_2samp


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, floor: float = 1e-4) -> float:
    """Population Stability Index with quantile bins fitted on the reference.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 major shift.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, bins=edges)[0] / len(reference)
    c = np.histogram(current, bins=edges)[0] / len(current)
    r, c = np.clip(r, floor, None), np.clip(c, floor, None)
    return float(np.sum((c - r) * np.log(c / r)))


@dataclass
class DriftReport:
    drifted: bool
    drifted_features: list[str]
    psi: dict[str, float]
    ks_stat: dict[str, float]
    ks_p: dict[str, float] = field(default_factory=dict)


class DataDriftDetector:
    """Flags a feature when PSI exceeds the threshold *and* the KS test rejects equality."""

    def __init__(self, reference: np.ndarray, names: list[str], psi_threshold: float = 0.2, ks_alpha: float = 1e-3):
        self.reference = np.asarray(reference)
        self.names = names
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha

    def check(self, current: np.ndarray) -> DriftReport:
        current = np.asarray(current)
        p, ks, kp, bad = {}, {}, {}, []
        for j, n in enumerate(self.names):
            p[n] = psi(self.reference[:, j], current[:, j])
            res = ks_2samp(self.reference[:, j], current[:, j])
            ks[n], kp[n] = float(res.statistic), float(res.pvalue)
            if p[n] > self.psi_threshold and kp[n] < self.ks_alpha:
                bad.append(n)
        return DriftReport(bool(bad), bad, p, ks, kp)


class DDM:
    """Drift Detection Method (Gama et al., 2004) over a stream of 0/1 prediction errors.

    Tracks the running error rate p and its std s. The minimum of p+s seen so far is the
    baseline; warning at p+s >= p_min + 2 s_min, drift at p+s >= p_min + 3 s_min.

    Two practical hardening changes versus the textbook version (measured in tests/test_drift.py):
    a longer warm-up (`min_instances`) and a minimum *absolute* error increase (`min_delta`).
    Without them, a lucky error-free start gives s_min ~ 0 and the very next error "drifts".
    """

    STABLE, WARNING, DRIFT = "stable", "warning", "drift"

    def __init__(
        self, min_instances: int = 100, warning_level: float = 2.0, drift_level: float = 3.0, min_delta: float = 0.05
    ):
        self.min_instances = min_instances
        self.min_delta = min_delta
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.p = 1.0
        self.s = 0.0
        self.p_min = float("inf")
        self.s_min = float("inf")

    def update(self, error: int) -> str:
        self.n += 1
        self.p += (error - self.p) / self.n
        self.s = math.sqrt(self.p * (1 - self.p) / self.n)
        if self.n < self.min_instances:
            return self.STABLE
        if self.p + self.s <= self.p_min + self.s_min:
            self.p_min, self.s_min = self.p, self.s
        level = self.p + self.s
        if level >= self.p_min + self.drift_level * self.s_min and self.p - self.p_min >= self.min_delta:
            self.reset()
            return self.DRIFT
        if level >= self.p_min + self.warning_level * self.s_min:
            return self.WARNING
        return self.STABLE
