"""Training pipeline: validate -> split -> train -> evaluate on holdout -> register."""

from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from .registry import Registry
from .validation import Schema, require_valid


def data_hash(X: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())
    return h.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True, timeout=5, cwd=Path(__file__).parent
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass
class TrainResult:
    version: int
    metrics: dict[str, float]
    train_seconds: float
    model: object
    X_train: np.ndarray


def train_and_register(
    registry: Registry,
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    schema: Schema | None = None,
    wrap=None,
) -> TrainResult:
    """Validate, train, evaluate on a stratified holdout, and register as a new NONE-stage version.

    `wrap` (optional) post-processes the fitted model before registration; the demo uses it to
    inject a deliberately faulty candidate.
    """
    schema = schema or Schema.default()
    require_valid(X, y, schema)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    t0 = time.perf_counter()
    params = {"max_iter": 100, "learning_rate": 0.1, "random_state": seed}
    model = HistGradientBoostingClassifier(**params).fit(Xtr, ytr)
    seconds = time.perf_counter() - t0
    proba = model.predict_proba(Xte)
    metrics = {
        "accuracy": float(accuracy_score(yte, proba.argmax(1))),
        "log_loss": float(log_loss(yte, proba)),
        "n_train": len(Xtr),
        "train_seconds": round(seconds, 4),
    }
    if wrap is not None:
        model = wrap(model)
    rng = np.random.default_rng(seed)
    ref = Xtr[rng.choice(len(Xtr), size=min(2000, len(Xtr)), replace=False)]
    version = registry.register(
        name, model, metrics, params={**params, "seed": seed}, data_hash=data_hash(X, y), git_sha=git_sha(),
        reference=ref,
    )
    return TrainResult(version, metrics, seconds, model, Xtr)
