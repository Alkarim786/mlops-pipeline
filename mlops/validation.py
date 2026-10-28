"""Hand-written data validation (schema + range + label checks), run before every training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import FEATURES


class DataValidationError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    min: float = -10.0
    max: float = 10.0


@dataclass
class Schema:
    features: list[FeatureSpec]
    min_rows: int = 500
    min_class_fraction: float = 0.05

    @classmethod
    def default(cls) -> Schema:
        return cls([FeatureSpec(n) for n in FEATURES])


def validate(X: np.ndarray, y: np.ndarray, schema: Schema) -> list[str]:
    problems: list[str] = []
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] != len(schema.features):
        return [f"expected {len(schema.features)} feature columns, got shape {X.shape}"]
    if len(X) != len(y):
        problems.append(f"X has {len(X)} rows but y has {len(y)}")
    if len(X) < schema.min_rows:
        problems.append(f"only {len(X)} rows, need at least {schema.min_rows}")
    for j, spec in enumerate(schema.features):
        col = X[:, j]
        if not np.isfinite(col).all():
            problems.append(f"{spec.name}: contains NaN/inf ({int((~np.isfinite(col)).sum())} values)")
            continue
        if col.min() < spec.min or col.max() > spec.max:
            problems.append(f"{spec.name}: values outside [{spec.min}, {spec.max}] (min={col.min():.2f}, max={col.max():.2f})")
    if not set(np.unique(y)).issubset({0, 1}):
        problems.append("labels must be 0/1")
    elif len(y) and min(y.mean(), 1 - y.mean()) < schema.min_class_fraction:
        problems.append(f"minority class fraction {min(y.mean(), 1 - y.mean()):.3f} < {schema.min_class_fraction}")
    return problems


def require_valid(X: np.ndarray, y: np.ndarray, schema: Schema) -> None:
    problems = validate(X, y, schema)
    if problems:
        raise DataValidationError(problems)
