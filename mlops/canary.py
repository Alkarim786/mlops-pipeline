"""Canary routing (sticky traffic split), per-arm statistics and the promote/rollback decision."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

PROD, CAND = "production", "candidate"


def bucket(request_id: str) -> float:
    """Deterministic hash of the request id into [0, 1): the same id always lands in the same arm."""
    return int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) / 2**32


def route(request_id: str, fraction: float) -> str:
    return CAND if bucket(request_id) < fraction else PROD


@dataclass
class ArmStats:
    n: int = 0
    errors: int = 0
    labelled: int = 0
    correct: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    pred_counts: dict[int, int] = field(default_factory=dict)

    def record(self, latency_ms: float, prediction: int | None) -> None:
        self.n += 1
        if prediction is None:
            self.errors += 1
            return
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > 5000:
            del self.latencies_ms[:1000]
        self.pred_counts[prediction] = self.pred_counts.get(prediction, 0) + 1

    def label(self, correct: bool) -> None:
        self.labelled += 1
        self.correct += int(correct)

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.labelled if self.labelled else None

    @property
    def error_rate(self) -> float:
        return self.errors / self.n if self.n else 0.0

    def p95_latency(self) -> float | None:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else None


@dataclass
class CanaryPolicy:
    min_labelled: int = 300  # labelled samples needed in *each* arm before judging
    max_labelled: int = 3000  # force a decision by here
    accuracy_epsilon: float = 0.01  # candidate may be at most this much worse and still be promoted
    alpha: float = 0.05  # significance for "candidate is worse"
    max_error_rate: float = 0.02
    max_latency_ratio: float = 5.0  # candidate p95 / production p95


@dataclass
class Decision:
    action: str  # "continue" | "promote" | "rollback"
    reason: str


def _one_sided_p_worse(c_correct: int, c_n: int, p_correct: int, p_n: int) -> float:
    """P-value of H0 (cand >= prod) vs H1 (cand < prod), pooled two-proportion z-test."""
    p1, p2 = c_correct / c_n, p_correct / p_n
    pooled = (c_correct + p_correct) / (c_n + p_n)
    se = math.sqrt(pooled * (1 - pooled) * (1 / c_n + 1 / p_n))
    if se == 0:
        return 1.0 if p1 >= p2 else 0.0
    z = (p1 - p2) / se
    return 0.5 * math.erfc(-z / math.sqrt(2))  # Phi(z)


def evaluate(prod: ArmStats, cand: ArmStats, policy: CanaryPolicy) -> Decision:
    if cand.n >= 20 and cand.error_rate > policy.max_error_rate:
        return Decision("rollback", f"candidate error rate {cand.error_rate:.3f} > {policy.max_error_rate}")
    pl, cl = prod.p95_latency(), cand.p95_latency()
    if pl and cl and cand.n >= 50 and cl > policy.max_latency_ratio * pl:
        return Decision("rollback", f"candidate p95 latency {cl:.2f}ms > {policy.max_latency_ratio}x production {pl:.2f}ms")
    if cand.labelled < policy.min_labelled or prod.labelled < policy.min_labelled:
        return Decision("continue", f"waiting for labels (cand {cand.labelled}, prod {prod.labelled}, need {policy.min_labelled})")
    ca, pa = cand.accuracy, prod.accuracy
    p_worse = _one_sided_p_worse(cand.correct, cand.labelled, prod.correct, prod.labelled)
    if ca < pa - policy.accuracy_epsilon and p_worse < policy.alpha:
        return Decision("rollback", f"candidate accuracy {ca:.3f} significantly below production {pa:.3f} (p={p_worse:.4f})")
    # Non-inferiority: promote once the lower confidence bound of (cand - prod) clears -epsilon.
    se = math.sqrt(ca * (1 - ca) / cand.labelled + pa * (1 - pa) / prod.labelled)
    lower = (ca - pa) - NormalDist().inv_cdf(1 - policy.alpha) * se
    if lower >= -policy.accuracy_epsilon:
        return Decision("promote", f"candidate accuracy {ca:.3f} vs production {pa:.3f}: non-inferior (lower bound {lower:+.3f})")
    if min(cand.labelled, prod.labelled) >= policy.max_labelled:
        if ca >= pa - policy.accuracy_epsilon:
            return Decision("promote", f"max samples reached; candidate {ca:.3f} vs production {pa:.3f}")
        return Decision("rollback", f"max samples reached; candidate {ca:.3f} < production {pa:.3f}")
    return Decision("continue", f"inconclusive (cand {ca:.3f}, prod {pa:.3f}, lower bound {lower:+.3f})")
