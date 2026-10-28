"""Serving engine: loads PRODUCTION (and the canary candidate), hot-reloads on registry changes,
routes traffic, records per-arm metrics and joins delayed labels back to predictions.

Framework-free so the FastAPI app and the offline simulator share exactly the same code.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

import numpy as np

from . import canary as cn
from .canary import CAND, PROD, ArmStats, CanaryPolicy, Decision
from .registry import GuardRejected, Registry


class ServingEngine:
    def __init__(self, registry: Registry, name: str, reload_interval_s: float = 0.0, pending_cap: int = 200_000):
        self.registry = registry
        self.name = name
        self.reload_interval_s = reload_interval_s
        self._lock = threading.RLock()
        self._last_check = 0.0
        self.prod_version: int | None = None
        self.cand_version: int | None = None
        self.fraction = 0.0
        self._models: dict[str, object] = {}
        self.stats: dict[str, ArmStats] = {PROD: ArmStats(), CAND: ArmStats()}
        self._pending: OrderedDict[str, tuple[str, int]] = OrderedDict()
        self._pending_cap = pending_cap
        self.reload_count = 0
        self.refresh(force=True)

    # ---- hot reload -----------------------------------------------------
    def refresh(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last_check < self.reload_interval_s:
            return False
        self._last_check = now
        prod = self.registry.stage_version(self.name, "PRODUCTION")
        canary = self.registry.active_canary(self.name)
        cand_v, frac = canary if canary else (None, 0.0)
        if prod == self.prod_version and cand_v == self.cand_version and frac == self.fraction:
            return False
        with self._lock:
            if prod != self.prod_version:
                self._models[PROD] = self.registry.load_model(self.name, prod) if prod is not None else None
                self.prod_version = prod
                self.reload_count += 1
            if cand_v != self.cand_version:
                self._models[CAND] = self.registry.load_model(self.name, cand_v) if cand_v is not None else None
                self.cand_version = cand_v
            self.fraction = frac
            # Fresh counters so both arms are compared over the same window.
            self.stats = {PROD: ArmStats(), CAND: ArmStats()}
            self._pending.clear()
        return True

    # ---- inference --------------------------------------------------------
    def predict(self, request_id: str, features) -> dict:
        self.refresh()
        with self._lock:
            arm = cn.route(request_id, self.fraction) if self._models.get(CAND) is not None else PROD
            model = self._models.get(arm)
            version = self.cand_version if arm == CAND else self.prod_version
        if model is None:
            raise RuntimeError("no PRODUCTION model registered")
        x = np.asarray(features, dtype=float).reshape(1, -1)
        t0 = time.perf_counter()
        try:
            proba = model.predict_proba(x)[0]
            pred = int(np.argmax(proba))
            ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self.stats[arm].record(ms, pred)
                self._pending[request_id] = (arm, pred)
                if len(self._pending) > self._pending_cap:
                    self._pending.popitem(last=False)
            return {"prediction": pred, "probability": float(proba[pred]), "version": version, "arm": arm, "latency_ms": ms}
        except Exception:
            with self._lock:
                self.stats[arm].record(0.0, None)
            raise

    def feedback(self, request_id: str, label: int) -> str | None:
        """Attach a (possibly delayed) ground-truth label; returns the arm that served it."""
        with self._lock:
            hit = self._pending.pop(request_id, None)
            if hit is None:
                return None
            arm, pred = hit
            self.stats[arm].label(pred == label)
            return arm

    def evaluate_canary(self, policy: CanaryPolicy) -> Decision | None:
        with self._lock:
            if self.cand_version is None:
                return None
            return cn.evaluate(self.stats[PROD], self.stats[CAND], policy)


class CanaryController:
    """Applies the canary decision to the registry (promote / archive) and ends the rollout."""

    def __init__(self, registry: Registry, engine: ServingEngine, policy: CanaryPolicy | None = None):
        self.registry = registry
        self.engine = engine
        self.policy = policy or CanaryPolicy()

    def step(self) -> Decision | None:
        d = self.engine.evaluate_canary(self.policy)
        if d is None or d.action == "continue":
            return d
        name, v = self.engine.name, self.engine.cand_version
        if d.action == "promote":
            try:
                self.registry.promote(name, v, d.reason, check_guard=False)
            except GuardRejected as e:  # only if a caller re-enables the guard
                self.registry.transition(name, v, "ARCHIVED", f"promotion guard: {e}")
                d = Decision("rollback", f"registry guard rejected promotion: {e}")
        else:
            self.registry.transition(name, v, "ARCHIVED", d.reason)
        self.registry.end_canary(name)
        self.engine.refresh(force=True)
        return d
