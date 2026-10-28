"""Closed-loop simulation: serve a drifting stream, detect drift, retrain, canary, promote/rollback."""

from __future__ import annotations

import os

# Single-row inference is latency-bound; thread fan-out per call only adds overhead.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .canary import PROD, CanaryPolicy
from .data import FEATURES, DriftSchedule, Stream, make_batch
from .drift import DDM, DataDriftDetector
from .engine import CanaryController, ServingEngine
from .pipeline import train_and_register
from .registry import Registry

NAME = "demo-classifier"


class ZeroedFeaturesModel:
    """Fault injection: a serving-side feature-pipeline bug (the four informative features arrive as 0).

    Its *offline* holdout metrics look fine because they are computed before wrapping, which is
    exactly the failure a canary exists to catch.
    """

    def __init__(self, inner, n_zeroed: int = 4):
        self.inner, self.n_zeroed = inner, n_zeroed

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        X[:, : self.n_zeroed] = 0.0
        return self.inner.predict_proba(X)


@dataclass
class Scenario:
    name: str
    description: str
    drift_at: int = 40
    shift: float = 0.0
    concept: float = 0.0
    fault_first_candidate: bool = False
    steps: int = 140
    batch: int = 100
    label_delay: int = 2
    canary_fraction: float = 0.2
    min_retrain_rows: int = 800


SCENARIOS = {
    "concept_drift": Scenario(
        "concept_drift", "P(y|X) changes at step 40 (the label rule rotates); P(X) is unchanged", concept=1.3
    ),
    "covariate_shift": Scenario(
        "covariate_shift", "P(X) shifts at step 40 (x0, x1 mean +2.5); the label rule is unchanged", shift=2.5
    ),
    "skewed_candidate": Scenario(
        "skewed_candidate",
        "moderate concept drift, but the first retrained candidate has a serving-side feature bug; the canary must roll it back",
        concept=0.6,
        fault_first_candidate=True,
    ),
}


@dataclass
class Result:
    scenario: str
    description: str
    timeline: list[dict] = field(default_factory=list)
    acc_by_step: list[float] = field(default_factory=list)
    prod_version_by_step: list[int] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def run(scenario: Scenario, workdir, seed: int = 0, verbose: bool = False) -> Result:
    registry = Registry(workdir)
    res = Result(scenario.name, scenario.description)

    def log(step: int, event: str, **kw):
        res.timeline.append({"step": step, "event": event, **kw})
        if verbose:
            print(f"  step {step:3d}  {event}  {kw}")

    # ---- bootstrap: train + promote v1 on the pre-drift regime ----------
    rng = np.random.default_rng(seed)
    X0, y0 = make_batch(4000, rng)
    t = train_and_register(registry, NAME, X0, y0, seed=seed)
    registry.stage(NAME, t.version, "bootstrap")
    registry.promote(NAME, t.version, "bootstrap")
    engine = ServingEngine(registry, NAME, reload_interval_s=60.0)  # transitions use refresh(force=True)
    policy = CanaryPolicy(min_labelled=200, max_labelled=500, accuracy_epsilon=0.02)
    controller = CanaryController(registry, engine, policy)
    detector = DataDriftDetector(registry.load_reference(NAME, t.version), FEATURES)
    ddm = DDM()
    log(0, "bootstrap", production=t.version, holdout_accuracy=round(t.metrics["accuracy"], 4))

    stream = Stream(DriftSchedule(scenario.drift_at, scenario.shift, scenario.concept), scenario.batch, seed + 1)
    labelled: deque = deque(maxlen=8000)  # (data_step, x, y)
    due: dict[int, list] = {}
    served: dict[str, int] = {}
    recent: deque = deque(maxlen=5)  # (step, X) of the last 5 batches
    state, attempts = "monitoring", 0
    start_from = 0
    data_alerts = 0
    warning_start: int | None = None
    first_alert: int | None = None
    promoted_step: int | None = None
    retrain_seconds: list[float] = []
    rollbacks = 0

    for step in range(scenario.steps):
        X, y = stream.batch(step)
        ids = [f"{step}-{i}" for i in range(len(X))]
        correct = 0
        for i, rid in enumerate(ids):
            out = engine.predict(rid, X[i])
            served[rid] = out["prediction"]
            correct += int(out["prediction"] == y[i])
        res.acc_by_step.append(correct / len(X))
        res.prod_version_by_step.append(engine.prod_version)
        due.setdefault(step + scenario.label_delay, []).append((step, ids, X, y))
        recent.append((step, X))

        # ---- delayed labels ------------------------------------------------
        for data_step, lids, lX, ly in due.pop(step, []):
            for j, rid in enumerate(lids):
                arm = engine.feedback(rid, int(ly[j]))
                labelled.append((data_step, lX[j], int(ly[j])))
                if arm == PROD:
                    verdict = ddm.update(int(served[rid] != ly[j]))
                    if verdict == DDM.WARNING and warning_start is None:
                        warning_start = data_step
                    elif verdict == DDM.STABLE:
                        warning_start = None
                    elif verdict == DDM.DRIFT and state == "monitoring":
                        first_alert = first_alert if first_alert is not None else step
                        log(step, "concept_drift_alert", detector="DDM", warning_since_step=warning_start)
                        state, start_from = "collecting", warning_start if warning_start is not None else step - 5
                        warning_start = None
                served.pop(rid, None)

        # ---- data drift check ----------------------------------------------
        if state == "monitoring" and len(recent) == recent.maxlen:
            rep = detector.check(np.vstack([b for _, b in recent]))
            data_alerts = data_alerts + 1 if rep.drifted else 0
            if data_alerts >= 2:
                first_alert = first_alert if first_alert is not None else step
                worst = max(rep.psi, key=rep.psi.get)
                log(step, "data_drift_alert", detector="PSI+KS", features=rep.drifted_features,
                    max_psi=round(rep.psi[worst], 3))
                # Estimate the change point: earliest recent batch that is itself flagged as drifted,
                # so the retraining set does not mix pre- and post-shift rows.
                flagged = [st for st, b in recent if detector.check(b).drifted]
                state, start_from, data_alerts = "collecting", (min(flagged) if flagged else step - 1), 0

        # ---- retrain when enough post-drift labelled data exists -------------
        if state == "collecting":
            rows = [(x, lab) for ds, x, lab in labelled if ds >= start_from]
            if len(rows) >= scenario.min_retrain_rows:
                Xr = np.array([r[0] for r in rows])
                yr = np.array([r[1] for r in rows])
                wrap = ZeroedFeaturesModel if (scenario.fault_first_candidate and attempts == 0) else None
                tr = train_and_register(registry, NAME, Xr, yr, seed=seed + attempts, wrap=wrap)
                attempts += 1
                retrain_seconds.append(tr.train_seconds)
                registry.stage(NAME, tr.version, "auto-retrain")
                registry.start_canary(NAME, tr.version, scenario.canary_fraction)
                engine.refresh(force=True)
                log(step, "retrained_and_canary_started", version=tr.version, rows=len(rows),
                    holdout_accuracy=round(tr.metrics["accuracy"], 4), train_seconds=round(tr.train_seconds, 3),
                    fraction=scenario.canary_fraction, faulty=wrap is not None)
                state = "canary"

        # ---- canary decision -----------------------------------------------
        elif state == "canary":
            d = controller.step()
            if d is not None and d.action != "continue":
                cand_stats = d.reason
                log(step, f"canary_{d.action}", reason=cand_stats)
                ddm.reset()
                if d.action == "promote":
                    promoted_step = step
                    detector = DataDriftDetector(registry.load_reference(NAME, engine.prod_version), FEATURES)
                    state, data_alerts = "monitoring", 0
                else:
                    rollbacks += 1
                    if attempts < 3:  # drift is unresolved: collect fresh data and try again
                        state, start_from = "collecting", step - 1
                    else:
                        state = "monitoring"

    # ---- summary --------------------------------------------------------------
    acc = np.array(res.acc_by_step)
    d0 = scenario.drift_at
    end_of_degraded = promoted_step if promoted_step is not None else scenario.steps
    res.summary = {
        "accuracy_before_drift": round(float(acc[5:d0].mean()), 4),
        "accuracy_during_degradation": round(float(acc[d0:end_of_degraded].mean()), 4),
        "accuracy_after_recovery_last20": round(float(acc[-20:].mean()), 4),
        "drift_injected_at_step": d0,
        "first_alert_step": first_alert,
        "detection_delay_steps": None if first_alert is None else first_alert - d0,
        "detection_delay_samples": None if first_alert is None else (first_alert - d0) * scenario.batch,
        "retrain_seconds": [round(s, 3) for s in retrain_seconds],
        "alerts": sum(e["event"].endswith("_alert") for e in res.timeline),
        "retrains": len(retrain_seconds),
        "promotions": sum(e["event"] == "canary_promote" for e in res.timeline),
        "rollbacks": rollbacks,
        "promoted_at_step": promoted_step,
        "final_production_version": engine.prod_version,
        "batch_size": scenario.batch,
        "label_delay_steps": scenario.label_delay,
    }
    return res


def main() -> None:
    import argparse
    import json
    import shutil
    import tempfile
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Run the closed-loop MLOps demo")
    ap.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    ap.add_argument("--out", default="docs/results.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    out = {}
    for n in names:
        work = Path(tempfile.mkdtemp(prefix=f"mlops-{n}-"))
        t0 = time.perf_counter()
        print(f"\n=== {n}: {SCENARIOS[n].description}")
        r = run(SCENARIOS[n], work, seed=args.seed, verbose=True)
        print(f"  summary: {json.dumps(r.summary)}  ({time.perf_counter() - t0:.1f}s wall)")
        out[n] = {"description": r.description, "summary": r.summary, "timeline": r.timeline,
                  "accuracy_by_step": [round(a, 3) for a in r.acc_by_step]}
        shutil.rmtree(work, ignore_errors=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
