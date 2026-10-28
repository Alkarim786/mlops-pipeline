"""End-to-end: a short closed loop (drift -> detect -> retrain -> canary -> promote/rollback)."""

from dataclasses import replace

from mlops.simulate import SCENARIOS, run

SMALL = {"steps": 100, "batch": 60, "min_retrain_rows": 500, "drift_at": 30}


def test_concept_drift_loop_recovers(tmp_path):
    r = run(replace(SCENARIOS["concept_drift"], **SMALL), tmp_path)
    s = r.summary
    assert s["first_alert_step"] is not None and s["detection_delay_steps"] <= 10
    assert s["promoted_at_step"] is not None and s["final_production_version"] >= 2
    assert s["accuracy_during_degradation"] < s["accuracy_before_drift"] - 0.15
    assert s["accuracy_after_recovery_last20"] > s["accuracy_during_degradation"] + 0.15


def test_skewed_candidate_is_rolled_back_then_clean_retrain_wins(tmp_path):
    r = run(replace(SCENARIOS["skewed_candidate"], **{**SMALL, "steps": 160}), tmp_path)
    events = [e["event"] for e in r.timeline]
    assert "canary_rollback" in events and "canary_promote" in events
    assert events.index("canary_rollback") < events.index("canary_promote")
    assert r.summary["rollbacks"] >= 1 and r.summary["accuracy_after_recovery_last20"] > 0.8
