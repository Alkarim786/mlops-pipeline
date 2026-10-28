import numpy as np

from mlops.data import FEATURES, make_batch
from mlops.drift import DDM, DataDriftDetector, psi
from mlops.validation import DataValidationError, Schema, require_valid, validate


def test_psi_small_for_same_distribution_large_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert psi(ref, rng.normal(size=5000)) < 0.05
    assert psi(ref, rng.normal(1.0, 1, size=5000)) > 0.25
    assert psi(ref, rng.normal(0, 2.0, size=5000)) > 0.1  # variance change is caught too


def test_psi_is_zero_for_identical_samples_and_handles_constant_reference():
    x = np.random.default_rng(1).normal(size=1000)
    assert psi(x, x) < 1e-9
    assert psi(np.ones(100), np.ones(100)) == 0.0


def test_detector_flags_only_shifted_features():
    rng = np.random.default_rng(0)
    ref, _ = make_batch(4000, rng)
    det = DataDriftDetector(ref, FEATURES)
    same, _ = make_batch(500, rng)
    assert not det.check(same).drifted
    shifted, _ = make_batch(500, rng, shift=1.5)
    rep = det.check(shifted)
    assert rep.drifted and set(rep.drifted_features) == {"x0", "x1"}


def test_detector_false_alarm_rate_is_low_on_stationary_data():
    rng = np.random.default_rng(3)
    ref, _ = make_batch(4000, rng)
    det = DataDriftDetector(ref, FEATURES)
    alarms = sum(det.check(make_batch(500, rng)[0]).drifted for _ in range(40))
    assert alarms <= 1


def test_ddm_detects_abrupt_error_rate_increase_quickly():
    rng = np.random.default_rng(0)
    ddm = DDM()
    for _ in range(1500):
        assert ddm.update(int(rng.random() < 0.08)) != DDM.DRIFT  # stationary: no alarm
    detected_after = None
    for i in range(1, 1500):
        if ddm.update(int(rng.random() < 0.40)) == DDM.DRIFT:
            detected_after = i
            break
    assert detected_after is not None and detected_after < 200


def _false_alarm_rate(first: int, trials: int = 1000, **kw) -> float:
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(trials):
        d = DDM(**kw)
        hits += any(d.update(int(e)) == DDM.DRIFT for e in rng.random(first) < 0.10)
    return hits / trials


def test_hardened_ddm_has_far_fewer_early_false_alarms_than_textbook():
    # Measured on a stationary 10%-error stream, first 300 samples after a reset.
    textbook = _false_alarm_rate(300, min_instances=30, min_delta=0.0)
    hardened = _false_alarm_rate(300)
    assert textbook > 0.15, textbook
    assert hardened < 0.10, hardened


def test_ddm_detection_delay_for_moderate_degradation():
    rng = np.random.default_rng(1)
    delays = []
    for _ in range(100):
        d = DDM()
        for e in rng.random(500) < 0.08:
            d.update(int(e))
        for i, e in enumerate(rng.random(2000) < 0.20, start=1):
            if d.update(int(e)) == DDM.DRIFT:
                delays.append(i)
                break
    assert len(delays) >= 95 and np.median(delays) < 400


def test_validation_catches_bad_data():
    rng = np.random.default_rng(0)
    X, y = make_batch(1000, rng)
    schema = Schema.default()
    assert validate(X, y, schema) == []
    bad = X.copy()
    bad[0, 0] = np.nan
    bad[1, 1] = 99
    problems = validate(bad, y, schema)
    assert any("NaN" in p for p in problems) and any("outside" in p for p in problems)
    assert any("only" in p for p in validate(X[:10], y[:10], schema))
    assert any("minority" in p for p in validate(X, np.zeros(len(y), dtype=int), schema))
    assert any("expected 6" in p for p in validate(X[:, :3], y, schema))
    try:
        require_valid(bad, y, schema)
        raise AssertionError("should raise")
    except DataValidationError as e:
        assert e.problems
