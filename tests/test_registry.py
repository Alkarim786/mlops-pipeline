import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlops.registry import ChecksumMismatch, GuardRejected, Registry, RegistryError


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "reg", epsilon=0.01)


def tiny_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    return LogisticRegression().fit(X, [0, 0, 1, 1])


def add(reg, acc, name="m"):
    return reg.register(name, tiny_model(), {"accuracy": acc}, data_hash="h", git_sha="abc")


def test_versions_increment_and_start_in_none(reg):
    assert add(reg, 0.9) == 1 and add(reg, 0.9) == 2
    assert reg.get("m", 1)["stage"] == "NONE"
    assert reg.get("m", 1)["git_sha"] == "abc" and reg.get("m", 1)["data_hash"] == "h"


def test_first_promotion_has_no_guard_and_next_archives_previous(reg):
    v1 = add(reg, 0.80)
    reg.stage("m", v1)
    reg.promote("m", v1)
    v2 = add(reg, 0.85)
    reg.stage("m", v2)
    reg.promote("m", v2)
    assert reg.stage_version("m", "PRODUCTION") == v2
    assert reg.get("m", v1)["stage"] == "ARCHIVED"


def test_guard_rejects_worse_candidate_but_allows_within_epsilon(reg):
    v1 = add(reg, 0.90)
    reg.stage("m", v1)
    reg.promote("m", v1)
    bad, ok = add(reg, 0.85), add(reg, 0.895)
    reg.stage("m", bad)
    with pytest.raises(GuardRejected):
        reg.promote("m", bad)
    assert reg.stage_version("m", "PRODUCTION") == v1  # rejected promotion changed nothing
    reg.stage("m", ok)
    reg.promote("m", ok)  # 0.895 >= 0.90 - 0.01
    assert reg.stage_version("m", "PRODUCTION") == ok


def test_guard_can_be_bypassed_explicitly(reg):
    v1 = add(reg, 0.95)
    reg.stage("m", v1)
    reg.promote("m", v1)
    v2 = add(reg, 0.60)
    reg.stage("m", v2)
    reg.promote("m", v2, check_guard=False)
    assert reg.stage_version("m", "PRODUCTION") == v2


def test_illegal_transitions(reg):
    v = add(reg, 0.9)
    with pytest.raises(RegistryError):
        reg.transition("m", v, "PRODUCTION")  # must pass through STAGING
    with pytest.raises(RegistryError):
        reg.transition("m", v, "BOGUS")
    with pytest.raises(RegistryError):
        reg.transition("m", 99, "STAGING")


def test_rollback_restores_previous_production(reg):
    v1, v2 = add(reg, 0.90), add(reg, 0.92)
    for v in (v1, v2):
        reg.stage("m", v)
        reg.promote("m", v)
    assert reg.rollback("m", "test") == v1
    assert reg.stage_version("m", "PRODUCTION") == v1
    assert reg.get("m", v2)["stage"] == "ARCHIVED"
    ev = [(e["version"], e["to_stage"]) for e in reg.events("m")]
    assert ev[-1] == (v1, "PRODUCTION")


def test_rollback_without_history_fails(reg):
    v = add(reg, 0.9)
    reg.stage("m", v)
    reg.promote("m", v)
    with pytest.raises(RegistryError):
        reg.rollback("m")


def test_checksum_detects_tampered_artifact(reg):
    v = add(reg, 0.9)
    assert reg.load_model("m", v) is not None
    art = reg.root / reg.get("m", v)["artifact"]
    art.write_bytes(art.read_bytes() + b"tamper")
    with pytest.raises(ChecksumMismatch):
        reg.load_model("m", v)


def test_canary_requires_staging_and_valid_fraction(reg):
    v = add(reg, 0.9)
    with pytest.raises(RegistryError):
        reg.start_canary("m", v, 0.1)
    reg.stage("m", v)
    with pytest.raises(RegistryError):
        reg.start_canary("m", v, 1.5)
    reg.start_canary("m", v, 0.1)
    assert reg.active_canary("m") == (v, 0.1)
    reg.end_canary("m")
    assert reg.active_canary("m") is None


def test_reference_sample_roundtrip(reg):
    ref = np.random.default_rng(0).normal(size=(50, 3))
    v = reg.register("m", tiny_model(), {"accuracy": 0.9}, reference=ref)
    np.testing.assert_array_equal(reg.load_reference("m", v), ref)
