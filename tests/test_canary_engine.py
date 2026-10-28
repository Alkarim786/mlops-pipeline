import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlops.app import create_app
from mlops.canary import CAND, PROD, ArmStats, CanaryPolicy, bucket, evaluate, route
from mlops.data import make_batch
from mlops.engine import CanaryController, ServingEngine
from mlops.pipeline import train_and_register
from mlops.registry import Registry


class Inverted:  # confidently wrong (module level so joblib can pickle it)
    def __init__(self, inner):
        self.inner = inner

    def predict_proba(self, A):
        return self.inner.predict_proba(A)[:, ::-1]


# ---- routing -------------------------------------------------------------
def test_split_proportion_matches_fraction():
    n, frac = 20000, 0.1
    hits = sum(route(f"req-{i}", frac) == CAND for i in range(n))
    sd = (n * frac * (1 - frac)) ** 0.5
    assert abs(hits - n * frac) < 4 * sd  # within 4 standard deviations of Binomial(n, p)


def test_routing_is_sticky_and_bucket_uniform():
    assert all(route("same-id", 0.5) == route("same-id", 0.5) for _ in range(10))
    b = np.array([bucket(f"x{i}") for i in range(5000)])
    assert 0.0 <= b.min() and b.max() < 1.0 and abs(b.mean() - 0.5) < 0.02


def test_fraction_is_monotone_a_request_in_canary_stays_when_fraction_grows():
    ids = [f"r{i}" for i in range(2000)]
    small = {i for i in ids if route(i, 0.05) == CAND}
    big = {i for i in ids if route(i, 0.25) == CAND}
    assert small <= big


# ---- decision logic --------------------------------------------------------
def arm(n=500, correct=450, errors=0, lat=1.0):
    a = ArmStats()
    for _ in range(n):
        a.record(lat, 1)
    a.errors, a.labelled, a.correct = errors, n, correct
    return a


POLICY = CanaryPolicy(min_labelled=300)


def test_waits_for_enough_labels():
    assert evaluate(arm(), arm(n=100, correct=90), POLICY).action == "continue"


def test_promotes_equal_or_better_candidate():
    assert evaluate(arm(correct=400), arm(correct=450), POLICY).action == "promote"


def test_equal_candidate_waits_for_evidence_then_is_promoted_by_non_inferiority():
    # 0.90 vs 0.90 at n=500 each: lower bound ~ -0.037 < -epsilon -> keep collecting ...
    assert evaluate(arm(correct=450), arm(correct=450), POLICY).action == "continue"
    # ... with 10x the evidence the bound clears -epsilon (0.01) and it is promoted.
    assert evaluate(arm(n=5000, correct=4500), arm(n=5000, correct=4500), POLICY).action == "promote"


def test_rolls_back_significantly_worse_candidate():
    d = evaluate(arm(correct=450), arm(correct=350), POLICY)
    assert d.action == "rollback" and "below production" in d.reason


def test_rolls_back_on_errors_and_latency():
    assert evaluate(arm(), arm(correct=450, errors=50), POLICY).action == "rollback"
    assert evaluate(arm(lat=1.0), arm(correct=450, lat=50.0), POLICY).action == "rollback"


def test_forced_decision_at_max_samples():
    pol = CanaryPolicy(min_labelled=10, max_labelled=100, alpha=1e-12)
    # slightly worse but not significant at this alpha -> inconclusive until max samples reached
    a, b = arm(n=100, correct=90), arm(n=100, correct=80)
    assert evaluate(a, b, pol).action == "rollback"


# ---- engine: hot reload, canary lifecycle ----------------------------------
@pytest.fixture
def setup(tmp_path):
    reg = Registry(tmp_path / "r")
    X, y = make_batch(2000, np.random.default_rng(0))
    v1 = train_and_register(reg, "m", X, y).version
    reg.stage("m", v1)
    reg.promote("m", v1)
    return reg, X, y


def test_hot_reload_picks_up_new_production(setup):
    reg, X, y = setup
    eng = ServingEngine(reg, "m", reload_interval_s=0.0)
    assert eng.predict("a", X[0])["version"] == 1
    v2 = train_and_register(reg, "m", X, y, seed=1).version
    reg.stage("m", v2)
    reg.promote("m", v2)
    assert eng.predict("b", X[1])["version"] == 2  # no restart needed
    assert eng.reload_count == 2
    reg.rollback("m")
    assert eng.predict("c", X[2])["version"] == 1


def test_canary_full_lifecycle_promotes_good_candidate(setup):
    reg, X, y = setup
    eng = ServingEngine(reg, "m")
    ctl = CanaryController(reg, eng, CanaryPolicy(min_labelled=150, max_labelled=400, accuracy_epsilon=0.03))
    v2 = train_and_register(reg, "m", X, y, seed=7).version
    reg.stage("m", v2)
    reg.start_canary("m", v2, 0.3)
    eng.refresh(force=True)
    Xs, ys = make_batch(1500, np.random.default_rng(5))
    arms = set()
    for i in range(len(Xs)):
        out = eng.predict(f"id{i}", Xs[i])
        arms.add(out["arm"])
        eng.feedback(f"id{i}", int(ys[i]))
    assert arms == {PROD, CAND}
    d = ctl.step()
    assert d.action == "promote"
    assert reg.stage_version("m", "PRODUCTION") == v2 and reg.active_canary("m") is None
    assert eng.prod_version == v2 and eng.cand_version is None


def test_canary_rolls_back_broken_candidate(setup):
    reg, X, y = setup
    eng = ServingEngine(reg, "m")
    ctl = CanaryController(reg, eng, CanaryPolicy(min_labelled=150))

    v2 = train_and_register(reg, "m", X, y, seed=3, wrap=Inverted).version
    reg.stage("m", v2)
    reg.start_canary("m", v2, 0.3)
    eng.refresh(force=True)
    Xs, ys = make_batch(1500, np.random.default_rng(6))
    for i in range(len(Xs)):
        eng.predict(f"id{i}", Xs[i])
        eng.feedback(f"id{i}", int(ys[i]))
    assert ctl.step().action == "rollback"
    assert reg.stage_version("m", "PRODUCTION") == 1
    assert reg.get("m", v2)["stage"] == "ARCHIVED"


def test_feedback_for_unknown_request_is_ignored(setup):
    reg, X, _ = setup
    eng = ServingEngine(reg, "m")
    assert eng.feedback("nope", 1) is None
    eng.predict("known", X[0])
    assert eng.feedback("known", 1) in (PROD, CAND)
    assert eng.feedback("known", 1) is None  # a label is consumed once


# ---- HTTP API ---------------------------------------------------------------
def test_http_api(setup, tmp_path):
    reg, X, y = setup
    client = TestClient(create_app(str(reg.root), "m", reload_interval_s=0.0, epsilon=0.05))
    assert client.get("/health").json()["production"] == 1
    r = client.post("/predict", json={"request_id": "r1", "features": X[0].tolist()})
    assert r.status_code == 200 and r.json()["prediction"] in (0, 1)
    assert client.post("/feedback", json={"request_id": "r1", "label": 1}).status_code == 200
    assert client.post("/feedback", json={"request_id": "zzz", "label": 1}).status_code == 404
    assert client.post("/predict", json={"request_id": "r2", "features": [1.0]}).status_code == 422
    assert "mlops_requests_total" in client.get("/metrics").text
    v2 = train_and_register(reg, "m", X, y, seed=9).version
    assert client.post(f"/registry/m/versions/{v2}/stage", json={"stage": "STAGING"}).status_code == 200
    assert client.post(f"/registry/m/versions/{v2}/stage", json={"stage": "PRODUCTION"}).status_code == 200
    assert client.get("/health").json()["production"] == v2
    assert client.post("/registry/m/rollback").json()["production"] == 1
    assert client.post("/registry/m/versions/99/stage", json={"stage": "STAGING"}).status_code == 400
    assert len(client.get("/registry/m/versions").json()) == 2
