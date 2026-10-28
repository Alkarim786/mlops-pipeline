"""FastAPI serving + registry admin API.  `uvicorn mlops.app:app` (env: MLOPS_ROOT, MLOPS_MODEL)."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .canary import CAND, PROD, CanaryPolicy
from .engine import CanaryController, ServingEngine
from .registry import GuardRejected, Registry, RegistryError


class PredictIn(BaseModel):
    request_id: str
    features: list[float] = Field(min_length=1)


class FeedbackIn(BaseModel):
    request_id: str
    label: int = Field(ge=0, le=1)


class StageIn(BaseModel):
    stage: str
    reason: str = ""


class CanaryIn(BaseModel):
    fraction: float = Field(gt=0, lt=1)


def create_app(root: str, model_name: str, reload_interval_s: float = 0.5, epsilon: float = 0.01) -> FastAPI:
    registry = Registry(root, epsilon=epsilon)
    engine = ServingEngine(registry, model_name, reload_interval_s=reload_interval_s)
    controller = CanaryController(registry, engine, CanaryPolicy())
    app = FastAPI(title="mlops-pipeline", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "production": engine.prod_version, "candidate": engine.cand_version}

    @app.post("/predict")
    def predict(body: PredictIn):
        try:
            return engine.predict(body.request_id, body.features)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        except Exception as e:  # noqa: BLE001  (bad feature vector etc. -> 422, not a 500)
            raise HTTPException(422, f"prediction failed: {e}")

    @app.post("/feedback")
    def feedback(body: FeedbackIn):
        arm = engine.feedback(body.request_id, body.label)
        if arm is None:
            raise HTTPException(404, "unknown or expired request_id")
        return {"arm": arm}

    @app.get("/registry/{name}/versions")
    def versions(name: str):
        return registry.list_versions(name)

    @app.post("/registry/{name}/versions/{version}/stage")
    def set_stage(name: str, version: int, body: StageIn):
        try:
            registry.transition(name, version, body.stage, body.reason)
        except GuardRejected as e:
            raise HTTPException(409, str(e))
        except RegistryError as e:
            raise HTTPException(400, str(e))
        engine.refresh(force=True)
        return {"ok": True}

    @app.post("/registry/{name}/rollback")
    def rollback(name: str):
        try:
            v = registry.rollback(name, "manual via API")
        except RegistryError as e:
            raise HTTPException(400, str(e))
        engine.refresh(force=True)
        return {"production": v}

    @app.post("/canary/{version}")
    def start_canary(version: int, body: CanaryIn):
        try:
            registry.start_canary(model_name, version, body.fraction)
        except RegistryError as e:
            raise HTTPException(400, str(e))
        engine.refresh(force=True)
        return {"canary": version, "fraction": body.fraction}

    @app.post("/canary/evaluate")
    def evaluate():
        d = controller.step()
        return {"decision": None if d is None else {"action": d.action, "reason": d.reason}}

    @app.get("/metrics")
    def metrics():
        """Prometheus text exposition."""
        lines = []
        for arm in (PROD, CAND):
            s = engine.stats[arm]
            lines += [
                f'mlops_requests_total{{arm="{arm}"}} {s.n}',
                f'mlops_errors_total{{arm="{arm}"}} {s.errors}',
                f'mlops_labelled_total{{arm="{arm}"}} {s.labelled}',
                f'mlops_correct_total{{arm="{arm}"}} {s.correct}',
            ]
            if s.p95_latency() is not None:
                lines.append(f'mlops_latency_p95_ms{{arm="{arm}"}} {s.p95_latency():.4f}')
        lines.append(f"mlops_model_reloads_total {engine.reload_count}")
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse("\n".join(lines) + "\n")

    return app


def _default_app() -> FastAPI:
    return create_app(os.environ.get("MLOPS_ROOT", "./registry-data"), os.environ.get("MLOPS_MODEL", "demo"))


def __getattr__(name):  # lazy: `uvicorn mlops.app:app` builds the app only when asked for it
    if name == "app":
        return _default_app()
    raise AttributeError(name)
