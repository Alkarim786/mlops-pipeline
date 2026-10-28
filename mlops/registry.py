"""Model registry: versions, stages, checksummed artifacts, guarded promotion, rollback.

Metadata lives in SQLite, artifacts on the local filesystem (an S3/MinIO backend would
only need to replace `_write_artifact` / `_read_artifact`).

Security note: models are stored with joblib (pickle). Only load registries you trust;
the SHA-256 check detects corruption/tampering of an artifact after registration, it does
not make untrusted pickles safe.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np

STAGES = ("NONE", "STAGING", "PRODUCTION", "ARCHIVED")
_ALLOWED = {
    ("NONE", "STAGING"),
    ("NONE", "ARCHIVED"),
    ("STAGING", "PRODUCTION"),
    ("STAGING", "ARCHIVED"),
    ("PRODUCTION", "ARCHIVED"),
    ("ARCHIVED", "STAGING"),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    name TEXT NOT NULL, version INTEGER NOT NULL, stage TEXT NOT NULL,
    metrics TEXT NOT NULL, params TEXT NOT NULL, data_hash TEXT NOT NULL, git_sha TEXT NOT NULL,
    artifact TEXT NOT NULL, sha256 TEXT NOT NULL, reference TEXT, created_at REAL NOT NULL,
    PRIMARY KEY (name, version));
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, version INTEGER NOT NULL,
    from_stage TEXT NOT NULL, to_stage TEXT NOT NULL, reason TEXT NOT NULL, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS rollouts (
    name TEXT PRIMARY KEY, version INTEGER NOT NULL, fraction REAL NOT NULL, started_at REAL NOT NULL);
"""


class RegistryError(Exception):
    pass


class GuardRejected(RegistryError):
    """Promotion refused because the candidate is worse than production beyond epsilon."""


class ChecksumMismatch(RegistryError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Registry:
    def __init__(self, root: str | Path, primary_metric: str = "accuracy", epsilon: float = 0.01):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "registry.db"
        self.primary_metric = primary_metric
        self.epsilon = epsilon
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commit or roll back as one transaction
                yield conn
        finally:
            conn.close()

    # ---- registration -------------------------------------------------
    def register(
        self,
        name: str,
        model: Any,
        metrics: dict[str, float],
        params: dict[str, Any] | None = None,
        data_hash: str = "",
        git_sha: str = "unknown",
        reference: np.ndarray | None = None,
    ) -> int:
        with self._conn() as c:
            row = c.execute("SELECT COALESCE(MAX(version), 0) AS v FROM versions WHERE name=?", (name,)).fetchone()
            version = row["v"] + 1
            vdir = self.root / "artifacts" / name / f"v{version}"
            vdir.mkdir(parents=True, exist_ok=True)
            art = vdir / "model.joblib"
            joblib.dump(model, art)
            ref_path = None
            if reference is not None:
                ref_path = vdir / "reference.npy"
                np.save(ref_path, reference)
            c.execute(
                "INSERT INTO versions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    name, version, "NONE", json.dumps(metrics), json.dumps(params or {}), data_hash, git_sha,
                    str(art.relative_to(self.root)), sha256_file(art),
                    str(ref_path.relative_to(self.root)) if ref_path else None, time.time(),
                ),
            )
        return version

    # ---- reads ----------------------------------------------------------
    def get(self, name: str, version: int) -> dict[str, Any]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM versions WHERE name=? AND version=?", (name, version)).fetchone()
        if r is None:
            raise RegistryError(f"{name} v{version} not found")
        d = dict(r)
        d["metrics"] = json.loads(d["metrics"])
        d["params"] = json.loads(d["params"])
        return d

    def list_versions(self, name: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT version FROM versions WHERE name=? ORDER BY version", (name,)).fetchall()
        return [self.get(name, r["version"]) for r in rows]

    def stage_version(self, name: str, stage: str) -> int | None:
        with self._conn() as c:
            r = c.execute(
                "SELECT version FROM versions WHERE name=? AND stage=? ORDER BY version DESC LIMIT 1", (name, stage)
            ).fetchone()
        return r["version"] if r else None

    def load_model(self, name: str, version: int) -> Any:
        meta = self.get(name, version)
        path = self.root / meta["artifact"]
        if sha256_file(path) != meta["sha256"]:
            raise ChecksumMismatch(f"{name} v{version}: artifact checksum mismatch")
        return joblib.load(path)

    def load_reference(self, name: str, version: int) -> np.ndarray | None:
        ref = self.get(name, version)["reference"]
        return np.load(self.root / ref) if ref else None

    def events(self, name: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM events WHERE name=? ORDER BY id", (name,)).fetchall()
        return [dict(r) for r in rows]

    # ---- stage transitions ----------------------------------------------
    def transition(
        self, name: str, version: int, to_stage: str, reason: str = "", check_guard: bool = True
    ) -> None:
        if to_stage not in STAGES:
            raise RegistryError(f"unknown stage {to_stage}")
        with self._conn() as c:
            r = c.execute("SELECT stage, metrics FROM versions WHERE name=? AND version=?", (name, version)).fetchone()
            if r is None:
                raise RegistryError(f"{name} v{version} not found")
            cur = r["stage"]
            if (cur, to_stage) not in _ALLOWED:
                raise RegistryError(f"illegal transition {cur} -> {to_stage}")
            if to_stage == "PRODUCTION":
                if check_guard:
                    self._guard(c, name, json.loads(r["metrics"]))
                old = c.execute(
                    "SELECT version FROM versions WHERE name=? AND stage='PRODUCTION'", (name,)
                ).fetchone()
                if old:
                    self._set_stage(c, name, old["version"], "PRODUCTION", "ARCHIVED", f"superseded by v{version}")
            self._set_stage(c, name, version, cur, to_stage, reason)

    def _guard(self, c: sqlite3.Connection, name: str, cand_metrics: dict[str, float]) -> None:
        prod = c.execute("SELECT version, metrics FROM versions WHERE name=? AND stage='PRODUCTION'", (name,)).fetchone()
        if prod is None:
            return
        pm = json.loads(prod["metrics"]).get(self.primary_metric)
        cm = cand_metrics.get(self.primary_metric)
        if pm is None or cm is None:
            raise GuardRejected(f"metric '{self.primary_metric}' missing; cannot compare")
        if cm < pm - self.epsilon:
            raise GuardRejected(
                f"candidate {self.primary_metric}={cm:.4f} < production {pm:.4f} - epsilon {self.epsilon}"
            )

    @staticmethod
    def _set_stage(c, name, version, from_stage, to_stage, reason):
        c.execute("UPDATE versions SET stage=? WHERE name=? AND version=?", (to_stage, name, version))
        c.execute(
            "INSERT INTO events (name, version, from_stage, to_stage, reason, ts) VALUES (?,?,?,?,?,?)",
            (name, version, from_stage, to_stage, reason, time.time()),
        )

    def stage(self, name: str, version: int, reason: str = "") -> None:
        self.transition(name, version, "STAGING", reason)

    def promote(self, name: str, version: int, reason: str = "", check_guard: bool = True) -> None:
        """STAGING -> PRODUCTION; the previous production version is archived atomically.

        The offline guard (candidate metric >= production metric - epsilon) protects manual and
        scheduled promotions. A canary promotion passes `check_guard=False`: after drift the
        production version's stored metric is stale (measured on the old distribution), so live
        A/B evidence on current traffic is the authority instead.
        """
        self.transition(name, version, "PRODUCTION", reason, check_guard)

    def rollback(self, name: str, reason: str = "") -> int:
        """Re-activate the version that was in production before the current one."""
        with self._conn() as c:
            cur = c.execute("SELECT version FROM versions WHERE name=? AND stage='PRODUCTION'", (name,)).fetchone()
            if cur is None:
                raise RegistryError("no production version to roll back")
            prev = c.execute(
                "SELECT version FROM events WHERE name=? AND to_stage='PRODUCTION' AND version<>? "
                "ORDER BY id DESC LIMIT 1",
                (name, cur["version"]),
            ).fetchone()
            if prev is None:
                raise RegistryError("no previous production version")
            self._set_stage(c, name, cur["version"], "PRODUCTION", "ARCHIVED", f"rollback: {reason}")
            self._set_stage(c, name, prev["version"], "ARCHIVED", "PRODUCTION", f"rollback restore: {reason}")
            return prev["version"]

    # ---- canary rollouts ------------------------------------------------
    def start_canary(self, name: str, version: int, fraction: float) -> None:
        if not 0 < fraction < 1:
            raise RegistryError("fraction must be in (0, 1)")
        with self._conn() as c:
            r = c.execute("SELECT stage FROM versions WHERE name=? AND version=?", (name, version)).fetchone()
            if r is None or r["stage"] != "STAGING":
                raise RegistryError("only a STAGING version can be canaried")
            c.execute(
                "INSERT OR REPLACE INTO rollouts VALUES (?,?,?,?)", (name, version, fraction, time.time())
            )

    def end_canary(self, name: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM rollouts WHERE name=?", (name,))

    def active_canary(self, name: str) -> tuple[int, float] | None:
        with self._conn() as c:
            r = c.execute("SELECT version, fraction FROM rollouts WHERE name=?", (name,)).fetchone()
        return (r["version"], r["fraction"]) if r else None
