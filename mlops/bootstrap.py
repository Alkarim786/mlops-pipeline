"""Idempotently create and promote a first model so the API has something to serve.

    python -m mlops.bootstrap --root ./registry-data --name demo
"""

from __future__ import annotations

import argparse

import numpy as np

from .data import make_batch
from .pipeline import train_and_register
from .registry import Registry


def bootstrap(root: str, name: str, seed: int = 0) -> int:
    reg = Registry(root)
    existing = reg.stage_version(name, "PRODUCTION")
    if existing is not None:
        return existing
    X, y = make_batch(4000, np.random.default_rng(seed))
    res = train_and_register(reg, name, X, y, seed=seed)
    reg.stage(name, res.version, "bootstrap")
    reg.promote(name, res.version, "bootstrap")
    return res.version


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./registry-data")
    ap.add_argument("--name", default="demo")
    args = ap.parse_args()
    print(f"production version of '{args.name}': v{bootstrap(args.root, args.name)}")


if __name__ == "__main__":
    main()
