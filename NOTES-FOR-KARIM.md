# Notes for Karim: mlops-pipeline (Project 27)

## Resume bullets (measured on my Ryzen 5 7520U, CPU only, single seeded run)

- Built an **MLOps platform from scratch in Python** (model registry with guarded promotion and rollback, drift detection, automated retraining, FastAPI serving with hot reload, canary deployment) instead of wrapping MLflow; **34 tests**, CI on GitHub Actions, Dockerised API.
- Implemented a **closed loop** that detected injected drift in **1 to 6 steps (100 to 600 requests)**, retrained in **under 0.5 s**, and restored accuracy from **0.54-0.60 back to 0.91** via automated canary promotion.
- Designed a **statistical canary controller** (sticky hash routing, non-inferiority promote rule, one-sided significance rollback) that **rolled back a faulty candidate (0.44 vs 0.78 live accuracy, p < 0.0001)** which had passed offline evaluation (0.91 holdout), then promoted a clean retrain; found and fixed a **24% early false-alarm rate in textbook DDM** (down to 5.6%).

## Likely interview questions and pointers

1. **Canary vs blue/green vs shadow?** Blue/green swaps all traffic at once (fast rollback, no gradual evidence). Canary exposes a small % and measures (`canary.py`). Shadow mirrors traffic with no user impact but cannot measure label-based accuracy as cleanly (Roadmap).
2. **Data drift vs concept drift, and how do you detect each?** Data drift: P(X) changes, detectable without labels (PSI, KS: `drift.py`). Concept drift: P(y|X) changes, needs labels (DDM on the error stream). The scenarios show both; covariate shift fires PSI+KS in 1 step, concept drift needs delayed labels (6 steps).
3. **What triggers retraining and what stops a bad retrain shipping?** Alert (drift) or schedule triggers it; validation blocks bad data; the canary compares live accuracy per arm and the registry can roll back. Mention the faulty-candidate scenario.
4. **Why does the canary bypass the registry's metric guard?** After drift the production metric is stale (old distribution). Live A/B on current traffic is the right evidence. Great "I found a bug and fixed it" story (see README decisions).
5. **How do you make this reproducible and safe to roll back?** Data hash + git SHA + seed + params + reference sample per version, SHA-256 artifact checks, event log, `rollback()` restores the previous production version, atomic SQLite transaction for stage changes.

## Understand this before interviews

- **Routing (`canary.py::route`)**: `sha256(request_id)` -> uniform [0,1); `< fraction` means candidate. Deterministic and monotone in `fraction` (a test proves the 5% set is a subset of the 25% set).
- **Decision (`canary.py::evaluate`)**: order matters. (1) error-rate and latency guards; (2) wait for `min_labelled` per arm; (3) rollback only if significantly worse by more than epsilon; (4) promote if the lower confidence bound of the difference >= -epsilon (non-inferiority); (5) else keep going, and at `max_labelled` decide by point estimate. Equal models need lots of data to be *proven* non-inferior, which is intentional.
- **Label join (`engine.py`)**: `predict` stores `request_id -> (arm, prediction)`; `feedback` pops it and updates that arm's accuracy. Stats and the pending table are reset on any rollout change so the two arms are compared over the same window.
- **DDM (`drift.py`)**: tracks running error rate p and std s; baseline is the min of p+s; warning at +2s, drift at +3s. The textbook version false-alarms when an early lucky streak gives s_min near 0; I added a 100-sample warm-up and a +5-point minimum rise (measured 24% -> 5.6%). It is still imperfect, which is why the canary is the safety net.
- **Closed loop (`simulate.py`)**: state machine `monitoring -> collecting -> canary -> monitoring`. Retraining waits for >= 800 post-change labelled rows; data-drift change point = earliest recent batch that is itself flagged; after a rollback with drift unresolved it goes back to collecting.
- **Limits to state yourself**: one seed (no confidence intervals on the scenario table), synthetic data, labels assumed to arrive for all requests, SQLite/local FS only, artifacts are pickles, no shadow mode or progressive rollout yet.
