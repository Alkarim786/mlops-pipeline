import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # single-row inference is latency-bound
