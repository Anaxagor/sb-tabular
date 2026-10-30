import os

# Keep CatBoost / torch from oversubscribing the machine and from writing
# catboost_info/ into the working directory during tests.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TQDM_DISABLE", "1")
