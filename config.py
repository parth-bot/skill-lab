"""Central settings for TweetMood Arena.

The decision-rule constants at the bottom are the same ones written in README section 3.
Fix them BEFORE the final run and do not change them after seeing the results.
"""
import os
from importlib.metadata import version

STUDENT_USN = "1MS25CI090"

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "tweet-mood-arena"
REGISTERED_MODEL = "tweet-mood-classifier"
TASK = "tweet-sentiment"
ARENA_RUN_NAME = f"arena_{TASK}"
MODEL_ARTIFACT_PATH = "model"

# ---- Data: TweetEval sentiment (3 classes) ----
DATASET_ID = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "sentiment"
DATASET_SPLIT = "validation"
SAMPLE_SIZE = 150
SEED = 7
LABELS = ["negative", "neutral", "positive"]  # TweetEval ids 0, 1, 2
DEVICE = "cpu"
MAX_LENGTH = 128

# ---- Contestants ----
# Contestant 1 starts as the champion (small and fast), contestant 2 is the challenger.
CONTESTANTS = {
    "A": {
        "run_name": "A_distilbert-multilingual-student",
        "hf_model_id": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        "role": "champion",
    },
    "B": {
        "run_name": "B_twitter-roberta-base",
        "hf_model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "role": "challenger",
    },
}

# ---- Decision rule (README section 3) ----
PRIMARY_METRIC = "macro_f1"   # higher is better
MIN_GAIN = 0.02               # challenger must beat champion by >= 2 percentage points
MAX_SLOWDOWN = 4.0            # reject challenger if avg latency > 4x champion's
# Tie-break: if the gain is below MIN_GAIN, the faster model keeps / takes the champion alias.

HIGHER_IS_BETTER = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
LOWER_IS_BETTER = ["avg_latency_ms", "p95_latency_ms", "total_inference_s"]


def library_versions() -> str:
    found = []
    for name in ("transformers", "torch", "mlflow", "datasets", "scikit-learn"):
        try:
            found.append(f"{name}={version(name)}")
        except Exception:
            found.append(f"{name}=not-installed")
    return ", ".join(found)
