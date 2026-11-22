"""MLflow pyfunc wrapper around a Hugging Face text-classification pipeline.

Only the model id is stored in the MLflow model; weights come from the local
Hugging Face cache (internet needed the first time).
"""
import re

import mlflow.pyfunc
import pandas as pd

import config as cfg

_URL = re.compile(r"https?://\S+")
_HANDLE = re.compile(r"@\w+")


def clean_tweet(text: str) -> str:
    """Light normalisation commonly used for tweet models."""
    text = _HANDLE.sub("@user", str(text))
    text = _URL.sub("http", text)
    return text.strip()


def to_label(raw_label: str) -> str:
    """Map each model's own label names onto negative / neutral / positive."""
    low = raw_label.strip().lower()
    if low.startswith("neg"):
        return "negative"
    if low.startswith("neu"):
        return "neutral"
    if low.startswith("pos"):
        return "positive"
    if low.startswith("label_"):  # generic LABEL_0/1/2, assumed to follow TweetEval order
        return cfg.LABELS[int(low.split("_")[1])]
    raise ValueError(f"Unknown label from model: {raw_label!r}")


class TweetMoodModel(mlflow.pyfunc.PythonModel):
    def __init__(self, hf_model_id: str):
        self.hf_model_id = hf_model_id
        self._pipe = None

    def load_context(self, context):
        self._ensure_pipe()

    def _ensure_pipe(self):
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline(
                "text-classification",
                model=self.hf_model_id,
                device=-1 if cfg.DEVICE == "cpu" else 0,
            )
        return self._pipe

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].tolist()
        else:
            texts = list(model_input)
        pipe = self._ensure_pipe()
        rows = []
        for t in texts:
            out = pipe(clean_tweet(t), truncation=True, max_length=cfg.MAX_LENGTH)[0]
            rows.append({"label": to_label(out["label"]), "confidence": float(out["score"])})
        return pd.DataFrame(rows)
