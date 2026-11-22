"""Run both contestants on the same tweets, log everything to MLflow, apply the decision rule.

Usage (tracking server must be running):  python run_arena.py
"""
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

import config as cfg
from tweet_model import TweetMoodModel

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
DATA = ROOT / "data"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


# ---------------------------------------------------------------- data
def load_eval_set() -> pd.DataFrame:
    """Fixed random sample of TweetEval validation tweets, cached in data/eval_tweets.csv."""
    path = DATA / "eval_tweets.csv"
    if path.exists():
        return pd.read_csv(path)
    from datasets import load_dataset

    ds = load_dataset(cfg.DATASET_ID, cfg.DATASET_CONFIG, split=cfg.DATASET_SPLIT)
    df = ds.to_pandas().sample(n=cfg.SAMPLE_SIZE, random_state=cfg.SEED).reset_index(drop=True)
    df["true_label"] = df["label"].map(dict(enumerate(cfg.LABELS)))
    df = df[["text", "true_label"]]
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------- scoring
def score(model: TweetMoodModel, df: pd.DataFrame):
    """Predict one tweet at a time so per-tweet latency is meaningful."""
    for t in df["text"].head(3):  # warm-up, not timed
        model.predict(None, [t])
    preds, confs, lat = [], [], []
    for t in df["text"]:
        start = time.perf_counter()
        out = model.predict(None, [t]).iloc[0]
        lat.append((time.perf_counter() - start) * 1000)
        preds.append(out["label"])
        confs.append(out["confidence"])
    return preds, confs, lat


def compute_metrics(y_true, y_pred, lat_ms) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=cfg.LABELS, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f,
        "avg_latency_ms": float(np.mean(lat_ms)),
        "p95_latency_ms": float(np.percentile(lat_ms, 95)),
        "total_inference_s": float(np.sum(lat_ms) / 1000),
    }


def decide(m_a: dict, m_b: dict):
    """A is the incumbent champion. Returns (winner_letter, explanation)."""
    gain = m_b[cfg.PRIMARY_METRIC] - m_a[cfg.PRIMARY_METRIC]
    slowdown = m_b["avg_latency_ms"] / m_a["avg_latency_ms"]
    if slowdown > cfg.MAX_SLOWDOWN:
        return "A", (f"Challenger rejected by latency guardrail: {slowdown:.1f}x slower "
                     f"(limit {cfg.MAX_SLOWDOWN:.0f}x). Macro-F1 gain was {gain:+.3f}.")
    if gain >= cfg.MIN_GAIN:
        return "B", (f"Challenger wins: macro-F1 gain {gain:+.3f} >= {cfg.MIN_GAIN} "
                     f"and latency {slowdown:.1f}x is inside the {cfg.MAX_SLOWDOWN:.0f}x limit.")
    if gain <= -cfg.MIN_GAIN:
        return "A", f"Champion stays: challenger is worse by {abs(gain):.3f} macro-F1."
    faster = "A" if m_a["avg_latency_ms"] <= m_b["avg_latency_ms"] else "B"
    return faster, (f"Gain {gain:+.3f} is inside +/-{cfg.MIN_GAIN}; tie-break goes to the "
                    f"faster model ({faster}).")


# ---------------------------------------------------------------- plots
def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=cfg.LABELS)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(3), cfg.LABELS, rotation=30)
    ax.set_yticks(range(3), cfg.LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bars(m_a, m_b, path):
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, [m_a[k] for k in keys], 0.4, label="A (champion)")
    ax.bar(x + 0.2, [m_b[k] for k in keys], 0.4, label="B (challenger)")
    ax.set_xticks(x, keys)
    ax.set_ylim(0, 1)
    ax.set_title("Quality metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_latency(lat_a, lat_b, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lat_a, bins=25, alpha=0.65, label="A (champion)")
    ax.hist(lat_b, bins=25, alpha=0.65, label="B (challenger)")
    ax.set_xlabel("latency per tweet (ms)")
    ax.set_ylabel("count")
    ax.set_title("Latency distribution (CPU, one tweet at a time)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- MLflow
def log_pyfunc(hf_model_id, example_in, example_out):
    kwargs = dict(python_model=TweetMoodModel(hf_model_id),
                  signature=infer_signature(example_in, example_out),
                  input_example=example_in)
    try:  # mlflow 3 uses `name`, mlflow 2 uses `artifact_path`
        return mlflow.pyfunc.log_model(name=cfg.MODEL_ARTIFACT_PATH, **kwargs)
    except TypeError:
        return mlflow.pyfunc.log_model(artifact_path=cfg.MODEL_ARTIFACT_PATH, **kwargs)


def run_contestant(letter, spec, df):
    model = TweetMoodModel(spec["hf_model_id"])
    print(f"[{letter}] scoring {len(df)} tweets with {spec['hf_model_id']} ...")
    preds, confs, lat = score(model, df)
    metrics = compute_metrics(df["true_label"], preds, lat)
    slug = spec["hf_model_id"].split("/")[-1]
    cm_path = OUT / f"confusion_{letter}_{slug}.png"
    plot_confusion(df["true_label"], preds, f"{letter}: {slug}", cm_path)
    pred_df = df.assign(pred_label=preds, confidence=confs, latency_ms=lat)
    pred_path = OUT / f"predictions_{letter}.csv"
    pred_df.to_csv(pred_path, index=False)

    with mlflow.start_run(run_name=spec["run_name"], nested=True) as run:
        mlflow.log_params({
            "hf_model_id": spec["hf_model_id"], "role": spec["role"], "task": cfg.TASK,
            "dataset": f"{cfg.DATASET_ID}/{cfg.DATASET_CONFIG}:{cfg.DATASET_SPLIT}",
            "sample_size": cfg.SAMPLE_SIZE, "seed": cfg.SEED, "device": cfg.DEVICE,
            "max_length": cfg.MAX_LENGTH, "student_usn": cfg.STUDENT_USN,
        })
        mlflow.set_tag("library_versions", cfg.library_versions())
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(pred_path))
        sample_in = df[["text"]].head(3)
        log_pyfunc(spec["hf_model_id"], sample_in, model.predict(None, sample_in))
        return run.info.run_id, metrics, pred_df, lat


def update_readme(table_md: str, decision_md: str):
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = re.sub(r"(<!-- RESULTS_START -->).*?(<!-- RESULTS_END -->)",
                  lambda m: f"{m.group(1)}\n{table_md}\n{m.group(2)}", text, flags=re.S)
    text = re.sub(r"(<!-- DECISION_START -->).*?(<!-- DECISION_END -->)",
                  lambda m: f"{m.group(1)}\n{decision_md}\n{m.group(2)}", text, flags=re.S)
    readme.write_text(text, encoding="utf-8")


def main():
    mlflow.set_tracking_uri(cfg.TRACKING_URI)
    mlflow.set_experiment(cfg.EXPERIMENT)
    df = load_eval_set()
    print(f"Eval set: {len(df)} tweets, class counts: {df['true_label'].value_counts().to_dict()}")

    with mlflow.start_run(run_name=cfg.ARENA_RUN_NAME) as parent:
        run_ids, metrics, preds, lats = {}, {}, {}, {}
        for letter, spec in cfg.CONTESTANTS.items():
            run_ids[letter], metrics[letter], preds[letter], lats[letter] = \
                run_contestant(letter, spec, df)

        winner, reason = decide(metrics["A"], metrics["B"])
        plot_bars(metrics["A"], metrics["B"], OUT / "quality_bars.png")
        plot_latency(lats["A"], lats["B"], OUT / "latency_hist.png")

        diff = preds["A"].assign(pred_B=preds["B"]["pred_label"])
        diff = diff[diff["pred_label"] != diff["pred_B"]].rename(columns={"pred_label": "pred_A"})
        diff[["text", "true_label", "pred_A", "pred_B"]].head(8).to_csv(
            OUT / "disagreements.csv", index=False)

        summary = {"decision": reason, "winner": winner, "A": metrics["A"], "B": metrics["B"],
                   "models": {k: v["hf_model_id"] for k, v in cfg.CONTESTANTS.items()}}
        (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

        mlflow.set_tags({"winner": winner, "decision_reason": reason,
                         "run_id_A": run_ids["A"], "run_id_B": run_ids["B"]})
        mlflow.log_params({"primary_metric": cfg.PRIMARY_METRIC, "min_gain": cfg.MIN_GAIN,
                           "max_slowdown": cfg.MAX_SLOWDOWN, "student_usn": cfg.STUDENT_USN})
        for letter in ("A", "B"):
            mlflow.log_metrics({f"{k}_{letter}": v for k, v in metrics[letter].items()})
        mlflow.log_artifacts(str(OUT))
        print(f"Parent run: {parent.info.run_id}")

    rows = ["| Metric | A (champion) | B (challenger) | B - A | Better |", "|---|---|---|---|---|"]
    for k in cfg.HIGHER_IS_BETTER + cfg.LOWER_IS_BETTER:
        a, b = metrics["A"][k], metrics["B"][k]
        better = "higher" if k in cfg.HIGHER_IS_BETTER else "lower"
        nd = 3 if k in cfg.HIGHER_IS_BETTER else 1
        rows.append(f"| {k} | {a:.{nd}f} | {b:.{nd}f} | {b - a:+.{nd}f} | {better} |")
    draft = (f"_Auto-generated draft, rewrite this in your own words after reading the results._\n\n"
             f"A = `{cfg.CONTESTANTS['A']['hf_model_id']}`, B = `{cfg.CONTESTANTS['B']['hf_model_id']}`. "
             f"{reason} Winner: **Model {winner}**.")
    update_readme("\n".join(rows), draft)
    print(reason)
    print("Next: python promote.py")


if __name__ == "__main__":
    main()
