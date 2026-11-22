"""Load a model from the registry by alias and score fresh tweets (data/new_texts.csv).

    python predict_by_alias.py                      # uses @champion
    python predict_by_alias.py --alias challenger
"""
import argparse
from pathlib import Path

import mlflow
import pandas as pd

import config as cfg

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="champion", choices=["champion", "challenger"])
    args = ap.parse_args()

    mlflow.set_tracking_uri(cfg.TRACKING_URI)
    uri = f"models:/{cfg.REGISTERED_MODEL}@{args.alias}"
    print(f"Loading {uri}")
    model = mlflow.pyfunc.load_model(uri)

    new = pd.read_csv(ROOT / "data" / "new_texts.csv")
    out = model.predict(new[["text"]])
    result = pd.concat([new.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
    result.insert(0, "alias", args.alias)
    (ROOT / "outputs").mkdir(exist_ok=True)
    result.to_csv(ROOT / "outputs" / f"new_text_predictions_{args.alias}.csv", index=False)
    pd.set_option("display.max_colwidth", 70, "display.width", 140)
    print(result[["text", "label", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
