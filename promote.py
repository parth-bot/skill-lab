"""Register both models and point the `champion` / `challenger` aliases at the right versions.

The winner is read from the `winner` tag of the latest arena run, so no result is typed by hand.
"""
import mlflow
from mlflow import MlflowClient

import config as cfg


def main():
    mlflow.set_tracking_uri(cfg.TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(cfg.EXPERIMENT)
    if exp is None:
        raise SystemExit("Experiment not found - run run_arena.py first.")
    runs = client.search_runs([exp.experiment_id],
                              filter_string=f"attributes.run_name = '{cfg.ARENA_RUN_NAME}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit("No arena run found - run run_arena.py first.")
    tags = runs[0].data.tags
    winner = tags["winner"]
    loser = "B" if winner == "A" else "A"

    versions = {}
    for letter in ("A", "B"):
        uri = f"runs:/{tags['run_id_' + letter]}/{cfg.MODEL_ARTIFACT_PATH}"
        mv = mlflow.register_model(uri, cfg.REGISTERED_MODEL)
        versions[letter] = mv.version
        print(f"Registered model {letter} as version {mv.version}")

    client.set_registered_model_alias(cfg.REGISTERED_MODEL, "champion", versions[winner])
    client.set_registered_model_alias(cfg.REGISTERED_MODEL, "challenger", versions[loser])
    print(f"champion   -> v{versions[winner]} (Model {winner})")
    print(f"challenger -> v{versions[loser]} (Model {loser})")
    print("Next: python predict_by_alias.py")


if __name__ == "__main__":
    main()
