#!/usr/bin/env bash
# One-shot runner: starts the MLflow server, runs the whole pipeline, then stops the server.
set -e
python -m mlflow server --host 127.0.0.1 --port 5000 > mlflow_server.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID" EXIT
sleep 8
python run_arena.py
python promote.py
python predict_by_alias.py
python predict_by_alias.py --alias challenger
echo "Done. Take your MLflow screenshots (compare runs, registry aliases, prediction output) into screenshots/ before committing."
