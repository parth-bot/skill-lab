# TweetMood Arena: Champion vs Challenger Tweet Sentiment Models with MLflow

Skill Lab (MLOps with MLflow) mini project · USN: 1MS25CI090

## 1. Task and models

- **Task:** 3-class tweet sentiment (negative / neutral / positive)
- **Dataset:** TweetEval `sentiment` config (`cardiffnlp/tweet_eval`), validation split, a fixed random sample of **150** tweets (seed 7), cached in `data/eval_tweets.csv` on the first run
- **Model A (champion):** `lxyuan/distilbert-base-multilingual-cased-sentiments-student` (distilled, fast baseline)
- **Model B (challenger):** `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa trained on tweets)
- **Fresh data:** 8 short tweets written by me in `data/new_texts.csv`, none from TweetEval

## 2. How to run

```bash
pip install -r requirements.txt

# terminal 1: tracking server (keep running)
python -m mlflow server --host 127.0.0.1 --port 5000

# terminal 2:
python run_arena.py          # scores A and B, logs to MLflow, applies the rule, fills sections 4 and 5
python promote.py            # registers both models, sets champion / challenger aliases
python predict_by_alias.py   # loads models:/tweet-mood-classifier@champion, predicts data/new_texts.csv
```

`python predict_by_alias.py --alias challenger` loads the other model. To flip the aliases, change them in the
MLflow UI (Models, then tweet-mood-classifier) and rerun the prediction script without editing it.
Browse runs at http://127.0.0.1:5000. The first run downloads both models from Hugging Face.

Files: `config.py` (settings and rule), `tweet_model.py` (pyfunc wrapper), `run_arena.py`, `promote.py`, `predict_by_alias.py`.

## 3. Decision rule (written before the final run)

| Part | Rule |
|---|---|
| Primary metric | **macro-F1** (higher is better), because the three classes are not balanced |
| Minimum gain | Challenger must beat the champion by **at least 0.02 (2 percentage points)**. With 150 tweets one tweet is about 0.7 points, so 2 points is roughly 3 tweets. |
| Latency guardrail | Reject the challenger if its average latency is more than **4x** the champion's |
| Tie-break | If the gain is inside +/-2 points, the **faster model** is the champion. If the challenger is worse by 2 points or more, the champion stays. |
| Direction | accuracy, macro precision, macro recall, macro-F1: higher is better. Latency and total inference time: lower is better. |

These values live in `config.py` and must not be changed after seeing results.

## 4. Results

<!-- RESULTS_START -->
Run `python run_arena.py` to fill this table.
<!-- RESULTS_END -->

Plots and CSVs are in `outputs/` and in the MLflow runs. Up to 8 tweets where the models disagree are in `outputs/disagreements.csv`.

## 5. Decision

<!-- DECISION_START -->
Run `python run_arena.py`, then write your own explanation here.
<!-- DECISION_END -->

## 6. Limitations and declaration

- Only 150 tweets, one dataset, one seed; a gap of 2 points is close to noise at this size.
- The Cardiff NLP model was trained on tweets from the same family of data as TweetEval, so its score may be optimistic.
- Latency is CPU only, one tweet at a time, and depends on the machine.
- The MLflow model stores the Hugging Face model id and loads weights from the local cache, so internet is needed the first time.
- **AI assistance:** I used an AI assistant (Claude) to help write the code and the first draft of this README. I will review the code and results myself and be able to explain them.
