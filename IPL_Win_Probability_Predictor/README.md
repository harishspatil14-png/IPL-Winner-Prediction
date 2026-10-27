# IPL Win Probability Predictor

A Flask + Scikit-learn web app that estimates **only the second-innings batting team's probability of winning** a chase. It uses a model trained from historical IPL ball-by-ball data. It does not show the bowling team's win probability and does not use random or hardcoded predictions.

## 1. Dataset

This project needs two genuine historical CSV files:

- `dataset/deliveries.csv` — ball-by-ball delivery records.
- `dataset/matches.csv` — match metadata, including match ID, venue, and winner.

You can obtain IPL ball-by-ball data from:
- Cricsheet's IPL data downloads: https://cricsheet.org/downloads/
- The IPL dataset repository and schema notes: https://github.com/ritesh-ojha/IPL-DATASET
- Kaggle IPL Complete Dataset (check the dataset's license and column schema before use).

Use a real dataset; do not create fabricated training data. The training script supports common Kaggle-style column names such as `id`/`match_id`, `inning`/`innings`, `total_runs`, `is_wicket`, `winner`, and `venue`. If your downloaded files use different column names, update the alias lists in `train_model.py`.

For many Kaggle datasets, `deliveries.csv` has columns such as `match_id`, `inning`, `over`, `ball`, `batting_team`, `bowling_team`, `total_runs`, `is_wicket`, and `extras_type`. Other common versions store dismissal data in `player_dismissed` and `dismissal_kind`, and illegal-ball data in `wide_runs` and `noball_runs`; these columns are supported as alternatives. `matches.csv` usually contains `id` or `match_id`, `winner`, and `venue`. Ensure the delivery and match IDs refer to the same matches.

## 2. Install and run in VS Code

Open this folder in VS Code. Open **Terminal → New Terminal** and run:

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

Place the real CSVs in `dataset/`, then train and evaluate:

```powershell
python train_model.py
```

Training creates:

- `model/ipl_win_model.joblib`
- `model/metadata.json`

Start the web app:

```powershell
python app.py
```

Open the local URL printed in the terminal, usually http://127.0.0.1:5000.

## 3. Inputs and calculations

- **Current score**: runs scored by the chasing team.
- **Wickets lost**: wickets lost, from 0 to 10.
- **Overs completed**: cricket notation, not decimal fractions. `14.3` means 14 overs and 3 legal balls, so 87 legal balls have been bowled.
- **Target**: score required to win.
- **Teams and venue**: choose exact options from the searchable suggestions.

For a 20-over innings:

- Legal balls bowled = `whole overs × 6 + balls in current over`.
- Balls remaining = `120 − legal balls bowled`.
- Runs remaining = `target − current score`.
- Current run rate = `current score ÷ overs bowled`.
- Required run rate = `runs remaining ÷ overs remaining`.

Wides and no-balls do not count as legal balls. The training script uses `extras_type` when available; if the column is missing, it prints a warning that legal-ball counts may be approximate. For best results, use a dataset that includes extras information.

## 4. Model and evaluation

The training pipeline uses a Random Forest classifier, one-hot encodes batting team, bowling team, and venue, and trains on second-innings states. The target label is whether the chasing team actually won the match. A match-level train/test split reduces leakage from multiple ball states in the same match. Accuracy, ROC-AUC, log loss, and a classification report are printed after training.

This is an educational baseline. Performance may vary by dataset and season; validate calibration and recent-season performance before presenting the probability as a reliable forecast. Do not interpret probability as certainty.

## 5. Troubleshooting

- **Model not found**: put the CSVs in `dataset/` and run `python train_model.py` before starting Flask.
- **No team/venue options**: training did not complete or `model/metadata.json` is missing.
- **Missing column**: inspect the actual CSV headers and add the column name to the alias list in `train_model.py`.
- **Training is slow**: use a recent Python version supported by scikit-learn and close unnecessary applications.
- **PowerShell activation blocked**: use Command Prompt activation, or run `.venv\Scripts\python.exe train_model.py` and `.venv\Scripts\python.exe app.py`.

## 6. Attribution

Credit the original dataset provider and comply with the dataset's license and attribution requirements. This project is not affiliated with or endorsed by the BCCI or IPL.
