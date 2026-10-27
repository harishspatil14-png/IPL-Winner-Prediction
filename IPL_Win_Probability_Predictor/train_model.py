"""
Train a second-innings IPL chasing win-probability model from historical data.

Expected files:
  dataset/deliveries.csv  (ball-by-ball IPL data)
  dataset/matches.csv     (match metadata including winner and venue)

Supports common Kaggle IPL columns (id/inning/over/ball/total_runs/is_wicket)
and datasets that record dismissals as player_dismissed/dismissal_kind.
"""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report

BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset"
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "batting_team", "bowling_team", "venue", "target",
    "score", "wickets_lost", "balls_remaining",
    "runs_remaining", "current_run_rate", "required_run_rate"
]
CAT = ["batting_team", "bowling_team", "venue"]
NUM = [c for c in FEATURES if c not in CAT]

def first_existing(df, names, required=True):
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"Missing expected column. Tried: {names}. Available columns: {list(df.columns)}")
    return None

def main():
    deliveries_path = DATA / "deliveries.csv"
    matches_path = DATA / "matches.csv"
    if not deliveries_path.exists():
        raise FileNotFoundError(
            f"Missing {deliveries_path}. Download a genuine IPL ball-by-ball CSV and place it there."
        )
    if not matches_path.exists():
        raise FileNotFoundError(
            f"Missing {matches_path}. Match metadata is required for venue and winner labels."
        )

    d = pd.read_csv(deliveries_path, low_memory=False)
    m = pd.read_csv(matches_path, low_memory=False)
    id_d = first_existing(d, ["match_id", "id", "ID"])
    id_m = first_existing(m, ["match_id", "id", "ID"])
    inning = first_existing(d, ["inning", "innings", "Innings"])
    over = first_existing(d, ["over", "overs", "Overs"])
    total_runs = first_existing(d, ["total_runs", "total_run", "TotalRun"])
    wicket = first_existing(d, ["is_wicket", "is_wicket_delivery", "IsWicketDelivery"], required=False)
    dismissed = first_existing(d, ["player_dismissed", "PlayerDismissed"], required=False)
    dismissal_kind = first_existing(d, ["dismissal_kind", "DismissalKind"], required=False)
    if wicket is None and dismissed is None:
        raise ValueError(
            "Missing wicket information. Expected an is_wicket column or player_dismissed. "
            f"Available columns: {list(d.columns)}"
        )
    bat = first_existing(d, ["batting_team", "BattingTeam"])
    bowl = first_existing(d, ["bowling_team", "BowlingTeam"])
    winner = first_existing(m, ["winner", "match_won_by", "winning_team"])
    venue_col = first_existing(m, ["venue", "Venue"], required=False)
    date_col = first_existing(m, ["date", "start_date"], required=False)

    d = d.copy()
    d["_match_id"] = d[id_d]
    d["_inning"] = pd.to_numeric(d[inning], errors="coerce")
    d["_over"] = pd.to_numeric(d[over], errors="coerce")
    d["_runs"] = pd.to_numeric(d[total_runs], errors="coerce").fillna(0)
    if wicket is not None:
        d["_wicket"] = pd.to_numeric(d[wicket], errors="coerce").fillna(0)
    else:
        dismissed_player = d[dismissed].fillna("").astype(str).str.strip().ne("")
        if dismissal_kind:
            kind = d[dismissal_kind].fillna("").astype(str).str.strip().str.lower()
            dismissed_player &= kind.ne("retired hurt")
        d["_wicket"] = dismissed_player.astype(int)
    d["_bat"] = d[bat].astype(str).str.strip()
    d["_bowl"] = d[bowl].astype(str).str.strip()

    meta = m.copy()
    meta["_match_id"] = meta[id_m]
    meta["_winner"] = meta[winner].astype(str).str.strip()
    meta["_venue"] = meta[venue_col].fillna("Unknown").astype(str).str.strip() if venue_col else "Unknown"
    if date_col:
        meta["_date"] = pd.to_datetime(meta[date_col], errors="coerce")
    else:
        meta["_date"] = pd.NaT

    # Only completed matches with a known winner and a usable first and second innings.
    valid_meta = meta[meta["_winner"].notna() & ~meta["_winner"].isin(["", "nan", "None", "no result", "No Result"])]
    valid_ids = set(valid_meta["_match_id"])
    d = d[d["_match_id"].isin(valid_ids) & d["_inning"].isin([1, 2])].copy()
    d = d.sort_values(["_match_id", "_inning", "_over"], kind="stable")

    # Build first-innings target (score + 1) from all first-innings deliveries.
    first = d[d["_inning"] == 1].groupby("_match_id")["_runs"].sum().rename("_first_total")
    second = d[d["_inning"] == 2].copy()
    second = second.join(first, on="_match_id")
    second = second.dropna(subset=["_first_total"])
    second["_target"] = second["_first_total"].astype(int) + 1

    # Determine legal deliveries from extras types or wide/no-ball run columns when available.
    extras_col = first_existing(d, ["extras_type", "extra_type", "ExtraType"], required=False)
    if extras_col:
        extra_text = d[extras_col].fillna("").astype(str).str.lower()
        d["_legal"] = ~extra_text.isin(["wides", "wide", "no ball", "noballs", "no-ball", "no_ball"])
    else:
        wide_runs = first_existing(d, ["wide_runs", "WideRuns"], required=False)
        noball_runs = first_existing(d, ["noball_runs", "no_ball_runs", "NoballRuns"], required=False)
        if wide_runs or noball_runs:
            illegal = pd.Series(False, index=d.index)
            if wide_runs is not None:
                illegal |= pd.to_numeric(d[wide_runs], errors="coerce").fillna(0).gt(0)
            if noball_runs is not None:
                illegal |= pd.to_numeric(d[noball_runs], errors="coerce").fillna(0).gt(0)
            d["_legal"] = ~illegal
        else:
            # Prefer datasets with explicit extras or wide/no-ball columns for exact ball counts.
            d["_legal"] = True
            warnings.warn("No extras_type or wide/no-ball columns found; legal-ball counts may be approximate.")

    legal = d[d["_inning"] == 2].copy()
    legal["_legal"] = legal["_legal"].astype(int)
    legal["_score"] = legal.groupby("_match_id")["_runs"].cumsum()
    legal["_wickets"] = legal.groupby("_match_id")["_wicket"].cumsum().clip(0, 10)
    legal["_balls_bowled"] = legal.groupby("_match_id")["_legal"].cumsum()
    legal = legal[legal["_match_id"].isin(set(second["_match_id"]))].copy()
    legal = legal.join(first, on="_match_id", rsuffix="_first")
    legal["_target"] = legal["_first_total"].astype(int) + 1
    legal = legal.merge(
        valid_meta[["_match_id", "_winner", "_venue", "_date"]].drop_duplicates("_match_id"),
        on="_match_id", how="left"
    )
    legal["_bat"] = legal["_bat"].astype(str).str.strip()
    legal["_bowl"] = legal["_bowl"].astype(str).str.strip()
    legal["_venue"] = legal["_venue"].fillna("Unknown").astype(str).str.strip()

    # Keep only states before the chase is already won/over; labels are the actual match result.
    legal = legal[
        (legal["_balls_bowled"] <= 120) &
        (legal["_score"] < legal["_target"]) &
        (legal["_wickets"] < 10) &
        (legal["_bat"] != "") & (legal["_bowl"] != "")
    ].copy()
    legal["_balls_remaining"] = 120 - legal["_balls_bowled"]
    legal["_runs_remaining"] = legal["_target"] - legal["_score"]
    legal["_overs_bowled"] = legal["_balls_bowled"] / 6.0
    legal["_overs_left"] = legal["_balls_remaining"] / 6.0
    legal["_current_run_rate"] = np.where(
        legal["_overs_bowled"] > 0, legal["_score"] / legal["_overs_bowled"], 0.0
    )
    legal["_required_run_rate"] = np.where(
        legal["_overs_left"] > 0, legal["_runs_remaining"] / legal["_overs_left"], 0.0
    )
    legal["_label"] = (legal["_winner"] == legal["_bat"]).astype(int)

    frame = pd.DataFrame({
        "batting_team": legal["_bat"],
        "bowling_team": legal["_bowl"],
        "venue": legal["_venue"],
        "target": legal["_target"],
        "score": legal["_score"],
        "wickets_lost": legal["_wickets"],
        "balls_remaining": legal["_balls_remaining"],
        "runs_remaining": legal["_runs_remaining"],
        "current_run_rate": legal["_current_run_rate"],
        "required_run_rate": legal["_required_run_rate"],
        "label": legal["_label"],
        "match_id": legal["_match_id"]
    })
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["label", "match_id"])
    if len(frame) < 1000 or frame["label"].nunique() < 2:
        raise ValueError(
            f"Only {len(frame)} usable training states found. Check the dataset columns and match metadata."
        )

    # Split by match rather than row to prevent states from one match leaking into both sets.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(frame[FEATURES], frame["label"], groups=frame["match_id"]))
    X_train, X_test = frame.iloc[train_idx][FEATURES], frame.iloc[test_idx][FEATURES]
    y_train, y_test = frame.iloc[train_idx]["label"], frame.iloc[test_idx]["label"]

    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("numeric", "passthrough", NUM)
    ])
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, min_samples_leaf=4, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n=== Hold-out evaluation (match-level split) ===")
    print(f"Training states: {len(train_idx):,} | Test states: {len(test_idx):,}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    try:
        print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
        print(f"Log loss: {log_loss(y_test, proba):.4f}")
    except ValueError:
        print("ROC-AUC/log loss unavailable for this test split.")
    print(classification_report(y_test, pred, digits=3, zero_division=0))

    joblib.dump(pipeline, MODEL_DIR / "ipl_win_model.joblib")
    metadata = {
        "teams": sorted(set(frame["batting_team"].unique()) | set(frame["bowling_team"].unique())),
        "venues": sorted(frame["venue"].dropna().unique().tolist()),
        "max_overs": 20,
        "training_rows": int(len(frame)),
        "matches": int(frame["match_id"].nunique()),
        "features": FEATURES
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved model: {MODEL_DIR / 'ipl_win_model.joblib'}")
    print(f"Saved metadata: {MODEL_DIR / 'metadata.json'}")
    print("Note: this is an educational model; evaluate calibration and recent-season performance before real-world use.")

if __name__ == "__main__":
    main()
