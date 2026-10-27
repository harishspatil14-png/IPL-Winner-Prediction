from pathlib import Path
import json
import math
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "ipl_win_model.joblib"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"

app = Flask(__name__)
FEATURES = [
    "batting_team", "bowling_team", "venue", "target",
    "score", "wickets_lost", "balls_remaining",
    "runs_remaining", "current_run_rate", "required_run_rate"
]

model = None
metadata = {}

def load_artifacts():
    global model, metadata
    if MODEL_PATH.exists() and METADATA_PATH.exists():
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    else:
        model, metadata = None, {}

load_artifacts()

def legal_balls_from_overs(overs):
    """Convert cricket notation such as 17.3 into legal balls (17*6+3)."""
    whole = int(overs)
    fractional = round((float(overs) - whole) * 10)
    if fractional < 0 or fractional > 5:
        raise ValueError("Overs must use cricket notation, e.g. 17.3 (not 17.5 decimal overs).")
    return whole * 6 + fractional

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/options")
def options():
    if not metadata:
        return jsonify({"error": "Model metadata not found. Run train_model.py first."}), 503
    return jsonify({
        "teams": metadata.get("teams", []),
        "venues": metadata.get("venues", []),
        "max_overs": metadata.get("max_overs", 20)
    })

@app.post("/api/predict")
def predict():
    if model is None:
        return jsonify({"error": "Trained model not found. Add the historical dataset and run python train_model.py."}), 503

    payload = request.get_json(silent=True) or {}
    try:
        batting = str(payload.get("batting_team", "")).strip()
        bowling = str(payload.get("bowling_team", "")).strip()
        venue = str(payload.get("venue", "")).strip()
        score = int(payload.get("score"))
        wickets = int(payload.get("wickets_lost"))
        target = int(payload.get("target"))
        overs = float(payload.get("overs_completed"))

        if not batting or not bowling or not venue:
            raise ValueError("Select a batting team, bowling team, and venue.")
        if batting == bowling:
            raise ValueError("Batting team and bowling team must be different.")
        if batting not in metadata.get("teams", []) or bowling not in metadata.get("teams", []):
            raise ValueError("Choose teams from the suggestions.")
        if venue not in metadata.get("venues", []):
            raise ValueError("Choose a venue from the suggestions.")
        if score < 0 or target < 1:
            raise ValueError("Score must be 0 or higher and target must be at least 1.")
        if wickets < 0 or wickets > 10:
            raise ValueError("Wickets lost must be between 0 and 10.")

        balls_bowled = legal_balls_from_overs(overs)
        max_balls = int(metadata.get("max_overs", 20)) * 6
        if balls_bowled < 0 or balls_bowled > max_balls:
            raise ValueError(f"Overs completed must be between 0.0 and {metadata.get('max_overs', 20)}.0.")
        if balls_bowled % 6 == 0 and overs != int(overs):
            raise ValueError("Use valid cricket overs notation, e.g. 12.0 or 12.3.")
        if balls_bowled == 0 and score > 0:
            raise ValueError("Overs must be greater than zero if runs have been scored.")
        if score >= target:
            return jsonify({
                "batting_team": batting, "win_probability": 100.0,
                "runs_remaining": 0, "balls_remaining": max(0, max_balls - balls_bowled),
                "current_run_rate": round(score / (balls_bowled / 6), 2) if balls_bowled else 0.0,
                "required_run_rate": 0.0,
                "summary": f"{batting} has reached the target and won the chase."
            })
        if wickets >= 10 or balls_bowled >= max_balls:
            return jsonify({
                "batting_team": batting, "win_probability": 0.0,
                "runs_remaining": target - score, "balls_remaining": 0,
                "current_run_rate": round(score / (balls_bowled / 6), 2) if balls_bowled else 0.0,
                "required_run_rate": 0.0,
                "summary": f"{batting} has no wickets or legal balls remaining to complete the chase."
            })

        balls_remaining = max_balls - balls_bowled
        runs_remaining = target - score
        overs_bowled = balls_bowled / 6
        overs_left = balls_remaining / 6
        crr = score / overs_bowled if overs_bowled else 0.0
        rrr = runs_remaining / overs_left if overs_left else 0.0

        row = pd.DataFrame([{
            "batting_team": batting, "bowling_team": bowling, "venue": venue,
            "target": target, "score": score, "wickets_lost": wickets,
            "balls_remaining": balls_remaining, "runs_remaining": runs_remaining,
            "current_run_rate": crr, "required_run_rate": rrr
        }], columns=FEATURES)

        probability = float(model.predict_proba(row)[0][1]) * 100
        probability = max(0.0, min(100.0, probability))
        summary = (
            f"{batting} need {runs_remaining} runs from {balls_remaining} legal balls, "
            f"with {10 - wickets} wicket(s) in hand. The required run rate is {rrr:.2f}."
        )
        return jsonify({
            "batting_team": batting,
            "win_probability": round(probability, 1),
            "runs_remaining": runs_remaining,
            "balls_remaining": balls_remaining,
            "current_run_rate": round(crr, 2),
            "required_run_rate": round(rrr, 2),
            "summary": summary
        })
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc) or "Please check all inputs."}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check that the trained model matches this application version."}), 500

if __name__ == "__main__":
    app.run(debug=True)
