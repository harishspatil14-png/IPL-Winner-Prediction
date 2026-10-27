# 🏏 IPL Winner Prediction

A Machine Learning web application that predicts the winning probability of the batting team during the **second innings of an IPL match** based on the current match situation.

## 📌 Project Overview

The IPL Winner Prediction system uses machine learning to estimate the chances of the chasing team winning a match. Users can enter match details and receive a predicted winning probability.

The application is built using Python, Flask, and Machine Learning, with a user-friendly web interface.

## ✨ Features

- 🏏 Predicts the second-innings batting team's winning probability.
- 📊 Uses match situation and team statistics for prediction.
- 🏟️ Supports venue selection.
- 🎯 Accepts match details such as current score, wickets, overs, and target.
- 🔍 Provides searchable, auto-complete dropdowns for team and venue selection.
- 🚫 Prevents selecting the same team as both batting and bowling teams.
- 🌐 Web interface powered by Flask.

## 🛠️ Technologies Used

- **Python** – Core programming language
- **Flask** – Web application framework
- **Pandas** – Data processing
- **Scikit-learn** – Machine Learning
- **HTML, CSS, JavaScript** – Frontend development
- **Gunicorn** – Production WSGI server
- **Render** – Web application deployment

## ⚙️ Input Parameters

The application uses the following match details:

- Batting Team
- Bowling Team
- Match Venue
- Current Score
- Wickets Lost
- Overs Completed
- Target Score

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/harishspatil14-png/IPL-Winner-Prediction.git
```

### 2. Navigate to the project folder

```bash
cd IPL-Winner-Prediction/IPL_Win_Probability_Predictor
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the application

```bash
python app.py
```

Open the local URL displayed in your terminal, usually:

`http://127.0.0.1:5000`

## ☁️ Deployment

This project can be deployed using Render.

- **Root Directory:** `IPL_Win_Probability_Predictor`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`

Ensure that the required model files and dependencies are included in the repository.

## 📂 Project Structure

```text
IPL-Winner-Prediction/
└── IPL_Win_Probability_Predictor/
    ├── app.py
    ├── train_model.py
    ├── requirements.txt
    ├── templates/
    ├── static/
    └── model files
```

*Note: The exact model filenames and additional files may vary depending on your implementation.*

## 🎯 Project Objective

The main objective is to demonstrate how Machine Learning can be applied to cricket match analysis to estimate the chasing team's winning probability based on the match situation.

## 👨‍💻 Author

**Harish Patil**

- GitHub: [harishspatil14-png](https://github.com/harishspatil14-png)

---

⭐ If you find this project interesting, consider giving the repository a star!
