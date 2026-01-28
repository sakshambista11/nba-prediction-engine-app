# 🏀 NBA Prediction Engine

This repository contains an end-to-end machine learning web app that predicts NBA game outcomes and lets you simulate custom 5v5 matchups. It uses an XGBoost classifier trained on historical NBA games and pulls active-season data for the interactive features.

---

## What this project does

### 1) Real-Game Predictor
- Predicts the winner of scheduled NBA games
- Builds model inputs from **team rolling averages (last 20 games)**

### 2) 5v5 Team Simulator
- Lets you create two custom lineups using **active 2024–25 player data**
- Aggregates individual player stats into a “synthetic team” profile
- Runs the same prediction pipeline to estimate a winner

---

## How it works

1. **Data ingestion**
   - Uses `nba_api` to retrieve current-season team/player data (2024–25)

2. **Feature engineering**
   - Computes **20-game rolling averages**
   - Uses ~18 stat categories (PTS, REB, AST, FG%, etc.)

3. **Model**
   - **XGBoost Classifier**
   - Trained on **~5,000 games (2019–2024)**
   - Reported accuracy: **~63%** (baseline project metric)

4. **App**
   - Built with **Streamlit** using a multi-page structure

---

## Tech stack

- **Data:** `nba_api` (live 2024–25 season data)
- **Modeling:** XGBoost
- **Frontend:** Streamlit
- **Preprocessing:** rolling averages + team/player aggregation

---

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/sakshambista11/nba-ai-predictor.git
    cd nba-ai-predictor
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:

    ```bash
    streamlit run Home.py
    ```
