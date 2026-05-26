# 🏀 NBA Win Probability AI

Predicts NBA game winners with 63% accuracy using XGBoost trained on 5,000+ games (2019–2024). Features a live Streamlit interface with two modes: a real-game team predictor and a custom 5v5 player simulator — both powered by live 2024–25 season data from the NBA API.

🚀 **[Live Demo](https://nba-prediction-engine-app.streamlit.app/)**

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

## How it works (high level)

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

1) Clone the repository
git clone https://github.com/YOUR_USERNAME/nba-ai-predictor.git
cd nba-ai-predictor

2) Install dependencies
pip install -r requirements.txt

3) Run the app
streamlit run Home.py