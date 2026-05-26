import streamlit as st
from utils import load_nba_model
import joblib
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from utils import team_ids

model = load_nba_model()


def get_team_stats(team_id):
    log = leaguegamelog.LeagueGameLog(season='2024-25', season_type_all_star='Regular Season')
    df = log.get_data_frames()[0]

    team_df = df[df["TEAM_ID"] == team_id].copy()
    team_df['GAME_DATE'] = pd.to_datetime(team_df['GAME_DATE'])
    team_df = team_df.sort_values(by='GAME_DATE')

    features = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']

    latest_stats = team_df[features].tail(20).mean().to_frame().T
    latest_stats.columns = [f'rolling_{c}' for c in latest_stats.columns]
    return latest_stats




st.title("🏀 NBA AI Predictor 🏀")

home_team = st.selectbox("Select Home Team", list(team_ids.keys()))
away_team = st.selectbox("Select Away Team", list(team_ids.keys()), index=1)

if st.button("Predict Winner"):
    with st.spinner("Fetching Game Stats"):
        home_id = team_ids[home_team]
        away_id = team_ids[away_team]

        home_stats = get_team_stats(home_id)
        away_stats = get_team_stats(away_id)

        matchup = pd.concat([home_stats.add_suffix('_home'), away_stats.add_suffix('_away')], axis=1)
        prob = model.predict_proba(matchup)[0][1]
        st.subheader(f"Prediction: {home_team} vs {away_team}")
        
        if prob > 0.5:
            st.success(f"🏆 The AI favors the **{home_team}** with {prob:.1%} confidence.")
        else:
            st.error(f"🏆 The AI favors the **{away_team}** with {1 - prob:.1%} confidence.")
