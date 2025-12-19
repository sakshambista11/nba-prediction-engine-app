import streamlit as st
from utils import load_nba_model
import joblib
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats

model = load_nba_model()

team_ids = {
    "Atlanta Hawks": 1610612737,
    "Boston Celtics": 1610612738,
    "Brooklyn Nets": 1610612751,
    "Charlotte Hornets": 1610612766,
    "Chicago Bulls": 1610612741,
    "Cleveland Cavaliers": 1610612739,
    "Dallas Mavericks": 1610612742,
    "Denver Nuggets": 1610612743,
    "Detroit Pistons": 1610612765,
    "Golden State Warriors": 1610612744,
    "Houston Rockets": 1610612745,
    "Indiana Pacers": 1610612754,
    "LA Clippers": 1610612746,
    "Los Angeles Lakers": 1610612747,
    "Memphis Grizzlies": 1610612763,
    "Miami Heat": 1610612748,
    "Milwaukee Bucks": 1610612749,
    "Minnesota Timberwolves": 1610612750,
    "New Orleans Pelicans": 1610612740,
    "New York Knicks": 1610612752,
    "Oklahoma City Thunder": 1610612760,
    "Orlando Magic": 1610612753,
    "Philadelphia 76ers": 1610612755,
    "Phoenix Suns": 1610612756,
    "Portland Trail Blazers": 1610612757,
    "Sacramento Kings": 1610612758,
    "San Antonio Spurs": 1610612759,
    "Toronto Raptors": 1610612761,
    "Utah Jazz": 1610612762,
    "Washington Wizards": 1610612764
}

def get_team_stats(team_id):
    log = leaguegamelog.LeagueGameLog(season='2023-24', season_type_all_star='Regular Season')
    df = log.get_data_frames()[0]

    team_df = df[df["TEAM_ID"] == team_id].copy()
    team_df['GAME_DATE'] = pd.to_datetime(team_df['GAME_DATE'])
    team_df = team_df.sort_values(by='GAME_DATE')

    features = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']

    latest_stats = team_df[features].tail(10).mean().to_frame().T
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
