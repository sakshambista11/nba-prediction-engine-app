import streamlit as st
import joblib
import pandas as pd
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import playercareerstats

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

features = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']

@st.cache_resource
def load_nba_model():
    return joblib.load('nba_model.pkl')

def get_player_id(playername):
    playerlist = players.find_players_by_full_name(playername)
    if not playerlist:
        raise ValueError(f"Player '{playername}' not found in NBA API.")
    return playerlist[0]['id']

def rolling_average(playerid):
    features = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
    stats = playergamelog.PlayerGameLog(playerid, "2024-25", "Regular Season")
    df = stats.get_data_frames()[0]
    df = pd.DataFrame(df)
    if df.empty:
        return pd.DataFrame()
    df["Target"] = df["WL"].map({"L": 0, "W":1})
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(by=["GAME_DATE"])
    for col in features:
        df[f'rolling_{col}'] = df[col].transform(lambda x: x.shift(1).rolling(20, min_periods = 1).mean())
    rolling = [x for x in df.columns if "rolling" in x]
    
    return df[rolling].tail(1)

def get_lineup_stats(player_ids):
    all_cols = [f'rolling_{c}' for c in ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']]
    sums_cols = [f'rolling_{c}' for c in ['PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']]
    avgs_cols = [f'rolling_{c}' for c in ['FG_PCT', 'FG3_PCT', 'FT_PCT']]
    data_append = []

    for i in player_ids:
        avg = rolling_average(i)
        if not avg.empty:
            data_append.append(avg)

    df_lineup = pd.concat(data_append)
    team_sums = df_lineup[sums_cols].sum().to_frame().T
    team_avgs = df_lineup[avgs_cols].mean().to_frame().T
    combined = pd.concat([team_sums, team_avgs], axis=1)
    
    return combined[all_cols]










