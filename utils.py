import streamlit as st
import joblib
import pandas as pd
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import playercareerstats

features = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']

@st.cache_resource
def load_nba_model():
    return joblib.load('nba_model.pkl')

def get_player_id(playername):
    playerlist = players.find_players_by_full_name(playername)
    print(playerlist)
    if playerlist:
        player_id = playerlist[0]['id']
    return player_id

def rolling_average(playerid):
    features = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
    stats = playergamelog.PlayerGameLog(playerid, "2024-25", "Regular Season")
    df = stats.get_data_frames()[0]
    df = pd.DataFrame(df)
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
        rolling_avg = rolling_average(i)
        data_append.append(rolling_avg)

    df_lineup = pd.concat(data_append)
    team_sums = df_lineup[sums_cols].sum().to_frame().T
    team_avgs = df_lineup[avgs_cols].mean().to_frame().T
    combined = pd.concat([team_sums, team_avgs], axis=1)
    
    return combined[all_cols]










