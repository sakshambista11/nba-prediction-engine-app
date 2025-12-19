import streamlit as st
import pandas as pd
from utils import load_nba_model, get_player_id, get_lineup_stats
from nba_api.stats.static import players

st.set_page_config(page_title="5v5 Simulator", layout="wide")
model = load_nba_model()

st.title("5v5 Custom Team Simulator")
st.write("Create a custom matchup using active players from the 2024-25 season.")

active_players = players.get_active_players()
player_names = sorted([p['full_name'] for p in active_players])

col1, col2 = st.columns(2)

with col1:
    st.header("🏠 Home Team Lineup")
    h_names = [st.selectbox(f"Home Player {i+1}", player_names, key=f"h{i}") for i in range(5)]

with col2:
    st.header("✈️ Away Team Lineup")
    a_names = [st.selectbox(f"Away Player {i+1}", player_names, key=f"a{i}") for i in range(5)]

if st.button("🚀 Run Simulation"):
    with st.spinner("Analyzing player chemistry and recent form..."):
        try:
            # 1. Get IDs for everyone
            h_ids = [get_player_id(name) for name in h_names]
            a_ids = [get_player_id(name) for name in a_names]
            
            # 2. Get Aggregated Stats
            home_team_stats = get_lineup_stats(h_ids)
            away_team_stats = get_lineup_stats(a_ids)
            
            # 3. Format for the Model (Match suffixes used in training)
            # The model expects [Home Features]_home then [Away Features]_away
            home_final = home_team_stats.add_suffix('_home')
            away_final = away_team_stats.add_suffix('_away')
            
            matchup_df = pd.concat([home_final, away_final], axis=1)
    
            # 3. FORCE CORRECT ORDER
            # This grabs the exact order the model was trained on
            train_feature_order = model.get_booster().feature_names
            matchup_df = matchup_df[train_feature_order]
    
            # 4. Predict
            prob = model.predict_proba(matchup_df)[0][1]

            st.divider()
            if prob > 0.5:
                st.success(f"🏆 **Home Lineup** is projected to win with **{prob:.1%}** confidence!")
            else:
                st.error(f"🏆 **Away Lineup** is projected to win with **{1-prob:.1%}** confidence!")
                
        except Exception as e:
            st.error(f"Error fetching data: {e}. Some players might not have 2024-25 game data yet.")