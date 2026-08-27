import pandas as pd
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "screener.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def fetch_latest_features():
    """
    Fetches the latest technical features for all stocks to use as the base for the screener.
    """
    conn = get_connection()
    query = """
    SELECT s.symbol, t.volatility_contraction_score, t.volume_surge_score, t.momentum_score 
    FROM technical_features t
    JOIN stocks s ON t.instrument_key = s.instrument_key
    WHERE t.timestamp = (SELECT MAX(timestamp) FROM technical_features)
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def mock_similarity_score(features_df, user_vol_surge, user_vcp_tightness):
    """
    Calculates a similarity score (0-100) based on how close the stock's current features 
    are to the user's targeted "ideal breakout DNA" parameters.
    """
    scores = []
    for _, row in features_df.iterrows():
        # Penalize if VCP is wider than target (we want tight VCP)
        vcp_penalty = max(0, row['volatility_contraction_score'] - user_vcp_tightness) * 5
        
        # Reward if volume surge is higher than target
        vol_bonus = max(0, row['volume_surge_score'] - user_vol_surge) * 10
        
        # Base score starts high, degrades based on distance from ideal momentum/VCP
        base_score = 90
        final_score = base_score - vcp_penalty + vol_bonus
        
        # Cap between 0 and 99
        final_score = max(0, min(99, final_score))
        scores.append(round(final_score))
        
    features_df['similarity_score'] = scores
    return features_df

if __name__ == "__main__":
    print("Testing Similarity Engine...")
    df = fetch_latest_features()
    if not df.empty:
        scored_df = mock_similarity_score(df, 3.0, 5.0)
        print(scored_df.head())
    else:
        print("No data found in database. Run build_db.py and feature_engineer.py first.")
