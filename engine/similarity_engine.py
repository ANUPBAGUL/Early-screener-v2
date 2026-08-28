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
    Queries the latest timestamp per individual stock, joining with fundamental data and 50-day avg volume.
    """
    conn = get_connection()
    query = """
    WITH avg_vol_50 AS (
        SELECT instrument_key, AVG(volume) as vol_50d_avg
        FROM (
            SELECT instrument_key, volume,
                   ROW_NUMBER() OVER (PARTITION BY instrument_key ORDER BY timestamp DESC) as rn
            FROM price_history
        )
        WHERE rn <= 50
        GROUP BY instrument_key
    )
    SELECT s.symbol, s.market_cap, s.debt_to_equity, s.revenue_growth, s.roce, s.peg_ratio, v.vol_50d_avg,
           t.volatility_contraction_score, t.volume_surge_score, t.momentum_score,
           t.sma_50, t.sma_150, t.sma_200, t.high_52w, t.stage_2_flag
    FROM technical_features t
    JOIN stocks s ON t.instrument_key = s.instrument_key
    JOIN avg_vol_50 v ON t.instrument_key = v.instrument_key
    WHERE (t.instrument_key, t.timestamp) IN (
        SELECT instrument_key, MAX(timestamp) 
        FROM technical_features 
        GROUP BY instrument_key
    )
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def calculate_similarity_score(features_df):
    """
    Calculates the true cosine similarity score (0-100) for all current stocks
    against the historical multibagger DNA library.
    Returns features_df with similarity_score, match_symbol, match_date, and match_return.
    """
    if features_df.empty:
        return features_df
        
    conn = get_connection()
    # Fetch historical DNA library
    dna_df = pd.read_sql_query("""
        SELECT s.symbol as dna_symbol, d.timestamp as dna_timestamp, 
               d.volatility_contraction_score, d.volume_surge_score, d.momentum_score,
               d.subsequent_return
        FROM multibagger_dna d
        JOIN stocks s ON d.instrument_key = s.instrument_key
    """, conn)
    conn.close()
    
    if dna_df.empty:
        # Fallback if DNA library is empty
        print("Warning: Multibagger DNA library is empty. Run breakout_model.py first.")
        features_df['similarity_score'] = 50
        features_df['match_symbol'] = "N/A"
        features_df['match_date'] = "N/A"
        features_df['match_return'] = 0.0
        return features_df
        
    cols = ['volatility_contraction_score', 'volume_surge_score', 'momentum_score']
    
    # Copy dataframes to avoid mutating inputs
    f_df = features_df.copy()
    d_df = dna_df.copy()
    
    # Fill NaNs with column means to support recent IPOs (momentum is NULL)
    for col in cols:
        mean_val = f_df[col].mean()
        if pd.isna(mean_val):
            mean_val = 0.0
        f_df[col] = f_df[col].fillna(mean_val)
        d_df[col] = d_df[col].fillna(mean_val)
        
    # Scale features using combined dataset z-score stats
    combined = pd.concat([f_df[cols], d_df[cols]], ignore_index=True)
    means = combined.mean()
    stds = combined.std().replace(0, 1.0)
    
    scaled_f = (f_df[cols] - means) / stds
    scaled_d = (d_df[cols] - means) / stds
    
    # Compute Cosine Similarity matrix: shape (N_features, M_dna)
    sim_matrix = cosine_similarity(scaled_f, scaled_d)
    
    scores = []
    match_symbols = []
    match_dates = []
    match_returns = []
    
    for i in range(len(features_df)):
        sim_row = sim_matrix[i]
        max_sim = np.max(sim_row)
        best_idx = np.argmax(sim_row)
        
        # Convert cosine similarity (-1 to 1) to percentage score (0-100)
        percentage_score = round(max(0.0, min(1.0, max_sim)) * 100)
        scores.append(percentage_score)
        
        best_match = dna_df.iloc[best_idx]
        match_symbols.append(best_match['dna_symbol'])
        # Strip timestamp timezone for clean display
        match_dates.append(best_match['dna_timestamp'].split('T')[0])
        match_returns.append(round(best_match['subsequent_return'] * 100, 1))
        
    features_df['similarity_score'] = scores
    features_df['match_symbol'] = match_symbols
    features_df['match_date'] = match_dates
    features_df['match_return'] = match_returns
    
    return features_df

def mock_similarity_score(features_df, user_vol_surge, user_vcp_tightness):
    """
    Wrapper mapping the old mock signature to the new genuine similarity score engine.
    """
    return calculate_similarity_score(features_df)

if __name__ == "__main__":
    print("Testing Similarity Engine...")
    df = fetch_latest_features()
    if not df.empty:
        scored_df = calculate_similarity_score(df)
        print(scored_df.head())
    else:
        print("No data found in database. Run build_db.py and feature_engineer.py first.")

