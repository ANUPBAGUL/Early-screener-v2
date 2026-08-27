import pandas as pd
import numpy as np
import sqlite3
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "screener.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def calculate_technical_features(df):
    """
    Computes technical features required for the multibagger screener.
    Expects a DataFrame with ['timestamp', 'close', 'volume', 'high', 'low']
    """
    df = df.sort_values('timestamp').copy()
    
    # 1. Volume Surge (Volume / 50-day average volume)
    df['vol_50d_avg'] = df['volume'].rolling(window=50).mean()
    df['volume_surge_score'] = df['volume'] / df['vol_50d_avg'].replace(0, np.nan)
    
    # 2. Volatility Contraction (VCP) - Standard Deviation of close over last 20 days as a % of price
    df['std_20d'] = df['close'].rolling(window=20).std()
    # Lower is tighter (better for VCP). We represent this as a % of the closing price.
    df['volatility_contraction_score'] = (df['std_20d'] / df['close']) * 100
    
    # 3. Momentum Score (Price vs 200 day moving average)
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df['momentum_score'] = ((df['close'] - df['ma_200']) / df['ma_200']) * 100
    
    return df

def run_feature_engineering():
    """
    Reads price_history from SQLite, calculates features, and saves them to technical_features table.
    """
    print("--- Running Feature Engineering ---")
    conn = get_connection()
    
    # Get all distinct instruments
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT instrument_key FROM price_history")
    instruments = [row[0] for row in cursor.fetchall()]
    
    for inst in instruments:
        df = pd.read_sql_query("SELECT * FROM price_history WHERE instrument_key = ?", conn, params=(inst,))
        if len(df) < 200:
            # Need at least 200 days for 200MA
            continue
            
        df = calculate_technical_features(df)
        
        # Drop NaN rows (first 200 days)
        df = df.dropna(subset=['volume_surge_score', 'volatility_contraction_score', 'momentum_score'])
        
        # Write back to technical_features
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO technical_features 
                (instrument_key, timestamp, volatility_contraction_score, volume_surge_score, momentum_score)
                VALUES (?, ?, ?, ?, ?)
            """, (
                row['instrument_key'], 
                row['timestamp'], 
                row['volatility_contraction_score'], 
                row['volume_surge_score'], 
                row['momentum_score']
            ))
            
    conn.commit()
    conn.close()
    print("Feature engineering complete!")

if __name__ == "__main__":
    run_feature_engineering()
