import pandas as pd
import numpy as np
import sqlite3
import os
import sys

# Centralized config integration
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

DB_PATH = config.DB_PATH

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
    if len(df) >= 200:
        df['ma_200'] = df['close'].rolling(window=200).mean()
        df['momentum_score'] = ((df['close'] - df['ma_200']) / df['ma_200']) * 100
    else:
        df['momentum_score'] = np.nan
        
    # 4. Stage 2 Trend Template Calculations
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_150'] = df['close'].rolling(window=150).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['high_52w'] = df['high'].rolling(window=250, min_periods=1).max()
    
    # Check 200 SMA trending up over 1 month (20 trading days)
    df['sma_200_1m_ago'] = df['sma_200'].shift(20)
    
    # Mark Minervini Trend Template conditions
    cond1 = df['close'] > df['sma_50']
    cond2 = df['close'] > df['sma_150']
    cond3 = df['close'] > df['sma_200']
    cond4 = df['sma_50'] > df['sma_150']
    cond5 = df['sma_150'] > df['sma_200']
    cond6 = df['sma_200'] > df['sma_200_1m_ago']
    cond7 = df['close'] >= (df['high_52w'] * 0.75)
    
    df['stage_2_flag'] = (cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7).astype(int)
    
    # 5. Pivot High — VCP Breakout Trigger Price
    # The max high over the last 15 trading days represents the tightest consolidation
    # handle peak. A stock trading above this level intraday (on volume) is the true
    # breakout trigger. This replaces the misleading "enter at yesterday's close" assumption.
    df['pivot_high'] = df['high'].rolling(window=15).max()
    
    # 6. Volume Profile Node Isolation (Rolling 30-Day Window)
    pocs = []
    densities = []
    
    closes = df['close'].values
    volumes = df['volume'].values
    highs = df['high'].values
    lows = df['low'].values
    
    n = len(df)
    for i in range(n):
        if i < 30:
            pocs.append(np.nan)
            densities.append(np.nan)
            continue
            
        c_slice = closes[i-29:i+1]
        v_slice = volumes[i-29:i+1]
        h_slice = highs[i-29:i+1]
        l_slice = lows[i-29:i+1]
        
        min_p = np.min(l_slice)
        max_p = np.max(h_slice)
        
        if max_p == min_p:
            pocs.append(min_p)
            densities.append(100.0)
            continue
            
        bins = np.linspace(min_p, max_p, 11)
        hist, bin_edges = np.histogram(c_slice, bins=bins, weights=v_slice)
        
        max_bin_idx = np.argmax(hist)
        poc_price = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx+1]) / 2.0
        
        tot_vol = np.sum(v_slice)
        if tot_vol > 0:
            mask = (c_slice >= poc_price * 0.97) & (c_slice <= poc_price * 1.03)
            density = (np.sum(v_slice[mask]) / tot_vol) * 100.0
        else:
            density = 0.0
            
        pocs.append(round(float(poc_price), 2))
        densities.append(round(float(density), 1))
        
    df['volume_node_poc'] = pocs
    df['volume_node_density'] = densities
    
    return df

def run_feature_engineering():
    """
    Reads price_history from SQLite, calculates features, and saves them to technical_features table.
    Calculates features incrementally and writes them in bulk.
    """
    print("--- Running Feature Engineering ---")
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get all distinct instruments
    cursor.execute("SELECT DISTINCT instrument_key FROM price_history")
    instruments = [row[0] for row in cursor.fetchall()]
    
    total_inserted = 0
    from datetime import datetime
    to_date = datetime.today().strftime('%Y-%m-%d')
    
    # Schema migration: add pivot_high, volume_node_poc, and volume_node_density if they don't exist
    try:
        cursor.execute("ALTER TABLE technical_features ADD COLUMN pivot_high REAL")
        conn.commit()
    except Exception:
        pass  # Column already exists
        
    try:
        cursor.execute("ALTER TABLE technical_features ADD COLUMN volume_node_poc REAL")
        cursor.execute("ALTER TABLE technical_features ADD COLUMN volume_node_density REAL")
        conn.commit()
    except Exception:
        pass  # Columns already exist
    
    for inst in instruments:
        # Delete today's recorded features first to prevent incomplete intraday calculation poisoning
        cursor.execute("DELETE FROM technical_features WHERE instrument_key = ? AND timestamp LIKE ?", (inst, to_date + "%"))
        
        # Check incremental checkpoint (only if sma_50 is already populated in DB)
        # If sma_50 is NULL for the max timestamp, we force recalculate for this stock
        cursor.execute("""
            SELECT MAX(timestamp) FROM technical_features 
            WHERE instrument_key = ? AND sma_50 IS NOT NULL
        """, (inst,))
        last_ts = cursor.fetchone()[0]
        
        # Load all history to calculate rolling indicators correctly
        df = pd.read_sql_query("SELECT * FROM price_history WHERE instrument_key = ? ORDER BY timestamp ASC", conn, params=(inst,))
        if len(df) < 50:
            # Need at least 50 days for volume surge
            continue
            
        df = calculate_technical_features(df)
        
        # Drop rows where base features (VCP and Volume Surge) are NaN
        df = df.dropna(subset=['volume_surge_score', 'volatility_contraction_score'])
        
        # Filter for incremental updates
        if last_ts:
            df = df[df['timestamp'] > last_ts]
            
        if df.empty:
            continue
            
        # Replace NaN values with None for SQLite insertion
        df = df.replace({np.nan: None})
        
        # Prepare data for bulk insert
        insert_data = [
            (
                row['instrument_key'], 
                row['timestamp'], 
                row['volatility_contraction_score'], 
                row['volume_surge_score'], 
                row['momentum_score'],
                row['sma_50'],
                row['sma_150'],
                row['sma_200'],
                row['high_52w'],
                row['stage_2_flag'],
                row.get('pivot_high'),
                row.get('volume_node_poc'),
                row.get('volume_node_density')
            )
            for _, row in df.iterrows()
        ]
        
        # Write back to technical_features in bulk
        cursor.executemany("""
            INSERT OR REPLACE INTO technical_features 
            (instrument_key, timestamp, volatility_contraction_score, volume_surge_score, momentum_score,
             sma_50, sma_150, sma_200, high_52w, stage_2_flag, pivot_high, volume_node_poc, volume_node_density)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, insert_data)
        
        total_inserted += len(insert_data)
        
    conn.commit()
    conn.close()
    print(f"Feature engineering complete! Inserted/updated {total_inserted} records.")

if __name__ == "__main__":
    run_feature_engineering()
