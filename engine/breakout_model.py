import pandas as pd
import numpy as np
import sqlite3
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from datetime import datetime, timedelta
import sys

# Centralized config integration
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

DB_PATH = config.DB_PATH
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "breakout_model.json")

def get_connection():
    return sqlite3.connect(DB_PATH)

def prepare_data():
    """
    Loads daily prices and features, labels breakouts (>= 50% gain in next 20 trading days),
    and returns a training dataset.
    """
    print("Loading data from database...")
    conn = get_connection()
    
    # Fetch price history for labeling (including high and low for path-dependent simulation)
    prices_df = pd.read_sql_query("""
        SELECT instrument_key, timestamp, open, high, low, close 
        FROM price_history 
        ORDER BY instrument_key, timestamp ASC
    """, conn)
    
    # Fetch features
    features_df = pd.read_sql_query("""
        SELECT instrument_key, timestamp, volatility_contraction_score, volume_surge_score, momentum_score 
        FROM technical_features
    """, conn)
    
    # Fetch fundamental snapshot (taken at model-training time, not at each historical date)
    # This eliminates the look-ahead bias of joining current-day fundamentals at query time.
    fundamentals_df = pd.read_sql_query("""
        SELECT instrument_key, debt_to_equity, price_to_book, roce
        FROM stocks
    """, conn)
    
    conn.close()
    
    print("Labeling historical breakouts...")
    # Calculate subsequent 20-day returns for each stock using path-dependent simulation
    # (Checking if TP is hit before SL is hit in subsequent 20 trading days)
    labeled_dfs = []
    for inst, group in prices_df.groupby('instrument_key'):
        group = group.sort_values('timestamp').copy()
        
        close_arr = group['close'].values
        high_arr = group['high'].values
        low_arr = group['low'].values
        n = len(group)
        
        labels = np.zeros(n, dtype=int)
        subsequent_returns = np.zeros(n, dtype=float)
        days_to_target = np.zeros(n, dtype=int)
        
        for i in range(n):
            if i + 20 >= n:
                labels[i] = 0
                subsequent_returns[i] = 0.0
                days_to_target[i] = 0
                continue
                
            close_val = close_arr[i]
            sl_price = close_val * (1.0 - 0.075)
            tp_price = close_val * (1.0 + config.BREAKOUT_LABEL_THRESHOLD)
            
            # Check subsequent 20 trading days path
            for j in range(i + 1, i + 21):
                day_low = low_arr[j]
                day_high = high_arr[j]
                
                # Check stop loss first (pessimistic)
                if day_low <= sl_price:
                    labels[i] = 0
                    subsequent_returns[i] = -0.075
                    days_to_target[i] = j - i
                    break
                elif day_high >= tp_price:
                    labels[i] = 1
                    subsequent_returns[i] = config.BREAKOUT_LABEL_THRESHOLD
                    days_to_target[i] = j - i
                    break
            else:
                # Time exit on day 20 close
                day_20_close = close_arr[i + 20]
                ret_val = (day_20_close - close_val) / close_val
                subsequent_returns[i] = ret_val
                labels[i] = 1 if ret_val >= config.BREAKOUT_LABEL_THRESHOLD else 0
                days_to_target[i] = 20
                
        group['label'] = labels
        group['subsequent_return'] = subsequent_returns
        group['days_to_target'] = days_to_target
        
        labeled_dfs.append(group[['instrument_key', 'timestamp', 'subsequent_return', 'label', 'days_to_target']])
        
    labeled_df = pd.concat(labeled_dfs, ignore_index=True)
    
    # Merge features with labels, then attach fundamental snapshot
    data = pd.merge(features_df, labeled_df, on=['instrument_key', 'timestamp'], how='inner')
    data = pd.merge(data, fundamentals_df, on='instrument_key', how='left')
    
    return data

def train_and_save_model():
    """
    Trains the XGBoost model and populates the multibagger_dna table in the database.
    """
    print("--- Training XGBoost Breakout Classifier ---")
    data = prepare_data()
    
    # Drop rows where base features are null
    data = data.dropna(subset=['volatility_contraction_score', 'volume_surge_score'])
    
    # Drop in-flight setups (unevaluated in the last 20 days of the dataset)
    if 'days_to_target' in data.columns:
        data = data[data['days_to_target'] > 0]
    
    # Separate features and labels
    feature_cols = ['volatility_contraction_score', 'volume_surge_score', 'momentum_score']
    X = data[feature_cols]
    y = data['label']
    
    print(f"Total samples: {len(data)}")
    print(f"Breakout samples (Class 1): {y.sum()} ({y.sum()/len(data)*100:.2f}%)")
    print(f"Non-breakout samples (Class 0): {len(data) - y.sum()}")
    
    if y.sum() == 0:
        print("ERROR: No historical breakouts found with >= 50% gain in 20 days. Cannot train model.")
        return
        
    # Split temporally to prevent leakage (Train: before 6 months ago, Test: last 6 months)
    split_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    print(f"Using dynamic temporal split date (6 months ago): {split_date}")
    train_mask = data['timestamp'] < split_date
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    
    if len(X_test) == 0 or y_test.sum() == 0:
        # Fallback to standard split if temporal split is too small
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
    # Calculate scale_pos_weight to handle heavy class imbalance
    ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"Class imbalance ratio (0/1): {ratio:.2f}")
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=ratio,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    print("Fitting model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\nModel Evaluation (Test Set):")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    # Save the model
    print(f"Saving model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH)
    
    # Populate multibagger_dna table with true historical breakouts
    populate_dna_library(data)

def populate_dna_library(data):
    """
    Saves true breakouts (subsequent_return >= 50%) to the multibagger_dna library.
    Snapshots fundamental values (D/E, P/B, ROCE) at model-training time so the
    6D similarity engine does not use stale current-day fundamentals for historical events.
    """
    print("\nPopulating Multibagger DNA Library...")
    # Filter for true breakout setups
    breakouts = data[data['label'] == 1].copy()
    
    # Replace NaNs with None for SQLite
    breakouts = breakouts.replace({np.nan: None})
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # --- Schema migration: add fundamental & days_to_target columns if they don't exist ---
    for col_def in ["debt_to_equity REAL", "price_to_book REAL", "roce REAL", "days_to_target INTEGER"]:
        try:
            cursor.execute(f"ALTER TABLE multibagger_dna ADD COLUMN {col_def}")
        except Exception:
            pass  # Column already exists — safe to ignore
    
    # Clean old library
    cursor.execute("DELETE FROM multibagger_dna")
    
    insert_data = [
        (
            row['instrument_key'],
            row['timestamp'],
            row['volatility_contraction_score'],
            row['volume_surge_score'],
            row['momentum_score'],
            row['subsequent_return'],
            row.get('debt_to_equity'),
            row.get('price_to_book'),
            row.get('roce'),
            int(row['days_to_target']) if 'days_to_target' in row and row['days_to_target'] is not None else 20
        )
        for _, row in breakouts.iterrows()
    ]
    
    cursor.executemany("""
        INSERT OR REPLACE INTO multibagger_dna
        (instrument_key, timestamp, volatility_contraction_score, volume_surge_score, momentum_score,
         subsequent_return, debt_to_equity, price_to_book, roce, days_to_target)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, insert_data)
    
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM multibagger_dna")
    dna_count = cursor.fetchone()[0]
    conn.close()
    
    print(f"Multibagger DNA Library populated with {dna_count} historical breakout states!")

if __name__ == "__main__":
    train_and_save_model()
