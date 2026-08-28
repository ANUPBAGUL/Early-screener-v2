import pandas as pd
import numpy as np
import sqlite3
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "screener.db")
MODEL_PATH = os.path.join(PROJECT_ROOT, "engine", "breakout_model.json")

def get_connection():
    return sqlite3.connect(DB_PATH)

def prepare_data():
    """
    Loads daily prices and features, labels breakouts (>= 50% gain in next 20 trading days),
    and returns a training dataset.
    """
    print("Loading data from database...")
    conn = get_connection()
    
    # Fetch price history for labeling
    prices_df = pd.read_sql_query("""
        SELECT instrument_key, timestamp, close 
        FROM price_history 
        ORDER BY instrument_key, timestamp ASC
    """, conn)
    
    # Fetch features
    features_df = pd.read_sql_query("""
        SELECT instrument_key, timestamp, volatility_contraction_score, volume_surge_score, momentum_score 
        FROM technical_features
    """, conn)
    
    conn.close()
    
    print("Labeling historical breakouts...")
    # Calculate subsequent 20-day returns for each stock
    labeled_dfs = []
    for inst, group in prices_df.groupby('instrument_key'):
        group = group.sort_values('timestamp').copy()
        
        # Max close in the subsequent 20 trading days (excluding current day)
        group['future_max_close'] = group['close'].shift(-1).iloc[::-1].rolling(window=20, min_periods=1).max().iloc[::-1]
        
        # Calculate subsequent return
        group['subsequent_return'] = (group['future_max_close'] - group['close']) / group['close']
        
        # Label: 1 if return >= 50% in subsequent 20 days, else 0
        group['label'] = (group['subsequent_return'] >= 0.50).astype(int)
        
        labeled_dfs.append(group[['instrument_key', 'timestamp', 'subsequent_return', 'label']])
        
    labeled_df = pd.concat(labeled_dfs, ignore_index=True)
    
    # Merge features with labels
    data = pd.merge(features_df, labeled_df, on=['instrument_key', 'timestamp'], how='inner')
    
    return data

def train_and_save_model():
    """
    Trains the XGBoost model and populates the multibagger_dna table in the database.
    """
    print("--- Training XGBoost Breakout Classifier ---")
    data = prepare_data()
    
    # Drop rows where base features are null
    data = data.dropna(subset=['volatility_contraction_score', 'volume_surge_score'])
    
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
        
    # Split temporally to prevent leakage (Train: before 2026, Test: 2026 onwards)
    train_mask = data['timestamp'] < '2026-01-01'
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
    """
    print("\nPopulating Multibagger DNA Library...")
    # Filter for true breakout setups
    breakouts = data[data['label'] == 1].copy()
    
    # Replace NaNs with None for SQLite
    breakouts = breakouts.replace({np.nan: None})
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Clean old library
    cursor.execute("DELETE FROM multibagger_dna")
    
    insert_data = [
        (
            row['instrument_key'],
            row['timestamp'],
            row['volatility_contraction_score'],
            row['volume_surge_score'],
            row['momentum_score'],
            row['subsequent_return']
        )
        for _, row in breakouts.iterrows()
    ]
    
    cursor.executemany("""
        INSERT OR REPLACE INTO multibagger_dna
        (instrument_key, timestamp, volatility_contraction_score, volume_surge_score, momentum_score, subsequent_return)
        VALUES (?, ?, ?, ?, ?, ?)
    """, insert_data)
    
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM multibagger_dna")
    dna_count = cursor.fetchone()[0]
    conn.close()
    
    print(f"Multibagger DNA Library populated with {dna_count} historical breakout states!")

if __name__ == "__main__":
    train_and_save_model()
