import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from upstox_client import UpstoxClient
import time
from datetime import datetime, timedelta

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path, override=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "screener.db")

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def fetch_top_stocks():
    """
    Downloads the master instrument list from Upstox and selects 700 equities.
    """
    print("Fetching master instrument list from Upstox...")
    url = 'https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz'
    df = pd.read_csv(url)
    
    # Filter for standard equities (not ETFs or options)
    df = df[(df['instrument_type'] == 'EQUITY') & (df['exchange'] == 'NSE_EQ')]
    
    # We'll select the top 700 by just slicing (Upstox usually orders them somewhat by relevance/alphabetically)
    # To be safer against penny stocks, we'll ensure price > 50
    df['last_price'] = pd.to_numeric(df['last_price'], errors='coerce')
    df = df[df['last_price'] > 50.0]
    
    top_700 = df.head(700)
    
    stocks = []
    for _, row in top_700.iterrows():
        stocks.append({
            "symbol": row['tradingsymbol'],
            "instrument_key": row['instrument_key']
        })
    return stocks

def initialize_database():
    print("Initializing Database...")
    conn = get_db_connection()
    # Execute the schema.sql
    schema_path = os.path.join(PROJECT_ROOT, "database", "schema.sql")
    if os.path.exists(schema_path):
        with open(schema_path, 'r') as f:
            conn.executescript(f.read())
    else:
        print(f"Warning: schema.sql not found at {schema_path}")
    conn.commit()
    conn.close()

def build_database():
    print("--- High-Growth Screener DB Builder ---")
    initialize_database()
    
    # 1. Init Upstox Client
    client = UpstoxClient()
    if not client.access_token:
        print("ERROR: UPSTOX_ACCESS_TOKEN is missing in .env")
        print("Please run upstox_client.py to get the login URL, authenticate, and save the token.")
        return
        
    stocks = fetch_top_stocks()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Global to_date is always today
    to_date = datetime.today().strftime('%Y-%m-%d')
    default_from_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    print(f"Checking updates for {len(stocks)} stocks up to {to_date}...")
    stats = {"stocks_updated": 0, "candles_added": 0}
    
    for idx, stock in enumerate(stocks):
        symbol = stock['symbol']
        inst_key = stock['instrument_key']
        
        # 1. Insert stock metadata
        cursor.execute("""
            INSERT OR IGNORE INTO stocks (instrument_key, symbol, exchange) 
            VALUES (?, ?, 'NSE')
        """, (inst_key, symbol))
        
        # 1.5 Determine from_date dynamically (Incremental Update)
        cursor.execute("SELECT MAX(timestamp) FROM price_history WHERE instrument_key = ?", (inst_key,))
        result = cursor.fetchone()
        
        if result and result[0]:
            # Extract the date part (e.g., '2024-08-25T00:00:00+05:30' -> '2024-08-25')
            last_date_str = result[0].split('T')[0]
            
            # If the last date in DB is today, skip fetching entirely
            if last_date_str == to_date:
                print(f"[{idx+1}/{len(stocks)}] {symbol} is already up to date. Skipping.")
                continue
                
            from_date = last_date_str
            print(f"[{idx+1}/{len(stocks)}] Fetching {symbol} from {from_date}...")
        else:
            from_date = default_from_date
            print(f"[{idx+1}/{len(stocks)}] Fetching new {symbol} (5 years)...")
        
        try:
            # 2. Fetch candles (Rate limit handling: sleep 0.2s = max 5 req/sec)
            candles = client.fetch_historical_candles(
                instrument_key=inst_key, 
                interval="day", 
                to_date=to_date, 
                from_date=from_date
            )
            
            # Upstox API returns None or empty if no new data
            if not candles:
                continue
            
            # Upstox returns data as: [timestamp, open, high, low, close, volume, oi]
            # timestamp is ISO format e.g., '2024-01-01T00:00:00+05:30'
            candles_added = 0
            for candle in candles:
                timestamp, open_p, high_p, low_p, close_p, volume, _ = candle
                
                cursor.execute("""
                    INSERT OR IGNORE INTO price_history 
                    (instrument_key, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (inst_key, timestamp, open_p, high_p, low_p, close_p, volume))
                
                # Check if it was actually inserted
                if cursor.rowcount > 0:
                    candles_added += 1
                    
            if candles_added > 0:
                stats["candles_added"] += candles_added
                stats["stocks_updated"] += 1
                
            conn.commit()
            time.sleep(0.2) # Rate limit protection
            
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
            
    conn.close()
    print("Database build complete!")
    return stats

if __name__ == "__main__":
    build_database()
