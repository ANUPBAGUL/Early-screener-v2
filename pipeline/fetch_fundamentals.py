import sqlite3
import os
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "screener.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def fetch_single_stock(symbol, inst_key):
    ticker_sym = f"{symbol}.NS"
    try:
        ticker = yf.Ticker(ticker_sym)
        info = ticker.info
        
        # 1. Market Cap (in Crores = divided by 10,000,000)
        market_cap_raw = info.get("marketCap", 0)
        market_cap_crores = market_cap_raw / 10000000.0 if market_cap_raw else 0.0
        
        # Helper for Net Worth / Total Equity
        book_value = info.get("bookValue")
        shares = info.get("sharesOutstanding")
        net_worth = book_value * shares if (book_value and shares) else None
        
        # 2. Debt to Equity ratio
        debt_to_equity = info.get("debtToEquity", None)
        if debt_to_equity is not None:
            debt_to_equity = debt_to_equity / 100.0  # yfinance returns e.g. 50.0 for 0.5 ratio
        else:
            # Try calculating dynamically from totalDebt and Net Worth
            total_debt = info.get("totalDebt")
            if total_debt and net_worth and net_worth > 0:
                debt_to_equity = total_debt / net_worth
            else:
                debt_to_equity = 0.0
            
        # 3. Revenue Growth (quarterly or yearly)
        revenue_growth = info.get("revenueGrowth", None)
        if revenue_growth is None:
            revenue_growth = 0.0
            
        # 4. Return on Capital Employed (ROCE) / returnOnEquity (ROE)
        roce = info.get("returnOnEquity", None)
        if roce is None:
            # Try calculating dynamically from Net Income and Net Worth
            net_income = info.get("netIncomeToCommon")
            if net_income and net_worth and net_worth > 0:
                roce = net_income / net_worth
            
            # Fallback to operating margins (EBIT margin) as a proxy if still None
            if roce is None:
                op_margin = info.get("operatingMargins", None)
                if op_margin is not None:
                    roce = op_margin * 1.5
                else:
                    roce = 0.0
                
        # 5. Price-to-Earnings-to-Growth (PEG) Ratio
        peg_ratio = info.get("trailingPegRatio", None)
        if peg_ratio is None:
            peg_ratio = info.get("pegRatio", None)
        if peg_ratio is None:
            # Try calculating dynamically: PE / (earningsGrowth * 100)
            pe = info.get("trailingPE")
            if pe is None:
                pe = info.get("forwardPE")
            
            eg = info.get("earningsGrowth")
            if eg is None:
                eg = info.get("revenueGrowth")
                
            if pe and eg and eg > 0:
                peg_ratio = pe / (eg * 100.0)
            else:
                peg_ratio = 0.0
                
        return {
            "instrument_key": inst_key,
            "symbol": symbol,
            "market_cap": market_cap_crores,
            "debt_to_equity": debt_to_equity,
            "revenue_growth": revenue_growth,
            "roce": roce,
            "peg_ratio": peg_ratio,
            "status": "success"
        }
    except Exception as e:
        return {
            "instrument_key": inst_key,
            "symbol": symbol,
            "status": "failed",
            "error": str(e)
        }

def update_fundamentals():
    print("--- Fetching Fundamental Data (including PEG Ratio) ---")
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get all stocks
    cursor.execute("SELECT symbol, instrument_key FROM stocks")
    stocks = cursor.fetchall()
    conn.close()
    
    print(f"Queueing {len(stocks)} stocks for fundamental fetching...")
    
    results = []
    # Fetch using 10 threads to avoid rate limits while maintaining speed
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_single_stock, sym, key): sym for sym, key in stocks}
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % 50 == 0 or completed == len(stocks):
                print(f"Progress: {completed}/{len(stocks)} stocks fetched.")
                
    # Update SQLite database
    conn = get_connection()
    cursor = conn.cursor()
    
    updated_count = 0
    for res in results:
        if res["status"] == "success":
            cursor.execute("""
                UPDATE stocks 
                SET market_cap = ?, debt_to_equity = ?, revenue_growth = ?, roce = ?, peg_ratio = ?
                WHERE instrument_key = ?
            """, (
                res["market_cap"],
                res["debt_to_equity"],
                res["revenue_growth"],
                res["roce"],
                res["peg_ratio"],
                res["instrument_key"]
            ))
            updated_count += 1
            
    conn.commit()
    conn.close()
    print(f"Successfully updated fundamentals (with PEG) for {updated_count} stocks!")

if __name__ == "__main__":
    update_fundamentals()
