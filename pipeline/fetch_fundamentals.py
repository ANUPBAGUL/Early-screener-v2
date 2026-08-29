import sqlite3
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys
from datetime import date as _date

# Centralized config integration
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

DB_PATH = config.DB_PATH

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def get_connection():
    return sqlite3.connect(DB_PATH)

def safe_float(val, default=0.0):
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        clean = re.sub(r'[^\d.-]', '', str(val))
        return float(clean) if clean else default
    except Exception:
        return default

def fetch_single_stock_from_screener(symbol, inst_key, name=None):
    """
    Fetches accurate standalone/consolidated fundamentals directly from Screener.in.
    Fallback to yfinance if Screener.in fails.
    """
    clean_sym = symbol.replace('.NS', '').replace('.BO', '').upper()
    urls = [
        f"https://www.screener.in/company/{clean_sym}/consolidated/",
        f"https://www.screener.in/company/{clean_sym}/"
    ]
    
    soup = None
    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=6)
            if resp.status_code == 200 and "Search companies" not in resp.text[:400]:
                soup = BeautifulSoup(resp.text, 'html.parser')
                break
        except Exception:
            pass
            
    # Search fallback by company name if direct symbol slug fails
    if not soup and name:
        clean_name = re.sub(r'(?i)\b(ltd|limited|inc|corp|corporation|pvt)\b', '', name).strip()
        try:
            s_resp = requests.get(f"https://www.screener.in/api/company/search/?q={requests.utils.quote(clean_name)}", headers=HEADERS, timeout=5)
            if s_resp.status_code == 200 and s_resp.json():
                target_url = "https://www.screener.in" + s_resp.json()[0]['url']
                resp = requests.get(target_url, headers=HEADERS, timeout=6)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
        except Exception:
            pass
            
    if not soup:
        # Secondary fallback to yfinance if Screener is unreachable for this stock
        return fetch_single_stock_yfinance(symbol, inst_key)

    # 1. Parse top-ratios
    top_ratios = {}
    top_ul = soup.find('ul', id='top-ratios')
    if top_ul:
        for li in top_ul.find_all('li'):
            n_el = li.find('span', class_='name')
            v_el = li.find('span', class_='number')
            if n_el and v_el:
                key = n_el.text.strip().lower()
                val_raw = v_el.text.strip().replace(',', '').replace('₹', '').replace('%', '').strip()
                top_ratios[key] = val_raw

    mcap = safe_float(top_ratios.get('market cap'), 0.0)
    pe = safe_float(top_ratios.get('stock p/e'), 0.0)
    price = safe_float(top_ratios.get('current price'), 0.0)
    book_val = safe_float(top_ratios.get('book value'), 0.0)
    pb = (price / book_val) if (price > 0 and book_val > 0) else 0.0
    
    roce_raw = top_ratios.get('roce')
    roe_raw = top_ratios.get('roe')
    
    if roce_raw is not None:
        roce_val = safe_float(roce_raw) / 100.0
    elif roe_raw is not None:
        roce_val = safe_float(roe_raw) / 100.0
    else:
        roce_val = 0.0
    
    # 2. Debt to Equity
    de_raw = top_ratios.get('debt to equity')
    if de_raw is not None:
        de_val = safe_float(de_raw, 0.0)
    else:
        bs_section = soup.find('section', id='balance-sheet')
        if bs_section:
            borrowings = 0.0
            equity_cap = 0.0
            reserves = 0.0
            for r in bs_section.find_all('tr'):
                header_td = r.find('td') or r.find('th')
                if not header_td: continue
                h_text = header_td.text.strip().lower()
                vals = []
                for td in r.find_all('td')[1:]:
                    v_clean = td.text.strip().replace('%', '').replace(',', '').strip()
                    if v_clean not in ['', '+', '-']:
                        try: vals.append(float(v_clean))
                        except ValueError: pass
                if not vals: continue
                if 'borrowings' in h_text:
                    borrowings = vals[-1]
                elif 'equity capital' in h_text:
                    equity_cap = vals[-1]
                elif 'reserves' in h_text:
                    reserves = vals[-1]
            total_equity = equity_cap + reserves
            de_val = (borrowings / total_equity) if total_equity > 0 else 0.0
        else:
            de_val = 0.0

    # 3. YoY Revenue Growth & Profit Growth (from Quarterly Table)
    rev_growth = 0.0
    eg_growth = 0.0
    icr = 999.0
    
    q_section = soup.find('section', id='quarters')
    if q_section:
        op = None
        interest = None
        for r in q_section.find_all('tr'):
            header_td = r.find('td') or r.find('th')
            if not header_td: continue
            h_text = header_td.text.strip().lower()
            
            vals_num = []
            for td in r.find_all('td')[1:]:
                v_clean = td.text.strip().replace('%', '').replace(',', '').strip()
                if v_clean not in ['', '+', '-']:
                    try: vals_num.append(float(v_clean))
                    except ValueError: pass
            
            if 'sales' in h_text or 'revenue' in h_text:
                if len(vals_num) >= 5:
                    latest = vals_num[-1]
                    last_year = vals_num[-5]
                    if last_year > 0:
                        rev_growth = (latest - last_year) / last_year
                        
            if 'net profit' in h_text:
                if len(vals_num) >= 5:
                    latest = vals_num[-1]
                    last_year = vals_num[-5]
                    if last_year > 0:
                        eg_growth = (latest - last_year) / last_year
                        
            if 'operating profit' in h_text and vals_num:
                op = vals_num[-1]
            if 'interest' in h_text and vals_num:
                interest = vals_num[-1]
                
        if op is not None and interest is not None and interest > 0:
            icr = op / interest

    # 4. Sector
    sector = "N/A"
    crumbs = soup.find_all('a', href=re.compile(r'/company/group/|/market/'))
    if crumbs:
        sector = crumbs[-1].text.strip()

    # 5. Dynamic PEG ratio
    peg_ratio = 0.0
    if pe > 0 and eg_growth > 0:
        peg_ratio = pe / (eg_growth * 100.0)

    # 6. Promoter Shareholding & Cash Flow
    promoter_holding = 0.0
    sh_section = soup.find('section', id='shareholding')
    if sh_section:
        for r in sh_section.find_all('tr'):
            td_el = r.find('td')
            if not td_el: continue
            title = td_el.text.strip().lower()
            if 'promoter' in title:
                vals = [safe_float(td.text.strip().replace('%', '')) for td in r.find_all('td')[1:] if td.text.strip().replace('%', '') not in ['', '+', '-']]
                if vals:
                    promoter_holding = vals[-1]
                    
    operating_cash_flow = 0.0
    cf_section = soup.find('section', id='cash-flow')
    if cf_section:
        for r in cf_section.find_all('tr'):
            td_el = r.find('td')
            if not td_el: continue
            title = td_el.text.strip().lower()
            if 'cash from operating activity' in title:
                vals = [safe_float(td.text.strip().replace(',', '')) for td in r.find_all('td')[1:] if td.text.strip().replace(',', '') not in ['', '+', '-']]
                if vals:
                    operating_cash_flow = vals[-1]

    # 7. Data Quality Score
    key_fields = [roce_val, de_val, rev_growth, eg_growth, icr, pe, pb]
    populated = sum(1 for v in key_fields if v is not None and v != 0.0)
    data_quality = 3 if populated >= 6 else (2 if populated >= 4 else (1 if populated >= 2 else 0))

    fundamentals_updated_at = str(_date.today())

    return {
        "instrument_key": inst_key,
        "symbol": symbol,
        "market_cap": round(mcap, 1),
        "debt_to_equity": round(de_val, 2),
        "revenue_growth": round(rev_growth, 4),
        "roce": round(roce_val, 4),
        "peg_ratio": round(peg_ratio, 2),
        "pe_ratio": round(pe, 2),
        "price_to_book": round(pb, 2),
        "earnings_growth": round(eg_growth, 4),
        "interest_coverage": round(icr, 1),
        "sector": sector,
        "days_to_earnings": None,
        "fundamentals_updated_at": fundamentals_updated_at,
        "data_quality": data_quality,
        "promoter_holding": round(promoter_holding, 2),
        "operating_cash_flow": round(operating_cash_flow, 1),
        "status": "success"
    }

def fetch_single_stock_yfinance(symbol, inst_key):
    """
    Secondary fallback via yfinance if Screener.in page is unavailable.
    """
    import yfinance as yf
    ticker_sym = f"{symbol}.NS"
    try:
        ticker = yf.Ticker(ticker_sym)
        info = ticker.info or {}
        market_cap_raw = info.get("marketCap", 0)
        market_cap_crores = market_cap_raw / 10000000.0 if market_cap_raw else 0.0
        
        debt_to_equity = info.get("debtToEquity", 0.0)
        if debt_to_equity: debt_to_equity /= 100.0
        
        roce = info.get("returnOnEquity", 0.0)
        pe_ratio = info.get("trailingPE", 0.0) or 0.0
        price_to_book = info.get("priceToBook", 0.0) or 0.0
        revenue_growth = info.get("revenueGrowth", 0.0) or 0.0
        earnings_growth = info.get("earningsGrowth", 0.0) or 0.0
        sector = info.get("sector", "N/A")
        
        held_insiders = info.get("heldPercentInsiders", 0.0) or 0.0
        promoter_holding = held_insiders * 100.0
        
        cfo_raw = info.get("operatingCashflows", 0.0) or 0.0
        operating_cash_flow = cfo_raw / 10000000.0
        
        return {
            "instrument_key": inst_key,
            "symbol": symbol,
            "market_cap": round(market_cap_crores, 1),
            "debt_to_equity": round(debt_to_equity, 2),
            "revenue_growth": round(revenue_growth, 4),
            "roce": round(roce, 4),
            "peg_ratio": 0.0,
            "pe_ratio": round(pe_ratio, 2),
            "price_to_book": round(price_to_book, 2),
            "earnings_growth": round(earnings_growth, 4),
            "interest_coverage": 5.0,
            "sector": sector,
            "days_to_earnings": None,
            "fundamentals_updated_at": str(_date.today()),
            "data_quality": 1,
            "promoter_holding": round(promoter_holding, 2),
            "operating_cash_flow": round(operating_cash_flow, 1),
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
    print("--- Fetching Fundamental Data directly from Screener.in (Clean Indian Financials) ---")
    conn = get_connection()
    cursor = conn.cursor()
    
    # Migration guards
    for col_def in [
        "days_to_earnings INTEGER",
        "fundamentals_updated_at TEXT",
        "data_quality INTEGER DEFAULT 0",
        "promoter_holding REAL",
        "operating_cash_flow REAL"
    ]:
        try:
            cursor.execute(f"ALTER TABLE stocks ADD COLUMN {col_def}")
        except Exception:
            pass
    conn.commit()
    
    cursor.execute("SELECT symbol, instrument_key, name FROM stocks")
    stocks = cursor.fetchall()
    conn.close()
    
    print(f"Queueing {len(stocks)} stocks for Screener.in fundamental extraction...")
    
    db_conn = get_connection()
    db_cursor = db_conn.cursor()
    
    results = []
    # Fetch using 10 concurrent workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_single_stock_from_screener, sym, key, name): (sym, key) 
            for sym, key, name in stocks
        }
        
        completed = 0
        updated_count = 0
        
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            
            if res and res.get("status") == "success":
                try:
                    db_cursor.execute("""
                        UPDATE stocks 
                        SET market_cap = ?, debt_to_equity = ?, revenue_growth = ?, roce = ?, peg_ratio = ?,
                            pe_ratio = ?, price_to_book = ?, earnings_growth = ?, interest_coverage = ?,
                            sector = ?, days_to_earnings = ?, fundamentals_updated_at = ?, data_quality = ?,
                            promoter_holding = ?, operating_cash_flow = ?
                        WHERE instrument_key = ?
                    """, (
                        res["market_cap"],
                        res["debt_to_equity"],
                        res["revenue_growth"],
                        res["roce"],
                        res["peg_ratio"],
                        res["pe_ratio"],
                        res["price_to_book"],
                        res["earnings_growth"],
                        res["interest_coverage"],
                        res["sector"],
                        res.get("days_to_earnings"),
                        res.get("fundamentals_updated_at"),
                        res.get("data_quality", 0),
                        res.get("promoter_holding", 0.0),
                        res.get("operating_cash_flow", 0.0),
                        res["instrument_key"]
                    ))
                    updated_count += 1
                except Exception as e:
                    print(f"Error saving {res.get('symbol')}: {e}")
            
            # Commit incrementally every 25 stocks
            if completed % 25 == 0 or completed == len(stocks):
                db_conn.commit()
                print(f"Progress: {completed}/{len(stocks)} stocks fetched and committed ({updated_count} updated).")
                
    db_conn.close()
    print(f"Successfully updated fundamentals from Screener.in for {updated_count} stocks!")

if __name__ == "__main__":
    update_fundamentals()
