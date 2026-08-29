import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys

# Ensure local directories are in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

DB_PATH = config.DB_PATH
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "breakout_model.json")

def calculate_fundamental_score(row):
    score = 0
    # 1. Solvency (Max 2 pts)
    de = row.get('debt_to_equity')
    if de is not None and not pd.isna(de) and de <= 0.5:
        score += 1
    icr = row.get('interest_coverage')
    if icr is not None and not pd.isna(icr) and icr >= 4.0:
        score += 1
        
    # 2. Profit Inflection (Max 2 pts)
    eg = row.get('earnings_growth')
    if eg is not None and not pd.isna(eg):
        if eg >= 1.0:
            score += 2
        elif eg >= 0.25:
            score += 1
            
    # 3. Capital Efficiency (Max 2 pts)
    roce = row.get('roce')
    if roce is not None and not pd.isna(roce):
        if roce >= 0.20:
            score += 2
        elif roce >= 0.15:
            score += 1
            
    # 4. Revenue Growth (Max 2 pts)
    rg = row.get('revenue_growth')
    if rg is not None and not pd.isna(rg):
        if rg >= 0.15:
            score += 2
        elif rg >= 0.10:
            score += 1
            
    # 5. Valuation Safety (Max 2 pts)
    pb = row.get('price_to_book')
    if pb is not None and not pd.isna(pb) and pb > 0.0 and pb <= 3.0:
        score += 1
    pe = row.get('pe_ratio')
    sector_avg = row.get('sector_avg_pe')
    if pe is not None and not pd.isna(pe) and sector_avg is not None and not pd.isna(sector_avg) and sector_avg > 0:
        if pe <= sector_avg:
            score += 1
            
    return score

def run_backtest_ledger(confluence_threshold=85.0, stop_loss_pct=7.5, target_profit_pct=15.0, include_fundamentals=False):
    """
    Runs the historical backtest over technical setups.
    Returns: (metrics_dict, trades_df, archetype_stats_df)
    """
    if not os.path.exists(DB_PATH) or not os.path.exists(MODEL_PATH):
        return {}, pd.DataFrame(), pd.DataFrame()
        
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Fetch setups from technical_features
    query = """
    SELECT t.instrument_key, t.timestamp, s.symbol, s.sector,
           t.volatility_contraction_score, t.volume_surge_score, t.momentum_score,
           s.roce, s.debt_to_equity, s.peg_ratio, s.pe_ratio, s.price_to_book, s.earnings_growth, s.interest_coverage,
           s.revenue_growth, t.stage_2_flag
    FROM technical_features t
    JOIN stocks s ON t.instrument_key = s.instrument_key
    WHERE t.volume_surge_score >= 1.5 AND t.volatility_contraction_score <= 10.0
    ORDER BY t.timestamp ASC
    """
    setups_df = pd.read_sql_query(query, conn)
    
    if setups_df.empty:
        conn.close()
        return {}, pd.DataFrame(), pd.DataFrame()
        
    # Map sectors to Indian market standards
    setups_df['sector'] = setups_df['sector'].map(config.INDIAN_SECTOR_MAP).fillna(setups_df['sector'])
    
    # Compute Sector Co-Breakout Count per date
    setups_df['is_breakout'] = 1
    setups_df['sector_breakouts_count'] = setups_df.groupby(['timestamp', 'sector'])['is_breakout'].transform('sum')
    setups_df.loc[
        setups_df['sector'].isna() | 
        (setups_df['sector'] == 'N/A') | 
        (setups_df['sector'] == 'None') | 
        (setups_df['sector'] == ''), 
        'sector_breakouts_count'
    ] = 0
    
    # Compute Sector Avg PE
    sector_pes = setups_df.groupby('sector')['pe_ratio'].transform('mean')
    setups_df['sector_avg_pe'] = sector_pes
    
    # 2. Compute F-Score & XGBoost Probabilities
    setups_df['f_score'] = setups_df.apply(calculate_fundamental_score, axis=1)
    
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    xgb_input = setups_df[['volatility_contraction_score', 'volume_surge_score', 'momentum_score']].copy()
    for col in xgb_input.columns:
        mean_val = xgb_input[col].mean()
        if pd.isna(mean_val) or np.isnan(mean_val):
            mean_val = 0.0
        xgb_input[col] = xgb_input[col].fillna(mean_val)
    probs = model.predict_proba(xgb_input)[:, 1]
    setups_df['breakout_prob'] = (probs * 100).round(1)
    
    # 3. Compute Confluence Score
    if include_fundamentals:
        setups_df['confluence_score'] = (setups_df['breakout_prob'] * 0.6) + (setups_df['f_score'] * 10.0 * 0.4)
    else:
        # Technical-Only (100% Bias-Free)
        setups_df['confluence_score'] = setups_df['breakout_prob']
        
    setups_df['confluence_score'] = np.where(
        setups_df['sector_breakouts_count'] >= 2,
        setups_df['confluence_score'] + 10.0,
        setups_df['confluence_score']
    )
    setups_df['confluence_score'] = setups_df['confluence_score'].clip(upper=100.0).round(1)
    
    # Filter by user-defined threshold
    setups_df = setups_df[setups_df['confluence_score'] >= confluence_threshold].copy()
    
    if setups_df.empty:
        conn.close()
        return {}, pd.DataFrame(), pd.DataFrame()
        
    # 4. Apply 20-day Cooldown per stock (using trading days, not calendar days)
    filtered_rows = []
    last_trigger_date = {}
    
    for _, row in setups_df.iterrows():
        sym = row['symbol']
        dt = pd.to_datetime(row['timestamp'])
        
        if sym in last_trigger_date:
            # np.busday_count gives exact Mon-Fri business days between dates
            trading_days_since = np.busday_count(
                last_trigger_date[sym].date(), dt.date()
            )
            if trading_days_since < 20:
                continue
                
        last_trigger_date[sym] = dt
        filtered_rows.append(row)
        
    setups_df = pd.DataFrame(filtered_rows)
    
    # 5. Fetch Forward 20-day Performance in batch (including high/low for path-dependent simulation)
    symbols = setups_df['symbol'].unique().tolist()
    placeholders = ",".join(["?"] * len(symbols))
    query_prices = f"""
    SELECT s.symbol, p.timestamp, p.high, p.low, p.close
    FROM price_history p
    JOIN stocks s ON p.instrument_key = s.instrument_key
    WHERE s.symbol IN ({placeholders})
    ORDER BY s.symbol, p.timestamp ASC
    """
    prices_df = pd.read_sql_query(query_prices, conn, params=symbols)
    conn.close()
    
    # Organize prices into dict of Series for rapid slicing
    price_series_dict = {}
    for sym, group in prices_df.groupby('symbol'):
        group = group.sort_values('timestamp')
        group['timestamp'] = pd.to_datetime(group['timestamp'])
        price_series_dict[sym] = group.reset_index(drop=True)
        
    # Compute actual returns and status for each setup
    trades = []
    
    for _, row in setups_df.iterrows():
        sym = row['symbol']
        setup_dt = pd.to_datetime(row['timestamp'])
        
        trade_info = {
            "Symbol": sym,
            "Date": setup_dt.strftime('%Y-%m-%d'),
            "Sector": row['sector'],
            "Confluence (%)": row['confluence_score'],
            "F-Score": int(row['f_score']),
            "XGBoost Prob (%)": row['breakout_prob'],
            "Setup Close": None,
            "Exit Price": None,
            "Return (%)": None,
            "Status": "ACTIVE",
            "Days to Exit": None
        }
        
        if sym in price_series_dict:
            series = price_series_dict[sym]
            
            # Find row index corresponding to setup date
            setup_indices = series[series['timestamp'] >= setup_dt].index
            if not setup_indices.empty:
                idx = setup_indices[0]
                setup_close = series.iloc[idx]['close']
                trade_info["Setup Close"] = setup_close
                
                # Slices subsequent 20 trading days (excluding the setup day itself)
                future_slice = series.iloc[idx+1:idx+21]
                
                if not future_slice.empty:
                    stop_loss_price = setup_close * (1.0 - stop_loss_pct / 100.0)
                    target_price = setup_close * (1.0 + target_profit_pct / 100.0)
                    
                    status = "ACTIVE"
                    exit_price = None
                    ret_val = None
                    
                    # Chronological simulation to handle Drawdown Blindspot
                    days_to_exit = None
                    if len(future_slice) == 20:
                        for day_idx, (_, day_row) in enumerate(future_slice.iterrows()):
                            # Check stop loss first (pessimistic)
                            if day_row['low'] <= stop_loss_price:
                                status = "FAILURE"
                                exit_price = stop_loss_price
                                ret_val = -stop_loss_pct
                                days_to_exit = day_idx + 1
                                break
                            elif day_row['high'] >= target_price:
                                status = "SUCCESS"
                                exit_price = target_price
                                ret_val = target_profit_pct
                                days_to_exit = day_idx + 1
                                break
                        else:
                            # Ended 20 days without hitting either target or stop.
                            # This is a TIME_EXIT — not a decisive SUCCESS or FAILURE.
                            # Classifying a +0.1% 20-day drift as "SUCCESS" inflates win-rate.
                            exit_price = future_slice.iloc[-1]['close']
                            ret_val = ((exit_price - setup_close) / setup_close) * 100
                            status = "TIME_EXIT"
                            days_to_exit = 20
                    else:
                        # In-flight/Active trade: check if already hit SL or target
                        for day_idx, (_, day_row) in enumerate(future_slice.iterrows()):
                            if day_row['low'] <= stop_loss_price:
                                status = "FAILURE"
                                exit_price = stop_loss_price
                                ret_val = -stop_loss_pct
                                days_to_exit = day_idx + 1
                                break
                            elif day_row['high'] >= target_price:
                                status = "SUCCESS"
                                exit_price = target_price
                                ret_val = target_profit_pct
                                days_to_exit = day_idx + 1
                                break
                        else:
                            status = "ACTIVE"
                            
                    trade_info["Exit Price"] = round(exit_price, 2) if exit_price is not None else None
                    trade_info["Return (%)"] = round(ret_val, 2) if ret_val is not None else None
                    trade_info["Status"] = status
                    trade_info["Days to Exit"] = days_to_exit
                else:
                    trade_info["Status"] = "ACTIVE"
                    
        trades.append(trade_info)
        
    trades_df = pd.DataFrame(trades)
    
    # 6. Compute Ledger Metrics
    completed_trades = trades_df[trades_df['Status'].isin(["SUCCESS", "FAILURE", "TIME_EXIT"])]
    active_count = len(trades_df[trades_df['Status'] == "ACTIVE"])
    
    total_completed = len(completed_trades)
    wins   = len(completed_trades[completed_trades['Status'] == "SUCCESS"])
    losses = len(completed_trades[completed_trades['Status'] == "FAILURE"])
    time_exits = len(completed_trades[completed_trades['Status'] == "TIME_EXIT"])
    
    # Win-rate is computed only on DECISIVE outcomes (target-hit vs stop-hit).
    # TIME_EXIT trades are excluded from this ratio so a +2% 20-day drift
    # cannot masquerade as the same "success" as a genuine +30% target hit.
    total_decisive = wins + losses
    win_rate = (wins / total_decisive * 100) if total_decisive > 0 else 0.0
    
    # Average return per decisive category (used for Kelly / payoff ratio)
    target_returns   = completed_trades[completed_trades['Status'] == "SUCCESS"]["Return (%)"]
    stop_returns     = completed_trades[completed_trades['Status'] == "FAILURE"]["Return (%)"]
    time_exit_returns = completed_trades[completed_trades['Status'] == "TIME_EXIT"]["Return (%)"]
    
    avg_gain = target_returns.mean() if not target_returns.empty else 0.0
    avg_loss = stop_returns.mean()   if not stop_returns.empty   else 0.0
    
    # Profit Factor: uses ALL completed trade returns (inclusive of time exits)
    # so the P&L picture reflects what actually happened in the market.
    all_positive = pd.concat([
        target_returns[target_returns > 0],
        time_exit_returns[time_exit_returns > 0]
    ])
    all_negative = pd.concat([
        stop_returns[stop_returns < 0],
        time_exit_returns[time_exit_returns < 0]
    ])
    total_gains  = all_positive.sum()
    total_losses = abs(all_negative.sum())
    profit_factor = (total_gains / total_losses) if total_losses > 0 else (999.0 if total_gains > 0 else 0.0)
    
    # Kelly Criterion (based on decisive trades: target-hit vs stop-hit)
    # Kelly % = W - (1 - W) / R  where R = avg_gain / |avg_loss|
    payoff_ratio = (avg_gain / abs(avg_loss)) if (avg_loss != 0) else 0.0
    
    if win_rate > 0 and avg_loss != 0:
        W = win_rate / 100.0
        R = payoff_ratio
        kelly = W - (1.0 - W) / R if R > 0 else 0.0
        kelly_pct = max(0.0, kelly * 100)
    else:
        kelly_pct = 0.0
    
    # --- Out-of-Sample (OOS) Win-Rate ---
    # The XGBoost model was trained on data older than 6 months. Signals from the
    # last 6 months were in the held-out TEST set — their win-rate is genuinely
    # out-of-sample. Signals before this boundary are in-sample and will show
    # optimistically inflated probabilities due to model memorisation.
    from datetime import datetime, timedelta
    oos_boundary = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    oos_decisive = completed_trades[
        (completed_trades['Date'] >= oos_boundary) &
        (completed_trades['Status'].isin(["SUCCESS", "FAILURE"]))
    ]
    oos_wins          = len(oos_decisive[oos_decisive['Status'] == "SUCCESS"])
    oos_decisive_count = len(oos_decisive)
    oos_win_rate      = round(oos_wins / oos_decisive_count * 100, 1) if oos_decisive_count > 0 else None
        
    # Calculate average days to target for SUCCESS trades
    success_trades = completed_trades[completed_trades['Status'] == "SUCCESS"]
    avg_days_to_target = success_trades['Days to Exit'].mean() if not success_trades.empty else 0.0
        
    metrics = {
        "win_rate":         round(win_rate, 1),   # Full history — mix of in-sample & OOS
        "oos_win_rate":     oos_win_rate,          # Last 6 months only — genuinely OOS
        "oos_decisive_count": oos_decisive_count,
        "total_flags":      len(trades_df),
        "completed_count":  total_completed,
        "decisive_count":   total_decisive,
        "time_exit_count":  time_exits,
        "active_count":     active_count,
        "avg_gain":         round(avg_gain, 1),
        "avg_loss":         round(avg_loss, 1),
        "profit_factor":    round(profit_factor, 2),
        "kelly_pct":        round(kelly_pct, 1),
        "payoff_ratio":     round(payoff_ratio, 2),
        "avg_days_to_target": round(avg_days_to_target, 1)
    }
    
    # 7. Compute Win-Rate Breakdown by Archetype Group
    # Note: We classify general/turnaround/cyclical/structural categories based on F-Score/Debt/PEG ranges
    archetypes_data = []
    
    # We assign rows in trades_df to archetypes dynamically for comparison
    # We load detailed metrics from setups_df back into trades_df to check archetype conditions
    if not setups_df.empty:
        details_df = setups_df[['symbol', 'timestamp', 'roce', 'debt_to_equity', 'price_to_book', 'earnings_growth', 'interest_coverage', 'revenue_growth', 'stage_2_flag']].copy()
        details_df['Date'] = pd.to_datetime(details_df['timestamp']).dt.strftime('%Y-%m-%d')
        trades_merged = pd.merge(trades_df, details_df, left_on=['Symbol', 'Date'], right_on=['symbol', 'Date'], how='inner')
        
        # Define masks
        completed_merged = trades_merged[trades_merged['Status'].isin(["SUCCESS", "FAILURE", "TIME_EXIT"])]
        
        archetypes_masks = {
            "Structural Compounder": (completed_merged['roce'] >= 0.20) & (completed_merged['debt_to_equity'] <= 0.10) & (completed_merged['stage_2_flag'] == 1),
            "Turnaround Multibagger": (completed_merged['debt_to_equity'] <= 0.75) & (completed_merged['roce'] >= 0.15) & (completed_merged['revenue_growth'] >= 0.15),
            "Cyclical Deep Value": (completed_merged['price_to_book'] <= 3.0) & (completed_merged['earnings_growth'] >= 1.0) & (completed_merged['interest_coverage'] >= 4.0),
            "General Breakout": pd.Series(True, index=completed_merged.index)
        }
        
        for arch_name, mask in archetypes_masks.items():
            subset = completed_merged[mask]
            sub_total = len(subset)
            sub_wins = len(subset[subset['Status'] == "SUCCESS"])
            # Win-rate per archetype: decisive trades only (ignore TIME_EXIT)
            sub_decisive = len(subset[subset['Status'].isin(["SUCCESS", "FAILURE"])])
            sub_win_rate = (sub_wins / sub_decisive * 100) if sub_decisive > 0 else 0.0
            sub_avg_ret = subset['Return (%)'].mean() if sub_total > 0 else 0.0
            
            archetypes_data.append({
                "Archetype": arch_name,
                "Trades": sub_total,
                "Win-Rate (%)": round(sub_win_rate, 1),
                "Avg Return (%)": round(sub_avg_ret, 1)
            })
            
    archetype_stats_df = pd.DataFrame(archetypes_data)
    
    if not trades_df.empty:
        trades_df = trades_df.sort_values(by='Date', ascending=False).reset_index(drop=True)
        
    return metrics, trades_df, archetype_stats_df
