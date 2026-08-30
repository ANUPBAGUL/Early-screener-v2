import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
import xgboost as xgb
import yfinance as yf
import config

# Ensure local paths are searched first to avoid importing global site-packages conflicts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "pipeline")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "engine")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Force reload of local modules to clear cached global site-packages from Streamlit memory
for mod in ["upstox_client", "similarity_engine", "feature_engineer", "build_db"]:
    if mod in sys.modules:
        del sys.modules[mod]

try:
    from similarity_engine import fetch_latest_features, calculate_similarity_score
    from upstox_client import UpstoxClient
    from dotenv import set_key
except ImportError as e:
    st.error(f"Failed to import modules: {e}")

# --- Page Configuration ---
st.set_page_config(
    page_title="High-Growth & Multibagger Screener",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling for Premium Dark Theme ---
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .metric-card {
        background-color: #1E1E24;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- Database Connection ---
DB_PATH = config.DB_PATH

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def calculate_fundamental_score(row):
    score = 0
    # 1. Solvency (Max 2 pts)
    de = row.get('debt_to_equity')
    if de is not None and not pd.isna(de):
        if de <= 0.5:
            score += 1
    icr = row.get('interest_coverage')
    if icr is not None and not pd.isna(icr):
        if icr >= 4.0:
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
    if pb is not None and not pd.isna(pb):
        if pb > 0.0 and pb <= 3.0:
            score += 1
    pe = row.get('pe_ratio')
    sector_avg = row.get('sector_avg_pe')
    if pe is not None and not pd.isna(pe) and sector_avg is not None and not pd.isna(sector_avg) and sector_avg > 0:
        if pe <= sector_avg:
            score += 1
            
    return score

@st.cache_data(ttl=86400)
def check_market_regime():
    """
    Fetches Nifty 50 monthly candles from Upstox and determines if Nifty is above/below monthly 10 EMA.
    Returns (is_bull, nifty_close, nifty_ema)
    """
    try:
        client = UpstoxClient()
        if not client.access_token:
            return None
        candles = client.fetch_historical_candles(config.INDEX_KEY, interval="month")
        if not candles:
            return None
        # Columns: [timestamp, open, high, low, close, volume, oi]
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        df = df.sort_values('timestamp').copy()
        df['close'] = pd.to_numeric(df['close'])
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        
        latest = df.iloc[-1]
        nifty_close = latest['close']
        nifty_ema = latest['ema_10']
        is_bull = nifty_close >= nifty_ema
        return is_bull, nifty_close, nifty_ema
    except Exception as e:
        print(f"Error checking Nifty monthly EMA: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache news feed for 1 hour to prevent HTTP blocking on user clicks
def fetch_stock_news(symbol):
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        return ticker.news
    except Exception:
        return []

def fetch_analogue_data(symbol, match_date):
    """
    Fetches price history for the historical match and slices it around the breakout date.
    """
    if not symbol or pd.isna(symbol) or not match_date or pd.isna(match_date):
        return None
        
    conn = get_connection()
    query = """
    SELECT timestamp, close, volume
    FROM price_history
    WHERE instrument_key = (SELECT instrument_key FROM stocks WHERE symbol = ? LIMIT 1)
    ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()
    
    if df.empty:
        return None
        
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
        match_dt = pd.to_datetime(match_date).date()
        
        # Find closest matching index
        df['date_diff'] = df['timestamp'].apply(lambda x: abs((x - match_dt).days))
        closest_idx = df['date_diff'].idxmin()
        
        # Slice 40 trading days before and 80 trading days after the breakout setup date
        start_idx = max(0, closest_idx - 40)
        end_idx = min(len(df) - 1, closest_idx + 80)
        
        sliced_df = df.iloc[start_idx:end_idx+1].copy()
        sliced_df.set_index('timestamp', inplace=True)
        return sliced_df
    except Exception as e:
        print(f"Error slicing analogue data: {e}")
        return None

# --- Try Loading XGBoost Model ---
xgb_loaded = False
try:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine", "breakout_model.json")
    if os.path.exists(model_path):
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        xgb_loaded = True
except Exception as e:
    st.sidebar.error(f"XGBoost Model Load Failure: {e}")

# --- Check Nifty Market Regime ---
market_status = check_market_regime()

# --- Sidebar: Authentication & Settings ---
with st.sidebar:
    st.header("🔑 Upstox Authentication")
    try:
        client = UpstoxClient()
        if client.access_token:
            st.success("✅ Authenticated with Upstox!")
            if st.button("Re-authenticate", key="reauth"):
                env_path = os.path.join(os.path.dirname(__file__), ".env")
                try:
                    from dotenv import unset_key
                    unset_key(env_path, "UPSTOX_ACCESS_TOKEN")
                except ImportError:
                    set_key(env_path, "UPSTOX_ACCESS_TOKEN", "")
                
                # Clear from active process memory
                if "UPSTOX_ACCESS_TOKEN" in os.environ:
                    del os.environ["UPSTOX_ACCESS_TOKEN"]
                
                st.success("Access token cleared. Reloading...")
                st.rerun()
        else:
            st.warning("Not Authenticated")
            if client.api_key and client.redirect_uri:
                login_url = client.get_login_url()
                st.markdown(f"[**Click Here to Login to Upstox**]({login_url})", unsafe_allow_html=True)
                
                auth_code = st.text_input("Paste Auth Code Here:")
                if st.button("Verify Code"):
                    try:
                        token = client.get_access_token(auth_code)
                        env_path = os.path.join(os.path.dirname(__file__), ".env")
                        set_key(env_path, "UPSTOX_ACCESS_TOKEN", token)
                        st.success("Successfully Authenticated! Reloading...")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Missing UPSTOX_API_KEY or UPSTOX_REDIRECT_URI in .env file.")
    except Exception as e:
        st.error(f"Error loading client: {e}")
        
    st.markdown("---")
    st.header("💾 Database Management")
    if st.button("Update Database Now", use_container_width=True):
        if 'client' in locals() and client.access_token:
            with st.spinner("Fetching latest data from Upstox (Smart Update)..."):
                try:
                    from pipeline.build_db import build_database
                    from engine.feature_engineer import run_feature_engineering
                    
                    stats = build_database()
                    if stats:
                        run_feature_engineering()
                        if stats['stocks_updated'] == 0:
                            st.info("✅ Database is already up to date! No new data to fetch.")
                        else:
                            st.success(f"✅ Success! Updated {stats['stocks_updated']} stocks with {stats['candles_added']} new daily candles.")
                except Exception as e:
                    st.error(f"Error during update: {e}")
        else:
            st.error("Please authenticate with Upstox first!")
            
    st.markdown("---")
    st.header("🤖 Machine Learning Model")
    
    # Interactive UI Selector for Target Threshold Scenario
    mode_options = {
        "🎯 15% (Target Predictor - Default)": 0.15,
        "📈 20% (Swing Growth - Broad Signals)": 0.20,
        "🚀 50% (Multibagger - Strict / Selective)": 0.50
    }
    
    current_thresh = getattr(config, 'BREAKOUT_LABEL_THRESHOLD', 0.15)
    current_index = 0 if current_thresh == 0.15 else (1 if current_thresh == 0.20 else 2)
    
    selected_mode = st.selectbox(
        "Select Model Target Scenario:",
        options=list(mode_options.keys()),
        index=current_index,
        help="Select what target return the XGBoost classifier is trained to predict over a 20-day holding period."
    )
    
    new_thresh = mode_options[selected_mode]
    if new_thresh != config.BREAKOUT_LABEL_THRESHOLD:
        config.BREAKOUT_LABEL_THRESHOLD = new_thresh
        st.warning(f"⚠️ Target mode changed to {selected_mode}. Click 'Retrain XGBoost Classifier' below to retrain model & update DNA library!")

    holding_period = st.slider(
        "Strategy Holding Period (Trading Days):",
        min_value=5, max_value=60,
        value=int(getattr(config, 'HOLDING_PERIOD', 20)),
        step=1,
        help="Configure the holding window for model training labels and backtesting exit criteria."
    )
    if holding_period != config.HOLDING_PERIOD:
        config.HOLDING_PERIOD = holding_period
        st.warning(f"⚠️ Strategy holding period changed to {holding_period} days. Click 'Retrain XGBoost Classifier' below to retrain the model and build the DNA library!")

    # Informative UI Callout Card
    st.markdown(
        f"""
        <div style="background-color: #1e293b; padding: 12px; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 12px;">
            <p style="margin: 0; font-size: 0.85rem; color: #94a3b8;"><b>Active Mode:</b> <span style="color: #60a5fa;"><b>≥+{int(new_thresh * 100)}% Close-to-Close Return ({config.HOLDING_PERIOD} Days)</b></span></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("ℹ️ How Target Thresholds Change UI Outcomes"):
        st.markdown("""
        - **20% (Swing Growth)**:
          - *Outcome:* High signal density & frequency.
          - *Trade-off:* Lower precision per signal, more frequent trades.
        - **30% (Target Hit - Recommended Default)**:
          - *Outcome:* Balanced signal frequency & optimal precision.
          - *Advantage:* Directly predicts if setup reaches your **+30% swing profit target**.
        - **50% (Multibagger)**:
          - *Outcome:* Very rare signals, highly conservative.
          - *Trade-off:* Requires extensive historical dataset; produces fewer flags.
        """)

    if st.button("Retrain XGBoost Classifier", use_container_width=True):
        with st.spinner("Training model & extracting DNA..."):
            try:
                from engine.breakout_model import train_and_save_model
                train_and_save_model()
                st.success(f"✅ XGBoost model retrained for {int(new_thresh*100)}% target mode and DNA library updated!")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")
                
    st.markdown("---")
    st.header("⚙️ Screener Parameters")
    
    strategy_mode = st.selectbox(
        "Select Strategy Mode:",
        ["🎯 Standard Swing Breakouts", "🚀 Ground-Floor Microcap Inflection"],
        help="Standard mode focuses on momentum breakouts in midcaps. Microcap Inflection mode targets sub-₹1,500 Cr microcap turnarounds."
    )
    is_microcap_mode = (strategy_mode == "🚀 Ground-Floor Microcap Inflection")
    
    if is_microcap_mode:
        st.sidebar.info("🚀 **Microcap Mode Active**: Forces MCAP to ₹50-1,500 Cr, volume to 30k+ shares, and applies Governance Filters (Promoter >= 45%, Positive CFO).")
        st.sidebar.warning("⚠️ **Risk Alert**: Microcap stocks require a wider **15% Stop-Loss** and a smaller **0.5% Position Sizing** to prevent risk of ruin.")
        
    st.subheader("Momentum & Breakout Settings")
    default_vol_surge = 1.5 if is_microcap_mode else 3.0
    vol_surge = st.slider(
        "Volume Surge Threshold (x Avg)", 
        min_value=1.0, max_value=10.0, value=default_vol_surge, step=0.5,
        help="How many times the average volume the breakout needs to be."
    )
    
    vcp_tightness = st.slider(
        "VCP Tightness Threshold (%)",
        min_value=1.0, max_value=15.0, value=5.0, step=0.5,
        help="Maximum allowable variance during the consolidation phase."
    )
    
    st.subheader("Market Context")
    similarity_threshold = st.slider(
        "Historical Similarity Match (%)",
        min_value=50, max_value=99, value=75, step=1,
        help="Minimum similarity to historical multibagger setups to flag."
    )
    
    st.subheader("Algorithmic Strategy Filters")
    
    # 1. Market Cap Range Slider (₹300 Cr to ₹10,000 Cr default)
    default_mcap_range = (50.0, 1500.0) if is_microcap_mode else (300.0, 10000.0)
    market_cap_range = st.slider(
        "Market Cap Range (₹ Crores)",
        min_value=10.0, max_value=50000.0, value=default_mcap_range, step=50.0,
        help="Configure the size constraint to target small-to-midcap stocks."
    )
    
    enforce_stage2 = st.checkbox(
        "Enforce Minervini Stage 2", 
        value=True, 
        help="Enforce: Price > 50 > 150 > 200 SMAs, and 200 SMA trending up."
    )
    enforce_liquidity = st.checkbox(
        "Enforce Average Daily Volume (>= 200k shares)", 
        value=True, 
        help="Filter out stocks with 50-day average daily volume under 200,000 shares."
    )
    enforce_quality = st.checkbox(
        "Enforce Quality (ROCE/ROE >= 18% & Debt/Equity <= 0.5)", 
        value=True, 
        help="Filter for capital efficiency (ROCE/ROE >= 18%) and fortress balance sheet (Debt/Equity <= 0.5)."
    )
    enforce_valuation = st.checkbox(
        "Enforce Valuation Filter (PEG < 1.0)",
        value=False,
        help="Filter out stocks with PEG ratio > 1.0 (Growth At a Reasonable Price)."
    )
    
    st.markdown("---")
    st.subheader("🛡️ Risk Management Settings")
    default_stop_loss = 15.0 if is_microcap_mode else 7.5
    stop_loss_pct = st.slider(
        "Initial Stop-Loss (%)",
        min_value=3.0, max_value=20.0, value=default_stop_loss, step=0.5,
        help="Strict initial stop-loss percentage to limit trade downside."
    )
    target_profit_pct = st.slider(
        "Swing Target Profit (%)",
        min_value=10.0, max_value=60.0, value=15.0, step=1.0,
        help="Take profit target for the first 50% of the position."
    )
    
    st.markdown("---")
    st.header("⚡ Intraday Workstation Mode")
    enable_intraday_mode = st.sidebar.toggle(
        "Enable Intraday Workstation", 
        value=False,
        help="Turn ON to expand the Top 15 Intraday Watchlist, Position Sizing Calculator, Nifty VWAP Kill-Switch, and Live 15-Min Breakout Scanner."
    )
    
    intraday_capital = config.INTRADAY_CONFIG["DEFAULT_CAPITAL"]
    intraday_risk_pct = config.INTRADAY_CONFIG["DEFAULT_RISK_PCT"]
    
    if enable_intraday_mode:
        intraday_capital = st.sidebar.number_input(
            "Total Trading Capital (₹)",
            min_value=10000.0, max_value=100000000.0,
            value=config.INTRADAY_CONFIG["DEFAULT_CAPITAL"],
            step=50000.0,
            help="Your total account capital available for trading."
        )
        intraday_risk_pct = st.sidebar.slider(
            "Risk Per Trade (%)",
            min_value=0.25, max_value=3.0, value=1.0, step=0.25,
            help="Maximum account equity you are willing to risk on a single trade."
        )
    
    # Market Kill Switch is informative only (does not clear/blank results)
            
    if st.button("Run Screener", use_container_width=True):
        st.session_state['run_screener'] = True

# --- Main Dashboard ---
st.title("🚀 High-Growth & Multibagger Screener")
st.markdown("Identify explosive momentum and potential multibaggers via volume anomalies and volatility contraction.")

# Display Market Regime status banner
if market_status is not None:
    is_bull, nifty_close, nifty_ema = market_status
    if is_bull:
        st.success(f"🟢 **Bull Market Regime**: Nifty 50 is trending above its Monthly 10 EMA (Close: {nifty_close:.1f} vs EMA: {nifty_ema:.1f}). Breakout setups are highly favorable.")
    else:
        st.warning(f"⚠️ **Bear Market Regime Active**: Nifty 50 is trading below its Monthly 10 EMA (Close: {nifty_close:.1f} vs EMA: {nifty_ema:.1f}). Long breakouts are still profitable (+1.61% average return), but drawdowns are higher. Sizing down by 50% is recommended.")

if st.session_state.get('run_screener', False):
    
    # 1. Market Kill Switch Sizing Down Warning (No Pause/Blanking)
    if market_status is not None and not is_bull and False:
        pass
    else:
        if market_status is not None and not is_bull:
            st.warning("⚠️ **Bear Market Regime Active**: Nifty 50 is trading below its Monthly 10 EMA. Expectancy remains positive (+1.61%), but drawdowns are historically higher. **Sizing down position size by 50% is strongly recommended**.")
        st.success(f"Scanning stocks for >{vol_surge}x volume surges and <{vcp_tightness}% VCP tightness...")
        
        # Fetch real data
        try:
            features_df = fetch_latest_features()
            if not features_df.empty:
                import numpy as np
                
                # A. Compute cosine similarity against DNA library
                scored_df = calculate_similarity_score(features_df)
                
                # B. Run XGBoost classifier probability if loaded
                if xgb_loaded:
                    feature_cols = ['volatility_contraction_score', 'volume_surge_score', 'momentum_score']
                    xgb_input = scored_df[feature_cols].copy()
                    for col in feature_cols:
                        mean_val = xgb_input[col].mean()
                        if pd.isna(mean_val) or np.isnan(mean_val):
                            mean_val = 0.0
                        xgb_input[col] = xgb_input[col].fillna(mean_val)
                    probs = model.predict_proba(xgb_input)[:, 1]
                    scored_df['breakout_prob'] = (probs * 100).round(1)
                else:
                    scored_df['breakout_prob'] = 0.0
                    
                # Map global sectors to standard Indian stock market sectors
                scored_df['sector'] = scored_df['sector'].map(config.INDIAN_SECTOR_MAP).fillna(scored_df['sector'])
                
                # C. Compute Dynamic Sector Average PE and Relative PE
                sector_pes = scored_df.groupby('sector')['pe_ratio'].transform('mean')
                scored_df['sector_avg_pe'] = sector_pes
                scored_df['relative_pe'] = (scored_df['pe_ratio'] / scored_df['sector_avg_pe'].replace(0, np.nan)).round(2)
                scored_df['relative_pe'] = scored_df['relative_pe'].fillna(1.0)
                
                # D. Calculate Sector Co-Breakout Clustering & F-Score / Confluence Score
                scored_df['is_breakout'] = (
                    (scored_df['volume_surge_score'] >= 1.5) & 
                    (scored_df['volatility_contraction_score'] <= 10.0)
                ).astype(int)
                
                # Count concurrent breakouts per sector (ignoring invalid/N/A sectors)
                scored_df['sector_breakouts_count'] = scored_df.groupby('sector')['is_breakout'].transform('sum')
                scored_df.loc[
                    scored_df['sector'].isna() | 
                    (scored_df['sector'] == 'N/A') | 
                    (scored_df['sector'] == 'None') | 
                    (scored_df['sector'] == ''), 
                    'sector_breakouts_count'
                ] = 0
                
                # Calculate F-Score
                scored_df['f_score'] = scored_df.apply(calculate_fundamental_score, axis=1)
                
                # Fix 2: Archetype-Specific Confluence Scores
                # Momentum paths (General, Structural): XGBoost signal dominates.
                # Value paths (Turnaround, Cyclical): F-Score dominates (fundamental inflection is the edge).
                sector_bonus = np.where(scored_df['sector_breakouts_count'] >= 2, 10.0, 0.0)
                
                for arch_key, w in config.CONFLUENCE_WEIGHTS.items():
                    col = f"confluence_{arch_key.lower()}"
                    scored_df[col] = (
                        (scored_df['breakout_prob'] * w['xgb']) +
                        (scored_df['f_score'] * 10.0 * w['fscore']) +
                        sector_bonus
                    ).clip(upper=100.0).round(1)
                
                # Keep a single representative score for sorting/display (General weights = default)
                scored_df['confluence_score'] = scored_df['confluence_general']
                
                # If Microcap Mode is active, apply overrides before filtering
                if is_microcap_mode:
                    # 1. Overwrite Stage 2 flag with relaxed Stage 1 (50 SMA > 150 SMA)
                    scored_df['stage_2_flag'] = np.where(scored_df['sma_50'] > scored_df['sma_150'], 1, 0)
                    
                    # 2. Apply Governance Guardrail
                    scored_df = scored_df[
                        (scored_df['promoter_holding'] >= config.MICROCAP_CONFIG["MIN_PROMOTER"]) &
                        (scored_df['operating_cash_flow'] > 0.0)
                    ].copy()
                    
                # 3. Compute Volatility-Adjusted Strategy Columns
                scored_df['dynamic_sl'] = (scored_df['volatility_contraction_score'] * 1.5).clip(5.0, 15.0).round(1)
                scored_df['tentative_days'] = scored_df['match_days'].apply(
                    lambda x: f"{int(x)} Days" if pd.notna(x) and x > 0 else f"{config.HOLDING_PERIOD} Days"
                )
                
                def get_strategy_suggestion(row):
                    close = row.get('close', 0.0)
                    sma50 = row.get('sma_50', 1.0)
                    sl = row.get('dynamic_sl', 7.5)
                    vcp = row.get('volatility_contraction_score', 5.0)
                    mcap = row.get('market_cap', 1000.0)
                    pbd = row.get('pbd_profile')
                    
                    if pbd == 'b':
                        return f"⚠️ Bearish b-Profile | Distribution Area | Avoid Breakouts"
                    
                    if sma50 > 0.0 and (close / sma50) > 1.20:
                        return f"⚠️ Over-Extended | Risk 0.25% | SL {sl}% | Wait for Pullback"
                    
                    if mcap < 1500.0:
                        return f"⚠️ Microcap Vol | Risk 0.5% | SL {sl}% | Governance OK"
                    elif vcp <= 4.0:
                        return f"🟢 Low Risk VCP | Risk 1.0% | SL {sl}% | GTT Buy | Trail 50D"
                    elif vcp > 7.0:
                        return f"🟡 Wide Base | Risk 0.75% | SL {sl}% | Buy Pullback"
                    else:
                        return f"🔵 Std Swing | Risk 1.0% | SL {sl}% | TP 15% GTT"
                        
                scored_df['risk_review'] = scored_df.apply(get_strategy_suggestion, axis=1)
                st.session_state['scored_df'] = scored_df

                # E. Apply standard user sliders (Volume Surge, VCP, Similarity) for the Momentum Engine
                tech_filtered_df = scored_df[
                    (scored_df['volume_surge_score'] >= vol_surge) & 
                    (scored_df['volatility_contraction_score'] <= vcp_tightness) &
                    (scored_df['similarity_score'] >= similarity_threshold)
                ].copy()
                
                # Initialize symbol tracking and records mapping
                all_flagged_symbols = set()
                global_matched_records = {}
                
                # Fix 3: Earnings risk filter (sidebar toggle)
                hide_earnings_risk = st.sidebar.toggle(
                    "🔔 Hide Earnings Risk (≤5 days)",
                    value=True,
                    help="Hide stocks with earnings announcements within 5 trading days. Buying into a VCP setup before earnings is high-risk."
                )
                
                def show_archetype_dataframe(df, tab_name, confluence_col='confluence_score'):
                    if df.empty:
                        st.info(f"ℹ️ No stocks matched {tab_name} parameters in this scan.")
                        return
                    
                    # Apply earnings risk filter
                    if hide_earnings_risk and 'days_to_earnings' in df.columns:
                        earnings_risk = df['days_to_earnings'].notna() & (df['days_to_earnings'] <= 5)
                        if earnings_risk.any():
                            st.warning(f"⚠️ {earnings_risk.sum()} stock(s) hidden — earnings within 5 trading days.")
                        df = df[~earnings_risk].copy()
                    
                    if df.empty:
                        st.info(f"ℹ️ No stocks matched {tab_name} after earnings filter.")
                        return
                    
                    df_sorted = df.sort_values(by=confluence_col, ascending=False)
                    df_sorted['Action'] = df_sorted[confluence_col].apply(
                        lambda x: "🚨 Parabolic Cluster" if x >= 85 else "🔥 High Conviction" if x >= 70 else "⚡ Breakout Alert"
                    )
                    
                    # Fix 3: Earnings Warning Badge
                    def earnings_badge(days):
                        if pd.isna(days) or days is None: return ""
                        if days <= 2: return f"🔴 EARNINGS {int(days)}d"
                        if days <= 5: return f"🟠 EARNINGS {int(days)}d"
                        if days <= 10: return f"🟡 EARNINGS {int(days)}d"
                        return ""
                    
                    # Fix 5: Data Quality Badge
                    def quality_badge(q):
                        if pd.isna(q): return "⬛"
                        q = int(q)
                        return {0: "⬛ No data", 1: "🟥 Partial", 2: "🟡 Good", 3: "🟢 Verified"}.get(q, "⬛")
                    
                    df_sorted['Earnings'] = df_sorted['days_to_earnings'].apply(earnings_badge) if 'days_to_earnings' in df_sorted.columns else ""
                    df_sorted['Data Quality'] = df_sorted['data_quality'].apply(quality_badge) if 'data_quality' in df_sorted.columns else "⬛"
                    
                    # Fix 4: Pivot High distance
                    if 'pivot_high' in df_sorted.columns:
                        df_sorted['pivot_high'] = df_sorted['pivot_high'].round(2)
                    
                    for _, row in df_sorted.iterrows():
                        all_flagged_symbols.add(row['symbol'])
                        global_matched_records[row['symbol']] = row.to_dict()
                        
                    # Determine which columns exist safely
                    base_cols = [
                        'symbol', 'Earnings', 'Data Quality', 'sector',
                        confluence_col, 'f_score', 'breakout_prob', 'similarity_score',
                        'volume_surge_score', 'volatility_contraction_score', 'market_cap', 'vol_50d_avg',
                        'promoter_holding', 'operating_cash_flow',
                        'pivot_high',
                        'roce', 'debt_to_equity', 'pe_ratio', 'relative_pe', 'price_to_book',
                        'earnings_growth', 'interest_coverage',
                        'match_symbol', 'match_date', 'match_return', 'Action'
                    ]
                    avail_cols = [c for c in base_cols if c in df_sorted.columns]
                    display_df = df_sorted[avail_cols].copy()
                    
                    if 'promoter_holding' in display_df.columns:
                        display_df['promoter_holding'] = display_df['promoter_holding'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) and x > 0.0 else "N/A")
                    if 'operating_cash_flow' in display_df.columns:
                        display_df['operating_cash_flow'] = display_df['operating_cash_flow'].apply(lambda x: f"{x:.1f}" if pd.notna(x) and x != 0.0 else "N/A")
                    
                    display_df[confluence_col] = display_df[confluence_col].round(1)
                    display_df['f_score'] = display_df['f_score'].astype(int)
                    display_df['breakout_prob'] = display_df['breakout_prob'].round(1)
                    display_df['similarity_score'] = display_df['similarity_score'].round(1)
                    display_df['volume_surge_score'] = display_df['volume_surge_score'].round(2)
                    display_df['volatility_contraction_score'] = display_df['volatility_contraction_score'].round(2)
                    display_df['market_cap'] = display_df['market_cap'].round(1)
                    display_df['vol_50d_avg'] = display_df['vol_50d_avg'].astype(int)
                    display_df['roce'] = (display_df['roce'] * 100).round(1).astype(str) + "%"
                    display_df['debt_to_equity'] = display_df['debt_to_equity'].round(2)
                    display_df['pe_ratio'] = display_df['pe_ratio'].round(1)
                    display_df['relative_pe'] = display_df['relative_pe'].round(2)
                    display_df['price_to_book'] = display_df['price_to_book'].round(2)
                    display_df['earnings_growth'] = (display_df['earnings_growth'] * 100).round(1).astype(str) + "%"
                    display_df['interest_coverage'] = display_df['interest_coverage'].apply(lambda x: "Infinite" if x >= 999.0 else f"{x:.1f}")
                    
                    col_rename = {
                        'symbol': 'Symbol', 'Earnings': '⚠️ Earnings',
                        'Data Quality': '📊 Data Quality', 'sector': 'Sector',
                        confluence_col: 'Confluence (%)', 'f_score': 'F-Score (10)',
                        'breakout_prob': 'Breakout Prob (%)', 'similarity_score': 'Similarity (%)',
                        'volume_surge_score': 'Volume Surge (x)', 'volatility_contraction_score': 'VCP Tightness (%)',
                        'market_cap': 'Market Cap (Cr)', 'vol_50d_avg': '50D Avg Vol',
                        'promoter_holding': 'Promoter Holding', 'operating_cash_flow': 'CFO (₹ Cr)',
                        'pivot_high': '🎯 Pivot High (₹)',
                        'roce': 'ROCE/ROE (%)', 'debt_to_equity': 'Debt-to-Equity',
                        'pe_ratio': 'P/E', 'relative_pe': 'Relative P/E',
                        'price_to_book': 'P/B', 'earnings_growth': 'YoY Profit Var',
                        'interest_coverage': 'Int. Coverage',
                        'match_symbol': 'Closest Match', 'match_date': 'Match Date',
                        'match_return': 'Hist. Return (%)', 'Action': 'Action'
                    }
                    display_df = display_df.rename(columns={k: v for k, v in col_rename.items() if k in display_df.columns})
                    
                    # Show data quality warning if many stocks have partial data
                    low_q = (df_sorted['data_quality'] <= 1).sum() if 'data_quality' in df_sorted.columns else 0
                    if low_q > 0:
                        st.caption(f"ℹ️ {low_q} stock(s) in this view have partial or no fundamental data (🟥/⬛). F-Score for these may be unreliable.")
                    
                    grid_key = f"grid_{tab_name.replace(' ', '_').lower()}"
                    event = st.dataframe(display_df, use_container_width=True, hide_index=True, on_select="rerun", key=grid_key, selection_mode="single-row")
                    if event and event.selection and event.selection.rows:
                        idx = event.selection.rows[0]
                        if idx < len(display_df):
                            selected_sym = display_df.iloc[idx]['Symbol']
                            st.session_state['selected_stock_view'] = selected_sym
                            st.session_state['selected_stock_context'] = tab_name.lower()
                            st.rerun()
                
                # Render Sector Co-Breakout Clustering Alerts
                clustered_sectors = scored_df[
                    (scored_df['is_breakout'] == 1) & 
                    (scored_df['sector_breakouts_count'] >= 2)
                ][['sector', 'sector_breakouts_count']].drop_duplicates()
                
                if not clustered_sectors.empty:
                    st.markdown("### 🚨 Institutional Sector Rotation Alerts")
                    cols_alert = st.columns(min(len(clustered_sectors), 4))
                    for idx, row_c in enumerate(clustered_sectors.itertuples()):
                        col_idx = idx % len(cols_alert)
                        with cols_alert[col_idx]:
                            st.info(f"🔥 **{row_c.sector}**\n\n**{row_c.sector_breakouts_count} concurrent breakouts** detected! Institutional flow confirmed (+10% Confluence bonus applied).")
                
                # ---------------------------------------------------------
                # ⚡ INTRADAY WORKSTATION SUITE (Active when Toggle is ON)
                # ---------------------------------------------------------
                if enable_intraday_mode:
                    st.markdown("## ⚡ Intraday Trading Workstation")
                    
                    # 1. Nifty 15-Min Intraday VWAP Kill-Switch Guard
                    nifty_vwap_bullish = True
                    try:
                        nifty_candles = client.fetch_historical_candles(config.INDEX_KEY, interval="15minute")
                        if nifty_candles and len(nifty_candles) > 0:
                            ndf = pd.DataFrame(nifty_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                            ndf['close'] = pd.to_numeric(ndf['close'])
                            ndf['volume'] = pd.to_numeric(ndf['volume'])
                            n_ltp = ndf.iloc[-1]['close']
                            n_vol_sum = ndf['volume'].sum()
                            n_vwap = (ndf['close'] * ndf['volume']).sum() / n_vol_sum if n_vol_sum > 0 else n_ltp
                            nifty_vwap_bullish = n_ltp >= n_vwap
                            
                            c_col1, c_col2, c_col3 = st.columns(3)
                            with c_col1:
                                if nifty_vwap_bullish:
                                    st.success(f"🟢 **Nifty Intraday Status**: BULLISH (LTP: {n_ltp:.1f} ≥ VWAP: {n_vwap:.1f})")
                                else:
                                    st.error(f"🔴 **Nifty Intraday Status**: BEARISH (LTP: {n_ltp:.1f} < VWAP: {n_vwap:.1f}) — Kill-Switch Active!")
                            with c_col2:
                                st.metric("Account Trading Capital", f"₹{intraday_capital:,.0f}")
                            with c_col3:
                                risk_amt = intraday_capital * (intraday_risk_pct / 100.0)
                                st.metric("Max Risk Per Trade", f"₹{risk_amt:,.0f} ({intraday_risk_pct}%)")
                    except Exception:
                        pass
                        
                    # 2. Generate Top 15 Intraday Watchlist Across All Archetypes
                    watchlist_candidates = scored_df[
                        (scored_df['confluence_score'] >= 70.0) &
                        (scored_df['stage_2_flag'] == 1) &
                        (scored_df['market_cap'] >= market_cap_range[0]) &
                        (scored_df['market_cap'] <= market_cap_range[1])
                    ].copy()
                    
                    if 'days_to_earnings' in watchlist_candidates.columns:
                        watchlist_candidates = watchlist_candidates[
                            watchlist_candidates['days_to_earnings'].isna() | (watchlist_candidates['days_to_earnings'] > 5)
                        ]
                        
                    watchlist_sorted = watchlist_candidates.sort_values(by='confluence_score', ascending=False).drop_duplicates(subset=['symbol']).head(15).copy()
                    
                    if watchlist_sorted.empty:
                        st.info("ℹ️ No high-conviction candidates (Confluence ≥ 70%) available for today's Intraday Watchlist.")
                    else:
                        st.markdown("### 📌 Today's Intraday Watchlist & Position Sizing Card (Top 15)")
                        
                        risk_per_trade = intraday_capital * (intraday_risk_pct / 100.0)
                        
                        w_list = []
                        for _, w_row in watchlist_sorted.iterrows():
                            sym = w_row['symbol']
                            sec = w_row['sector']
                            conf = round(w_row['confluence_score'], 1)
                            close_p = w_row['close'] if 'close' in w_row and not pd.isna(w_row['close']) else w_row.get('vol_50d_avg', 0)
                            pivot_p = w_row['pivot_high'] if 'pivot_high' in w_row and not pd.isna(w_row['pivot_high']) else round(close_p * 1.02, 2)
                            
                            sl_p = round(pivot_p * (1.0 - 0.015), 2)
                            risk_per_share = pivot_p - sl_p
                            
                            shares_qty = int(risk_per_trade / risk_per_share) if risk_per_share > 0 else 0
                            capital_req = shares_qty * pivot_p
                            
                            dist_pct = round(((pivot_p - close_p) / close_p) * 100, 1) if close_p > 0 else 0.0
                            
                            q_badge = {0: "⬛", 1: "🟥", 2: "🟡", 3: "🟢"}.get(int(w_row.get('data_quality', 0)), "⬛")
                            
                            w_list.append({
                                "Symbol": sym,
                                "Sector": sec,
                                "Confluence (%)": conf,
                                "Setup Close (₹)": round(close_p, 2),
                                "🎯 Pivot High (₹)": round(pivot_p, 2),
                                "Dist to Pivot (%)": f"+{dist_pct}%",
                                "Intraday SL (₹)": sl_p,
                                "Shares to Buy": shares_qty,
                                "Capital Req (₹)": f"₹{int(capital_req):,}",
                                "Data Quality": q_badge
                            })
                            
                        w_df = pd.DataFrame(w_list)
                        st.dataframe(w_df, use_container_width=True, hide_index=True)
                        
                        # CSV Export Button
                        csv_data = w_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Intraday Watchlist (CSV for Broker Terminal)",
                            data=csv_data,
                            file_name=f"intraday_watchlist_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # 3. Upstox 15-Min Live Trigger Scanner Button
                        st.markdown("### ⚡ Live Market-Hours Intraday Scanner")
                        if st.button("⚡ Scan Watchlist Live (Upstox 15-Min Feed)", use_container_width=True):
                            if not nifty_vwap_bullish:
                                st.warning("⚠️ Intraday Kill-Switch Active: Nifty 50 is trading below VWAP. Exercise caution with long breakouts.")
                            
                            with st.spinner("Fetching live 15-min intraday candles from Upstox for Watchlist stocks..."):
                                breakouts_found = 0
                                for _, item in w_df.iterrows():
                                    sym = item['Symbol']
                                    pivot_val = item['🎯 Pivot High (₹)']
                                    shares = item['Shares to Buy']
                                    sl_val = item['Intraday SL (₹)']
                                    
                                    ikey = None
                                    if 'instrument_key' in scored_df.columns:
                                        match_inst = scored_df[scored_df['symbol'] == sym]['instrument_key'].values
                                        if len(match_inst) > 0: ikey = match_inst[0]
                                    if not ikey:
                                        try:
                                            conn_tmp = get_connection()
                                            cur_tmp = conn_tmp.cursor()
                                            cur_tmp.execute("SELECT instrument_key FROM stocks WHERE symbol = ?", (sym,))
                                            r_tmp = cur_tmp.fetchone()
                                            if r_tmp: ikey = r_tmp[0]
                                            conn_tmp.close()
                                        except Exception:
                                            pass
                                    if not ikey: continue
                                    
                                    try:
                                        c15 = client.fetch_historical_candles(ikey, interval="15minute")
                                        if c15 and len(c15) >= 2:
                                            cdf15 = pd.DataFrame(c15, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                                            cdf15['close'] = pd.to_numeric(cdf15['close'])
                                            cdf15['volume'] = pd.to_numeric(cdf15['volume'])
                                            
                                            latest_ltp = cdf15.iloc[-1]['close']
                                            latest_vol = cdf15.iloc[-1]['volume']
                                            avg_15m_vol = cdf15['volume'].mean()
                                            
                                            if latest_ltp >= pivot_val and latest_vol >= (avg_15m_vol * 1.5):
                                                breakouts_found += 1
                                                st.error(f"🚨 **LIVE BREAKOUT ALERT**: **{sym}** crossed Pivot High! LTP: ₹{latest_ltp:.2f} ≥ Trigger: ₹{pivot_val:.2f} | 15-Min Vol: {latest_vol:,} (Spike > 1.5x)")
                                                st.info(f"👉 **Execution**: Buy **{shares} shares** of {sym} | Intraday SL: ₹{sl_val:.2f} | Target: ₹{round(pivot_val * 1.3, 2):.2f}")
                                                
                                                st.components.v1.html("""
                                                    <audio autoplay>
                                                        <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
                                                    </audio>
                                                """, height=0)
                                    except Exception:
                                        pass
                                        
                                if breakouts_found == 0:
                                    st.success("✅ Watchlist scan complete. No intraday breakouts triggered at this moment. Standing by.")
                    st.markdown("---")

                # Tab Structure
                tab_horizon, tab_all, tab_turnaround, tab_cyclical, tab_structural, tab_backtest = st.tabs([
                    "⏳ Time Horizon Discovery",
                    "🔍 General Breakout", 
                    "🔄 Turnaround Multibagger", 
                    "📈 Cyclical Value", 
                    "💎 Structural Compounder",
                    "📊 Strategy Backtest Ledger"
                ])
                
                # Define overridden values based on Strategy Mode
                mcap_min = config.MICROCAP_CONFIG["MIN_MCAP"] if is_microcap_mode else market_cap_range[0]
                mcap_max = config.MICROCAP_CONFIG["MAX_MCAP"] if is_microcap_mode else market_cap_range[1]
                vol_min = config.MICROCAP_CONFIG["MIN_VOLUME"] if is_microcap_mode else (config.ARCHETYPES["GENERAL"]["MIN_VOLUME"] if enforce_liquidity else 0.0)

                if is_microcap_mode:
                    st.warning("🚀 **Ground-Floor Microcap Inflection Mode Active**: Enforcing ₹50-1,500 Cr Market Cap, 30k+ volume, and Governance Filters (Promoter >= 45%, Positive CFO). Sizing down to **0.5% allocation** is recommended.")

                with tab_horizon:
                    st.markdown("### ⏳ Time Horizon Discovery Dashboard")
                    st.markdown("Simplify your stock selection by choosing setups matching your trading timeline.")
                    
                    col_h1, col_h2, col_h3 = st.columns(3)
                    
                    with col_h1:
                        st.subheader("⚡ Intraday Watchlist (1-Day)")
                        st.caption("Target quick 2% to 4% momentum moves. Exit before market close.")
                        
                        # Filter for high probability breakout candidates with fast analogue reaction (1-3 days)
                        intra_candidates = scored_df[
                            (scored_df['market_cap'] >= mcap_min) &
                            (scored_df['market_cap'] <= mcap_max) &
                            (scored_df['stage_2_flag'] == 1) &
                            (scored_df['match_days'] <= 3)
                        ].sort_values(by='breakout_prob', ascending=False).head(5).copy()
                        
                        if not intra_candidates.empty:
                            # Populate global symbols
                            for _, r in intra_candidates.iterrows():
                                all_flagged_symbols.add(r['symbol'])
                                global_matched_records[r['symbol']] = r.to_dict()
                                
                            display_intra = intra_candidates[['symbol', 'pivot_high', 'close', 'dynamic_sl', 'risk_review']].copy()
                            display_intra['Trigger Dist (%)'] = ((display_intra['pivot_high'] - display_intra['close']) / display_intra['close'] * 100).round(1).clip(lower=0.0)
                            display_intra['pivot_high'] = display_intra['pivot_high'].round(2)
                            display_intra = display_intra[['symbol', 'pivot_high', 'Trigger Dist (%)', 'dynamic_sl', 'risk_review']]
                            display_intra = display_intra.rename(columns={
                                'symbol': 'Symbol', 'pivot_high': '🎯 Buy GTT (₹)',
                                'dynamic_sl': 'SL (%)', 'risk_review': 'Risk & Strategy Suggestion'
                            })
                            
                            event_intra = st.dataframe(display_intra, use_container_width=True, hide_index=True, on_select="rerun", key="grid_intra", selection_mode="single-row")
                            if event_intra and event_intra.selection and event_intra.selection.rows:
                                idx = event_intra.selection.rows[0]
                                if idx < len(display_intra):
                                    selected_sym = display_intra.iloc[idx]['Symbol']
                                    st.session_state['selected_stock_view'] = selected_sym
                                    st.session_state['selected_stock_context'] = 'intraday'
                                    st.rerun()
                                    
                            st.info("💡 **Execution**: Enable the *Intraday Workstation* toggle below to receive live 15-min alerts when these trigger.")
                        else:
                            st.info("No intraday candidates found.")
                            
                    with col_h2:
                        st.subheader("🎯 Swing Watchlist (7-30 Days)")
                        st.caption("Target a +15% swing return on VCP breakout setups.")
                        
                        # Filter for swing candidates with medium-velocity breakout paths (7-30 days)
                        swing_candidates = tech_filtered_df[
                            (tech_filtered_df['market_cap'] >= mcap_min) &
                            (tech_filtered_df['market_cap'] <= mcap_max) &
                            (tech_filtered_df['match_days'] >= 7) &
                            (tech_filtered_df['match_days'] <= 30)
                        ].sort_values(by='confluence_score', ascending=False).head(5).copy()
                        
                        if not swing_candidates.empty:
                            # Populate global symbols
                            for _, r in swing_candidates.iterrows():
                                all_flagged_symbols.add(r['symbol'])
                                global_matched_records[r['symbol']] = r.to_dict()
                                
                            display_swing = swing_candidates[['symbol', 'pivot_high', 'confluence_score', 'dynamic_sl', 'tentative_days', 'risk_review']].copy()
                            display_swing['pivot_high'] = display_swing['pivot_high'].round(2)
                            display_swing = display_swing.rename(columns={
                                'symbol': 'Symbol', 'pivot_high': '🎯 Buy GTT (₹)',
                                'confluence_score': 'Confluence (%)', 'dynamic_sl': 'SL (%)',
                                'tentative_days': 'Tentative Days', 'risk_review': 'Risk & Strategy Suggestion'
                            })
                            
                            event_swing = st.dataframe(display_swing, use_container_width=True, hide_index=True, on_select="rerun", key="grid_swing", selection_mode="single-row")
                            if event_swing and event_swing.selection and event_swing.selection.rows:
                                idx = event_swing.selection.rows[0]
                                if idx < len(display_swing):
                                    selected_sym = display_swing.iloc[idx]['Symbol']
                                    st.session_state['selected_stock_view'] = selected_sym
                                    st.session_state['selected_stock_context'] = 'swing'
                                    st.rerun()
                        else:
                            st.info("No swing candidates found.")
                            
                    with col_h3:
                        st.subheader("💎 Long-Term Watch (30-250 Days)")
                        st.caption("Hold elite compounders; exit only below major trendlines.")
                        
                        # Filter for long-term compounders with structural trends (30-250 days)
                        long_candidates = scored_df[
                            (scored_df['market_cap'] >= mcap_min) &
                            (scored_df['market_cap'] <= mcap_max) &
                            (scored_df['roce'] >= 0.18) &
                            (scored_df['debt_to_equity'] <= 0.5) &
                            (scored_df['stage_2_flag'] == 1) &
                            (scored_df['match_days'] >= 30) &
                            (scored_df['match_days'] <= 250)
                        ].sort_values(by='confluence_score', ascending=False).head(5).copy()
                        
                        if not long_candidates.empty:
                            # Populate global symbols
                            for _, r in long_candidates.iterrows():
                                all_flagged_symbols.add(r['symbol'])
                                global_matched_records[r['symbol']] = r.to_dict()
                                
                            display_long = long_candidates[['symbol', 'roce', 'debt_to_equity', 'dynamic_sl', 'risk_review']].copy()
                            display_long['roce'] = (display_long['roce'] * 100).round(1).astype(str) + "%"
                            display_long['debt_to_equity'] = display_long['debt_to_equity'].round(2)
                            display_long = display_long.rename(columns={
                                'symbol': 'Symbol', 'roce': 'ROCE (%)',
                                'debt_to_equity': 'Debt-to-Equity', 'dynamic_sl': 'SL (%)',
                                'risk_review': 'Risk & Strategy Suggestion'
                            })
                            
                            event_long = st.dataframe(display_long, use_container_width=True, hide_index=True, on_select="rerun", key="grid_long", selection_mode="single-row")
                            if event_long and event_long.selection and event_long.selection.rows:
                                idx = event_long.selection.rows[0]
                                if idx < len(display_long):
                                    selected_sym = display_long.iloc[idx]['Symbol']
                                    st.session_state['selected_stock_view'] = selected_sym
                                    st.session_state['selected_stock_context'] = 'long_term'
                                    st.rerun()
                                    
                            st.info("💡 **Exit Rule**: Exit if the daily close drops below the 50D SMA or 200D SMA.")
                        else:
                            st.info("No long-term compounders found.")

                with tab_all:
                    st.markdown("### 🔍 General Breakout Screener")
                    cfg_g = config.ARCHETYPES["GENERAL"]
                    g_df = tech_filtered_df[
                        (tech_filtered_df['market_cap'] >= mcap_min) & 
                        (tech_filtered_df['market_cap'] <= mcap_max)
                    ].copy()
                    
                    if enforce_stage2:
                        g_df = g_df[g_df['stage_2_flag'] == 1]
                    if enforce_liquidity:
                        g_df = g_df[g_df['vol_50d_avg'] >= vol_min]
                    if enforce_quality:
                        g_df = g_df[
                            (g_df['roce'] >= cfg_g['MIN_ROCE']) & 
                            (g_df['debt_to_equity'] <= cfg_g['MAX_DEBT'])
                        ]
                    if enforce_valuation:
                        g_df = g_df[
                            (g_df['peg_ratio'] > 0.0) & 
                            (g_df['peg_ratio'] <= cfg_g['MAX_PEG'])
                        ]
                    show_archetype_dataframe(g_df, "General Breakout", confluence_col='confluence_general')
                    
                with tab_turnaround:
                    cfg_t = config.ARCHETYPES["TURNAROUND"]
                    st.markdown("### 🔄 Turnaround Multibaggers")
                    st.markdown("Identify distressed or out-of-favor companies emerging from losses, cutting debt, or recovering margins.")
                    st.markdown(rf"*(Rules: Cap ₹{int(cfg_t['MIN_MCAP'])}Cr - ₹{int(cfg_t['MAX_MCAP'])}Cr | Debt to Equity $\le$ {cfg_t['MAX_DEBT']} | Rev Growth $\ge$ {int(cfg_t['MIN_REV_GROWTH']*100)}% | ROCE/ROE $\ge$ {int(cfg_t['MIN_ROCE']*100)}% | Volume Spike $\ge$ {cfg_t['MIN_VOL_SURGE']}x)*")
                    t_mcap_min = mcap_min if is_microcap_mode else cfg_t['MIN_MCAP']
                    t_mcap_max = mcap_max if is_microcap_mode else cfg_t['MAX_MCAP']
                    
                    t_df = scored_df[
                        (scored_df['market_cap'] >= t_mcap_min) & 
                        (scored_df['market_cap'] <= t_mcap_max) &
                        (scored_df['debt_to_equity'] <= cfg_t['MAX_DEBT']) &
                        (scored_df['revenue_growth'] >= cfg_t['MIN_REV_GROWTH']) &
                        (scored_df['roce'] >= cfg_t['MIN_ROCE']) &
                        (scored_df['volume_surge_score'] >= cfg_t['MIN_VOL_SURGE'])
                    ].copy()
                    if enforce_liquidity:
                        t_df = t_df[t_df['vol_50d_avg'] >= vol_min]
                    show_archetype_dataframe(t_df, "Turnaround Multibagger", confluence_col='confluence_turnaround')
                    
                with tab_cyclical:
                    cfg_c = config.ARCHETYPES["CYCLICAL"]
                    st.markdown("### 📈 Cyclical Deep Value Plays")
                    st.markdown("Industrial/commodity assets at the trough of the global cycle experiencing a violent quarterly profit inflection.")
                    st.markdown(rf"*(Rules: Price-to-Book $\le$ {cfg_c['MAX_PB']} | YoY Profit Var $\ge$ {int(cfg_c['MIN_EARNINGS_GROWTH']*100)}% | Interest Coverage $\ge$ {cfg_c['MIN_ICR']} | ROCE/ROE $\ge$ {int(cfg_c['MIN_ROCE']*100)}%)*")
                    c_df = scored_df[
                        (scored_df['price_to_book'] > 0.0) & 
                        (scored_df['price_to_book'] <= cfg_c['MAX_PB']) &
                        (scored_df['earnings_growth'] >= cfg_c['MIN_EARNINGS_GROWTH']) &
                        (scored_df['interest_coverage'] >= cfg_c['MIN_ICR']) &
                        (scored_df['roce'] >= cfg_c['MIN_ROCE'])
                    ].copy()
                    if enforce_liquidity:
                        c_df = c_df[c_df['vol_50d_avg'] >= 200000.0]
                    show_archetype_dataframe(c_df, "Cyclical Deep Value", confluence_col='confluence_cyclical')
                    
                with tab_structural:
                    cfg_s = config.ARCHETYPES["STRUCTURAL"]
                    st.markdown("### 💎 Structural Growth Compounders")
                    st.markdown("Elite wealth compounders with strong moats, high pricing power, clean balance sheets, and consistent growth.")
                    st.markdown(rf"*(Rules: ROCE/ROE $\ge$ {int(cfg_s['MIN_ROCE']*100)}% | Debt-to-Equity $\le$ {cfg_s['MAX_DEBT']} | Rev Growth $\ge$ {int(cfg_s['MIN_REV_GROWTH']*100)}% | Stage 2 Uptrend Enforced)*")
                    s_df = tech_filtered_df[
                        (tech_filtered_df['market_cap'] >= mcap_min) &
                        (tech_filtered_df['market_cap'] <= mcap_max) &
                        (tech_filtered_df['roce'] >= cfg_s['MIN_ROCE']) & 
                        (tech_filtered_df['debt_to_equity'] <= cfg_s['MAX_DEBT']) &
                        (tech_filtered_df['revenue_growth'] >= cfg_s['MIN_REV_GROWTH']) &
                        (tech_filtered_df['stage_2_flag'] == 1)
                    ].copy()
                    if enforce_liquidity:
                        s_df = s_df[s_df['vol_50d_avg'] >= vol_min]
                    show_archetype_dataframe(s_df, "Structural Growth Compounder")
                
                with tab_backtest:
                    st.markdown("### 📊 Strategy Backtest & Win-Rate Ledger")
                    st.markdown("Evaluate the historical hit-rate and sizing statistics for setups exceeding your Confluence Score threshold.")
                    
                    thresh_pct = int(config.BREAKOUT_LABEL_THRESHOLD * 100)
                    st.caption(f"💡 **Active Model Training Mode:** `≥+{thresh_pct}% 20-Day Close-to-Close Return`. You can change this mode (20%, 30%, 50%) in the sidebar & retrain the model to compare scenarios.")
                    
                    # Cutoff Slider
                    backtest_cutoff = st.slider(
                        "Confluence Cutoff Threshold (%)",
                        min_value=70.0, max_value=95.0, value=85.0, step=1.0,
                        help="Select the minimum Confluence Score to count as a signal."
                    )
                    
                    # Lookahead Bias Toggle Option
                    backtest_include_funds = st.checkbox(
                        "🔬 Include Fundamental Filters (Lookahead Proxy)",
                        value=False,
                        help="Enable this to apply F-Score fundamentals filter to the backtester. Note: uses current fundamentals as a structural proxy, introducing minor lookahead bias. Disable for a pure, 100% bias-free technical backtest."
                    )
                    
                    @st.cache_data(ttl=3600)
                    def get_cached_backtest(cutoff, sl_pct, tp_pct, include_funds):
                        from engine.backtest_ledger import run_backtest_ledger
                        return run_backtest_ledger(cutoff, sl_pct, tp_pct, include_funds)
                        
                    with st.spinner("Executing historical backtest..."):
                        metrics_b, trades_df_b, arch_df_b = get_cached_backtest(
                            backtest_cutoff, stop_loss_pct, target_profit_pct, backtest_include_funds
                        )
                        
                    if not metrics_b or trades_df_b.empty:
                        st.warning("⚠️ No historical signals met this high confluence threshold in the database.")
                    else:
                        # Metrics Row 1: Win-Rate (split into full-history vs out-of-sample)
                        col_b1, col_b1b, col_b2, col_b3, col_b4 = st.columns(5)
                        with col_b1:
                            oos_delta = None
                            if metrics_b['oos_win_rate'] is not None:
                                oos_delta = f"OOS ({metrics_b['oos_decisive_count']} signals): {metrics_b['oos_win_rate']}%"
                            st.metric(
                                f"Win-Rate — Full History",
                                f"{metrics_b['win_rate']}%",
                                delta=oos_delta,
                                help=f"Hit-rate across all history. Includes IN-SAMPLE signals (pre-6mo) where the XGBoost model already saw the data during training — these will be optimistically high. Compare against the OOS rate."
                            )
                        with col_b1b:
                            if metrics_b['oos_win_rate'] is not None:
                                st.metric(
                                    f"Win-Rate — OOS (Last 6 Months)",
                                    f"{metrics_b['oos_win_rate']}%",
                                    delta=f"{metrics_b['oos_decisive_count']} decisive signals",
                                    help=f"Hit-rate on signals from the last 6 months only. These were in the model's HELD-OUT TEST SET and never seen during training — this is the genuine out-of-sample performance number to trust."
                                )
                            else:
                                st.metric("Win-Rate — OOS (Last 6 Months)", "N/A", help="No decisive signals in the last 6 months at this threshold.")
                        with col_b2:
                            st.metric(
                                "Total Signals Logged",
                                f"{metrics_b['total_flags']}",
                                f"{metrics_b['decisive_count']} decisive / {metrics_b['time_exit_count']} time-exits / {metrics_b['active_count']} active",
                                help="Decisive = hit target or stop. Time-exits = expired at Day 20. Active = still in the 20-day window."
                            )
                        with col_b3:
                            st.metric("Avg Return on Success", f"+{metrics_b['avg_gain']}%", help="Average return achieved by successful trades.")
                        with col_b4:
                            st.metric("Avg Return on Failures", f"{metrics_b['avg_loss']}%", help=f"Average return of failed trades (combination of stopped-out trades and time-exits on Day 20.")
                            
                        # Sizing Metrics
                        col_b5, col_b6, col_b7, col_b8 = st.columns(4)
                        with col_b5:
                            st.metric("Profit Factor", f"{metrics_b['profit_factor']}x", help="Ratio of cumulative percentage gains on winning trades to cumulative percentage losses on losing trades.")
                        with col_b6:
                            st.metric("Payoff Ratio (W/L Size)", f"{metrics_b['payoff_ratio']}x", help="Ratio of average winning trade size to average losing trade size (Risk-Reward payoff expectancy).")
                        with col_b7:
                            st.metric("Kelly Allocation Size", f"{metrics_b['kelly_pct']}%", help="Suggested allocation percentage of total capital per trade to maximize growth without risk of ruin.")
                        with col_b8:
                            st.metric("Avg Days to Target", f"{metrics_b.get('avg_days_to_target', 0.0)} Days", help="Average number of trading days it takes for successful breakout setups to reach the profit target.")
                            
                        # Archetype Win-rate Breakdown
                        st.markdown("#### 📈 Performance Breakdown by Archetype Group")
                        display_arch = arch_df_b.copy()
                        emoji_map = {
                            "Structural Compounder": "💎 Structural Compounder",
                            "Turnaround Multibagger": "🔄 Turnaround Multibagger",
                            "Cyclical Deep Value": "📈 Cyclical Deep Value",
                            "General Breakout": "🔍 General Breakout"
                        }
                        display_arch['Archetype'] = display_arch['Archetype'].map(emoji_map).fillna(display_arch['Archetype'])
                        st.dataframe(display_arch, use_container_width=True, hide_index=True)
                        
                        # Detailed Trade Log
                        st.markdown("#### 📋 Completed and Active Trade Logs")
                        display_trades = trades_df_b.copy()
                        display_trades['Confluence (%)'] = display_trades['Confluence (%)'].round(1)
                        display_trades['XGBoost Prob (%)'] = display_trades['XGBoost Prob (%)'].round(1)
                        display_trades['Status'] = display_trades['Status'].apply(
                            lambda x: "🟢 TARGET HIT" if x == "SUCCESS" else "🔴 STOP HIT" if x == "FAILURE" else "🟠 TIME EXIT" if x == "TIME_EXIT" else "🟡 ACTIVE"
                        )
                        st.dataframe(display_trades, use_container_width=True, hide_index=True)
                
                st.session_state['matched_records'] = global_matched_records
                st.session_state['flagged_symbols'] = sorted(list(all_flagged_symbols))
        except Exception as e:
            st.error(f"Could not load real data from database: {e}")
            import traceback
            st.write(traceback.format_exc())
        
        st.subheader("Historical Evidence & Risk Management Viewer")
        flagged_syms = st.session_state.get('flagged_symbols', [])
        selected_stock = None
        if flagged_syms:
            # Sync selectbox with table row click
            default_stock = st.session_state.get('selected_stock_view')
            default_idx = 0
            if default_stock in flagged_syms:
                default_idx = flagged_syms.index(default_stock)
            selected_stock = st.selectbox(
                "Select a stock to view historical matches:", 
                flagged_syms, 
                index=default_idx,
                key="selected_stock_selectbox"
            )
            st.session_state['selected_stock_view'] = selected_stock
            
            # Catalyst & News Intelligence Expander
            with st.expander("🗞️ Catalyst & News Intelligence (Live Feed)"):
                try:
                    news_list = fetch_stock_news(selected_stock)
                    
                    if news_list:
                        for item in news_list[:4]:  # Show top 4 articles
                            content = item.get('content', {})
                            title = content.get('title', 'No Title')
                            summary = content.get('summary', '')
                            pub_time = content.get('displayTime', '')
                            provider = content.get('provider', {}).get('displayName', 'Unknown')
                            url = content.get('clickThroughUrl', {}).get('url', '#')
                            
                            st.markdown(f"##### [{title}]({url})")
                            st.markdown(f"*{provider} | {pub_time}*")
                            if summary:
                                st.markdown(f"> {summary}")
                            st.markdown("---")
                    else:
                        st.info("No recent news articles found for this stock on Yahoo Finance.")
                except Exception as ex:
                    st.warning(f"Could not load news feed: {ex}")
        else:
            st.warning("⚠️ No stocks matched search filters across any of the archetype tabs. Try loosening parameters.")
        
        st.write(f"DEBUG: selected_stock = {selected_stock}")
        st.write(f"DEBUG: matched_records in session_state = {'matched_records' in st.session_state}")
        if 'matched_records' in st.session_state:
            st.write(f"DEBUG: keys in matched_records = {list(st.session_state['matched_records'].keys())}")
        st.write(f"DEBUG: scored_df in session_state = {'scored_df' in st.session_state}")

        if selected_stock:
            try:
                record = None
                if 'matched_records' in st.session_state:
                    record = st.session_state['matched_records'].get(selected_stock)
                
                if not record and 'scored_df' in st.session_state:
                    sdf = st.session_state['scored_df']
                    match_rows = sdf[sdf['symbol'] == selected_stock]
                    if not match_rows.empty:
                        record = match_rows.iloc[0].to_dict()
                
                if record:
                    match_sym = record['match_symbol']
                    match_dt = record['match_date']
                    match_ret = record['match_return']
                    sim_score = record['similarity_score']
                    match_days = record.get('match_days', 20)
                    
                    conn = get_connection()
                    # Get last 250 trading days for the CURRENT stock
                    query = """
                    SELECT timestamp, close, volume
                    FROM price_history
                    WHERE instrument_key = (SELECT instrument_key FROM stocks WHERE symbol = ? LIMIT 1)
                    ORDER BY timestamp DESC
                    LIMIT 250
                    """
                    chart_df = pd.read_sql_query(query, conn, params=(selected_stock,))
                    conn.close()
                    
                    if not chart_df.empty:
                        # Chart 1: Current Stock Price & Volume
                        chart_df = chart_df.sort_values('timestamp')
                        chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp']).dt.date
                        chart_df.set_index('timestamp', inplace=True)
                        chart_df['50-Day MA'] = chart_df['close'].rolling(window=50).mean()
                        
                        # Fetch current price for risk metrics
                        current_price = chart_df.iloc[-1]['close']
                        
                        # Dynamically calculated stop loss and profit targets based on context
                        context = st.session_state.get('selected_stock_context', 'swing')
                        
                        sma_50_raw = record.get('sma_50')
                        sma_50_val = sma_50_raw if not pd.isna(sma_50_raw) else 0.0
                        sma_150_raw = record.get('sma_150')
                        sma_150_val = sma_150_raw if not pd.isna(sma_150_raw) else 0.0
                        sma_200_raw = record.get('sma_200')
                        sma_200_val = sma_200_raw if not pd.isna(sma_200_raw) else 0.0
                        
                        # Chart 2: Analogue (Historical match)
                        analogue_df = fetch_analogue_data(match_sym, match_dt)
                        
                        if context == 'intraday':
                            current_sl = 1.5
                            current_tp = 3.0
                            context_title = "⚡ Intraday Scalping Workspace"
                            context_advice = f"⚡ **Intraday Strategy**: Target quick +3.0% scalp (**Rs. {current_price * 1.03:.2f}**) with a tight 1.5% Stop Loss (**Rs. {current_price * 0.985:.2f}**). Sell 100% of your position before market close (3:30 PM IST)."
                            st.info(context_advice)
                        elif context in ['long_term', 'structural growth compounder', 'structural growth compounder ']:
                            current_sl = record.get('dynamic_sl', 15.0)
                            current_tp = 30.0
                            context_title = "💎 Long-Term Wealth Compounder Workspace"
                            context_advice = f"💎 **Long-Term Strategy**: Buy breakouts. Target +30.0% swing exit (**Rs. {current_price * 1.30:.2f}**) on 50% of the shares. Hold the remaining 50% runner indefinitely, trailing exits below the 50D SMA (**Rs. {sma_50_val:.1f}**) or 200D SMA (**Rs. {sma_200_val:.1f}**)."
                            st.success(context_advice)
                        else:
                            current_sl = record.get('dynamic_sl', stop_loss_pct)
                            current_tp = target_profit_pct
                            context_title = "🎯 Swing Breakout Workspace"
                            context_advice = f"🎯 **Swing Strategy**: Target +{current_tp}% (**Rs. {current_price * (1.0 + current_tp/100.0):.2f}**) with a volatility-adjusted SL of {current_sl}% (**Rs. {current_price * (1.0 - current_sl/100.0):.2f}**). Lock in profits on 50% of the shares, and trail the rest."
                            st.warning(context_advice)
                            
                        stop_loss_mult = 1.0 - (current_sl / 100.0)
                        target_mult = 1.0 + (current_tp / 100.0)
                        
                        stop_loss_price = current_price * stop_loss_mult
                        target_price = current_price * target_mult
                        
                        # Show risk management box
                        st.markdown(f"### 🛡️ {context_title}")
                        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                        with col_r1:
                            st.metric("Pivot Entry Price", f"Rs. {current_price:.2f}")
                        with col_r2:
                            st.metric(f"Dynamic Initial Stop ({current_sl}%)", f"Rs. {stop_loss_price:.2f}", f"-{current_sl}%", delta_color="inverse")
                        with col_r3:
                            st.metric(f"Timeframe Target ({current_tp}%)", f"Rs. {target_price:.2f}", f"+{current_tp}%")
                        with col_r4:
                            st.metric(
                                "Runner Stop (50D / 200D)", 
                                f"Rs. {sma_50_val:.1f} / {sma_200_val:.1f}", 
                                "UNCAPPED Target", 
                                help="Book profit on 50% of the position at the swing target. Leave the remaining 50% uncapped as a runner, trailing the 50D or 200D SMA close."
                            )
                        
                        # 6. Auction Market Volume Profile Analysis
                        poc_price = record.get('volume_node_poc')
                        density_val = record.get('volume_node_density')
                        
                        is_extended = False
                        if sma_50_val > 0.0 and (current_price / sma_50_val) > 1.20:
                            is_extended = True
                            st.error(f"⚠️ **OVER-EXTENDED WARNING**: This stock is trading {(current_price/sma_50_val - 1.0)*100:.1f}% above its 50-day moving average. Buying breakouts here carries extreme pullback risk. We strongly recommend reducing your risk sizing.")
                        
                        pbd_shape = record.get('pbd_profile')
                        
                        if pbd_shape == 'b':
                            st.error("🚨 **BEARISH DISTRIBUTION STRUCTURE**: This stock exhibits a **b-Profile** shape, meaning the majority of its volume occurred at the bottom of its range after a drop. Buying breakouts here has a very low historical success rate. Sizing down to a minimum (0.1% risk) is recommended.")
                        
                        if poc_price is not None and density_val is not None:
                            st.markdown("### 📊 Auction Market Volume Profile (30-Day)")
                            col_vp1, col_vp2, col_vp3, col_vp4 = st.columns(4)
                            with col_vp1:
                                st.metric("Point of Control (POC)", f"Rs. {poc_price:.2f}", help="The price level where the maximum volume occurred over the last 30 trading days.")
                            with col_vp2:
                                density_desc = "Heavy Accumulation" if density_val >= 50.0 else "Light Trading"
                                st.metric("Volume Area Density", f"{density_val:.1f}%", density_desc)
                            with col_vp3:
                                dist_from_poc = ((current_price - poc_price) / poc_price) * 100
                                if dist_from_poc > 15.0:
                                    st.metric("Dist. to Base", f"+{dist_from_poc:.1f}%", "Chase Risk (High)", delta_color="inverse")
                                else:
                                    st.metric("Dist. to Base", f"+{dist_from_poc:.1f}%", "Within Base (Low)")
                            with col_vp4:
                                if pbd_shape == 'P':
                                    shape_desc = "Accumulation (Bull)"
                                elif pbd_shape == 'b':
                                    shape_desc = "Distribution (Bear)"
                                else:
                                    shape_desc = "Balance (Neutral)"
                                st.metric("PbD Profile Shape", f"{pbd_shape}-Profile", shape_desc)
                        
                        # Volatility-Adjusted Position Sizer
                        st.markdown("### 🧮 Volatility-Adjusted Position Sizer")
                        col_sz1, col_sz2, col_sz3 = st.columns(3)
                        with col_sz1:
                            total_cap = st.number_input("Total Trading Capital (Rs.)", min_value=10000.0, value=500000.0, step=10000.0, key="sizer_cap")
                        with col_sz2:
                            # Dynamic capital risk scaling based on extensions/microcaps/bearish profiles
                            default_risk = 0.1 if pbd_shape == 'b' else (0.25 if is_extended else (0.5 if record.get('market_cap', 1000.0) < 1500.0 else 1.0))
                            risk_pct = st.slider("Max Capital Risk per Trade (%)", min_value=0.1, max_value=5.0, value=default_risk, step=0.1, key="sizer_risk")
                        with col_sz3:
                            rupee_risk = total_cap * (risk_pct / 100.0)
                            price_diff = current_price * (current_sl / 100.0)
                            shares_to_buy = int(rupee_risk / price_diff) if price_diff > 0 else 0
                            st.metric("Suggested Shares to Buy", f"{shares_to_buy:,} Shares", f"Risk: Rs. {rupee_risk:,.0f} ({risk_pct}%)")
                            
                        st.markdown(f"**Position Capital Required**: Rs. {shares_to_buy * current_price:,.2f} | **Worst Case Loss**: Rs. {shares_to_buy * price_diff:,.2f}")
                            
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            corp_name = record.get('name', '')
                            full_title = f"{selected_stock} — {corp_name}" if corp_name else selected_stock
                            st.subheader(f"📊 Current Stock: {full_title}")
                            st.markdown(f"**Close Price & 50-day Moving Average**")
                            st.line_chart(chart_df[['close', '50-Day MA']])
                            st.markdown(f"**Trading Volume**")
                            st.bar_chart(chart_df['volume'])
                            
                        with col2:
                            st.subheader(f"⏳ Historical Match: {match_sym}")
                            st.info(f"💡 **{selected_stock}** has a **{sim_score}% similarity** to **{match_sym}** on **{match_dt}**. Subsequent to this setup, **{match_sym}** reached its target, rallying **+{match_ret}%** in **{int(match_days)} trading days**.")
                            
                            if analogue_df is not None and not analogue_df.empty:
                                # Checkbox for Ghost Chart
                                show_ghost = st.checkbox("Show Ghost Chart Projection (T+20)", value=True, help="Overlay the historical match's subsequent 20-day return path projected onto the current close price.")
                                
                                if show_ghost:
                                    try:
                                        temp_df = analogue_df.reset_index()
                                        match_date_obj = pd.to_datetime(match_dt).date()
                                        temp_df['date_diff'] = temp_df['timestamp'].apply(lambda x: abs((x - match_date_obj).days))
                                        closest_match_row_idx = temp_df['date_diff'].idxmin()
                                        
                                        base_price = temp_df.iloc[closest_match_row_idx]['close']
                                        subsequent_days = temp_df.iloc[closest_match_row_idx:closest_match_row_idx+21].copy()
                                        
                                        if len(subsequent_days) > 1:
                                            subsequent_days['pct_return'] = ((subsequent_days['close'] - base_price) / base_price) * 100
                                            subsequent_days['projected_price'] = current_price * (1.0 + subsequent_days['pct_return'] / 100.0)
                                            subsequent_days['Trading Day'] = [f"T+{i}" for i in range(len(subsequent_days))]
                                            subsequent_days.set_index('Trading Day', inplace=True)
                                            
                                            st.markdown(f"📈 **Ghost Chart Projection (Target: Rs. {current_price * (1.0 + match_ret/100.0):.2f})**")
                                            st.line_chart(subsequent_days['projected_price'])
                                    except Exception as ex:
                                        st.error(f"Could not construct Ghost Chart projection: {ex}")
                                
                                st.markdown(f"**Historical Close Price (Window around Breakout Date {match_dt})**")
                                st.line_chart(analogue_df['close'])
                                st.markdown(f"**Historical Trading Volume**")
                                st.bar_chart(analogue_df['volume'])
                            else:
                                st.warning("Historical price data for matching analogue not found in database.")
                    else:
                        st.warning("No historical data available for plotting.")
            except Exception as e:
                st.error(f"Error loading charts: {e}")
                
else:
    st.info("👈 Tweak your parameters in the sidebar and click **Run Screener** to start hunting.")
