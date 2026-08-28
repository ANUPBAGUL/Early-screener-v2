import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
import xgboost as xgb

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
DB_PATH = os.path.join("database", "screener.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

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
        candles = client.fetch_historical_candles("NSE_INDEX|Nifty 50", interval="month")
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

def fetch_analogue_data(symbol, match_date):
    """
    Fetches price history for the historical match and slices it around the breakout date.
    """
    conn = get_connection()
    query = """
    SELECT p.timestamp, p.close, p.volume
    FROM price_history p
    JOIN stocks s ON p.instrument_key = s.instrument_key
    WHERE s.symbol = ?
    ORDER BY p.timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()
    
    if df.empty:
        return None
        
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
    return sliced_df[['close', 'volume']]

# --- Try Loading XGBoost Model ---
xgb_loaded = False
try:
    model_path = os.path.join("engine", "breakout_model.json")
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
    if st.button("Retrain XGBoost Classifier", use_container_width=True):
        with st.spinner("Training model & extracting DNA..."):
            try:
                from engine.breakout_model import train_and_save_model
                train_and_save_model()
                st.success("✅ XGBoost model retrained and DNA library updated!")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")
                
    st.markdown("---")
    st.header("⚙️ Screener Parameters")
    
    st.subheader("Momentum & Breakout Settings")
    vol_surge = st.slider(
        "Volume Surge Threshold (x Avg)", 
        min_value=1.0, max_value=10.0, value=3.0, step=0.5,
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
    market_cap_range = st.slider(
        "Market Cap Range (₹ Crores)",
        min_value=100.0, max_value=50000.0, value=(300.0, 10000.0), step=100.0,
        help="Configure the size constraint to target small-to-midcap stocks (default ₹300 cr to ₹10,000 cr)."
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
    
    bypass_kill_switch = False
    if market_status is not None:
        is_bull, nifty_close, nifty_ema = market_status
        if not is_bull:
            bypass_kill_switch = st.checkbox(
                "Bypass Market Kill Switch", 
                value=False,
                help="Enable screening even when Nifty 50 trades below its Monthly 10 EMA."
            )
            
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
        st.error(f"🔴 **Bear Market Kill Switch Active**: Nifty 50 is trading below its Monthly 10 EMA (Close: {nifty_close:.1f} vs EMA: {nifty_ema:.1f}). Long breakout setups have high failure rates. Signals paused by default.")

if st.session_state.get('run_screener', False):
    
    # 1. Market Kill Switch Pause Check
    if market_status is not None and not is_bull and not bypass_kill_switch:
        st.warning("⛔ Screener results paused due to active Market Kill Switch. Check 'Bypass Market Kill Switch' in the sidebar to bypass.")
        real_data = pd.DataFrame()
    else:
        st.success(f"Scanning stocks for >{vol_surge}x volume surges and <{vcp_tightness}% VCP tightness...")
        
        # Fetch real data
        real_data = None
        try:
            features_df = fetch_latest_features()
            if not features_df.empty:
                # A. Compute cosine similarity against DNA library
                scored_df = calculate_similarity_score(features_df)
                
                # B. Run XGBoost classifier probability if loaded
                if xgb_loaded:
                    feature_cols = ['volatility_contraction_score', 'volume_surge_score', 'momentum_score']
                    probs = model.predict_proba(scored_df[feature_cols])[:, 1]
                    scored_df['breakout_prob'] = (probs * 100).round(1)
                else:
                    scored_df['breakout_prob'] = 0.0
                    
                # C. Apply standard user sliders (Volume Surge & VCP)
                filtered_df = scored_df[
                    (scored_df['volume_surge_score'] >= vol_surge) & 
                    (scored_df['volatility_contraction_score'] <= vcp_tightness) &
                    (scored_df['similarity_score'] >= similarity_threshold)
                ].copy()
                
                # D. Apply Size (Market Cap Range) Filter
                filtered_df = filtered_df[
                    (filtered_df['market_cap'] >= market_cap_range[0]) & 
                    (filtered_df['market_cap'] <= market_cap_range[1])
                ]
                
                # E. Apply Minervini Stage 2 Uptrend template
                if enforce_stage2:
                    filtered_df = filtered_df[filtered_df['stage_2_flag'] == 1]
                    
                # F. Apply Institutional Liquidity Filters (ADV >= 200k)
                if enforce_liquidity:
                    filtered_df = filtered_df[filtered_df['vol_50d_avg'] >= 200000.0]
                    
                # G. Apply Capital Efficiency (18% ROCE/ROE) and Leverage Filters
                if enforce_quality:
                    filtered_df = filtered_df[
                        (filtered_df['roce'] >= 0.18) & 
                        (filtered_df['debt_to_equity'] <= 0.5)
                    ]
                    
                # H. Apply Valuation PEG Filter
                if enforce_valuation:
                    filtered_df = filtered_df[
                        (filtered_df['peg_ratio'] > 0.0) & 
                        (filtered_df['peg_ratio'] <= 1.0)
                    ]
                
                # Sort by similarity score
                real_data = filtered_df.sort_values(by='similarity_score', ascending=False)
                
                # Format UI Action
                real_data['Action'] = real_data['similarity_score'].apply(
                    lambda x: "🚨 Parabolic" if x > 90 else "🔥 High Conviction" if x > 75 else "⚡ Breakout Alert"
                )
                
                # Store records in session state
                st.session_state['matched_records'] = real_data.set_index('symbol').to_dict('index')
                
                # Format main table columns
                real_data = real_data[[
                    'symbol', 'volume_surge_score', 'volatility_contraction_score', 'market_cap', 'vol_50d_avg',
                    'roce', 'debt_to_equity', 'peg_ratio', 'similarity_score', 'breakout_prob', 
                    'match_symbol', 'match_date', 'match_return', 'Action'
                ]]
                real_data['volume_surge_score'] = real_data['volume_surge_score'].round(2)
                real_data['volatility_contraction_score'] = real_data['volatility_contraction_score'].round(2)
                real_data['market_cap'] = real_data['market_cap'].round(1)
                real_data['vol_50d_avg'] = real_data['vol_50d_avg'].astype(int)
                real_data['roce'] = (real_data['roce'] * 100).round(1).astype(str) + "%"
                real_data['debt_to_equity'] = real_data['debt_to_equity'].round(2)
                real_data['peg_ratio'] = real_data['peg_ratio'].round(2)
                
                real_data.columns = [
                    "Symbol", "Volume Surge (x)", "VCP Tightness (%)", "Market Cap (Cr)", "50D Avg Vol",
                    "ROCE/ROE (%)", "Debt-to-Equity", "PEG Ratio", "Similarity Score (%)", "Breakout Prob (%)", 
                    "Closest Match", "Match Date", "Hist. Return (%)", "Action"
                ]
        except Exception as e:
            st.error(f"Could not load real data from database: {e}")
            import traceback
            st.write(traceback.format_exc())
            
    if real_data is not None and not real_data.empty:
        display_data = real_data
    else:
        if real_data is not None and real_data.empty:
            st.warning("⚠️ No stocks matched your search filters. Try loosening parameters or disabling filters.")
        else:
            st.warning("⚠️ Using mock data! The database is empty or the API pipeline hasn't been run yet.")
        display_data = None
        
    if display_data is not None:
        st.subheader(f"Top Screener Results ({len(display_data)} Stocks Flagged)")
        st.dataframe(display_data, use_container_width=True, hide_index=True)
        
        st.subheader("Historical Evidence & Risk Management Viewer")
        selected_stock = st.selectbox("Select a stock to view historical matches:", display_data["Symbol"].tolist())
        
        if selected_stock and 'matched_records' in st.session_state:
            try:
                record = st.session_state['matched_records'].get(selected_stock)
                
                if record:
                    match_sym = record['match_symbol']
                    match_dt = record['match_date']
                    match_ret = record['match_return']
                    sim_score = record['similarity_score']
                    
                    conn = get_connection()
                    # Get last 250 trading days for the CURRENT stock
                    query = """
                    SELECT p.timestamp, p.close, p.volume
                    FROM price_history p
                    JOIN stocks s ON p.instrument_key = s.instrument_key
                    WHERE s.symbol = ?
                    ORDER BY p.timestamp DESC
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
                        stop_loss_price = current_price * 0.925 # 7.5% stop loss
                        target_price = current_price * 1.30     # 30% target profit
                        
                        sma_50_raw = record.get('sma_50')
                        sma_50_val = sma_50_raw if sma_50_raw is not None else 0.0
                        sma_150_raw = record.get('sma_150')
                        sma_150_val = sma_150_raw if sma_150_raw is not None else 0.0
                        
                        # Chart 2: Analogue (Historical match)
                        analogue_df = fetch_analogue_data(match_sym, match_dt)
                        
                        # Show risk management box
                        st.markdown("### 🛡️ Hybrid Breakout Risk Management (Dual-Allocation)")
                        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                        with col_r1:
                            st.metric("Pivot Entry Price", f"Rs. {current_price:.2f}")
                        with col_r2:
                            st.metric("Strict Initial Stop (7.5%)", f"Rs. {stop_loss_price:.2f}", "-7.5%", delta_color="inverse")
                        with col_r3:
                            st.metric("Swing Exit (50% position)", f"Rs. {target_price:.2f}", "+30.0%")
                        with col_r4:
                            st.metric(
                                "Runner Stop (50D / 150D)", 
                                f"Rs. {sma_50_val:.1f} / {sma_150_val:.1f}", 
                                "UNCAPPED Target", 
                                help="Book profit on 50% of the position at +30%. Leave the remaining 50% uncapped as a runner, exiting only if the daily close falls below the 50-day or 150-day SMA."
                            )
                            
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader(f"📊 Current Stock: {selected_stock}")
                            st.markdown(f"**Close Price & 50-day Moving Average**")
                            st.line_chart(chart_df[['close', '50-Day MA']])
                            st.markdown(f"**Trading Volume**")
                            st.bar_chart(chart_df['volume'])
                            
                        with col2:
                            st.subheader(f"⏳ Historical Match: {match_sym}")
                            st.info(f"💡 **{selected_stock}** has a **{sim_score}% similarity** to **{match_sym}** on **{match_dt}**. Subsequent to this setup, **{match_sym}** rallied **+{match_ret}%** over the next 20 trading days.")
                            
                            if analogue_df is not None and not analogue_df.empty:
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
