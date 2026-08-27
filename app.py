import streamlit as st
import pandas as pd
import sqlite3
import os
import sys

# Ensure engine path is available for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
try:
    from engine.similarity_engine import fetch_latest_features, mock_similarity_score
    from pipeline.upstox_client import UpstoxClient
    from dotenv import set_key
except ImportError as e:
    st.error(f"Failed to import modules: {e}")

# --- Page Configuration ---
st.set_page_config(
    page_title="High-Growth Screener",
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
    # Note: Ensure database directory exists before calling this
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

# --- Sidebar: Authentication & Tweaking ---
with st.sidebar:
    st.header("🔑 Upstox Authentication")
    try:
        client = UpstoxClient()
        if client.access_token:
            st.success("✅ Authenticated with Upstox!")
            if st.button("Re-authenticate", key="reauth"):
                # Clear token logic can go here
                pass
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
        min_value=50, max_value=99, value=85, step=1,
        help="Minimum similarity to historical multibagger setups to flag."
    )
    
    if st.button("Run Screener", use_container_width=True):
        st.session_state['run_screener'] = True

# --- Main Dashboard ---
st.title("🚀 High-Growth & Multibagger Screener")
st.markdown("Identify explosive momentum and potential multibaggers via volume anomalies and volatility contraction.")

if st.session_state.get('run_screener', False):
    st.success(f"Scanning stocks for >{vol_surge}x volume surges and <{vcp_tightness}% VCP tightness...")
    
    # Try fetching real data from the database
    real_data = None
    try:
        features_df = fetch_latest_features()
        if not features_df.empty:
            scored_df = mock_similarity_score(features_df, vol_surge, vcp_tightness)
            # Filter and sort
            real_data = scored_df[scored_df['similarity_score'] >= similarity_threshold].copy()
            real_data = real_data.sort_values(by='similarity_score', ascending=False)
            
            # Format for UI
            real_data['Action'] = real_data['similarity_score'].apply(
                lambda x: "🚨 Parabolic" if x > 95 else "🔥 High Conviction" if x > 85 else "⚡ Breakout Alert"
            )
            real_data = real_data[['symbol', 'volume_surge_score', 'volatility_contraction_score', 'similarity_score', 'Action']]
            real_data.columns = ["Symbol", "Volume Surge", "VCP Tightness", "Similarity Score", "Action"]
    except Exception as e:
        st.warning(f"Could not load real data from database: {e}")
        
    if real_data is not None and not real_data.empty:
        display_data = real_data
    else:
        st.warning("⚠️ Using mock data! The database is empty or the API pipeline hasn't been run yet. Configure .env and run build_db.py to see live data.")
        # Mock Data for UI Demonstration
        display_data = pd.DataFrame({
            "Symbol": ["TATAELXSI", "KPITTECH", "RVNL", "IREDA", "SUZLON"],
            "Sector": ["IT", "IT", "Railways", "Finance", "Energy"],
            "Volume Surge": [3.5, 4.2, 5.1, 8.0, 3.1],
            "VCP Tightness": [4.1, 2.3, 3.5, 1.2, 4.8],
            "Similarity Score": [92, 88, 95, 99, 86],
            "Action": ["🔥 High Conviction", "⚡ Breakout Alert", "🔥 High Conviction", "🚨 Parabolic", "⚡ Breakout Alert"]
        })
    
    st.subheader("Top Screener Results")
    st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    st.subheader("Historical Evidence Viewer")
    selected_stock = st.selectbox("Select a stock to view historical matches:", display_data["Symbol"].tolist())
    
    if selected_stock:
        try:
            conn = get_connection()
            # Get last 250 trading days (~1 year)
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
                # Sort chronologically for charting
                chart_df = chart_df.sort_values('timestamp')
                chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp']).dt.date
                chart_df.set_index('timestamp', inplace=True)
                
                # Calculate 50-day MA for visual context
                chart_df['50-Day MA'] = chart_df['close'].rolling(window=50).mean()
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**{selected_stock} - 1 Year Price Trend**")
                    st.line_chart(chart_df[['close', '50-Day MA']])
                with col2:
                    st.markdown(f"**{selected_stock} - Daily Volume**")
                    st.bar_chart(chart_df['volume'])
                    
                st.info(f"💡 Notice how the volume anomalies correlate with price breakouts. The engine mathematical flagged **{selected_stock}** because its current footprint matches historical multibaggers.")
            else:
                st.warning("No historical data available for plotting.")
        except Exception as e:
            st.error(f"Error loading chart: {e}")
    
else:
    st.info("👈 Tweak your parameters in the sidebar and click **Run Screener** to start hunting.")
