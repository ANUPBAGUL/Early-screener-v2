import os
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=env_path, override=True)

# Centralized Database Configuration
DB_NAME = os.getenv("DATABASE_NAME", "screener.db")
DB_PATH = os.path.join(PROJECT_ROOT, "database", DB_NAME)

# Ingestion & API Configuration
INDEX_KEY = os.getenv("INDEX_KEY", "NSE_INDEX|Nifty 50")
UNIVERSE_LIMIT = int(os.getenv("UNIVERSE_LIMIT", "1200"))

# Mapping global Morningstar sectors to Indian stock market standard sectors
INDIAN_SECTOR_MAP = {
    "Consumer Defensive": "FMCG",
    "Consumer Cyclical": "Auto & Consumer Durables",
    "Basic Materials": "Metals, Mining & Chemicals",
    "Financial Services": "Banking & Financial Services",
    "Real Estate": "Realty",
    "Industrials": "Capital Goods & Infrastructure",
    "Healthcare": "Pharma & Healthcare",
    "Utilities": "Power & Utilities",
    "Technology": "IT & Software",
    "Communication Services": "Telecom & Media",
    "Energy": "Oil, Gas & Energy"
}

# Quantitative Archetypes Threshold Constants
ARCHETYPES = {
    "GENERAL": {
        "MIN_ROCE": 0.18,
        "MAX_DEBT": 0.50,
        "MAX_PEG": 1.0,
        "MIN_VOLUME": 200000.0
    },
    "TURNAROUND": {
        "MIN_MCAP": 1000.0,
        "MAX_MCAP": 100000.0,
        "MAX_DEBT": 0.75,
        "MIN_REV_GROWTH": 0.15,
        "MIN_ROCE": 0.15,
        "MIN_VOL_SURGE": 1.5
    },
    "CYCLICAL": {
        "MAX_PB": 3.0,
        "MIN_EARNINGS_GROWTH": 1.0,
        "MIN_ICR": 4.0,
        "MIN_ROCE": 0.15
    },
    "STRUCTURAL": {
        "MIN_ROCE": 0.20,
        "MAX_DEBT": 0.10,
        "MIN_REV_GROWTH": 0.15
    }
}

# XGBoost label threshold — controls what the model is trained to predict.
# Change this value and retrain (python engine/breakout_model.py) to switch scenarios.
# 0.20 = swing growth screener (broad signals, less selective)
# 0.15 = target-hit predictor (model predicts "will setup hit the +15% swing target?") [DEFAULT]
# 0.50 = multibagger screener (rare signals, very conservative — requires large historical dataset)
BREAKOUT_LABEL_THRESHOLD = 0.15

# Per-archetype Confluence Score weights.
# Momentum/technical paths: XGBoost signal dominates (breakout probability is the edge).
# Value/fundamental paths: F-Score dominates (fundamental inflection precedes chart momentum).
CONFLUENCE_WEIGHTS = {
    "GENERAL":    {"xgb": 0.80, "fscore": 0.20},
    "STRUCTURAL": {"xgb": 0.80, "fscore": 0.20},
    "TURNAROUND": {"xgb": 0.30, "fscore": 0.70},
    "CYCLICAL":   {"xgb": 0.30, "fscore": 0.70},
}

# Intraday Workstation Default Parameters
INTRADAY_CONFIG = {
    "DEFAULT_CAPITAL": 500000.0,         # Default Total Capital (₹5 Lakhs)
    "DEFAULT_RISK_PCT": 1.0,             # Default Risk per Trade (1.0% = ₹5,000)
    "STOP_LOSS_OFFSET_PCT": 1.5,         # Intraday SL offset below Pivot High (%)
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", "")
}

# Ground-Floor Microcap Inflection Mode Configurations
MICROCAP_CONFIG = {
    "MIN_MCAP": 50.0,            # ₹50 Crores min cap
    "MAX_MCAP": 1500.0,          # ₹1,500 Crores max cap
    "MIN_VOLUME": 30000.0,       # 30,000 shares 50D average daily volume
    "MIN_PROMOTER": 45.0,        # 45% minimum promoter holding
}
