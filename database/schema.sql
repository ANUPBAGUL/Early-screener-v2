-- High-Growth Screener SQLite Schema

CREATE TABLE IF NOT EXISTS stocks (
    instrument_key TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    name TEXT,
    segment TEXT,
    exchange TEXT,
    market_cap REAL,
    debt_to_equity REAL,
    revenue_growth REAL,
    roce REAL,
    peg_ratio REAL,
    pe_ratio REAL,
    price_to_book REAL,
    earnings_growth REAL,
    interest_coverage REAL,
    sector TEXT,
    days_to_earnings INTEGER,           -- trading days until next earnings announcement
    fundamentals_updated_at TEXT,       -- ISO date when yfinance last refreshed this row
    data_quality INTEGER DEFAULT 0      -- 0=no data, 1=partial, 2=good, 3=high quality
);

CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_key TEXT,
    timestamp DATETIME,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    UNIQUE(instrument_key, timestamp),
    FOREIGN KEY(instrument_key) REFERENCES stocks(instrument_key)
);

CREATE TABLE IF NOT EXISTS technical_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_key TEXT,
    timestamp DATETIME,
    volatility_contraction_score REAL,
    volume_surge_score REAL,
    momentum_score REAL,
    sma_50 REAL,
    sma_150 REAL,
    sma_200 REAL,
    high_52w REAL,
    stage_2_flag INTEGER,
    pivot_high REAL,                    -- max(high) over last 15 trading days; VCP breakout trigger price
    UNIQUE(instrument_key, timestamp),
    FOREIGN KEY(instrument_key) REFERENCES stocks(instrument_key)
);

CREATE TABLE IF NOT EXISTS multibagger_dna (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_key TEXT,
    timestamp DATETIME,
    volatility_contraction_score REAL,
    volume_surge_score REAL,
    momentum_score REAL,
    subsequent_return REAL,
    debt_to_equity REAL,
    price_to_book REAL,
    roce REAL,
    UNIQUE(instrument_key, timestamp),
    FOREIGN KEY(instrument_key) REFERENCES stocks(instrument_key)
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_price_history_instrument ON price_history(instrument_key);
CREATE INDEX IF NOT EXISTS idx_price_history_timestamp ON price_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_technical_features_instrument ON technical_features(instrument_key);
CREATE INDEX IF NOT EXISTS idx_technical_features_timestamp ON technical_features(timestamp);
CREATE INDEX IF NOT EXISTS idx_multibagger_dna_instrument ON multibagger_dna(instrument_key);
CREATE INDEX IF NOT EXISTS idx_multibagger_dna_timestamp ON multibagger_dna(timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
