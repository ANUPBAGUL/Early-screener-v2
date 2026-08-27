-- High-Growth Screener SQLite Schema

CREATE TABLE IF NOT EXISTS stocks (
    instrument_key TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    name TEXT,
    segment TEXT,
    exchange TEXT
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
    UNIQUE(instrument_key, timestamp),
    FOREIGN KEY(instrument_key) REFERENCES stocks(instrument_key)
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_price_history_instrument ON price_history(instrument_key);
CREATE INDEX IF NOT EXISTS idx_price_history_timestamp ON price_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_technical_features_instrument ON technical_features(instrument_key);
CREATE INDEX IF NOT EXISTS idx_technical_features_timestamp ON technical_features(timestamp);
