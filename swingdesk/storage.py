from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from swingdesk.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker     TEXT NOT NULL,
    date       TEXT NOT NULL,
    open       REAL,
    high       REAL,
    low        REAL,
    close      REAL,
    volume     REAL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);

CREATE TABLE IF NOT EXISTS news (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT NOT NULL,
    title         TEXT NOT NULL,
    link          TEXT NOT NULL UNIQUE,
    published     TEXT,
    summary       TEXT,
    tickers       TEXT,
    fetched_at    TEXT DEFAULT CURRENT_TIMESTAMP,
    -- Sentiment fields (populated by analyze.sentiment via Claude)
    sentiment     TEXT,        -- bullish | bearish | neutral
    impact        TEXT,        -- high | medium | low
    event_type    TEXT,        -- earnings | downgrade | guidance | macro | ...
    rationale     TEXT,        -- one-line explanation
    analyzed_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_published ON news(published);
CREATE INDEX IF NOT EXISTS idx_news_analyzed ON news(analyzed_at);

CREATE TABLE IF NOT EXISTS signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    ticker       TEXT NOT NULL,
    setup        TEXT NOT NULL,
    direction    TEXT NOT NULL,
    entry        REAL,
    stoploss     REAL,
    target       REAL,
    rr           REAL,
    score        REAL,
    notes        TEXT,
    universe     TEXT DEFAULT 'main'    -- 'main' | 'smallcap'
);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_generated ON signals(generated_at);

CREATE TABLE IF NOT EXISTS watchlist (
    ticker TEXT PRIMARY KEY,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS smallcap_watchlist (
    ticker TEXT PRIMARY KEY,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT NOT NULL,
    setup         TEXT,
    side          TEXT DEFAULT 'long',           -- long | short
    qty           INTEGER NOT NULL,
    entry_price   REAL NOT NULL,
    entry_date    TEXT NOT NULL,
    stoploss      REAL,
    target        REAL,
    initial_stop  REAL,                          -- never modified (vs stoploss which trails)
    high_water    REAL,                          -- highest close seen — used for trailing
    status        TEXT DEFAULT 'open',           -- open | closed
    exit_price    REAL,
    exit_date     TEXT,
    exit_reason   TEXT,                          -- target | stoploss | manual | trail | time_stop
    pnl           REAL,
    pnl_pct       REAL,
    r_multiple    REAL,
    is_paper      INTEGER DEFAULT 1,             -- 1 = paper, 0 = real
    signal_id     INTEGER,                       -- which scan signal triggered it (nullable)
    notes         TEXT,
    last_price    REAL,                          -- updated by mark-to-market
    last_marked   TEXT,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS earnings_calendar (
    ticker        TEXT PRIMARY KEY,
    next_earnings TEXT,                          -- ISO date of next earnings
    last_earnings TEXT,
    updated_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS macro_indicators (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    close       REAL,
    volume      REAL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_macro_ticker ON macro_indicators(ticker);

CREATE TABLE IF NOT EXISTS holdings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT NOT NULL,
    qty           REAL NOT NULL,
    avg_price     REAL NOT NULL,
    last_price    REAL,
    invested      REAL,                          -- qty * avg_price
    current_value REAL,                          -- qty * last_price
    pnl           REAL,
    pnl_pct       REAL,
    source        TEXT,                          -- "groww", "manual", etc.
    imported_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_holdings_ticker ON holdings(ticker);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker            TEXT PRIMARY KEY,
    short_name        TEXT,
    sector            TEXT,
    industry          TEXT,
    market_cap        REAL,                       -- in INR (full notional)
    shares_outstanding REAL,                      -- total shares issued
    float_shares      REAL,                       -- free-float shares (tradable)
    trailing_pe       REAL,
    forward_pe        REAL,
    price_to_book     REAL,
    return_on_equity  REAL,                       -- 0.15 means 15%
    debt_to_equity    REAL,                       -- yfinance gives this as %, we store as fraction (e.g. 1.5)
    profit_margin     REAL,                       -- 0.15 means 15%
    operating_margin  REAL,
    earnings_growth   REAL,                       -- YoY, 0.20 means +20%
    revenue_growth    REAL,
    current_ratio     REAL,
    dividend_yield    REAL,
    beta              REAL,
    quality_score     REAL,                       -- 0-100, computed by analyze.quality
    updated_at        TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_fund_sector ON fundamentals(sector);
CREATE INDEX IF NOT EXISTS idx_fund_quality ON fundamentals(quality_score);
CREATE INDEX IF NOT EXISTS idx_pos_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_pos_ticker ON positions(ticker);
CREATE INDEX IF NOT EXISTS idx_pos_paper ON positions(is_paper);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    setup       TEXT NOT NULL,
    entry_date  TEXT,
    exit_date   TEXT,
    entry       REAL,
    exit        REAL,
    stoploss    REAL,
    target      REAL,
    outcome     TEXT,
    r           REAL,
    bars_held   INTEGER,
    planned_rr  REAL
);
CREATE INDEX IF NOT EXISTS idx_bt_run ON backtest_trades(run_id);
CREATE INDEX IF NOT EXISTS idx_bt_setup ON backtest_trades(setup);

CREATE TABLE IF NOT EXISTS delivery (
    ticker      TEXT NOT NULL,                  -- yfinance-style, e.g. RELIANCE.NS
    date        TEXT NOT NULL,                  -- ISO yyyy-mm-dd
    traded_qty  REAL,                           -- total shares traded
    deliv_qty   REAL,                           -- shares actually delivered
    deliv_pct   REAL,                           -- delivery as % of traded (0-100)
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_delivery_ticker ON delivery(ticker);

CREATE TABLE IF NOT EXISTS deals (
    deal_type   TEXT NOT NULL,                  -- 'bulk' | 'block'
    date        TEXT NOT NULL,                  -- ISO yyyy-mm-dd
    ticker      TEXT NOT NULL,                  -- yfinance-style, e.g. RELIANCE.NS
    security    TEXT,
    client      TEXT,
    side        TEXT,                           -- 'BUY' | 'SELL'
    qty         REAL,
    price       REAL,
    PRIMARY KEY (deal_type, date, ticker, client, side, qty)
);
CREATE INDEX IF NOT EXISTS idx_deals_ticker ON deals(ticker);

CREATE TABLE IF NOT EXISTS prices_intraday (
    ticker      TEXT NOT NULL,                  -- yfinance-style, e.g. RELIANCE.NS
    interval    TEXT NOT NULL,                  -- '5m' | '15m' | '1h'
    datetime    TEXT NOT NULL,                  -- ISO timestamp (exchange tz)
    open        REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, interval, datetime)
);
CREATE INDEX IF NOT EXISTS idx_intraday_ticker ON prices_intraday(ticker, interval);

CREATE TABLE IF NOT EXISTS autotrader_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    asof               TEXT NOT NULL,             -- evaluation date (YYYY-MM-DD)
    equity             REAL,
    free_cash          REAL,
    realized_pnl       REAL,
    unrealized_pnl     REAL,
    peak_equity        REAL,
    drawdown_pct       REAL,
    portfolio_heat_pct REAL,
    n_open             INTEGER,
    opened             INTEGER,
    closed             INTEGER,
    halted             INTEGER DEFAULT 0,
    note               TEXT,
    created_at         TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS plans (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    planned_amount  REAL NOT NULL,                 -- ₹ you intend to deploy on this name
    target_price    REAL,                          -- optional desired entry level
    stop_price      REAL,
    note            TEXT,
    status          TEXT DEFAULT 'idea',           -- idea | watching | entered | dropped
    conviction      REAL,                          -- Decision.conviction snapshot at log time
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_plans_ticker ON plans(ticker);
"""


def _resolve_db(db_path: Path | None) -> Path:
    """Resolve DB path at call time so monkeypatching DB_PATH (in tests) works."""
    if db_path is not None:
        return db_path
    # Read the *current* module attribute, not a stale default.
    import swingdesk.storage as _self
    return _self.DB_PATH


def init_db(db_path: Path | None = None) -> None:
    db = _resolve_db(db_path)
    with sqlite3.connect(db) as con:
        # 1. Ensure base tables exist (no-op if already present — does not modify columns).
        for stmt in SCHEMA.split(";"):
            s = stmt.strip()
            if s.startswith("CREATE TABLE"):
                con.execute(s)
        # 2. Migrate columns on `news` for DBs created before Week 2.
        existing = {r[1] for r in con.execute("PRAGMA table_info(news)")}
        for col, ddl in [
            ("sentiment",   "ALTER TABLE news ADD COLUMN sentiment TEXT"),
            ("impact",      "ALTER TABLE news ADD COLUMN impact TEXT"),
            ("event_type",  "ALTER TABLE news ADD COLUMN event_type TEXT"),
            ("rationale",   "ALTER TABLE news ADD COLUMN rationale TEXT"),
            ("analyzed_at", "ALTER TABLE news ADD COLUMN analyzed_at TEXT"),
        ]:
            if col not in existing:
                con.execute(ddl)
        # 2b. Migrate `positions` for Week 5 trailing-stop columns.
        existing_pos = {r[1] for r in con.execute("PRAGMA table_info(positions)")}
        for col, ddl in [
            ("initial_stop", "ALTER TABLE positions ADD COLUMN initial_stop REAL"),
            ("high_water",   "ALTER TABLE positions ADD COLUMN high_water REAL"),
        ]:
            if col not in existing_pos:
                con.execute(ddl)
        # 2c. Migrate `signals` for the universe label (small-cap separation).
        existing_sig = {r[1] for r in con.execute("PRAGMA table_info(signals)")}
        if "universe" not in existing_sig:
            con.execute("ALTER TABLE signals ADD COLUMN universe TEXT DEFAULT 'main'")
        # 2d. Migrate `fundamentals` for the size/float columns (manipulation section).
        existing_fund = {r[1] for r in con.execute("PRAGMA table_info(fundamentals)")}
        for col, ddl in [
            ("shares_outstanding", "ALTER TABLE fundamentals ADD COLUMN shares_outstanding REAL"),
            ("float_shares",       "ALTER TABLE fundamentals ADD COLUMN float_shares REAL"),
        ]:
            if col not in existing_fund:
                con.execute(ddl)
        # 3. Now safe to create indexes (the columns they reference exist).
        for stmt in SCHEMA.split(";"):
            s = stmt.strip()
            if s.startswith("CREATE INDEX"):
                con.execute(s)
        con.commit()


@contextmanager
def connect(db_path: Path | None = None):
    con = sqlite3.connect(_resolve_db(db_path))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def upsert_prices(ticker: str, df: pd.DataFrame) -> int:
    """Insert/replace OHLCV rows. df has DatetimeIndex and OHLCV columns."""
    if df is None or df.empty:
        return 0
    out = df.copy()
    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    out = out.reset_index().rename(
        columns={
            "index": "date",
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    out.columns = [c.lower() for c in out.columns]
    out["ticker"] = ticker
    cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
    out = out[cols]
    rows = list(out.itertuples(index=False, name=None))
    with connect() as con:
        con.executemany(
            "INSERT OR REPLACE INTO prices (ticker,date,open,high,low,close,volume) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def load_prices(ticker: str, days: int | None = None) -> pd.DataFrame:
    q = "SELECT date,open,high,low,close,volume FROM prices WHERE ticker=? ORDER BY date"
    with connect() as con:
        df = pd.read_sql_query(q, con, params=(ticker,), parse_dates=["date"])
    df = df.set_index("date")
    if days:
        df = df.tail(days)
    return df


def insert_news(items: list[dict]) -> int:
    if not items:
        return 0
    rows = [
        (
            it.get("source"),
            it.get("title"),
            it.get("link"),
            it.get("published"),
            it.get("summary"),
            ",".join(it.get("tickers") or []),
        )
        for it in items
    ]
    with connect() as con:
        cur = con.executemany(
            "INSERT OR IGNORE INTO news (source,title,link,published,summary,tickers) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
        return cur.rowcount or 0


def load_news(limit: int = 100, ticker: str | None = None) -> pd.DataFrame:
    cols = ("source,title,link,published,summary,tickers,"
            "sentiment,impact,event_type,rationale,analyzed_at")
    if ticker:
        q = (f"SELECT {cols} FROM news WHERE tickers LIKE ? "
             "ORDER BY published DESC LIMIT ?")
        params = (f"%{ticker}%", limit)
    else:
        q = f"SELECT {cols} FROM news ORDER BY published DESC LIMIT ?"
        params = (limit,)
    with connect() as con:
        return pd.read_sql_query(q, con, params=params)


def load_unanalyzed_news(limit: int = 200) -> pd.DataFrame:
    """News rows that don't yet have sentiment, prioritizing items with ticker matches."""
    q = ("SELECT id, source, title, summary, tickers FROM news "
         "WHERE analyzed_at IS NULL "
         "ORDER BY (tickers != '') DESC, published DESC "
         "LIMIT ?")
    with connect() as con:
        return pd.read_sql_query(q, con, params=(limit,))


def update_news_sentiment(rows: list[dict]) -> int:
    """Each row: {id, sentiment, impact, event_type, rationale}."""
    if not rows:
        return 0
    with connect() as con:
        con.executemany(
            "UPDATE news SET sentiment=?, impact=?, event_type=?, rationale=?, "
            "analyzed_at=CURRENT_TIMESTAMP WHERE id=?",
            [(r["sentiment"], r["impact"], r["event_type"], r["rationale"], r["id"])
             for r in rows],
        )
    return len(rows)


def insert_position(pos: dict) -> int:
    """Insert a new position. Returns the new row id."""
    cols = ["ticker", "setup", "side", "qty", "entry_price", "entry_date",
            "stoploss", "target", "is_paper", "signal_id", "notes"]
    vals = [pos.get(c) for c in cols]
    with connect() as con:
        cur = con.execute(
            f"INSERT INTO positions ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})",
            vals,
        )
        return cur.lastrowid


def update_position(pos_id: int, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = pd.Timestamp.utcnow().isoformat()
    sets = ", ".join(f"{k}=?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE positions SET {sets} WHERE id=?",
                    list(fields.values()) + [pos_id])


def load_positions(*, status: str | None = None, is_paper: bool | None = None) -> pd.DataFrame:
    q = "SELECT * FROM positions WHERE 1=1"
    params: list = []
    if status:
        q += " AND status=?"
        params.append(status)
    if is_paper is not None:
        q += " AND is_paper=?"
        params.append(int(is_paper))
    q += " ORDER BY entry_date DESC, id DESC"
    with connect() as con:
        return pd.read_sql_query(q, con, params=tuple(params))


def get_position(pos_id: int) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM positions WHERE id=?", (pos_id,)).fetchone()
    return dict(row) if row else None


def log_autotrader_step(row: dict) -> int:
    """Append one paper-autotrader evaluation to the run log. Returns row id."""
    cols = ["asof", "equity", "free_cash", "realized_pnl", "unrealized_pnl",
            "peak_equity", "drawdown_pct", "portfolio_heat_pct", "n_open",
            "opened", "closed", "halted", "note"]
    vals = [row.get(c) for c in cols]
    with connect() as con:
        cur = con.execute(
            f"INSERT INTO autotrader_log ({','.join(cols)}) "
            f"VALUES ({','.join(['?'] * len(cols))})", vals)
        return cur.lastrowid


def load_autotrader_log(limit: int = 200) -> pd.DataFrame:
    """Most-recent-first autotrader run log (equity curve + per-step actions)."""
    with connect() as con:
        return pd.read_sql_query(
            "SELECT * FROM autotrader_log ORDER BY id DESC LIMIT ?", con,
            params=(limit,))


def upsert_earnings(ticker: str, next_earnings: str | None,
                    last_earnings: str | None = None) -> None:
    with connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO earnings_calendar "
            "(ticker, next_earnings, last_earnings, updated_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (ticker, next_earnings, last_earnings),
        )


def get_next_earnings(ticker: str) -> str | None:
    with connect() as con:
        row = con.execute(
            "SELECT next_earnings FROM earnings_calendar WHERE ticker=?",
            (ticker,),
        ).fetchone()
    return row["next_earnings"] if row else None


def upsert_macro(ticker: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    out = df.copy()
    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    out = out.reset_index().rename(columns={"index": "date", "Date": "date",
                                             "Close": "close", "Volume": "volume"})
    out.columns = [c.lower() for c in out.columns]
    out["ticker"] = ticker
    rows = [(r.ticker, r.date, r.close, r.volume)
            for r in out[["ticker", "date", "close", "volume"]].itertuples(index=False)]
    with connect() as con:
        con.executemany(
            "INSERT OR REPLACE INTO macro_indicators (ticker,date,close,volume) "
            "VALUES (?,?,?,?)", rows,
        )
    return len(rows)


def load_macro(ticker: str, days: int | None = None) -> pd.DataFrame:
    q = "SELECT date, close, volume FROM macro_indicators WHERE ticker=? ORDER BY date"
    with connect() as con:
        df = pd.read_sql_query(q, con, params=(ticker,), parse_dates=["date"])
    df = df.set_index("date")
    if days:
        df = df.tail(days)
    return df


def upsert_intraday(ticker: str, df: pd.DataFrame, interval: str) -> int:
    """Insert/replace intraday OHLCV bars. df has a DatetimeIndex + OHLCV cols."""
    if df is None or df.empty:
        return 0
    out = df.copy()
    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d %H:%M:%S")
    out = out.reset_index().rename(columns={
        "index": "datetime", "Datetime": "datetime", "Date": "datetime",
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj_close", "Volume": "volume",
    })
    out.columns = [c.lower() for c in out.columns]
    rows = [(ticker, interval, r.datetime, getattr(r, "open", None),
             getattr(r, "high", None), getattr(r, "low", None),
             getattr(r, "close", None), getattr(r, "volume", None))
            for r in out.itertuples(index=False)]
    with connect() as con:
        con.executemany(
            "INSERT OR REPLACE INTO prices_intraday "
            "(ticker,interval,datetime,open,high,low,close,volume) "
            "VALUES (?,?,?,?,?,?,?,?)", rows,
        )
    return len(rows)


def load_intraday(ticker: str, interval: str = "5m",
                  days: int | None = None) -> pd.DataFrame:
    """Load intraday bars for a ticker/interval, oldest→newest, datetime-indexed."""
    q = ("SELECT datetime,open,high,low,close,volume FROM prices_intraday "
         "WHERE ticker=? AND interval=? ORDER BY datetime")
    with connect() as con:
        df = pd.read_sql_query(q, con, params=(ticker, interval),
                               parse_dates=["datetime"])
    df = df.set_index("datetime")
    if days:
        cutoff = df.index.max() - pd.Timedelta(days=days)
        df = df[df.index >= cutoff]
    return df


def list_macro_tickers() -> list[str]:
    with connect() as con:
        rows = con.execute(
            "SELECT DISTINCT ticker FROM macro_indicators ORDER BY ticker"
        ).fetchall()
    return [r["ticker"] for r in rows]


def replace_holdings(rows: list[dict], source: str = "groww") -> int:
    """Holdings are a snapshot — replace whole table on each import."""
    with connect() as con:
        con.execute("DELETE FROM holdings")
        if not rows:
            return 0
        cols = ["ticker", "qty", "avg_price", "last_price", "invested",
                "current_value", "pnl", "pnl_pct", "source"]
        placeholders = ",".join(["?"] * len(cols))
        for r in rows:
            r.setdefault("source", source)
        con.executemany(
            f"INSERT INTO holdings ({','.join(cols)}) VALUES ({placeholders})",
            [tuple(r.get(c) for c in cols) for r in rows],
        )
    return len(rows)


def load_holdings() -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query(
            "SELECT * FROM holdings ORDER BY current_value DESC NULLS LAST", con
        )


def holdings_tickers() -> list[str]:
    """Distinct list of tickers currently held (used to enrich the data layer)."""
    with connect() as con:
        rows = con.execute(
            "SELECT DISTINCT ticker FROM holdings ORDER BY ticker"
        ).fetchall()
    return [r["ticker"] for r in rows]


# ---- Investment plans (intended ₹ per ticker; distinct from positions/holdings) --
# `plans` is *intent* — how much you mean to deploy on a name and where. It is
# decoupled from `positions` (paper/real trades) and `holdings` (Groww snapshot,
# replaced wholesale on import). One active plan per ticker.

def upsert_plan(plan: dict) -> int:
    """Insert or update the active plan for a ticker. Returns its row id.

    An existing plan whose status is not 'dropped' is updated in place (so the
    intended amount/levels for a name stay a single row); otherwise a new row is
    inserted. Recognised keys: ticker, planned_amount, target_price, stop_price,
    note, status, conviction."""
    ticker = plan.get("ticker")
    plan = {**plan, "status": plan.get("status") or "idea"}   # never insert NULL status
    fields = ["planned_amount", "target_price", "stop_price", "note",
              "status", "conviction"]
    with connect() as con:
        row = con.execute(
            "SELECT id FROM plans WHERE ticker=? AND (status IS NULL OR status!='dropped') "
            "ORDER BY id DESC LIMIT 1", (ticker,)).fetchone()
        if row:
            pid = row["id"]
            sets = ", ".join(f"{k}=?" for k in fields if k in plan)
            vals = [plan[k] for k in fields if k in plan]
            if sets:
                con.execute(
                    f"UPDATE plans SET {sets}, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    vals + [pid])
            return pid
        cols = ["ticker"] + fields
        vals = [plan.get(c) for c in cols]
        cur = con.execute(
            f"INSERT INTO plans ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})",
            vals)
        return cur.lastrowid


def load_plans(status: str | None = None) -> pd.DataFrame:
    q = "SELECT * FROM plans"
    params: tuple = ()
    if status:
        q += " WHERE status=?"
        params = (status,)
    q += " ORDER BY created_at DESC, id DESC"
    with connect() as con:
        return pd.read_sql_query(q, con, params=params)


def update_plan(plan_id: int, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = pd.Timestamp.utcnow().isoformat()
    sets = ", ".join(f"{k}=?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE plans SET {sets} WHERE id=?",
                    list(fields.values()) + [plan_id])


def delete_plan(plan_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM plans WHERE id=?", (plan_id,))


def combined_universe(include_smallcaps: bool = False,
                      include_discovery: bool = False) -> list[str]:
    """Watchlist + holdings deduped — the full set of tickers we should
    keep prices/fundamentals/news data for.

    Optionally include the curated small-cap or large/mid-cap discovery
    universes — useful when running `news` or `sentiment` so headlines
    mentioning those names get tagged properly.
    """
    wl = get_watchlist()
    held = holdings_tickers()
    universe = set(wl) | set(held)
    if include_smallcaps:
        # Lazy import to avoid circular dependency
        from swingdesk.analyze.smallcaps import SMALLCAP_UNIVERSE
        universe |= set(SMALLCAP_UNIVERSE)
    if include_discovery:
        from swingdesk.analyze.discovery import DISCOVERY_UNIVERSE
        universe |= set(DISCOVERY_UNIVERSE)
    return sorted(universe)


def upsert_fundamentals(rows: list[dict]) -> int:
    """Insert/replace fundamentals for many tickers."""
    if not rows:
        return 0
    cols = ["ticker", "short_name", "sector", "industry", "market_cap",
            "shares_outstanding", "float_shares",
            "trailing_pe", "forward_pe", "price_to_book", "return_on_equity",
            "debt_to_equity", "profit_margin", "operating_margin",
            "earnings_growth", "revenue_growth", "current_ratio",
            "dividend_yield", "beta", "quality_score"]
    placeholders = ",".join(["?"] * len(cols))
    with connect() as con:
        con.executemany(
            f"INSERT OR REPLACE INTO fundamentals ({','.join(cols)}, updated_at) "
            f"VALUES ({placeholders}, CURRENT_TIMESTAMP)",
            [tuple(r.get(c) for c in cols) for r in rows],
        )
    return len(rows)


def get_fundamentals(ticker: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM fundamentals WHERE ticker=?",
                          (ticker,)).fetchone()
    return dict(row) if row else None


def load_fundamentals(min_quality: float | None = None) -> pd.DataFrame:
    q = "SELECT * FROM fundamentals"
    params: tuple = ()
    if min_quality is not None:
        q += " WHERE quality_score >= ?"
        params = (min_quality,)
    q += " ORDER BY quality_score DESC NULLS LAST"
    with connect() as con:
        return pd.read_sql_query(q, con, params=params)


def upsert_delivery(rows: list[dict]) -> int:
    """Insert/replace security-wise delivery rows (ticker, date, qtys, pct)."""
    if not rows:
        return 0
    cols = ["ticker", "date", "traded_qty", "deliv_qty", "deliv_pct"]
    placeholders = ",".join(["?"] * len(cols))
    with connect() as con:
        con.executemany(
            f"INSERT OR REPLACE INTO delivery ({','.join(cols)}) VALUES ({placeholders})",
            [tuple(r.get(c) for c in cols) for r in rows],
        )
    return len(rows)


def load_delivery(ticker: str, days: int | None = None) -> pd.DataFrame:
    q = "SELECT date, traded_qty, deliv_qty, deliv_pct FROM delivery WHERE ticker=? ORDER BY date"
    with connect() as con:
        df = pd.read_sql_query(q, con, params=(ticker,), parse_dates=["date"])
    df = df.set_index("date")
    if days:
        df = df.tail(days)
    return df


def upsert_deals(rows: list[dict]) -> int:
    """Insert/replace bulk/block deal rows."""
    if not rows:
        return 0
    cols = ["deal_type", "date", "ticker", "security", "client", "side", "qty", "price"]
    placeholders = ",".join(["?"] * len(cols))
    with connect() as con:
        con.executemany(
            f"INSERT OR REPLACE INTO deals ({','.join(cols)}) VALUES ({placeholders})",
            [tuple(r.get(c) for c in cols) for r in rows],
        )
    return len(rows)


def load_deals(ticker: str | None = None, days: int | None = None) -> pd.DataFrame:
    q = "SELECT deal_type, date, ticker, security, client, side, qty, price FROM deals"
    params: tuple = ()
    if ticker:
        q += " WHERE ticker=?"
        params = (ticker,)
    q += " ORDER BY date DESC"
    with connect() as con:
        df = pd.read_sql_query(q, con, params=params, parse_dates=["date"])
    if days and not df.empty:
        cutoff = df["date"].max() - pd.Timedelta(days=days)
        df = df[df["date"] >= cutoff]
    return df


def load_earnings_calendar() -> pd.DataFrame:
    q = ("SELECT ticker, next_earnings, last_earnings, updated_at "
         "FROM earnings_calendar ORDER BY next_earnings ASC")
    with connect() as con:
        return pd.read_sql_query(q, con)


def open_positions_for_ticker(ticker: str) -> pd.DataFrame:
    """All open positions for one ticker (used to prevent duplicate paper entries)."""
    q = "SELECT * FROM positions WHERE ticker=? AND status='open'"
    with connect() as con:
        return pd.read_sql_query(q, con, params=(ticker,))


def save_backtest_trades(run_id: str, trades_df: pd.DataFrame) -> int:
    if trades_df is None or trades_df.empty:
        return 0
    df = trades_df.copy()
    df["run_id"] = run_id
    cols = ["run_id", "ticker", "setup", "entry_date", "exit_date", "entry",
            "exit", "stoploss", "target", "outcome", "r", "bars_held", "planned_rr"]
    df = df[cols]
    rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
    with connect() as con:
        con.executemany(
            "INSERT INTO backtest_trades "
            "(run_id,ticker,setup,entry_date,exit_date,entry,exit,stoploss,target,outcome,r,bars_held,planned_rr) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def load_backtest_trades(run_id: str | None = None) -> pd.DataFrame:
    if run_id:
        q = "SELECT * FROM backtest_trades WHERE run_id=? ORDER BY entry_date"
        params = (run_id,)
    else:
        q = "SELECT * FROM backtest_trades ORDER BY entry_date"
        params = ()
    with connect() as con:
        return pd.read_sql_query(q, con, params=params)


def list_backtest_runs() -> pd.DataFrame:
    q = ("SELECT run_id, COUNT(*) as n_trades, MIN(entry_date) as start, "
         "MAX(entry_date) as end FROM backtest_trades GROUP BY run_id "
         "ORDER BY run_id DESC")
    with connect() as con:
        return pd.read_sql_query(q, con)


def recent_sentiment_for_ticker(ticker: str, days: int = 7) -> pd.DataFrame:
    """Recent analyzed news for a ticker, used by the scoring engine."""
    q = ("SELECT title, sentiment, impact, event_type, rationale, published "
         "FROM news WHERE tickers LIKE ? AND sentiment IS NOT NULL "
         "AND published >= datetime('now', ?) "
         "ORDER BY published DESC")
    with connect() as con:
        return pd.read_sql_query(q, con, params=(f"%{ticker}%", f"-{days} days"))


def save_signals(signals: list[dict], universe: str = "main") -> int:
    """Persist signals with a universe label ('main' | 'smallcap').
    A per-signal `universe` key overrides the parameter for that row."""
    if not signals:
        return 0
    rows = [
        (
            s.get("ticker"),
            s.get("setup"),
            s.get("direction", "long"),
            s.get("entry"),
            s.get("stoploss"),
            s.get("target"),
            s.get("rr"),
            s.get("score"),
            s.get("notes"),
            s.get("universe") or universe,
        )
        for s in signals
    ]
    with connect() as con:
        con.executemany(
            "INSERT INTO signals (ticker,setup,direction,entry,stoploss,target,"
            "rr,score,notes,universe) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def load_signals(limit: int = 50, universe: str | None = None) -> pd.DataFrame:
    """Load signals optionally filtered by universe. None → both."""
    base = ("SELECT generated_at,ticker,setup,direction,entry,stoploss,target,"
            "rr,score,notes,universe FROM signals")
    if universe:
        q = f"{base} WHERE universe=? ORDER BY generated_at DESC LIMIT ?"
        params: tuple = (universe, limit)
    else:
        q = f"{base} ORDER BY generated_at DESC LIMIT ?"
        params = (limit,)
    with connect() as con:
        return pd.read_sql_query(q, con, params=params)


def get_watchlist() -> list[str]:
    with connect() as con:
        rows = con.execute("SELECT ticker FROM watchlist ORDER BY ticker").fetchall()
    return [r["ticker"] for r in rows]


def set_watchlist(tickers: list[str]) -> None:
    with connect() as con:
        con.execute("DELETE FROM watchlist")
        con.executemany("INSERT INTO watchlist (ticker) VALUES (?)", [(t,) for t in tickers])


def seed_watchlist_if_empty(default: list[str]) -> None:
    if not get_watchlist():
        set_watchlist(default)


# ---- Small-cap watchlist (independent from the main watchlist) ----------------

def get_smallcap_watchlist() -> list[str]:
    with connect() as con:
        rows = con.execute(
            "SELECT ticker FROM smallcap_watchlist ORDER BY ticker"
        ).fetchall()
    return [r["ticker"] for r in rows]


def set_smallcap_watchlist(tickers: list[str]) -> None:
    with connect() as con:
        con.execute("DELETE FROM smallcap_watchlist")
        con.executemany(
            "INSERT INTO smallcap_watchlist (ticker) VALUES (?)",
            [(t,) for t in tickers],
        )


def add_to_smallcap_watchlist(ticker: str) -> bool:
    """Idempotent add. Returns True if newly added, False if already present."""
    with connect() as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO smallcap_watchlist (ticker) VALUES (?)",
            (ticker,),
        )
        return cur.rowcount > 0


def remove_from_smallcap_watchlist(ticker: str) -> bool:
    with connect() as con:
        cur = con.execute(
            "DELETE FROM smallcap_watchlist WHERE ticker=?", (ticker,)
        )
        return cur.rowcount > 0
