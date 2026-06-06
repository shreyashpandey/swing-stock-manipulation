"""Import + analyze a Groww-style holdings report.

Groww's portfolio export (CSV or Excel) typically has columns like:
    Stock Name | Symbol | Quantity | Average Buy Price | Current Price |
    Total Invested | Current Value | P&L | P&L %

We accept any reasonable variant — column names are matched against an
alias map, and overrides can be passed via `--map`. Holdings replace the
whole holdings table on each import (it's a snapshot, not history).

Per-stock analysis combines four lenses:
    1. Fundamentals — quality score from analyze/quality.py
    2. Technicals — where price sits vs 20/50/200 EMA, RSI, any active setup
    3. News sentiment — recent bullish/bearish/impact counts
    4. P&L — current unrealized gain/loss

The recommendation engine considers all four and emits one of:
    BUY_MORE, HOLD, REDUCE, SELL, NO_DATA
along with a short, specific reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rich.console import Console

from swingdesk.analyze import early_exits as early_exits_mod
from swingdesk.analyze import exits as exits_mod
from swingdesk.analyze import quality as quality_mod
from swingdesk.analyze.setups import scan_ticker
from swingdesk.analyze.technicals import add_indicators, trend_quality
from swingdesk.storage import (
    get_fundamentals,
    load_news,
    load_prices,
    recent_sentiment_for_ticker,
    replace_holdings,
)

console = Console()

# Column aliases — any of these resolve to the canonical field.
COLUMN_ALIASES = {
    "symbol":    ["symbol", "stock", "stock name", "stocks", "tradingsymbol",
                  "scrip", "scrip code", "instrument", "ticker", "stock symbol",
                  "company", "company name", "name"],
    "qty":       ["qty", "quantity", "shares", "no. of shares", "holding qty",
                  "qty.", "no of shares", "holdings"],
    "avg_price": ["avg price", "average price", "average buy price",
                  "avg. buy price", "avg buy price", "buy avg", "avg cost",
                  "average cost", "avg. price", "avg. buy", "buy price",
                  "avg buy", "average buy"],
    "last_price": ["last price", "current price", "ltp", "market price", "cmp",
                   "last traded price", "live price", "current"],
    "invested":  ["invested", "total invested", "investment", "buy value",
                  "invested value", "total investment"],
    "current_value": ["current value", "market value", "value", "current val",
                      "market val", "present value"],
    "pnl":       ["p&l", "pnl", "profit/loss", "unrealized p&l", "unrealised p&l",
                  "profit", "p & l", "net p&l", "total p&l"],
    "pnl_pct":   ["p&l %", "pnl %", "p&l (%)", "% gain", "return %",
                  "p&l%", "% change", "returns", "return"],
}


def _normalize_columns(df: pd.DataFrame, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    overrides = overrides or {}
    lower = {c.lower().strip(): c for c in df.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in overrides:
            rename[overrides[canonical]] = canonical
            continue
        for alias in aliases:
            if alias in lower:
                rename[lower[alias]] = canonical
                break
    return df.rename(columns=rename)


# Company-name → NSE ticker resolver. Built around the curated watchlist plus
# common Nifty-100 names. Used when the broker export gives "Reliance Industries
# Ltd" instead of "RELIANCE". Match is case-insensitive substring match against
# the keys (longest-first to avoid "Tata Motors" matching "Tata").
NAME_TO_TICKER = {
    # Large caps
    "reliance industries": "RELIANCE.NS",
    "reliance": "RELIANCE.NS",
    "tata consultancy services": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "axis bank": "AXISBANK.NS",
    "kotak mahindra bank": "KOTAKBANK.NS",
    "kotak bank": "KOTAKBANK.NS",
    "state bank of india": "SBIN.NS",
    "state bank": "SBIN.NS",
    "induslnd bank": "INDUSINDBK.NS",
    "indusind bank": "INDUSINDBK.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "airtel": "BHARTIARTL.NS",
    "larsen & toubro": "LT.NS",
    "larsen and toubro": "LT.NS",
    "l&t": "LT.NS",
    "hindustan unilever": "HINDUNILVR.NS",
    "itc": "ITC.NS",
    "asian paints": "ASIANPAINT.NS",
    "maruti suzuki": "MARUTI.NS",
    "maruti": "MARUTI.NS",
    "tata motors": "TATAMOTORS.NS",
    "mahindra & mahindra": "M&M.NS",
    "mahindra and mahindra": "M&M.NS",
    "tata steel": "TATASTEEL.NS",
    "jsw steel": "JSWSTEEL.NS",
    "hindalco": "HINDALCO.NS",
    "vedanta": "VEDL.NS",
    "ongc": "ONGC.NS",
    "oil and natural gas": "ONGC.NS",
    "coal india": "COALINDIA.NS",
    "ntpc": "NTPC.NS",
    "powergrid": "POWERGRID.NS",
    "power grid": "POWERGRID.NS",
    "tata power": "TATAPOWER.NS",
    "jsw energy": "JSWENERGY.NS",
    "adani enterprises": "ADANIENT.NS",
    "adani ports": "ADANIPORTS.NS",
    "adani green": "ADANIGREEN.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "bajaj finserv": "BAJAJFINSV.NS",
    "bajaj auto": "BAJAJ-AUTO.NS",
    "sun pharma": "SUNPHARMA.NS",
    "dr reddy": "DRREDDY.NS",
    "cipla": "CIPLA.NS",
    "divi's lab": "DIVISLAB.NS",
    "divis lab": "DIVISLAB.NS",
    "apollo hospital": "APOLLOHOSP.NS",
    "nestle": "NESTLEIND.NS",
    "britannia": "BRITANNIA.NS",
    "titan": "TITAN.NS",
    "wipro": "WIPRO.NS",
    "hcl technologies": "HCLTECH.NS",
    "hcl tech": "HCLTECH.NS",
    "tech mahindra": "TECHM.NS",
    "lti mindtree": "LTIM.NS",
    "ltimindtree": "LTIM.NS",
    "persistent systems": "PERSISTENT.NS",
    "coforge": "COFORGE.NS",
    "trent": "TRENT.NS",
    "dixon technologies": "DIXON.NS",
    "polycab": "POLYCAB.NS",
    "tvs motor": "TVSMOTOR.NS",
    "eicher motors": "EICHERMOT.NS",
    "hero motocorp": "HEROMOTOCO.NS",
    "bharat electronics": "BEL.NS",
    "hindustan aeronautics": "HAL.NS",
    "bharat forge": "BHARATFORG.NS",
    "ultratech cement": "ULTRACEMCO.NS",
    "grasim": "GRASIM.NS",
    "dlf": "DLF.NS",
    "godrej properties": "GODREJPROP.NS",
    "oberoi realty": "OBEROIRLTY.NS",
    "hdfc life": "HDFCLIFE.NS",
    "sbi life": "SBILIFE.NS",
    "central depository services": "CDSL.NS",
    "cdsl": "CDSL.NS",
    "bpcl": "BPCL.NS",
    "bharat petroleum": "BPCL.NS",
    "indian oil": "IOC.NS",
    "ioc": "IOC.NS",
    # Newer / mid-cap names commonly held by retail
    "suzlon energy": "SUZLON.NS",
    "suzlon": "SUZLON.NS",
    "swiggy": "SWIGGY.NS",
    "vodafone idea": "IDEA.NS",
    "vodafone": "IDEA.NS",
    "yes bank": "YESBANK.NS",
    "waaree energies": "WAAREEENER.NS",
    "waaree": "WAAREEENER.NS",
    "zomato": "ZOMATO.NS",
    "paytm": "PAYTM.NS",
    "one 97 communications": "PAYTM.NS",
    "nykaa": "NYKAA.NS",
    "fsn e-commerce": "NYKAA.NS",
    "policybazaar": "POLICYBZR.NS",
    "pb fintech": "POLICYBZR.NS",
    "irctc": "IRCTC.NS",
    "irfc": "IRFC.NS",
    "indian railway finance": "IRFC.NS",
    "rec": "RECLTD.NS",
    "rec limited": "RECLTD.NS",
    "power finance": "PFC.NS",
    "pfc": "PFC.NS",
    "ireda": "IREDA.NS",
    "mazagon dock": "MAZDOCK.NS",
    "cochin shipyard": "COCHINSHIP.NS",
    "bemco": "BEML.NS",
    "beml": "BEML.NS",
    "rvnl": "RVNL.NS",
    "rail vikas": "RVNL.NS",
    "lichousing": "LICHSGFIN.NS",
    "lic housing": "LICHSGFIN.NS",
    "lic of india": "LICI.NS",
    "life insurance corp": "LICI.NS",
    "tata gold etf": "TATAGOLD.NS",
    "tata gold": "TATAGOLD.NS",
}


# Direct corrections for broker symbols that resolve to the WRONG / no ticker.
# Groww exports sometimes dump a concatenated long name into the symbol column
# (e.g. "COALINDIALTD", "BANKOFBARODA") which neither looks like a ticker nor
# matches the name map cleanly. Key = the raw uppercased symbol as Groww emits
# it (after -EQ etc. stripping); value = correct NSE ticker, or None if the
# instrument has no tradable yfinance equity series (SGBs, some ETFs).
# All non-None targets verified against yfinance.
TICKER_CORRECTIONS: dict[str, str | None] = {
    "ADANIPORT&SEZLTD": "ADANIPORTS.NS",
    "ADANIPOWERLTD": "ADANIPOWER.NS",
    "BAJAJHINDUSTHANSUGARLT": "BAJAJHIND.NS",
    "BANKOFBARODA": "BANKBARODA.NS",
    "CENTRALDEPOSER(I)LTD": "CDSL.NS",
    "COALINDIALTD": "COALINDIA.NS",
    "ETERNALLIMITED": "ETERNAL.NS",          # ex-Zomato, renamed 2025
    "FEDERALBANKLTD": "FEDERALBNK.NS",
    "GTLINFRA.LTD": "GTLINFRA.NS",
    "INDIANRAILWAYFINCORPL": "IRFC.NS",
    "INDIANRENEWABLEENERGY": "IREDA.NS",
    "MAHINDRALIFESPACEDEVLTD": "MAHLIFE.NS",
    "MOSCHIPTECHNOLOGIESLTD": "MOSCHIP.NS",
    "MSTCLIMITED": "MSTCLTD.NS",
    "NHPCLTD": "NHPC.NS",
    "OLAELECTRICMOBILITYLTD": "OLAELEC.NS",
    "ONEMOBIKWIKSYSTEMSLTD": "MOBIKWIK.NS",
    "ORIENTGREENPOWERCOLTD": "GREENPOWER.NS",
    "POWERFINCORPLTD.": "PFC.NS",
    "POWERFINCORPLTD": "PFC.NS",
    "SHREERENUKASUGARSLTD": "RENUKA.NS",
    "SHRIRAMPIST.&RINGLTD": "SHRIPISTON.NS",
    "SJVNLTD": "SJVN.NS",
    # Tata Motors demerged (2025): old TATAMOTORS.NS no longer trades. Map to
    # the passenger-vehicle successor that retail holders track. (You may also
    # separately hold the commercial-vehicle entity — add it manually if so.)
    "TATAMOTORS": "TMPV.NS",
    "TATAAML-TATSILV": None,                 # Tata Silver ETF — no clean series
    "2.50%GOLDBONDS2031SR-III": None,        # Sovereign Gold Bond — no equity data
    "AVENUESAILIMITED": None,                # unresolved — correct manually
}


def _apply_correction(sym: str) -> str | None | bool:
    """If `sym` (raw, uppercased, suffix-stripped) is a known-bad broker symbol,
    return its correction. Returns False when there's no correction entry so the
    caller can fall through to the normal resolver."""
    key = sym.upper().strip()
    if key in TICKER_CORRECTIONS:
        return TICKER_CORRECTIONS[key]          # str ticker, or None (untradable)
    # Also try the .NS-stripped form
    if key.endswith(".NS") and key[:-3] in TICKER_CORRECTIONS:
        return TICKER_CORRECTIONS[key[:-3]]
    return False


def _resolve_from_name(name: str) -> str | None:
    """Try to map a company name to its NSE ticker. Returns None if no match.

    Handles three input shapes:
      1. "Reliance Industries Limited" — proper company name with spaces
      2. "RELIANCEINDUSTRIESLIMITED" — concatenated no-space form (Groww CSV)
      3. "RELIANCE INDUSTRIES LTD" — uppercase with spaces
    """
    lc = name.lower().strip()
    # Strip common *trailing* suffixes (with or without leading space).
    # NOTE: "india" is deliberately NOT here — it wrongly truncates names like
    # "coalindia" → "coal". Country/qualifier words are handled by substring
    # matching against the name map instead.
    suffixes = ["limited", "ltd.", "ltd", "corporation", "corp.", "corp",
                "industries", "inc.", "inc", "company"]
    # Try suffix removal with both " suffix" and "suffix" (concatenated)
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if lc.endswith(" " + suf):
                lc = lc[: -(len(suf) + 1)].strip()
                changed = True
                break
            elif lc.endswith(suf) and len(lc) > len(suf):
                lc = lc[: -len(suf)].strip()
                changed = True
                break
    # Direct match
    if lc in NAME_TO_TICKER:
        return NAME_TO_TICKER[lc]
    # Try inserting spaces between camelCase / lower runs (for concat names)
    # e.g. "renergy" → already handled; "vodafoneidea" → "vodafone idea"
    # We use the longest-substring match below as the catchall.
    for key in sorted(NAME_TO_TICKER, key=len, reverse=True):
        # Direct substring or "stripped" substring match
        key_no_space = key.replace(" ", "")
        if key in lc or key_no_space in lc:
            return NAME_TO_TICKER[key]
    return None


def _normalize_symbol(sym: str) -> str:
    """Match the format we use elsewhere (NSE suffix .NS).

    Handles three input shapes:
      1. "RELIANCE" / "RELIANCE.NS" / "RELIANCE-EQ"  → "RELIANCE.NS"
      2. "Reliance Industries Limited"               → "RELIANCE.NS"  (name lookup)
      3. Anything else                               → "<UPPERCASED>.NS" (best-effort)
    """
    s_raw = str(sym).strip()
    # Highest priority: an explicit correction for a known-bad broker symbol.
    corrected = _apply_correction(s_raw)
    if corrected is not False:
        # str → use the corrected ticker; None → untradable instrument (SGB,
        # some ETFs): keep the original symbol unchanged so it stays visible,
        # flagged by the data-health check, and the remap stays idempotent.
        return corrected if corrected else s_raw
    upper_check = s_raw.upper()
    # Already carries an NSE/BSE suffix and survived the correction check above →
    # it's a finished ticker. Return as-is (idempotent; avoids re-appending .NS
    # to longer symbols like ADANIPOWER.NS that fail the length heuristic below).
    if upper_check.endswith(".NS") or upper_check.endswith(".BO"):
        return upper_check
    # Looks like a real ticker already? Real NSE tickers are short (≤12) and
    # don't end in common suffix words like "LIMITED" / "LTD" / "INDUSTRIES".
    looks_like_ticker = (
        " " not in s_raw
        and len(s_raw) <= 12
        and not any(upper_check.endswith(suf) for suf in
                    ("LIMITED", "LTD", "CORPORATION", "INDUSTRIES", "COMPANY"))
    )
    if looks_like_ticker:
        s = upper_check
        # Strip ONLY known NSE series codes (-EQ, -BE, -BL, -BZ, -SM, -ST).
        # Keep legitimate hyphenated tickers like BAJAJ-AUTO intact.
        for series in ("-EQ", "-BE", "-BL", "-BZ", "-SM", "-ST"):
            if s.endswith(series):
                s = s[: -len(series)]
                break
        if "." in s:
            return s
        return f"{s}.NS"
    # Looks like a company name — try the resolver
    resolved = _resolve_from_name(s_raw)
    if resolved:
        return resolved
    # Last-ditch: uppercase and slap on .NS (will probably fail yfinance lookup
    # but at least it's a deterministic transformation the user can correct)
    return f"{s_raw.upper().replace(' ', '')}.NS"


def _find_header_row(raw: pd.DataFrame, max_scan: int = 40) -> int:
    """Groww (and many broker exports) put a metadata preamble before the
    actual table. Scan the first `max_scan` rows looking for one that
    contains recognisable column names. Returns 0-based row index.

    A row qualifies as the header if its cells collectively cover ≥ 3
    DISTINCT category keywords. This prevents false matches on metadata
    rows like 'Name, Shreyash Pandey' where 'name' alone shouldn't count.
    """
    # Grouped so each cell can claim at most one category hit. This means
    # a row needs at least 3 different column types to qualify.
    keyword_groups = [
        {"stock", "symbol", "scrip", "isin", "instrument", "ticker", "tradingsymbol"},
        {"quantity", "qty", "shares", "holdings"},
        {"average", "avg", "buy price", "cost", "avg price"},
        {"current price", "ltp", "market price", "cmp", "live price"},
        {"invested", "investment", "buy value"},
        {"current value", "market value", "current val"},
        {"p&l", "pnl", "profit", "return", "% change", "% gain"},
    ]

    def _categories_hit(row) -> int:
        cells = [str(c).strip().lower() for c in row.tolist() if pd.notna(c)]
        hit = 0
        for group in keyword_groups:
            for c in cells:
                if any(kw in c for kw in group):
                    hit += 1
                    break  # only one hit per category per row
        return hit

    best_row, best_score = 0, 0
    for i in range(min(max_scan, len(raw))):
        score = _categories_hit(raw.iloc[i])
        if score >= 3 and score > best_score:
            best_row, best_score = i, score
    return best_row


def _read_raw(path: Path) -> pd.DataFrame:
    """Read the file with no header so we can scan for the real one.

    Broker exports often have variable column counts (e.g. an 8-column
    metadata header followed by a 9-column data table). pandas can't handle
    this directly — it locks columns to the first row's width and either
    errors or silently drops the longer rows.

    Fix: pre-scan with the csv module to find the MAX column count, then
    read with that many synthetic column names so no row is truncated.
    """
    if path.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(path, header=None)

    import csv
    max_cols = 0
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            max_cols = max(max_cols, len(row))
    if max_cols == 0:
        return pd.DataFrame()
    names = list(range(max_cols))
    return pd.read_csv(path, header=None, names=names, skip_blank_lines=True,
                       engine="python")


def parse(path: str | Path, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a holdings CSV/Excel into a normalized DataFrame.

    Required logical columns: symbol, qty, avg_price.
    Other columns are derived if missing.

    Tolerant of the broker-style preamble (metadata rows above the table) —
    we scan for the real header row before parsing the body.
    """
    p = Path(path)
    raw = _read_raw(p)
    header_row = _find_header_row(raw)
    # Re-read using the detected header row. Use raw to extract the actual
    # header values, then build a regular DataFrame from the body so we
    # don't fight pandas' column-count lock-in.
    if p.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(p, header=header_row)
    else:
        # Trim raw to header_row..end, promote header_row to columns,
        # drop fully-empty columns (Groww often pads with trailing commas).
        header_values = raw.iloc[header_row].tolist()
        body = raw.iloc[header_row + 1:].reset_index(drop=True)
        # Rename columns using the detected header, fall back to position
        rename_map = {}
        for i, h in enumerate(header_values):
            if pd.notna(h) and str(h).strip():
                rename_map[i] = str(h).strip()
        body = body.rename(columns=rename_map)
        # Drop columns whose names are still numeric (empty headers)
        body = body[[c for c in body.columns if isinstance(c, str)]]
        df = body
    # Drop blank or all-NaN rows
    df = df.dropna(how="all")
    # Drop columns that are entirely empty (Groww often has these as separators)
    df = df.dropna(axis=1, how="all")
    df = _normalize_columns(df, overrides)

    missing = {"symbol", "qty", "avg_price"} - set(df.columns)
    if missing:
        # Show first few rows so the user can spot where the real headers live
        sample = df.head(3).to_string(index=False)
        raise ValueError(
            f"missing required columns: {missing}.\n"
            f"Detected columns after auto-header-scan: {list(df.columns)}\n"
            f"First 3 rows:\n{sample}\n\n"
            f"If the headers above don't look right, use --map to override, e.g.:\n"
            f"  --map symbol='Stock Name',qty=Quantity,avg_price='Avg. buy price'"
        )

    df["symbol"] = df["symbol"].apply(_normalize_symbol)
    # Coerce all known-numeric columns. The header-scan path leaves them as
    # strings, so explicit conversion is required before arithmetic.
    for col in ("qty", "avg_price", "last_price", "invested",
                "current_value", "pnl", "pnl_pct"):
        if col in df.columns:
            # Strip currency symbols and commas before numeric conversion
            df[col] = df[col].astype(str).str.replace(r"[₹$,]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["symbol", "qty", "avg_price"])
    df = df[df["qty"] > 0]

    # Derive missing columns from what we have
    if "invested" not in df.columns:
        df["invested"] = df["qty"] * df["avg_price"]
    if "current_value" not in df.columns and "last_price" in df.columns:
        df["current_value"] = df["qty"] * df["last_price"]
    if "pnl" not in df.columns and "current_value" in df.columns:
        df["pnl"] = df["current_value"] - df["invested"]
    if "pnl_pct" not in df.columns and "pnl" in df.columns:
        df["pnl_pct"] = (df["pnl"] / df["invested"]) * 100

    return df.reset_index(drop=True)


def import_csv(path: str | Path, overrides: dict[str, str] | None = None,
               source: str = "groww") -> int:
    df = parse(path, overrides)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "ticker": r["symbol"],
            "qty": float(r["qty"]),
            "avg_price": float(r["avg_price"]),
            "last_price": float(r["last_price"]) if "last_price" in df.columns and pd.notna(r["last_price"]) else None,
            "invested": float(r["invested"]) if pd.notna(r.get("invested")) else None,
            "current_value": float(r["current_value"]) if "current_value" in df.columns and pd.notna(r["current_value"]) else None,
            "pnl": float(r["pnl"]) if "pnl" in df.columns and pd.notna(r["pnl"]) else None,
            "pnl_pct": float(r["pnl_pct"]) if "pnl_pct" in df.columns and pd.notna(r["pnl_pct"]) else None,
        })
    n = replace_holdings(rows, source=source)
    console.print(f"[green]imported {n} holdings from {path}[/green]")
    return n


# ---- analysis ----------------------------------------------------------------

@dataclass
class HoldingAnalysis:
    ticker: str
    qty: float
    avg_price: float
    last_price: float | None
    pnl_pct: float | None
    portfolio_weight: float | None     # fraction of total portfolio value
    # Lens scores
    quality_score: float | None
    quality_verdict: str               # "strong" | "ok" | "weak" | "unknown"
    technical_state: str               # "uptrend" | "weakening" | "broken" | "unknown"
    rsi: float | None
    above_50ema: bool | None
    above_200ema: bool | None
    # Volume-flow
    mfi: float | None = None           # Money Flow Index — volume-weighted RSI
    buy_pressure_20d: float | None = None
    sentiment_bullish: int = 0
    sentiment_bearish: int = 0
    sentiment_top_event: str | None = None
    active_setup: str | None = None
    # Exit plan
    initial_stop: float | None = None
    trailing_stop: float | None = None
    book_partial_at: float | None = None
    full_target: float | None = None
    risk_reward: float | None = None
    # Trend-quality classification (real / weak / false uptrend)
    trend_verdict: str | None = None
    trend_label: str | None = None
    # Early-exit warnings (over and above the BUY/HOLD/SELL recommendation)
    early_exit_action: str | None = None         # EXIT | TRIM_50 | TRIM_25 | WATCH | NONE
    early_warnings: list[str] = field(default_factory=list)
    # AI thesis
    ai_narrative: str | None = None
    ai_conviction: int | None = None
    ai_action: str | None = None
    ai_risks: list[str] | None = None
    ai_catalyst: str | None = None
    # Output
    recommendation: str = "HOLD"        # BUY_MORE | HOLD | REDUCE | SELL | NO_DATA
    reasons: list[str] = field(default_factory=list)


def _technical_state(df: pd.DataFrame) -> tuple[str, float | None, bool | None, bool | None]:
    """Classify technical state from indicator-augmented OHLCV."""
    if df.empty or len(df) < 50:
        return "unknown", None, None, None
    last = df.iloc[-1]
    rsi = float(last["rsi14"]) if pd.notna(last.get("rsi14")) else None
    close = float(last["close"])
    above_50 = (pd.notna(last.get("ema50")) and close > last["ema50"]) if "ema50" in df.columns else None
    above_200 = (pd.notna(last.get("ema200")) and close > last["ema200"]) if "ema200" in df.columns else None

    if above_50 and above_200:
        state = "uptrend"
    elif above_200 and not above_50:
        state = "weakening"        # above long-term, below medium-term
    elif not above_50 and not above_200:
        state = "broken"
    else:
        state = "ok"
    return state, rsi, above_50, above_200


def _quality_verdict(q: float | None) -> str:
    if q is None:
        return "unknown"
    if q >= 75:
        return "strong"
    if q >= 60:
        return "ok"
    return "weak"


def _recommend(a: HoldingAnalysis) -> tuple[str, list[str]]:
    """Apply the recommendation rubric. Returns (recommendation, reasons)."""
    reasons: list[str] = []

    # If we have zero data, can't recommend
    if a.quality_score is None and a.technical_state == "unknown":
        return "NO_DATA", ["no fundamentals or price history yet"]

    sell_pts = 0
    buy_pts = 0

    # --- SELL signals ---
    if a.quality_verdict == "weak":
        sell_pts += 2
        reasons.append("weak fundamentals (quality < 60)")
    if a.technical_state == "broken":
        sell_pts += 2
        reasons.append("below both 50 + 200 EMA")
    if a.sentiment_bearish >= 2 and a.sentiment_bullish == 0:
        sell_pts += 1
        reasons.append(f"{a.sentiment_bearish} bearish news, no bullish offset")
    if a.rsi is not None and a.rsi < 30 and a.technical_state in ("broken", "weakening"):
        sell_pts += 1
        reasons.append(f"oversold (RSI {a.rsi:.0f}) in downtrend")
    if a.portfolio_weight is not None and a.portfolio_weight > 0.35:
        sell_pts += 1
        reasons.append(f"oversized position ({a.portfolio_weight*100:.0f}% of portfolio)")

    # --- BUY-MORE signals ---
    if a.quality_verdict == "strong" and a.technical_state == "uptrend":
        buy_pts += 2
        reasons.append("strong fundamentals + uptrend")
    if a.sentiment_bullish >= 2 and a.sentiment_bearish == 0:
        buy_pts += 1
        reasons.append(f"{a.sentiment_bullish} bullish news, no bearish")
    if a.active_setup:
        buy_pts += 1
        reasons.append(f"fresh signal: {a.active_setup}")

    # Net decision
    net = buy_pts - sell_pts
    if sell_pts >= 3 and buy_pts == 0:
        rec = "SELL"
    elif net >= 2:
        rec = "BUY_MORE"
    elif net <= -1:
        rec = "REDUCE"
    else:
        rec = "HOLD"
        if not reasons:
            reasons.append("no strong signal in either direction")
    return rec, reasons


def analyze_one(ticker: str, qty: float, avg_price: float,
                last_price: float | None = None,
                portfolio_value: float | None = None,
                pnl_pct: float | None = None,
                use_ai_thesis: bool = False) -> HoldingAnalysis:
    """Run all lenses on one holding: fundamentals + technicals + sentiment
    + volume flow + exit levels + (optionally) AI thesis from Claude."""
    # Quality
    fund = get_fundamentals(ticker)
    qscore = fund.get("quality_score") if fund else None

    # Technicals + volume flow — need indicator-augmented price frame
    df = load_prices(ticker)
    technical_state = "unknown"
    rsi = above_50 = above_200 = mfi = buy_pressure = None
    if not df.empty and len(df) >= 50:
        df = add_indicators(df)
        technical_state, rsi, above_50, above_200 = _technical_state(df)
        last = df.iloc[-1]
        mfi = float(last["mfi14"]) if pd.notna(last.get("mfi14")) else None
        buy_pressure = float(last["buy_pressure_20"]) if pd.notna(last.get("buy_pressure_20")) else None

    # Backfill a live price from the latest bar when the broker export omitted
    # one (Groww frequently does) — sizing, P&L and weight all need it.
    if last_price is None and not df.empty:
        last_price = float(df.iloc[-1]["close"])

    # Sentiment — last 7 days
    sent_df = recent_sentiment_for_ticker(ticker, days=7)
    bull = bear = 0
    top_event = None
    if not sent_df.empty:
        bull = int((sent_df["sentiment"] == "bullish").sum())
        bear = int((sent_df["sentiment"] == "bearish").sum())
        high_impact = sent_df[sent_df["impact"] == "high"]
        if not high_impact.empty:
            top_event = str(high_impact.iloc[0].get("event_type"))

    # Active setup
    active_setup = None
    sigs = scan_ticker(ticker)
    if sigs:
        active_setup = sigs[0]["setup"]

    # Portfolio weight
    weight = None
    if portfolio_value and last_price:
        weight = (qty * last_price) / portfolio_value

    # Exit plan
    plan = exits_mod.compute(ticker, avg_buy=avg_price)

    # Trend-quality read
    tq = None
    if not df.empty and len(df) >= 60:
        tq = trend_quality(df)
    # Early-exit warning evaluation
    ee = early_exits_mod.evaluate(ticker)

    a = HoldingAnalysis(
        ticker=ticker, qty=qty, avg_price=avg_price, last_price=last_price,
        pnl_pct=pnl_pct, portfolio_weight=weight,
        quality_score=qscore, quality_verdict=_quality_verdict(qscore),
        technical_state=technical_state, rsi=rsi,
        above_50ema=above_50, above_200ema=above_200,
        mfi=mfi, buy_pressure_20d=buy_pressure,
        sentiment_bullish=bull, sentiment_bearish=bear,
        sentiment_top_event=top_event, active_setup=active_setup,
        initial_stop=plan.initial_stop if plan else None,
        trailing_stop=plan.trailing_stop if plan else None,
        book_partial_at=plan.book_partial_at if plan else None,
        full_target=plan.full_target if plan else None,
        risk_reward=plan.risk_reward if plan else None,
        trend_verdict=(tq or {}).get("verdict") if tq else None,
        trend_label=(tq or {}).get("label") if tq else None,
        early_exit_action=ee.action,
        early_warnings=[w.reason for w in ee.warnings],
    )
    a.recommendation, a.reasons = _recommend(a)

    # If early-exit action is CRITICAL/HIGH, force the recommendation
    # to reflect it (so the BUY/HOLD/SELL column matches reality).
    if ee.action == "EXIT" and a.recommendation in ("BUY_MORE", "HOLD"):
        a.recommendation = "SELL"
        a.reasons = [f"early-exit signal: {a.early_warnings[0] if a.early_warnings else 'multiple warnings'}"]
    elif ee.action in ("TRIM_50", "TRIM_25") and a.recommendation == "BUY_MORE":
        a.recommendation = "REDUCE"
        a.reasons = [f"early-exit signal: {a.early_warnings[0] if a.early_warnings else 'warnings present'}"]

    # AI thesis (lazy import to avoid loading anthropic when not needed)
    if use_ai_thesis:
        from swingdesk.analyze import thesis as thesis_mod
        # Build recent_news df for the thesis (top 5 by published)
        news_df = load_news(limit=10, ticker=ticker)
        tech_state = {
            "state": technical_state, "rsi": rsi,
            "above_50ema": above_50, "above_200ema": above_200,
            "mfi": mfi, "buy_pressure": buy_pressure,
            "active_setup": active_setup,
        }
        t = thesis_mod.generate(
            ticker=ticker, qty=qty, avg_buy=avg_price, last_price=last_price,
            pnl_pct=pnl_pct, fundamentals=fund, technical_state=tech_state,
            recent_news=news_df,
        )
        if t is not None:
            a.ai_narrative = t.narrative
            a.ai_conviction = t.conviction
            a.ai_action = t.action
            a.ai_risks = t.risks
            a.ai_catalyst = t.catalyst_to_watch

    return a


def remap_existing_tickers() -> dict:
    """Re-run ticker normalisation (including TICKER_CORRECTIONS) over the
    holdings already saved in the DB and re-save them with corrected symbols.

    Lets a user fix a bad import without re-uploading the file. Returns
    {"changed": [(old, new), ...], "untradable": [tickers with no equity data]}.
    """
    from swingdesk.storage import load_holdings, replace_holdings

    df = load_holdings()
    if df.empty:
        return {"changed": [], "untradable": []}

    keep = ["ticker", "qty", "avg_price", "last_price", "invested",
            "current_value", "pnl", "pnl_pct", "source"]
    changed: list[tuple[str, str]] = []
    untradable: list[str] = []
    rows: list[dict] = []
    for _, r in df.iterrows():
        old = r["ticker"]
        new = _normalize_symbol(old)
        if _apply_correction(old) is None:        # explicit "no equity series"
            untradable.append(old)
        if new != old:
            changed.append((old, new))
        row = {}
        for c in keep:
            v = r.get(c)
            row[c] = None if (v is None or (isinstance(v, float) and pd.isna(v))) else v
        row["ticker"] = new
        rows.append(row)
    replace_holdings(rows)
    return {"changed": changed, "untradable": untradable}


def analyze_portfolio(holdings_df: pd.DataFrame,
                      use_ai_thesis: bool = False) -> list[HoldingAnalysis]:
    """Run analysis on every row in the holdings dataframe."""
    portfolio_value = float(holdings_df["current_value"].sum()) if "current_value" in holdings_df.columns else None

    results: list[HoldingAnalysis] = []
    for _, r in holdings_df.iterrows():
        a = analyze_one(
            ticker=r["ticker"],
            qty=float(r["qty"]),
            avg_price=float(r["avg_price"]),
            last_price=float(r["last_price"]) if pd.notna(r.get("last_price")) else None,
            portfolio_value=portfolio_value,
            pnl_pct=float(r["pnl_pct"]) if pd.notna(r.get("pnl_pct")) else None,
            use_ai_thesis=use_ai_thesis,
        )
        results.append(a)
    return results


def portfolio_summary(analyses: list[HoldingAnalysis]) -> dict:
    """Aggregate portfolio-level observations."""
    if not analyses:
        return {}
    sectors_concentration = {}
    for a in analyses:
        fund = get_fundamentals(a.ticker)
        sector = fund["sector"] if fund and fund.get("sector") else "Unknown"
        sectors_concentration[sector] = sectors_concentration.get(sector, 0) + (a.portfolio_weight or 0)

    buy_more = [a.ticker for a in analyses if a.recommendation == "BUY_MORE"]
    sell = [a.ticker for a in analyses if a.recommendation == "SELL"]
    reduce = [a.ticker for a in analyses if a.recommendation == "REDUCE"]
    hold = [a.ticker for a in analyses if a.recommendation == "HOLD"]

    concentrated = [a for a in analyses if (a.portfolio_weight or 0) > 0.25]
    sector_concentrated = [s for s, w in sectors_concentration.items() if w > 0.40]

    return {
        "n_holdings": len(analyses),
        "buy_more": buy_more,
        "sell": sell,
        "reduce": reduce,
        "hold": hold,
        "concentrated_positions": [a.ticker for a in concentrated],
        "sector_concentration": sectors_concentration,
        "concentrated_sectors": sector_concentrated,
    }
