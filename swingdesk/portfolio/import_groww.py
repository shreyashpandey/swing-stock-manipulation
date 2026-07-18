"""Import trade history from Groww (or any broker) via CSV.

Groww doesn't publish a stable CSV schema, so this importer is flexible:
it accepts any CSV with common column names and infers the rest. Use
`--map name=Symbol,qty=Quantity,...` to override the auto-detected mapping.

Expected logical fields:
    symbol    — stock ticker (we append .NS automatically if missing)
    side      — buy/sell  (case-insensitive)
    qty       — share count (int)
    price     — execution price per share
    date      — trade date (any format pandas can parse)
    [exchange] — optional, defaults to NSE

The importer matches buy/sell pairs by FIFO per ticker:
    each buy opens a position; the next same-ticker sell of equal or
    greater qty closes it (partial fills get split across positions).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from rich.console import Console

from swingdesk.storage import insert_position, update_position, upsert_trades

console = Console()

# Common column name variants seen in Indian broker exports.
COLUMN_ALIASES = {
    "symbol":   ["symbol", "stock", "stock name", "tradingsymbol",
                 "scrip", "scrip code", "instrument", "ticker"],
    "side":     ["side", "trade type", "type", "transaction", "action", "buy/sell"],
    "qty":      ["qty", "quantity", "shares", "qty.", "no. of shares"],
    "price":    ["price", "avg price", "average price", "rate", "trade price"],
    "date":     ["date", "trade date", "executed at", "order date", "timestamp"],
    "exchange": ["exchange", "venue", "exchange name"],
    # Optional — picked up only if present.
    "segment":  ["product", "product type", "producttype", "order type", "ordertype"],
    "charges":  ["charges", "total charges", "taxes & charges", "taxes and charges",
                 "brokerage & charges", "total taxes"],
    "order_id": ["order id", "order no", "order_id", "trade id", "tradeid"],
}


def _norm_symbol(sym: str) -> str:
    """Use holdings.py's robust normaliser (TICKER_CORRECTIONS + name lookup) so
    imported tickers match the rest of the app; fall back to the simple form."""
    try:
        from swingdesk.portfolio.holdings import _normalize_symbol as _h
        return _h(sym)
    except Exception:
        return _normalize_symbol(sym)


def _norm_segment(val) -> str:
    """Map a broker product/segment code to 'delivery' | 'intraday'."""
    s = str(val or "").upper()
    if any(t in s for t in ("MIS", "INTRADAY", "INTRA")):
        return "intraday"
    return "delivery"   # CNC / DELIVERY / NRML / blank → delivery (the swing default)


def _dedupe_key(date_s: str, ticker: str, side: str, qty: float, price: float) -> str:
    raw = f"{date_s}|{ticker}|{side}|{qty}|{price}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _normalize_columns(df: pd.DataFrame, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Rename columns to the canonical logical names. Returns a new DataFrame."""
    overrides = overrides or {}
    lower = {c.lower().strip(): c for c in df.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        # explicit override wins
        if canonical in overrides:
            rename[overrides[canonical]] = canonical
            continue
        for alias in aliases:
            if alias in lower:
                rename[lower[alias]] = canonical
                break
    return df.rename(columns=rename)


def _normalize_symbol(sym: str) -> str:
    """Ensure NSE suffix. 'RELIANCE' → 'RELIANCE.NS', 'TCS-EQ' → 'TCS.NS'."""
    s = str(sym).strip().upper()
    # Strip series suffixes like -EQ, -BE
    if "-" in s:
        s = s.split("-")[0]
    if "." in s:
        return s  # already suffixed
    return f"{s}.NS"


def parse_csv(path: str | Path, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Load and normalize a broker CSV. Returns DataFrame with canonical columns."""
    df = pd.read_csv(path)
    df = _normalize_columns(df, overrides)
    required = {"symbol", "side", "qty", "price", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"missing required columns: {missing}. "
            f"Found: {list(df.columns)}. Use --map to override (e.g. --map symbol=Stock)."
        )
    df["symbol"] = df["symbol"].apply(_norm_symbol)
    # 'B'/'BUY'/'Buy' → buy, 'S'/'SELL' → sell.
    df["side"] = df["side"].astype(str).str.lower().str.strip().map(
        lambda v: "buy" if v in ("b", "buy") else "sell" if v in ("s", "sell") else v)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").astype("Int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    # Try ISO first (YYYY-MM-DD), fall back to dayfirst for DD/MM/YYYY.
    parsed = pd.to_datetime(df["date"], errors="coerce", format="ISO8601")
    if parsed.isna().any():
        parsed = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["date"] = parsed
    # Optional columns (only coerce if the export carried them).
    if "charges" in df.columns:
        df["charges"] = (df["charges"].astype(str).str.replace(r"[₹,]", "", regex=True)
                         .pipe(pd.to_numeric, errors="coerce"))
    df = df.dropna(subset=["symbol", "side", "qty", "price", "date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def import_trades(path: str | Path, *, overrides: dict[str, str] | None = None,
                  is_paper: bool = False) -> dict:
    """FIFO-match buys and sells from a broker CSV into positions.

    Buys open new positions (is_paper=False by default — these are real trades
    from your broker, not paper). Sells close the oldest open position on the
    same ticker. Partial sells split the remaining qty into a new position.
    """
    df = parse_csv(path, overrides)
    if df.empty:
        return {"buys": 0, "sells": 0, "matched": 0, "opened": 0}

    # In-memory FIFO queue per ticker: list of {pos_id, qty_remaining, entry_price}
    open_q: dict[str, list[dict]] = {}
    buys = sells = matched = opened = 0

    for _, row in df.iterrows():
        sym = row["symbol"]
        qty = int(row["qty"])
        price = float(row["price"])
        date_s = row["date"].strftime("%Y-%m-%d")

        if row["side"] == "buy":
            buys += 1
            pos_id = insert_position({
                "ticker": sym,
                "setup": "imported",
                "side": "long",
                "qty": qty,
                "entry_price": price,
                "entry_date": date_s,
                "stoploss": None,
                "target": None,
                "is_paper": int(is_paper),
                "signal_id": None,
                "notes": "imported from broker CSV",
            })
            opened += 1
            open_q.setdefault(sym, []).append(
                {"pos_id": pos_id, "qty_remaining": qty, "entry_price": price}
            )
        elif row["side"] == "sell":
            sells += 1
            queue = open_q.get(sym, [])
            remaining_to_sell = qty
            while remaining_to_sell > 0 and queue:
                head = queue[0]
                consumed = min(head["qty_remaining"], remaining_to_sell)
                head["qty_remaining"] -= consumed
                remaining_to_sell -= consumed

                if head["qty_remaining"] == 0:
                    # Full close
                    pnl = (price - head["entry_price"]) * consumed
                    update_position(
                        head["pos_id"],
                        status="closed",
                        exit_price=price,
                        exit_date=date_s,
                        exit_reason="broker_sell",
                        pnl=round(pnl, 2),
                        pnl_pct=round((price - head["entry_price"]) / head["entry_price"] * 100, 3),
                    )
                    queue.pop(0)
                    matched += 1
                else:
                    # Partial close: shrink the original position to the closed portion
                    # and insert a new open position for the remainder.
                    pnl = (price - head["entry_price"]) * consumed
                    update_position(
                        head["pos_id"],
                        qty=consumed,
                        status="closed",
                        exit_price=price,
                        exit_date=date_s,
                        exit_reason="broker_sell_partial",
                        pnl=round(pnl, 2),
                        pnl_pct=round((price - head["entry_price"]) / head["entry_price"] * 100, 3),
                    )
                    new_id = insert_position({
                        "ticker": sym,
                        "setup": "imported",
                        "side": "long",
                        "qty": head["qty_remaining"],
                        "entry_price": head["entry_price"],
                        "entry_date": date_s,  # carry forward but mark today as recreation
                        "is_paper": int(is_paper),
                        "notes": f"residual after partial sell of #{head['pos_id']}",
                    })
                    head["pos_id"] = new_id
                    matched += 1

            if remaining_to_sell > 0:
                console.print(
                    f"[yellow]warning: sell of {qty} {sym} on {date_s} couldn't "
                    f"fully match — {remaining_to_sell} shares unmatched (short?)[/yellow]"
                )

    return {"buys": buys, "sells": sells, "matched": matched, "opened": opened}


def import_tradebook(path: str | Path, *, overrides: dict[str, str] | None = None,
                     source: str = "groww") -> dict:
    """Import a Groww order-history / tradebook export into the `trades` table
    (raw executions). Idempotent — re-importing the same file adds nothing new.

    This is the source the P&L report FIFO-matches into round-trips; it is kept
    separate from `positions` (the paper/real lifecycle) so broker history never
    pollutes the journal. Returns ``{"new": n, "rows": total}``."""
    df = parse_csv(path, overrides)
    if df.empty:
        return {"new": 0, "rows": 0}
    has_seg, has_exch = "segment" in df.columns, "exchange" in df.columns
    has_charges, has_oid = "charges" in df.columns, "order_id" in df.columns
    rows = []
    for _, r in df.iterrows():
        sym, side = r["symbol"], r["side"]
        qty, price = float(r["qty"]), float(r["price"])
        date_s = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
        seg = _norm_segment(r["segment"]) if has_seg else "delivery"
        exch = (str(r["exchange"]).upper() if has_exch and pd.notna(r["exchange"]) else "NSE")
        charges = float(r["charges"]) if has_charges and pd.notna(r["charges"]) else None
        oid = str(r["order_id"]) if has_oid and pd.notna(r["order_id"]) else None
        rows.append({
            "dedupe_key": oid or _dedupe_key(date_s, sym, side, qty, price),
            "ticker": sym, "side": side, "qty": qty, "price": price,
            "trade_date": date_s, "segment": seg, "exchange": exch, "charges": charges,
        })
    new = upsert_trades(rows, source=source)
    console.print(f"  trades: {len(rows)} parsed, {new} new")
    return {"new": new, "rows": len(rows)}


# Groww "Tax P&L" / Capital-Gains export: already-matched round trips.
TAX_PNL_ALIASES = {
    "symbol":     ["symbol", "stock", "stock name", "scrip", "scrip name",
                   "company", "company name", "security", "instrument"],
    "qty":        ["qty", "quantity", "quantity sold", "shares", "sold quantity"],
    "buy_date":   ["buy date", "purchase date", "acquisition date", "buy_date", "date of purchase"],
    "buy_price":  ["buy price", "purchase price", "buy avg", "buy average", "acquisition price",
                   "purchase value per unit"],
    "sell_date":  ["sell date", "sale date", "sell_date", "date of sale"],
    "sell_price": ["sell price", "sale price", "sell avg", "sale value per unit"],
    "gross_pnl":  ["realized p&l", "realised p&l", "realized pnl", "profit", "gain",
                   "net p&l", "pnl", "realised profit", "gain/loss", "p&l"],
    "charges":    ["charges", "total charges", "taxes & charges", "expenses"],
}


def parse_tax_pnl(path: str | Path, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Parse a Groww Tax-P&L / Capital-Gains export into matched round trips.

    Returns canonical columns: ``symbol, qty, buy_date, sell_date, buy_price,
    sell_price`` and, when present, ``gross_pnl`` / ``charges`` (Groww's *actual*
    figures, preferred over the model). Format varies, so a preamble is skipped
    and `overrides` can remap columns (same mechanism as the tradebook)."""
    raw = pd.read_csv(path, header=None, dtype=str)
    header_row = _find_tax_header(raw)
    df = pd.read_csv(path, skiprows=header_row)
    overrides = overrides or {}
    lower = {c.lower().strip(): c for c in df.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in TAX_PNL_ALIASES.items():
        if canonical in overrides:
            rename[overrides[canonical]] = canonical
            continue
        for a in aliases:
            if a in lower:
                rename[lower[a]] = canonical
                break
    df = df.rename(columns=rename)
    required = {"symbol", "qty", "buy_date", "sell_date", "buy_price", "sell_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Tax P&L: missing columns {missing}. Found {list(df.columns)}. "
            "Use column overrides (e.g. symbol=Stock,buy_date=Buy Date).")
    df["symbol"] = df["symbol"].apply(_norm_symbol)
    for c in ("qty", "buy_price", "sell_price", "gross_pnl", "charges"):
        if c in df.columns:
            df[c] = (df[c].astype(str).str.replace(r"[₹,]", "", regex=True)
                     .pipe(pd.to_numeric, errors="coerce"))
    for c in ("buy_date", "sell_date"):
        parsed = pd.to_datetime(df[c], errors="coerce", format="ISO8601")
        if parsed.isna().any():
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        df[c] = parsed
    df = df.dropna(subset=["symbol", "qty", "buy_date", "sell_date", "buy_price", "sell_price"])
    return df.reset_index(drop=True)


def _find_tax_header(raw: pd.DataFrame, max_scan: int = 30) -> int:
    """Find the header row of a Tax-P&L export (Groww prepends a metadata block)."""
    flat_aliases = {a for al in TAX_PNL_ALIASES.values() for a in al}
    for i in range(min(max_scan, len(raw))):
        cells = {str(v).lower().strip() for v in raw.iloc[i].tolist()}
        if len(cells & flat_aliases) >= 3:
            return i
    return 0
