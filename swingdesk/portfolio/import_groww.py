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

from pathlib import Path

import pandas as pd
from rich.console import Console

from swingdesk.storage import insert_position, update_position

console = Console()

# Common column name variants seen in Indian broker exports.
COLUMN_ALIASES = {
    "symbol":   ["symbol", "stock", "stock name", "tradingsymbol",
                 "scrip", "scrip code", "instrument", "ticker"],
    "side":     ["side", "trade type", "type", "transaction", "action", "buy/sell"],
    "qty":      ["qty", "quantity", "shares", "qty.", "no. of shares"],
    "price":    ["price", "avg price", "average price", "rate", "trade price"],
    "date":     ["date", "trade date", "executed at", "order date", "timestamp"],
    "exchange": ["exchange", "segment", "venue"],
}


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
    df["symbol"] = df["symbol"].apply(_normalize_symbol)
    df["side"] = df["side"].astype(str).str.lower().str.strip()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").astype("Int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    # Try ISO first (YYYY-MM-DD), fall back to dayfirst for DD/MM/YYYY.
    parsed = pd.to_datetime(df["date"], errors="coerce", format="ISO8601")
    if parsed.isna().any():
        parsed = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["date"] = parsed
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
