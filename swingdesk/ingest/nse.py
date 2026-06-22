"""Pull NSE data that yfinance doesn't carry — the strongest manipulation tells:

    * **Security-wise delivery %** — of the shares that traded, how many were
      actually *delivered* (taken into a demat account) vs churned intraday. A
      price run-up on *falling* delivery % is the classic operator footprint:
      the move is circular intraday trading, not real accumulation.
    * **Bulk & block deals** — large single-party trades NSE forces to be
      disclosed. Repeated same-party activity around a run-up flags an operator.

All three come from NSE's public CSV archives (no login). NSE is finicky: it
403s API calls without a primed cookie + browser User-Agent, and archive files
simply 404 on market holidays. Everything here fails soft — a missing file or a
blocked request returns empty, never raises, so the daily pipeline keeps going.

Symbol convention: NSE uses bare symbols ("RELIANCE"); the rest of SwingDesk
uses yfinance tickers ("RELIANCE.NS"). We translate at the boundary and store
the yfinance form so delivery/deals join cleanly with prices & fundamentals.
"""
from __future__ import annotations

import io
from datetime import date, datetime, timedelta

import pandas as pd
import requests
from rich.console import Console

from swingdesk.storage import upsert_deals, upsert_delivery

console = Console()

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
_DELIVERY_URL = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
_BULK_URL = "https://archives.nseindia.com/content/equities/bulk.csv"
_BLOCK_URL = "https://archives.nseindia.com/content/equities/block.csv"


# --- symbol <-> ticker helpers -------------------------------------------------
def to_symbol(ticker: str) -> str | None:
    """yfinance ticker -> NSE symbol. Returns None for non-NSE tickers (.BO)."""
    t = ticker.strip().upper()
    if t.endswith(".NS"):
        return t[:-3]
    if "." not in t:
        return t
    return None  # .BO and others aren't on NSE


def to_ticker(symbol: str) -> str:
    """NSE symbol -> yfinance ticker."""
    return f"{symbol.strip().upper()}.NS"


def _session() -> requests.Session:
    """A cookie-primed session. NSE blocks archive fetches until the homepage
    has handed out its cookies to a browser-looking client."""
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=12)
    except requests.RequestException:
        pass  # archives sometimes work without the cookie anyway — try regardless
    return s


def _get_csv(s: requests.Session, url: str) -> pd.DataFrame | None:
    try:
        r = s.get(url, timeout=25)
    except requests.RequestException as e:
        console.print(f"[yellow]NSE fetch failed: {e}[/yellow]")
        return None
    if r.status_code != 200 or not r.text.strip():
        return None
    try:
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]  # NSE pads headers with spaces
    return df


def _num(v) -> float | None:
    """NSE marks non-deliverable rows with '-'. Coerce to float or None."""
    try:
        f = float(str(v).strip().replace(",", ""))
        return f if f == f else None
    except (TypeError, ValueError):
        return None


# --- delivery % ----------------------------------------------------------------
def fetch_delivery_bhavcopy(on: date, session: requests.Session | None = None) -> pd.DataFrame:
    """All EQ-series delivery rows for one trading day. Empty on a holiday/miss.

    Returns columns: symbol, traded_qty, deliv_qty, deliv_pct (one row/symbol)."""
    s = session or _session()
    url = _DELIVERY_URL.format(ddmmyyyy=on.strftime("%d%m%Y"))
    df = _get_csv(s, url)
    if df is None or "SYMBOL" not in df.columns:
        return pd.DataFrame()
    df = df[df.get("SERIES", "").astype(str).str.strip() == "EQ"].copy()
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "symbol": df["SYMBOL"].astype(str).str.strip(),
        "traded_qty": df["TTL_TRD_QNTY"].map(_num),
        "deliv_qty": df["DELIV_QTY"].map(_num),
        "deliv_pct": df["DELIV_PER"].map(_num),
    })
    return out


def ingest_delivery(tickers: list[str], days: int = 20, end: date | None = None) -> int:
    """Backfill delivery % for `tickers` over the last `days` trading sessions.

    Walks business days backwards from `end` (today), fetching one bhavcopy per
    day and keeping only our universe's symbols. Holidays 404 and are skipped.
    Returns the number of (ticker, date) rows written."""
    end = end or datetime.now().date()
    wanted = {to_symbol(t) for t in tickers}
    wanted.discard(None)
    if not wanted:
        console.print("[yellow]no NSE tickers in universe — nothing to fetch[/yellow]")
        return 0

    s = _session()
    written, got_days = 0, 0
    # Pull a few extra calendar days so weekends/holidays don't starve the count.
    bdays = pd.bdate_range(end=pd.Timestamp(end), periods=days + 6)
    for ts in sorted(bdays, reverse=True):
        if got_days >= days:
            break
        day = ts.date()
        bhav = fetch_delivery_bhavcopy(day, session=s)
        if bhav.empty:
            continue
        got_days += 1
        iso = day.isoformat()
        rows = [
            {
                "ticker": to_ticker(r.symbol),
                "date": iso,
                "traded_qty": r.traded_qty,
                "deliv_qty": r.deliv_qty,
                "deliv_pct": r.deliv_pct,
            }
            for r in bhav[bhav["symbol"].isin(wanted)].itertuples()
        ]
        written += upsert_delivery(rows)
        console.print(f"  delivery: {iso} -> {len(rows)} symbols")
    console.print(f"[green]saved {written} delivery rows[/green]")
    return written


# --- bulk / block deals --------------------------------------------------------
def _parse_deals(df: pd.DataFrame, deal_type: str, wanted: set[str] | None) -> list[dict]:
    if df is None or df.empty or "Symbol" not in df.columns:
        return []
    price_col = next((c for c in df.columns if c.lower().startswith("trade price")), None)
    rows = []
    for d in df.to_dict("records"):
        symbol = str(d.get("Symbol", "")).strip().upper()
        if not symbol or (wanted is not None and symbol not in wanted):
            continue
        try:
            iso = pd.to_datetime(d.get("Date"), dayfirst=True).date().isoformat()
        except Exception:
            continue
        rows.append({
            "deal_type": deal_type,
            "date": iso,
            "ticker": to_ticker(symbol),
            "security": str(d.get("Security Name", "")).strip() or None,
            "client": str(d.get("Client Name", "")).strip() or None,
            "side": str(d.get("Buy/Sell", "")).strip().upper() or None,
            "qty": _num(d.get("Quantity Traded")),
            "price": _num(d.get(price_col)) if price_col else None,
        })
    return rows


def ingest_deals(tickers: list[str] | None = None) -> int:
    """Fetch the current bulk + block deal archives and store them. If
    `tickers` is given, only deals for those symbols are kept."""
    s = _session()
    wanted = None
    if tickers:
        wanted = {to_symbol(t) for t in tickers}
        wanted.discard(None)

    total = 0
    for deal_type, url in (("bulk", _BULK_URL), ("block", _BLOCK_URL)):
        df = _get_csv(s, url)
        rows = _parse_deals(df, deal_type, wanted)
        total += upsert_deals(rows)
        console.print(f"  {deal_type} deals: {len(rows)} rows")
    console.print(f"[green]saved {total} deal rows[/green]")
    return total


def ingest(tickers: list[str], days: int = 20) -> dict[str, int]:
    """Convenience: delivery backfill + latest bulk/block deals in one call."""
    return {
        "delivery": ingest_delivery(tickers, days=days),
        "deals": ingest_deals(tickers),
    }
