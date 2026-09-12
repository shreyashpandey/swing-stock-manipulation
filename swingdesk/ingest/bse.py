"""Pull BSE bulk/block deals through BSE's public API.

This complements the existing NSE-only deal ingestion so Institutional Flow can
see a broader public disclosed tape. BSE exposes a date-range JSON API used by
its own bulk/block deal page.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import requests
from rich.console import Console

from swingdesk.storage import connect, upsert_deals

console = Console()

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.bseindia.com",
    "Referer": "https://www.bseindia.com/markets/equity/EQReports/BulknBlockDeals?flag=1",
}
_API = "https://api.bseindia.com/BseIndiaAPI/api/BulkDealData_ng/w"
_TYPE_MAP = {"bulk": "1", "block": "2"}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


def _wanted_symbols(tickers: list[str] | None) -> set[str] | None:
    if not tickers:
        return None
    out: set[str] = set()
    for ticker in tickers:
        t = str(ticker or "").strip().upper()
        if not t:
            continue
        if t.endswith(".NS") or t.endswith(".BO"):
            t = t[:-3]
        out.add(t)
    return out or None


def _known_nse_symbols() -> set[str]:
    q = """
        SELECT ticker FROM watchlist
        UNION
        SELECT ticker FROM holdings
        UNION
        SELECT ticker FROM fundamentals
        UNION
        SELECT DISTINCT ticker FROM prices
    """
    with connect() as con:
        rows = [str(r[0]).strip().upper() for r in con.execute(q).fetchall() if r[0]]
    return {r[:-3] for r in rows if r.endswith(".NS")}


def _preferred_ticker(symbol: str, known_nse_symbols: set[str]) -> str:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return ""
    return f"{sym}.NS" if sym in known_nse_symbols else f"{sym}.BO"


def _records_to_rows(records: list[dict], deal_type: str, wanted: set[str] | None,
                     known_nse_symbols: set[str]) -> list[dict]:
    rows: list[dict] = []
    for rec in records:
        symbol = str(rec.get("scripname") or "").strip().upper()
        if not symbol or (wanted is not None and symbol not in wanted):
            continue
        raw_date = rec.get("DEAL_DATE")
        try:
            iso = pd.to_datetime(raw_date).date().isoformat()
        except Exception:
            continue
        side = str(rec.get("TRANSACTION_TYPE") or "").strip().upper()
        rows.append({
            "exchange": "BSE",
            "deal_type": deal_type,
            "date": iso,
            "ticker": _preferred_ticker(symbol, known_nse_symbols),
            "security": symbol,
            "client": str(rec.get("CLIENT_NAME") or "").strip() or None,
            "side": "BUY" if side.startswith("P") else "SELL" if side.startswith("S") else side,
            "qty": float(rec.get("QUANTITY") or 0) or None,
            "price": float(rec.get("PRICE") or 0) or None,
        })
    return rows


def fetch_deals(start: date, end: date, deal_type: str = "bulk",
                tickers: list[str] | None = None,
                session: requests.Session | None = None) -> list[dict]:
    """Fetch BSE bulk/block deals for a date range as common-schema rows."""
    s = session or _session()
    wanted = _wanted_symbols(tickers)
    known_nse = _known_nse_symbols()
    params = {
        "DealType": _TYPE_MAP[deal_type],
        "sc_code": "",
        "FDate": start.strftime("%d/%m/%Y"),
        "TDate": end.strftime("%d/%m/%Y"),
    }
    try:
        r = s.get(_API, params=params, timeout=25)
        if r.status_code != 200:
            return []
        payload = r.json()
    except Exception as e:
        console.print(f"[yellow]BSE {deal_type} fetch failed: {e}[/yellow]")
        return []
    return _records_to_rows(payload.get("Table") or [], deal_type, wanted, known_nse)


def ingest_deals(tickers: list[str] | None = None, days: int = 30,
                 end: date | None = None) -> int:
    """Fetch recent BSE bulk + block deals and store them."""
    end = end or datetime.now().date()
    start = end - timedelta(days=max(days - 1, 0))
    s = _session()
    total = 0
    for deal_type in ("bulk", "block"):
        rows = fetch_deals(start, end, deal_type=deal_type, tickers=tickers, session=s)
        total += upsert_deals(rows)
        console.print(f"  bse {deal_type} deals: {len(rows)} rows")
    console.print(f"[green]saved {total} BSE deal rows[/green]")
    return total
