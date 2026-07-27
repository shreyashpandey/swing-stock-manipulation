"""Pre-results watchlist scanner.

This is a descriptive event-prep view: upcoming result dates, available
fundamental growth/margin trends, recent institutional tape, brokerage tone, and
pre-result price behavior. It deliberately avoids "buy/sell" language.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from swingdesk.analyze.institutional import BrokerageAction, InstitutionalFlow
from swingdesk.storage import (
    combined_universe,
    load_earnings_calendar,
    load_fundamentals,
    load_prices,
)


@dataclass
class ResultsWatchRow:
    ticker: str
    result_date: str | None
    days_left: int | None
    window: str
    revenue_growth: float | None
    earnings_growth: float | None
    profit_margin: float | None
    operating_margin: float | None
    pre_result_move_pct: float | None
    broker_tone: str
    broker_count: int
    institutional_flow: str
    institutional_value_cr: float
    marquee: str
    gap_risk: str
    readiness: str
    notes: str


def _parse_date(value) -> date | None:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def _pct(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value) * 100, 1)


def _pre_result_move(prices: pd.DataFrame, sessions: int = 20) -> float | None:
    if prices is None or prices.empty or "close" not in prices:
        return None
    close = prices["close"].dropna()
    if len(close) < 2:
        return None
    lookback = min(sessions, len(close) - 1)
    first = float(close.iloc[-(lookback + 1)])
    last = float(close.iloc[-1])
    if first <= 0:
        return None
    return round((last / first - 1) * 100, 1)


def _broker_tone(actions: list[BrokerageAction], ticker: str) -> tuple[str, int]:
    hits = [a for a in actions if a.ticker == ticker]
    if not hits:
        return "No recent call", 0
    bull = sum(1 for a in hits if a.sentiment == "bullish")
    bear = sum(1 for a in hits if a.sentiment == "bearish")
    if bull > bear:
        return "Positive", len(hits)
    if bear > bull:
        return "Cautious", len(hits)
    return "Mixed", len(hits)


def _flow_map(flows: list[InstitutionalFlow]) -> dict[str, InstitutionalFlow]:
    return {f.ticker: f for f in flows}


def _gap_risk(days_left: int | None, move_pct: float | None,
              earnings_growth_pct: float | None, broker_tone: str,
              flow_side: str) -> str:
    points = 0
    if days_left is not None and days_left <= 3:
        points += 2
    elif days_left is not None and days_left <= 10:
        points += 1
    if move_pct is not None and abs(move_pct) >= 10:
        points += 2
    elif move_pct is not None and abs(move_pct) >= 5:
        points += 1
    if earnings_growth_pct is not None and earnings_growth_pct < 0:
        points += 1
    if broker_tone == "Cautious":
        points += 1
    if flow_side == "SELL":
        points += 1
    if points >= 5:
        return "High"
    if points >= 3:
        return "Elevated"
    if points >= 1:
        return "Moderate"
    return "Low"


def _readiness(days_left: int | None, gap_risk: str, broker_tone: str,
               flow_side: str, marquee: str, move_pct: float | None,
               earnings_growth_pct: float | None,
               revenue_growth_pct: float | None,
               profit_margin_pct: float | None) -> tuple[str, str]:
    notes: list[str] = []
    if days_left is None:
        notes.append("Result date unavailable")
    elif days_left <= 3:
        notes.append("Result is very near")
    elif days_left <= 10:
        notes.append("Result is within 10 days")

    if move_pct is not None and move_pct >= 8:
        notes.append("Price has already run up")
    elif move_pct is not None and move_pct <= -8:
        notes.append("Price is weak into results")

    if broker_tone in {"Positive", "Cautious", "Mixed"}:
        notes.append(f"Brokerage tone: {broker_tone.lower()}")
    if flow_side == "BUY":
        notes.append("Disclosed deal flow is net buy")
    elif flow_side == "SELL":
        notes.append("Disclosed deal flow is net sell")
    if marquee:
        notes.append("Marquee institution seen on deal tape")

    growth_ok = any(v is not None and v >= 10 for v in (earnings_growth_pct, revenue_growth_pct))
    margin_ok = profit_margin_pct is not None and profit_margin_pct >= 10

    if gap_risk == "High":
        label = "High gap risk"
    elif flow_side == "BUY" and (marquee or broker_tone == "Positive"):
        label = "Institutional accumulation"
    elif broker_tone == "Cautious" or flow_side == "SELL":
        label = "Expectation caution"
    elif move_pct is not None and move_pct >= 8 and growth_ok:
        label = "Expectation heavy"
    elif growth_ok and margin_ok:
        label = "Clean setup"
    else:
        label = "Watchlist only"
    return label, "; ".join(notes)


def build_results_watch(
    tickers: list[str],
    earnings: pd.DataFrame,
    fundamentals: pd.DataFrame,
    price_fn,
    flows: list[InstitutionalFlow] | None = None,
    actions: list[BrokerageAction] | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """Pure core for tests and UI.

    `fundamentals` uses yfinance-style stored growth/margin values as the
    available proxy. Dedicated analyst-consensus ingestion can later add true
    estimate columns without changing the scanner contract.
    """
    today = today or date.today()
    flows_by_ticker = _flow_map(flows or [])
    actions = actions or []

    earn_by_ticker = {}
    if earnings is not None and not earnings.empty:
        for r in earnings.itertuples():
            earn_by_ticker[getattr(r, "ticker")] = getattr(r, "next_earnings", None)

    fund_by_ticker = {}
    if fundamentals is not None and not fundamentals.empty:
        fund_by_ticker = {r["ticker"]: r for r in fundamentals.to_dict("records")}

    rows: list[ResultsWatchRow] = []
    for ticker in tickers:
        result_date = _parse_date(earn_by_ticker.get(ticker))
        days_left = (result_date - today).days if result_date else None
        if days_left is not None and days_left < -7:
            continue
        if days_left is None:
            window = "Unknown"
        elif days_left < 0:
            window = "Recently declared"
        elif days_left <= 3:
            window = "0-3d"
        elif days_left <= 10:
            window = "4-10d"
        elif days_left <= 30:
            window = "11-30d"
        else:
            window = "Later"

        f = fund_by_ticker.get(ticker, {})
        revenue_growth = _pct(f.get("revenue_growth"))
        earnings_growth = _pct(f.get("earnings_growth"))
        profit_margin = _pct(f.get("profit_margin"))
        operating_margin = _pct(f.get("operating_margin"))
        move_pct = _pre_result_move(price_fn(ticker))
        broker_tone, broker_count = _broker_tone(actions, ticker)
        flow = flows_by_ticker.get(ticker)
        flow_side = flow.net_side if flow else "None"
        flow_value_cr = round((flow.net_value or 0) / 1e7, 2) if flow else 0.0
        marquee = " · ".join(flow.marquee) if flow and flow.marquee else ""
        gap_risk = _gap_risk(days_left, move_pct, earnings_growth, broker_tone, flow_side)
        readiness, notes = _readiness(
            days_left, gap_risk, broker_tone, flow_side, marquee, move_pct,
            earnings_growth, revenue_growth, profit_margin)

        rows.append(ResultsWatchRow(
            ticker=ticker,
            result_date=result_date.isoformat() if result_date else None,
            days_left=days_left,
            window=window,
            revenue_growth=revenue_growth,
            earnings_growth=earnings_growth,
            profit_margin=profit_margin,
            operating_margin=operating_margin,
            pre_result_move_pct=move_pct,
            broker_tone=broker_tone,
            broker_count=broker_count,
            institutional_flow=flow_side,
            institutional_value_cr=flow_value_cr,
            marquee=marquee,
            gap_risk=gap_risk,
            readiness=readiness,
            notes=notes,
        ))

    out = pd.DataFrame([r.__dict__ for r in rows])
    if out.empty:
        return out
    out["_sort_days"] = out["days_left"].fillna(9999)
    out["_sort_risk"] = out["gap_risk"].map({"High": 0, "Elevated": 1, "Moderate": 2, "Low": 3}).fillna(4)
    out = out.sort_values(["_sort_days", "_sort_risk", "ticker"]).drop(columns=["_sort_days", "_sort_risk"])
    return out.reset_index(drop=True)


def scan(tickers: list[str] | None = None, days_ahead: int = 45,
         include_unknown_dates: bool = False) -> pd.DataFrame:
    from swingdesk.analyze import institutional as inst

    tickers = tickers or combined_universe(include_smallcaps=False)
    out = build_results_watch(
        list(tickers),
        load_earnings_calendar(),
        load_fundamentals(),
        load_prices,
        flows=inst.recent_institutional_flow(days=30, only_marquee=False, limit=100),
        actions=inst.brokerage_actions(days=14, limit=100),
    )
    if out.empty:
        return out
    if not include_unknown_dates:
        out = out[out["days_left"].notna()]
    out = out[(out["days_left"].isna()) | (out["days_left"] <= days_ahead)]
    return out.reset_index(drop=True)
