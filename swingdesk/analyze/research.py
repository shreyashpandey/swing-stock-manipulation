"""Local, user-directed research workflows; no network calls or trading actions."""
from __future__ import annotations

import json
import math
import operator
import re

import numpy as np
import pandas as pd

from swingdesk import storage

METRICS = {
    "close": "Close (₹)", "change_1d": "Return across 1 stored observation (%)",
    "change_5d": "Return across 5 stored observations (%)", "change_21d": "2Return across 1 stored observation (%)",
    "rvol": "Volume / prior 20 stored observations average",
    "market_cap_cr": "Market cap (₹ crore)", "trailing_pe": "Trailing P/E",
    "roe_pct": "ROE (%)", "revenue_growth_pct": "Revenue growth (%)",
    "debt_to_equity": "Debt / equity", "quality_score": "Quality score",
}
OPS = {">=": operator.ge, "<=": operator.le}
PRICE_METRICS = ["close", "change_1d", "change_5d", "change_21d", "rvol"]
KINDS = {"watchlist", "screen", "alert"}


def normalize_tickers(tickers: list[str]) -> list[str]:
    result = []
    for raw in tickers:
        ticker = raw.strip().upper()
        if not ticker:
            continue
        if not re.fullmatch(r"[A-Z0-9&^_.=-]{1,40}", ticker):
            raise ValueError(f"Invalid ticker: {ticker}")
        if ticker not in result:
            result.append(ticker)
    return result


def validate_rules(rules: list[dict]) -> None:
    for rule in rules:
        if rule.get("metric") not in METRICS or rule.get("op") not in OPS:
            raise ValueError("Select a supported metric and comparison.")
        if isinstance(rule.get("value"), bool) or not isinstance(rule.get("value"), (int, float)) or not math.isfinite(rule["value"]):
            raise ValueError("Thresholds must be finite numbers.")


def save_item(kind: str, name: str, payload: dict) -> None:
    name = name.strip()
    if kind not in KINDS or not name or len(name) > 80:
        raise ValueError("Choose a valid item type and a name of 1–80 characters.")
    payload = dict(payload)
    payload["tickers"] = normalize_tickers(payload.get("tickers", []))
    if not payload["tickers"]:
        raise ValueError("Add at least one ticker.")
    if kind in {"screen", "alert"}:
        validate_rules(payload.get("rules", []))
    if kind == "alert" and not payload.get("rules"):
        raise ValueError("An alert needs at least one condition.")
    if kind == "alert" and any(r["metric"] not in PRICE_METRICS for r in payload["rules"]):
        raise ValueError("Alerts support daily price and volume metrics only.")
    with storage.connect() as con:
        con.execute(
            "INSERT INTO research_items(kind,name,payload) VALUES (?,?,?) "
            "ON CONFLICT(kind,name) DO UPDATE SET payload=excluded.payload, "
            "updated_at=CURRENT_TIMESTAMP",
            (kind, name, json.dumps(payload, allow_nan=False)),
        )


def saved_items(kind: str) -> dict[str, dict]:
    with storage.connect() as con:
        return {r["name"]: json.loads(r["payload"]) for r in con.execute(
            "SELECT name,payload FROM research_items WHERE kind=? ORDER BY name", (kind,)
        )}


def delete_item(kind: str, name: str) -> None:
    with storage.connect() as con:
        con.execute("DELETE FROM research_items WHERE kind=? AND name=?", (kind, name))


def snapshot(tickers: list[str]) -> pd.DataFrame:
    """Keep missing data visible. Returns count stored observations; window dates expose coverage gaps."""
    funds = storage.load_fundamentals()
    fund_map = funds.set_index("ticker").to_dict("index") if not funds.empty else {}
    rows = []
    for ticker in normalize_tickers(tickers):
        df = storage.load_prices(ticker, days=22)
        f = fund_map.get(ticker, {})
        row = {"ticker": ticker, "sector": f.get("sector") or "Unknown",
               "price_as_of": None, "fundamentals_as_of": f.get("updated_at")}
        row.update({metric: np.nan for metric in METRICS})
        row.update({f"change_{n}d_from": None for n in (1, 5, 21)})
        row["rvol_from"] = None
        if not df.empty:
            close = pd.to_numeric(df.close, errors="coerce")
            row.update(close=close.iloc[-1], price_as_of=str(df.index[-1].date()))
            for sessions in (1, 5, 21):
                if len(close) > sessions and close.iloc[-sessions-1] > 0:
                    row[f"change_{sessions}d_from"] = str(df.index[-sessions-1].date())
                    row[f"change_{sessions}d"] = (close.iloc[-1] / close.iloc[-sessions-1] - 1) * 100
            if len(df) >= 21:
                prior = pd.to_numeric(df.volume.iloc[-21:-1], errors="coerce")
                if prior.notna().all() and prior.mean() > 0:
                    row["rvol_from"] = str(df.index[-21].date())
                    row["rvol"] = df.volume.iloc[-1] / prior.mean()
        for key in ("trailing_pe", "debt_to_equity", "quality_score"):
            row[key] = f.get(key, np.nan)
        for target, source, scale in (("market_cap_cr", "market_cap", 1e-7),
                                      ("roe_pct", "return_on_equity", 100),
                                      ("revenue_growth_pct", "revenue_growth", 100)):
            value = f.get(source)
            row[target] = value * scale if value is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=["ticker", "sector", "price_as_of", "fundamentals_as_of", *METRICS, "change_1d_from", "change_5d_from", "change_21d_from", "rvol_from"]).replace([np.inf, -np.inf], np.nan)


def filter_snapshot(frame: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """AND conditions; unknown values never satisfy an active condition."""
    validate_rules(rules)
    mask = pd.Series(True, index=frame.index)
    for rule in rules:
        values = pd.to_numeric(frame[rule["metric"]], errors="coerce")
        mask &= values.notna() & OPS[rule["op"]](values, rule["value"])
    return frame.loc[mask].copy()


def comparison_series(tickers: list[str], sessions: int = 126) -> pd.DataFrame:
    """Rebase on shared observed dates, without forward-filling absent prices."""
    series = {}
    for ticker in normalize_tickers(tickers):
        prices = storage.load_prices(ticker, days=sessions)
        if prices.empty:
            return pd.DataFrame()
        series[ticker] = pd.to_numeric(prices.close, errors="coerce")
    if not series:
        return pd.DataFrame()
    common = pd.concat(series, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    common = common.loc[(common > 0).all(axis=1)]
    if len(common) < 2:
        return pd.DataFrame()
    return common.div(common.iloc[0]).mul(100)


def evaluate_alerts(today=None) -> pd.DataFrame:
    """Manual condition checks only. Stale/unknown rows cannot report a match."""
    today = pd.Timestamp(today or pd.Timestamp.now(tz="Asia/Kolkata").date()).date()
    rows = []
    for name, item in saved_items("alert").items():
        frame = snapshot(item["tickers"])
        matches = set(filter_snapshot(frame, item["rules"]).ticker)
        for row in frame.to_dict("records"):
            age = ((today - pd.Timestamp(row["price_as_of"]).date()).days
                   if pd.notna(row["price_as_of"]) else None)
            unknown = any(pd.isna(row[r["metric"]]) for r in item["rules"])
            status = ("Missing data" if unknown or age is None else
                      "Stale data" if age > 4 or age < 0 else
                      "Condition met" if row["ticker"] in matches else "Not met")
            rows.append({"alert": name, "status": status, **row})
    return pd.DataFrame(rows)
