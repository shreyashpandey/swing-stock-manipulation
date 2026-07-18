"""The 🧭 Board — one fused, ranked, colour-coded view of the whole universe.

Builds on :func:`decision.decide_universe` (which already fuses factor rank +
technicals + trend + manipulation + liquidity + turnover + news + regime) and
adds the missing pieces the user asked for:

  * **US / global per-stock exposure** — NASDAQ-overnight / USD-INR / Brent
    betas via :func:`spillover.stock_sensitivities`, plus a market-level header
    (regime + next-day spillover + market pulse).
  * **Composite labels** — each row gets a single ``category`` (which drives its
    row colour) and a ``tags`` string that *couples* every matching signal
    ("News · Uptrend · Liquid", "⚠ Manip · Turnover surge").
  * **A strict "satisfies all factors" gate** for the top-20 table, with a
    looser top-up when fewer than 20 names clear every check.

The categorisation / tagging / gating are pure functions of a per-row dict so
they're unit-testable without touching the DB or the network.
"""
from __future__ import annotations

import pandas as pd

from swingdesk.analyze import decision as decision_mod
from swingdesk.analyze import spillover as spillover_mod
from swingdesk.config import ACCOUNT_CAPITAL, RISK_PER_TRADE_PCT

# Category → row colour (rgba). Order here is also the precedence order: risk
# flags first so a manipulation/illiquidity warning is never hidden by a
# positive tag. "Neutral" gets no tint.
CATEGORY_COLOR = {
    "🔴 Manipulation":   "rgba(192,57,43,0.28)",
    "🟠 Illiquid":       "rgba(230,126,34,0.25)",
    "🟢 News-backed":    "rgba(31,138,58,0.22)",
    "🔵 Turnover surge": "rgba(41,128,185,0.22)",
    "🟣 US tailwind":    "rgba(142,68,173,0.22)",
    "🟦 Factor leader":  "rgba(22,160,133,0.20)",
    "⚪ Neutral":        "",
}

TURNOVER_SURGE_MULT = 3.0    # today's traded value ≥ N× its 60-day norm
NASDAQ_BETA_HI = 0.4         # "meaningfully US-tech sensitive"
N_GATES = 8                  # number of strict checks

BOARD_COLS = [
    "ticker", "category", "tags", "action", "conviction", "setups", "sentiment_net",
    "turn_today_pct", "turn_avg_pct", "today_x_avg", "liq_tier", "liq_score",
    "float_turnover_pct", "amihud", "manip_tier", "factor_quintile", "tech_tilt",
    "trend_verdict", "expectancy_r", "nasdaq_beta", "us_read", "regime_label",
    "outlook_expected", "entry", "stoploss", "target", "rr", "n_gates", "passes_strict",
]


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


# ---- pure label / gate helpers (no DB) --------------------------------------
def _us_tailwind(r: dict) -> bool:
    nb = r.get("nasdaq_beta")
    exp = r.get("outlook_expected")
    return (nb is not None and nb >= NASDAQ_BETA_HI
            and r.get("regime_label") != "risk-off"
            and (exp is None or exp >= 0))


def categorize(r: dict) -> str:
    """The single category that drives a row's colour (caution wins)."""
    if r.get("manip_tier") in ("High", "Elevated"):
        return "🔴 Manipulation"
    if r.get("liq_tier") in ("illiquid", "untradeable"):
        return "🟠 Illiquid"
    if (r.get("sentiment_net") or 0) >= 2 and r.get("tech_tilt") in ("BUY", "STRONG BUY"):
        return "🟢 News-backed"
    if (r.get("today_x_avg") or 0) >= TURNOVER_SURGE_MULT:
        return "🔵 Turnover surge"
    if _us_tailwind(r):
        return "🟣 US tailwind"
    if r.get("factor_quintile") == 1:
        return "🟦 Factor leader"
    return "⚪ Neutral"


def tags(r: dict) -> str:
    """Couple every matching descriptor (independent of the row colour)."""
    out: list[str] = []
    if r.get("manip_tier") in ("High", "Elevated"):
        out.append("⚠ Manip")
    if r.get("liq_tier") in ("illiquid", "untradeable"):
        out.append("⚠ Illiquid")
    elif r.get("liq_tier") == "liquid":
        out.append("Liquid")
    net = r.get("sentiment_net") or 0
    if net >= 2:
        out.append("News+")
    elif net <= -2:
        out.append("⚠ News−")
    if (r.get("today_x_avg") or 0) >= TURNOVER_SURGE_MULT:
        out.append("Turnover surge")
    if r.get("setups"):
        out.insert(0, "🎯 " + str(r["setups"]).split(",")[0].strip())   # lead with the fired setup
    if r.get("trend_verdict") == "real":
        out.append("Uptrend")
    if _us_tailwind(r):
        out.append("US-tailwind")
    if r.get("factor_quintile") in (1, 2):
        out.append(f"Q{int(r['factor_quintile'])}")
    if r.get("tech_tilt") in ("BUY", "STRONG BUY"):
        out.append("Bull tilt")
    return " · ".join(out) if out else "—"


def gate_flags(r: dict) -> list[bool]:
    """The eight 'satisfies all factors' checks, in order."""
    exp = r.get("expectancy_r")
    return [
        r.get("factor_quintile") in (1, 2),
        r.get("liq_tier") in ("liquid", "moderate"),
        r.get("manip_tier") in ("Low", "n/a"),
        r.get("tech_tilt") in ("BUY", "STRONG BUY"),
        r.get("trend_verdict") in ("real", "weak"),
        (r.get("sentiment_net") or 0) >= 0,
        not (r.get("regime_label") == "risk-off" and (r.get("nasdaq_beta") or 0) >= 1.0),
        exp is None or exp > 0,        # MC skipped in fast mode → don't penalise
    ]


def gate_count(r: dict) -> int:
    return sum(gate_flags(r))


def passes_strict(r: dict) -> bool:
    return all(gate_flags(r))


# ---- global + per-stock context ---------------------------------------------
def global_context() -> dict:
    """Market-level backdrop: regime + next-day US→NIFTY spillover + pulse."""
    from swingdesk.ingest import macro as macro_mod
    return {
        "regime": _safe(spillover_mod.regime),
        "outlook": _safe(spillover_mod.next_day_outlook),
        "pulse": _safe(macro_mod.market_pulse) or {},
    }


def stock_global(ticker: str) -> dict:
    """Per-stock US/global betas from spillover.stock_sensitivities."""
    out = {"nasdaq_beta": None, "usdinr_beta": None, "brent_beta": None, "us_read": ""}
    sens = _safe(spillover_mod.stock_sensitivities, ticker)
    if sens is None or sens.empty:
        return out
    for _, row in sens.iterrows():
        d = str(row["driver"]).upper()
        if "NASDAQ" in d:
            out["nasdaq_beta"] = round(float(row["beta"]), 2)
        elif "USD" in d or "INR" in d:
            out["usdinr_beta"] = round(float(row["beta"]), 2)
        elif "BRENT" in d:
            out["brent_beta"] = round(float(row["beta"]), 2)
    # One-line read = the strongest non-NIFTY driver (df is sorted by |r| desc).
    non_nifty = sens[~sens["driver"].astype(str).str.upper().str.contains("NIFTY")]
    if not non_nifty.empty:
        out["us_read"] = str(non_nifty.iloc[0]["read"])
    return out


def build_board(tickers: list[str], *, capital: float = ACCOUNT_CAPITAL,
                risk_pct: float = RISK_PER_TRADE_PCT, run_montecarlo: bool = False,
                use_ml: bool = False, top_us: int = 40,
                ctx: dict | None = None) -> pd.DataFrame:
    """Rank + categorise the universe. Per-stock US betas are computed only for
    the top ``top_us`` names by conviction (each is an OLS regression)."""
    decisions = decision_mod.decide_universe(
        list(dict.fromkeys(tickers)), capital=capital, risk_pct=risk_pct,
        run_montecarlo=run_montecarlo, use_ml=use_ml)
    if not decisions:
        return pd.DataFrame(columns=BOARD_COLS)

    ctx = ctx or global_context()
    outlook_expected = ctx["outlook"].expected_pct if ctx.get("outlook") else None

    ranked = sorted(decisions, key=lambda d: d.conviction or 0, reverse=True)
    enrich = {d.ticker for d in ranked[:top_us]}

    rows = []
    for d in ranked:
        g = stock_global(d.ticker) if d.ticker in enrich else \
            {"nasdaq_beta": None, "usdinr_beta": None, "brent_beta": None, "us_read": ""}
        r = {
            "ticker": d.ticker, "action": d.action, "conviction": d.conviction,
            "sentiment_net": d.sentiment_net, "today_x_avg": d.today_vs_avg_value_mult,
            "turn_today_pct": d.today_turnover_pct, "turn_avg_pct": d.avg_turnover_pct,
            "liq_tier": d.liq_tier, "liq_score": d.liq_score,
            "float_turnover_pct": d.float_turnover_pct, "amihud": d.amihud,
            "manip_tier": d.manip_tier, "factor_quintile": d.factor_quintile,
            "tech_tilt": d.tech_tilt, "trend_verdict": d.trend_verdict,
            "setups": ", ".join(d.setups) if d.setups else "",
            "expectancy_r": d.expectancy_r, "regime_label": d.regime_label,
            "outlook_expected": outlook_expected,
            "nasdaq_beta": g["nasdaq_beta"], "us_read": g["us_read"],
            "entry": d.entry, "stoploss": d.stoploss, "target": d.target, "rr": d.rr,
        }
        r["category"] = categorize(r)
        r["tags"] = tags(r)
        r["n_gates"] = gate_count(r)
        r["passes_strict"] = passes_strict(r)
        rows.append(r)
    return pd.DataFrame(rows, columns=BOARD_COLS)


def top_picks(board: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Strict 'satisfies all factors' names first; if fewer than ``n``, top up
    with the closest names (≥5 of 8 checks), flagged as 'fill'."""
    if board.empty:
        return board
    strict = board[board["passes_strict"]].sort_values("conviction", ascending=False).copy()
    strict["pick"] = "strict"
    if len(strict) >= n:
        return strict.head(n).reset_index(drop=True)
    need = n - len(strict)
    rest = board[~board["ticker"].isin(strict["ticker"]) & (board["n_gates"] >= 5)].copy()
    rest = rest.sort_values(["n_gates", "conviction"], ascending=[False, False])
    rest["pick"] = "fill"
    return pd.concat([strict, rest.head(need)], ignore_index=True)
