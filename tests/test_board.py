"""The 🧭 Board: categorisation precedence, tag coupling, strict gate, top-up."""
from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk.analyze import board
from swingdesk.storage import upsert_fundamentals, upsert_prices


# ---- pure helpers (deterministic, no DB) ------------------------------------
def _row(**over):
    base = dict(manip_tier="Low", liq_tier="liquid", sentiment_net=0, tech_tilt="NEUTRAL",
                today_x_avg=1.0, nasdaq_beta=None, regime_label="neutral",
                outlook_expected=0.0, factor_quintile=3, trend_verdict="weak",
                expectancy_r=None)
    base.update(over)
    return base


def test_category_precedence_risk_first():
    assert board.categorize(_row(manip_tier="High")) == "🔴 Manipulation"
    assert board.categorize(_row(liq_tier="illiquid")) == "🟠 Illiquid"
    # manipulation outranks illiquid
    assert board.categorize(_row(manip_tier="High", liq_tier="untradeable")) == "🔴 Manipulation"


def test_category_positive_buckets():
    assert board.categorize(_row(sentiment_net=3, tech_tilt="BUY")) == "🟢 News-backed"
    assert board.categorize(_row(today_x_avg=5.0)) == "🔵 Turnover surge"
    assert board.categorize(_row(nasdaq_beta=0.8, regime_label="risk-on")) == "🟣 US tailwind"
    assert board.categorize(_row(factor_quintile=1)) == "🟦 Factor leader"
    assert board.categorize(_row()) == "⚪ Neutral"


def test_us_tailwind_requires_not_risk_off():
    assert board._us_tailwind(_row(nasdaq_beta=0.9, regime_label="neutral")) is True
    assert board._us_tailwind(_row(nasdaq_beta=0.9, regime_label="risk-off")) is False
    assert board._us_tailwind(_row(nasdaq_beta=0.1)) is False


def test_tags_couple_multiple_signals():
    t = board.tags(_row(sentiment_net=3, trend_verdict="real", liq_tier="liquid",
                        factor_quintile=1, tech_tilt="STRONG BUY"))
    for piece in ("News+", "Uptrend", "Liquid", "Q1", "Bull tilt"):
        assert piece in t
    # risk tags surface even alongside positives
    assert "⚠ Manip" in board.tags(_row(manip_tier="High", sentiment_net=3))


def test_strict_gate_all_must_hold():
    good = _row(factor_quintile=1, liq_tier="liquid", manip_tier="Low",
                tech_tilt="BUY", trend_verdict="real", sentiment_net=1,
                regime_label="neutral", expectancy_r=0.5)
    assert board.passes_strict(good) is True and board.gate_count(good) == board.N_GATES
    # break one check → fails strict but most gates still pass
    bad = dict(good); bad["manip_tier"] = "Elevated"
    assert board.passes_strict(bad) is False
    assert board.gate_count(bad) == board.N_GATES - 1


def test_top_picks_strict_first_then_fill():
    rows = []
    # 3 strict passers, 4 near-misses (7 gates), 1 weak (3 gates)
    for i in range(3):
        rows.append(_row(ticker=f"S{i}.NS", factor_quintile=1, liq_tier="liquid",
                         manip_tier="Low", tech_tilt="BUY", trend_verdict="real",
                         sentiment_net=1, expectancy_r=0.5, conviction=90 - i))
    for i in range(4):
        rows.append(_row(ticker=f"F{i}.NS", factor_quintile=1, liq_tier="liquid",
                         manip_tier="Elevated", tech_tilt="BUY", trend_verdict="real",
                         sentiment_net=1, expectancy_r=0.5, conviction=70 - i))
    rows.append(_row(ticker="W.NS", conviction=10))
    df = pd.DataFrame([{**r, "category": board.categorize(r), "tags": board.tags(r),
                        "n_gates": board.gate_count(r), "passes_strict": board.passes_strict(r)}
                       for r in rows])
    picks = board.top_picks(df, n=5)
    assert len(picks) == 5
    assert list(picks[picks["pick"] == "strict"]["ticker"]) == ["S0.NS", "S1.NS", "S2.NS"]
    assert (picks["pick"] == "fill").sum() == 2          # topped up from the near-misses
    assert "W.NS" not in set(picks["ticker"])            # too few gates to qualify


# ---- integration: build_board on seeded data --------------------------------
def _seed(ticker, price, vol, drift, n=320, quality=70, seed=1):
    rng = np.random.default_rng(seed)
    close = price * np.cumprod(1 + rng.normal(drift, 0.012, n))
    v = np.full(n, float(vol)) * rng.uniform(0.9, 1.1, n)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    upsert_prices(ticker, pd.DataFrame({"open": close * 0.999, "high": close * 1.01,
                                        "low": close * 0.99, "close": close, "volume": v}, index=idx))
    upsert_fundamentals([{"ticker": ticker, "market_cap": 2.0e11, "float_shares": 2.0e8,
                          "shares_outstanding": 4.0e8, "quality_score": quality, "sector": "IT",
                          "return_on_equity": 0.18, "trailing_pe": 22, "earnings_growth": 0.15,
                          "revenue_growth": 0.12, "debt_to_equity": 0.3}])


def test_build_board_end_to_end(tmp_db):
    _seed("GOOD.NS", 500, 2_000_000, 0.004, quality=80, seed=1)
    _seed("MEH.NS", 300, 1_500_000, 0.0, quality=60, seed=2)
    _seed("ILQ.NS", 500, 2_000, -0.002, seed=3)     # illiquid + downtrend
    df = board.build_board(["GOOD.NS", "MEH.NS", "ILQ.NS"], run_montecarlo=False)
    assert list(df.columns) == board.BOARD_COLS
    assert len(df) == 3
    cats = dict(zip(df["ticker"], df["category"]))
    assert cats["ILQ.NS"] == "🟠 Illiquid"               # illiquid flagged
    # every row has a category and a tags string
    assert df["category"].notna().all() and df["tags"].notna().all()
    # top_picks never errors and is capped
    assert len(board.top_picks(df, n=20)) <= 3
