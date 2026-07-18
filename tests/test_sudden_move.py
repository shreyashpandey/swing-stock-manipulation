from __future__ import annotations

import numpy as np
import pandas as pd

from swingdesk import storage
from swingdesk.analyze import sudden_move


def _coil_df(n: int = 150, *, breakout_at: int | None = None) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    for i in range(1, n):
        close[i] = close[i - 1] * (1 + 0.0008)
    # Last 40 bars compress below the high with quiet upward pressure.
    start = max(0, n - 40)
    close[start:] = np.linspace(close[start], close[start] * 1.035, n - start)
    if breakout_at is not None and breakout_at + 1 < n:
        close[breakout_at + 1:] *= 1.045

    open_ = close * 0.998
    high = close * 1.006
    low = close * 0.994
    volume = np.full(n, 500_000.0)
    volume[start:] = np.linspace(550_000, 1_000_000, n - start)
    # More volume on up days creates a positive signed-volume read.
    up = np.r_[False, np.diff(close) > 0]
    volume[up] *= 1.35
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_score_frame_recognizes_compressed_prebreakout_setup():
    df = _coil_df()
    row = sudden_move.score_frame(
        "COIL.NS",
        df,
        fund={"market_cap": 8e9, "float_shares": 4e7, "shares_outstanding": 8e7},
        sentiment_score=14,
        global_score=6,
        include_manipulation=False,
    )
    assert row is not None
    assert row.radar_score >= 55
    assert row.compression > 50
    assert row.prebreakout > 50
    assert row.catalyst > 50
    assert "near breakout zone" in row.reasons


def test_scan_returns_ranked_dataframe(tmp_db):
    storage.upsert_prices("COIL.NS", _coil_df())
    storage.upsert_prices("DULL.NS", _coil_df().assign(volume=100_000.0))
    storage.upsert_fundamentals([
        {"ticker": "COIL.NS", "market_cap": 8e9, "float_shares": 4e7,
         "shares_outstanding": 8e7, "quality_score": 70, "sector": "IT"},
        {"ticker": "DULL.NS", "market_cap": 4e11, "float_shares": 2e9,
         "shares_outstanding": 3e9, "quality_score": 60, "sector": "FMCG"},
    ])
    df = sudden_move.scan(["COIL.NS", "DULL.NS"], limit=5)
    assert list(df.columns) == sudden_move.RADAR_COLS
    assert len(df) == 2
    assert df.iloc[0]["radar_score"] >= df.iloc[1]["radar_score"]


def test_backtest_reports_hits_after_high_radar_scores(tmp_db):
    # Build a coil where the next bar after day 120 jumps >3%.
    df = _coil_df(150, breakout_at=120)
    storage.upsert_prices("BT.NS", df)
    storage.upsert_fundamentals([
        {"ticker": "BT.NS", "market_cap": 8e9, "float_shares": 4e7,
         "shares_outstanding": 8e7, "quality_score": 70, "sector": "IT"},
    ])
    trades, summary = sudden_move.backtest(
        ["BT.NS"], min_score=45, move_threshold_pct=3.0, horizon=1)
    assert summary["n"] == len(trades)
    assert summary["n"] > 0
    assert {"ticker", "date", "radar_score", "hit_high"}.issubset(trades.columns)
    assert trades["hit_high"].any()

