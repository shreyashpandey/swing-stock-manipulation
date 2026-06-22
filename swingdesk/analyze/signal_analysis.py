"""Fuse the four analytics layers (Rank · Range · Risk · Global) onto each live
signal, so the Signals page can show a single, decision-ready row per signal.

For every signal it answers:
  Rank   — where does this name sit in the cross-sectional factor ranking?
  Range  — given the signal's OWN target/stop, P(target before stop) + expectancy?
  Risk   — how many shares / what % of capital at the configured risk?
  Global — what moves it (beta to NIFTY and to US tech overnight)?

Heavy bits (Monte-Carlo barrier sim, per-stock regressions, the cross-sectional
factor table) are computed once per ticker and reused, so this scales to a full
signal list. It's still meant to run behind a button, not on every rerun.
"""
from __future__ import annotations

import pandas as pd

from swingdesk.analyze import expected_range as erange
from swingdesk.analyze import factors as factors_mod
from swingdesk.analyze import risk as risk_mod
from swingdesk.analyze import spillover
from swingdesk.config import ACCOUNT_CAPITAL, RISK_PER_TRADE_PCT


def analyze_signals(signals: list[dict],
                    capital: float = ACCOUNT_CAPITAL,
                    risk_pct: float = RISK_PER_TRADE_PCT,
                    mc_sims: int = 2000) -> pd.DataFrame:
    """One enriched row per signal across all four layers. Empty input → empty."""
    if not signals:
        return pd.DataFrame()

    tickers = sorted({s["ticker"] for s in signals})
    # Rank: one cross-sectional factor table over all signalled names.
    ftab = factors_mod.factor_table(tickers)
    fmap = {r["ticker"]: r for r in ftab.to_dict("records")} if not ftab.empty else {}
    sens_cache: dict[str, pd.DataFrame] = {}

    rows = []
    for s in signals:
        tk = s["ticker"]
        entry, sl, tgt = s.get("entry"), s.get("stoploss"), s.get("target")
        direction = s.get("direction", "long")

        # --- Risk: fixed-fractional position size
        ps = (risk_mod.position_size(entry, sl, capital=capital, risk_pct=risk_pct)
              if (entry and sl) else None)

        # --- Range: P(target before stop) + expectancy from the signal's levels
        p_tgt = exp_r = None
        if direction == "long" and entry and sl and tgt and entry > sl and tgt > entry:
            tpct = (tgt - entry) / entry * 100
            spct = (entry - sl) / entry * 100
            tvs = erange.target_vs_stop(tk, target_pct=tpct, stop_pct=spct, n_sims=mc_sims)
            if tvs:
                p_tgt, exp_r = tvs.p_target_first, tvs.expectancy_r
        er = erange.expected_range(tk, horizon_days=10)
        exp_move = er.expected_move_pct if er else None

        # --- Global: betas to NIFTY (market) and NASDAQ (US tech, overnight)
        if tk not in sens_cache:
            sens_cache[tk] = spillover.stock_sensitivities(tk)
        sens = sens_cache[tk]
        b_nifty = b_nasdaq = None
        if not sens.empty:
            for _, r in sens.iterrows():
                if str(r["driver"]).startswith("NIFTY"):
                    b_nifty = r["beta"]
                elif str(r["driver"]).startswith("NASDAQ"):
                    b_nasdaq = r["beta"]

        fr = fmap.get(tk, {})
        rows.append({
            "ticker": tk, "setup": s.get("setup"), "dir": direction,
            "entry": entry, "stoploss": sl, "target": tgt, "rr": s.get("rr"),
            # Rank
            "rank_q": fr.get("quintile"), "rank_score": fr.get("composite"),
            # Range
            "p_target_first": p_tgt, "exp_R": exp_r, "exp_move_10d_pct": exp_move,
            # Risk
            "shares": ps.shares if ps else None,
            "pct_cap": ps.pct_of_capital if ps else None,
            "risk_amt": ps.risk_amount if ps else None,
            # Global
            "beta_nifty": b_nifty, "beta_nasdaq": b_nasdaq,
        })

    df = pd.DataFrame(rows)
    # Best ideas first: highest reward-vs-risk expectancy, then factor rank.
    if "exp_R" in df.columns:
        df = df.sort_values(["exp_R", "rank_score"], ascending=[False, False],
                            na_position="last").reset_index(drop=True)
    return df
