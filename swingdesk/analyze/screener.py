"""Combined screener: 'a few good stocks' that are attractive on factors AND
actually tradeable, each with the reasons attached.

This fuses the cross-sectional factor rank (momentum/quality/value/low-vol/trend)
with the liquidity profile (turnover-vs-mcap, volume-vs-float, ₹ ADV, impact).
The whole point: a name that looks great on the chart but is illiquid is flagged
'good but illiquid' — not surfaced as a buy. Liquidity is a veto, not a tiebreak.
"""
from __future__ import annotations

import pandas as pd

from swingdesk.analyze import factors as factors_mod
from swingdesk.analyze import liquidity as liq_mod
from swingdesk.storage import load_fundamentals

_FACTOR_LABELS = {
    "momentum_z": "momentum", "quality_z": "quality", "value_z": "value",
    "low_vol_z": "low-vol", "trend_z": "uptrend",
}
# Verdict sort priority (lower = shown first).
_VERDICT_RANK = {"✅ good & tradeable": 0, "⚠ good but illiquid": 1,
                 "⚠ good, liquidity unknown": 1, "➖ weak factors": 2}


def _factor_reasons(row: dict) -> list[str]:
    """Factors this stock is strong on (cross-sectional z > 0.5)."""
    out = []
    for col, label in _FACTOR_LABELS.items():
        z = row.get(col)
        if z is not None and pd.notna(z) and z > 0.5:
            out.append(label)
    return out


def screen(tickers: list[str], fund_df: pd.DataFrame | None = None,
           min_liquidity: float = 40.0) -> pd.DataFrame:
    """Rank `tickers` by factors, gated/annotated by liquidity. Returns one row
    per name with a verdict + combined reasons, best (good & tradeable) first."""
    if not tickers:
        return pd.DataFrame()
    ftab = factors_mod.factor_table(tickers)
    if ftab.empty:
        return pd.DataFrame()
    fmap = {r["ticker"]: r for r in ftab.to_dict("records")}

    fund_df = fund_df if fund_df is not None else load_fundamentals()
    fund = ({r["ticker"]: r for r in fund_df.to_dict("records")}
            if fund_df is not None and not fund_df.empty else {})

    rows = []
    for t, fr in fmap.items():
        composite = fr.get("composite")
        liq = liq_mod.liquidity_profile(t, fund.get(t))
        factor_ok = composite is not None and pd.notna(composite) and composite > 0
        liq_known = liq is not None
        liq_ok = liq_known and liq.score >= min_liquidity

        if factor_ok and liq_ok:
            verdict = "✅ good & tradeable"
        elif factor_ok and liq_known and not liq_ok:
            verdict = "⚠ good but illiquid"
        elif factor_ok and not liq_known:
            verdict = "⚠ good, liquidity unknown"
        else:
            verdict = "➖ weak factors"

        fac = _factor_reasons(fr)
        why = (("Strong " + ", ".join(fac)) if fac else "no standout factor")
        if liq is not None:
            why += " · " + liq.reasons[0]
            if verdict == "⚠ good but illiquid":
                why += " · ⚠ illiquid — size tiny or skip"

        rows.append({
            "ticker": t,
            "verdict": verdict,
            "factor_composite": round(float(composite), 3) if pd.notna(composite) else None,
            "quintile": fr.get("quintile"),
            "liq_tier": liq.tier if liq else "n/a",
            "liq_score": liq.score if liq else None,
            "adv_cr": liq.adv_value_cr if liq else None,
            "turnover_pct": liq.turnover_pct if liq else None,
            "float_turnover_pct": liq.float_turnover_pct if liq else None,
            "amihud": liq.amihud if liq else None,
            "why": why,
        })

    df = pd.DataFrame(rows)
    df["_v"] = df["verdict"].map(_VERDICT_RANK).fillna(3)
    df = df.sort_values(["_v", "factor_composite"], ascending=[True, False],
                        na_position="last").drop(columns="_v").reset_index(drop=True)
    return df
