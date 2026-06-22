"""Detect unusual / potentially manipulated trading activity.

These are *operator-footprint* heuristics — the signatures a stock leaves when
someone is pushing it around rather than the market fairly discovering price:
a disproportionate amount of money churning relative to the company's size, a
day's volume eating a big slice of the tradable float, returns that are far
outside the stock's own normal range, and thin liquidity that lets a small
order move the tape a long way.

Everything here is computed from data we already have — daily OHLCV (the
`prices` table) plus `market_cap` / `shares_outstanding` / `float_shares` (the
`fundamentals` table, sourced from yfinance). No NSE scraping yet.

Each detector returns a small dict:
    metric values (raw, human-readable) + a 0-100 ``score`` (higher = more
    unusual) + a ``notes`` list of plain-English observations.
``scorecard()`` blends the four into one 0-100 ``risk_score`` per ticker.

The *strongest* manipulation tells — security-wise **delivery %**, **bulk /
block deals**, **promoter pledge** changes — live on NSE and are not in
yfinance. ``NSE_HOOKS`` documents where those plug in; ``scorecard()`` already
tolerates their absence and records the gap in ``data_gaps``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- Tunable thresholds (one place, so the glossary can quote them) -----------
WINDOW = 20          # baseline lookback (~1 trading month)
RUN_WINDOW = 10      # window for spotting a clustered run-up
MIN_BARS = WINDOW + 5  # need this many bars before a read is meaningful

# Saturation points — the value at which a sub-metric maxes its component to 100.
TURNOVER_MCAP_FULL = 0.03    # 3% of market cap traded in one day = extreme churn
TURNOVER_SPIKE_FULL = 5.0    # today's value 5x the 20-day median = extreme spike
VOLUME_MULT_FULL = 5.0       # 5x average volume = extreme
FLOAT_TURN_FULL = 0.10       # 10% of free float traded in a day = extreme
RETURN_Z_FULL = 4.0          # a 4-sigma daily move = extreme
GAP_FULL = 0.08              # an 8% overnight gap = extreme
BIG_MOVE = 0.095             # ~near the 10% circuit band — counts as a "large up-move"
ILLIQUID_VALUE_CR = 50.0     # median daily turnover below ₹50 cr = thinly traded

# How the detectors combine into the headline risk score. NSE-sourced detectors
# (delivery, deals) are included; the scorecard renormalises over whatever is
# actually available, so missing NSE data just reweights the price-based four.
WEIGHTS = {
    "turnover_mcap": 0.22,
    "volume_float": 0.22,
    "abnormal_return": 0.18,
    "amihud": 0.10,
    "delivery": 0.20,
    "deals": 0.08,
}

TIER_ELEVATED = 30
TIER_HIGH = 60


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _baseline(series: pd.Series, window: int = WINDOW) -> pd.Series:
    """The `window` bars *before* today — excludes the current bar so a spike
    never contaminates the norm it's being measured against."""
    return series.iloc[:-1].tail(window).dropna()


def _enough(df: pd.DataFrame) -> bool:
    return df is not None and not df.empty and len(df) >= MIN_BARS


# --- Detector 1: traded value vs market cap -----------------------------------
def turnover_vs_marketcap(df: pd.DataFrame, market_cap: float | None) -> dict | None:
    """'Order value vs market cap' — the rupee value changing hands today
    (close × volume) measured against the whole company's market cap, and how
    far that sits above the stock's own 20-day norm.

    A large cap normally turns over a fraction of a percent of its value a day.
    When a day's traded value balloons to several percent of market cap, or
    many times the recent median, money is moving in a way the float doesn't
    usually justify."""
    if not _enough(df) or not market_cap:
        return None
    # Today's traded value, its ratio to market cap, and the spike vs the prior
    # 20-day baseline all come from the shared market_metrics layer — the same
    # code the liquidity/screener tab uses, so the two tabs can't drift apart.
    from swingdesk.analyze.market_metrics import _today_turnover
    today_value, turnover_ratio, median_value, spike_mult = _today_turnover(
        df, market_cap, base_window=WINDOW)

    comp_abs = _clamp(turnover_ratio / TURNOVER_MCAP_FULL * 100)
    comp_spike = _clamp((spike_mult - 1) / (TURNOVER_SPIKE_FULL - 1) * 100) if spike_mult == spike_mult else 0.0
    score = 0.4 * comp_abs + 0.6 * comp_spike

    notes = []
    notes.append(
        f"₹{today_value/1e7:,.1f} cr traded today = {turnover_ratio*100:.2f}% of the "
        f"₹{market_cap/1e7:,.0f} cr market cap."
    )
    if spike_mult == spike_mult:
        notes.append(f"That's {spike_mult:.1f}× the 20-day median traded value.")
    if turnover_ratio >= TURNOVER_MCAP_FULL:
        notes.append("⚠ Daily churn above 3% of market cap — heavy, unusual rotation.")
    if spike_mult == spike_mult and spike_mult >= TURNOVER_SPIKE_FULL:
        notes.append("⚠ Traded value spiked 5×+ above its own norm.")

    return {
        "today_value_cr": round(today_value / 1e7, 2),
        "turnover_ratio_pct": round(turnover_ratio * 100, 3),
        "median_value_cr": round(median_value / 1e7, 2) if median_value == median_value else None,
        "spike_mult": round(spike_mult, 2) if spike_mult == spike_mult else None,
        "score": round(score, 1),
        "notes": notes,
    }


# --- Detector 2: volume vs float ----------------------------------------------
def volume_float_spike(
    df: pd.DataFrame,
    float_shares: float | None = None,
    shares_outstanding: float | None = None,
) -> dict | None:
    """How big today's volume is versus (a) the stock's own average volume and
    (b) its tradable float. An operator accumulating or distributing has to
    trade size; when one day chews through a large fraction of the free float,
    or many times the average, that's their footprint.

    Float denominator preference: free float → shares outstanding → (if neither
    is known) derived shares = market_cap/price is left to the caller. The note
    records which basis was used so the read isn't silently optimistic."""
    if not _enough(df):
        return None
    vol = df["volume"]
    today_vol = float(vol.iloc[-1])
    base = _baseline(vol)
    mean20 = float(base.mean()) if len(base) else float("nan")
    std20 = float(base.std(ddof=0)) if len(base) else float("nan")

    volume_mult = today_vol / mean20 if mean20 and mean20 > 0 else float("nan")
    volume_z = (today_vol - mean20) / std20 if std20 and std20 > 0 else float("nan")

    basis = float_shares or shares_outstanding
    basis_label = "free float" if float_shares else ("shares outstanding" if shares_outstanding else None)
    float_turn = today_vol / basis if basis and basis > 0 else float("nan")

    comp_mult = _clamp((volume_mult - 1) / (VOLUME_MULT_FULL - 1) * 100) if volume_mult == volume_mult else 0.0
    comp_float = _clamp(float_turn / FLOAT_TURN_FULL * 100) if float_turn == float_turn else 0.0
    # If we have no float basis, lean entirely on the volume multiple.
    score = 0.6 * comp_mult + 0.4 * comp_float if basis else comp_mult

    notes = []
    if volume_mult == volume_mult:
        notes.append(f"Today's volume is {volume_mult:.1f}× the 20-day average ({volume_z:+.1f}σ).")
    if float_turn == float_turn:
        notes.append(f"{float_turn*100:.1f}% of {basis_label} changed hands today.")
        if float_turn >= FLOAT_TURN_FULL:
            notes.append("⚠ A day's volume ate 10%+ of the tradable float.")
    else:
        notes.append("Float basis unknown — read rests on the volume multiple alone.")
    if volume_mult == volume_mult and volume_mult >= VOLUME_MULT_FULL:
        notes.append("⚠ Volume 5×+ its average — abnormal participation.")

    return {
        "volume_mult": round(volume_mult, 2) if volume_mult == volume_mult else None,
        "volume_z": round(volume_z, 2) if volume_z == volume_z else None,
        "float_turnover_pct": round(float_turn * 100, 2) if float_turn == float_turn else None,
        "float_basis": basis_label,
        "score": round(score, 1),
        "notes": notes,
    }


# --- Detector 3: abnormal return / circuit run-up -----------------------------
def abnormal_return(df: pd.DataFrame) -> dict | None:
    """How extreme today's move is versus the stock's own daily-return
    distribution, plus the overnight gap and any *cluster* of big up-days — the
    run-up phase of a pump rather than one isolated jump."""
    if not _enough(df):
        return None
    close = df["close"]
    ret = close.pct_change()
    today_ret = float(ret.iloc[-1]) if pd.notna(ret.iloc[-1]) else 0.0

    base = _baseline(ret)
    mean20 = float(base.mean()) if len(base) else 0.0
    std20 = float(base.std(ddof=0)) if len(base) else float("nan")
    ret_z = (today_ret - mean20) / std20 if std20 and std20 > 0 else float("nan")

    # Overnight gap (today's open vs yesterday's close).
    gap = float("nan")
    if "open" in df.columns and len(close) >= 2:
        prev_close = float(close.iloc[-2])
        if prev_close > 0:
            gap = (float(df["open"].iloc[-1]) - prev_close) / prev_close

    # Clustered run-up: large up-days within the recent window + the run's return.
    recent = ret.tail(RUN_WINDOW)
    big_up_days = int((recent >= BIG_MOVE).sum())
    run_return = float(close.iloc[-1] / close.iloc[-1 - RUN_WINDOW] - 1) if len(close) > RUN_WINDOW else float("nan")

    comp_z = _clamp(abs(ret_z) / RETURN_Z_FULL * 100) if ret_z == ret_z else 0.0
    comp_gap = _clamp(abs(gap) / GAP_FULL * 100) if gap == gap else 0.0
    comp_run = _clamp(big_up_days / 3 * 100)  # 3 near-circuit up-days in 10 → maxed
    score = 0.45 * comp_z + 0.25 * comp_gap + 0.30 * comp_run

    notes = []
    if ret_z == ret_z:
        notes.append(f"Today moved {today_ret*100:+.1f}% — a {ret_z:+.1f}σ day vs its 20-day norm.")
    if gap == gap and abs(gap) >= 0.02:
        notes.append(f"Opened on a {gap*100:+.1f}% gap.")
    if big_up_days >= 2:
        notes.append(
            f"⚠ {big_up_days} near-circuit (+{BIG_MOVE*100:.0f}%+) up-days in the last "
            f"{RUN_WINDOW} sessions — a clustered run-up ({run_return*100:+.0f}% over the stretch)."
        )
    if ret_z == ret_z and abs(ret_z) >= RETURN_Z_FULL:
        notes.append("⚠ A 4σ+ daily move — far outside normal behaviour.")

    return {
        "today_return_pct": round(today_ret * 100, 2),
        "return_z": round(ret_z, 2) if ret_z == ret_z else None,
        "gap_pct": round(gap * 100, 2) if gap == gap else None,
        "big_up_days_10": big_up_days,
        "run_return_pct": round(run_return * 100, 1) if run_return == run_return else None,
        "score": round(score, 1),
        "notes": notes,
    }


# --- Detector 4: Amihud illiquidity -------------------------------------------
def amihud_illiquidity(df: pd.DataFrame) -> dict | None:
    """Amihud (2002) illiquidity = average of |daily return| ÷ rupee value
    traded. It measures how far price gets pushed per rupee of trading: a thin,
    illiquid stock returns a large number, meaning a modest order moves the tape
    a long way — exactly the kind of stock that's cheap to manipulate.

    We pair the raw Amihud figure with the plainer median daily turnover (₹ cr):
    when that median is low the stock is structurally easy to push, so any
    volume spike the other detectors flag carries more weight."""
    if not _enough(df):
        return None
    ret = df["close"].pct_change().abs()
    traded_value = df["close"] * df["volume"]
    valid = traded_value > 0
    illiq_daily = (ret[valid] / traded_value[valid]).tail(WINDOW)
    # Scale to "price-impact per ₹1 cr traded" so the number is readable.
    amihud = float(illiq_daily.mean() * 1e7) if len(illiq_daily) else float("nan")

    median_value_cr = float((traded_value.tail(WINDOW).median()) / 1e7)

    # Illiquidity score: high when the median daily turnover is low.
    comp = _clamp((1 - median_value_cr / ILLIQUID_VALUE_CR) * 100)
    score = comp

    notes = []
    notes.append(f"Median daily turnover ≈ ₹{median_value_cr:,.1f} cr over the last {WINDOW} days.")
    if amihud == amihud:
        notes.append(f"Amihud illiquidity ≈ {amihud:.4f} (price-impact per ₹1 cr traded).")
    if median_value_cr < ILLIQUID_VALUE_CR:
        notes.append(
            f"⚠ Thinly traded (< ₹{ILLIQUID_VALUE_CR:.0f} cr/day) — a small order can move it a lot, "
            "so treat any volume/return spike here as higher-risk."
        )

    return {
        "amihud": round(amihud, 5) if amihud == amihud else None,
        "median_value_cr": round(median_value_cr, 2),
        "score": round(score, 1),
        "notes": notes,
    }


# --- NSE detectors (data via swingdesk.ingest.nse) ----------------------------
DELIV_HEALTHY = 45.0     # delivery % an anchor: a clean stock often delivers ~40-60%
DELIV_FLOOR = 10.0       # at/below this, almost pure intraday churn -> maxed
DEALS_WINDOW = 30        # days of bulk/block deal history to weigh

NSE_HOOKS = {
    "delivery_spike": "Security-wise delivery %: a price run-up on FALLING delivery % "
                      "(mostly intraday churn, few shares actually delivered) is a classic "
                      "pump tell. Source: NSE security-wise delivery bhavcopy.",
    "bulk_block_deals": "Bulk & block deals: large single-party trades disclosed by NSE. "
                        "Repeated same-entity activity around a run-up flags an operator.",
    "promoter_pledge": "Promoter pledge / holding changes: rising pledge or shrinking "
                       "promoter holding alongside a price spike. Source: NSE/BSE filings. "
                       "Not wired yet.",
}


def delivery_spike(ticker: str, df: pd.DataFrame | None = None,
                   delivery_df: pd.DataFrame | None = None) -> dict | None:
    """Flag a price run-up riding on *falling* delivery % — the move is intraday
    churn rather than shares genuinely being accumulated.

    `delivery_df` (index=date, columns incl. ``deliv_pct``) can be injected;
    otherwise it's read from storage. `df` is the price frame, used only to tell
    whether price is actually rising over the recent window."""
    if delivery_df is None:
        try:
            from swingdesk.storage import load_delivery
            delivery_df = load_delivery(ticker, days=WINDOW + 5)
        except Exception:
            return None
    if delivery_df is None or delivery_df.empty or "deliv_pct" not in delivery_df.columns:
        return None
    pct = delivery_df["deliv_pct"].dropna()
    if len(pct) < 5:
        return None

    latest = float(pct.iloc[-1])
    avg_prior = float(pct.iloc[:-1].tail(WINDOW).mean())

    # Is price rising over the run window? (drives the divergence component)
    run_return = float("nan")
    if df is not None and not df.empty and len(df) > RUN_WINDOW:
        run_return = float(df["close"].iloc[-1] / df["close"].iloc[-1 - RUN_WINDOW] - 1)

    comp_low = _clamp((DELIV_HEALTHY - latest) / (DELIV_HEALTHY - DELIV_FLOOR) * 100)
    comp_div = 0.0
    rising = run_return == run_return and run_return > 0.05
    if rising and avg_prior > 0 and latest < avg_prior:
        comp_div = _clamp((avg_prior - latest) / avg_prior * 200)  # 50% rel. drop -> 100
    score = 0.45 * comp_low + 0.55 * comp_div

    notes = [f"Delivery {latest:.0f}% today vs {avg_prior:.0f}% 20-day average."]
    if rising:
        notes.append(f"Price is up {run_return*100:+.0f}% over {RUN_WINDOW} sessions.")
        if comp_div > 40:
            notes.append("⚠ Run-up on falling delivery — looks like intraday churn, not real accumulation.")
    if latest <= DELIV_FLOOR + 10:
        notes.append(f"⚠ Very low delivery ({latest:.0f}%) — most volume never left the day.")

    return {
        "delivery_pct": round(latest, 1),
        "delivery_avg": round(avg_prior, 1),
        "run_return_pct": round(run_return * 100, 1) if run_return == run_return else None,
        "score": round(score, 1),
        "notes": notes,
    }


def bulk_block_deals(ticker: str, deals_df: pd.DataFrame | None = None) -> dict | None:
    """Weigh recent bulk/block deals: a *repeated* same-party presence around a
    stock is the operator tell (a one-off institutional block usually isn't).

    `deals_df` (columns incl. ``client``, ``side``, ``deal_type``, ``qty``) can
    be injected; otherwise read from storage for the last 30 days."""
    if deals_df is None:
        try:
            from swingdesk.storage import load_deals
            deals_df = load_deals(ticker, days=DEALS_WINDOW)
        except Exception:
            return None
    if deals_df is None or deals_df.empty:
        return None

    n = len(deals_df)
    clients = deals_df["client"].fillna("?")
    repeat = int(clients.value_counts().iloc[0]) if n else 0
    top_client = clients.value_counts().index[0] if n else None
    n_block = int((deals_df["deal_type"] == "block").sum())

    comp_count = _clamp(n / 5 * 100)
    comp_rep = _clamp((repeat - 1) / 2 * 100)   # same party 3+ times -> maxed
    score = 0.4 * comp_count + 0.6 * comp_rep

    notes = [f"{n} bulk/block deal(s) in the last {DEALS_WINDOW} days ({n_block} block)."]
    if repeat >= 2:
        notes.append(f"⚠ '{top_client}' appears in {repeat} of them — repeated same-party activity.")

    return {
        "n_deals": n,
        "n_block": n_block,
        "max_same_party": repeat,
        "score": round(score, 1),
        "notes": notes,
    }


# --- The scorecard: blend everything into one read ----------------------------
def _tier(score: float) -> str:
    if score >= TIER_HIGH:
        return "High"
    if score >= TIER_ELEVATED:
        return "Elevated"
    return "Low"


def scorecard(ticker: str, df: pd.DataFrame, fundamentals: dict | None = None) -> dict:
    """Run all detectors and blend them into one 0-100 ``risk_score`` for a
    ticker. Higher = more unusual / manipulation-prone activity *right now*.

    Returns a dict with ``risk_score``, ``tier`` (Low/Elevated/High),
    per-detector ``components``, a flat ``notes`` list (the ⚠ items bubble to
    the top), and ``data_gaps`` for anything that couldn't be computed (missing
    fundamentals, NSE sources not wired)."""
    fundamentals = fundamentals or {}
    market_cap = fundamentals.get("market_cap")
    float_shares = fundamentals.get("float_shares")
    shares_outstanding = fundamentals.get("shares_outstanding")

    # Derive total shares from market_cap/price when neither share count is known.
    if not float_shares and not shares_outstanding and market_cap and _enough(df):
        last_price = float(df["close"].iloc[-1])
        if last_price > 0:
            shares_outstanding = market_cap / last_price

    data_gaps: list[str] = []
    if not _enough(df):
        return {
            "ticker": ticker, "risk_score": None, "tier": "n/a",
            "components": {}, "notes": [], "verdict": "Not enough price history to assess.",
            "data_gaps": [f"need ≥ {MIN_BARS} daily bars, have {0 if df is None else len(df)}"],
        }

    components = {
        "turnover_mcap": turnover_vs_marketcap(df, market_cap),
        "volume_float": volume_float_spike(df, float_shares, shares_outstanding),
        "abnormal_return": abnormal_return(df),
        "amihud": amihud_illiquidity(df),
        "delivery": delivery_spike(ticker, df),
        "deals": bulk_block_deals(ticker),
    }
    if components["turnover_mcap"] is None:
        data_gaps.append("market_cap missing — turnover-vs-market-cap skipped (refresh fundamentals)")
    if not float_shares and not shares_outstanding and not market_cap:
        data_gaps.append("no share count / market cap — float-turnover skipped")
    if components["delivery"] is None:
        data_gaps.append("no NSE delivery data — run the NSE refresh (delivery-% check skipped)")
    if components["deals"] is None:
        data_gaps.append("no NSE bulk/block deals on record for this stock")
    data_gaps.append("promoter pledge / holding changes — not wired (see NSE_HOOKS)")

    # Weighted blend over whatever components are available; renormalise weights.
    total_w, acc = 0.0, 0.0
    for key, comp in components.items():
        if comp is not None:
            w = WEIGHTS[key]
            acc += w * comp["score"]
            total_w += w
    risk_score = round(acc / total_w, 1) if total_w else None

    # Notes: warnings (⚠) first, then the rest, tagged by source detector.
    warn, info = [], []
    for comp in components.values():
        if not comp:
            continue
        for n in comp["notes"]:
            (warn if n.startswith("⚠") else info).append(n)
    notes = warn + info

    tier = _tier(risk_score) if risk_score is not None else "n/a"
    verdict = {
        "High": "High unusual-activity footprint — investigate before trading; size down or avoid.",
        "Elevated": "Some unusual activity — worth a closer look at the flagged metrics.",
        "Low": "Activity looks normal for this stock.",
        "n/a": "Insufficient data.",
    }[tier]

    return {
        "ticker": ticker,
        "risk_score": risk_score,
        "tier": tier,
        "components": components,
        "notes": notes,
        "verdict": verdict,
        "data_gaps": data_gaps,
    }
