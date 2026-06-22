from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Augment OHLCV with technical + volume-flow indicators."""
    if df.empty:
        return df
    out = df.copy()
    # Price-based
    out["ema20"] = ta.ema(out["close"], length=20)
    out["ema50"] = ta.ema(out["close"], length=50)
    out["ema200"] = ta.ema(out["close"], length=200)
    out["sma20"] = ta.sma(out["close"], length=20)
    out["rsi14"] = ta.rsi(out["close"], length=14)
    macd = ta.macd(out["close"])
    if macd is not None and not macd.empty:
        out["macd"] = macd.iloc[:, 0]
        out["macd_signal"] = macd.iloc[:, 1]
        out["macd_hist"] = macd.iloc[:, 2]
    out["atr14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    out["vol_avg20"] = out["volume"].rolling(20).mean()
    out["high20"] = out["high"].rolling(20).max()
    out["low20"] = out["low"].rolling(20).min()
    out["high55"] = out["high"].rolling(55).max()

    # Bollinger Bands (20, 2σ) — volatility envelope. bb_pct: where close sits
    # in the band (0=lower, 1=upper); bb_width: band width / mid (squeeze gauge).
    bb = ta.bbands(out["close"], length=20, std=2)
    if bb is not None and not bb.empty:
        out["bb_lower"] = bb.iloc[:, 0]
        out["bb_mid"] = bb.iloc[:, 1]
        out["bb_upper"] = bb.iloc[:, 2]
        out["bb_width"] = bb.iloc[:, 3]      # BBB — bandwidth %
        out["bb_pct"] = bb.iloc[:, 4]        # BBP — %B
        # Rolling floor of bandwidth → detect a "squeeze" (low-vol coil).
        out["bb_width_min60"] = out["bb_width"].rolling(60).min()

    # ADX (14) — trend STRENGTH, plus directional indicators. ADX > 25 = a real
    # trend; DI+ > DI- = bulls in control. This is the "is it even trending?" gate.
    adx = ta.adx(out["high"], out["low"], out["close"], length=14)
    if adx is not None and not adx.empty:
        out["adx14"] = adx.iloc[:, 0]
        out["di_plus"] = adx.iloc[:, 1]
        out["di_minus"] = adx.iloc[:, 2]

    # Stochastic (14,3,3) — momentum oscillator for TIMING (overbought/oversold +
    # %K/%D crosses). Pairs well with a trend filter for pullback entries.
    stoch = ta.stoch(out["high"], out["low"], out["close"])
    if stoch is not None and not stoch.empty:
        out["stoch_k"] = stoch.iloc[:, 0]
        out["stoch_d"] = stoch.iloc[:, 1]

    # Supertrend (10, 3) — an explicit BUY/SELL regime line. Direction flips
    # +1 (long/green) ↔ -1 (short/red); the line itself is a trailing stop.
    st = ta.supertrend(out["high"], out["low"], out["close"], length=10, multiplier=3.0)
    if st is not None and not st.empty:
        out["supertrend"] = st.iloc[:, 0]
        out["supertrend_dir"] = st.iloc[:, 1]   # +1 bullish, -1 bearish

    # CCI (20) — deviation from the typical-price mean; >+100 strong up-thrust,
    # <-100 washed-out. Used as a confirmation, not a standalone trigger.
    out["cci20"] = ta.cci(out["high"], out["low"], out["close"], length=20)

    # Volume-flow indicators — these reveal whether smart money is
    # accumulating (buying on dips) or distributing (selling into strength).
    out["obv"] = ta.obv(out["close"], out["volume"])              # cumulative
    out["mfi14"] = ta.mfi(out["high"], out["low"], out["close"],  # volume-weighted RSI
                          out["volume"], length=14)
    ad = ta.ad(out["high"], out["low"], out["close"], out["volume"])
    out["ad_line"] = ad if ad is not None else None               # Accum/Dist

    # OBV slope: positive = accumulation, negative = distribution
    out["obv_slope_10"] = out["obv"] - out["obv"].shift(10)

    # Up-volume / down-volume ratio over last 20 days — "buying pressure"
    direction = (out["close"] > out["close"].shift(1)).astype(int).replace(0, -1)
    signed_vol = out["volume"] * direction
    out["buy_pressure_20"] = (
        signed_vol.rolling(20).apply(lambda s: s[s > 0].sum() / max(1, abs(s.sum())),
                                     raw=False)
    )
    return out


def _slope_dir(series: pd.Series, window: int) -> int:
    """Direction of a series over `window` bars: +1 up, -1 down, 0 flat/unknown."""
    if series is None or len(series) < window + 1:
        return 0
    a, b = series.iloc[-1], series.iloc[-1 - window]
    if pd.isna(a) or pd.isna(b) or a == b:
        return 0
    return 1 if a > b else -1


def trend_quality(df: pd.DataFrame, window: int = 20) -> dict | None:
    """Classify a price uptrend as volume-confirmed ('real') or diverging ('false').

    A genuine uptrend has price AND volume-flow rising together. A false one has
    price grinding up while smart money distributes — OBV/AD flat-to-down, weak
    up-day volume, bearish MFI divergence. Returns a dict with:
      verdict  : "real" | "weak" | "false" | "no_uptrend"
      label    : human-readable verdict
      score    : 0-100 (share of the 5 volume checks that confirm the trend)
      reasons  : list of ✓/✗ explanation strings
    or None if there isn't enough indicator-augmented data.
    """
    need = ("close", "ema20", "ema50", "obv", "mfi14", "ad_line", "volume")
    if df is None or df.empty or len(df) < window + 6 or any(c not in df.columns for c in need):
        return None
    last = df.iloc[-1]

    # Is it even an uptrend? (rising close + 20EMA above 50EMA + price above 50EMA)
    in_uptrend = (
        _slope_dir(df["close"], window) > 0
        and not pd.isna(last["ema20"]) and not pd.isna(last["ema50"])
        and last["ema20"] > last["ema50"] and last["close"] > last["ema50"]
    )
    if not in_uptrend:
        return {
            "verdict": "no_uptrend", "label": "No clear uptrend", "score": 0,
            "in_uptrend": False,
            "reasons": ["Price isn't in a defined uptrend "
                        "(needs a rising close with 20EMA > 50EMA > price-support)."],
        }

    reasons: list[str] = []
    confirms = 0

    # 1) OBV — cumulative volume flow tracking price?
    if _slope_dir(df["obv"], window) > 0:
        confirms += 1
        reasons.append("✓ OBV rising with price — buyers absorbing supply (accumulation).")
    else:
        reasons.append("✗ OBV flat/falling while price rises — bearish divergence (distribution).")

    # 2) Accumulation/Distribution line — closing near highs on volume?
    if _slope_dir(df["ad_line"], window) > 0:
        confirms += 1
        reasons.append("✓ A/D line rising — closes landing near the high of range on volume.")
    else:
        reasons.append("✗ A/D line not rising — closes drifting toward the low of range.")

    # 3) MFI — healthy and not bearishly diverging from price
    mfi_now, mfi_then = last["mfi14"], df["mfi14"].iloc[-1 - window]
    if not pd.isna(mfi_now):
        diverging = not pd.isna(mfi_then) and mfi_now < mfi_then - 5
        if mfi_now >= 50 and not diverging:
            confirms += 1
            reasons.append(f"✓ MFI {mfi_now:.0f} ≥ 50 and not diverging — money flow supports price.")
        else:
            reasons.append(f"✗ MFI {mfi_now:.0f} weak or fading vs {window}d ago — money flow lags price.")

    # 4) Up-day vs down-day volume share over the window
    win = df.tail(window)
    chg = win["close"].diff()
    up_vol = float(win.loc[chg > 0, "volume"].sum())
    dn_vol = float(win.loc[chg < 0, "volume"].sum())
    up_share = up_vol / (up_vol + dn_vol) if (up_vol + dn_vol) > 0 else 0.5
    if up_share >= 0.55:
        confirms += 1
        reasons.append(f"✓ {up_share:.0%} of volume traded on up-days — buyers in control.")
    else:
        reasons.append(f"✗ Only {up_share:.0%} of volume on up-days — the advance lacks volume.")

    # 5) Volume expanding into the advance (not running on fumes)?
    recent = df["volume"].tail(5).mean()
    prior = df["volume"].iloc[-window:-5].mean()
    if not pd.isna(recent) and not pd.isna(prior) and prior > 0 and recent >= prior:
        confirms += 1
        reasons.append("✓ Volume expanding into the advance — healthy participation.")
    else:
        reasons.append("✗ Volume contracting into the advance — rally running on fumes.")

    score = round(confirms / 5 * 100)
    if confirms >= 4:
        verdict, label = "real", "Real uptrend — volume-confirmed"
    elif confirms >= 2:
        verdict, label = "weak", "Weak / unconfirmed uptrend — be selective"
    else:
        verdict, label = "false", "False uptrend — distribution risk"
    return {"verdict": verdict, "label": label, "score": score,
            "in_uptrend": True, "up_volume_share": round(up_share, 2),
            "reasons": reasons}


def money_flow_read(df: pd.DataFrame, window: int = 20) -> dict:
    """Plain-English read-out of OBV and MFI for the latest bar.

    Returns keys: obv, mfi (one-line verdicts) plus obv_detail, mfi_detail
    (a second sentence spelling out *why* / what to watch).
    """
    out: dict[str, str | None] = {"obv": None, "mfi": None,
                                  "obv_detail": None, "mfi_detail": None}
    if df is None or df.empty:
        return out
    last = df.iloc[-1]
    if "obv" in df.columns and len(df) > window:
        obv_d, price_d = _slope_dir(df["obv"], window), _slope_dir(df["close"], window)
        if obv_d > 0 and price_d > 0:
            out["obv"] = "🟢 OBV rising with price — accumulation; the trend has volume behind it."
        elif obv_d > 0 and price_d <= 0:
            out["obv"] = "🟡 OBV rising while price is flat/down — quiet accumulation; watch for a breakout."
        elif obv_d < 0 and price_d > 0:
            out["obv"] = "🔴 OBV falling while price rises — distribution; bearish divergence, trend at risk."
        else:
            out["obv"] = "🔴 OBV falling with price — selling pressure dominant."
        # Detail: is OBV at a fresh high/low vs the window? (leads price turns)
        obv_win = df["obv"].tail(window + 1)
        if not obv_win.isna().all():
            at_high = last["obv"] >= obv_win.max() - 1e-9
            at_low = last["obv"] <= obv_win.min() + 1e-9
            if at_high:
                out["obv_detail"] = ("OBV just printed a fresh "
                    f"{window}-day high — buyers stepped up before price confirmed; "
                    "often leads a breakout.")
            elif at_low:
                out["obv_detail"] = ("OBV at a fresh "
                    f"{window}-day low — net selling; any price strength is suspect "
                    "until OBV turns up.")
            else:
                out["obv_detail"] = ("OBV measures cumulative up-day minus down-day "
                    "volume. When it confirms price you can trust the move; when it "
                    "diverges, the crowd is quietly exiting.")
    mfi = last.get("mfi14") if "mfi14" in df.columns else None
    if mfi is not None and not pd.isna(mfi):
        m = float(mfi)
        if m >= 80:
            out["mfi"] = f"🔴 MFI {m:.0f} — overbought; money flow stretched, pullback risk."
        elif m <= 20:
            out["mfi"] = f"🟢 MFI {m:.0f} — oversold; potential bounce zone."
        elif m >= 50:
            out["mfi"] = f"🟢 MFI {m:.0f} — healthy buying-side money flow."
        else:
            out["mfi"] = f"🟡 MFI {m:.0f} — weak money flow, sellers slightly ahead."
        # Detail: trend of MFI over the window + divergence check
        mfi_then = df["mfi14"].iloc[-1 - window] if len(df) > window else None
        if mfi_then is not None and not pd.isna(mfi_then):
            delta = m - float(mfi_then)
            price_d = _slope_dir(df["close"], window)
            if price_d > 0 and delta < -5:
                out["mfi_detail"] = (f"MFI fell {abs(delta):.0f} pts while price rose "
                    "— bearish divergence; buying conviction is fading even as price climbs.")
            elif price_d < 0 and delta > 5:
                out["mfi_detail"] = (f"MFI rose {delta:.0f} pts while price fell "
                    "— bullish divergence; sellers are exhausting, watch for a turn.")
            else:
                out["mfi_detail"] = ("MFI is a volume-weighted RSI (0–100): >80 buyers "
                    "exhausted, <20 sellers exhausted, 50 the neutral line. Divergence "
                    "from price is the signal worth watching.")
    return out


def volume_profile_read(df: pd.DataFrame, profile: pd.DataFrame | None = None,
                        bins: int = 24, lookback: int = 60,
                        value_area_pct: float = 0.70) -> dict | None:
    """Interpret a volume-by-price profile: Point of Control, Value Area, and
    where current price sits relative to the high-volume nodes.

    Returns a dict with numeric levels + a list of plain-English `reasons`,
    or None if there isn't enough data.
    """
    if profile is None:
        profile = volume_profile(df, bins=bins, lookback=lookback)
    if profile is None or profile.empty or df is None or df.empty:
        return None

    price = float(df["close"].iloc[-1])
    total = float(profile["volume"].sum()) or 1.0

    # Point of Control — the single most-traded price level (fairest value).
    poc_row = profile.loc[profile["volume"].idxmax()]
    poc = float(poc_row["price_mid"])

    # Value Area — the contiguous-ish set of levels holding ~70% of volume.
    ordered = profile.sort_values("volume", ascending=False)
    cum, va_levels = 0.0, []
    for _, r in ordered.iterrows():
        va_levels.append(r)
        cum += float(r["volume"])
        if cum >= value_area_pct * total:
            break
    va_low = min(float(r["price_low"]) for r in va_levels)
    va_high = max(float(r["price_high"]) for r in va_levels)

    # High-volume nodes (>=70% of POC volume) above & below current price → S/R.
    poc_vol = float(poc_row["volume"])
    hvn = profile[profile["volume"] >= 0.7 * poc_vol]
    above = hvn[hvn["price_mid"] > price]["price_mid"]
    below = hvn[hvn["price_mid"] < price]["price_mid"]
    resistance = float(above.min()) if not above.empty else None
    support = float(below.max()) if not below.empty else None

    reasons: list[str] = []
    # Where is price vs the value area?
    if price > va_high:
        reasons.append(f"🟡 Price ₹{price:,.0f} is **above the value area** "
                       f"(₹{va_low:,.0f}–₹{va_high:,.0f}) — extended into thin volume; "
                       "moves up here are easy but unsupported, prone to snap back to value.")
    elif price < va_low:
        reasons.append(f"🟢 Price ₹{price:,.0f} is **below the value area** "
                       f"(₹{va_low:,.0f}–₹{va_high:,.0f}) — trading at a discount to where "
                       "most shares changed hands; a reclaim of the value area is the bull trigger.")
    else:
        reasons.append(f"⚪ Price ₹{price:,.0f} sits **inside the value area** "
                       f"(₹{va_low:,.0f}–₹{va_high:,.0f}) — fair value, balanced; "
                       "wait for a break out of this range for a directional edge.")
    # POC relationship
    if poc > price:
        reasons.append(f"Heaviest trading (Point of Control) is **overhead at ₹{poc:,.0f}** "
                       "— acts as a magnet/resistance; lots of trapped buyers may sell into a rally there.")
    else:
        reasons.append(f"Heaviest trading (Point of Control) is **below at ₹{poc:,.0f}** "
                       "— a volume shelf that tends to support pullbacks.")
    # Nearest S/R nodes
    if support is not None:
        reasons.append(f"Nearest high-volume support: **₹{support:,.0f}** "
                       f"({(price-support)/price*100:.1f}% below) — natural place for a stop to sit under.")
    if resistance is not None:
        reasons.append(f"Nearest high-volume resistance: **₹{resistance:,.0f}** "
                       f"({(resistance-price)/price*100:.1f}% above) — first target / where to expect supply.")
    if support is None and resistance is None:
        reasons.append("No dominant high-volume node nearby — volume is spread thin, "
                       "so price can move quickly with little friction.")

    return {
        "poc": round(poc, 2),
        "value_area_low": round(va_low, 2),
        "value_area_high": round(va_high, 2),
        "support": round(support, 2) if support is not None else None,
        "resistance": round(resistance, 2) if resistance is not None else None,
        "price": round(price, 2),
        "reasons": reasons,
    }


def signal_scoreboard(df: pd.DataFrame) -> dict | None:
    """Aggregate the major indicators into ONE clear buy/sell tilt.

    Every indicator casts a bull (+) / bear (−) / neutral vote with a weight.
    Returns:
      tilt   : "STRONG BUY" | "BUY" | "NEUTRAL" | "SELL" | "STRONG SELL"
      score  : −100..+100 (weighted net, normalised)
      bull/bear/neutral : lists of plain-English reasons
    or None if indicators aren't computed yet.
    """
    need = ("close", "ema50", "ema200", "rsi14", "macd", "macd_signal")
    if df is None or df.empty or len(df) < 60 or any(c not in df.columns for c in need):
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(last["close"])

    bull: list[str] = []
    bear: list[str] = []
    neutral: list[str] = []
    score = 0.0

    def vote(weight: float, cond_bull, cond_bear, bull_msg, bear_msg,
             neutral_msg=None):
        nonlocal score
        if cond_bull:
            score += weight
            bull.append(bull_msg)
        elif cond_bear:
            score -= weight
            bear.append(bear_msg)
        elif neutral_msg:
            neutral.append(neutral_msg)

    def fv(col):
        v = last.get(col)
        return float(v) if v is not None and not pd.isna(v) else None

    # --- Trend (EMA stack) — weight 2 ---
    e50, e200 = fv("ema50"), fv("ema200")
    if e50 and e200:
        vote(2.0, close > e50 > e200, close < e50 < e200,
             "Price above 50 & 200 EMA — full uptrend stack",
             "Price below 50 & 200 EMA — full downtrend stack",
             "Price tangled in the EMAs — no clean trend")

    # --- Supertrend regime — weight 2 (the explicit buy/sell line) ---
    stdir = fv("supertrend_dir")
    stval = fv("supertrend")
    if stdir is not None:
        flipped = (prev.get("supertrend_dir") is not None
                   and not pd.isna(prev.get("supertrend_dir"))
                   and float(prev["supertrend_dir"]) != stdir)
        tag = " (just flipped)" if flipped else ""
        vote(2.0, stdir > 0, stdir < 0,
             f"Supertrend BULLISH{tag}" + (f" — trailing support ₹{stval:,.0f}" if stval else ""),
             f"Supertrend BEARISH{tag}" + (f" — overhead resistance ₹{stval:,.0f}" if stval else ""))

    # --- ADX trend strength + direction — weight 2 ---
    adx, dip, dim = fv("adx14"), fv("di_plus"), fv("di_minus")
    if adx is not None and dip is not None and dim is not None:
        if adx >= 25 and dip > dim:
            score += 2.0
            bull.append(f"ADX {adx:.0f} (strong) with DI+ > DI− — trend up and powering")
        elif adx >= 25 and dim > dip:
            score -= 2.0
            bear.append(f"ADX {adx:.0f} (strong) with DI− > DI+ — downtrend in force")
        elif adx < 20:
            neutral.append(f"ADX {adx:.0f} — weak/no trend; range-bound, breakouts may fail")
        else:
            (bull if dip > dim else bear).append(
                f"ADX {adx:.0f} (building) — {'bulls' if dip > dim else 'bears'} edging ahead")
            score += 0.5 if dip > dim else -0.5

    # --- MACD — weight 1.5 ---
    macd, sig = fv("macd"), fv("macd_signal")
    hist = fv("macd_hist")
    if macd is not None and sig is not None:
        cross_up = (prev.get("macd") is not None and not pd.isna(prev.get("macd"))
                    and prev["macd"] <= prev["macd_signal"] and macd > sig)
        cross_dn = (prev.get("macd") is not None and not pd.isna(prev.get("macd"))
                    and prev["macd"] >= prev["macd_signal"] and macd < sig)
        if cross_up:
            score += 1.5; bull.append("MACD just crossed ABOVE signal — fresh momentum")
        elif cross_dn:
            score -= 1.5; bear.append("MACD just crossed BELOW signal — momentum rolling over")
        else:
            vote(1.0, macd > sig, macd < sig,
                 "MACD above signal" + (" & histogram rising" if hist and hist > 0 else ""),
                 "MACD below signal — bearish momentum")

    # --- RSI — weight 1 ---
    rsi = fv("rsi14")
    if rsi is not None:
        if rsi > 75:
            score -= 0.5; bear.append(f"RSI {rsi:.0f} — overbought, pullback risk")
        elif rsi < 25:
            neutral.append(f"RSI {rsi:.0f} — deeply oversold; bounce candidate but knife-catchy")
        else:
            vote(1.0, 50 <= rsi <= 70, rsi < 45,
                 f"RSI {rsi:.0f} — healthy bullish zone",
                 f"RSI {rsi:.0f} — momentum weak")

    # --- Stochastic — weight 1 ---
    k, d = fv("stoch_k"), fv("stoch_d")
    if k is not None and d is not None:
        if k > 80:
            neutral.append(f"Stochastic {k:.0f} — overbought")
        elif k < 20 and k > d:
            score += 1.0; bull.append(f"Stochastic turning up from oversold ({k:.0f})")
        else:
            vote(0.7, k > d, k < d, "Stochastic %K above %D — short-term up",
                 "Stochastic %K below %D — short-term down")

    # --- Bollinger %B — weight 0.8 ---
    bbp = fv("bb_pct")
    if bbp is not None:
        if bbp > 1.0:
            score += 0.8; bull.append("Price riding ABOVE the upper Bollinger band — strong thrust")
        elif bbp < 0.0:
            neutral.append("Price below the lower Bollinger band — stretched; mean-reversion zone")
        else:
            vote(0.6, bbp > 0.6, bbp < 0.4, "Upper half of the Bollinger range",
                 "Lower half of the Bollinger range")

    # --- Money flow (MFI + OBV) — weight 1 ---
    mfi = fv("mfi14")
    if mfi is not None:
        vote(1.0, mfi >= 55, mfi < 45,
             f"MFI {mfi:.0f} — money flowing in", f"MFI {mfi:.0f} — money flowing out")
    if "obv" in df.columns:
        obv_d = _slope_dir(df["obv"], 20)
        vote(0.8, obv_d > 0, obv_d < 0,
             "OBV rising — accumulation", "OBV falling — distribution")

    # --- Normalise to −100..100 (max attainable ≈ 12.1) and label ---
    max_w = 12.1
    norm = max(-100, min(100, round(score / max_w * 100)))
    n_bull, n_bear = len(bull), len(bear)
    if norm >= 55 and n_bull >= n_bear + 2:
        tilt = "STRONG BUY"
    elif norm >= 20:
        tilt = "BUY"
    elif norm <= -55 and n_bear >= n_bull + 2:
        tilt = "STRONG SELL"
    elif norm <= -20:
        tilt = "SELL"
    else:
        tilt = "NEUTRAL"
    return {"tilt": tilt, "score": norm, "bull": bull, "bear": bear,
            "neutral": neutral, "n_bull": n_bull, "n_bear": n_bear}


def volume_profile(df: pd.DataFrame, bins: int = 24,
                   lookback: int = 60) -> pd.DataFrame:
    """Volume-by-price histogram for the last `lookback` bars.

    Each bar contributes its volume to the price bucket of its (high+low+close)/3
    typical price. Returns a DataFrame [price_low, price_high, volume, pct].
    Used to find high-volume nodes (price levels where lots of shares changed
    hands — they tend to act as support/resistance).
    """
    if df.empty or len(df) < 5:
        return pd.DataFrame()
    window = df.tail(lookback).copy()
    if "typical" not in window:
        window["typical"] = (window["high"] + window["low"] + window["close"]) / 3
    low, high = window["typical"].min(), window["typical"].max()
    if low == high:
        return pd.DataFrame()
    edges = np.linspace(low, high, bins + 1)
    # np.histogram weights each bar's typical price by its volume
    hist, _ = np.histogram(window["typical"], bins=edges, weights=window["volume"])
    total = hist.sum() or 1
    rows = []
    for i in range(bins):
        rows.append({
            "price_low": float(edges[i]),
            "price_high": float(edges[i + 1]),
            "price_mid": float((edges[i] + edges[i + 1]) / 2),
            "volume": float(hist[i]),
            "pct": float(hist[i] / total * 100),
        })
    return pd.DataFrame(rows)
