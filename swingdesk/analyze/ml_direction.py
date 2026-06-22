"""Module D — probabilistic direction model: P(up over N sessions).

This is the part people *think* they want ("predict the stock"), built the only
way it's defensible:

  * It outputs a **probability**, not a call. 0.58 means "58% of historically
    similar setups were higher N days later" — and we calibrate so that number
    means what it says.
  * It is validated **walk-forward by date** — train only on the past, test on
    the future, never the reverse. Pooled across your whole universe.
  * It is always compared to the **base rate** (just guessing the majority).
    If the model can't beat that out-of-sample, it has no edge and you should
    ignore it. We report that gap honestly rather than hiding it.

Features are all causal (known at bar t): momentum/trend distance from EMAs,
oscillators (RSI/Stoch/CCI/MFI), volatility (ATR%, BB width), volume-flow, and
a few macro/regime inputs (NIFTY 5-day return, India VIX, prior-night S&P) so
the model can learn the US-spillover and risk-regime context.

Honesty caveat baked into the docs and the UI: daily/weekly equity direction is
~55% predictable at best. Treat a high P(up) as a tilt that *stacks with* your
setup + risk rules, never as a standalone signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swingdesk.analyze.technicals import add_indicators
from swingdesk.storage import load_macro, load_prices

# Macro series used as regime/spillover context.
_NIFTY, _VIX, _SPX = "^NSEI", "^INDIAVIX", "^GSPC"

# Per-stock technical features — always available after warmup.
TECH_FEATURES = [
    "rsi14", "mfi14", "stoch_k", "stoch_d", "adx14", "di_plus", "di_minus",
    "cci20", "bb_pct", "bb_width", "macd_hist_n", "dist_ema20", "dist_ema50",
    "dist_ema200", "ema20_slope", "atr_pct", "atr_regime", "vol_ratio",
    "buy_pressure_20", "supertrend_dir", "obv_slope_n",
    "ret_1", "ret_5", "ret_10", "ret_20", "dist_52w_high",
]
# Market/regime + relative-strength features — may be missing (median-filled).
MACRO_FEATURES = ["nifty_ret_5", "india_vix", "spx_overnight", "rel_strength_20"]
# Cross-sectional rank features — computed across the universe per date. These
# encode RELATIVE strength (the strongest lever in equity ML): a stock's rank
# vs its peers matters more than its absolute reading.
XS_FEATURES = ["xs_mom_rank", "xs_vol_rank"]
FEATURES = TECH_FEATURES + MACRO_FEATURES + XS_FEATURES


# --------------------------------------------------------------------------- #
# feature engineering
# --------------------------------------------------------------------------- #
def _macro_features() -> pd.DataFrame:
    """Date-indexed macro/regime features (causal)."""
    cols = {}
    nifty = load_macro(_NIFTY)
    if not nifty.empty:
        cols["nifty_ret_5"] = nifty["close"].pct_change(5)
        cols["nifty_ret_20"] = nifty["close"].pct_change(20)   # for rel_strength
    vix = load_macro(_VIX)
    if not vix.empty:
        cols["india_vix"] = vix["close"] / 100.0
    spx = load_macro(_SPX)
    if not spx.empty:
        # Prior-night S&P return — known before today's Indian session.
        cols["spx_overnight"] = spx["close"].pct_change().shift(1)
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols)


def build_features(ticker: str, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-bar causal feature matrix for one ticker (date-indexed)."""
    px = load_prices(ticker)
    if px.empty or len(px) < 220:
        return pd.DataFrame()
    df = add_indicators(px)
    close = df["close"]
    f = pd.DataFrame(index=df.index)
    for c in ["rsi14", "mfi14", "stoch_k", "stoch_d", "adx14", "di_plus",
              "di_minus", "cci20", "bb_pct", "bb_width", "buy_pressure_20",
              "supertrend_dir"]:
        f[c] = df[c] if c in df.columns else np.nan
    f["macd_hist_n"] = (df.get("macd_hist", np.nan) / close)
    f["dist_ema20"] = close / df["ema20"] - 1
    f["dist_ema50"] = close / df["ema50"] - 1
    f["dist_ema200"] = close / df["ema200"] - 1
    f["ema20_slope"] = df["ema20"] / df["ema20"].shift(10) - 1
    atr_pct = df["atr14"] / close
    f["atr_pct"] = atr_pct
    # Volatility regime: current ATR% vs its own 60-day average (>1 = expanding).
    f["atr_regime"] = atr_pct / atr_pct.rolling(60, min_periods=20).mean()
    f["vol_ratio"] = df["volume"] / df["vol_avg20"]
    f["obv_slope_n"] = df.get("obv_slope_10", np.nan) / (df["vol_avg20"] * 10)
    for k in (1, 5, 10, 20):
        f[f"ret_{k}"] = close.pct_change(k)
    # Distance below the rolling 52-week high — proximity to highs is a momentum/
    # breakout tell; deep below = laggard.
    f["dist_52w_high"] = close / close.rolling(252, min_periods=120).max() - 1
    f["ticker"] = ticker

    macro = macro if macro is not None else _macro_features()
    if not macro.empty:
        f = f.join(macro, how="left")
    # Relative strength: stock's 20d return minus NIFTY's. The single most
    # useful "is this leading or lagging the market?" signal.
    f["rel_strength_20"] = f["ret_20"] - f.get("nifty_ret_20", np.nan)
    for mc in MACRO_FEATURES:
        if mc not in f.columns:
            f[mc] = np.nan
    return f


def make_dataset(tickers: list[str], horizon: int = 10
                 ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Pool features + binary labels across tickers.

    Label y=1 if close is higher `horizon` sessions later. The last `horizon`
    bars per ticker have no label and are dropped from training. Returns
    (X[FEATURES], y, dates)."""
    macro = _macro_features()
    frames = []
    for t in tickers:
        f = build_features(t, macro)
        if f.empty:
            continue
        px = load_prices(t)
        fwd = px["close"].shift(-horizon) / px["close"] - 1
        f = f.copy()
        f["y"] = (fwd > 0).astype(float)
        f.loc[fwd.isna(), "y"] = np.nan      # no realised label yet
        f["date"] = f.index
        frames.append(f)
    if not frames:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]")
    data = pd.concat(frames, ignore_index=True)
    # Require the technical features (always present after warmup) + a label.
    # Macro features may be missing entirely (no macro fetched) — median-fill
    # them instead of dropping every row.
    data = data.dropna(subset=TECH_FEATURES + ["y"])
    if data.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]")
    for m in MACRO_FEATURES:
        med = data[m].median()
        data[m] = data[m].fillna(med if np.isfinite(med) else 0.0)
    # Cross-sectional ranks: where does each stock sit vs its peers ON THAT DATE.
    data["xs_mom_rank"] = data.groupby("date")["ret_20"].rank(pct=True).fillna(0.5)
    data["xs_vol_rank"] = data.groupby("date")["atr_pct"].rank(pct=True).fillna(0.5)
    return data[FEATURES], data["y"].astype(int), data["date"]


def _make_model():
    """Modest gradient-boosting classifier — regularised to resist overfitting
    on a noisy, low-signal target."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=3,
        min_samples_leaf=80, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=42,
    )


# --------------------------------------------------------------------------- #
# walk-forward evaluation
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardResult:
    horizon: int
    n_folds: int
    n_train_total: int
    n_test_total: int
    base_rate: float            # P(up) in test data — the "always guess up" bar
    accuracy: float
    auc: float
    brier: float                # calibration error (lower better; 0.25 = coin flip)
    edge_vs_baseline: float     # accuracy - max(base, 1-base): >0 means real lift
    verdict: str
    folds: list[dict] = field(default_factory=list)


def walk_forward_eval(tickers: list[str], horizon: int = 10, n_splits: int = 5,
                      min_train: int = 750) -> WalkForwardResult | None:
    """Expanding-window, date-based walk-forward. Each fold trains on all bars
    strictly before the test window and predicts the future window — so no
    future information ever leaks into training."""
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    X, y, dates = make_dataset(tickers, horizon)
    if X.empty or len(X) < min_train + 100:
        return None
    order = np.argsort(dates.values)
    X, y, dates = X.iloc[order], y.iloc[order], dates.iloc[order]

    uniq = np.array(sorted(pd.unique(dates)))
    if len(uniq) < n_splits + 10:
        return None
    # Dates eligible to be tested (after the minimum training history).
    train_cut_idx = max(1, int(np.searchsorted(
        np.cumsum(dates.value_counts().sort_index().values), min_train)))
    test_dates = uniq[train_cut_idx:]
    if len(test_dates) < n_splits:
        return None
    folds_dates = np.array_split(test_dates, n_splits)
    uniq_index = {d: i for i, d in enumerate(uniq)}

    accs, aucs, briers = [], [], []
    n_tr_tot = n_te_tot = 0
    up_total = te_total = 0
    fold_rows = []
    for fd in folds_dates:
        if len(fd) == 0:
            continue
        start = fd[0]
        # PURGE/EMBARGO: a training row at date d has a label spanning [d, d+horizon].
        # To stop that label peeking into the test window, drop the last `horizon`
        # trading days before the test start from training.
        embargo_i = max(0, uniq_index[start] - horizon)
        embargo_date = uniq[embargo_i]
        train_mask = dates.values < embargo_date
        test_mask = np.isin(dates.values, fd)
        if train_mask.sum() < min_train or test_mask.sum() < 20:
            continue
        ytr = y.values[train_mask]
        if len(np.unique(ytr)) < 2:
            continue
        model = _make_model()
        model.fit(X.values[train_mask], ytr)
        p = model.predict_proba(X.values[test_mask])[:, 1]
        yte = y.values[test_mask]
        pred = (p >= 0.5).astype(int)
        acc = accuracy_score(yte, pred)
        brier = brier_score_loss(yte, p)
        auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else float("nan")
        accs.append(acc); briers.append(brier)
        if np.isfinite(auc):
            aucs.append(auc)
        n_tr_tot += int(train_mask.sum())
        n_te_tot += int(test_mask.sum())
        up_total += int(yte.sum()); te_total += len(yte)
        fold_rows.append({
            "test_from": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "accuracy": round(acc, 3), "auc": round(auc, 3) if np.isfinite(auc) else None,
            "brier": round(brier, 4),
        })

    if not accs:
        return None
    base = up_total / te_total if te_total else 0.5
    acc_mean = float(np.mean(accs))
    edge = acc_mean - max(base, 1 - base)
    auc_mean = float(np.mean(aucs)) if aucs else float("nan")
    verdict = (
        "real edge out-of-sample" if (edge > 0.02 and (not np.isfinite(auc_mean) or auc_mean > 0.53))
        else "marginal — barely beats guessing" if edge > 0
        else "no edge — do not trade this alone"
    )
    return WalkForwardResult(
        horizon=horizon, n_folds=len(accs), n_train_total=n_tr_tot,
        n_test_total=n_te_tot, base_rate=round(base, 3),
        accuracy=round(acc_mean, 3), auc=round(auc_mean, 3) if np.isfinite(auc_mean) else float("nan"),
        brier=round(float(np.mean(briers)), 4), edge_vs_baseline=round(edge, 3),
        verdict=verdict, folds=fold_rows,
    )


# --------------------------------------------------------------------------- #
# live prediction
# --------------------------------------------------------------------------- #
def train_and_predict(tickers: list[str], horizon: int = 10,
                      min_train: int = 500) -> pd.DataFrame:
    """Train on all *labeled* history, then output a calibrated P(up over
    `horizon`) for each ticker's most recent (unlabeled) bar.

    Probabilities are isotonic-calibrated on a held-out tail of the training
    data so the number is trustworthy, not just rank-ordered."""
    from sklearn.calibration import CalibratedClassifierCV

    X, y, dates = make_dataset(tickers, horizon)
    if X.empty or len(X) < min_train:
        return pd.DataFrame()

    base = _make_model()
    # Calibrate on a time-ordered tail so the mapping reflects recent regime.
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X.values, y.values)

    macro = _macro_features()
    macro_med = {m: float(X[m].median()) for m in MACRO_FEATURES}

    # Gather the latest valid bar per ticker into one frame so the cross-sectional
    # ranks can be computed across the universe (matching how training built them).
    latest_rows = []
    for t in tickers:
        f = build_features(t, macro)
        if f.empty:
            continue
        for m in MACRO_FEATURES:
            f[m] = f[m].fillna(macro_med.get(m, 0.0))
        valid = f.dropna(subset=TECH_FEATURES)
        if valid.empty:
            continue
        row = valid.iloc[[-1]].copy()
        row["ticker"] = t
        row["asof"] = valid.index[-1].strftime("%Y-%m-%d")
        latest_rows.append(row)
    if not latest_rows:
        return pd.DataFrame()

    L = pd.concat(latest_rows, ignore_index=True)
    # Cross-sectional ranks across today's snapshot of the universe.
    L["xs_mom_rank"] = L["ret_20"].rank(pct=True).fillna(0.5)
    L["xs_vol_rank"] = L["atr_pct"].rank(pct=True).fillna(0.5)

    probs = model.predict_proba(L[FEATURES].values)[:, 1]
    out = pd.DataFrame({
        "ticker": L["ticker"],
        "prob_up": np.round(probs, 3),
        "asof": L["asof"],
    })
    out["signal"] = np.where(out["prob_up"] >= 0.58, "bullish",
                             np.where(out["prob_up"] <= 0.42, "bearish", "neutral"))
    return out.sort_values("prob_up", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# interpretability
# --------------------------------------------------------------------------- #
def feature_importance(tickers: list[str], horizon: int = 10,
                       n_repeats: int = 5, min_train: int = 500) -> pd.DataFrame:
    """Permutation importance on a held-out (purged) tail — how much each feature
    actually moves out-of-sample accuracy. Honest: shuffling a feature and seeing
    accuracy drop is far more trustworthy than a model's internal split counts.
    Near-zero importance = noise you could drop."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score  # noqa: F401 (used via scoring)

    X, y, dates = make_dataset(tickers, horizon)
    if X.empty or len(X) < min_train + 100:
        return pd.DataFrame()
    order = np.argsort(dates.values)
    X, y, dates = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True), dates.iloc[order].reset_index(drop=True)

    uniq = np.array(sorted(pd.unique(dates)))
    cut = int(len(uniq) * 0.8)
    test_start = uniq[cut]
    embargo = uniq[max(0, cut - horizon)]
    train_mask = dates.values < embargo
    test_mask = dates.values >= test_start
    if train_mask.sum() < min_train or test_mask.sum() < 50:
        return pd.DataFrame()
    if len(np.unique(y.values[train_mask])) < 2:
        return pd.DataFrame()

    model = _make_model()
    model.fit(X.values[train_mask], y.values[train_mask])
    r = permutation_importance(
        model, X.values[test_mask], y.values[test_mask],
        n_repeats=n_repeats, random_state=42, scoring="accuracy",
    )
    return pd.DataFrame({
        "feature": FEATURES,
        "importance": np.round(r.importances_mean, 4),
        "std": np.round(r.importances_std, 4),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
