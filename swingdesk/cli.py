from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python swingdesk/cli.py` in addition to `python -m swingdesk.cli`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from swingdesk.analyze import score, sentiment
from swingdesk.analyze.setups import scan_all
from swingdesk.config import DEFAULT_WATCHLIST
from swingdesk.ingest import earnings as earnings_ingest
from swingdesk.ingest import global_news as global_news_ingest
from swingdesk.ingest import fundamentals as fundamentals_ingest
from swingdesk.ingest import macro as macro_ingest
from swingdesk.ingest import nse as nse_ingest
from swingdesk.ingest import news_rss, prices
from swingdesk.notify import telegram
from swingdesk.portfolio import journal as pj
from swingdesk.portfolio import positions as portfolio
from swingdesk.portfolio import paper_trader as paper_trader_mod
from swingdesk.portfolio import holdings as holdings_mod
from swingdesk.portfolio import import_groww
from swingdesk.portfolio import reconcile as reconcile_mod
from swingdesk.storage import (
    add_to_smallcap_watchlist,
    combined_universe,
    get_position,
    get_smallcap_watchlist,
    get_watchlist,
    holdings_tickers,
    init_db,
    load_positions,
    load_signals,
    remove_from_smallcap_watchlist,
    save_signals,
    seed_watchlist_if_empty,
    set_smallcap_watchlist,
    set_watchlist,
)

console = Console()


def cmd_init(args):
    init_db()
    seed_watchlist_if_empty(DEFAULT_WATCHLIST)
    console.print(f"[green]initialized DB and watchlist ({len(get_watchlist())} tickers)[/green]")


def cmd_prices(args):
    init_db()
    seed_watchlist_if_empty(DEFAULT_WATCHLIST)
    universe = combined_universe()  # watchlist + holdings
    console.print(f"[bold]Fetching prices for {len(universe)} tickers "
                  f"({len(get_watchlist())} watchlist + {len(holdings_tickers())} holdings) "
                  f"(period={args.period})[/bold]")
    prices.ingest(universe, period=args.period, workers=args.workers)


def cmd_news(args):
    init_db()
    seed_watchlist_if_empty(DEFAULT_WATCHLIST)
    # Tag headlines against the entire investable universe — watchlist +
    # holdings + (optionally) discovery + smallcap pools — so news mentioning
    # a small cap you don't yet own still gets surfaced.
    universe = combined_universe(include_smallcaps=args.include_smallcaps,
                                  include_discovery=args.include_discovery)
    console.print(f"[bold]Fetching news (matching against {len(universe)} tickers)[/bold]")
    n = news_rss.ingest(universe)
    console.print(f"[green]{n} new news items[/green]")
    if getattr(args, "retag", False):
        m = news_rss.retag_news(universe)
        console.print(f"[green]re-tagged {m} existing news rows[/green]")


def cmd_scan(args):
    init_db()
    seed_watchlist_if_empty(DEFAULT_WATCHLIST)
    wl = get_watchlist()
    console.print(f"[bold]Scanning {len(wl)} tickers for setups[/bold]")
    sigs = scan_all(wl, persist=False)
    if sigs:
        sigs = score.enrich(sigs)
        save_signals(sigs)
    console.print(f"[green]{len(sigs)} signals generated[/green]")


def cmd_sentiment(args):
    init_db()
    sentiment.ingest(max_items=args.max)


def cmd_notify(args):
    sigs_df = load_signals(limit=args.limit)
    if sigs_df.empty:
        console.print("[yellow]no signals to notify[/yellow]")
        return
    sigs = sigs_df.to_dict(orient="records")
    sent = telegram.send_signals(sigs)
    console.print(f"[green]sent: {bool(sent)}[/green]")


def cmd_schedule(args):
    from swingdesk import scheduler
    scheduler.run(once=args.once)


def cmd_backtest(args):
    """Run a walk-forward backtest over the watchlist and report metrics."""
    from datetime import datetime

    from swingdesk.backtest import engine, metrics
    from swingdesk.storage import save_backtest_trades

    init_db()
    seed_watchlist_if_empty(DEFAULT_WATCHLIST)
    wl = [args.ticker] if args.ticker else get_watchlist()

    console.print(f"[bold]Backtesting {len(wl)} ticker(s), max_hold={args.max_hold}[/bold]")
    trades = engine.backtest_universe(wl, max_hold=args.max_hold)

    if trades.empty:
        console.print("[yellow]no trades produced (insufficient data or no setups fired)[/yellow]")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_backtest_trades(run_id, trades)

    summary = metrics.summarize(trades)
    console.rule(f"[bold]Backtest summary — run_id={run_id}")
    cols = ["setup", "n_trades", "win_rate", "avg_r", "total_r",
            "expectancy", "profit_factor", "max_drawdown_r", "max_consec_losses",
            "avg_bars_held"]
    console.print(summary[cols].to_string(index=False))

    console.rule("[bold]Edge gating")
    for _, row in summary.iterrows():
        if row["setup"] == "ALL":
            continue
        ok, fails = metrics.gate(row.to_dict())
        if ok:
            console.print(f"  [green]PASS[/green]  {row['setup']:<20}  expectancy={row['expectancy']} pf={row['profit_factor']}")
        else:
            console.print(f"  [red]FAIL[/red]  {row['setup']:<20}  reasons: {', '.join(fails)}")


def cmd_run(args):
    cmd_prices(args)
    cmd_news(args)
    cmd_sentiment(args)
    cmd_scan(args)
    # Refresh earnings calendar once a day so the blackout filter stays accurate
    if not args.skip_earnings:
        try:
            earnings_ingest.ingest(get_watchlist())
        except Exception as e:
            console.print(f"[yellow]earnings refresh failed (non-fatal): {e}[/yellow]")
    # Mark open paper positions: trail stops, auto-close on SL/target
    mark = portfolio.mark_to_market()
    if mark.get("closed"):
        console.print(f"[yellow]auto-closed {mark['closed']} paper positions[/yellow]")
    if mark.get("trailed"):
        console.print(f"[cyan]trailed stops on {mark['trailed']} positions[/cyan]")
    # Auto paper-trade today's top signals
    sigs_df = load_signals(limit=20)
    if not sigs_df.empty:
        sigs = sigs_df.to_dict(orient="records")
        result = portfolio.auto_paper_trade(sigs, min_composite=args.min_score)
        console.print(f"[bold]paper-trade: opened {result['opened']}, "
                      f"skipped {result['skipped']}[/bold]")
        telegram.send_signals(sigs)


def cmd_paper_step(args):
    """Run one step of the execution-cost-aware paper autotrader (loop-friendly)."""
    cfg = paper_trader_mod.AutoTraderConfig(
        min_score=args.min_score, algo=args.algo,
        max_portfolio_heat_pct=args.max_heat, kill_switch_dd_pct=args.kill_dd,
        flatten_on_kill=args.flatten_on_kill)
    universe = get_watchlist()
    rep = paper_trader_mod.step(cfg, universe=universe, refresh=args.refresh)
    state = "⛔ HALTED" if rep.halted else "🟢 active"
    console.print(
        f"[bold]paper-step[/bold] {rep.asof}  {state}  "
        f"equity ₹{rep.equity:,.0f}  dd {rep.drawdown_pct:.1f}%  "
        f"heat {rep.portfolio_heat_pct:.1f}%  "
        f"[green]+{len(rep.entries)} opened[/green] "
        f"[yellow]{len(rep.exits)} closed[/yellow]  {rep.n_open} open")
    for e in rep.entries:
        console.print(f"  [green]open[/green] {e['ticker']:>15} x{e['qty']} "
                      f"@ ₹{e['fill']} (+{e['slip_bps']:.0f}bps) · {e['setup']}")
    for x in rep.exits:
        console.print(f"  [yellow]exit[/yellow] {x['ticker']:>15} {x['exit_reason']} "
                      f"P&L ₹{x['pnl']}")
    for n in rep.notes:
        console.print(f"  [dim]{n}[/dim]")


# --- portfolio commands ---------------------------------------------------------

def cmd_positions(args):
    df = load_positions(status=args.status if args.status != "all" else None,
                       is_paper=None if args.both else (not args.real))
    if df.empty:
        console.print("[yellow]no positions[/yellow]")
        return
    cols = ["id", "ticker", "setup", "qty", "entry_price", "stoploss", "target",
            "status", "exit_price", "pnl", "pnl_pct", "r_multiple",
            "is_paper", "entry_date", "exit_date"]
    cols = [c for c in cols if c in df.columns]
    console.print(df[cols].to_string(index=False))


def cmd_open(args):
    sl = float(args.stoploss)
    tgt = float(args.target)
    res = portfolio.open_position(
        ticker=args.ticker.upper(),
        entry=float(args.entry),
        stoploss=sl,
        target=tgt,
        setup=args.setup,
        is_paper=not args.real,
        notes=args.notes,
        qty=args.qty,
    )
    if res["status"] != "opened":
        console.print(f"[red]rejected: {res.get('reason')}[/red]")


def cmd_close(args):
    res = portfolio.close_position(args.id, exit_price=float(args.price),
                                   exit_reason=args.reason)
    if res["status"] == "not_found":
        console.print(f"[red]position #{args.id} not found[/red]")


def cmd_mark(args):
    res = portfolio.mark_to_market(refresh=args.refresh)
    console.print(
        f"checked={res['checked']}  closed={res['closed']}  marked={res.get('marked', 0)}  "
        f"trailed={res.get('trailed', 0)}  fetched={res.get('fetched', 0)}"
    )
    if res.get("skipped"):
        console.print(f"[yellow]no data for: {', '.join(res['skipped'])}[/yellow]")


def cmd_sync(args):
    """One-shot: pull fresh prices for all holdings + mark-to-market.
    Use this any time you want to refresh your paper book against live prices."""
    init_db()
    held = holdings_tickers()
    pos_df = load_positions(status="open")
    open_tickers = sorted(set(pos_df["ticker"].tolist()) if not pos_df.empty else [])
    universe = sorted(set(held) | set(open_tickers))
    if not universe:
        console.print("[yellow]no holdings or open positions yet[/yellow]")
        return
    console.print(f"[bold]Syncing {len(universe)} tickers (holdings + open positions)[/bold]")
    prices.ingest(universe, period="6mo", workers=6)
    res = portfolio.mark_to_market(refresh=False)
    console.print(
        f"[green]sync complete:[/green] marked {res.get('marked', 0)} positions, "
        f"closed {res['closed']} on SL/target, trailed {res.get('trailed', 0)}"
    )


def cmd_journal(args):
    s = pj.stats(is_paper=None if args.both else (not args.real))
    console.rule(f"[bold]Portfolio stats ({'real+paper' if args.both else ('REAL' if args.real else 'paper')})")
    console.print(f"  trades total      {s.n_trades}  (open: {s.n_open}, closed: {s.n_closed})")
    console.print(f"  win/loss          {s.wins}/{s.losses}  ({s.win_rate * 100:.1f}%)")
    console.print(f"  total P&L         ₹{s.total_pnl}")
    console.print(f"  avg P&L / trade   ₹{s.avg_pnl}")
    console.print(f"  best / worst      ₹{s.best_trade}  /  ₹{s.worst_trade}")
    console.print(f"  avg R / total R   {s.avg_r}  /  {s.total_r}")
    console.print(f"  avg holding       {s.avg_holding_days} days")
    console.print(f"  open risk         ₹{s.open_risk}")
    console.print(f"  unrealized P&L    ₹{s.unrealized_pnl}")

    by_s = pj.by_setup(is_paper=None if args.both else (not args.real))
    if not by_s.empty:
        console.rule("[bold]By setup")
        console.print(by_s.to_string(index=False))


def cmd_discover(args):
    """Find new stocks to invest in (outside watchlist + holdings)."""
    from swingdesk.analyze import discovery
    init_db()
    console.print(f"[bold]Scanning {len(discovery.DISCOVERY_UNIVERSE)} discovery candidates[/bold]")
    opps = discovery.scan()

    # High-conviction panel first — these are the "invest without much thought" picks
    if not args.no_conviction:
        hc = discovery.high_conviction(opps)
        if hc:
            console.rule(f"[bold green]🎯 HIGH-CONVICTION ({len(hc)} names) — all 4 lenses align")
            for o in hc[:5]:
                console.print(
                    f"  [bold]{o.ticker:>15}[/bold]  ({o.company})  score={o.composite_score:.1f}"
                )
                for r in o.reasons:
                    console.print(f"      [dim]· {r}[/dim]")

    filtered = [o for o in opps
                if o.quality_score is not None and o.quality_score >= args.min_quality]
    if args.active_only:
        filtered = [o for o in filtered if o.active_setup]
    if args.conviction_only:
        filtered = [o for o in filtered if o.conviction == "high"]

    console.rule(f"[bold]Ranked list (top {args.limit})"
                 f" — quality ≥ {args.min_quality}"
                 f"{', active-setup-only' if args.active_only else ''}"
                 f"{', conviction=high' if args.conviction_only else ''}")
    badge = {"high": "🟢", "medium": "🟡", "low": "⚪"}
    for o in filtered[:args.limit]:
        setup = f"[green]{o.active_setup}[/green]" if o.active_setup else "-"
        mom = f"{o.momentum_20d_pct:+.1f}%" if o.momentum_20d_pct else "—"
        tq = o.trend_verdict or "?"
        console.print(
            f"  {badge.get(o.conviction, '⚪')} {o.composite_score:>5.1f}  "
            f"{o.ticker:>15}  Q={o.quality_score:>5}  "
            f"trend={o.technical_state:<10}  trend-q={tq:<7}  "
            f"RSI={o.rsi or '?':<5}  20d={mom:<7}  setup={setup}"
        )
        for r in o.reasons[:3]:
            console.print(f"        [dim]· {r}[/dim]")


def cmd_sc_watchlist(args):
    """Manage the SEPARATE small-cap watchlist (independent from the main one)."""
    if args.add:
        added = 0
        for t in [x.strip().upper() for x in args.add.split(",") if x.strip()]:
            if add_to_smallcap_watchlist(t):
                added += 1
                console.print(f"[green]added {t}[/green]")
            else:
                console.print(f"[yellow]{t} already present[/yellow]")
        console.print(f"[bold]Total new: {added}[/bold]")
    elif args.remove:
        for t in [x.strip().upper() for x in args.remove.split(",") if x.strip()]:
            if remove_from_smallcap_watchlist(t):
                console.print(f"[green]removed {t}[/green]")
            else:
                console.print(f"[yellow]{t} not in watchlist[/yellow]")
    elif args.set:
        tickers = [t.strip().upper() for t in args.set.split(",") if t.strip()]
        set_smallcap_watchlist(tickers)
        console.print(f"[green]small-cap watchlist set to {len(tickers)} tickers[/green]")
    elif args.clear:
        set_smallcap_watchlist([])
        console.print("[green]small-cap watchlist cleared[/green]")
    else:
        wl = get_smallcap_watchlist()
        if not wl:
            console.print("[yellow](small-cap watchlist is empty — use --add or --set)[/yellow]")
        else:
            console.print(f"[bold]Small-cap watchlist ({len(wl)} tickers):[/bold]")
            for t in wl:
                console.print(f"  {t}")


def cmd_sc_scan(args):
    """Scan the small-cap WATCHLIST (not the full universe) and persist
    signals tagged with universe='smallcap' so they appear in the Signals
    tab with a clear label."""
    from swingdesk.analyze import score
    from swingdesk.analyze.setups import scan_all
    init_db()
    wl = get_smallcap_watchlist()
    if not wl:
        console.print("[yellow]small-cap watchlist is empty. Run "
                      "`swingdesk smallcaps` to discover candidates and add them.[/yellow]")
        return
    console.print(f"[bold]Scanning {len(wl)} small-cap watchlist tickers[/bold]")
    sigs = scan_all(wl, persist=False, universe="smallcap")
    if sigs:
        sigs = score.enrich(sigs)
        save_signals(sigs, universe="smallcap")
    console.print(f"[green]{len(sigs)} small-cap signals generated[/green]")


def cmd_smallcaps(args):
    """Find small-cap stocks to swing-trade. Separate from main Discover —
    same scoring + extra liquidity filter for thin-stock safety."""
    from swingdesk.analyze import smallcaps
    init_db()
    console.print(f"[bold]Scanning {len(smallcaps.SMALLCAP_UNIVERSE)} small-cap candidates[/bold]")
    opps = smallcaps.scan()

    if not args.no_conviction:
        hc = smallcaps.high_conviction(opps)
        if hc:
            console.rule(f"[bold green]🎯 SMALL-CAP HIGH-CONVICTION ({len(hc)})")
            for o in hc[:5]:
                console.print(
                    f"  [bold]{o.ticker:>15}[/bold]  ({o.company})  score={o.composite_score:.1f}"
                )
                for r in o.reasons:
                    console.print(f"      [dim]· {r}[/dim]")

    filtered = [o for o in opps
                if o.quality_score is not None and o.quality_score >= args.min_quality]
    if args.active_only:
        filtered = [o for o in filtered if o.active_setup]
    if args.conviction_only:
        filtered = [o for o in filtered if o.conviction == "high"]

    console.rule(f"[bold]Top {args.limit} small-caps "
                 f"(quality ≥ {args.min_quality}"
                 f"{', active-setup-only' if args.active_only else ''}"
                 f"{', conviction=high' if args.conviction_only else ''})")
    badge = {"high": "🟢", "medium": "🟡", "low": "⚪"}
    for o in filtered[:args.limit]:
        setup = f"[green]{o.active_setup}[/green]" if o.active_setup else "-"
        mom = f"{o.momentum_20d_pct:+.1f}%" if o.momentum_20d_pct else "—"
        tq = o.trend_verdict or "?"
        console.print(
            f"  {badge.get(o.conviction, '⚪')} {o.composite_score:>5.1f}  "
            f"{o.ticker:>15}  Q={o.quality_score:>5}  "
            f"trend={o.technical_state:<10}  trend-q={tq:<7}  "
            f"RSI={o.rsi or '?':<5}  20d={mom:<7}  setup={setup}"
        )
        for r in o.reasons[:3]:
            console.print(f"        [dim]· {r}[/dim]")


def cmd_warnings(args):
    """Show early-exit warnings for every open holding."""
    from swingdesk.analyze import early_exits
    init_db()
    held = holdings_tickers()
    if not held:
        console.print("[yellow]no holdings yet — import a CSV first[/yellow]")
        return
    console.rule(f"[bold]Early-exit warnings across {len(held)} holdings")
    flagged = 0
    for ticker in held:
        read = early_exits.evaluate(ticker)
        if read.action == "NONE":
            continue
        flagged += 1
        action_color = {"EXIT": "red", "TRIM_50": "orange3",
                        "TRIM_25": "yellow", "WATCH": "dim"}.get(read.action, "white")
        console.print(f"  [{action_color}]{read.action:<8}[/{action_color}] "
                      f"{ticker:>15}  severity={read.severity_total}")
        for w in read.warnings:
            console.print(f"        [{action_color}]·[/{action_color}] {w.reason}")
    if flagged == 0:
        console.print("[green]No warnings on any holding. Clean book.[/green]")


def cmd_enrich(args):
    """One-shot: pull prices + fundamentals + news for every ticker in
    holdings (watchlist already covered). Use this right after importing a
    new holdings CSV so the analyzer has data for everything."""
    init_db()
    held = holdings_tickers()
    if not held:
        console.print("[yellow]no holdings yet — import a CSV first[/yellow]")
        return
    console.print(f"[bold]Enriching data for {len(held)} held tickers[/bold]")
    prices.ingest(held, period=args.period, workers=6)
    fundamentals_ingest.ingest(held)
    # News is universe-wide already; the ticker tagging will pick up any matches
    if not args.skip_news:
        news_rss.ingest(combined_universe())
    console.print("[green]enrichment complete — re-run `holdings --analyze` to see updated analysis[/green]")


def cmd_holdings(args):
    """Import Groww holdings CSV/Excel and analyze each position."""
    from swingdesk.storage import load_holdings

    overrides = {}
    if args.map:
        for kv in args.map.split(","):
            k, _, v = kv.partition("=")
            if k and v:
                overrides[k.strip()] = v.strip()

    if args.file:
        n = holdings_mod.import_csv(args.file, overrides=overrides, source=args.source)
        console.print(f"[green]imported {n} holdings[/green]")

    df = load_holdings()
    if df.empty:
        console.print("[yellow]no holdings — pass --file to import[/yellow]")
        return

    if args.analyze or args.file:
        console.rule("[bold]Per-holding analysis")
        results = holdings_mod.analyze_portfolio(df, use_ai_thesis=args.ai)
        summary = holdings_mod.portfolio_summary(results)

        color = {"BUY_MORE": "green", "HOLD": "yellow",
                 "REDUCE": "orange3", "SELL": "red", "NO_DATA": "dim"}
        for a in results:
            c = color.get(a.recommendation, "white")
            weight = f"{a.portfolio_weight*100:.1f}%" if a.portfolio_weight else "?"
            pnl = f"{a.pnl_pct:+.1f}%" if a.pnl_pct is not None else "?"
            qs = f"Q={a.quality_score:.0f}" if a.quality_score is not None else "Q=?"
            mfi = f"MFI={a.mfi:.0f}" if a.mfi is not None else ""
            bp = f"buy_pr={a.buy_pressure_20d:.2f}" if a.buy_pressure_20d is not None else ""
            console.print(
                f"  [{c}]{a.recommendation:<9}[/{c}] {a.ticker:>15}  "
                f"weight={weight:<6} P&L={pnl:<7} {qs} "
                f"tech={a.technical_state:<10} {mfi:<8} {bp:<13} "
                f"news +{a.sentiment_bullish}/-{a.sentiment_bearish}"
            )
            for r in a.reasons[:3]:
                console.print(f"      [dim]· {r}[/dim]")
            # Exit plan if computable
            if a.initial_stop or a.full_target:
                line = "      [cyan]exit plan:[/cyan] "
                if a.initial_stop:
                    line += f"hard stop ₹{a.initial_stop} | "
                if a.trailing_stop:
                    line += f"trail ₹{a.trailing_stop} | "
                if a.book_partial_at:
                    line += f"trim 1/3 @ ₹{a.book_partial_at} | "
                if a.full_target:
                    line += f"target ₹{a.full_target}"
                if a.risk_reward:
                    line += f" (R:R={a.risk_reward})"
                console.print(line)
            # AI thesis if available
            if a.ai_narrative:
                console.print(f"      [magenta]AI ({a.ai_conviction}/100, {a.ai_action}):[/magenta] "
                              f"{a.ai_narrative}")
                if a.ai_catalyst:
                    console.print(f"      [dim]· catalyst: {a.ai_catalyst}[/dim]")

        console.rule("[bold]Portfolio summary")
        console.print(f"  holdings:    {summary['n_holdings']}")
        console.print(f"  [green]BUY_MORE: {', '.join(summary['buy_more']) or '—'}[/green]")
        console.print(f"  [yellow]HOLD:     {', '.join(summary['hold']) or '—'}[/yellow]")
        console.print(f"  [orange3]REDUCE:   {', '.join(summary['reduce']) or '—'}[/orange3]")
        console.print(f"  [red]SELL:     {', '.join(summary['sell']) or '—'}[/red]")

        if summary["concentrated_positions"]:
            console.print(f"\n  [red]⚠ Concentrated positions (>25%):[/red] "
                          f"{', '.join(summary['concentrated_positions'])}")
        if summary["concentrated_sectors"]:
            console.print(f"  [red]⚠ Concentrated sectors (>40%):[/red] "
                          f"{', '.join(summary['concentrated_sectors'])}")
    else:
        console.print(df.to_string(index=False))


def cmd_fundamentals(args):
    init_db()
    seed_watchlist_if_empty(DEFAULT_WATCHLIST)
    # --extended fetches fundamentals across the broad ~Nifty-500 universe so
    # company-name -> ticker aliases populate (lights up news tagging for names
    # like VA Tech Wabag / Aegis / Kirloskar). It's a heavy one-time yfinance run.
    universe = combined_universe(include_smallcaps=args.extended,
                                  include_discovery=args.extended,
                                  include_extended=args.extended)
    console.print(f"[bold]Fetching fundamentals for {len(universe)} tickers"
                  f"{' (extended universe)' if args.extended else ' (watchlist + holdings)'}[/bold]")
    fundamentals_ingest.ingest(universe)


def cmd_nse(args):
    """Pull NSE delivery % + bulk/block deals for the universe (manipulation section)."""
    from swingdesk.ingest import bse as bse_ingest
    init_db()
    seed_watchlist_if_empty(DEFAULT_WATCHLIST)
    universe = combined_universe(include_smallcaps=args.extended,
                                  include_discovery=args.extended,
                                  include_extended=args.extended)
    console.print(f"[bold]Fetching NSE delivery ({args.days}d) + deals for "
                  f"{len(universe)} tickers[/bold]")
    if args.all_deals:
        # Capture EVERY bulk/block deal market-wide (not just our universe) so the
        # institutional-flow scanner sees who's buying anything, anywhere.
        nse_ingest.ingest_delivery(universe, days=args.days)
        nse_ingest.ingest_deals(None)
        bse_ingest.ingest_deals(None, days=args.days)
    else:
        nse_ingest.ingest(universe, days=args.days)


def cmd_catalyst(args):
    """News-first scanner: stocks where a catalyst is driving a move."""
    from swingdesk.analyze import catalyst
    init_db()
    hits = catalyst.scan_catalysts(days=args.days, limit=args.limit)
    if not hits:
        console.print("[yellow]No recent catalysts. Run `news` + `sentiment` first.[/yellow]")
        return
    console.rule(f"[bold]News-catalyst scanner — last {args.days}d ({len(hits)} names)")
    arrow = {"bullish": "[green]▲[/green]", "bearish": "[red]▼[/red]", "mixed": "[yellow]◆[/yellow]"}
    for h in hits:
        new = "" if h.in_universe else "  [magenta]🆕 outside watchlist[/magenta]"
        ret = f"{h.ret_pct:+.1f}%" if h.ret_pct is not None else "—"
        rvol = f"{h.rvol:.1f}x" if h.rvol is not None else "—"
        console.print(
            f"  {arrow.get(h.direction, '◆')} {h.strength:>5.1f}  {h.ticker:>14}  "
            f"news={h.bullish}↑/{h.bearish}↓(hi={h.high_impact})  "
            f"{ret:<7} vol={rvol:<6} {h.top_event or ''}{new}"
        )
        if h.top_headline:
            console.print(f"        [dim]{h.top_headline[:110]}[/dim]")


def cmd_institutional(args):
    """Smart-money: bulk/block deal flow + sell-side brokerage calls."""
    from swingdesk.analyze import institutional as inst
    init_db()
    flows = inst.recent_institutional_flow(days=args.days, only_marquee=args.marquee,
                                           limit=args.limit)
    console.rule(f"[bold]Institutional deal flow — last {args.days}d"
                 f"{' (marquee only)' if args.marquee else ''}")
    if not flows:
        console.print("[yellow]No deals stored. Run `nse --all-deals` first.[/yellow]")
    for f in flows:
        side = ("[green]NET BUY[/green]" if f.net_side == "BUY" else
                "[red]NET SELL[/red]" if f.net_side == "SELL" else "FLAT")
        mq = f"  [bold cyan]{' · '.join(f.marquee)}[/bold cyan]" if f.marquee else ""
        cr = f" ₹{f.net_value / 1e7:,.1f}cr" if f.net_value else ""
        console.print(f"  {f.ticker:>14}  {side}{cr}  deals={f.n_deals}{mq}")
        if f.exchanges:
            console.print(f"        [dim]exchanges: {' / '.join(f.exchanges)}[/dim]")
        for name, cside in f.clients[:4]:
            console.print(f"        [dim]{cside:<4} {name[:60]}[/dim]")

    investors = inst.investor_activity(days=args.days, limit=args.limit)
    if investors:
        console.rule("[bold]Top named investors on the disclosed tape")
        for inv in investors[: min(10, len(investors))]:
            side = ("[green]NET BUY[/green]" if inv.net_side == "BUY" else
                    "[red]NET SELL[/red]" if inv.net_side == "SELL" else "FLAT")
            mq = f"  [bold cyan]{inv.marquee}[/bold cyan]" if inv.marquee else ""
            tickers = ", ".join(inv.tickers[:3]) + ("…" if len(inv.tickers) > 3 else "")
            console.print(f"  {inv.client[:40]:<40} {side}  ₹{abs(inv.net_value) / 1e7:,.1f}cr{mq}")
            console.print(f"        [dim]stocks: {tickers or '—'} · exchanges: {' / '.join(inv.exchanges) or '—'}[/dim]")

    actions = inst.brokerage_actions(days=args.days, limit=args.limit)
    if actions:
        console.rule("[bold]Brokerage calls (sell-side desks)")
        for a in actions:
            color = "green" if a.sentiment == "bullish" else "red"
            console.print(f"  [{color}]{a.broker}[/{color}]  {a.action}  "
                          f"{a.ticker or '—':>12}  [dim]{a.headline[:90]}[/dim]")


def cmd_screen(args):
    """Rank watchlist by fundamental quality. Optionally filter or replace the watchlist."""
    from swingdesk.analyze import quality as quality_mod
    from swingdesk.storage import load_fundamentals

    df = load_fundamentals()
    if df.empty:
        console.print("[yellow]no fundamentals yet — run `swingdesk fundamentals` first[/yellow]")
        return
    # Apply hard quality filter
    keep = []
    for _, r in df.iterrows():
        ok, fails = quality_mod.passes_quality_bar(r.to_dict(), min_score=args.min_score)
        if ok:
            keep.append(r["ticker"])

    cols = ["ticker", "short_name", "sector", "quality_score",
            "return_on_equity", "trailing_pe", "debt_to_equity",
            "profit_margin", "earnings_growth", "revenue_growth"]
    cols = [c for c in cols if c in df.columns]
    df_sorted = df[cols].sort_values("quality_score", ascending=False, na_position="last")
    console.rule(f"[bold]Quality ranking — passing bar (≥{args.min_score}): {len(keep)} of {len(df)}")
    console.print(df_sorted.to_string(index=False))

    if args.apply:
        from swingdesk.storage import set_watchlist
        set_watchlist(keep)
        console.print(f"\n[green]watchlist replaced with {len(keep)} quality-screened names[/green]")


def cmd_earnings(args):
    init_db()
    seed_watchlist_if_empty(DEFAULT_WATCHLIST)
    wl = get_watchlist()
    console.print(f"[bold]Fetching earnings dates for {len(wl)} tickers[/bold]")
    earnings_ingest.ingest(wl)


def cmd_optimize(args):
    from swingdesk.backtest.optimizer import optimize, ParamGrid
    init_db()
    wl = [args.ticker] if args.ticker else get_watchlist()
    results = optimize(wl, args.setup, ParamGrid())
    if results.empty:
        console.print("[yellow]no results — insufficient data[/yellow]")
        return
    console.rule(f"[bold]Top 10 parameter combos — {args.setup}")
    console.print(results.head(10).to_string(index=False))
    console.rule("[bold]Bottom 5 (for contrast)")
    console.print(results.tail(5).to_string(index=False))


def cmd_reconcile(args):
    df = reconcile_mod.reconcile(latest_run=not args.all_runs,
                                 min_paper_trades=args.min_trades)
    if df.empty:
        console.print("[yellow]no data to reconcile yet[/yellow]")
        return
    console.rule("[bold]Paper vs Backtest drift")
    console.print(df.to_string(index=False))
    flagged = df[df["verdict"] == "underperforming"]
    if not flagged.empty:
        console.print(f"\n[red]⚠ {len(flagged)} setup(s) underperforming backtest — investigate[/red]")


def cmd_import(args):
    overrides = {}
    if args.map:
        for kv in args.map.split(","):
            k, _, v = kv.partition("=")
            if k and v:
                overrides[k.strip()] = v.strip()
    res = import_groww.import_trades(args.csv, overrides=overrides, is_paper=args.paper)
    console.print(f"[green]import: {res}[/green]")


def cmd_signals(args):
    df = load_signals(limit=args.limit)
    if df.empty:
        console.print("[yellow]no signals yet — run `swingdesk scan` first[/yellow]")
        return
    console.print(df.to_string(index=False))


def cmd_watchlist(args):
    if args.set:
        tickers = [t.strip().upper() for t in args.set.split(",") if t.strip()]
        set_watchlist(tickers)
        console.print(f"[green]watchlist set to {len(tickers)} tickers[/green]")
    else:
        wl = get_watchlist()
        console.print("\n".join(wl) if wl else "[yellow](empty)[/yellow]")


def cmd_global_news(args):
    init_db()
    n = global_news_ingest.ingest(per_feed=args.per_feed)
    console.print(f"[green]{n} new global impact item(s)[/green]")

    from swingdesk.analyze import global_impact
    df = global_impact.recent_impacts(days=args.days, limit=args.limit)
    if df.empty:
        console.print("[yellow]no mapped global impacts yet[/yellow]")
        return
    cols = ["published", "source", "cue", "direction", "impact_score",
            "sectors", "tickers", "title"]
    cols = [c for c in cols if c in df.columns]
    console.print(df[cols].to_string(index=False))


def cmd_sudden_radar(args):
    from swingdesk.analyze import sudden_move
    init_db()
    universe = combined_universe(include_smallcaps=args.include_smallcaps,
                                  include_discovery=args.include_discovery)
    df = sudden_move.scan(universe, include_intraday=args.intraday, limit=args.limit,
                          include_smallcaps=False, include_discovery=False)
    if df.empty:
        console.print("[yellow]no radar rows — fetch prices/fundamentals first[/yellow]")
        return
    cols = ["ticker", "radar_score", "readiness", "last", "compression",
            "accumulation", "prebreakout", "relative_strength", "anomaly_score",
            "anomaly_volume_z", "anomaly_range_z", "anomaly_turnover_z", "catalyst",
            "intraday_confirm", "manip_penalty", "reasons", "risks"]
    console.rule("[bold]Sudden Move Radar — setup pressure, not a guarantee")
    console.print(df[cols].to_string(index=False))


def cmd_sudden_backtest(args):
    from swingdesk.analyze import sudden_move
    init_db()
    tickers = [args.ticker] if args.ticker else combined_universe(
        include_smallcaps=args.include_smallcaps,
        include_discovery=args.include_discovery)
    trades, summary = sudden_move.backtest(
        tickers, min_score=args.min_score,
        move_threshold_pct=args.move_threshold,
        horizon=args.horizon)
    console.rule("[bold]Sudden Move Radar backtest")
    console.print(summary)
    if trades.empty:
        console.print("[yellow]no qualifying historical setups[/yellow]")
        return
    console.print(
        trades.sort_values("radar_score", ascending=False).head(args.limit).to_string(index=False)
    )


def cmd_market_flow(args):
    from swingdesk.analyze import market_flow
    init_db()
    state = market_flow.current_flow()
    console.rule("[bold]Market flow")
    console.print({
        "flow": state.flow,
        "score": state.score,
        "nifty_trend": state.nifty_trend,
        "volatility": state.volatility,
        "reason": state.reason,
    })
    if args.backtest:
        stats, _ = market_flow.strategy_flow_backtest(
            combined_universe(include_smallcaps=args.include_smallcaps),
            max_hold=args.max_hold, min_trades=args.min_trades)
        if stats.empty:
            console.print("[yellow]no strategy-flow stats — run macro + prices first[/yellow]")
        else:
            console.rule("[bold]Strategy edge by market flow")
            console.print(stats.to_string(index=False))


def cmd_confluence(args):
    from swingdesk.analyze import market_flow
    init_db()
    universe = combined_universe(include_smallcaps=args.include_smallcaps,
                                  include_discovery=args.include_discovery)
    df = market_flow.confluence_board(universe, limit=args.limit,
                                      include_intraday=args.intraday,
                                      target_move_pct=args.target_move)
    if df.empty:
        console.print("[yellow]no confluence rows — run prices/fundamentals/macro first[/yellow]")
        return
    console.rule("[bold]Confluence Board — market-flow aware ranking")
    console.print(df.to_string(index=False))


def cmd_explosive_backtest(args):
    from swingdesk.analyze import market_flow
    init_db()
    universe = [args.ticker] if args.ticker else combined_universe(
        include_smallcaps=args.include_smallcaps,
        include_discovery=args.include_discovery)
    trades, summary = market_flow.explosive_move_backtest(
        universe, target_move_pct=args.target_move,
        min_explosive_score=args.min_score)
    console.rule("[bold]Explosive move backtest")
    console.print(summary)
    if not trades.empty:
        console.print(trades.sort_values("explosive_score", ascending=False).head(args.limit).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swingdesk", description="Local swing-trading signal app")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="Initialize database and watchlist")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("prices", help="Fetch OHLCV for watchlist")
    p.add_argument("--period", default="2y")
    p.add_argument("--workers", type=int, default=6)
    p.set_defaults(func=cmd_prices)

    p = sub.add_parser("news", help="Fetch RSS news and tag tickers")
    p.add_argument("--include-smallcaps", action="store_true", default=True,
                   help="Tag against the small-cap universe too (default: True)")
    p.add_argument("--include-discovery", action="store_true", default=True,
                   help="Tag against the discovery universe too (default: True)")
    p.add_argument("--retag", action="store_true",
                   help="Also re-tag previously stored news with the current universe/aliases")
    p.set_defaults(func=cmd_news)

    p = sub.add_parser("scan", help="Run technical scanners + sentiment scoring, persist signals")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("sentiment", help="Run Claude sentiment on unanalyzed news")
    p.add_argument("--max", type=int, default=200, help="Max headlines to analyze")
    p.set_defaults(func=cmd_sentiment)

    p = sub.add_parser("notify", help="Push the latest signals to Telegram")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_notify)

    p = sub.add_parser("schedule", help="Run the daemon (pre-market + post-close jobs)")
    p.add_argument("--once", action="store_true", help="Run full pipeline once and exit")
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("backtest", help="Walk-forward backtest of all setups over price history")
    p.add_argument("--ticker", help="Backtest a single ticker only")
    p.add_argument("--max-hold", type=int, default=20, help="Time-stop in bars (default 20)")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("run", help="prices + news + sentiment + scan + paper-trade + notify (the daily run)")
    p.add_argument("--period", default="2y")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--min-score", type=float, default=75.0,
                   help="Min composite score to auto paper-trade (default 75 — be picky)")
    p.add_argument("--skip-earnings", action="store_true",
                   help="Skip refreshing the earnings calendar (faster)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("paper-step",
                       help="One step of the paper autotrader (exit → enter, exec-cost fills, kill-switch)")
    p.add_argument("--min-score", type=float, default=60.0,
                   help="Min signal score to enter (default 60)")
    p.add_argument("--algo", default="vwap", choices=["vwap", "twap", "pov", "is"],
                   help="Execution algo used to price entry fills (default vwap)")
    p.add_argument("--max-heat", type=float, default=6.0,
                   help="Max portfolio heat %% (total open risk vs capital, default 6)")
    p.add_argument("--kill-dd", type=float, default=10.0,
                   help="Halt new entries at this equity drawdown %% (default 10)")
    p.add_argument("--flatten-on-kill", action="store_true",
                   help="Close all positions when the kill-switch trips")
    p.add_argument("--refresh", action="store_true",
                   help="Re-download fresh prices before evaluating")
    p.set_defaults(func=cmd_paper_step)

    # ---- portfolio commands ----
    p = sub.add_parser("positions", help="List positions (paper by default)")
    p.add_argument("--status", default="all", choices=["all", "open", "closed"])
    p.add_argument("--real", action="store_true", help="Show real positions (default: paper)")
    p.add_argument("--both", action="store_true", help="Show both paper and real")
    p.set_defaults(func=cmd_positions)

    p = sub.add_parser("open", help="Manually open a position")
    p.add_argument("ticker")
    p.add_argument("--entry", required=True, type=float)
    p.add_argument("--stoploss", required=True, type=float)
    p.add_argument("--target", required=True, type=float)
    p.add_argument("--qty", type=int, help="Override auto-sizing")
    p.add_argument("--setup", default="manual")
    p.add_argument("--real", action="store_true", help="Mark as a REAL trade (default: paper)")
    p.add_argument("--notes", default=None)
    p.set_defaults(func=cmd_open)

    p = sub.add_parser("close", help="Close a position by id")
    p.add_argument("id", type=int)
    p.add_argument("--price", required=True, type=float)
    p.add_argument("--reason", default="manual",
                   choices=["manual", "target", "stoploss", "time_stop"])
    p.set_defaults(func=cmd_close)

    p = sub.add_parser("mark", help="Mark-to-market: update open positions with latest prices")
    p.add_argument("--refresh", action="store_true",
                   help="Re-download fresh prices before marking (vs use cached data)")
    p.set_defaults(func=cmd_mark)

    p = sub.add_parser("sync", help="Pull fresh prices for all holdings/positions + mark-to-market (one shot)")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("journal", help="Portfolio stats and per-setup breakdown")
    p.add_argument("--real", action="store_true", help="Only real positions (default: paper)")
    p.add_argument("--both", action="store_true")
    p.set_defaults(func=cmd_journal)

    p = sub.add_parser("import", help="Import broker trade CSV (Groww and similar)")
    p.add_argument("csv", help="Path to CSV file")
    p.add_argument("--map", help='Column overrides, e.g. "symbol=Stock,qty=Quantity"')
    p.add_argument("--paper", action="store_true",
                   help="Mark imported trades as paper (default: real, since they came from a broker)")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("holdings", help="Import Groww holdings + analyze (BUY/HOLD/SELL per position)")
    p.add_argument("--file", help="Path to holdings CSV/XLSX (skip to use last imported)")
    p.add_argument("--map", help='Column overrides, e.g. "symbol=Stock,avg_price=Avg"')
    p.add_argument("--source", default="groww", help="Source label (default: groww)")
    p.add_argument("--analyze", action="store_true", help="Force re-analyze even if no new file")
    p.add_argument("--ai", action="store_true",
                   help="Include Claude AI thesis for each holding (uses API)")
    p.set_defaults(func=cmd_holdings)

    p = sub.add_parser("enrich", help="Fetch prices+fundamentals+news for holdings (run after importing)")
    p.add_argument("--period", default="1y")
    p.add_argument("--skip-news", action="store_true")
    p.set_defaults(func=cmd_enrich)

    p = sub.add_parser("discover", help="Find new stocks to invest in (outside watchlist + holdings)")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--min-quality", type=float, default=60)
    p.add_argument("--active-only", action="store_true",
                   help="Only show stocks with a setup firing today")
    p.add_argument("--conviction-only", action="store_true",
                   help="Show only HIGH conviction names in the ranked list")
    p.add_argument("--no-conviction", action="store_true",
                   help="Suppress the high-conviction summary panel at the top")
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("warnings", help="Early-exit warnings on your holdings (trend-break / volume divergence / bearish news / sector weak / VIX spike)")
    p.set_defaults(func=cmd_warnings)

    p = sub.add_parser("sc-watchlist", help="Manage the small-cap watchlist (separate from main)")
    p.add_argument("--add", help="Comma-separated tickers to ADD to the small-cap WL")
    p.add_argument("--remove", help="Comma-separated tickers to REMOVE")
    p.add_argument("--set", help="REPLACE the small-cap WL with these tickers")
    p.add_argument("--clear", action="store_true", help="Empty the small-cap WL")
    p.set_defaults(func=cmd_sc_watchlist)

    p = sub.add_parser("sc-scan", help="Scan the small-cap watchlist (tags signals with universe='smallcap')")
    p.set_defaults(func=cmd_sc_scan)

    p = sub.add_parser("smallcaps", help="Discovery scanner over a curated ~95-name small-cap universe")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--min-quality", type=float, default=55,
                   help="Lower default than mid/large-cap discover — smallcaps with Q≥55 are still solid")
    p.add_argument("--active-only", action="store_true")
    p.add_argument("--conviction-only", action="store_true")
    p.add_argument("--no-conviction", action="store_true")
    p.set_defaults(func=cmd_smallcaps)

    p = sub.add_parser("macro", help="Refresh macro indicators (Nifty, USD/INR, Crude, S&P 500)")
    p.add_argument("--period", default="1y")
    p.set_defaults(func=lambda a: (init_db(), macro_ingest.ingest(period=a.period)))

    # ---- Week 5 commands ----
    p = sub.add_parser("fundamentals", help="Fetch fundamental ratios per ticker (yfinance)")
    p.add_argument("--extended", action="store_true",
                   help="Fetch across the broad ~Nifty-500 universe (populates name->ticker aliases)")
    p.set_defaults(func=cmd_fundamentals)

    p = sub.add_parser("nse", help="Fetch NSE delivery %% + bulk/block deals (manipulation section)")
    p.add_argument("--extended", action="store_true",
                   help="Cover the broad ~Nifty-500 universe")
    p.add_argument("--all-deals", action="store_true",
                   help="Capture every bulk/block deal market-wide (for the institutional-flow scanner)")
    p.add_argument("--days", type=int, default=20,
                   help="Trading days of delivery history to backfill (default 20)")
    p.set_defaults(func=cmd_nse)

    p = sub.add_parser("catalyst", help="News-first scanner: stocks where a catalyst is driving a move")
    p.add_argument("--days", type=int, default=3, help="Lookback window in days")
    p.add_argument("--limit", type=int, default=25, help="Max names to show")
    p.set_defaults(func=cmd_catalyst)

    p = sub.add_parser("institutional", help="Smart-money: bulk/block deal flow + brokerage calls")
    p.add_argument("--days", type=int, default=30, help="Lookback window in days")
    p.add_argument("--limit", type=int, default=30, help="Max names to show")
    p.add_argument("--marquee", action="store_true",
                   help="Only stocks with a marquee institution (BlackRock/MS/JPM/…) on the tape")
    p.set_defaults(func=cmd_institutional)

    p = sub.add_parser("global-news", help="Fetch global market news and map Indian impact")
    p.add_argument("--per-feed", type=int, default=40, help="Max RSS items per feed")
    p.add_argument("--days", type=int, default=3, help="Show mapped impacts from last N days")
    p.add_argument("--limit", type=int, default=30, help="Rows to print")
    p.set_defaults(func=cmd_global_news)

    p = sub.add_parser("sudden-radar", help="Rank stocks with pressure for sudden upside moves")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--intraday", action="store_true",
                   help="Use stored intraday bars for ORB/VWAP/RVOL confirmation")
    p.add_argument("--include-smallcaps", action="store_true", default=True)
    p.add_argument("--include-discovery", action="store_true")
    p.set_defaults(func=cmd_sudden_radar)

    p = sub.add_parser("sudden-backtest", help="Backtest sudden-move radar hit rates")
    p.add_argument("--ticker", help="Backtest one ticker only")
    p.add_argument("--min-score", type=float, default=70.0)
    p.add_argument("--move-threshold", type=float, default=3.0,
                   help="Next-horizon high/close move counted as a hit")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--include-smallcaps", action="store_true", default=True)
    p.add_argument("--include-discovery", action="store_true")
    p.set_defaults(func=cmd_sudden_backtest)

    p = sub.add_parser("market-flow", help="Show current market flow and strategy edge by flow")
    p.add_argument("--backtest", action="store_true",
                   help="Backtest setup families grouped by historical market flow")
    p.add_argument("--max-hold", type=int, default=20)
    p.add_argument("--min-trades", type=int, default=5)
    p.add_argument("--include-smallcaps", action="store_true")
    p.set_defaults(func=cmd_market_flow)

    p = sub.add_parser("confluence", help="Fuse all tabs into one market-flow-aware board")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--intraday", action="store_true")
    p.add_argument("--target-move", type=float, default=10.0,
                   help="One-day intraday move target to rank for (e.g. 10 or 15)")
    p.add_argument("--include-smallcaps", action="store_true", default=True)
    p.add_argument("--include-discovery", action="store_true")
    p.set_defaults(func=cmd_confluence)

    p = sub.add_parser("explosive-backtest", help="Backtest +10/+15%% one-day explosive filter")
    p.add_argument("--ticker", help="Backtest one ticker only")
    p.add_argument("--target-move", type=float, default=10.0)
    p.add_argument("--min-score", type=float, default=70.0)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--include-smallcaps", action="store_true", default=True)
    p.add_argument("--include-discovery", action="store_true")
    p.set_defaults(func=cmd_explosive_backtest)

    p = sub.add_parser("screen", help="Rank watchlist by fundamental quality")
    p.add_argument("--min-score", type=float, default=60.0,
                   help="Composite quality threshold (default 60)")
    p.add_argument("--apply", action="store_true",
                   help="Replace watchlist with the names that pass")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("earnings", help="Refresh upcoming-earnings calendar for the watchlist")
    p.set_defaults(func=cmd_earnings)

    p = sub.add_parser("optimize", help="Grid-search SL/target/hold parameters for a setup")
    p.add_argument("--setup", required=True,
                   choices=["breakout_20d", "pullback_ema20", "volume_thrust", "ema_20_50_cross"])
    p.add_argument("--ticker", help="Single ticker (otherwise full watchlist)")
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("reconcile", help="Compare live paper-trade stats vs backtest expectation")
    p.add_argument("--min-trades", type=int, default=5,
                   help="Min paper trades to draw a verdict (default 5)")
    p.add_argument("--all-runs", action="store_true",
                   help="Aggregate all backtest runs (default: latest only)")
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("signals", help="Show recent signals")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_signals)

    p = sub.add_parser("watchlist", help="View or set watchlist")
    p.add_argument("--set", help="Comma-separated tickers (e.g. RELIANCE.NS,TCS.NS)")
    p.set_defaults(func=cmd_watchlist)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
