from __future__ import annotations

from datetime import date

from swingdesk.analyze import market_calendar as mc


def test_build_calendar_sorted_and_nonempty():
    ev = mc.build_calendar(2026)
    assert len(ev) > 30
    starts = [e.start for e in ev]
    assert starts == sorted(starts)  # chronological


def test_budget_is_feb_first():
    ev = mc.build_calendar(2026)
    budget = [e for e in ev if e.name == "Union Budget"]
    assert len(budget) == 1
    assert budget[0].start == date(2026, 2, 1)
    assert budget[0].bias == mc.VOLATILE


def test_monthly_expiry_is_last_thursday():
    ev = mc.build_calendar(2026)
    expiries = [e for e in ev if e.category == mc.CAT_EXPIRY]
    assert len(expiries) == 12
    for e in expiries:
        assert e.start.weekday() == 3  # Thursday
        # No later Thursday exists in that month.
        assert (e.start.replace(day=1).month == e.start.month)
        nxt = date(e.start.year, e.start.month, min(e.start.day + 7, 28))
        assert not (nxt.month == e.start.month and nxt.weekday() == 3 and nxt.day > e.start.day)


def test_fixed_holidays_present():
    ev = mc.build_calendar(2026)
    holidays = {e.start for e in ev if e.category == mc.CAT_HOLIDAY}
    assert date(2026, 1, 26) in holidays   # Republic Day
    assert date(2026, 8, 15) in holidays   # Independence Day
    assert date(2026, 12, 25) in holidays  # Christmas


def test_biases_are_valid_buckets():
    valid = {mc.BULLISH, mc.BEARISH, mc.VOLATILE, mc.NEUTRAL}
    assert all(e.bias in valid for e in mc.build_calendar(2026))


def test_curated_events_flagged_approximate():
    ev = mc.build_calendar(2026)
    muhurat = [e for e in ev if "Muhurat" in e.name]
    assert muhurat and muhurat[0].approximate is True
    assert muhurat[0].bias == mc.BULLISH


def test_uncurated_year_has_only_deterministic():
    assert mc.has_curated(2026) is True
    assert mc.has_curated(2099) is False
    ev = mc.build_calendar(2099)
    assert all(not e.approximate for e in ev)          # no floating events
    assert all(e.category != mc.CAT_FESTIVE for e in ev)


def test_events_carry_sector_tilts():
    ev = mc.build_calendar(2026)
    festive = [e for e in ev if e.name == "Festive-consumption season"][0]
    assert mc.SEC_CONS_CYC in festive.sectors
    # A broad event has no specific sector tilt.
    santa = [e for e in ev if e.name == "Santa Claus rally"][0]
    assert santa.sectors == []


def test_upcoming_bullish_filters_to_bullish_only():
    today = date(2026, 9, 20)
    bull = mc.upcoming_bullish(2026, today, within_days=40)
    assert bull, "expected the festive-consumption window to show up"
    assert all(e.bias == mc.BULLISH for e in bull)


def test_upcoming_window():
    today = date(2026, 1, 25)
    up = mc.upcoming(2026, today, within_days=15)
    # Budget (Feb 1) and the Economic Survey (Jan 31) fall in the next 15 days.
    names = {e.name for e in up}
    assert "Union Budget" in names
    assert all(today <= e.start <= date(2026, 2, 9) for e in up)
