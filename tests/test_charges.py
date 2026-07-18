"""Indian charge model + capital-gains classification."""
from __future__ import annotations

from swingdesk.analyze import charges as c


def test_delivery_round_trip_itemisation():
    buy = c.leg_charges("buy", 500, 100)        # ₹50,000 turnover
    sell = c.leg_charges("sell", 550, 100)      # ₹55,000 turnover
    # Brokerage capped at ₹20 (0.1% would be ₹50/₹55).
    assert buy["brokerage"] == 20.0 and sell["brokerage"] == 20.0
    # STT 0.1% on BOTH delivery legs.
    assert buy["stt"] == 50.0 and sell["stt"] == 55.0
    # Stamp duty only on the buy; DP charge only on the (delivery) sell.
    assert buy["stamp_duty"] > 0 and sell["stamp_duty"] == 0.0
    assert buy["dp_charge"] == 0.0 and sell["dp_charge"] > 0
    # Round trip == sum of legs.
    rt = c.round_trip_charges(500, 550, 100)
    assert rt["total"] == round(buy["total"] + sell["total"], 2)
    # Sanity: a ₹50k→₹55k delivery round trip costs roughly 0.3–0.4% of turnover.
    assert 150 < rt["total"] < 250


def test_brokerage_is_lower_of_cap_and_pct():
    # Tiny order: 0.1% of ₹1,000 = ₹1 < ₹20 cap.
    small = c.leg_charges("buy", 10, 100)
    assert small["brokerage"] == 1.0
    # Large order: 0.1% of ₹50,000 = ₹50 → capped at ₹20.
    big = c.leg_charges("buy", 500, 100)
    assert big["brokerage"] == 20.0


def test_intraday_has_no_dp_and_sell_only_stt():
    sell = c.leg_charges("sell", 550, 100, segment="intraday")
    assert sell["dp_charge"] == 0.0
    assert sell["stt"] == round(55000 * 0.025 / 100, 2)   # 0.025% intraday
    buy = c.leg_charges("buy", 500, 100, segment="intraday")
    assert buy["stt"] == 0.0                               # intraday: no STT on buy


def test_exit_charges_are_sell_side_only():
    ex = c.exit_charges(550, 100)
    sell = c.leg_charges("sell", 550, 100)
    assert ex == sell
    assert ex["stamp_duty"] == 0.0 and ex["dp_charge"] > 0


def test_zero_turnover_is_all_zero():
    z = c.leg_charges("buy", 0, 100)
    assert all(v == 0.0 for v in z.values())


def test_gain_classification_boundary_at_365_days():
    stcg = c.classify_gain("2024-01-01", "2024-12-31", 5000)   # 364 days
    ltcg = c.classify_gain("2024-01-01", "2025-01-01", 5000)   # 366 days
    assert stcg["gain_type"] == "STCG" and stcg["tax_rate_pct"] == 20.0
    assert ltcg["gain_type"] == "LTCG" and ltcg["tax_rate_pct"] == 12.5
    assert ltcg["tax_est"] == round(5000 * 12.5 / 100, 2)
    # A loss is never taxed.
    assert c.classify_gain("2024-01-01", "2024-03-01", -3000)["tax_est"] == 0.0
