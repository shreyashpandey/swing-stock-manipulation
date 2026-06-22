"""CRUD round-trip for the `plans` table (intended ₹ per ticker)."""
from __future__ import annotations

from swingdesk.storage import delete_plan, load_plans, update_plan, upsert_plan


def test_plan_crud_roundtrip(tmp_db):
    pid = upsert_plan({"ticker": "RELIANCE.NS", "planned_amount": 50000,
                       "target_price": 2800, "note": "on dip", "status": "watching"})
    assert pid > 0
    df = load_plans()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "RELIANCE.NS"
    assert row["planned_amount"] == 50000
    assert row["status"] == "watching"


def test_upsert_updates_active_plan_in_place(tmp_db):
    upsert_plan({"ticker": "TCS.NS", "planned_amount": 20000})
    upsert_plan({"ticker": "TCS.NS", "planned_amount": 35000, "status": "idea"})
    df = load_plans()
    assert len(df) == 1                      # one active plan per ticker
    assert df.iloc[0]["planned_amount"] == 35000


def test_dropped_plan_does_not_block_a_new_one(tmp_db):
    pid = upsert_plan({"ticker": "INFY.NS", "planned_amount": 10000})
    update_plan(pid, status="dropped")
    upsert_plan({"ticker": "INFY.NS", "planned_amount": 25000})   # fresh plan
    active = load_plans()
    active = active[active["status"] != "dropped"]
    assert len(active) == 1
    assert active.iloc[0]["planned_amount"] == 25000


def test_status_filter_and_delete(tmp_db):
    upsert_plan({"ticker": "SBIN.NS", "planned_amount": 15000, "status": "entered"})
    pid = upsert_plan({"ticker": "ITC.NS", "planned_amount": 8000, "status": "idea"})
    assert len(load_plans(status="entered")) == 1
    assert len(load_plans(status="idea")) == 1
    delete_plan(pid)
    assert len(load_plans(status="idea")) == 0
