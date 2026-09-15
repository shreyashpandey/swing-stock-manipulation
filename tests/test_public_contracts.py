import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.swingdesk_api.contracts import Instrument, Provenance, ScannerEvidence
from backend.swingdesk_api.projection import project_radar_observations


@pytest.fixture
def context():
    return {
        "instrument": Instrument(instrument_id="synthetic-1", symbol="TEST.NS", exchange="NSE"),
        "provenance": Provenance(
            batch_id="synthetic-batch-1", source_id="synthetic",
            session_date="2026-09-11", observed_at="2026-09-11T15:30:00+05:30",
            available_at="2026-09-11T16:00:00+05:30", coverage="complete",
            freshness="current", adjustment_basis="raw"),
    }


@pytest.fixture
def row():
    return dict(ticker="TEST.NS", last=100., range_20d_pct=5.,
                volume_mult=1.5, adv_value_cr=10.)


def test_projection_excludes_internal_and_prescriptive_fields(context, row):
    row.update(action="BUY", allocation_amount=50000, target=150,
               reasons=["Buy now"], radar_score=99, manip_penalty=10)
    result = project_radar_observations(row, **context)
    payload = result.model_dump_json()
    assert result.status == "ready"
    assert [o.value for o in result.observations] == [100, 5, 1.5, 10]
    for forbidden in ("BUY", "allocation_amount", "target", "reasons", "radar_score", "manip_penalty"):
        assert forbidden not in payload
    assert ScannerEvidence.model_validate_json(payload) == result


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_missing_values_remain_explicit(context, row, value):
    row["volume_mult"] = value
    result = project_radar_observations(row, **context)
    assert result.status == "limited"
    metric = result.observations[2]
    assert metric.value is None and metric.status == "missing"
    assert metric.missing_reason == ("not_supplied" if value is None else "non_finite")
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


@pytest.mark.parametrize("value", [True, "100", -10, 0])
def test_invalid_prices_fail(context, row, value):
    row["last"] = value
    with pytest.raises(ValueError):
        project_radar_observations(row, **context)


@pytest.mark.parametrize("field,value", [("coverage", "unknown"), ("coverage", "partial"),
                                         ("freshness", "stale"), ("adjustment_basis", "unknown")])
def test_incomplete_provenance_cannot_be_ready(context, row, field, value):
    context["provenance"] = Provenance.model_validate({
        **context["provenance"].model_dump(), field: value})
    result = project_radar_observations(row, **context)
    assert result.status == "limited"
    with pytest.raises(ValidationError):
        ScannerEvidence.model_validate({**result.model_dump(), "status": "ready"})


def test_mismatch_and_unreviewed_fields_rejected(context, row):
    with pytest.raises(ValueError, match="symbol"):
        project_radar_observations({**row, "ticker": "OTHER.NS"}, **context)
    result = project_radar_observations(row, **context).model_dump()
    with pytest.raises(ValidationError):
        ScannerEvidence.model_validate({**result, "recommendation": "BUY"})
    result["observations"][0]["unit"] = "percent"
    with pytest.raises(ValidationError):
        ScannerEvidence.model_validate(result)


@pytest.mark.parametrize("field,value", [
    ("observed_at", "2026-09-11T15:30:00"),
    ("available_at", "2026-09-10T16:00:00+05:30"),
    ("session_date", "2026-09-12"),
])
def test_invalid_timestamps_rejected(context, field, value):
    with pytest.raises(ValidationError):
        Provenance.model_validate({**context["provenance"].model_dump(), field: value})


def test_schema_disallows_extra_fields():
    schema = ScannerEvidence.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Observation"]["additionalProperties"] is False
    artifact = Path(__file__).resolve().parents[1] / "backend/swingdesk_api/scanner-evidence-v1.schema.json"
    assert json.loads(artifact.read_text()) == schema
