"""Versioned observation contracts. No engine models are serialized directly."""
from datetime import date
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictFloat, model_validator

Identifier = Annotated[str, Field(strict=True, min_length=1, max_length=128,
                                 pattern=r"^[A-Za-z0-9_.:-]+$")]
MetricName = Literal["close", "range_20_observations_pct", "volume_multiple", "average_daily_value_cr"]
Unit = Literal["INR", "percent", "multiple", "INR_crore"]
UNITS: dict[str, str] = {
    "close": "INR", "range_20_observations_pct": "percent",
    "volume_multiple": "multiple", "average_daily_value_cr": "INR_crore",
}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False,
                              validate_default=True, revalidate_instances="always")


class Instrument(Contract):
    instrument_id: Identifier
    symbol: Annotated[str, Field(strict=True, min_length=1, max_length=40,
                                pattern=r"^[A-Z0-9&^_.=-]+$")]
    exchange: Literal["NSE", "BSE"]


class Provenance(Contract):
    batch_id: Identifier
    source_id: Identifier
    formula_version: Literal["legacy-radar-observations-v1"] = "legacy-radar-observations-v1"
    session_date: date
    observed_at: AwareDatetime
    available_at: AwareDatetime
    coverage: Literal["complete", "partial", "unknown"]
    freshness: Literal["current", "stale", "unknown"]
    adjustment_basis: Literal["raw", "adjusted", "unknown"]

    @model_validator(mode="after")
    def chronological(self):
        if self.available_at < self.observed_at:
            raise ValueError("available_at cannot precede observed_at")
        if self.session_date > self.observed_at.date():
            raise ValueError("session_date cannot follow observed_at")
        return self


class Observation(Contract):
    metric: MetricName
    value: StrictFloat | None
    unit: Unit
    status: Literal["available", "missing"]
    missing_reason: Literal["not_supplied", "non_finite"] | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.unit != UNITS[self.metric]:
            raise ValueError("Unit does not match metric")
        if self.status == "available":
            if self.value is None or self.missing_reason is not None:
                raise ValueError("Available observations need a value and no missing reason")
            if self.value < 0 or (self.metric == "close" and self.value == 0):
                raise ValueError("Invalid non-positive price or negative magnitude")
        elif self.value is not None or self.missing_reason is None:
            raise ValueError("Missing observations need a null value and a reason")
        return self


class ScannerEvidence(Contract):
    schema_version: Literal["1.0"] = "1.0"
    instrument: Instrument
    provenance: Provenance
    observations: tuple[Observation, ...] = Field(min_length=4, max_length=4)
    status: Literal["ready", "limited"]

    @model_validator(mode="after")
    def complete_contract(self):
        if {item.metric for item in self.observations} != set(UNITS):
            raise ValueError("Every allowlisted metric must appear exactly once")
        ready = (self.provenance.coverage == "complete"
                 and self.provenance.freshness == "current"
                 and self.provenance.adjustment_basis != "unknown"
                 and all(item.status == "available" for item in self.observations))
        if self.status != ("ready" if ready else "limited"):
            raise ValueError("Evidence status must reflect coverage and missing data")
        return self
