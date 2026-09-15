"""Pure projection of frozen engine rows. No database, network or recomputation."""
import math
from collections.abc import Mapping
from numbers import Real

from .contracts import Instrument, Observation, Provenance, ScannerEvidence, UNITS

# Adding a new internal field never adds it to the customer contract automatically.
ENGINE_FIELDS = {
    "close": "last",
    "range_20_observations_pct": "range_20d_pct",
    "volume_multiple": "volume_mult",
    "average_daily_value_cr": "adv_value_cr",
}


def project_radar_observations(row: Mapping, *, instrument: Instrument,
                               provenance: Provenance) -> ScannerEvidence:
    """Caller supplies explicit source/batch context; missing metadata is not guessed.

    This adapter is an internal preview until batch provenance and the actual
    output scope have been reviewed. It does not certify source rights.
    """
    if row.get("ticker") != instrument.symbol:
        raise ValueError("Engine row and instrument symbol do not match")
    observations = []
    for metric, field in ENGINE_FIELDS.items():
        value = row.get(field)
        reason = None
        if value is None:
            reason = "not_supplied"
        elif isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field} must be a numeric observation")
        elif not math.isfinite(value):
            reason = "non_finite"
        observations.append(Observation(
            metric=metric, unit=UNITS[metric],
            value=None if reason else float(value),
            status="missing" if reason else "available", missing_reason=reason,
        ))
    ready = (provenance.coverage == "complete" and provenance.freshness == "current"
             and provenance.adjustment_basis != "unknown"
             and all(item.status == "available" for item in observations))
    return ScannerEvidence(instrument=instrument, provenance=provenance,
                           observations=tuple(observations), status="ready" if ready else "limited")
