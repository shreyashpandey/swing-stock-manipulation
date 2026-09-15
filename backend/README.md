# Backend foundation

This is the first B02 implementation increment: validated observation DTOs and a
pure projection from a frozen internal radar row. It does not expose HTTP routes,
connect to the local SQLite store, authenticate users, or publish scanner results.

Install with `python -m pip install -e '.[backend]'`. Pydantic 2.13.4 is pinned to
the version used for this increment. Export the JSON Schema with:

```sh
python -m backend.swingdesk_api.export_schema
```

The checked-in `swingdesk_api/scanner-evidence-v1.schema.json` is the client-facing
schema artifact; a regression check verifies that it matches the Python models.

`project_radar_observations` requires an explicit instrument identity and batch
provenance. It copies only close, stored-observation range, volume multiple, and
average daily traded value. It never serializes internal free-text explanations,
recommendations, allocation amounts, composite scores or manipulation claims.
NaN/infinity become explicit missing observations; invalid types fail validation.
Source identifiers are references, not evidence of data rights. A future worker
must resolve these references to durable source and batch records.

`ready` means that all four values are available and the supplied coverage,
freshness and adjustment metadata meet the contract; it is not a trading signal
or legal clearance. The caller must obtain those facts from a validated batch.
Unknown coverage cannot become ready. The legacy engine cannot establish full
provenance by itself, so this adapter is presently for internal previews.

Next: B03 durable calendar/adjustment/batch contracts, followed by B04 identity,
PostgreSQL migrations and owner-scoped API repositories. B02 substantive output
review remains pending. The existing Streamlit interface remains the running app.

Validation behavior follows the [Pydantic model documentation](https://docs.pydantic.dev/latest/concepts/models/)
and [configuration reference](https://docs.pydantic.dev/latest/api/config/).
