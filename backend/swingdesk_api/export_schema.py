"""Print the versioned evidence JSON Schema: python -m backend.swingdesk_api.export_schema."""
import json

from .contracts import ScannerEvidence


def main():
    print(json.dumps(ScannerEvidence.model_json_schema(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
