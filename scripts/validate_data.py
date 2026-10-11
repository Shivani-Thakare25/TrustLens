"""Usage: python scripts/validate_data.py <raw.csv> [--out reports/data_validation.json]"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingestion import load_raw  # noqa: E402
from src.data.validation import validate_dataframe  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report = validate_dataframe(load_raw(a.data))
    text = json.dumps(report.to_dict(), indent=2)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    sys.exit(0 if report.ok else 1)
