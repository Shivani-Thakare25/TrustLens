"""Download a CSV/JSON dataset from a direct HTTPS URL you have the right to use.
Usage: python scripts/download_data.py --url https://... --out data/raw/reviews.csv
       python scripts/download_data.py --list-sources
No dataset is bundled or assumed. Check each dataset's license and label semantics (see DATA_SOURCES.md).
"""

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAX_BYTES = 500 * 1024 * 1024
SOURCES = [
    "Ott et al. Deceptive Opinion Spam Corpus",
    "Yelp reviews (Yelp Open Dataset / YelpChi, YelpNYC, YelpZip)",
    "Amazon review datasets (e.g. UCSD / McAuley lab)",
    "Hugging Face datasets with a suitable license",
    "Kaggle datasets where terms permit",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--out", default="data/raw/reviews.csv")
    ap.add_argument("--list-sources", action="store_true")
    a = ap.parse_args()
    if a.list_sources or not a.url:
        print("Candidate public sources (verify license and label meaning before use):")
        print("\n".join(f"  - {s}" for s in SOURCES))
        sys.exit(0)
    if not a.url.startswith("https://"):
        sys.exit("Only https:// URLs are allowed.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with urllib.request.urlopen(a.url, timeout=60) as r, open(a.out, "wb") as f:
        while chunk := r.read(1 << 20):
            total += len(chunk)
            if total > MAX_BYTES:
                f.close()
                Path(a.out).unlink()
                sys.exit("Download exceeds the 500 MB safety limit.")
            f.write(chunk)
    print(f"saved {total} bytes to {a.out}")
