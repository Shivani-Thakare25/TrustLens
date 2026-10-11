"""Creates the final, submission-ready project archive ZIP.
Excludes virtual environments, pycache, git history, and large temporary files.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "mlruns",
    ".dvc",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".DS_Store",
}


def create_submission_archive(out_path: Path | None = None) -> Path:
    if out_path is None:
        out_path = ROOT.parent / "TrustLens_Final_Submission.zip"

    print(f"Creating final submission archive at: {out_path}")
    total_files = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in ROOT.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(ROOT)
            parts = set(rel.parts)
            if parts & EXCLUDE_DIRS:
                continue
            if file_path.suffix in EXCLUDE_EXTS:
                continue
            zf.write(file_path, arcname=f"TrustLens/{rel.as_posix()}")
            total_files += 1

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Archive successfully created: {total_files} files packaged ({size_mb:.2f} MB).")
    return out_path


if __name__ == "__main__":
    create_submission_archive()
