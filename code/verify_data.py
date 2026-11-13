#!/usr/bin/env python3
"""Verify extracted Zenodo analysis tables against their file manifest."""

from __future__ import annotations

from framework import PROJECT_ROOT, DATA_ROOT

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = PROJECT_ROOT
DEFAULT_MANIFEST = ROOT / "zenodo" / 'figure_analysis_inputs.tsv'


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        choices=(2, 3, 4, 5, 6),
        help="Check only files used by these figures.",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Check file presence and byte size without computing SHA-256 checksums.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    selected = set(args.figures or ())
    failures: list[str] = []
    checked = 0
    total_bytes = 0

    with args.manifest.expanduser().resolve().open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            row_figures = {int(value) for value in row["paper_figures"].split(",")}
            if selected and not selected.intersection(row_figures):
                continue
            path = root / row["source_path"]
            expected_size = int(row["bytes"])
            checked += 1
            if not path.is_file():
                failures.append(f"MISSING  {path.relative_to(root)}")
                continue
            actual_size = path.stat().st_size
            total_bytes += actual_size
            if actual_size != expected_size:
                failures.append(
                    f"SIZE     {path.relative_to(root)}: {actual_size} != {expected_size}"
                )
                continue
            if not args.no_hash:
                actual_hash = sha256(path)
                if actual_hash != row["sha256"]:
                    failures.append(
                        f"SHA256   {path.relative_to(root)}: {actual_hash} != {row['sha256']}"
                    )

    if checked == 0:
        parser.error("No files in this manifest match the selected figures; choose the corresponding input manifest.")
    if failures:
        print("\n".join(failures))
        raise SystemExit(f"Verification failed: {len(failures)} of {checked} files")
    mode = "size" if args.no_hash else "size and SHA-256"
    print(f"Verified {checked} files ({total_bytes:,} bytes) by {mode}.")


if __name__ == "__main__":
    main()

