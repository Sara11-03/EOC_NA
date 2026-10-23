#!/usr/bin/env python3
"""Download municipality GeoJSON data used by the case-study scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow execution as `python scripts/...py` without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eoc_na.geo import OPENPOLIS_COMUNI_URL, download_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=OPENPOLIS_COMUNI_URL)
    parser.add_argument("--out", type=Path, default=Path("data/raw/comuni.geojson"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = download_file(args.url, args.out, overwrite=args.overwrite)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
