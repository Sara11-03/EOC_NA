#!/usr/bin/env python3
"""Run from any working directory; see --help."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eoc_geo.cli import export_qpu_main, main_guard
if __name__ == "__main__":
    main_guard(export_qpu_main)
