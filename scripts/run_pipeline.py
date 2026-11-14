#!/usr/bin/env python3
"""Command-line entry point for the complete WATTWISE ML pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline import _json_ready, run_pipeline  # noqa: E402


if __name__ == "__main__":
    metrics = run_pipeline(PROJECT_ROOT)
    print(json.dumps(metrics, indent=2, default=_json_ready))

