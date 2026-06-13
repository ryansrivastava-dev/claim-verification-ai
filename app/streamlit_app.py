"""Alternate Streamlit entry point for local runs from the app/ folder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.streamlit_app_core import main


main(project_root=ROOT)
