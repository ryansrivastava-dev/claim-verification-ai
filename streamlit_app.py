"""Streamlit entry point for Evidence-Based Fact Verification."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.streamlit_app_core import main


main(project_root=ROOT)
