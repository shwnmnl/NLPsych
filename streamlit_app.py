from __future__ import annotations

import os
import sys
from pathlib import Path

for env_key, env_value in {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ.setdefault(env_key, env_value)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
SRC_STR = str(SRC)
if SRC_STR not in sys.path:
    sys.path.insert(0, SRC_STR)

import streamlit as st


def main() -> None:
    # Streamlit >=1.36 supports explicit multipage routing via st.navigation.
    # This keeps a single canonical pages directory under src/nlpsych_app/pages.
    if hasattr(st, "navigation") and hasattr(st, "Page"):
        app_script = ROOT / "src" / "nlpsych_app" / "app.py"
        docs_script = ROOT / "src" / "nlpsych_app" / "pages" / "Docs.py"
        nav = st.navigation(
            [
                st.Page(str(app_script), title="Main", icon="💬", default=True, url_path=""),
                st.Page(str(docs_script), title="Docs", icon="📘", url_path="Docs"),
            ]
        )
        nav.run()
        return

    # Fallback for older Streamlit versions.
    from nlpsych_app.app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
