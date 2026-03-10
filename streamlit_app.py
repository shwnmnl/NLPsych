from __future__ import annotations

from pathlib import Path

import streamlit as st


def main() -> None:
    # Streamlit >=1.36 supports explicit multipage routing via st.navigation.
    # This keeps a single canonical pages directory under src/nlpsych_app/pages.
    if hasattr(st, "navigation") and hasattr(st, "Page"):
        root = Path(__file__).resolve().parent
        app_script = root / "src" / "nlpsych_app" / "app.py"
        docs_script = root / "src" / "nlpsych_app" / "pages" / "Docs.py"
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
