"""Docs page shim for root-level Streamlit entrypoints.

This lets `streamlit_app.py` discover the Docs page via `pages/Docs.py`
while keeping the canonical implementation inside `nlpsych_app.pages.Docs`.
"""

from nlpsych_app.pages.Docs import main


if __name__ == "__main__":
    main()
