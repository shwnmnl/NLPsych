from __future__ import annotations

from pathlib import Path
import base64
from importlib.resources import files
import streamlit as st


def _candidate_project_roots() -> list[Path]:
    roots: list[Path] = []
    # Streamlit Cloud/local runs from repo root
    cwd = Path.cwd().resolve()
    roots.append(cwd)
    # Source-tree runs (e.g., streamlit run src/nlpsych_app/app.py)
    roots.append(Path(__file__).resolve().parents[3])
    # De-duplicate while preserving order
    seen: set[Path] = set()
    ordered: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            ordered.append(root)
    return ordered


def _load_docs_markdown() -> str:
    for root in _candidate_project_roots():
        doc_path = root / "docs" / "project_overview.md"
        if doc_path.exists():
            return doc_path.read_text(encoding="utf-8")
    # Optional packaged docs path for future wheel inclusion.
    try:
        packaged_doc = files("nlpsych_app") / "docs" / "project_overview.md"
        if packaged_doc.is_file():
            return packaged_doc.read_text(encoding="utf-8")
    except Exception:
        pass
    return (
        "# NLPsych Docs\n\n"
        "Documentation file not found. Expected: docs/project_overview.md\n"
    )


def _load_logo_bytes() -> bytes:
    for root in _candidate_project_roots():
        logo_path = root / "assets" / "NLPsych_logo.png"
        if logo_path.exists():
            return logo_path.read_bytes()
    try:
        pkg_logo = files("nlpsych_app") / "assets" / "NLPsych_logo.png"
        if pkg_logo.is_file():
            return pkg_logo.read_bytes()
    except Exception:
        pass
    return b""


def main() -> None:
    st.set_page_config(
        page_title="NLPsych Docs",
        page_icon="📘",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    logo_b = _load_logo_bytes()
    data_uri = ""
    if logo_b:
        data_uri = "data:image/png;base64," + base64.b64encode(logo_b).decode("utf-8")

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
            display: none;
        }
        section[data-testid="stSidebar"] .stSidebarNav {
            display: none;
        }
        .nlpsych-sidebar-card {
            background: #ffffff;
            border: 1px solid #e9e9e9;
            border-radius: 12px;
            padding: 12px 14px;
            margin: 10px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .nlpsych-sidebar-title {
            font-weight: 700;
            margin-bottom: 6px;
        }
        .nlpsych-sidebar-muted {
            color: #666666;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        if data_uri:
            st.markdown(
                f"""
                <div style="text-align:center; margin-bottom: 6px;">
                  <img src="{data_uri}" width="120">
                </div>
                <div style="text-align:center; font-weight:700; font-size:1.25rem;">NLPsych</div>
                <div style="text-align:center; color:#666; font-size:0.9rem; margin-bottom: 8px;">
                  Natural Language Psychometrics
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("---")
        back_link_rendered = False
        for candidate in ("src/nlpsych_app/app.py", "streamlit_app.py", "app.py"):
            try:
                st.page_link(candidate, label="Back to App", icon="🏠")
                back_link_rendered = True
                break
            except Exception:
                continue
        if not back_link_rendered:
            st.markdown("[Back to App](/)")
        st.markdown("---")
        st.markdown("📘 Docs")
        doc_sections = [
            ("Overview", "overview"),
            ("Orientation: Using Text Embeddings for Statistical Analysis", "orientation-using-text-embeddings-for-statistical-analysis"),
            ("Feature Highlights", "feature-highlights"),
            ("Architecture", "architecture"),
            ("Installation", "installation"),
            ("Quickstart (Library)", "quickstart-library"),
            ("Streamlit App", "streamlit-app"),
            ("Module Reference", "module-reference"),
            ("Data Flow: Text → Report", "data-flow-text-report"),
            ("Testing & Quality", "testing--quality"),
            ("Support & Contributing", "support--contributing"),
        ]
        links_html = (
            "<ul style='margin:0.25rem 0 0 1rem; padding:0;'>"
            + "".join(
                f"<li><a href='#{anchor}' target='_self'>{label}</a></li>"
                for label, anchor in doc_sections
            )
            + "</ul>"
        )
        st.markdown(links_html, unsafe_allow_html=True)

    md = _load_docs_markdown()
    st.markdown(md)


if __name__ == "__main__":
    main()
