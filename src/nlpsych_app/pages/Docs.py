from __future__ import annotations

from pathlib import Path
import base64
import streamlit as st


def _load_docs_markdown() -> str:
    root = Path(__file__).resolve().parents[3]
    doc_path = root / "docs" / "project_overview.md"
    if doc_path.exists():
        return doc_path.read_text(encoding="utf-8")
    return (
        "# NLPsych Docs\n\n"
        "Documentation file not found. Expected: docs/project_overview.md\n"
    )


def _load_logo_bytes() -> bytes:
    root = Path(__file__).resolve().parents[3]
    logo_path = root / "assets" / "NLPsych_logo.png"
    if logo_path.exists():
        return logo_path.read_bytes()
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
                  <a href="/" style="text-decoration:none;">
                    <img src="{data_uri}" width="120">
                  </a>
                </div>
                <div style="text-align:center; font-weight:700; font-size:1.25rem;">NLPsych</div>
                <div style="text-align:center; color:#666; font-size:0.9rem; margin-bottom: 8px;">
                  Natural Language Psychometrics
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("---")
        try:
            st.page_link("app.py", label="Back to App", icon="🏠")
        except Exception:
            st.markdown("[Back to App](/)")
        st.markdown("---")
        st.markdown("📘 Docs")
        doc_sections = [
            ("Overview", "overview"),
            ("Beginner Tutorial (Non-ML)", "beginner-tutorial-non-ml"),
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
        links_md = "\n".join(f"- [{label}](#{anchor})" for label, anchor in doc_sections)
        st.markdown(links_md)

    md = _load_docs_markdown()
    st.markdown(md)


if __name__ == "__main__":
    main()
