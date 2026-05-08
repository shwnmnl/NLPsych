from __future__ import annotations
import html
import inspect
import io
import sys
import textwrap
import math
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import spacy 
import base64
from pathlib import Path
from importlib.resources import files
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import LinearSVC

_SRC_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT_STR = str(_SRC_ROOT)
if _SRC_ROOT_STR not in sys.path:
    sys.path.insert(0, _SRC_ROOT_STR)

from nlpsych.utils import get_spacy_pipeline_base, get_st_model_base
from nlpsych.descriptive_stats import (
    descriptive_stats,
    build_descriptive_summary_table,
    descriptive_summary_table_to_markdown,
    descriptive_summary_table_to_latex,
)
from nlpsych.embedding import (
    embed_text_columns_simple_base,
    reduce_embeddings,
    build_plot_df,
    plot_projection,
)
from nlpsych.topic_modeling import (
    build_topic_plot,
    fit_topic_model,
    build_topic_assignments,
    summarize_topics as summarize_topics_table,
    merge_topic_assignments,
    resolve_projection_from_topic_match,
    update_topic_representation,
)
from nlpsych_app.topic_state import (
    clear_topic_state,
    store_topic_results,
)
from plotly.subplots import make_subplots
from nlpsych.modeling import auto_cv_with_permutation, apply_multiple_comparisons_correction
from nlpsych.report import build_report_payload, interpret_model_row, summarize_model_row


ASSETS = files("nlpsych_app") / "assets"
TEMP_DEMO_CSV = _SRC_ROOT.parent / "demo_data" / "nlpsych_dp_demo.csv"
EMBEDDING_MODEL_OPTIONS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "all-distilroberta-v1",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "distiluse-base-multilingual-cased-v2",
]

def main():
    # ===== Session state init =====
    def _init_state():
        ss = st.session_state
        ss.setdefault("stats_df", None)
        ss.setdefault("overall", None)
        ss.setdefault("meta_df", None)
        ss.setdefault("embeddings", None)
        ss.setdefault("texts", None)
        ss.setdefault("embedding_opts", None)
        ss.setdefault("projection_opts", None)
        ss.setdefault("plot_df", None)
        ss.setdefault("topic_model", None)
        ss.setdefault("topic_assignments_df", None)
        ss.setdefault("topic_summary_df", None)
        ss.setdefault("cluster_opts", None)
        ss.setdefault("topic_repr_opts", None)
        ss.setdefault("topic_plot_fig", None)
        ss.setdefault("topic_plot_kind", None)
        ss.setdefault("topic_plot_opts", None)
        ss.setdefault("results_df", None)
        ss.setdefault("preds", None)
    _init_state()

    # ===== Section: Cached resources =====
    # For reshowing cached data when clicking from one tab to another
    def _opts_same(a: dict | None, b: dict | None) -> bool:
        return a is not None and b is not None and a == b

    @st.cache_resource(show_spinner=False)
    def get_spacy_pipeline() -> spacy.Language:
        """UI cached spaCy pipeline"""
        return get_spacy_pipeline_base(allow_download=True)

    @st.cache_resource(show_spinner=False)
    def get_st_model(model_name: str = "all-MiniLM-L6-v2") -> "SentenceTransformer":
        """UI cached SentenceTransformer model keyed by model_name"""
        return get_st_model_base(model_name)

    @st.cache_data(show_spinner=False)
    def embed_text_columns_simple_ui(
        series_list: List[pd.Series],
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
    ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        return embed_text_columns_simple_base(series_list, model_name=model_name, normalize=normalize)

    # ===== Section: Utility for downloads =====
    def df_to_csv_download(
        df: pd.DataFrame,
        filename: str,
        include_index: bool = True,
        button_label: str = "Download CSV",
    ):
        csv = df.to_csv(index=include_index).encode("utf-8")
        st.download_button(
            button_label,
            data=csv,
            file_name=filename,
            mime="text/csv",
            key=f"dl_csv_{filename}",
        )


    def text_download(text: str, filename: str, button_label: str, mime: str = "text/plain"):
        st.download_button(
            button_label,
            data=text.encode("utf-8"),
            file_name=filename,
            mime=mime,
            key=f"dl_text_{filename}",
        )


    def npy_download(arr: np.ndarray, filename: str):
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        st.download_button("Download NPY", data=buf, file_name=filename, mime="application/octet-stream")

    def _field_label(label: str, tooltip: str | None = None):
        safe_label = html.escape(label)
        if tooltip:
            safe_tooltip = html.escape(tooltip)
            st.markdown(
                f"<div class=\"nlpsych-field-label\">{safe_label}<span class=\"nlpsych-help\" title=\"{safe_tooltip}\">?</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class=\"nlpsych-field-label\">{safe_label}</div>",
                unsafe_allow_html=True,
            )

    def render_cached_topic_plot(fig: object) -> None:
        """Render a cached BERTopic visualization."""
        if isinstance(fig, go.Figure):
            st.plotly_chart(fig, width='stretch')
            return
        st.write(fig)

    # ===== Logo loading helper =====
    def load_logo_bytes(filename="NLPsych_logo.png") -> bytes:
        # 1) Dev/repo path for local runs
        dev_path = Path(__file__).resolve().parents[2] / "assets" / filename
        if dev_path.exists():
            return dev_path.read_bytes()

        # 2) Packaged asset for pip installs
        pkg_res = files("nlpsych_app") / "assets" / filename
        with pkg_res.open("rb") as f:
            return f.read()

    # ===== Section: Streamlit UI =====
    # Page config and title
    st.set_page_config(
        page_title="NLPsych",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    # Reduce top padding so main content starts higher on load
    st.markdown(
        """
        <style>
        .block-container { padding-top: 0.5rem; }
        .nlpsych-field-label {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-weight: 600;
            font-size: 0.9rem;
            margin: 0.35rem 0 0.1rem 0;
        }
        .nlpsych-help {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 14px;
            height: 14px;
            border-radius: 999px;
            border: 1px solid #9aa0a6;
            color: #6b6f76;
            font-size: 10px;
            line-height: 14px;
            cursor: help;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Centered logo
    logo_b = load_logo_bytes("NLPsych_logo.png")
    data_uri = "data:image/png;base64," + base64.b64encode(logo_b).decode("utf-8")

    # Sidebar styling and content (left rail)
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
            display: none;
        }
        section[data-testid="stSidebar"] .stSidebarNav {
            display: none;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 0rem !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
            padding-top: 0rem !important;
        }
        section[data-testid="stSidebar"] .element-container:first-child {
            margin-top: -2.5rem;
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

    def _sidebar_card(title: str, body_html: str):
        st.sidebar.markdown(
            f"""
            <div class="nlpsych-sidebar-card">
              <div class="nlpsych-sidebar-title">{title}</div>
              <div class="nlpsych-sidebar-muted">{body_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom: 6px;">
              <img src="{data_uri}" width="36" height="36" style="border-radius:4px;">
              <div style="font-weight:700; font-size:1.2rem;">NLPsych</div>
            </div>
            <div style="color:#666; font-size:0.9rem; margin-bottom: 8px;">
              <i>Natural Language Psychometrics</i>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _sidebar_card(
            "About",
            "NLPsych turns free-text datasets into descriptive statistics, semantic embeddings, and lightweight predictive models.",
        )
        # _sidebar_card(
        #     "Links",
        #     "<ul style='margin:0 0 0 1rem; padding:0;'>"
        #     "<li>Documentation (add link)</li>"
        #     "<li>GitHub (add link)</li>"
        #     "<li>Issues (add link)</li>"
        #     "<li>Contact (add link)</li>"
        #     "</ul>",
        # )

        st.markdown("---")
        st.markdown("📘 Docs")
        docs_link_rendered = False
        for candidate in ("pages/Docs.py", "src/nlpsych_app/pages/Docs.py"):
            try:
                st.page_link(candidate, label="Open Documentation", icon="📖")
                docs_link_rendered = True
                break
            except Exception:
                continue
        if not docs_link_rendered:
            st.markdown("[Open Documentation](Docs)")

        doc_sections = [
            ("Overview", "overview"),
            (
                "Orientation: Using Text Embeddings for Statistical Analysis",
                "orientation-using-text-embeddings-for-statistical-analysis",
            ),
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
        docs_links_html = (
            "<ul style='margin:0.25rem 0 0 1rem; padding:0;'>"
            + "".join(
                f"<li><a href='Docs#{anchor}' target='_self'>{html.escape(label)}</a></li>"
                for label, anchor in doc_sections
            )
            + "</ul>"
        )
        st.markdown(docs_links_html, unsafe_allow_html=True)
    center_left, center_col, center_right = st.columns([1, 2, 1])
    with center_col:
        st.markdown(
            f"""
            <p style="text-align:center;">
                <img src="{data_uri}" width="150">
            </p>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<h1 style='text-align: center; color: black;'>NLPsych</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align:center;">
            NLPsych (<i>Natural Language Psychometrics</i>) lets you upload a file, pick your text columns, and get instant descriptive statistics, visualize semantic embeddings, and run predictive models all in one streamlined workflow.
            </div>
            <br>
            """,
            unsafe_allow_html=True
        )
        with st.expander("How to use NLPsych"):
            st.markdown(
                """
                **Usage**

                1. Upload a CSV/TSV/XLSX (or toggle demo data) and confirm the detected text columns.
                2. Use the tabs to run descriptive statistics, explore embeddings, train models, and assemble a report.
                3. Download any tables or embedding arrays with the buttons provided in each tab.

                **Tips**

                - Use the “Use demo data” option to explore features before bringing your own dataset.
                - Keep text columns fairly clean (one document per cell) for best spaCy and embedding results.
                - Recompute stats or embeddings after changing preprocessing options to refresh cached results.
                """
            )
            st.info(
                "This public demo runs on third-party servers. See [docs](Docs#streamlit-app) for local setup instructions.",
                icon="ℹ️",
            )

        # Uploader at the top, centered block
        uploaded = st.file_uploader("Upload CSV/TSV/XLSX", type=["csv", "tsv", "xlsx"])

        # Wide content starts here
        use_demo = st.checkbox("Use demo data if no file", value=False)

    # Load data
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix == ".tsv":
            df = pd.read_csv(uploaded, sep="\t")
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)
    else:
        if use_demo:
            if not TEMP_DEMO_CSV.exists():
                st.error(f"Temporary demo file not found: {TEMP_DEMO_CSV}")
                st.stop()
            df = pd.read_csv(TEMP_DEMO_CSV)
            st.info("Using temporary demo CSV from the repository")
        else:
            st.stop()

    st.write("Preview")
    st.dataframe(df.head(), width='stretch')

    # Column selection just once, used by tabs
    candidate_text_cols = [c for c in df.columns if df[c].dtype == object]
    text_cols = st.multiselect("Select text columns", candidate_text_cols, default=candidate_text_cols[:1])
    if not text_cols:
        st.warning("Select at least one text column.")
        st.stop()

    tabs = st.tabs(["Overview", "Descriptive stats", "Embeddings", "Topics / Clusters", "Modeling", "Report"])



    # ===== Tab: Overview =====
    with tabs[0]:
        st.subheader("Dataset overview")
        st.write(f"Rows: {len(df)}")
        st.write(f"Selected text columns: {text_cols}")

    # ===== Tab: Descriptive stats =====
    with tabs[1]:
        st.subheader("Per row and overall stats")

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            _field_label(
                "Alphabetic only",
                "Keep only alphabetic tokens (filters out numbers and punctuation).",
            )
            words_must_be_alpha = st.checkbox(
                "Alphabetic only",
                value=True,
                label_visibility="collapsed",
            )
        with col_b:
            _field_label(
                "Drop stopwords",
                "Remove common stopwords (e.g., the, and) before counting.",
            )
            drop_stopwords = st.checkbox(
                "Drop stopwords",
                value=False,
                label_visibility="collapsed",
            )
        with col_c:
            _field_label(
                "Use lemmas for uniques",
                "Count unique words by lemma (base form) instead of surface form.",
            )
            use_lemmas = st.checkbox(
                "Use lemmas for uniques",
                value=True,
                label_visibility="collapsed",
            )
        with col_d:
            _field_label(
                "Sentence length mode",
                "Unweighted = average per-sentence length; ratio = total tokens / total sentences.",
            )
            avg_sent_mode = st.selectbox(
                "Sentence length mode",
                ["unweighted", "ratio"],
                label_visibility="collapsed",
            )

        current_opts = {
            "words_must_be_alpha": words_must_be_alpha,
            "drop_stopwords": drop_stopwords,
            "use_lemmas": use_lemmas,
            "avg_sent_mode": avg_sent_mode,
            "text_cols": tuple(text_cols),
        }

        cached_df = st.session_state.get("stats_df")
        cached_overall = st.session_state.get("overall")
        cached_opts = st.session_state.get("stats_opts")

        cache_is_fresh = _opts_same(cached_opts, current_opts) and cached_df is not None and cached_overall is not None

        col_left, col_right = st.columns([1, 3])
        with col_left:
            if cached_df is None or cached_overall is None:
                stats_button_label = "Compute descriptive stats"
            else:
                stats_button_label = "Recompute descriptive stats"
            run_stats = st.button(stats_button_label, key="btn_stats_compute")

        with col_right:
            if cache_is_fresh:
                st.caption("Showing cached results")
            elif cached_df is not None:
                st.caption("Cached results exist but options changed. Press Recompute descriptive stats to refresh.")

        def render_descriptive_outputs(stats_df_show: pd.DataFrame, overall_show):
            st.write("Per row stats")
            st.dataframe(stats_df_show, width='stretch')
            df_to_csv_download(stats_df_show, "per_row_stats.csv")

            summary_df = build_descriptive_summary_table(
                stats_df=stats_df_show,
                overall_obj=overall_show,
                decimals=3,
            )
            st.write("Summary table (paper-ready)")
            if summary_df.empty:
                st.info("Summary table could not be built from the current stats output.")
            else:
                st.dataframe(summary_df, width='stretch')
                dl_col_a, dl_col_b, dl_col_c = st.columns(3)
                with dl_col_a:
                    df_to_csv_download(
                        summary_df,
                        "descriptive_summary_table.csv",
                        include_index=False,
                        button_label="Download summary CSV",
                    )
                md_table = descriptive_summary_table_to_markdown(summary_df, decimals=3)
                latex_table = descriptive_summary_table_to_latex(summary_df, decimals=3)
                with dl_col_b:
                    text_download(
                        md_table,
                        "descriptive_summary_table.md",
                        button_label="Download summary Markdown",
                        mime="text/markdown",
                    )
                with dl_col_c:
                    text_download(
                        latex_table,
                        "descriptive_summary_table.tex",
                        button_label="Download summary LaTeX",
                        mime="text/plain",
                    )
                with st.expander("Copy-ready formats"):
                    st.caption("Markdown and LaTeX both use a vertical metric layout for easier manuscript copy/paste.")
                    st.text_area(
                        "Markdown table",
                        value=md_table,
                        height=260,
                        key="stats_summary_markdown_text",
                    )
                    st.text_area(
                        "LaTeX table",
                        value=latex_table,
                        height=220,
                        key="stats_summary_latex_text",
                    )

            st.write("Overall stats")
            st.json(overall_show)

        if run_stats:
            try:
                nlp = get_spacy_pipeline()
                series_list = [df[c] for c in text_cols]
                stats_df, overall = descriptive_stats(
                    *series_list,
                    use_lemmas_for_uniques=use_lemmas,
                    split_overall="both",
                    nlp=nlp,
                    n_process=1,
                    batch_size=256,
                    words_must_be_alpha=words_must_be_alpha,
                    drop_stopwords=drop_stopwords,
                    avg_sentence_mode=avg_sent_mode
                )
                render_descriptive_outputs(stats_df, overall)

                st.session_state["stats_df"] = stats_df
                st.session_state["overall"] = overall
                st.session_state["stats_opts"] = current_opts
            except Exception as e:
                st.exception(e)
        else:
            if cached_df is None or cached_overall is None:
                st.info("Press Compute descriptive stats to generate stats.")
            else:
                render_descriptive_outputs(cached_df, cached_overall)


    # ===== Tab: Embeddings =====
    with tabs[2]:
        st.subheader("Embeddings and projection")

        cached_topic_model = st.session_state.get("topic_model")
        cached_cluster_opts = st.session_state.get("cluster_opts")
        topic_match_available = (
            cached_topic_model is not None
            and isinstance(cached_cluster_opts, dict)
            and str(cached_cluster_opts.get("cluster_reduce_method", "")).lower() in {"umap", "pca"}
        )
        match_topic_reducer = False
        if topic_match_available:
            match_topic_reducer = st.checkbox(
                "Match fitted topic reducer settings",
                value=False,
                help=(
                    "Use the fitted BERTopic clustering reducer family and compatible settings "
                    "for a fresh 2D/3D display projection."
                ),
            )
        elif cached_topic_model is not None:
            st.caption(
                "Matched projection mode becomes available after fitting topics with a cached UMAP or PCA clustering reducer."
            )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            embed_model_options = EMBEDDING_MODEL_OPTIONS + ["Custom"]
            model_choice = st.selectbox(
                "SentenceTransformer model",
                embed_model_options,
                index=0
            )
            if model_choice == "Custom":
                custom_embed_model = st.text_input(
                    "Custom SentenceTransformer model",
                    EMBEDDING_MODEL_OPTIONS[0]
                ).strip()
                model_name = custom_embed_model or EMBEDDING_MODEL_OPTIONS[0]
            else:
                model_name = model_choice
        with col_b:
            if match_topic_reducer:
                reduce_method = str(cached_cluster_opts.get("cluster_reduce_method", "umap")).lower()
                st.text_input(
                    "Projection reducer",
                    value=reduce_method.upper(),
                    disabled=True,
                )
            else:
                reduce_method = st.selectbox(
                    "Projection reducer",
                    ["pca", "umap", "tsne"],
                )
        with col_c:
            n_components = st.selectbox("Projection dimensions", [2, 3], index=0)

        umap_n_neighbors = None
        umap_min_dist = None
        umap_metric = "cosine"
        tsne_perplexity = None
        tsne_learning_rate = None
        tsne_n_iter = None
        tsne_metric = "cosine"
        projection_random_state = 1

        if match_topic_reducer:
            matched_reduce_method = str(cached_cluster_opts.get("cluster_reduce_method", "umap")).upper()
            st.caption(
                f"Matched projection mode is active. The plot will be refit with the fitted topic "
                f"reducer family (`{matched_reduce_method}`) and compatible settings for `{int(n_components)}D` display. "
                "This is not literal reuse of the BERTopic-fitted reducer unless the dimensions already match."
            )
        elif reduce_method == "umap":
            with st.expander("Projection UMAP settings"):
                _field_label(
                    "UMAP n_neighbors",
                    "Controls local vs global structure; higher favors global patterns.",
                )
                umap_n_neighbors = st.number_input(
                    "UMAP n_neighbors",
                    min_value=2,
                    max_value=200,
                    value=15,
                    step=1,
                    label_visibility="collapsed",
                )
                _field_label(
                    "UMAP min_dist",
                    "Minimum distance between points; lower values pack clusters tighter.",
                )
                umap_min_dist = st.number_input(
                    "UMAP min_dist",
                    min_value=0.0,
                    max_value=0.99,
                    value=0.1,
                    step=0.01,
                    format="%.2f",
                    label_visibility="collapsed",
                )
                _field_label(
                    "UMAP metric",
                    "Distance metric used to measure similarity between points.",
                )
                umap_metric = st.selectbox(
                    "UMAP metric",
                    ["cosine", "euclidean", "manhattan"],
                    index=0,
                    label_visibility="collapsed",
                )
        elif reduce_method == "tsne":
            with st.expander("Projection t-SNE settings"):
                _field_label(
                    "Auto perplexity",
                    "Choose perplexity automatically based on sample size.",
                )
                auto_perp = st.checkbox(
                    "Auto perplexity",
                    value=True,
                    label_visibility="collapsed",
                )
                _field_label(
                    "Perplexity",
                    "Effective number of neighbors; lower emphasizes local structure.",
                )
                perp_val = st.number_input(
                    "Perplexity",
                    min_value=2.0,
                    max_value=200.0,
                    value=30.0,
                    step=1.0,
                    disabled=auto_perp,
                    label_visibility="collapsed",
                )
                tsne_perplexity = None if auto_perp else float(perp_val)

                _field_label(
                    "Auto learning rate",
                    "Choose learning rate automatically based on sample size.",
                )
                auto_lr = st.checkbox(
                    "Auto learning rate",
                    value=True,
                    label_visibility="collapsed",
                )
                _field_label(
                    "Learning rate",
                    "Step size for optimization; too high can distort results.",
                )
                lr_val = st.number_input(
                    "Learning rate",
                    min_value=10.0,
                    max_value=2000.0,
                    value=200.0,
                    step=10.0,
                    disabled=auto_lr,
                    label_visibility="collapsed",
                )
                tsne_learning_rate = None if auto_lr else float(lr_val)

                _field_label(
                    "Iterations",
                    "Optimization steps; higher is more stable but slower.",
                )
                tsne_n_iter = int(st.number_input(
                    "Iterations",
                    min_value=250,
                    max_value=5000,
                    value=1000,
                    step=250,
                    label_visibility="collapsed",
                ))
                _field_label(
                    "t-SNE metric",
                    "Distance metric used to measure similarity between points.",
                )
                tsne_metric = st.selectbox(
                    "t-SNE metric",
                    ["cosine", "euclidean"],
                    index=0,
                    label_visibility="collapsed",
                )

        non_text_cols = [c for c in df.columns if c not in text_cols]

        current_embedding_opts = {
            "model_name": model_name,
            "text_cols": tuple(text_cols),
        }
        manual_projection_opts = {
            "reduce_method": reduce_method,
            "n_components": int(n_components),
            "umap_n_neighbors": int(umap_n_neighbors) if umap_n_neighbors is not None else None,
            "umap_min_dist": float(umap_min_dist) if umap_min_dist is not None else None,
            "umap_metric": umap_metric,
            "tsne_perplexity": float(tsne_perplexity) if tsne_perplexity is not None else None,
            "tsne_learning_rate": float(tsne_learning_rate) if tsne_learning_rate is not None else None,
            "tsne_n_iter": int(tsne_n_iter) if tsne_n_iter is not None else None,
            "tsne_metric": tsne_metric,
            "random_state": int(projection_random_state),
        }
        current_projection_opts = resolve_projection_from_topic_match(
            manual_projection_opts,
            match_topic_reducer=match_topic_reducer,
            cluster_opts=cached_cluster_opts,
            default_random_state=projection_random_state,
        )
        current_embed_opts = {
            **current_embedding_opts,
            **current_projection_opts,
        }

        cached_meta = st.session_state.get("meta_df")
        cached_emb = st.session_state.get("embeddings")
        cached_texts = st.session_state.get("texts")
        cached_plot_df = st.session_state.get("plot_df")
        cached_embedding_opts = st.session_state.get("embedding_opts")
        cached_projection_opts = st.session_state.get("projection_opts")

        cached_topic_assignments = st.session_state.get("topic_assignments_df")

        embedding_cache_is_fresh = (
            _opts_same(cached_embedding_opts, current_embedding_opts)
            and cached_meta is not None
            and cached_emb is not None
            and cached_texts is not None
        )
        projection_cache_is_fresh = (
            embedding_cache_is_fresh
            and _opts_same(cached_projection_opts, current_projection_opts)
            and cached_plot_df is not None
        )

        stats_df = st.session_state.get("stats_df")
        stats_color_options: list[str] = []
        if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
            stats_numeric_cols = [
                c for c in stats_df.columns
                if pd.api.types.is_numeric_dtype(stats_df[c])
            ]
            stats_color_options = [f"Stats: {c}" for c in stats_numeric_cols]

        topic_color_options: list[str] = []
        if cached_topic_assignments is not None:
            topic_color_options = ["topic_label", "topic_id", "is_outlier"]

        def _compute_embeddings() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
            meta_df, emb, texts = embed_text_columns_simple_ui(
                [df[c] for c in text_cols],
                model_name=model_name,
                normalize=True,
            )
            return meta_df, emb, texts

        def _compute_projection(
            meta_df: pd.DataFrame,
            emb: np.ndarray,
            texts: list[str],
        ) -> pd.DataFrame:
            projection_method = current_projection_opts["reduce_method"]
            reduction_metric = "cosine"
            if projection_method == "umap":
                reduction_metric = current_projection_opts.get("umap_metric", "cosine")
            elif projection_method == "tsne":
                reduction_metric = current_projection_opts.get("tsne_metric", "cosine")
            Z = reduce_embeddings(
                embeddings=emb,
                method=projection_method,
                n_components=int(current_projection_opts["n_components"]),
                random_state=int(current_projection_opts.get("random_state", 1)),
                metric=reduction_metric,
                umap_n_neighbors=current_projection_opts.get("umap_n_neighbors"),
                umap_min_dist=current_projection_opts.get("umap_min_dist"),
                tsne_perplexity=current_projection_opts.get("tsne_perplexity"),
                tsne_learning_rate=current_projection_opts.get("tsne_learning_rate"),
                tsne_n_iter=current_projection_opts.get("tsne_n_iter"),
            )
            plot_df = build_plot_df(Z, meta_df, texts)
            return plot_df

        def _store_embedding_results(meta_df: pd.DataFrame, emb: np.ndarray, texts: list[str]) -> None:
            st.session_state["meta_df"] = meta_df
            st.session_state["embeddings"] = emb
            st.session_state["texts"] = texts
            st.session_state["embedding_opts"] = current_embedding_opts

        def _store_projection_results(plot_df: pd.DataFrame) -> None:
            st.session_state["plot_df"] = plot_df
            st.session_state["projection_opts"] = current_projection_opts
            st.session_state["embed_opts"] = current_embed_opts

        style_col, plot_col = st.columns([1, 3])
        with style_col:
            if cached_plot_df is None:
                button_label = "Compute embeddings"
            else:
                button_label = "Recompute embeddings"
            run_embed = st.button(button_label, key="btn_embed_compute")

            if projection_cache_is_fresh:
                st.caption("Showing cached results. Styling changes do not require recompute.")
            elif cached_plot_df is not None:
                st.caption("Cached results exist but options changed. Press Recompute embeddings to refresh.")

            st.markdown("**Plot styling**")
            try:
                if "height" in inspect.signature(st.container).parameters:
                    style_container = st.container(height=560, border=True)
                else:
                    style_container = st.container()
            except Exception:
                style_container = st.container()

            with style_container:
                plot_height = st.slider("Plot height (px)", min_value=350, max_value=1200, value=500, step=25)
                color_by_options = (
                    ["source_column"]
                    + topic_color_options
                    + non_text_cols
                    + stats_color_options
                    + ["None"]
                )
                color_by_choice = st.selectbox("Color points by", color_by_options, index=0)
                point_size = st.slider("Point size", min_value=2, max_value=20, value=8)
                point_opacity = st.slider("Point opacity", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
                show_legend = st.checkbox("Show legend", value=True)
                color_mode = st.selectbox("Color interpretation", ["Auto", "Categorical", "Continuous"], index=0)

                template_options = {
                    "Plotly White": "plotly_white",
                    "Plotly Dark": "plotly_dark",
                    "Simple White": "simple_white",
                    "GGPlot2": "ggplot2",
                    "Seaborn": "seaborn",
                    "Presentation": "presentation",
                    "Default (Plotly)": None,
                }
                template_choice = st.selectbox("Theme", list(template_options.keys()), index=0)
                template_value = template_options[template_choice]

                transparent_bg = st.checkbox("Transparent background", value=False)
                use_custom_bg = st.checkbox("Custom background colors", value=False, disabled=transparent_bg)
                if transparent_bg:
                    paper_bgcolor = "rgba(0,0,0,0)"
                    plot_bgcolor = "rgba(0,0,0,0)"
                    hide_axes = True
                elif use_custom_bg:
                    paper_bgcolor = st.color_picker("Paper background", "#ffffff")
                    plot_bgcolor = st.color_picker("Plot background", "#ffffff")
                    hide_axes = False
                else:
                    paper_bgcolor = None
                    plot_bgcolor = None
                    hide_axes = False

                def _parse_color_list(raw: str) -> list[str]:
                    return [c.strip() for c in raw.split(",") if c.strip()]

                color_discrete_sequence = None
                color_continuous_scale = None
                inferred_type = "categorical"
                if color_by_choice in df.columns:
                    inferred_type = "continuous" if pd.api.types.is_numeric_dtype(df[color_by_choice]) else "categorical"
                elif color_by_choice.startswith("Stats: "):
                    inferred_type = "continuous"
                elif color_by_choice in ("source_column", "index", "topic_label", "is_outlier"):
                    inferred_type = "categorical"
                elif color_by_choice == "topic_id":
                    inferred_type = "continuous"

                if color_by_choice != "None":
                    show_continuous = color_mode == "Continuous" or (color_mode == "Auto" and inferred_type == "continuous")
                    show_categorical = color_mode == "Categorical" or (color_mode == "Auto" and inferred_type == "categorical")

                    if show_continuous:
                        cont_scales = {
                            "Default (Plotly)": None,
                            "Viridis": px.colors.sequential.Viridis,
                            "Plasma": px.colors.sequential.Plasma,
                            "Cividis": px.colors.sequential.Cividis,
                            "Turbo": px.colors.sequential.Turbo,
                            "Blues": px.colors.sequential.Blues,
                            "Reds": px.colors.sequential.Reds,
                        }
                        cont_choice = st.selectbox("Continuous color scale", list(cont_scales.keys()), index=0)
                        color_continuous_scale = cont_scales[cont_choice]

                    if show_categorical:
                        discrete_palettes = {
                            "Plotly (default)": None,
                            "D3": px.colors.qualitative.D3,
                            "G10": px.colors.qualitative.G10,
                            "T10": px.colors.qualitative.T10,
                            "Plotly": px.colors.qualitative.Plotly,
                            "Dark24": px.colors.qualitative.Dark24,
                            "Set1": px.colors.qualitative.Set1,
                            "Pastel1": px.colors.qualitative.Pastel1,
                            "Set2": px.colors.qualitative.Set2,
                        }
                        palette_choice = st.selectbox("Discrete color palette", list(discrete_palettes.keys()), index=0)
                        custom_palette = st.text_input("Custom palette (comma-separated hex, optional)", "")
                        if custom_palette.strip():
                            color_discrete_sequence = _parse_color_list(custom_palette)
                        else:
                            color_discrete_sequence = discrete_palettes[palette_choice]

        plot_style = {
            "color_by": None if color_by_choice == "None" else color_by_choice,
            "color_mode": color_mode,
            "point_size": int(point_size),
            "point_opacity": float(point_opacity),
            "color_discrete_sequence": color_discrete_sequence,
            "color_continuous_scale": color_continuous_scale,
            "template": template_value,
            "plot_bgcolor": plot_bgcolor,
            "paper_bgcolor": paper_bgcolor,
            "show_legend": bool(show_legend),
            "hide_axes": bool(hide_axes),
            "plot_height": int(plot_height),
        }

        def _render_embed_block(plot_df, emb, meta_df, style: dict):
            plot_df_use = plot_df
            color_by = style.get("color_by")
            color_mode = style.get("color_mode", "Auto")
            if "text" in plot_df_use.columns:
                def _wrap_hover_text(value: str, width: int = 70) -> str:
                    if value is None:
                        return ""
                    text = str(value).replace("\r", " ").replace("\n", " ")
                    wrapped = "<br>".join(textwrap.wrap(text, width=width))
                    return wrapped or text

                plot_df_use = plot_df_use.copy()
                plot_df_use["text"] = plot_df_use["text"].map(_wrap_hover_text)

            if color_by and color_by.startswith("Stats: "):
                if not isinstance(stats_df, pd.DataFrame) or stats_df.empty:
                    st.warning("Compute descriptive stats to use stats-based coloring.")
                    color_by = None
                else:
                    metric_name = color_by.replace("Stats: ", "", 1)
                    if metric_name not in stats_df.columns:
                        st.warning(f"Descriptive stat '{metric_name}' not found.")
                        color_by = None
                    else:
                        stats_series = stats_df[metric_name]
                        if isinstance(stats_series, pd.Series):
                            if not stats_series.index.is_unique:
                                st.warning(
                                    "Descriptive stats index has duplicates; averaging values for coloring."
                                )
                                stats_series = stats_series.groupby(level=[0, 1]).mean()
                            idx = pd.MultiIndex.from_frame(plot_df_use[["source_column", "index"]])
                            plot_df_use = plot_df_use.copy()
                            plot_df_use[color_by] = stats_series.reindex(idx).to_numpy()

            if color_by and color_by not in plot_df_use.columns and color_by in df.columns:
                color_series = df[color_by]
                if not color_series.index.is_unique:
                    st.warning(
                        f"Column '{color_by}' has duplicate indices; using the first occurrence for coloring."
                    )
                    color_series = color_series.groupby(level=0).first()
                plot_df_use = plot_df_use.copy()
                plot_df_use[color_by] = plot_df_use["index"].map(color_series)
            elif color_by and color_by not in plot_df_use.columns:
                st.warning(f"Color-by column '{color_by}' not found; defaulting to no color.")
                color_by = None

            resolved_mode = None
            if color_by and color_by in plot_df_use.columns:
                series = plot_df_use[color_by]
                is_numeric = pd.api.types.is_numeric_dtype(series)
                if color_mode == "Categorical":
                    plot_df_use = plot_df_use.copy()
                    plot_df_use[color_by] = series.astype(str)
                    resolved_mode = "categorical"
                elif color_mode == "Continuous":
                    if is_numeric:
                        resolved_mode = "continuous"
                    else:
                        coerced = pd.to_numeric(series, errors="coerce")
                        if coerced.notna().any():
                            plot_df_use = plot_df_use.copy()
                            plot_df_use[color_by] = coerced
                            resolved_mode = "continuous"
                        else:
                            st.warning(f"Column '{color_by}' is not numeric; falling back to categorical colors.")
                            plot_df_use = plot_df_use.copy()
                            plot_df_use[color_by] = series.astype(str)
                            resolved_mode = "categorical"
                else:
                    if is_numeric:
                        resolved_mode = "continuous"
                    else:
                        plot_df_use = plot_df_use.copy()
                        plot_df_use[color_by] = series.astype(str)
                        resolved_mode = "categorical"

            discrete_seq = style.get("color_discrete_sequence") if resolved_mode == "categorical" else None
            continuous_scale = style.get("color_continuous_scale") if resolved_mode == "continuous" else None

            fig = plot_projection(
                plot_df_use,
                n_components=plot_df_use.filter(like="dim_").shape[1],
                color_by=color_by,
                point_size=style.get("point_size", 8),
                point_opacity=style.get("point_opacity", 0.9),
                color_discrete_sequence=discrete_seq,
                color_continuous_scale=continuous_scale,
                template=style.get("template"),
                plot_bgcolor=style.get("plot_bgcolor"),
                paper_bgcolor=style.get("paper_bgcolor"),
                show_legend=style.get("show_legend", True),
                hide_axes=style.get("hide_axes", False),
            )
            plot_height = style.get("plot_height")
            if plot_height:
                fig.update_layout(height=plot_height)
            st.plotly_chart(fig, width='stretch')
            st.write("Projection coordinates")
            st.dataframe(plot_df.head(), width='stretch')
            df_to_csv_download(plot_df, "projection.csv")
            st.write("Raw embeddings")
            npy_download(emb, "embeddings.npy")
            emb_cols = [f"e{i+1}" for i in range(emb.shape[1])]
            emb_df = pd.DataFrame(emb, columns=emb_cols)
            emb_df = pd.concat([meta_df.reset_index(drop=True), emb_df], axis=1)
            df_to_csv_download(emb_df, "embeddings_with_meta.csv")

        display_meta = cached_meta
        display_emb = cached_emb
        display_plot_df = cached_plot_df
        plot_error = None

        if run_embed:
            try:
                if embedding_cache_is_fresh:
                    meta_df = cached_meta
                    emb = cached_emb
                    texts = cached_texts
                else:
                    meta_df, emb, texts = _compute_embeddings()
                    clear_topic_state(st.session_state)
                    _store_embedding_results(meta_df, emb, texts)
                plot_df = _compute_projection(meta_df, emb, texts)
                _store_projection_results(plot_df)
                display_meta = meta_df
                display_emb = emb
                display_plot_df = plot_df
            except Exception as e:
                plot_error = e

        if display_plot_df is not None and display_emb is not None and display_meta is not None:
            if cached_topic_assignments is not None:
                try:
                    display_plot_df = merge_topic_assignments(display_plot_df, cached_topic_assignments)
                except Exception:
                    pass

        with plot_col:
            if plot_error is not None:
                st.exception(plot_error)
            elif display_plot_df is None or display_emb is None or display_meta is None:
                st.info("Press Compute embeddings to generate embeddings and projection.")
            else:
                _render_embed_block(display_plot_df, display_emb, display_meta, plot_style)


    # ===== Tab: Topics / Clusters =====
    with tabs[3]:
        st.subheader("Cluster discovery and topic modeling")

        cached_meta = st.session_state.get("meta_df")
        cached_emb = st.session_state.get("embeddings")
        cached_texts = st.session_state.get("texts")
        cached_embedding_opts = st.session_state.get("embedding_opts")
        cached_topic_model = st.session_state.get("topic_model")
        cached_topic_assignments = st.session_state.get("topic_assignments_df")
        cached_topic_summary = st.session_state.get("topic_summary_df")
        cached_cluster_opts = st.session_state.get("cluster_opts")
        cached_topic_repr_opts = st.session_state.get("topic_repr_opts")
        cached_topic_plot_fig = st.session_state.get("topic_plot_fig")
        cached_topic_plot_kind = st.session_state.get("topic_plot_kind")
        cached_topic_plot_opts = st.session_state.get("topic_plot_opts")
        cached_plot_df = st.session_state.get("plot_df")
        embedding_cache_ready = (
            cached_meta is not None
            and cached_emb is not None
            and cached_texts is not None
            and cached_embedding_opts is not None
        )

        def _render_topic_outputs(assignments_df: pd.DataFrame, summary_df: pd.DataFrame):
            total_docs = int(len(assignments_df))
            noise_mask = assignments_df["topic_id"].eq(-1)
            noise_count = int(noise_mask.sum())
            non_noise_topics = int(assignments_df.loc[~noise_mask, "topic_id"].nunique())
            noise_share = float(noise_count / total_docs) if total_docs else 0.0

            metric_cols = st.columns(4)
            metric_cols[0].metric("Documents", f"{total_docs}")
            metric_cols[1].metric("Non-noise topics", f"{non_noise_topics}")
            metric_cols[2].metric("Noise documents", f"{noise_count}")
            metric_cols[3].metric("Noise share", f"{noise_share:.1%}")

            st.write("Topic summary")
            st.dataframe(summary_df, width='stretch')
            df_to_csv_download(summary_df, "topic_summary.csv", include_index=False)

            st.write("Topic assignments")
            st.dataframe(assignments_df, width='stretch')
            df_to_csv_download(assignments_df, "topic_assignments.csv", include_index=False)

        if embedding_cache_ready:
            cached_model_name = cached_embedding_opts.get("model_name", "unknown model")
            cached_text_cols = cached_embedding_opts.get("text_cols", ())
            st.caption(
                f"Using cached embeddings from `{cached_model_name}` on {len(cached_texts)} text cells "
                f"across {len(cached_text_cols)} selected text column(s). "
                "After fitting topics here, go back to `Embeddings` to color the projection by topic."
            )
        else:
            st.info("Compute embeddings in the Embeddings tab first. This tab uses the current cached embeddings.")

        cluster_col, cluster_results_col = st.columns([1, 3])
        with cluster_col:
            with st.expander("Clustering reducer", expanded=True):
                _field_label(
                    "Clustering reducer",
                    "Reducer used inside BERTopic before HDBSCAN clustering.",
                )
                topic_cluster_reduce_method = st.selectbox(
                    "Clustering reducer",
                    ["umap", "pca"],
                    index=0,
                    label_visibility="collapsed",
                    key="topic_tab_cluster_reduce_method",
                )
                _field_label(
                    "Reducer dimensions",
                    "Intermediate dimensionality passed into HDBSCAN.",
                )
                max_topic_reduce_dims = 50
                if embedding_cache_ready:
                    max_topic_reduce_dims = max(
                        2,
                        min(50, int(min(cached_emb.shape[0], cached_emb.shape[1]))),
                    )
                topic_cluster_reduce_n_components = int(st.number_input(
                    "Reducer dimensions",
                    min_value=2,
                    max_value=max_topic_reduce_dims,
                    value=min(5, max_topic_reduce_dims),
                    step=1,
                    label_visibility="collapsed",
                    key="topic_tab_cluster_reduce_n_components",
                ))
                topic_umap_n_neighbors = 15
                topic_umap_min_dist = 0.0
                topic_umap_metric = "cosine"
                if topic_cluster_reduce_method == "umap":
                    _field_label(
                        "UMAP n_neighbors",
                        "Controls local vs global topic structure; higher values preserve broader neighborhoods.",
                    )
                    topic_umap_n_neighbors = int(st.number_input(
                        "Topic UMAP n_neighbors",
                        min_value=2,
                        max_value=200,
                        value=15,
                        step=1,
                        label_visibility="collapsed",
                        key="topic_tab_umap_n_neighbors",
                    ))
                    _field_label(
                        "UMAP min_dist",
                        "Lower values keep local neighborhoods tighter before clustering.",
                    )
                    topic_umap_min_dist = float(st.number_input(
                        "Topic UMAP min_dist",
                        min_value=0.0,
                        max_value=0.99,
                        value=0.0,
                        step=0.01,
                        format="%.2f",
                        label_visibility="collapsed",
                        key="topic_tab_umap_min_dist",
                    ))
                    _field_label(
                        "UMAP metric",
                        "Distance metric used for the BERTopic clustering reducer.",
                    )
                    topic_umap_metric = st.selectbox(
                        "Topic UMAP metric",
                        ["cosine", "euclidean", "manhattan"],
                        index=0,
                        label_visibility="collapsed",
                        key="topic_tab_umap_metric",
                    )

            with st.expander("HDBSCAN settings"):
                _field_label(
                    "HDBSCAN min_cluster_size",
                    "Minimum number of documents required to form a cluster.",
                )
                topic_hdbscan_min_cluster_size = int(st.number_input(
                    "HDBSCAN min_cluster_size",
                    min_value=2,
                    max_value=200,
                    value=10,
                    step=1,
                    label_visibility="collapsed",
                    key="topic_tab_hdbscan_min_cluster_size",
                ))
                _field_label(
                    "HDBSCAN min_samples",
                    "Auto uses HDBSCAN defaults; lower values usually produce more clusters.",
                )
                auto_min_samples = st.checkbox(
                    "Auto min_samples",
                    value=True,
                    key="topic_tab_auto_min_samples",
                )
                topic_hdbscan_min_samples = None
                if not auto_min_samples:
                    topic_hdbscan_min_samples = int(st.number_input(
                        "HDBSCAN min_samples",
                        min_value=1,
                        max_value=200,
                        value=5,
                        step=1,
                        label_visibility="collapsed",
                        key="topic_tab_hdbscan_min_samples",
                    ))

            with st.expander("Topic representation settings"):
                _field_label(
                    "Top words per topic",
                    "Number of c-TF-IDF terms to retain in each topic representation.",
                )
                topic_top_n_words = int(st.number_input(
                    "Top words per topic",
                    min_value=3,
                    max_value=20,
                    value=10,
                    step=1,
                    label_visibility="collapsed",
                    key="topic_tab_top_n_words",
                ))
                _field_label(
                    "N-gram range",
                    "Choose unigram-only labels or allow bigrams in topic representations.",
                )
                ngram_range_label = st.selectbox(
                    "N-gram range",
                    ["1-1", "1-2"],
                    index=1,
                    label_visibility="collapsed",
                    key="topic_tab_ngram_range",
                )
                st.caption(
                    "English topic-term cleanup is enabled by default: stopword filtering, "
                    "frequent-word downweighting, `max_df=0.9`, lowercase normalization, "
                    "unicode accent stripping, and auto `min_df` (1 for tiny corpora, otherwise 2)."
                )

            current_cluster_opts = {
                "cluster_reduce_method": topic_cluster_reduce_method,
                "cluster_reduce_n_components": int(topic_cluster_reduce_n_components),
                "umap_n_neighbors": int(topic_umap_n_neighbors) if topic_cluster_reduce_method == "umap" else None,
                "umap_min_dist": float(topic_umap_min_dist) if topic_cluster_reduce_method == "umap" else None,
                "umap_metric": topic_umap_metric if topic_cluster_reduce_method == "umap" else None,
                "hdbscan_min_cluster_size": int(topic_hdbscan_min_cluster_size),
                "hdbscan_min_samples": int(topic_hdbscan_min_samples) if topic_hdbscan_min_samples is not None else None,
                "random_state": 42,
            }
            current_topic_repr_opts = {
                "top_n_words": int(topic_top_n_words),
                "ngram_range": (1, 1) if ngram_range_label == "1-1" else (1, 2),
            }

            cluster_results_exist = (
                cached_topic_model is not None
                and cached_topic_assignments is not None
                and cached_topic_summary is not None
            )
            cluster_fit_cache_is_current = (
                embedding_cache_ready
                and cluster_results_exist
                and _opts_same(cached_cluster_opts, current_cluster_opts)
            )
            cluster_repr_cache_is_current = (
                cluster_fit_cache_is_current
                and _opts_same(cached_topic_repr_opts, current_topic_repr_opts)
            )

            discover_topics = st.button(
                "Discover clusters/topics",
                key="btn_topic_discover",
                disabled=not embedding_cache_ready,
            )
            refresh_topic_terms = st.button(
                "Refresh topic terms",
                key="btn_topic_refresh_terms",
                disabled=not (
                    cluster_fit_cache_is_current
                    and embedding_cache_ready
                ),
                help=(
                    "Rebuild c-TF-IDF topic labels with the current representation settings "
                    "without recomputing the fitted clustering reducer or HDBSCAN clusters."
                ),
            )

            if cluster_repr_cache_is_current:
                st.caption("Showing cached cluster/topic results.")
            elif cluster_fit_cache_is_current:
                st.caption(
                    "Clusters are current, but topic-term settings changed. "
                    "Press Refresh topic terms to relabel topics without reclustering."
                )
            elif cluster_results_exist and embedding_cache_ready:
                st.caption(
                    "Cached cluster/topic results exist but clustering settings changed. "
                    "Press Discover clusters/topics to refit BERTopic."
                )

        topic_error = None
        if refresh_topic_terms:
            try:
                if (
                    not embedding_cache_ready
                    or cached_topic_model is None
                    or cached_topic_assignments is None
                    or cached_topic_summary is None
                    or not _opts_same(cached_cluster_opts, current_cluster_opts)
                ):
                    raise RuntimeError(
                        "Current clusters are stale or unavailable. Recompute clusters before refreshing topic terms."
                    )
                topic_model_obj = update_topic_representation(
                    cached_topic_model,
                    cached_texts,
                    topics=cached_topic_assignments["topic_id"].tolist(),
                    top_n_words=current_topic_repr_opts["top_n_words"],
                    ngram_range=current_topic_repr_opts["ngram_range"],
                )
                refreshed_topics = np.asarray(
                    getattr(topic_model_obj, "topics_", cached_topic_assignments["topic_id"].to_numpy()),
                    dtype=int,
                )
                refreshed_probs = getattr(topic_model_obj, "probabilities_", None)
                topic_assignments_df = build_topic_assignments(
                    cached_meta,
                    cached_texts,
                    refreshed_topics,
                    refreshed_probs,
                    topic_model_obj,
                )
                topic_summary_df = summarize_topics_table(
                    topic_model_obj,
                    topic_assignments_df,
                )
                store_topic_results(
                    st.session_state,
                    topic_model=topic_model_obj,
                    topic_assignments_df=topic_assignments_df,
                    topic_summary_df=topic_summary_df,
                    cluster_opts=current_cluster_opts,
                    topic_repr_opts=current_topic_repr_opts,
                )
                st.rerun()
            except Exception as e:
                topic_error = e

        if discover_topics:
            try:
                if not embedding_cache_ready:
                    raise RuntimeError("Compute embeddings in the Embeddings tab first.")
                topic_model_obj, topic_ids, topic_probs = fit_topic_model(
                    cached_texts,
                    cached_emb,
                    cluster_reduce_method=current_cluster_opts["cluster_reduce_method"],
                    cluster_reduce_n_components=current_cluster_opts["cluster_reduce_n_components"],
                    umap_n_neighbors=current_cluster_opts["umap_n_neighbors"] or 15,
                    umap_min_dist=current_cluster_opts["umap_min_dist"] or 0.0,
                    umap_metric=current_cluster_opts["umap_metric"] or "cosine",
                    hdbscan_min_cluster_size=current_cluster_opts["hdbscan_min_cluster_size"],
                    hdbscan_min_samples=current_cluster_opts["hdbscan_min_samples"],
                    top_n_words=current_topic_repr_opts["top_n_words"],
                    ngram_range=current_topic_repr_opts["ngram_range"],
                    random_state=current_cluster_opts["random_state"],
                )
                topic_assignments_df = build_topic_assignments(
                    cached_meta,
                    cached_texts,
                    topic_ids,
                    topic_probs,
                    topic_model_obj,
                )
                topic_summary_df = summarize_topics_table(
                    topic_model_obj,
                    topic_assignments_df,
                )
                store_topic_results(
                    st.session_state,
                    topic_model=topic_model_obj,
                    topic_assignments_df=topic_assignments_df,
                    topic_summary_df=topic_summary_df,
                    cluster_opts=current_cluster_opts,
                    topic_repr_opts=current_topic_repr_opts,
                )
                st.rerun()
            except Exception as e:
                topic_error = e

        with cluster_results_col:
            if topic_error is not None:
                st.exception(topic_error)
            elif cluster_results_exist and embedding_cache_ready:
                _render_topic_outputs(cached_topic_assignments, cached_topic_summary)
            else:
                st.info("Discover clusters/topics to fit BERTopic on the current cached embeddings.")

        if cluster_results_exist and embedding_cache_ready and topic_error is None:
            available_topic_ids = [
                int(topic_id)
                for topic_id in cached_topic_summary["topic_id"].tolist()
                if int(topic_id) != -1
            ] if "topic_id" in cached_topic_summary.columns else []

            st.markdown("**BERTopic visualizations**")
            st.caption(
                "BERTopic visualizations are topic-model diagnostics and are distinct from the "
                "document projection in `Embeddings`."
            )
            st.caption(
                "The intertopic map is a topic-level 2D visualization generated by BERTopic, "
                "not the same as the document-level projection."
            )

            if not available_topic_ids:
                st.info("No non-noise topics are available to visualize yet.")
            else:
                topic_plot_reduced_embeddings = None
                if isinstance(cached_plot_df, pd.DataFrame):
                    dim_cols = [c for c in cached_plot_df.columns if c.startswith("dim_")]
                    if len(dim_cols) == 2 and len(cached_plot_df) == len(cached_texts):
                        try:
                            plot_text_ok = True
                            if "text" in cached_plot_df.columns:
                                plot_text_ok = cached_plot_df["text"].reset_index(drop=True).equals(pd.Series(cached_texts))
                            if plot_text_ok:
                                topic_plot_reduced_embeddings = cached_plot_df[dim_cols].to_numpy(dtype=float)
                        except Exception:
                            topic_plot_reduced_embeddings = None

                plot_label_to_kind = {
                    "Intertopic distance map": "intertopic_map",
                    "Topic word scores": "barchart",
                    "Topic similarity heatmap": "heatmap",
                    "Topic hierarchy": "hierarchy",
                    "Term score decline": "term_rank",
                    "Documents": "documents",
                }
                plot_control_col, plot_display_col = st.columns([1.15, 1.85])
                with plot_control_col:
                    plot_label = st.selectbox(
                        "Plot type",
                        list(plot_label_to_kind.keys()),
                        key="topic_plot_type",
                    )
                    topic_plot_kind = plot_label_to_kind[plot_label]

                    plot_topics = None
                    plot_top_n_topics = None
                    plot_use_ctfidf = None
                    plot_n_words = None
                    plot_autoscale = None
                    plot_n_clusters = None
                    plot_orientation = None
                    plot_log_scale = None
                    plot_sample = None
                    plot_hide_annotations = None
                    plot_hide_document_hover = None
                    plot_reuse_projection = False

                    supports_subset_mode = topic_plot_kind not in {"term_rank"}
                    if supports_subset_mode:
                        topic_subset_mode = st.selectbox(
                            "Topic subset",
                            ["All topics", "Top N topics", "Selected topic IDs"],
                            key="topic_plot_subset_mode",
                        )
                        if topic_subset_mode == "All topics":
                            plot_topics = None
                        elif topic_subset_mode == "Top N topics":
                            plot_top_n_topics = int(st.number_input(
                                "Top N topics",
                                min_value=1,
                                max_value=max(1, len(available_topic_ids)),
                                value=min(8, len(available_topic_ids)),
                                step=1,
                                key="topic_plot_top_n_topics",
                            ))
                        else:
                            plot_topics = st.multiselect(
                                "Selected topic IDs",
                                options=available_topic_ids,
                                default=[],
                                key="topic_plot_selected_topics",
                            )
                    else:
                        plot_topics = st.multiselect(
                            "Highlighted topic IDs",
                            options=available_topic_ids,
                            default=[],
                            key="topic_plot_term_rank_topics",
                        )
                        if not plot_topics:
                            plot_topics = None

                    if topic_plot_kind == "intertopic_map":
                        plot_use_ctfidf = st.checkbox(
                            "Use c-TF-IDF space",
                            value=False,
                            key="topic_plot_use_ctfidf_topics",
                        )
                    elif topic_plot_kind == "barchart":
                        plot_n_words = int(st.number_input(
                            "Words per topic",
                            min_value=3,
                            max_value=20,
                            value=5,
                            step=1,
                            key="topic_plot_n_words",
                        ))
                        plot_autoscale = st.checkbox(
                            "Autoscale label height",
                            value=False,
                            key="topic_plot_autoscale",
                        )
                    elif topic_plot_kind == "heatmap":
                        cluster_heatmap = st.checkbox(
                            "Order by topic clusters",
                            value=False,
                            key="topic_plot_heatmap_cluster_ordering",
                        )
                        if cluster_heatmap:
                            plot_n_clusters = int(st.number_input(
                                "Heatmap clusters",
                                min_value=1,
                                max_value=max(1, len(available_topic_ids)),
                                value=min(max(1, len(available_topic_ids)), 8),
                                step=1,
                                key="topic_plot_heatmap_n_clusters",
                            ))
                        plot_use_ctfidf = st.checkbox(
                            "Use c-TF-IDF space",
                            value=False,
                            key="topic_plot_use_ctfidf_heatmap",
                        )
                    elif topic_plot_kind == "hierarchy":
                        plot_orientation = st.selectbox(
                            "Orientation",
                            ["left", "bottom"],
                            index=0,
                            key="topic_plot_hierarchy_orientation",
                        )
                        plot_use_ctfidf = st.checkbox(
                            "Use c-TF-IDF space",
                            value=True,
                            key="topic_plot_use_ctfidf_hierarchy",
                        )
                    elif topic_plot_kind == "term_rank":
                        plot_log_scale = st.checkbox(
                            "Log scale",
                            value=False,
                            key="topic_plot_log_scale",
                        )
                    elif topic_plot_kind == "documents":
                        plot_sample = float(st.slider(
                            "Sample share per topic",
                            min_value=0.05,
                            max_value=1.0,
                            value=1.0,
                            step=0.05,
                            key="topic_plot_documents_sample",
                        ))
                        plot_hide_annotations = st.checkbox(
                            "Hide topic annotations",
                            value=False,
                            key="topic_plot_documents_hide_annotations",
                        )
                        plot_hide_document_hover = st.checkbox(
                            "Hide document hover text",
                            value=False,
                            key="topic_plot_documents_hide_hover",
                        )
                        if topic_plot_reduced_embeddings is not None:
                            plot_reuse_projection = st.checkbox(
                                "Reuse current cached 2D projection",
                                value=False,
                                key="topic_plot_documents_use_cached_projection",
                            )
                            st.caption(
                                "Off by default. When disabled, the app computes a stable 2D "
                                "PCA projection from the cached embeddings for BERTopic's "
                                "document view."
                            )
                        else:
                            st.caption(
                                "The app will compute a stable 2D PCA projection from the cached "
                                "embeddings before calling BERTopic's document view."
                            )
                    default_plot_titles = {
                        "intertopic_map": "<b>Intertopic Distance Map</b>",
                        "barchart": "<b>Topic Word Scores</b>",
                        "heatmap": "<b>Similarity Matrix</b>",
                        "hierarchy": "<b>Hierarchical Clustering</b>",
                        "term_rank": "<b>Term score decline per Topic</b>",
                        "documents": "<b>Documents and Topics</b>",
                    }
                    default_plot_sizes = {
                        "intertopic_map": (650, 650),
                        "barchart": (250, 250),
                        "heatmap": (800, 800),
                        "hierarchy": (1000, 600),
                        "term_rank": (800, 500),
                        "documents": (1200, 750),
                    }
                    default_width, default_height = default_plot_sizes[topic_plot_kind]
                    with st.expander("Advanced options"):
                        plot_title = st.text_input(
                            "Plot title",
                            value=default_plot_titles[topic_plot_kind],
                            key="topic_plot_title",
                        )
                        plot_width = int(st.number_input(
                            "Plot width",
                            min_value=200,
                            max_value=2400,
                            value=default_width,
                            step=50,
                            key="topic_plot_width",
                        ))
                        plot_height = int(st.number_input(
                            "Plot height",
                            min_value=200,
                            max_value=2400,
                            value=default_height,
                            step=50,
                            key="topic_plot_height",
                        ))

                    current_topic_plot_opts = {
                        "docs_source": "cached_texts",
                        "topics": tuple(int(topic_id) for topic_id in plot_topics) if plot_topics is not None else None,
                        "top_n_topics": int(plot_top_n_topics) if plot_top_n_topics is not None else None,
                        "use_ctfidf": plot_use_ctfidf,
                        "n_words": int(plot_n_words) if plot_n_words is not None else None,
                        "autoscale": plot_autoscale,
                        "n_clusters": int(plot_n_clusters) if plot_n_clusters is not None else None,
                        "orientation": plot_orientation,
                        "log_scale": plot_log_scale,
                        "sample": plot_sample,
                        "hide_annotations": plot_hide_annotations,
                        "hide_document_hover": plot_hide_document_hover,
                        "use_reduced_embeddings": bool(plot_reuse_projection),
                        "title": plot_title,
                        "width": int(plot_width),
                        "height": int(plot_height),
                    }
                    topic_plot_cache_is_current = (
                        cluster_repr_cache_is_current
                        and cached_topic_plot_fig is not None
                        and cached_topic_plot_kind == topic_plot_kind
                        and _opts_same(cached_topic_plot_opts, current_topic_plot_opts)
                    )

                    generate_topic_plot = st.button(
                        "Generate plot",
                        key="btn_topic_generate_plot",
                        disabled=not cluster_results_exist,
                    )

                    if topic_plot_cache_is_current:
                        st.caption("Showing cached BERTopic visualization.")
                    elif cached_topic_plot_fig is not None:
                        st.caption("BERTopic plot settings changed. Press Generate plot to refresh the visualization.")

                topic_plot_error = None
                if generate_topic_plot:
                    try:
                        plot_embeddings = None
                        plot_reduced_embeddings = None
                        plot_docs = None
                        if topic_plot_kind == "documents":
                            plot_docs = cached_texts
                            if current_topic_plot_opts["use_reduced_embeddings"]:
                                plot_reduced_embeddings = topic_plot_reduced_embeddings
                            else:
                                plot_embeddings = cached_emb
                        if plot_embeddings is not None:
                            plot_reduced_embeddings = reduce_embeddings(
                                np.asarray(plot_embeddings),
                                method="pca",
                                n_components=2,
                                random_state=1,
                            )
                            plot_embeddings = None
                        topic_plot_fig = build_topic_plot(
                            cached_topic_model,
                            topic_plot_kind,
                            docs=plot_docs,
                            topics=current_topic_plot_opts["topics"],
                            embeddings=plot_embeddings,
                            reduced_embeddings=plot_reduced_embeddings,
                            top_n_topics=current_topic_plot_opts["top_n_topics"],
                            sample=current_topic_plot_opts["sample"],
                            hide_annotations=current_topic_plot_opts["hide_annotations"],
                            hide_document_hover=current_topic_plot_opts["hide_document_hover"],
                            use_ctfidf=current_topic_plot_opts["use_ctfidf"],
                            n_words=current_topic_plot_opts["n_words"],
                            autoscale=current_topic_plot_opts["autoscale"],
                            n_clusters=current_topic_plot_opts["n_clusters"],
                            orientation=current_topic_plot_opts["orientation"],
                            log_scale=current_topic_plot_opts["log_scale"],
                            title=current_topic_plot_opts["title"],
                            width=current_topic_plot_opts["width"],
                            height=current_topic_plot_opts["height"],
                        )
                        st.session_state["topic_plot_fig"] = topic_plot_fig
                        st.session_state["topic_plot_kind"] = topic_plot_kind
                        st.session_state["topic_plot_opts"] = current_topic_plot_opts
                        st.rerun()
                    except Exception as e:
                        topic_plot_error = e

                with plot_display_col:
                    if topic_plot_error is not None:
                        st.exception(topic_plot_error)
                    elif topic_plot_cache_is_current:
                        render_cached_topic_plot(cached_topic_plot_fig)
                    else:
                        st.info("Choose a BERTopic plot and press Generate plot.")


    # ===== Tab: Modeling =====
    with tabs[4]:
        st.subheader("Predictive modeling with cross validation and permutation test")
        st.caption("Uses embeddings as features. Select a non text target column.")

        non_text_cols = [c for c in df.columns if c not in text_cols]

        def _detect_task(y: pd.Series, max_unique_for_class: int = 20) -> str:
            if y.dtype.name in {"category", "object", "bool"}:
                return "classification"
            if pd.api.types.is_integer_dtype(y) and y.nunique() <= max_unique_for_class:
                return "classification"
            if pd.api.types.is_float_dtype(y) and y.nunique() <= max_unique_for_class:
                return "classification"
            return "regression"

        classification_metrics = [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
        ]
        regression_metrics = [
            "r2",
            "mae",
            "mse",
            "rmse",
        ]

        classifier_model_options = {
            "Logistic Regression": (
                LogisticRegression(max_iter=1000, solver="lbfgs"),
                {"C": [0.1, 1.0, 10.0]},
            ),
            "Linear SVM": (
                LinearSVC(),
                {"C": [0.1, 1.0, 10.0]},
            ),
            "Random Forest": (
                RandomForestClassifier(n_estimators=200, random_state=42),
                {"n_estimators": [100, 300], "max_depth": [None, 10]},
            ),
        }
        regression_model_options = {
            "Ridge": (
                Ridge(alpha=1.0),
                {"alpha": [0.1, 1.0, 10.0]},
            ),
            "Lasso": (
                Lasso(alpha=0.1),
                {"alpha": [0.01, 0.1, 1.0]},
            ),
            "Random Forest": (
                RandomForestRegressor(n_estimators=200, random_state=42),
                {"n_estimators": [100, 300], "max_depth": [None, 10]},
            ),
        }

        def _parse_grid_values(raw: str):
            parts = [p.strip() for p in str(raw).split(",") if p.strip()]
            parsed = []
            for p in parts:
                low = p.lower()
                if low == "none":
                    parsed.append(None)
                    continue
                try:
                    if "." in p:
                        parsed.append(float(p))
                    else:
                        parsed.append(int(p))
                    continue
                except ValueError:
                    parsed.append(p)
            return parsed

        def _grid_inputs(label: str, defaults: dict):
            st.caption(f"{label} parameter grid (comma-separated values)")
            grid: dict = {}
            for param_name, default_vals in defaults.items():
                default_text = ", ".join("None" if v is None else str(v) for v in default_vals)
                _field_label(
                    f"{label}: {param_name}",
                    "Comma-separated values to try when tuning. Use None to allow default.",
                )
                raw = st.text_input(
                    f"{label}: {param_name}",
                    default_text,
                    label_visibility="collapsed",
                )
                values = _parse_grid_values(raw)
                if values:
                    grid[param_name] = values
            return grid

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.markdown("**1. Targets**")
            _field_label(
                "Target columns",
                "Columns you want to predict. You can use auto task detection or force a task type.",
            )
            target_cols = st.multiselect(
                "Target columns",
                non_text_cols,
                default=[c for c in non_text_cols if c.lower().startswith("target")][:1],
                label_visibility="collapsed",
            )
            group_options = ["None"] + non_text_cols
            _field_label(
                "Group column (optional)",
                "Keeps rows with the same ID in the same fold using GroupKFold (stratified for classification when available).",
            )
            group_choice = st.selectbox(
                "Group column (optional)",
                group_options,
                index=0,
                disabled=len(non_text_cols) == 0,
                label_visibility="collapsed",
            )
            group_col = None if group_choice == "None" else group_choice

            _field_label(
                "Task mode",
                "Auto infers each target task from dtype and unique-value count. You can also force one task globally.",
            )
            task_mode_options = {
                "Auto detect per target": "auto",
                "Force classification": "classification",
                "Force regression": "regression",
            }
            task_mode_label = st.selectbox(
                "Task mode",
                list(task_mode_options.keys()),
                index=0,
                label_visibility="collapsed",
            )
            task_mode = task_mode_options[task_mode_label]

            max_unique_for_class = 20
            if task_mode == "auto":
                _field_label(
                    "Auto-detect threshold (numeric unique values)",
                    "Numeric targets with unique-value count less than or equal to this threshold are treated as classification.",
                )
                max_unique_for_class = int(
                    st.number_input(
                        "Auto-detect threshold (numeric unique values)",
                        min_value=1,
                        max_value=200,
                        value=20,
                        step=1,
                        label_visibility="collapsed",
                    )
                )

            target_task_overrides: dict[str, str] = {}
            target_tasks: dict[str, str] = {}
            if target_cols:
                auto_detected_tasks = {
                    tgt: _detect_task(df[tgt].dropna(), max_unique_for_class=max_unique_for_class)
                    for tgt in target_cols
                }
                with st.expander("Per-target task overrides (optional)"):
                    st.caption("Set explicit task types for selected targets. Leave as Auto to use the global task mode.")
                    for tgt in target_cols:
                        detected = auto_detected_tasks[tgt]
                        override_choice = st.selectbox(
                            f"Task for {tgt}",
                            [
                                f"Auto ({detected})",
                                "Classification",
                                "Regression",
                            ],
                            index=0,
                            key=f"task_override__{tgt}",
                        )
                        if override_choice == "Classification":
                            target_task_overrides[str(tgt)] = "classification"
                        elif override_choice == "Regression":
                            target_task_overrides[str(tgt)] = "regression"

                for tgt in target_cols:
                    if str(tgt) in target_task_overrides:
                        target_tasks[tgt] = target_task_overrides[str(tgt)]
                    elif task_mode != "auto":
                        target_tasks[tgt] = task_mode
                    else:
                        target_tasks[tgt] = auto_detected_tasks[tgt]
                unique_tasks = sorted(set(target_tasks.values()))
            else:
                target_tasks = {}
                unique_tasks = []

            st.markdown("**2. Embeddings**")
            modeling_model_options = EMBEDDING_MODEL_OPTIONS + ["Custom"]
            embed_opts = st.session_state.get("embed_opts") if "embed_opts" in st.session_state else None
            embed_default = embed_opts.get("model_name") if isinstance(embed_opts, dict) else None
            prev_embed_default = st.session_state.get("modeling_embed_default")
            current_choice = st.session_state.get("modeling_model_choice")
            current_custom = st.session_state.get("modeling_custom_model")
            if embed_default:
                should_sync = False
                if current_choice is None:
                    should_sync = True
                elif current_choice == prev_embed_default:
                    should_sync = True
                elif current_choice == "Custom" and prev_embed_default and current_custom == prev_embed_default:
                    should_sync = True

                if should_sync:
                    if embed_default in modeling_model_options:
                        st.session_state["modeling_model_choice"] = embed_default
                    else:
                        st.session_state["modeling_model_choice"] = "Custom"
                        st.session_state["modeling_custom_model"] = embed_default
                st.session_state["modeling_embed_default"] = embed_default

            _field_label(
                "SentenceTransformer model for modeling",
                "Select the text embedding model. Larger models can capture more nuance but run slower.",
            )
            modeling_choice = st.selectbox(
                "SentenceTransformer model for modeling",
                modeling_model_options,
                index=(
                    modeling_model_options.index(embed_default)
                    if embed_default in modeling_model_options
                    else modeling_model_options.index("Custom")
                    if embed_default
                    else 0
                ),
                key="modeling_model_choice",
                label_visibility="collapsed",
            )
            if modeling_choice == "Custom":
                _field_label(
                    "Custom modeling SentenceTransformer model",
                    "Enter a Hugging Face model name or a local path.",
                )
                custom_modeling_model = st.text_input(
                    "Custom modeling SentenceTransformer model",
                    st.session_state.get("modeling_custom_model", embed_default or EMBEDDING_MODEL_OPTIONS[0]),
                    key="modeling_custom_model",
                    label_visibility="collapsed",
                ).strip()
                model_name_modeling = custom_modeling_model or EMBEDDING_MODEL_OPTIONS[0]
            else:
                model_name_modeling = modeling_choice
            reuse_embed = st.checkbox(
                "Reuse cached embeddings if available",
                value=True,
                help="Embeddings are numeric versions of your text. Reuse to save time if text and model are unchanged.",
            )
            text_feature_mode = "average"
            if len(text_cols) > 1:
                text_feature_mode_options = {
                    "Average selected text columns per row": "average",
                    "Run separate models per text column": "separate",
                }
                _field_label(
                    "How to use multiple text columns",
                    "Average = mean-pool embeddings across selected text columns for each row. Separate = run one independent model set per text column.",
                )
                text_feature_mode_label = st.selectbox(
                    "How to use multiple text columns",
                    list(text_feature_mode_options.keys()),
                    index=0,
                    key="modeling_text_feature_mode",
                    label_visibility="collapsed",
                )
                text_feature_mode = text_feature_mode_options[text_feature_mode_label]
            elif len(text_cols) == 1:
                st.caption("One text column selected, so modeling uses that column directly.")

            st.markdown("**3. Models**")
            show_classifier = (not unique_tasks) or ("classification" in unique_tasks)
            show_regressor = (not unique_tasks) or ("regression" in unique_tasks)
            if len(unique_tasks) > 1:
                st.caption("Mixed target types detected; configure both classifier and regressor settings.")

            classifier_model_name = None
            regressor_model_name = None
            if show_classifier and show_regressor:
                model_col_a, model_col_b = st.columns(2)
                with model_col_a:
                    _field_label(
                        "Classifier model",
                        "Model for predicting categories. Simple models are faster; complex models can capture non-linear patterns.",
                    )
                    classifier_model_name = st.selectbox(
                        "Classifier model",
                        list(classifier_model_options.keys()),
                        index=0,
                        label_visibility="collapsed",
                    )
                with model_col_b:
                    _field_label(
                        "Regressor model",
                        "Model for predicting numeric values. Simple models are faster; complex models can capture non-linear patterns.",
                    )
                    regressor_model_name = st.selectbox(
                        "Regressor model",
                        list(regression_model_options.keys()),
                        index=0,
                        label_visibility="collapsed",
                    )
            elif show_classifier:
                _field_label(
                    "Classifier model",
                    "Model for predicting categories. Simple models are faster; complex models can capture non-linear patterns.",
                )
                classifier_model_name = st.selectbox(
                    "Classifier model",
                    list(classifier_model_options.keys()),
                    index=0,
                    label_visibility="collapsed",
                )
            elif show_regressor:
                _field_label(
                    "Regressor model",
                    "Model for predicting numeric values. Simple models are faster; complex models can capture non-linear patterns.",
                )
                regressor_model_name = st.selectbox(
                    "Regressor model",
                    list(regression_model_options.keys()),
                    index=0,
                    label_visibility="collapsed",
                )

            feature_selection_choice = "None"
            reduction_choice = "None"
            model_reduce_method = None
            model_reduce_n_components = None

        if target_cols:
            if len(unique_tasks) == 1:
                metric_options = classification_metrics if unique_tasks[0] == "classification" else regression_metrics
            else:
                metric_options = classification_metrics + regression_metrics
        else:
            metric_options = []

        with right_col:
            st.markdown("**4. Evaluation**")
            run_permutation = True

            _field_label(
                "CV folds",
                "How many train/test splits to average over. Higher is more stable but slower.",
            )
            cv_folds = st.number_input(
                "CV folds",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                label_visibility="collapsed",
            )
            n_perm = 500
            if run_permutation:
                _field_label(
                    "Permutation iterations",
                    "Times to shuffle the target to estimate chance performance. Higher is more reliable but slower.",
                )
                n_perm = st.number_input(
                    "Permutation iterations",
                    min_value=50,
                    max_value=2000,
                    value=500,
                    step=50,
                    label_visibility="collapsed",
                )
            if target_cols and len(unique_tasks) > 1:
                st.caption("Mixed target types detected; showing both classification and regression metrics.")

            _field_label(
                "Additional metrics to report",
                "Extra evaluation scores to include in the report. Does not change training.",
            )
            report_metrics = st.multiselect(
                "Additional metrics to report",
                metric_options,
                default=["accuracy"] if "accuracy" in metric_options else (metric_options[:1] if metric_options else []),
                label_visibility="collapsed",
            )
            correction_choice = "None"
            correction_method = None
            correction_scope = "all_tests"
            if run_permutation:
                _field_label(
                    "Multiple comparisons correction",
                    "Adjust p-values when testing multiple targets and/or multiple feature sets.",
                )
                correction_options = {
                    "FDR (Benjamini–Hochberg)": "fdr_bh",
                    "Bonferroni": "bonferroni",
                    "Holm": "holm",
                    "None": None,
                }
                correction_choice = st.selectbox(
                    "Multiple comparisons correction",
                    list(correction_options.keys()),
                    index=0,
                    label_visibility="collapsed",
                )
                correction_method = correction_options[correction_choice]
                if text_feature_mode == "separate" and len(text_cols) > 1 and len(target_cols) > 1:
                    correction_scope_options = {
                        "Across all tests (targets x feature sets)": "all_tests",
                        "Within each target across feature sets": "within_target",
                        "Within each feature set across targets": "within_feature_set",
                    }
                    _field_label(
                        "Correction scope",
                        "Choose which family of hypothesis tests should be corrected together when separate feature sets are modeled.",
                    )
                    correction_scope_label = st.selectbox(
                        "Correction scope",
                        list(correction_scope_options.keys()),
                        index=0,
                        label_visibility="collapsed",
                    )
                    correction_scope = correction_scope_options[correction_scope_label]

            tune_hyperparams = False
            k_best = 0
            percentile = 20
            variance_threshold = 0.0
            classifier_param_grid = {}
            regressor_param_grid = {}
            model_reduce_components_grid = None
            k_best_grid = None
            percentile_grid = None
            variance_threshold_grid = None

            with st.expander("Advanced (feature selection, dimensionality reduction, tuning)"):
                st.markdown("**Feature selection**")
                _field_label(
                    "Feature selection",
                    "Optionally keep only the most informative features. None uses all features.",
                )
                feature_selection_choice = st.selectbox(
                    "Feature selection",
                    ["None", "KBest", "Percentile", "Variance threshold"],
                    index=0,
                    label_visibility="collapsed",
                )

                if feature_selection_choice == "KBest":
                    _field_label(
                        "Top features (K)",
                        "Keep the top K most informative features.",
                    )
                    k_best = st.number_input(
                        "Top features (K)",
                        min_value=5,
                        max_value=2000,
                        value=200,
                        step=25,
                        label_visibility="collapsed",
                    )
                    percentile = 20
                    variance_threshold = 0.0
                elif feature_selection_choice == "Percentile":
                    _field_label(
                        "Top percentile",
                        "Keep the top X percent of features by importance.",
                    )
                    percentile = st.number_input(
                        "Top percentile",
                        min_value=1,
                        max_value=100,
                        value=20,
                        step=5,
                        label_visibility="collapsed",
                    )
                    k_best = 0
                    variance_threshold = 0.0
                elif feature_selection_choice == "Variance threshold":
                    _field_label(
                        "Variance threshold",
                        "Remove features that barely change across samples.",
                    )
                    variance_threshold = st.number_input(
                        "Variance threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.01,
                        label_visibility="collapsed",
                    )
                    k_best = 0
                    percentile = 20
                else:
                    k_best = 0
                    percentile = 20
                    variance_threshold = 0.0

                st.markdown("**Dimensionality reduction**")
                _field_label(
                    "Reduce embeddings within CV",
                    "Optionally compress embeddings inside cross-validation to avoid leakage.",
                )
                reduction_choice = st.selectbox(
                    "Reduce embeddings within CV",
                    ["None", "PCA"],
                    index=0,
                    label_visibility="collapsed",
                )
                if reduction_choice == "PCA":
                    _field_label(
                        "PCA components",
                        "Number of dimensions after compression.",
                    )
                    model_reduce_n_components = int(
                        st.number_input(
                            "PCA components",
                            min_value=2,
                            max_value=1000,
                            value=50,
                            step=1,
                            label_visibility="collapsed",
                        )
                    )
                    model_reduce_method = "pca"
                else:
                    model_reduce_method = None
                    model_reduce_n_components = None

                _field_label(
                    "Tune hyperparameters (light grid)",
                    "Try a small grid of settings. This can improve results but increases runtime.",
                )
                tune_hyperparams = st.checkbox(
                    "Tune hyperparameters (light grid)",
                    value=False,
                    label_visibility="collapsed",
                )
                if tune_hyperparams:
                    st.markdown("**Dimensionality reduction tuning**")
                    if model_reduce_method == "pca":
                        _field_label(
                            "PCA components grid (optional)",
                            "Comma-separated values to try, e.g., 20, 50, 100.",
                        )
                        comps_raw = st.text_input(
                            "PCA components grid (optional)",
                            "",
                            label_visibility="collapsed",
                        )
                        model_reduce_components_grid = _parse_grid_values(comps_raw) if comps_raw.strip() else None

                    st.markdown("**Feature selection ranges**")
                    if feature_selection_choice == "KBest":
                        _field_label(
                            "K values grid (optional)",
                            "Comma-separated values to try, e.g., 100, 200, 300.",
                        )
                        k_raw = st.text_input(
                            "K values grid (optional)",
                            "",
                            label_visibility="collapsed",
                        )
                        k_best_grid = _parse_grid_values(k_raw) if k_raw.strip() else None
                    elif feature_selection_choice == "Percentile":
                        _field_label(
                            "Percentile grid (optional)",
                            "Comma-separated values to try, e.g., 10, 20, 30.",
                        )
                        p_raw = st.text_input(
                            "Percentile grid (optional)",
                            "",
                            label_visibility="collapsed",
                        )
                        percentile_grid = _parse_grid_values(p_raw) if p_raw.strip() else None
                    elif feature_selection_choice == "Variance threshold":
                        _field_label(
                            "Variance threshold grid (optional)",
                            "Comma-separated values to try, e.g., 0.0, 0.01, 0.05.",
                        )
                        v_raw = st.text_input(
                            "Variance threshold grid (optional)",
                            "",
                            label_visibility="collapsed",
                        )
                        variance_threshold_grid = _parse_grid_values(v_raw) if v_raw.strip() else None

                    if show_classifier and classifier_model_name:
                        classifier_param_grid = _grid_inputs(
                            "Classifier",
                            classifier_model_options[classifier_model_name][1],
                        )
                    if show_regressor and regressor_model_name:
                        regressor_param_grid = _grid_inputs(
                            "Regressor",
                            regression_model_options[regressor_model_name][1],
                        )

        model_opts_current = {
            "target_cols": tuple(target_cols),
            "target_tasks": tuple(sorted((str(k), str(v)) for k, v in target_tasks.items())),
            "task_mode": task_mode,
            "max_unique_for_class": int(max_unique_for_class),
            "target_task_overrides": tuple(sorted(target_task_overrides.items())),
            "group_col": group_col,
            "cv_folds": int(cv_folds),
            "n_perm": int(n_perm),
            "text_cols": tuple(text_cols),
            "text_feature_mode": text_feature_mode,
            "reuse_embed": bool(reuse_embed),
            "model_name_modeling": model_name_modeling,
            "report_metrics": tuple(report_metrics),
            "correction_method": correction_choice,
            "correction_scope": correction_scope,
            "classifier_model_name": classifier_model_name,
            "regressor_model_name": regressor_model_name,
            "classifier_param_grid": classifier_param_grid,
            "regressor_param_grid": regressor_param_grid,
            "tune_hyperparams": bool(tune_hyperparams),
            "feature_selection": feature_selection_choice,
            "k_best": int(k_best),
            "percentile": int(percentile),
            "variance_threshold": float(variance_threshold),
            "reduce_method": model_reduce_method,
            "reduce_n_components": int(model_reduce_n_components) if model_reduce_n_components is not None else None,
            "reduce_components_grid": tuple(model_reduce_components_grid) if model_reduce_components_grid else None,
            "k_best_grid": tuple(k_best_grid) if k_best_grid else None,
            "percentile_grid": tuple(percentile_grid) if percentile_grid else None,
            "variance_threshold_grid": tuple(variance_threshold_grid) if variance_threshold_grid else None,
        }

        cached_results = st.session_state.get("results_df")
        cached_preds = st.session_state.get("preds")
        cached_model_opts = st.session_state.get("model_opts")

        cache_fresh = cached_results is not None and cached_preds is not None and cached_model_opts == model_opts_current

        with left_col:
            model_button_label = "Run modeling" if cached_results is None else "Recompute modeling"
            run_model = st.button(model_button_label, key="btn_model_compute")

            if cache_fresh:
                st.caption("Showing cached results")
            elif cached_results is not None:
                st.caption("Cached results exist but options changed. Press Recompute modeling to refresh.")

        results_container = st.container()

        plot_style_col, plot_out_col = st.columns([1, 3])
        with plot_style_col:
            st.markdown("**Permutation plot styling**")
            try:
                import inspect

                if "height" in inspect.signature(st.container).parameters:
                    style_container = st.container(height=560, border=True)
                else:
                    style_container = st.container()
            except Exception:
                style_container = st.container()

            with style_container:
                plot_cols = st.selectbox("Grid columns", [1, 2, 3], index=1, key="perm_plot_cols")
                show_titles = st.checkbox("Show subplot titles", value=True, key="perm_show_titles")
                show_legend = st.checkbox("Show legend", value=False, key="perm_show_legend")
                show_observed = st.checkbox("Show observed score line", value=True, key="perm_show_observed")
                nbins = st.slider(
                    "Histogram bins",
                    min_value=10,
                    max_value=80,
                    value=30,
                    step=5,
                    key="perm_nbins",
                )
                bar_opacity = st.slider(
                    "Bar opacity",
                    min_value=0.2,
                    max_value=1.0,
                    value=0.8,
                    step=0.05,
                    key="perm_bar_opacity",
                )
                color_mode = st.selectbox(
                    "Bar color mode",
                    ["Palette", "Single color"],
                    index=0,
                    key="perm_color_mode",
                )

                palette_options = {
                    "Plotly": px.colors.qualitative.Plotly,
                    "D3": px.colors.qualitative.D3,
                    "Set2": px.colors.qualitative.Set2,
                    "Dark24": px.colors.qualitative.Dark24,
                    "Pastel1": px.colors.qualitative.Pastel1,
                }
                if color_mode == "Single color":
                    bar_color = st.color_picker("Bar color", "#4c78a8", key="perm_bar_color")
                    palette = None
                else:
                    palette_choice = st.selectbox(
                        "Palette",
                        list(palette_options.keys()),
                        index=0,
                        key="perm_palette_choice",
                    )
                    palette = palette_options[palette_choice]
                    bar_color = None

                line_color = st.color_picker(
                    "Observed line color",
                    "#d62728",
                    key="perm_observed_line_color",
                )
                line_width = st.slider(
                    "Observed line width",
                    min_value=1,
                    max_value=6,
                    value=2,
                    step=1,
                    key="perm_observed_line_width",
                )
                height_per_row = st.slider(
                    "Height per row (px)",
                    min_value=240,
                    max_value=700,
                    value=320,
                    step=20,
                    key="perm_height_per_row",
                )
                x_axis_label = st.text_input("X-axis label", "Score", key="perm_x_axis_label")
                y_axis_label = st.text_input("Y-axis label", "Count", key="perm_y_axis_label")
                figure_title = st.text_input("Figure title (optional)", "", key="perm_figure_title")
                figure_title_align = st.selectbox(
                    "Figure title alignment",
                    ["Center", "Left"],
                    index=0,
                    key="perm_figure_title_align",
                )

                target_label_map: dict[str, str] = {}
                target_bar_colors: dict[str, str] = {}
                target_line_colors: dict[str, str] = {}

                style_targets: list[str] = []
                style_results = st.session_state.get("results_df")
                if isinstance(style_results, pd.DataFrame) and len(style_results) and "target" in style_results.columns:
                    style_targets = [str(t) for t in style_results["target"].astype(str).tolist()]
                elif target_cols:
                    style_targets = [str(t) for t in target_cols]
                style_targets = list(dict.fromkeys(style_targets))

                per_target_custom = st.checkbox(
                    "Customize each histogram (labels and colors)",
                    value=False,
                    disabled=not style_targets,
                    key="perm_per_target_custom",
                )
                if not style_targets:
                    st.caption("Run modeling (or select targets) to enable per-target overrides.")

                if per_target_custom and style_targets:
                    with st.expander("Per-target overrides", expanded=False):
                        st.caption("Set custom subplot labels and colors per target.")
                        for i, tgt in enumerate(style_targets):
                            key_suffix = f"{i}_{''.join(ch if ch.isalnum() else '_' for ch in tgt)}"
                            default_hist_color = "#4c78a8"
                            if isinstance(bar_color, str) and bar_color.strip():
                                default_hist_color = bar_color
                            elif palette:
                                candidate = palette[i % len(palette)]
                                if isinstance(candidate, str) and candidate.startswith("#"):
                                    default_hist_color = candidate

                            target_label = st.text_input(
                                f"Label for {tgt}",
                                value=tgt,
                                key=f"perm_target_label_{key_suffix}",
                            )
                            target_hist_color = st.color_picker(
                                f"Histogram color for {tgt}",
                                value=default_hist_color,
                                key=f"perm_target_hist_color_{key_suffix}",
                            )
                            target_obs_color = st.color_picker(
                                f"Observed line color for {tgt}",
                                value=line_color,
                                key=f"perm_target_line_color_{key_suffix}",
                            )

                            if isinstance(target_label, str) and target_label.strip() and target_label.strip() != tgt:
                                target_label_map[tgt] = target_label.strip()
                            target_bar_colors[tgt] = target_hist_color
                            target_line_colors[tgt] = target_obs_color

                theme_options = {
                    "Plotly White": "plotly_white",
                    "Plotly Dark": "plotly_dark",
                    "Simple White": "simple_white",
                    "GGPlot2": "ggplot2",
                    "Seaborn": "seaborn",
                    "Presentation": "presentation",
                    "Default (Plotly)": None,
                }
                theme_choice = st.selectbox(
                    "Plot theme",
                    list(theme_options.keys()),
                    index=0,
                    key="perm_plot_theme",
                )
                theme_value = theme_options[theme_choice]

        perm_plot_style = {
            "grid_cols": int(plot_cols),
            "show_titles": bool(show_titles),
            "show_legend": bool(show_legend),
            "show_observed": bool(show_observed),
            "nbins": int(nbins),
            "bar_opacity": float(bar_opacity),
            "bar_color": bar_color,
            "palette": palette,
            "line_color": line_color,
            "line_width": int(line_width),
            "height_per_row": int(height_per_row),
            "template": theme_value,
            "x_axis_label": x_axis_label,
            "y_axis_label": y_axis_label,
            "figure_title": figure_title.strip() or None,
            "figure_title_align": figure_title_align.lower(),
            "target_label_map": target_label_map,
            "target_bar_colors": target_bar_colors,
            "target_line_colors": target_line_colors,
        }

        def _render_model_table(results_df: pd.DataFrame, model_config=None):
            try:
                if "cv_folds_used" in results_df.columns and len(results_df):
                    requested_folds = int(cv_folds)
                    reduced = results_df[results_df["cv_folds_used"] < requested_folds]
                    if len(reduced):
                        targets = ", ".join(reduced["target"].astype(str).tolist())
                        st.warning(
                            f"CV folds reduced from {requested_folds} due to limited data for: {targets}"
                        )
            except Exception:
                pass
            display_df = results_df.copy()
            display_df = display_df.drop(
                columns=["cv_scores", "perm_scores", "target_name"],
                errors="ignore",
            )
            if "p_adjusted" in display_df.columns or (
                "p_adjust_method" in display_df.columns and display_df["p_adjust_method"].notna().any()
            ):
                display_df = display_df.drop(columns=["p_fdr"], errors="ignore")
            if "p_adjust_method" in display_df.columns and display_df["p_adjust_method"].isna().all():
                display_df = display_df.drop(columns=["p_adjust_method"])
            if "p_adjust_scope" in display_df.columns and display_df["p_adjust_scope"].isna().all():
                display_df = display_df.drop(columns=["p_adjust_scope"])
            st.write("Results")
            st.dataframe(display_df, width='stretch')
            df_to_csv_download(results_df, "model_results.csv")
            if "observed" in results_df.columns and results_df["observed"].isna().any():
                st.warning(
                    "Some targets have undefined primary scores (often R2 with too-small test folds). "
                    "Reduce CV folds or use more samples."
                )
            if len(results_df):
                st.markdown("**Interpretation per target**")
                include_adjusted_in_writeup = len(results_df) > 1
                expand_single = len(results_df) == 1
                for idx, (_, row) in enumerate(results_df.iterrows(), start=1):
                    tgt = str(row.get("target", "")).strip() or f"target_{idx}"
                    bullets = interpret_model_row(row, model_config=model_config)
                    summary_sentence = summarize_model_row(
                        row,
                        include_adjusted=include_adjusted_in_writeup,
                        model_config=model_config,
                    )
                    with st.expander(f"{idx}. {tgt}", expanded=expand_single):
                        for b in bullets:
                            st.markdown(f"- {b}")
                        st.markdown(f"*Example write-up sentence: {summary_sentence}*")

        def _render_perm_plots(results_df: pd.DataFrame, style: dict):
            if len(results_df):
                perm_rows = []
                for _, row in results_df.iterrows():
                    tgt = row.get("target", "target")
                    perm_scores = row.get("perm_scores", [])
                    observed = float(row.get("observed", 0.0))
                    if isinstance(perm_scores, tuple):
                        perm_scores = list(perm_scores)
                    elif isinstance(perm_scores, np.ndarray):
                        perm_scores = perm_scores.tolist()
                    if isinstance(perm_scores, list) and len(perm_scores):
                        perm_rows.append((tgt, perm_scores, observed))

                if perm_rows:
                    target_label_map = style.get("target_label_map") or {}
                    target_bar_colors = style.get("target_bar_colors") or {}
                    target_line_colors = style.get("target_line_colors") or {}

                    def _target_label(target_name) -> str:
                        key = str(target_name)
                        val = target_label_map.get(key)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
                        return key

                    n_plots = len(perm_rows)
                    grid_cols = max(1, min(style.get("grid_cols", 2), n_plots))
                    grid_rows = int(math.ceil(n_plots / grid_cols))
                    show_titles = style.get("show_titles", True)
                    subplot_titles = [_target_label(t[0]) for t in perm_rows] if show_titles else None

                    fig = make_subplots(
                        rows=grid_rows,
                        cols=grid_cols,
                        subplot_titles=subplot_titles,
                    )

                    palette = style.get("palette") or []
                    bar_color = style.get("bar_color")
                    nbins = int(style.get("nbins", 30))
                    bar_opacity = float(style.get("bar_opacity", 0.8))
                    show_legend = bool(style.get("show_legend", False))
                    show_observed = bool(style.get("show_observed", True))
                    line_color = style.get("line_color", "#d62728")
                    line_width = int(style.get("line_width", 2))

                    for i, (tgt, perm_scores, observed) in enumerate(perm_rows):
                        row_idx = (i // grid_cols) + 1
                        col_idx = (i % grid_cols) + 1
                        target_key = str(tgt)
                        display_name = _target_label(target_key)
                        target_override_color = target_bar_colors.get(target_key)
                        if target_override_color:
                            color = target_override_color
                        elif bar_color:
                            color = bar_color
                        elif palette:
                            color = palette[i % len(palette)]
                        else:
                            color = None

                        fig.add_trace(
                            go.Histogram(
                                x=perm_scores,
                                nbinsx=nbins,
                                marker_color=color,
                                opacity=bar_opacity,
                                name=display_name,
                                showlegend=show_legend,
                            ),
                            row=row_idx,
                            col=col_idx,
                        )
                        if show_observed:
                            line_color_for_target = target_line_colors.get(target_key, line_color)
                            fig.add_vline(
                                x=observed,
                                line_color=line_color_for_target,
                                line_width=line_width,
                                row=row_idx,
                                col=col_idx,
                            )

                    template = style.get("template")
                    if template:
                        fig.update_layout(template=template)
                    layout_kwargs = {
                        "showlegend": show_legend,
                        "bargap": 0.1,
                        "height": int(style.get("height_per_row", 320)) * grid_rows + 60,
                    }
                    figure_title = style.get("figure_title")
                    title_align = str(style.get("figure_title_align", "center")).lower()
                    if isinstance(figure_title, str) and figure_title.strip():
                        if title_align == "left":
                            layout_kwargs["title"] = {
                                "text": figure_title.strip(),
                                "x": 0.0,
                                "xanchor": "left",
                                "xref": "paper",
                            }
                        else:
                            layout_kwargs["title"] = {
                                "text": figure_title.strip(),
                                "x": 0.5,
                                "xanchor": "center",
                                "xref": "paper",
                            }
                    fig.update_layout(**layout_kwargs)
                    fig.update_xaxes(title_text=str(style.get("x_axis_label", "Score")))
                    fig.update_yaxes(title_text=str(style.get("y_axis_label", "Count")))

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displaylogo": False, "toImageButtonOptions": {"format": "png"}},
                    )
                else:
                    st.info("Permutation plots will appear here after modeling.")

        results_df_current = None
        results_notice = None
        results_error = None

        if run_model:
            if not target_cols:
                results_notice = ("warning", "Select at least one target column.")
            else:
                try:
                    if reuse_embed and st.session_state.get("embeddings") is not None and st.session_state.get("meta_df") is not None:
                        meta_df = st.session_state["meta_df"]
                        emb = st.session_state["embeddings"]
                    else:
                        meta_df, emb, texts = embed_text_columns_simple_ui(
                            [df[c] for c in text_cols],
                            model_name=model_name_modeling,
                            normalize=True
                        )
                    X = pd.DataFrame(emb, index=pd.MultiIndex.from_frame(meta_df[["source_column", "index"]]))
                    Y = df[target_cols]
                    group_series = df[group_col] if group_col else None

                    classifier_model = classifier_model_options[classifier_model_name][0] if classifier_model_name else None
                    regressor_model = regression_model_options[regressor_model_name][0] if regressor_model_name else None
                    model_kwargs_base = {
                        "Y": Y,
                        "cv": int(cv_folds),
                        "n_permutations": int(n_perm),
                        "run_permutation": bool(run_permutation),
                        "random_state": 42,
                        "max_unique_for_class": int(max_unique_for_class),
                        "scale_X": True,
                        "report_metrics": report_metrics,
                        "correction_method": correction_method,
                        "tune_hyperparams": bool(tune_hyperparams),
                        "classifier_model": classifier_model,
                        "regressor_model": regressor_model,
                        "classifier_param_grid": classifier_param_grid,
                        "regressor_param_grid": regressor_param_grid,
                        "feature_selection": (
                            "kbest"
                            if feature_selection_choice == "KBest"
                            else "percentile"
                            if feature_selection_choice == "Percentile"
                            else "variance"
                            if feature_selection_choice == "Variance threshold"
                            else None
                        ),
                        "k_best": int(k_best),
                        "percentile": int(percentile),
                        "variance_threshold": float(variance_threshold),
                        "reduce_method": model_reduce_method,
                        "reduce_n_components": model_reduce_n_components,
                        "reduce_components_grid": model_reduce_components_grid,
                        "k_best_grid": k_best_grid,
                        "percentile_grid": percentile_grid,
                        "variance_threshold_grid": variance_threshold_grid,
                        "groups": group_series,
                        "task_mode": task_mode,
                        "target_task_overrides": target_task_overrides,
                    }
                    supported_params = set(inspect.signature(auto_cv_with_permutation).parameters)
                    if text_feature_mode == "separate" and len(text_cols) > 1:
                        results_parts = []
                        preds = {}
                        available_sources = set(meta_df["source_column"].astype(str).tolist())
                        for source in text_cols:
                            source_name = str(source)
                            if source_name not in available_sources:
                                continue
                            try:
                                X_source = X.xs(source_name, level=0, drop_level=True)
                            except KeyError:
                                continue
                            model_kwargs = {
                                "X": X_source,
                                **model_kwargs_base,
                                "correction_method": None,
                            }
                            filtered_model_kwargs = {
                                k: v for k, v in model_kwargs.items() if k in supported_params
                            }
                            source_results_df, source_preds = auto_cv_with_permutation(**filtered_model_kwargs)
                            if len(source_results_df):
                                source_results_df = source_results_df.copy()
                                source_results_df["feature_set"] = source_name
                                source_results_df["target_name"] = source_results_df["target"].astype(str)
                                source_results_df["target"] = source_results_df["target"].map(
                                    lambda target_name: f"{source_name} :: {target_name}"
                                )
                                results_parts.append(source_results_df)
                            for target_name, pred_vals in source_preds.items():
                                preds[f"{source_name} :: {target_name}"] = pred_vals
                        if results_parts:
                            results_df = pd.concat(results_parts, ignore_index=True, sort=False)
                            correction_group_cols = None
                            if correction_scope == "within_target":
                                correction_group_cols = ["target_name"]
                            elif correction_scope == "within_feature_set":
                                correction_group_cols = ["feature_set"]
                            results_df = apply_multiple_comparisons_correction(
                                results_df,
                                correction_method=correction_method,
                                group_cols=correction_group_cols,
                                scope_label=correction_scope,
                            )
                        else:
                            results_df = pd.DataFrame()
                    else:
                        X_by_row = X.groupby(level=1).mean()
                        model_kwargs = {
                            "X": X_by_row,
                            **model_kwargs_base,
                        }
                        filtered_model_kwargs = {
                            k: v for k, v in model_kwargs.items() if k in supported_params
                        }
                        results_df, preds = auto_cv_with_permutation(**filtered_model_kwargs)

                    st.session_state["results_df"] = results_df
                    st.session_state["preds"] = preds
                    st.session_state["model_opts"] = model_opts_current
                    results_df_current = results_df
                except Exception as e:
                    results_error = e
        else:
            if cached_results is None:
                results_notice = ("info", "Press Run modeling to compute results.")
            else:
                results_df_current = cached_results

        with results_container:
            if results_error is not None:
                st.exception(results_error)
            elif results_notice is not None:
                level, msg = results_notice
                if level == "warning":
                    st.warning(msg)
                else:
                    st.info(msg)
            elif results_df_current is not None:
                _render_model_table(results_df_current, model_config=st.session_state.get("model_opts"))

        with plot_out_col:
            if results_df_current is not None:
                _render_perm_plots(results_df_current, perm_plot_style)
            elif results_error is None:
                st.info("Permutation plots will appear here after modeling.")


    # ===== Tab: Report =====
    with tabs[5]:
        st.subheader("Session report")
        st.caption("Summarizes what you ran and adds brief interpretations.")

        include_stats = st.checkbox("Include descriptive stats", value=st.session_state.get("stats_df") is not None)
        include_embed = st.checkbox("Include embeddings and projection", value=st.session_state.get("plot_df") is not None)
        include_model = st.checkbox("Include modeling", value=st.session_state.get("results_df") is not None)

        fmt = st.radio("Format", ["Markdown", "HTML"], horizontal=True, key="radio_report_format")
        assemble = st.button("Assemble report", key="btn_report_assemble")

        if assemble:
            stats_df = st.session_state.get("stats_df") if include_stats else None
            overall = st.session_state.get("overall") if include_stats else None
            plot_df = st.session_state.get("plot_df") if include_embed else None
            results_df = st.session_state.get("results_df") if include_model else None
            stats_opts = st.session_state.get("stats_opts") if include_stats else None
            embed_opts = st.session_state.get("embed_opts") if include_embed else None
            model_opts = st.session_state.get("model_opts") if include_model else None

            try:
                from inspect import signature

                sig = signature(build_report_payload)
                if "stats_config" in sig.parameters:
                    html_report, md_report = build_report_payload(
                        text_cols=text_cols,
                        stats_df=stats_df,
                        overall_obj=overall,
                        plot_df=plot_df,
                        results_df=results_df,
                        stats_config=stats_opts,
                        embed_config=embed_opts,
                        model_config=model_opts
                    )
                else:
                    html_report, md_report = build_report_payload(
                        text_cols=text_cols,
                        stats_df=stats_df,
                        overall_obj=overall,
                        plot_df=plot_df,
                        results_df=results_df
                    )
            except Exception:
                html_report, md_report = build_report_payload(
                    text_cols=text_cols,
                    stats_df=stats_df,
                    overall_obj=overall,
                    plot_df=plot_df,
                    results_df=results_df
                )

            if fmt == "Markdown":
                st.markdown(md_report)
                st.download_button(
                    "Download Markdown",
                    data=md_report.encode("utf-8"),
                    file_name="nlpsych_report.md",
                    mime="text/markdown"
                )
            else:
                render_html = st.checkbox("Render HTML in app", value=True)
                if render_html:
                    st.markdown(html_report, unsafe_allow_html=True)
                st.download_button(
                    "Download HTML",
                    data=html_report.encode("utf-8"),
                    file_name="nlpsych_report.html",
                    mime="text/html"
                )

if __name__ == "__main__":
    main()
