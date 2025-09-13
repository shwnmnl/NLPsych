from __future__ import annotations
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import spacy 
from pathlib import Path
from importlib.resources import files
from nlpsych.utils import get_spacy_pipeline_base, get_st_model_base
from nlpsych.descriptive_stats import spacy_descriptive_stats
from nlpsych.embedding import (
    embed_text_columns_simple_base,
    reduce_embeddings,
    build_plot_df,
    plot_projection,
)
import base64
from nlpsych.modeling import auto_cv_with_permutation
from nlpsych.report import build_report_payload

ASSETS = files("nlpsych_app") / "assets"

def main():
    # ===== Session state init =====
    def _init_state():
        ss = st.session_state
        ss.setdefault("stats_df", None)
        ss.setdefault("overall", None)
        ss.setdefault("meta_df", None)
        ss.setdefault("embeddings", None)
        ss.setdefault("plot_df", None)
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
    def embed_text_columns_simple_ui(series_list: List[pd.Series], model_name="all-MiniLM-L6-v2", normalize=True):
        return embed_text_columns_simple_base(series_list, model_name=model_name, normalize=normalize)

    # ===== Section: Utility for downloads =====
    def df_to_csv_download(df: pd.DataFrame, filename: str):
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name=filename, mime="text/csv")


    def npy_download(arr: np.ndarray, filename: str):
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        st.download_button("Download NPY", data=buf, file_name=filename, mime="application/octet-stream")

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
    st.set_page_config(page_title="NLPsych", page_icon="💬", layout="centered")
    # Centered logo
    logo_b = load_logo_bytes("NLPsych_logo.png")
    data_uri = "data:image/png;base64," + base64.b64encode(logo_b).decode("utf-8")
    st.markdown(
        f"""
        <p style="text-align:center;">
            <img src="{data_uri}" width="150">
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 style='text-align: center; color: black;'>NLPsych</h1>", unsafe_allow_html=True)
    st.markdown("""
    NLPsych (*Natural Language Psychometrics*) lets you upload a CSV file, pick your text columns, and get instant descriptive statistics, visualize semantic embeddings, and run predictive models all in one streamlined workflow.
    """)

    # Uploader at the top, no sidebar
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_demo = st.checkbox("Use demo data if no file", value=True)

    # Load data
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        if use_demo:
            df = pd.DataFrame(
                {
                    "text_a": [
                        "Hello world. This is a tiny test.",
                        "I propose a simple method for estimating brain connectivity. Results suggest strong default mode network involvement. However, the sample was small.",
                        "lol that movie was unreal, i could not believe the ending tbh",
                        "Patient denies chest pain. Vitals stable. Recommend follow up in two weeks.",
                        "Consider the problem of induction. How can we justify any expectation of regularity",
                        "A sudden pang of memory, a ripple on the surface of thought, and then stillness.",
                    ],
                    "text_b": [
                        "Buy now and save big. Limited time offer.",
                        "Deep reinforcement learning agents can overfit. Regularization and data augmentation help.",
                        "Today I made pasta with garlic and olive oil. It was perfect.",
                        "To be is to be perceived, said Bishop Berkeley.",
                        "The quick brown fox jumps over the lazy dog.",
                        "This is a very very very repetitive sentence sentence sentence.",
                    ],
                    "target_demo": [0, 1, 0, 1, 0, 1],
                }
            )
            st.info("Using built in demo data")
        else:
            st.stop()

    st.write("Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Column selection just once, used by tabs
    candidate_text_cols = [c for c in df.columns if df[c].dtype == object]
    text_cols = st.multiselect("Select text columns", candidate_text_cols, default=candidate_text_cols[:1])
    if not text_cols:
        st.warning("Select at least one text column.")
        st.stop()

    tabs = st.tabs(["Overview", "Descriptive stats", "Embeddings", "Modeling", "Report"])



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
            words_must_be_alpha = st.checkbox("Alphabetic only", value=True)
        with col_b:
            drop_stopwords = st.checkbox("Drop stopwords", value=False)
        with col_c:
            use_lemmas = st.checkbox("Use lemmas for uniques", value=True)
        with col_d:
            avg_sent_mode = st.selectbox("Sentence length mode", ["unweighted", "ratio"])

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
            run_stats = st.button("Compute descriptive stats", key="btn_stats_compute")
            recompute = st.button("Recompute", key="btn_stats_recompute") if cache_is_fresh else False

        with col_right:
            if cache_is_fresh:
                st.caption("Showing cached results")
            elif cached_df is not None:
                st.caption("Cached results exist but options changed. Press Recompute to refresh.")

        if cache_is_fresh and not recompute and not run_stats:
            st.write("Per row stats")
            st.dataframe(cached_df, use_container_width=True)
            df_to_csv_download(cached_df, "per_row_stats.csv")
            st.write("Overall stats")
            st.json(cached_overall)
        elif run_stats or recompute:
            try:
                nlp = get_spacy_pipeline()
                series_list = [df[c] for c in text_cols]
                stats_df, overall = spacy_descriptive_stats(
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
                st.write("Per row stats")
                st.dataframe(stats_df, use_container_width=True)
                df_to_csv_download(stats_df, "per_row_stats.csv")
                st.write("Overall stats")
                st.json(overall)

                st.session_state["stats_df"] = stats_df
                st.session_state["overall"] = overall
                st.session_state["stats_opts"] = current_opts
            except Exception as e:
                st.exception(e)
        else:
            if cached_df is None:
                st.info("Press Compute to generate stats.")
            else:
                st.write("Per row stats")
                st.dataframe(cached_df, use_container_width=True)
                df_to_csv_download(cached_df, "per_row_stats.csv")
                st.write("Overall stats")
                st.json(cached_overall)


    # ===== Tab: Embeddings =====
    with tabs[2]:
        st.subheader("Embeddings and projection")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            model_name = st.text_input("SentenceTransformer model", "all-MiniLM-L6-v2")
        with col_b:
            reduce_method = st.selectbox("Reduction method", ["pca", "umap", "tsne"])
        with col_c:
            n_components = st.selectbox("Dimensions", [2, 3], index=0)

        current_embed_opts = {
            "model_name": model_name,
            "reduce_method": reduce_method,
            "n_components": int(n_components),
            "text_cols": tuple(text_cols),
        }

        cached_meta = st.session_state.get("meta_df")
        cached_emb = st.session_state.get("embeddings")
        cached_plot_df = st.session_state.get("plot_df")
        cached_embed_opts = st.session_state.get("embed_opts")

        cache_is_fresh = (
            _opts_same(cached_embed_opts, current_embed_opts)
            and cached_meta is not None
            and cached_emb is not None
            and cached_plot_df is not None
        )

        col_left, col_right = st.columns([1, 3])
        with col_left:
            run_embed = st.button("Embed and plot", key="btn_embed_compute")
            recompute_embed = st.button("Recompute", key="btn_embed_recompute") if cache_is_fresh else False

        with col_right:
            if cache_is_fresh:
                st.caption("Showing cached results")
            elif cached_plot_df is not None:
                st.caption("Cached results exist but options changed. Press Recompute to refresh.")

        def _render_embed_block(plot_df, emb, meta_df):
            fig = plot_projection(plot_df, n_components=plot_df.filter(like="dim_").shape[1], color_by="source_column", point_size=8)
            st.plotly_chart(fig, use_container_width=True)
            st.write("Projection coordinates")
            st.dataframe(plot_df.head(), use_container_width=True)
            df_to_csv_download(plot_df, "projection.csv")
            st.write("Raw embeddings")
            npy_download(emb, "embeddings.npy")
            emb_cols = [f"e{i+1}" for i in range(emb.shape[1])]
            emb_df = pd.DataFrame(emb, columns=emb_cols)
            emb_df = pd.concat([meta_df.reset_index(drop=True), emb_df], axis=1)
            df_to_csv_download(emb_df, "embeddings_with_meta.csv")

        if cache_is_fresh and not recompute_embed and not run_embed:
            _render_embed_block(cached_plot_df, cached_emb, cached_meta)
        elif run_embed or recompute_embed:
            try:
                meta_df, emb, texts = embed_text_columns_simple_ui(
                    [df[c] for c in text_cols],
                    model_name=model_name,
                    normalize=True
                )
                Z = reduce_embeddings(embeddings=emb, method=reduce_method, n_components=int(n_components))
                plot_df = build_plot_df(Z, meta_df, texts)
                _render_embed_block(plot_df, emb, meta_df)

                st.session_state["meta_df"] = meta_df
                st.session_state["embeddings"] = emb
                st.session_state["plot_df"] = plot_df
                st.session_state["embed_opts"] = current_embed_opts
            except Exception as e:
                st.exception(e)
        else:
            if cached_plot_df is None:
                st.info("Press Compute to generate embeddings and projection.")
            else:
                _render_embed_block(cached_plot_df, cached_emb, cached_meta)


    # ===== Tab: Modeling =====
    with tabs[3]:
        st.subheader("Quick model with cross validation and permutation test")
        st.caption("Uses embeddings as features. Select a non text target column.")

        non_text_cols = [c for c in df.columns if c not in text_cols]

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            target_cols = st.multiselect(
                "Target columns",
                non_text_cols,
                default=[c for c in non_text_cols if c.lower().startswith("target")][:1]
            )
        with col_b:
            cv_folds = st.number_input("CV folds", min_value=3, max_value=10, value=5, step=1)
        with col_c:
            n_perm = st.number_input("Permutation iterations", min_value=50, max_value=2000, value=200, step=50)

        col_d, col_e = st.columns(2)
        with col_d:
            reuse_embed = st.checkbox("Reuse cached embeddings if available", value=True)
        with col_e:
            model_name_modeling = st.text_input("SentenceTransformer model for modeling", "all-MiniLM-L6-v2")

        model_opts_current = {
            "target_cols": tuple(target_cols),
            "cv_folds": int(cv_folds),
            "n_perm": int(n_perm),
            "text_cols": tuple(text_cols),
            "reuse_embed": bool(reuse_embed),
            "model_name_modeling": model_name_modeling,
        }

        cached_results = st.session_state.get("results_df")
        cached_preds = st.session_state.get("preds")
        cached_model_opts = st.session_state.get("model_opts")

        cache_fresh = cached_results is not None and cached_preds is not None and cached_model_opts == model_opts_current

        left, right = st.columns([1, 3])
        with left:
            run_model = st.button("Run modeling", key="btn_model_compute")
            recompute_model = st.button("Recompute", key="btn_model_recompute") if cache_fresh else False

        with right:
            if cache_fresh:
                st.caption("Showing cached results")
            elif cached_results is not None:
                st.caption("Cached results exist but options changed. Press Recompute to refresh.")

        def _render_model_results(results_df: pd.DataFrame):
            st.write("Results")
            st.dataframe(results_df, use_container_width=True)
            df_to_csv_download(results_df, "model_results.csv")
            if len(results_df):
                first = results_df.iloc[0]
                tgt = first.get("target", "target")
                perm_scores = first.get("perm_scores", [])
                observed = float(first.get("observed", 0.0))
                if isinstance(perm_scores, list) and len(perm_scores):
                    fig_hist = px.histogram(x=perm_scores, nbins=30, title=f"Permutation scores for {tgt}")
                    fig_hist.add_vline(x=observed)
                    st.plotly_chart(fig_hist, use_container_width=True)

        if cache_fresh and not recompute_model and not run_model:
            _render_model_results(cached_results)
        elif run_model or recompute_model:
            if not target_cols:
                st.warning("Select at least one target column.")
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
                    X_by_row = X.groupby(level=1).mean()
                    Y = df[target_cols]

                    results_df, preds = auto_cv_with_permutation(
                        X=X_by_row,
                        Y=Y,
                        cv=int(cv_folds),
                        n_permutations=int(n_perm),
                        random_state=42,
                        scale_X=True
                    )

                    _render_model_results(results_df)

                    st.session_state["results_df"] = results_df
                    st.session_state["preds"] = preds
                    st.session_state["model_opts"] = model_opts_current
                except Exception as e:
                    st.exception(e)
        else:
            if cached_results is None:
                st.info("Press Run modeling to compute results.")
            else:
                _render_model_results(cached_results)


    # ===== Tab: Report =====
    with tabs[4]:
        st.subheader("Session report")
        st.caption("Summarizes what you ran and adds brief interpretations.")

        include_stats = st.checkbox("Include descriptive stats", value=st.session_state.get("stats_df") is not None)
        include_embed = st.checkbox("Include embeddings and projection", value=st.session_state.get("plot_df") is not None)
        include_model = st.checkbox("Include modeling", value=st.session_state.get("results_df") is not None)

        fmt = st.radio("Format", ["HTML", "Markdown"], horizontal=True, key="radio_report_format")
        assemble = st.button("Assemble report", key="btn_report_assemble")

        if assemble:
            stats_df = st.session_state.get("stats_df") if include_stats else None
            overall = st.session_state.get("overall") if include_stats else None
            plot_df = st.session_state.get("plot_df") if include_embed else None
            results_df = st.session_state.get("results_df") if include_model else None

            html_report, md_report = build_report_payload(
                text_cols=text_cols,
                stats_df=stats_df,
                overall_obj=overall,
                plot_df=plot_df,
                results_df=results_df
            )

            if fmt == "HTML":
                st.markdown(html_report, unsafe_allow_html=True)
                st.download_button(
                    "Download HTML",
                    data=html_report.encode("utf-8"),
                    file_name="nlpsych_report.html",
                    mime="text/html"
                )
            else:
                st.markdown(md_report)
                st.download_button(
                    "Download Markdown",
                    data=md_report.encode("utf-8"),
                    file_name="nlpsych_report.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main()