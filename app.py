# ===== Section: Imports and optional dependencies =====
import os
import io
import base64
from functools import lru_cache
from dataclasses import dataclass
from typing import Literal, Union, Dict, Any, Iterable, Optional, Tuple, List

import streamlit as st
import pandas as pd
import numpy as np

# Plotting
import plotly.express as px

# NLP
import spacy, sys, subprocess
from collections import Counter

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# Modeling
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.base import clone
from sklearn.utils import check_random_state
from statsmodels.stats.multitest import multipletests


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

# For reshowing cached data when clicking from one tab to another
def _opts_same(a: dict | None, b: dict | None) -> bool:
    return a is not None and b is not None and a == b

# ===== Section: Cached resources =====
import spacy, sys, subprocess
import streamlit as st

@st.cache_resource(show_spinner=False)
def get_spacy_pipeline():
    """
    Load a light English spaCy pipeline that excludes NER and parser,
    and ensures a fast rule based sentencizer is present.
    On first run, if the model is missing, download it.
    If download fails, fall back to a blank English pipeline.
    """
    def _load():
        nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser"])
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

    try:
        return _load()
    except OSError:
        with st.spinner("Downloading spaCy model en_core_web_sm…"):
            try:
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                return _load()
            except Exception as e:
                st.warning(
                    "Could not download en_core_web_sm. Falling back to a blank English pipeline."
                )
                nlp = spacy.blank("en")
                if "sentencizer" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")
                return nlp


@st.cache_resource(show_spinner=False)
def get_st_model(model_name: str = "all-MiniLM-L6-v2"):
    if not HAS_ST:
        st.error(
            "sentence-transformers is not installed. "
            "Install it with: pip install sentence-transformers"
        )
        raise RuntimeError("SentenceTransformer missing")
    return SentenceTransformer(model_name)


# ===== Section: Core text stats =====
SplitOverall = Literal["combined", "per_column", "both"]

def spacy_descriptive_stats(
    *series_list: Iterable[pd.Series],
    use_lemmas_for_uniques: bool = True,
    split_overall: SplitOverall = "combined",
    nlp: Optional[spacy.Language] = None,
    n_process: int = 1,
    batch_size: int = 256,
    words_must_be_alpha: bool = True,
    drop_stopwords: bool = False,
    avg_sentence_mode: Literal["unweighted", "ratio"] = "unweighted"
) -> Tuple[pd.DataFrame, Union[Dict[str, Any], Dict[str, Dict[str, Any]]]]:
    """
    Compute per row text stats and aggregate overall.
    DataFrame index is MultiIndex of source_column and original index.
    avg_sentence_mode unweighted means mean of per sentence token counts.
    avg_sentence_mode ratio means total tokens divided by total sentences.
    """
    if nlp is None:
        nlp = get_spacy_pipeline()

    has_lemmatizer = "lemmatizer" in nlp.pipe_names
    if use_lemmas_for_uniques and not has_lemmatizer:
        st.warning("Lemmatizer not found in pipeline. Falling back to surface forms for uniques.")
        use_lemmas_for_uniques = False

    rows = []
    for s in series_list:
        if not isinstance(s, pd.Series):
            raise ValueError("All inputs must be pandas Series")
        src = s.name if s.name is not None else "<unnamed>"
        for idx, text in s.dropna().items():
            if isinstance(text, str) and text.strip():
                rows.append({"text": text, "source_column": src, "index": idx})

    if not rows:
        return pd.DataFrame(), {}

    texts = [r["text"] for r in rows]

    def keep_token(t: spacy.tokens.Token) -> bool:
        if t.is_space or t.is_punct:
            return False
        if words_must_be_alpha and not t.is_alpha:
            return False
        if drop_stopwords and t.is_stop:
            return False
        return True

    stats_rows = []
    unique_sets = []
    pos_full_list = []

    # Process texts
    for doc, meta in zip(nlp.pipe(texts, batch_size=batch_size, n_process=n_process), rows):
        tokens = [t for t in doc if not t.is_space]
        word_tokens = [t for t in tokens if keep_token(t)]
        chars = len(meta["text"])
        word_count = len(word_tokens)
        sentences = list(doc.sents)
        sentence_count = len(sentences)

        # Average word length
        avg_word_length = float(np.mean([len(t.text) for t in word_tokens])) if word_count else 0.0

        # Average sentence length
        if sentence_count:
            if avg_sentence_mode == "unweighted":
                per_sent_counts = [len([t for t in sent if keep_token(t)]) for sent in sentences]
                avg_sentence_length = float(np.mean(per_sent_counts))
            else:
                total_in_sent = sum(len([t for t in sent if keep_token(t)]) for sent in sentences)
                avg_sentence_length = float(total_in_sent) / float(sentence_count) if sentence_count else 0.0
        else:
            avg_sentence_length = 0.0

        # Vocabulary set per row
        if use_lemmas_for_uniques:
            uniques = {t.lemma_.lower() for t in word_tokens if t.lemma_}
        else:
            uniques = {t.text.lower() for t in word_tokens}

        unique_words = len(uniques)
        lexical_diversity = float(unique_words / word_count) if word_count else 0.0
        pos_counts = Counter(t.pos_ for t in word_tokens)

        unique_sets.append((meta["source_column"], meta["index"], uniques))
        pos_full_list.append((meta["source_column"], meta["index"], pos_counts))

        stats_rows.append(
            {
                "source_column": meta["source_column"],
                "index": meta["index"],
                "char_count": int(chars),
                "word_count": int(word_count),
                "sentence_count": int(sentence_count),
                "avg_word_length": float(avg_word_length),
                "avg_sentence_length": float(avg_sentence_length),
                "unique_words": int(unique_words),
                "lexical_diversity": float(lexical_diversity),
                "noun_count": int(sum(1 for t in word_tokens if t.pos_ in {"NOUN", "PROPN"})),
                "verb_count": int(sum(1 for t in word_tokens if t.pos_ in {"VERB", "AUX"})),
                "adjective_count": int(sum(1 for t in word_tokens if t.pos_ == "ADJ")),
                "adverb_count": int(sum(1 for t in word_tokens if t.pos_ == "ADV")),
                "pos_full": dict(pos_counts),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df.set_index(["source_column", "index"], inplace=True)
    stats_df.sort_index(inplace=True)

    uniques_by_row = {(src, idx): u for src, idx, u in unique_sets}
    pos_by_row = {(src, idx): c for src, idx, c in pos_full_list}

    def compute_overall(index_like) -> Dict[str, Any]:
        sub = stats_df.loc[index_like]
        total_words = int(sub["word_count"].sum())
        vocab_union = set()
        for key in sub.index:
            vocab_union |= uniques_by_row.get(key, set())
        pos_total = Counter()
        for key in sub.index:
            pos_total += pos_by_row.get(key, Counter())

        # Average sentence length aggregation mirrors per row choice
        if len(sub):
            if avg_sentence_mode == "unweighted":
                avg_sentence_length_agg = float(sub["avg_sentence_length"].mean())
            else:
                total_sentences = int(sub["sentence_count"].sum())
                avg_sentence_length_agg = float(total_words) / float(total_sentences) if total_sentences else 0.0
        else:
            avg_sentence_length_agg = 0.0

        return {
            "total_chars": int(sub["char_count"].sum()),
            "total_words": total_words,
            "total_sentences": int(sub["sentence_count"].sum()),
            "avg_word_length": float(sub["avg_word_length"].mean()) if len(sub) else 0.0,
            "avg_sentence_length": float(avg_sentence_length_agg),
            "total_unique_words": int(len(vocab_union)),
            "lexical_diversity": float(len(vocab_union) / total_words) if total_words else 0.0,
            "pos_distribution_basic": {
                "nouns": int(sub["noun_count"].sum()),
                "verbs": int(sub["verb_count"].sum()),
                "adjectives": int(sub["adjective_count"].sum()),
                "adverbs": int(sub["adverb_count"].sum()),
            },
            "pos_distribution_full": dict(pos_total),
        }

    if stats_df.empty:
        return stats_df, {}

    if split_overall == "combined":
        return stats_df, compute_overall(stats_df.index)
    if split_overall == "per_column":
        out = {}
        for col, subdf in stats_df.groupby(level=0):
            out[col] = compute_overall(subdf.index)
        return stats_df, out
    if split_overall == "both":
        per_col = {}
        for col, subdf in stats_df.groupby(level=0):
            per_col[col] = compute_overall(subdf.index)
        return stats_df, {"combined": compute_overall(stats_df.index), "per_column": per_col}
    return stats_df, compute_overall(stats_df.index)


# ===== Section: Embeddings and reduction =====
@st.cache_data(show_spinner=False)
def embed_text_columns_simple(series_list: List[pd.Series], model_name="all-MiniLM-L6-v2", normalize=True):
    model = get_st_model(model_name)
    texts = []
    meta = []
    for s in series_list:
        if not isinstance(s, pd.Series):
            raise ValueError("All inputs must be pandas Series")
        for idx, text in s.dropna().items():
            if isinstance(text, str) and text.strip():
                texts.append(text)
                meta.append({"index": idx, "source_column": s.name})
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=normalize)
    meta_df = pd.DataFrame(meta)
    return meta_df, embeddings, texts


def reduce_embeddings(
    embeddings: np.ndarray,
    method: Literal["umap", "tsne", "pca"] = "pca",
    n_components: int = 2,
    random_state: int = 42,
    metric: str = "cosine"
):
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")

    if method == "umap":
        if not HAS_UMAP:
            st.warning("UMAP not found. Falling back to PCA.")
            method = "pca"
        else:
            reducer = umap.UMAP(n_components=n_components, metric=metric, random_state=random_state)
            Z = reducer.fit_transform(embeddings)
            return Z

    if method == "tsne":
        n = len(embeddings)
        # Perplexity must be less than n and usually less than about n/3
        safe_perp = max(5, min(30, (n - 1) // 3 if n > 10 else max(2, n // 2)))
        reducer = TSNE(
            n_components=n_components,
            metric="cosine" if metric == "cosine" else "euclidean",
            init="pca",
            random_state=random_state,
            perplexity=safe_perp
        )
        Z = reducer.fit_transform(embeddings)
        return Z


    reducer = PCA(n_components=n_components, random_state=random_state)
    Z = reducer.fit_transform(embeddings)
    return Z


def build_plot_df(Z: np.ndarray, meta: pd.DataFrame, texts: List[str]):
    n_components = Z.shape[1]
    cols = [f"dim_{i+1}" for i in range(n_components)]
    plot_df = meta.reset_index(drop=True).copy()
    for i, c in enumerate(cols):
        plot_df[c] = Z[:, i]
    plot_df["text"] = list(texts)
    return plot_df


def plot_projection(plot_df: pd.DataFrame, n_components: int = 2, color_by: Optional[str] = "source_column", point_size: int = 7):
    hover_cols = ["text"]
    for c in ["source_column", "index"]:
        if c in plot_df.columns:
            hover_cols.append(c)

    if n_components == 2:
        fig = px.scatter(
            plot_df,
            x="dim_1",
            y="dim_2",
            color=color_by if color_by in plot_df.columns else None,
            hover_data=hover_cols,
            opacity=0.9
        )
        fig.update_traces(marker=dict(size=point_size))
    else:
        fig = px.scatter_3d(
            plot_df,
            x="dim_1",
            y="dim_2",
            z="dim_3",
            color=color_by if color_by in plot_df.columns else None,
            hover_data=hover_cols,
            opacity=0.9
        )
        fig.update_traces(marker=dict(size=point_size))
    fig.update_layout(template="plotly_white")
    return fig


# ===== Section: Modeling with cross validation and permutation test =====
@dataclass
class TargetResult:
    target: str
    task: str
    metric_name: str
    observed: float
    p_value: float
    p_fdr: float
    cv_scores: List[float]
    perm_scores: List[float]
    cv_folds_used: int


def _detect_task(y: pd.Series, max_unique_for_class: int = 20) -> str:
    if y.dtype.name in {"category", "object", "bool"}:
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= max_unique_for_class:
        return "classification"
    if pd.api.types.is_float_dtype(y) and y.nunique() <= max_unique_for_class:
        return "classification"
    return "regression"


def _cv_iterator(task: str, y: np.ndarray, cv: int, random_state: int):
    n = len(y)
    if n < 2:
        raise ValueError("Not enough samples for cross validation")

    if task == "classification":
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError("Classification requires at least two classes after alignment with X")
        min_per_class = int(counts.min())
        n_splits = max(2, min(cv, min_per_class))
        if n_splits < cv:
            try:
                import streamlit as st
                st.info(f"Reducing CV folds from {cv} to {n_splits} due to small class counts")
            except Exception:
                pass
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        n_splits = max(2, min(cv, n))
        if n_splits < cv:
            try:
                import streamlit as st
                st.info(f"Reducing CV folds from {cv} to {n_splits} due to small sample size")
            except Exception:
                pass
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _primary_metric(task: str):
    if task == "classification":
        return "accuracy", accuracy_score, True
    return "r2", r2_score, True


def _fit_and_score_cv(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    base_model,
    cv_splits,
    scorer
) -> Tuple[List[float], np.ndarray]:
    scores = []
    preds = np.empty_like(y, dtype=float)
    for train, test in cv_splits.split(X, y if task == "classification" else None):
        model = clone(base_model)
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        preds[test] = y_pred
        scores.append(scorer(y[test], y_pred))
    return scores, preds


def _perm_test(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    base_model,
    cv_splits,
    scorer,
    observed: float,
    n_perm: int,
    larger_is_better: bool,
    rng: np.random.RandomState
):
    perm_scores = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        scores, _ = _fit_and_score_cv(X, y_perm, task, base_model, cv_splits, scorer)
        perm_scores.append(np.mean(scores))
    if larger_is_better:
        p = (1.0 + np.sum(np.asarray(perm_scores) >= observed)) / (n_perm + 1.0)
    else:
        p = (1.0 + np.sum(np.asarray(perm_scores) <= observed)) / (n_perm + 1.0)
    return perm_scores, p


def auto_cv_with_permutation(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    cv: int = 5,
    n_permutations: int = 200,
    random_state: int = 42,
    max_unique_for_class: int = 20,
    classifier_model=None,
    regressor_model=None,
    scale_X: bool = True
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:

    if classifier_model is None:
        classifier_model = LogisticRegression(max_iter=1000)
    if regressor_model is None:
        regressor_model = Ridge(alpha=1.0)

    rng = check_random_state(random_state)

    X_mat = X.values.astype(float)
    if scale_X:
        scaler = StandardScaler().fit(X_mat)
        X_mat = scaler.transform(X_mat)

    results: List[TargetResult] = []
    cv_preds: Dict[str, np.ndarray] = {}

    for target in Y.columns:
        y_raw = Y[target].dropna()
        common_idx = X.index.intersection(y_raw.index)
        y_raw = y_raw.loc[common_idx]

        if len(y_raw) < 2:
            try:
                import streamlit as st
                st.warning(f"Skipping {target} because fewer than two samples remain after alignment")
            except Exception:
                pass
            continue

        X_sub = X_mat[[X.index.get_loc(i) for i in common_idx]]

        task = _detect_task(y_raw, max_unique_for_class)
        metric_name, scorer, larger_is_better = _primary_metric(task)

        if task == "classification":
            classes = np.unique(y_raw.values)
            if len(classes) < 2:
                try:
                    import streamlit as st
                    st.warning(f"Skipping {target} because it has a single class after alignment")
                except Exception:
                    pass
                continue
            le = LabelEncoder()
            y = le.fit_transform(y_raw.values)
            base_model = make_pipeline(LogisticRegression(max_iter=1000)) if classifier_model is None else make_pipeline(classifier_model)
        else:
            y = y_raw.values.astype(float)
            base_model = make_pipeline(Ridge(alpha=1.0)) if regressor_model is None else make_pipeline(regressor_model)

        cv_splits = _cv_iterator(task, y, cv, random_state)
        try:
            actual_folds = cv_splits.get_n_splits()
        except Exception:
            actual_folds = cv

        cv_scores, preds = _fit_and_score_cv(X_sub, y, task, base_model, cv_splits, scorer)
        observed = float(np.mean(cv_scores))

        perm_scores, p_val = _perm_test(
            X_sub, y, task, base_model, cv_splits, scorer,
            observed, n_permutations, larger_is_better, rng
        )

        results.append(TargetResult(
            target=target,
            task=task,
            metric_name=metric_name,
            observed=observed,
            p_value=float(p_val),
            p_fdr=np.nan,
            cv_scores=cv_scores,
            perm_scores=perm_scores,
            cv_folds_used=int(actual_folds)
        ))
        cv_preds[target] = preds

    if not results:
        return pd.DataFrame(columns=[
            "target","task","metric_name","observed","p_value","p_fdr",
            "cv_scores","perm_scores","cv_folds_used"
        ]), {}

    # FDR correction across targets
    p_vals = [r.p_value for r in results]
    reject, p_fdr, _, _ = multipletests(p_vals, method="fdr_bh")
    for r, adj in zip(results, p_fdr):
        r.p_fdr = float(adj)

    results_df = pd.DataFrame([r.__dict__ for r in results])
    return results_df, cv_preds



# ===== Section: Utility for downloads =====
def df_to_csv_download(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=True).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=filename, mime="text/csv")


def npy_download(arr: np.ndarray, filename: str):
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    st.download_button("Download NPY", data=buf, file_name=filename, mime="application/octet-stream")


# ===== Section: Streamlit UI =====
# Page config and title
st.set_page_config(page_title="NLPsych", layout="wide")
st.title("NLPsych")
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


# ===== Pretty report helpers =====
from datetime import datetime

def _pick_overall(overall_obj):
    if isinstance(overall_obj, dict) and "combined" in overall_obj:
        return overall_obj["combined"]
    return overall_obj or {}

def interpret_stats(overall: dict) -> list[str]:
    out = []
    if not overall:
        return out
    ld = overall.get("lexical_diversity")
    if ld is not None:
        if ld >= 0.8:
            out.append("Lexical diversity is very high which suggests a rich vocabulary with little repetition.")
        elif ld >= 0.6:
            out.append("Lexical diversity is moderate which suggests some repetition and good variety.")
        else:
            out.append("Lexical diversity is relatively low which suggests frequent repetition of word types.")
    asl = overall.get("avg_sentence_length")
    if asl is not None:
        if asl >= 20:
            out.append("Average sentence length is long which often reflects complex structure.")
        elif asl >= 12:
            out.append("Average sentence length is mid range which often reflects balanced structure.")
        else:
            out.append("Average sentence length is short which often reflects simple direct phrasing.")
    posb = overall.get("pos_distribution_basic", {})
    total_words = overall.get("total_words", 0) or 0
    if total_words and posb:
        nouns = posb.get("nouns", 0)
        verbs = posb.get("verbs", 0)
        if nouns + verbs:
            nv_ratio = nouns / max(1, verbs)
            if nv_ratio >= 2:
                out.append("Noun to verb ratio is high which can indicate dense nominal style.")
            elif nv_ratio <= 0.7:
                out.append("Verb to noun ratio is high which can indicate action oriented style.")
    return out

def interpret_model_row(row: pd.Series) -> list[str]:
    out = []
    metric = str(row.get("metric_name", "")).upper()
    score = float(row.get("observed", 0.0))
    p = float(row.get("p_value", 1.0))
    p_fdr = float(row.get("p_fdr", 1.0))
    folds = int(row.get("cv_folds_used", 0)) if pd.notnull(row.get("cv_folds_used", np.nan)) else None
    task = row.get("task", "classification")

    if task == "classification":
        if score >= 0.80:
            out.append(f"{metric} indicates strong discrimination.")
        elif score >= 0.60:
            out.append(f"{metric} indicates moderate discrimination.")
        else:
            out.append(f"{metric} indicates weak discrimination.")
    else:
        if score >= 0.50:
            out.append(f"{metric} indicates strong fit.")
        elif score >= 0.20:
            out.append(f"{metric} indicates moderate fit.")
        elif score >= 0.00:
            out.append(f"{metric} indicates weak fit.")
        else:
            out.append(f"{metric} is negative which is worse than a constant baseline and suggests poor generalization.")

    if p_fdr < 0.05:
        out.append("Permutation test with FDR correction indicates a statistically reliable effect.")
    elif p < 0.05:
        out.append("Permutation test is nominally significant but not after FDR correction.")
    else:
        out.append("Permutation test does not indicate statistical significance.")

    if folds is not None and folds > 0:
        out.append(f"Cross validation used {folds} folds.")
    return out

def _to_html_table(df: pd.DataFrame, index=False):
    return df.to_html(index=index, classes="table", border=0, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x)

def build_report_payload(
    text_cols: list[str],
    stats_df: Optional[pd.DataFrame],
    overall_obj: Optional[dict],
    plot_df: Optional[pd.DataFrame],
    results_df: Optional[pd.DataFrame]
) -> tuple[str, str]:
    """Returns HTML and Markdown versions of the same report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Prepare sections
    stats_overall = _pick_overall(overall_obj) if overall_obj else {}
    stats_rows = []
    if stats_overall:
        stats_rows = [
            ["Total characters", stats_overall.get("total_chars", "NA")],
            ["Total words", stats_overall.get("total_words", "NA")],
            ["Total sentences", stats_overall.get("total_sentences", "NA")],
            ["Average word length", round(float(stats_overall.get("avg_word_length", 0.0)), 3)],
            ["Average sentence length", round(float(stats_overall.get("avg_sentence_length", 0.0)), 3)],
            ["Total unique words", stats_overall.get("total_unique_words", "NA")],
            ["Lexical diversity", round(float(stats_overall.get("lexical_diversity", 0.0)), 3)],
        ]
    stats_table_df = pd.DataFrame(stats_rows, columns=["Metric", "Value"]) if stats_rows else None
    stats_interps = interpret_stats(stats_overall) if stats_overall else []

    # Embeddings quick note
    emb_note = None
    if plot_df is not None and len(plot_df):
        dims = 3 if "dim_3" in plot_df.columns else 2
        emb_note = f"Projection computed with {dims} dimensions. Downloadable files are available from the Embeddings tab."

    # Modeling table and interpretations
    model_table_df = None
    model_interps = []
    if results_df is not None and len(results_df):
        show_cols = ["target", "task", "metric_name", "observed", "p_value", "p_fdr", "cv_folds_used"]
        show_cols = [c for c in show_cols if c in results_df.columns]
        model_table_df = results_df[show_cols].copy()
        model_table_df["observed"] = model_table_df["observed"].astype(float).round(3)
        if "p_value" in model_table_df:
            model_table_df["p_value"] = model_table_df["p_value"].astype(float).round(4)
        if "p_fdr" in model_table_df:
            model_table_df["p_fdr"] = model_table_df["p_fdr"].astype(float).round(4)
        for _, row in results_df.iterrows():
            tgt = str(row.get("target", "unknown"))
            bullets = interpret_model_row(row)
            model_interps.append((tgt, bullets))

    # HTML with light CSS
    css = """<style>
    .report { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; line-height: 1.5; color: #111; }
    .header { margin-bottom: 1rem; }
    .title { font-size: 1.75rem; font-weight: 700; margin: 0; }
    .subtitle { color: #555; margin: .25rem 0 0 0; }
    .card { background: #fff; border: 1px solid #eee; border-radius: 12px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .section-title { font-size: 1.2rem; font-weight: 700; margin: 0 0 8px 0; }
    .table { border-collapse: collapse; width: 100%; }
    .table th, .table td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #f0f0f0; }
    .table th { background: #fafafa; font-weight: 600; }
    .muted { color: #666; }
    ul { margin: .25rem 0 .5rem 1.2rem; }
    li { margin: .15rem 0; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f4f6f8; font-size: .875rem; margin-right: 6px; }
    </style>"""

    html_parts = []
    html_parts.append('<div class="report">')
    html_parts.append('<div class="header">')
    html_parts.append(f'<p class="title">NLPsych session report</p>')
    html_parts.append(f'<p class="subtitle">Generated at {ts}</p>')
    html_parts.append('</div>')

    # Overview card
    html_parts.append('<div class="card">')
    html_parts.append('<div class="section-title">Overview</div>')
    html_parts.append(f'<p>Selected text columns</p>')
    html_parts.append("".join([f'<span class="pill">{c}</span>' for c in text_cols]) or '<p class="muted">None</p>')
    html_parts.append('</div>')

    # Stats card
    html_parts.append('<div class="card">')
    html_parts.append('<div class="section-title">Descriptive statistics</div>')
    if stats_table_df is None:
        html_parts.append('<p class="muted">No descriptive statistics were available. Run the Descriptive stats tab first.</p>')
    else:
        html_parts.append(_to_html_table(stats_table_df, index=False))
        if stats_interps:
            html_parts.append('<p><strong>Interpretation</strong></p>')
            html_parts.append('<ul>')
            for s in stats_interps:
                html_parts.append(f'<li>{s}</li>')
            html_parts.append('</ul>')
    html_parts.append('</div>')

    # Embeddings card
    html_parts.append('<div class="card">')
    html_parts.append('<div class="section-title">Embeddings and projection</div>')
    if emb_note is None:
        html_parts.append('<p class="muted">No embeddings or projection were available. Run the Embeddings tab first.</p>')
    else:
        html_parts.append(f'<p>{emb_note}</p>')
    html_parts.append('</div>')

    # Modeling card
    html_parts.append('<div class="card">')
    html_parts.append('<div class="section-title">Modeling results</div>')
    if model_table_df is None:
        html_parts.append('<p class="muted">No modeling results were available. Run the Modeling tab first.</p>')
    else:
        html_parts.append(_to_html_table(model_table_df, index=False))
        if model_interps:
            html_parts.append('<p><strong>Interpretation per target</strong></p>')
            html_parts.append('<ul>')
            for tgt, bullets in model_interps:
                html_parts.append(f'<li><strong>{tgt}</strong>')
                if bullets:
                    html_parts.append('<ul>')
                    for b in bullets:
                        html_parts.append(f'<li>{b}</li>')
                    html_parts.append('</ul>')
                html_parts.append('</li>')
            html_parts.append('</ul>')
    html_parts.append('</div>')

    html_parts.append('</div>')  # end report

    html_report = css + "\n" + "\n".join(html_parts) 

    # Markdown version for users who prefer plain text
    md_parts = []
    md_parts.append("# NLPsych session report")
    md_parts.append(f"Generated at {ts}")
    md_parts.append(f"Selected text columns: {', '.join(text_cols)}\n")
    md_parts.append("## Descriptive statistics")
    if stats_table_df is None:
        md_parts.append("No descriptive statistics were available. Run the Descriptive stats tab first.")
    else:
        md_parts.append(stats_table_df.to_markdown(index=False))
        if stats_interps:
            md_parts.append("Interpretation")
            for s in stats_interps:
                md_parts.append(f"* {s}")
    md_parts.append("\n## Embeddings and projection")
    if emb_note is None:
        md_parts.append("No embeddings or projection were available. Run the Embeddings tab first.")
    else:
        md_parts.append(emb_note)
    md_parts.append("\n## Modeling results")
    if model_table_df is None:
        md_parts.append("No modeling results were available. Run the Modeling tab first.")
    else:
        md_parts.append(model_table_df.to_markdown(index=False))
        md_parts.append("Interpretation per target")
        for tgt, bullets in model_interps:
            md_parts.append(f"* {tgt}")
            for b in bullets:
                md_parts.append(f"  * {b}")

    md_report = "\n\n".join(md_parts)
    return html_report, md_report



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
        run_embed = st.button("Compute embeddings and plot", key="btn_embed_compute")
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
            meta_df, emb, texts = embed_text_columns_simple(
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
                    meta_df, emb, texts = embed_text_columns_simple(
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