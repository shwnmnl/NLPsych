from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

try:
    from bertopic import BERTopic  # type: ignore
    from bertopic.vectorizers import ClassTfidfTransformer  # type: ignore

    HAS_BERTOPIC = True
    _BERTOPIC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised via fallback path
    BERTopic = None  # type: ignore[assignment]
    ClassTfidfTransformer = None  # type: ignore[assignment]
    HAS_BERTOPIC = False
    _BERTOPIC_IMPORT_ERROR = exc

try:
    import hdbscan  # type: ignore

    HAS_HDBSCAN = True
    _HDBSCAN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised via fallback path
    hdbscan = None  # type: ignore[assignment]
    HAS_HDBSCAN = False
    _HDBSCAN_IMPORT_ERROR = exc

try:
    import umap  # type: ignore

    HAS_TOPIC_UMAP = True
    _UMAP_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised via fallback path
    umap = None  # type: ignore[assignment]
    HAS_TOPIC_UMAP = False
    _UMAP_IMPORT_ERROR = exc

def _require_topic_modeling_dependencies(
    *,
    require_hdbscan: bool = False,
    require_umap: bool = False,
) -> None:
    missing: list[str] = []
    if not HAS_BERTOPIC:
        missing.append("bertopic")
    if require_hdbscan and not HAS_HDBSCAN:
        missing.append("hdbscan")
    if require_umap and not HAS_TOPIC_UMAP:
        missing.append("umap-learn")
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "Topic modeling requires optional dependencies "
            f"({missing_str}). Install them with "
            "`pip install nlpsych` or add the missing packages manually."
        )


def _resolved_min_df(
    n_docs: int,
    vectorizer_min_df: Optional[int | float],
) -> int | float:
    if vectorizer_min_df is not None:
        return vectorizer_min_df
    return 2 if int(n_docs) >= 20 else 1


def _build_vectorizer_model(
    n_docs: int,
    *,
    ngram_range: tuple[int, int],
    vectorizer_stop_words: str | list[str] | None = "english",
    vectorizer_min_df: Optional[int | float] = None,
    vectorizer_max_df: Optional[int | float] = 0.9,
    vectorizer_strip_accents: Optional[str] = "unicode",
) -> CountVectorizer:
    kwargs: dict[str, Any] = {
        "ngram_range": (int(ngram_range[0]), int(ngram_range[1])),
        "stop_words": vectorizer_stop_words,
        "min_df": _resolved_min_df(n_docs, vectorizer_min_df),
        "lowercase": True,
        "strip_accents": vectorizer_strip_accents,
    }
    if vectorizer_max_df is not None:
        kwargs["max_df"] = vectorizer_max_df
    return CountVectorizer(**kwargs)


def _build_ctfidf_model(
    *,
    reduce_frequent_words: bool = True,
) -> object:
    _require_topic_modeling_dependencies()
    return ClassTfidfTransformer(reduce_frequent_words=bool(reduce_frequent_words))


def _resolve_cluster_reduce_n_components(
    cluster_reduce_n_components: Optional[int],
    umap_n_components: int,
) -> int:
    n_components = umap_n_components if cluster_reduce_n_components is None else cluster_reduce_n_components
    n_components_int = int(n_components)
    if n_components_int < 1:
        raise ValueError("cluster_reduce_n_components must be at least 1")
    return n_components_int


def _build_cluster_reducer(
    *,
    cluster_reduce_method: str,
    cluster_reduce_n_components: int,
    n_samples: int,
    n_features: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    random_state: int,
) -> object:
    method = str(cluster_reduce_method).lower()
    if method == "umap":
        _require_topic_modeling_dependencies(require_umap=True)
        if int(n_samples) <= (int(cluster_reduce_n_components) + 1):
            return PCA(
                n_components=min(int(cluster_reduce_n_components), max(1, int(n_samples))),
                random_state=int(random_state),
            )
        resolved_n_neighbors = max(2, min(int(umap_n_neighbors), int(n_samples) - 1))
        return umap.UMAP(
            n_neighbors=resolved_n_neighbors,
            n_components=int(cluster_reduce_n_components),
            min_dist=float(umap_min_dist),
            metric=str(umap_metric),
            init="random",
            low_memory=True,
            n_jobs=1,
            random_state=int(random_state),
        )
    if method == "pca":
        max_components = min(int(n_samples), int(n_features))
        if int(cluster_reduce_n_components) > max_components:
            raise ValueError(
                "cluster_reduce_n_components cannot exceed min(n_samples, n_features) "
                f"for PCA reduction; got {int(cluster_reduce_n_components)} with "
                f"n_samples={int(n_samples)} and n_features={int(n_features)}"
            )
        return PCA(
            n_components=int(cluster_reduce_n_components),
            random_state=int(random_state),
        )
    raise ValueError("cluster_reduce_method must be one of: 'umap', 'pca'")


def _format_topic_label(topic_model: object, topic_id: int, max_terms: int = 3) -> str:
    if int(topic_id) == -1:
        return "Outlier / noise"
    terms = _topic_terms(topic_model, int(topic_id), limit=max_terms)
    if terms:
        return f"Topic {int(topic_id)}: {', '.join(terms)}"
    return f"Topic {int(topic_id)}"


def _topic_terms(topic_model: object, topic_id: int, limit: Optional[int] = None) -> list[str]:
    if int(topic_id) == -1:
        return []
    try:
        topic_pairs = topic_model.get_topic(int(topic_id)) or []
    except Exception:
        return []
    words = [str(word) for word, _score in topic_pairs if str(word).strip()]
    if limit is not None:
        return words[: int(limit)]
    return words


def _resolved_topic_ids_for_plot(
    topic_model: object,
    *,
    topics: Optional[Sequence[int]] = None,
    top_n_topics: Optional[int] = None,
) -> list[int]:
    if topics is not None:
        return [int(topic_id) for topic_id in topics]
    try:
        topic_info = topic_model.get_topic_info()
    except Exception as exc:
        raise ValueError("Topic ids could not be derived from topic_model") from exc
    if not isinstance(topic_info, pd.DataFrame) or "Topic" not in topic_info.columns:
        raise ValueError("Topic ids could not be derived from topic_model")
    topic_info_use = topic_info.copy()
    if "Count" in topic_info_use.columns:
        topic_info_use = topic_info_use.sort_values("Count", ascending=False)
    topic_ids = [int(topic_id) for topic_id in topic_info_use["Topic"].tolist() if int(topic_id) != -1]
    if top_n_topics is not None:
        resolved_n = int(top_n_topics)
        if resolved_n < 1:
            raise ValueError("top_n_topics must be at least 1")
        topic_ids = topic_ids[:resolved_n]
    return topic_ids


def _to_dense_2d_array(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Topic representations must be a 2D matrix")
    return arr


def _project_topic_representations_2d(representations: np.ndarray, *, random_state: int = 42) -> np.ndarray:
    n_rows = int(representations.shape[0])
    if n_rows <= 0:
        return np.empty((0, 2), dtype=float)
    if n_rows == 1:
        return np.array([[0.0, 0.0]], dtype=float)
    if n_rows == 2:
        return np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float)
    reducer = PCA(
        n_components=min(2, int(representations.shape[1]), n_rows),
        random_state=int(random_state),
    )
    coords = reducer.fit_transform(representations)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(n_rows, dtype=float)])
    return coords[:, :2]


def _build_small_topic_intertopic_map(
    topic_model: object,
    *,
    topics: Optional[Sequence[int]] = None,
    top_n_topics: Optional[int] = None,
    use_ctfidf: bool = False,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> go.Figure:
    topic_ids = _resolved_topic_ids_for_plot(
        topic_model,
        topics=topics,
        top_n_topics=top_n_topics,
    )
    if not topic_ids:
        raise ValueError("No non-noise topics are available to visualize")

    topic_info = topic_model.get_topic_info()
    topic_info = topic_info.loc[topic_info["Topic"].isin(topic_ids), :].copy()
    count_map = {}
    if "Count" in topic_info.columns:
        count_map = {int(row["Topic"]): int(row["Count"]) for _, row in topic_info.iterrows()}

    all_topics = sorted([int(topic_id) for topic_id in getattr(topic_model, "get_topics")().keys() if int(topic_id) != -1])
    topic_indices = [all_topics.index(int(topic_id)) for topic_id in topic_ids]

    if use_ctfidf:
        base = getattr(topic_model, "c_tf_idf_", None)
    else:
        base = getattr(topic_model, "topic_embeddings_", None)
    if base is None:
        base = np.eye(len(all_topics), dtype=float)

    base_arr = _to_dense_2d_array(base)
    if base_arr.shape[0] < max(topic_indices) + 1:
        base_arr = _to_dense_2d_array(np.eye(len(all_topics), dtype=float))
    coords = _project_topic_representations_2d(base_arr[topic_indices], random_state=42)

    topic_words = [" | ".join(_topic_terms(topic_model, topic_id, limit=5)) for topic_id in topic_ids]
    topic_counts = np.array([float(count_map.get(int(topic_id), 1)) for topic_id in topic_ids], dtype=float)
    if np.allclose(topic_counts.max(), topic_counts.min()):
        marker_sizes = np.full(len(topic_ids), 36.0, dtype=float)
    else:
        marker_sizes = 30.0 + 36.0 * (topic_counts - topic_counts.min()) / (topic_counts.max() - topic_counts.min())

    customdata = np.column_stack(
        [
            np.asarray(topic_ids, dtype=int),
            np.asarray(topic_words, dtype=object),
            np.asarray(topic_counts, dtype=int),
        ]
    )
    fig = go.Figure(
        data=[
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers+text",
                text=[f"Topic {int(topic_id)}" for topic_id in topic_ids],
                textposition="top center",
                customdata=customdata,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Words: %{customdata[1]}<br>"
                    "Count: %{customdata[2]}<extra></extra>"
                ),
                marker=dict(
                    size=marker_sizes.tolist(),
                    color=topic_counts.tolist(),
                    colorscale="Blues",
                    showscale=False,
                    opacity=0.9,
                    line=dict(width=1, color="#455A64"),
                ),
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        title=title or "<b>Intertopic Distance Map</b>",
        width=None if width is None else int(width),
        height=None if height is None else int(height),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def _normalize_representative_docs(docs: Sequence[str], representative_docs: int) -> str:
    cleaned = []
    for doc in docs[: max(0, int(representative_docs))]:
        text = str(doc).replace("\r", " ").replace("\n", " ").strip()
        if not text:
            continue
        cleaned.append(text[:160] + ("..." if len(text) > 160 else ""))
    return " || ".join(cleaned)


def _document_probabilities(
    topic_model: object,
    texts: Sequence[str],
    topics: np.ndarray,
    probs: Optional[np.ndarray],
) -> np.ndarray:
    try:
        doc_info = topic_model.get_document_info(list(texts))
        if len(doc_info) == len(texts) and "Probability" in doc_info.columns:
            values = pd.to_numeric(doc_info["Probability"], errors="coerce").to_numpy(dtype=float)
            if len(values) == len(texts):
                return values
    except Exception:
        pass

    if probs is None:
        return np.full(len(texts), np.nan, dtype=float)

    probs_arr = np.asarray(probs, dtype=float)
    if probs_arr.ndim == 1:
        if probs_arr.shape[0] != len(texts):
            raise ValueError("Probability array does not align with topics")
        return probs_arr

    if probs_arr.ndim != 2 or probs_arr.shape[0] != len(texts):
        raise ValueError("Probability array must align with topics")

    topic_ids: list[int] = []
    try:
        info = topic_model.get_topic_info()
        if isinstance(info, pd.DataFrame) and "Topic" in info.columns:
            topic_ids = [int(topic_id) for topic_id in info["Topic"].tolist() if int(topic_id) != -1]
    except Exception:
        topic_ids = []

    topic_index = {topic_id: idx for idx, topic_id in enumerate(topic_ids)}
    out = np.full(len(texts), np.nan, dtype=float)
    for row_idx, topic_id in enumerate(np.asarray(topics, dtype=int)):
        if int(topic_id) == -1:
            continue
        col_idx = topic_index.get(int(topic_id))
        if col_idx is not None and col_idx < probs_arr.shape[1]:
            out[row_idx] = float(probs_arr[row_idx, col_idx])
        else:
            row = probs_arr[row_idx]
            if np.isfinite(row).any():
                out[row_idx] = float(np.nanmax(row))
    return out


def fit_topic_model(
    texts: Sequence[str],
    embeddings: np.ndarray,
    *,
    cluster_reduce_method: str = "umap",
    cluster_reduce_n_components: Optional[int] = None,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    umap_min_dist: float = 0.0,
    umap_metric: str = "cosine",
    hdbscan_min_cluster_size: int = 10,
    hdbscan_min_samples: Optional[int] = None,
    top_n_words: int = 10,
    ngram_range: tuple[int, int] = (1, 2),
    vectorizer_stop_words: str | list[str] | None = "english",
    vectorizer_min_df: Optional[int | float] = None,
    vectorizer_max_df: Optional[int | float] = 0.9,
    vectorizer_strip_accents: Optional[str] = "unicode",
    reduce_frequent_words: bool = True,
    calculate_probabilities: bool = True,
    random_state: int = 42,
) -> tuple[object, np.ndarray, np.ndarray | None]:
    """
    Fit a BERTopic model using precomputed sentence embeddings.
    """
    text_list = ["" if text is None else str(text) for text in texts]
    emb_arr = np.asarray(embeddings)
    if emb_arr.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if len(text_list) != emb_arr.shape[0]:
        raise ValueError("texts and embeddings must have the same number of rows")
    if len(text_list) == 0:
        raise ValueError("texts must contain at least one non-empty document")
    if any(not text.strip() for text in text_list):
        raise ValueError("texts must not contain empty documents")
    if len(ngram_range) != 2:
        raise ValueError("ngram_range must contain exactly two integers")
    reduce_method = str(cluster_reduce_method).lower()
    _require_topic_modeling_dependencies(
        require_hdbscan=True,
        require_umap=reduce_method == "umap",
    )
    resolved_reduce_n_components = _resolve_cluster_reduce_n_components(
        cluster_reduce_n_components,
        umap_n_components,
    )
    vectorizer_model = _build_vectorizer_model(
        len(text_list),
        ngram_range=ngram_range,
        vectorizer_stop_words=vectorizer_stop_words,
        vectorizer_min_df=vectorizer_min_df,
        vectorizer_max_df=vectorizer_max_df,
        vectorizer_strip_accents=vectorizer_strip_accents,
    )
    ctfidf_model = _build_ctfidf_model(
        reduce_frequent_words=reduce_frequent_words,
    )
    cluster_reducer_model = _build_cluster_reducer(
        cluster_reduce_method=reduce_method,
        cluster_reduce_n_components=resolved_reduce_n_components,
        n_samples=len(text_list),
        n_features=int(emb_arr.shape[1]),
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        random_state=random_state,
    )

    hdbscan_kwargs: dict[str, Any] = {
        "min_cluster_size": int(hdbscan_min_cluster_size),
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "prediction_data": True,
    }
    if hdbscan_min_samples is not None:
        hdbscan_kwargs["min_samples"] = int(hdbscan_min_samples)
    hdbscan_model = hdbscan.HDBSCAN(**hdbscan_kwargs)

    topic_model = BERTopic(
        umap_model=cluster_reducer_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        top_n_words=int(top_n_words),
        n_gram_range=(int(ngram_range[0]), int(ngram_range[1])),
        calculate_probabilities=bool(calculate_probabilities),
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(text_list, embeddings=emb_arr)
    return topic_model, np.asarray(topics, dtype=int), None if probs is None else np.asarray(probs)


def update_topic_representation(
    topic_model: object,
    texts: Sequence[str],
    *,
    topics: Optional[Sequence[int]] = None,
    top_n_words: int = 10,
    ngram_range: tuple[int, int] = (1, 2),
    vectorizer_stop_words: str | list[str] | None = "english",
    vectorizer_min_df: Optional[int | float] = None,
    vectorizer_max_df: Optional[int | float] = 0.9,
    vectorizer_strip_accents: Optional[str] = "unicode",
    reduce_frequent_words: bool = True,
) -> object:
    """
    Refresh c-TF-IDF topic representations without recomputing clusters.
    """
    _require_topic_modeling_dependencies()

    text_list = ["" if text is None else str(text) for text in texts]
    if len(text_list) == 0:
        raise ValueError("texts must contain at least one non-empty document")
    if any(not text.strip() for text in text_list):
        raise ValueError("texts must not contain empty documents")
    if len(ngram_range) != 2:
        raise ValueError("ngram_range must contain exactly two integers")

    vectorizer_model = _build_vectorizer_model(
        len(text_list),
        ngram_range=ngram_range,
        vectorizer_stop_words=vectorizer_stop_words,
        vectorizer_min_df=vectorizer_min_df,
        vectorizer_max_df=vectorizer_max_df,
        vectorizer_strip_accents=vectorizer_strip_accents,
    )
    ctfidf_model = _build_ctfidf_model(
        reduce_frequent_words=reduce_frequent_words,
    )
    topic_list = None if topics is None else [int(topic_id) for topic_id in topics]
    topic_model.update_topics(
        text_list,
        topics=topic_list,
        top_n_words=int(top_n_words),
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
    )
    return topic_model


def resolve_projection_from_topic_match(
    manual_projection_opts: Mapping[str, Any],
    *,
    match_topic_reducer: bool = False,
    cluster_opts: Optional[Mapping[str, Any]] = None,
    default_random_state: int = 1,
) -> dict[str, Any]:
    """
    Resolve projection settings for the Embeddings tab, optionally matching the
    fitted BERTopic clustering reducer family and compatible settings.
    """
    resolved = dict(manual_projection_opts)
    resolved["match_topic_reducer"] = bool(match_topic_reducer)
    resolved["matched_cluster_opts"] = None
    resolved["random_state"] = int(resolved.get("random_state", default_random_state))
    if not match_topic_reducer:
        return resolved
    if cluster_opts is None:
        raise ValueError("cluster_opts are required when match_topic_reducer=True")

    cluster_snapshot = dict(cluster_opts)
    reduce_method = str(cluster_snapshot.get("cluster_reduce_method", "umap")).lower()
    if reduce_method not in {"umap", "pca"}:
        raise ValueError("Matched topic projection requires a fitted topic reducer of 'umap' or 'pca'")

    n_components = int(resolved.get("n_components", 2))
    matched = {
        "reduce_method": reduce_method,
        "n_components": n_components,
        "umap_n_neighbors": None,
        "umap_min_dist": None,
        "umap_metric": "cosine",
        "tsne_perplexity": None,
        "tsne_learning_rate": None,
        "tsne_n_iter": None,
        "tsne_metric": "cosine",
        "random_state": int(cluster_snapshot.get("random_state", 42)),
        "match_topic_reducer": True,
        "matched_cluster_opts": cluster_snapshot,
    }
    if reduce_method == "umap":
        matched["umap_n_neighbors"] = int(cluster_snapshot.get("umap_n_neighbors", 15))
        matched["umap_min_dist"] = float(cluster_snapshot.get("umap_min_dist", 0.0))
        matched["umap_metric"] = str(cluster_snapshot.get("umap_metric", "cosine"))
    return matched


def build_topic_plot(
    topic_model: object,
    plot_type: str,
    *,
    docs: Optional[Sequence[str]] = None,
    topics: Optional[Sequence[int]] = None,
    embeddings: Optional[np.ndarray] = None,
    reduced_embeddings: Optional[np.ndarray] = None,
    top_n_topics: Optional[int] = None,
    sample: Optional[float] = None,
    hide_annotations: Optional[bool] = None,
    hide_document_hover: Optional[bool] = None,
    use_ctfidf: Optional[bool] = None,
    n_words: Optional[int] = None,
    autoscale: Optional[bool] = None,
    n_clusters: Optional[int] = None,
    orientation: Optional[str] = None,
    log_scale: Optional[bool] = None,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    """
    Dispatch to a supported BERTopic visualization method with validation.
    """
    plot_specs: dict[str, dict[str, Any]] = {
        "intertopic_map": {
            "method": "visualize_topics",
            "allowed": {"topics", "top_n_topics", "use_ctfidf", "title", "width", "height"},
        },
        "barchart": {
            "method": "visualize_barchart",
            "allowed": {"topics", "top_n_topics", "n_words", "autoscale", "title", "width", "height"},
        },
        "heatmap": {
            "method": "visualize_heatmap",
            "allowed": {"topics", "top_n_topics", "n_clusters", "use_ctfidf", "title", "width", "height"},
        },
        "hierarchy": {
            "method": "visualize_hierarchy",
            "allowed": {"topics", "top_n_topics", "orientation", "use_ctfidf", "title", "width", "height"},
        },
        "term_rank": {
            "method": "visualize_term_rank",
            "allowed": {"topics", "log_scale", "title", "width", "height"},
        },
        "documents": {
            "method": "visualize_documents",
            "allowed": {
                "docs",
                "topics",
                "embeddings",
                "reduced_embeddings",
                "sample",
                "hide_annotations",
                "hide_document_hover",
                "title",
                "width",
                "height",
            },
        },
    }
    plot_key = str(plot_type).lower()
    spec = plot_specs.get(plot_key)
    if spec is None:
        raise ValueError(
            "plot_type must be one of: intertopic_map, barchart, heatmap, hierarchy, "
            "term_rank, documents"
        )

    doc_list = None if docs is None else ["" if doc is None else str(doc) for doc in docs]
    topic_list = None if topics is None else [int(topic_id) for topic_id in topics]
    if topic_list is not None and len(topic_list) == 0:
        raise ValueError("topics must contain at least one topic id when provided")
    if topic_list is not None and top_n_topics is not None:
        raise ValueError("Provide either topics or top_n_topics, not both")
    if top_n_topics is not None and "top_n_topics" not in spec["allowed"] and "topics" in spec["allowed"]:
        try:
            topic_info = topic_model.get_topic_info()
        except Exception as exc:
            raise ValueError(
                f"plot_type='{plot_key}' does not support top_n_topics directly and topic ids could not be derived"
            ) from exc
        if not isinstance(topic_info, pd.DataFrame) or "Topic" not in topic_info.columns:
            raise ValueError(
                f"plot_type='{plot_key}' does not support top_n_topics directly and topic ids could not be derived"
            )
        topic_info_use = topic_info.copy()
        if "Count" in topic_info_use.columns:
            topic_info_use = topic_info_use.sort_values("Count", ascending=False)
        topic_candidates = [
            int(topic_id)
            for topic_id in topic_info_use["Topic"].tolist()
            if int(topic_id) != -1
        ]
        resolved_n = int(top_n_topics)
        if resolved_n < 1:
            raise ValueError("top_n_topics must be at least 1")
        topic_list = topic_candidates[:resolved_n]
        top_n_topics = None

    option_values = {
        "docs": doc_list,
        "topics": topic_list,
        "embeddings": None if embeddings is None else np.asarray(embeddings),
        "reduced_embeddings": None if reduced_embeddings is None else np.asarray(reduced_embeddings),
        "top_n_topics": None if top_n_topics is None else int(top_n_topics),
        "sample": None if sample is None else float(sample),
        "hide_annotations": hide_annotations,
        "hide_document_hover": hide_document_hover,
        "use_ctfidf": use_ctfidf,
        "n_words": None if n_words is None else int(n_words),
        "autoscale": autoscale,
        "n_clusters": None if n_clusters is None else int(n_clusters),
        "orientation": None if orientation is None else str(orientation),
        "log_scale": log_scale,
        "title": None if title is None else str(title),
        "width": None if width is None else int(width),
        "height": None if height is None else int(height),
    }

    for arg_name, arg_value in option_values.items():
        if arg_value is None:
            continue
        if arg_name not in spec["allowed"]:
            raise ValueError(f"{arg_name} is not supported for plot_type='{plot_key}'")
    if option_values["top_n_topics"] is not None and option_values["top_n_topics"] < 1:
        raise ValueError("top_n_topics must be at least 1")
    if plot_key == "documents" and option_values["docs"] is None:
        raise ValueError(f"docs are required for plot_type='{plot_key}'")
    if option_values["sample"] is not None and not (0.0 < option_values["sample"] <= 1.0):
        raise ValueError("sample must be between 0 and 1")
    if option_values["n_words"] is not None and option_values["n_words"] < 1:
        raise ValueError("n_words must be at least 1")
    if option_values["n_clusters"] is not None and option_values["n_clusters"] < 1:
        raise ValueError("n_clusters must be at least 1")
    if option_values["width"] is not None and option_values["width"] < 1:
        raise ValueError("width must be at least 1")
    if option_values["height"] is not None and option_values["height"] < 1:
        raise ValueError("height must be at least 1")
    if option_values["orientation"] is not None and option_values["orientation"] not in {"left", "bottom"}:
        raise ValueError("orientation must be either 'left' or 'bottom'")
    if option_values["reduced_embeddings"] is not None:
        reduced_arr = option_values["reduced_embeddings"]
        if reduced_arr.ndim != 2 or reduced_arr.shape[1] != 2:
            raise ValueError("reduced_embeddings must be a 2D array with exactly 2 columns")
        if doc_list is not None and reduced_arr.shape[0] != len(doc_list):
            raise ValueError("reduced_embeddings must align with docs")
    if option_values["embeddings"] is not None:
        emb_arr = option_values["embeddings"]
        if emb_arr.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if doc_list is not None and emb_arr.shape[0] != len(doc_list):
            raise ValueError("embeddings must align with docs")
    method = getattr(topic_model, spec["method"], None)
    if method is None:
        raise AttributeError(f"topic_model does not support {spec['method']}()")
    method_module = str(getattr(method, "__module__", ""))
    if (
        plot_key == "intertopic_map"
        and "bertopic" in method_module.lower()
        and len(_resolved_topic_ids_for_plot(
            topic_model,
            topics=topic_list,
            top_n_topics=top_n_topics,
        )) <= 3
    ):
        return _build_small_topic_intertopic_map(
            topic_model,
            topics=topic_list,
            top_n_topics=top_n_topics,
            use_ctfidf=bool(option_values["use_ctfidf"]),
            title=option_values["title"],
            width=option_values["width"],
            height=option_values["height"],
        )

    call_kwargs = {
        key: value
        for key, value in option_values.items()
        if value is not None and key in spec["allowed"]
    }
    figure = method(**call_kwargs)
    if plot_key == "documents" and isinstance(figure, go.Figure):
        payload = figure.to_plotly_json()
        changed = False
        for trace in payload.get("data", []):
            if trace.get("type") == "scattergl":
                trace["type"] = "scatter"
                changed = True
        if changed:
            return go.Figure(payload)
    return figure


def build_topic_assignments(
    meta_df: pd.DataFrame,
    texts: Sequence[str],
    topics: np.ndarray,
    probs: Optional[np.ndarray],
    topic_model: object,
) -> pd.DataFrame:
    """
    Combine metadata, texts, topic ids, and document-level topic probabilities.
    """
    if len(meta_df) != len(texts):
        raise ValueError("meta_df and texts must have the same number of rows")

    topics_arr = np.asarray(topics, dtype=int)
    if len(topics_arr) != len(texts):
        raise ValueError("topics and texts must have the same number of rows")

    topic_probs = _document_probabilities(topic_model, texts, topics_arr, probs)
    assignments = meta_df.reset_index(drop=True).copy()
    assignments["text"] = list(texts)
    assignments["topic_id"] = topics_arr
    assignments["topic_label"] = [
        _format_topic_label(topic_model, int(topic_id))
        for topic_id in topics_arr
    ]
    assignments["topic_probability"] = topic_probs
    assignments["is_outlier"] = assignments["topic_id"].eq(-1)

    ordered_cols = [
        "index",
        "source_column",
        "text",
        "topic_id",
        "topic_label",
        "topic_probability",
        "is_outlier",
    ]
    remaining_cols = [col for col in assignments.columns if col not in ordered_cols]
    return assignments[ordered_cols + remaining_cols]


def summarize_topics(
    topic_model: object,
    assignments_df: pd.DataFrame,
    *,
    representative_docs: int = 3,
) -> pd.DataFrame:
    """
    Build a compact per-topic summary table from a fitted BERTopic model.
    """
    required_cols = {"topic_id", "topic_label", "text"}
    missing = required_cols.difference(assignments_df.columns)
    if missing:
        raise ValueError(
            "assignments_df is missing required columns: "
            + ", ".join(sorted(missing))
        )

    total_docs = int(len(assignments_df))
    if total_docs == 0:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "topic_label",
                "count",
                "share",
                "top_terms",
                "representative_docs",
            ]
        )

    topic_info = None
    try:
        topic_info = topic_model.get_topic_info()
    except Exception:
        topic_info = None

    if isinstance(topic_info, pd.DataFrame) and "Topic" in topic_info.columns:
        topic_ids = [int(topic_id) for topic_id in topic_info["Topic"].tolist()]
    else:
        topic_ids = [int(topic_id) for topic_id in assignments_df["topic_id"].drop_duplicates().tolist()]

    representative_doc_map: dict[int, Sequence[str]] = {}
    try:
        representative_doc_map = topic_model.get_representative_docs() or {}
    except Exception:
        representative_doc_map = {}

    rows: list[dict[str, Any]] = []
    for topic_id in topic_ids:
        topic_mask = assignments_df["topic_id"].eq(int(topic_id))
        topic_rows = assignments_df.loc[topic_mask]
        if topic_rows.empty:
            continue
        topic_label = _format_topic_label(topic_model, int(topic_id))
        if int(topic_id) == -1:
            top_terms = ""
        else:
            top_terms = ", ".join(_topic_terms(topic_model, int(topic_id)))
        rep_docs = representative_doc_map.get(int(topic_id))
        if not rep_docs:
            rep_docs = topic_rows["text"].tolist()
        elif isinstance(rep_docs, str):
            rep_docs = [rep_docs]
        rows.append(
            {
                "topic_id": int(topic_id),
                "topic_label": topic_label,
                "count": int(len(topic_rows)),
                "share": float(len(topic_rows) / total_docs),
                "top_terms": top_terms,
                "representative_docs": _normalize_representative_docs(
                    list(rep_docs),
                    representative_docs=int(representative_docs),
                ),
            }
        )

    return pd.DataFrame(rows)


def merge_topic_assignments(
    plot_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach topic assignment columns onto a projection table without reordering rows.
    """
    if len(plot_df) != len(assignments_df):
        raise ValueError("plot_df and assignments_df must have the same number of rows")

    plot_reset = plot_df.reset_index(drop=True).copy()
    assign_reset = assignments_df.reset_index(drop=True)
    for col in ("index", "source_column", "text"):
        if col in plot_reset.columns and col in assign_reset.columns:
            if not plot_reset[col].equals(assign_reset[col]):
                raise ValueError(f"plot_df and assignments_df must align on '{col}'")

    for col in ("topic_id", "topic_label", "topic_probability", "is_outlier"):
        if col not in assign_reset.columns:
            raise ValueError(f"assignments_df is missing required column '{col}'")
        plot_reset[col] = assign_reset[col].to_numpy()
    return plot_reset


__all__ = [
    "fit_topic_model",
    "build_topic_assignments",
    "summarize_topics",
    "merge_topic_assignments",
    "update_topic_representation",
    "build_topic_plot",
]
