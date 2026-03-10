from __future__ import annotations
from typing import List, Optional, Literal, Tuple

import warnings
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from nlpsych.utils import get_st_model_base

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def embed_text_columns_simple_base(
        series_list: List[pd.Series], 
        model_name="all-MiniLM-L6-v2", 
        normalize=True
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Encode one or more text Series with SentenceTransformers.

    Parameters
    ----------
    series_list
        List of pandas Series. Each row that is a non empty string will be used.
        The Series.name is preserved as source_column.
    model_name
        Hugging Face model name or local path.
    normalize
        If True, return L2 normalized embeddings.

    Returns
    -------
    meta_df
        DataFrame with two columns, source_column and index, aligned to embeddings.
    embeddings
        NumPy array of shape (n_samples, dim).
    texts
        The raw texts in the same order as the embeddings.
    """
    model = get_st_model_base(model_name)
    texts: List[str] = []
    meta: List[dict] = []
    for s in series_list:
        if not isinstance(s, pd.Series):
            raise ValueError("All inputs must be pandas Series")
        for idx, text in s.dropna().items():
            if isinstance(text, str) and text.strip():
                texts.append(text)
                meta.append({"index": idx, "source_column": s.name})

    if not texts:
        return pd.DataFrame(columns=["index", "source_column"]), np.empty((0, 0)), []

    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=normalize)
    meta_df = pd.DataFrame(meta)
    return meta_df, embeddings, texts

    
def reduce_embeddings(
    embeddings: np.ndarray,
    method: Literal["umap", "tsne", "pca"] = "pca",
    n_components: int = 2,
    random_state: int = 42,
    metric: str = "cosine",
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
    tsne_perplexity: Optional[float] = None,
    tsne_learning_rate: Optional[float] = None,
    tsne_n_iter: Optional[int] = None,
) -> np.ndarray:
    """
    Reduce embeddings to 2 or 3 dimensions using UMAP, t SNE, or PCA.

    Returns
    -------
    Z
        Array of shape (n_samples, n_components).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")
    n = embeddings.shape[0]
    if n == 0:
        return np.empty((0, n_components))
    
    if method == "umap":
        if not HAS_UMAP:
            warnings.warn("UMAP not available. Falling back to PCA.", RuntimeWarning)
            method = "pca"
        else:
            umap_kwargs = {
                "n_components": n_components,
                "metric": metric,
                "random_state": random_state,
            }
            if umap_n_neighbors is not None:
                umap_kwargs["n_neighbors"] = int(umap_n_neighbors)
            if umap_min_dist is not None:
                umap_kwargs["min_dist"] = float(umap_min_dist)
            reducer = umap.UMAP(**umap_kwargs)
            return reducer.fit_transform(embeddings)

    if method == "tsne":
        n = len(embeddings)
        # Perplexity must be less than n and usually less than about n/3
        safe_perp = max(5, min(30, (n - 1) // 3 if n > 10 else max(2, n // 2)))
        if tsne_perplexity is not None:
            try:
                requested = float(tsne_perplexity)
            except (TypeError, ValueError):
                requested = safe_perp
            max_perp = max(1.0, float(n - 1))
            if requested < 1.0:
                warnings.warn("t-SNE perplexity must be >= 1. Using 1.0.", RuntimeWarning)
                requested = 1.0
            if requested >= max_perp:
                adj = max(1.0, max_perp - 1.0) if max_perp > 1.0 else 1.0
                warnings.warn(
                    f"t-SNE perplexity {requested} is too large for n={n}. Using {adj}.",
                    RuntimeWarning
                )
                requested = adj
            safe_perp = requested
        tsne_kwargs = {
            "n_components": n_components,
            "metric": "cosine" if metric == "cosine" else "euclidean",
            "init": "pca",
            "random_state": random_state,
            "perplexity": float(safe_perp),
            "learning_rate": float(tsne_learning_rate) if tsne_learning_rate is not None else "auto",
        }
        if tsne_n_iter is not None:
            try:
                from inspect import signature

                sig = signature(TSNE)
                if "n_iter" in sig.parameters:
                    tsne_kwargs["n_iter"] = int(tsne_n_iter)
                elif "max_iter" in sig.parameters:
                    tsne_kwargs["max_iter"] = int(tsne_n_iter)
            except Exception:
                tsne_kwargs["n_iter"] = int(tsne_n_iter)
        reducer = TSNE(**tsne_kwargs)
        return reducer.fit_transform(embeddings)

    # PCA default
    reducer = PCA(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(embeddings)


def build_plot_df(Z: np.ndarray, meta: pd.DataFrame, texts: List[str]) -> pd.DataFrame:
    """
    Combine reduced coordinates with metadata and texts for plotting.
    """
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D array")
    if len(meta) != Z.shape[0] or len(texts) != Z.shape[0]:
        raise ValueError("meta and texts must align with Z rows")
    
    n_components = Z.shape[1]
    cols = [f"dim_{i+1}" for i in range(n_components)]
    plot_df = meta.reset_index(drop=True).copy()
    for i, c in enumerate(cols):
        plot_df[c] = Z[:, i]
    plot_df["text"] = list(texts)
    return plot_df


def plot_projection(
    plot_df: pd.DataFrame,
    n_components: int = 2,
    color_by: Optional[str] = "source_column",
    point_size: int = 7,
    point_opacity: float = 0.9,
    color_discrete_sequence: Optional[list] = None,
    color_continuous_scale: Optional[list] = None,
    template: Optional[str] = "plotly_white",
    plot_bgcolor: Optional[str] = None,
    paper_bgcolor: Optional[str] = None,
    show_legend: bool = True,
    hide_axes: bool = False,
):
    """
    Build a Plotly figure for the projection.
    Styling options let callers override color palettes, opacity, and backgrounds.
    """
    hover_cols = ["text"]
    for c in ["source_column", "index"]:
        if c in plot_df.columns:
            hover_cols.append(c)

    scatter_kwargs = {
        "color": color_by if color_by in plot_df.columns else None,
        "hover_data": hover_cols,
        "opacity": point_opacity,
    }
    if color_discrete_sequence is not None:
        scatter_kwargs["color_discrete_sequence"] = color_discrete_sequence
    if color_continuous_scale is not None:
        scatter_kwargs["color_continuous_scale"] = color_continuous_scale

    if n_components == 2:
        fig = px.scatter(
            plot_df,
            x="dim_1",
            y="dim_2",
            **scatter_kwargs
        )
        fig.update_traces(marker=dict(size=point_size, opacity=point_opacity))
    else:
        fig = px.scatter_3d(
            plot_df,
            x="dim_1",
            y="dim_2",
            z="dim_3",
            **scatter_kwargs
        )
        fig.update_traces(marker=dict(size=point_size, opacity=point_opacity))

    if template:
        fig.update_layout(template=template)
    if plot_bgcolor is not None:
        fig.update_layout(plot_bgcolor=plot_bgcolor)
        if n_components == 3:
            fig.update_layout(scene=dict(bgcolor=plot_bgcolor))
    if paper_bgcolor is not None:
        fig.update_layout(paper_bgcolor=paper_bgcolor)
    fig.update_layout(showlegend=bool(show_legend))
    if not show_legend:
        fig.update_coloraxes(showscale=False)

    if hide_axes:
        if n_components == 2:
            fig.update_xaxes(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                ticks="",
                title_text=None,
            )
            fig.update_yaxes(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                ticks="",
                title_text=None,
            )
        else:
            fig.update_layout(
                scene=dict(
                    xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title_text=None),
                    yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title_text=None),
                    zaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title_text=None),
                )
            )
    return fig
