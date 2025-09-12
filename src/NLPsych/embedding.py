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
    metric: str = "cosine"
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
            reducer = umap.UMAP(n_components=n_components, metric=metric, random_state=random_state)
            return reducer.fit_transform(embeddings)

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


def plot_projection(plot_df: pd.DataFrame, 
                    n_components: int = 2, 
                    color_by: Optional[str] = "source_column", 
                    point_size: int = 7
):
    """
    Build a Plotly figure for the projection.
    """
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

