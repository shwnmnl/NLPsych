import numpy as np
import pandas as pd
import pytest

import nlpsych.embedding as embedding
from nlpsych.embedding import (
    build_plot_df,
    embed_text_columns_simple_base,
    plot_projection,
    reduce_embeddings,
)


class FakeSentenceTransformer:
    def __init__(self):
        self.calls = []

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        self.calls.append(
            {
                "texts": list(texts),
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            dtype=float,
        )


def test_embed_text_columns_simple_base_filters_inputs_and_preserves_alignment(monkeypatch):
    model = FakeSentenceTransformer()
    monkeypatch.setattr(embedding, "get_st_model_base", lambda model_name: model)

    s1 = pd.Series(
        ["Alpha text", "   ", None, 123, "Beta text"],
        index=[10, 11, 12, 13, 14],
        name="note",
    )
    s2 = pd.Series(
        ["Gamma text"],
        index=[20],
        name="essay",
    )

    meta_df, emb, texts = embed_text_columns_simple_base(
        [s1, s2],
        model_name="fake-model",
        normalize=False,
    )

    assert texts == ["Alpha text", "Beta text", "Gamma text"]
    assert meta_df.to_dict("records") == [
        {"index": 10, "source_column": "note"},
        {"index": 14, "source_column": "note"},
        {"index": 20, "source_column": "essay"},
    ]
    assert emb.shape == (3, 2)
    assert model.calls == [
        {
            "texts": ["Alpha text", "Beta text", "Gamma text"],
            "convert_to_numpy": True,
            "normalize_embeddings": False,
        }
    ]


def test_embed_text_columns_simple_base_returns_empty_when_no_valid_texts(monkeypatch):
    model = FakeSentenceTransformer()
    monkeypatch.setattr(embedding, "get_st_model_base", lambda model_name: model)

    s = pd.Series([None, "", "   ", 42], name="note")

    meta_df, emb, texts = embed_text_columns_simple_base([s])

    assert meta_df.empty
    assert list(meta_df.columns) == ["index", "source_column"]
    assert emb.shape == (0, 0)
    assert texts == []
    assert model.calls == []


def test_reduce_embeddings_umap_falls_back_to_pca_when_umap_missing(monkeypatch):
    class FakePCA:
        last_kwargs = None

        def __init__(self, **kwargs):
            FakePCA.last_kwargs = kwargs

        def fit_transform(self, X):
            return np.asarray(X)[:, :2]

    monkeypatch.setattr(embedding, "HAS_UMAP", False)
    monkeypatch.setattr(embedding, "PCA", FakePCA)

    X = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ]
    )

    with pytest.warns(RuntimeWarning, match="UMAP not available"):
        Z = reduce_embeddings(X, method="umap", n_components=2, random_state=7)

    assert Z.shape == (3, 2)
    assert FakePCA.last_kwargs == {"n_components": 2, "random_state": 7}


def test_reduce_embeddings_tsne_clips_invalid_perplexity_and_uses_supported_iter_kw(monkeypatch):
    class FakeTSNE:
        last_kwargs = None
        last_X = None

        def __init__(
            self,
            *,
            n_components,
            metric,
            init,
            random_state,
            perplexity,
            learning_rate,
            max_iter=None,
        ):
            FakeTSNE.last_kwargs = {
                "n_components": n_components,
                "metric": metric,
                "init": init,
                "random_state": random_state,
                "perplexity": perplexity,
                "learning_rate": learning_rate,
                "max_iter": max_iter,
            }

        def fit_transform(self, X):
            FakeTSNE.last_X = np.asarray(X)
            return np.zeros((len(X), 2), dtype=float)

    monkeypatch.setattr(embedding, "TSNE", FakeTSNE)

    X = np.arange(18, dtype=float).reshape(6, 3)

    with pytest.warns(RuntimeWarning, match="too large"):
        Z = reduce_embeddings(
            X,
            method="tsne",
            n_components=2,
            metric="cosine",
            tsne_perplexity=99,
            tsne_learning_rate=150,
            tsne_n_iter=750,
            random_state=3,
        )

    assert Z.shape == (6, 2)
    assert FakeTSNE.last_X.shape == (6, 3)
    assert FakeTSNE.last_kwargs == {
        "n_components": 2,
        "metric": "cosine",
        "init": "pca",
        "random_state": 3,
        "perplexity": 4.0,
        "learning_rate": 150.0,
        "max_iter": 750,
    }


def test_build_plot_df_and_plot_projection_hide_axes():
    meta = pd.DataFrame(
        {
            "index": [1, 2],
            "source_column": ["a", "b"],
        }
    )
    texts = ["first", "second"]
    Z = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)

    plot_df = build_plot_df(Z, meta, texts)
    fig = plot_projection(
        plot_df,
        n_components=2,
        color_by="source_column",
        show_legend=False,
        hide_axes=True,
    )

    assert plot_df.columns.tolist() == ["index", "source_column", "dim_1", "dim_2", "text"]
    assert plot_df["text"].tolist() == texts
    assert len(fig.data) == 2
    assert fig.layout.showlegend is False
    assert fig.layout.xaxis.showticklabels is False
    assert fig.layout.yaxis.showticklabels is False
