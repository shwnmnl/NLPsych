import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from nlpsych.topic_modeling import (
    build_topic_assignments,
    build_topic_plot,
    fit_topic_model,
    merge_topic_assignments,
    resolve_projection_from_topic_match,
    summarize_topics,
    update_topic_representation,
)
import nlpsych.topic_modeling as topic_modeling
from nlpsych_app.topic_state import (
    clear_topic_state,
    store_topic_results,
)


class FakeTopicModel:
    def __init__(self):
        self._topics = {
            0: [("therapy", 0.5), ("mood", 0.3), ("anxiety", 0.2)],
            1: [("market", 0.5), ("stocks", 0.3), ("returns", 0.2)],
        }

    def get_topic(self, topic_id: int):
        return self._topics.get(int(topic_id), [])

    def get_topic_info(self):
        return pd.DataFrame(
            {
                "Topic": [0, 1, -1],
                "Count": [2, 2, 1],
            }
        )

    def get_document_info(self, docs):
        return pd.DataFrame(
            {
                "Document": list(docs),
                "Probability": [0.91, 0.87, 0.95, np.nan, 0.89],
            }
        )

    def get_representative_docs(self):
        return {
            0: [
                "Therapy session focused on sleep and mood regulation.",
                "Mood improved after several sessions and less anxiety.",
            ],
            1: [
                "Markets rallied after the earnings report.",
                "Stock returns were volatile across the quarter.",
            ],
        }


class FakeUpdatableTopicModel:
    def __init__(self):
        self.last_update = None

    def update_topics(
        self,
        docs,
        topics=None,
        top_n_words=10,
        vectorizer_model=None,
        ctfidf_model=None,
        representation_model=None,
    ):
        self.last_update = {
            "docs": list(docs),
            "topics": topics,
            "top_n_words": top_n_words,
            "vectorizer_model": vectorizer_model,
            "ctfidf_model": ctfidf_model,
            "representation_model": representation_model,
        }


class FakeReducer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCTFIDF:
    def __init__(self, reduce_frequent_words=False):
        self.reduce_frequent_words = reduce_frequent_words


class FakeBERTopicModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.docs = None
        self.embeddings = None

    def fit_transform(self, docs, embeddings=None):
        self.docs = list(docs)
        self.embeddings = None if embeddings is None else np.asarray(embeddings)
        return [0] * len(self.docs), np.ones(len(self.docs), dtype=float)


class FakePlotTopicModel:
    def __init__(self):
        self.calls = []
        self.topic_embeddings_ = np.array(
            [
                [1.0, 0.0, 0.1],
                [0.2, 0.9, 0.1],
                [0.1, 0.2, 0.95],
            ],
            dtype=float,
        )
        self.c_tf_idf_ = self.topic_embeddings_.copy()
        self.topic_sizes_ = {0: 12, 1: 9, 2: 4}
        self.custom_labels_ = None
        self._outliers = 1

    def _record(self, name: str, **kwargs):
        self.calls.append((name, kwargs))
        return {"plot_name": name, "kwargs": kwargs}

    def get_topic_info(self):
        return pd.DataFrame(
            {
                "Topic": [0, 1, 2, -1],
                "Count": [12, 9, 4, 2],
            }
        )

    def get_topics(self):
        return {
            0: [("therapy", 0.5)],
            1: [("market", 0.5)],
            2: [("weather", 0.5)],
            -1: [],
        }

    def visualize_topics(self, **kwargs):
        return self._record("visualize_topics", **kwargs)

    def visualize_barchart(self, **kwargs):
        return self._record("visualize_barchart", **kwargs)

    def visualize_heatmap(self, **kwargs):
        return self._record("visualize_heatmap", **kwargs)

    def visualize_hierarchy(self, **kwargs):
        return self._record("visualize_hierarchy", **kwargs)

    def visualize_term_rank(self, **kwargs):
        return self._record("visualize_term_rank", **kwargs)

    def visualize_documents(self, **kwargs):
        return self._record("visualize_documents", **kwargs)


class FakeScatterglTopicModel:
    def visualize_documents(self, **kwargs):
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=[0.1, 0.2],
                y=[0.3, 0.4],
                mode="markers",
                name="topic-0",
            )
        )
        return fig


FakePlotTopicModel.visualize_topics.__module__ = "bertopic.plotting._topics"


def _sample_topic_inputs():
    meta_df = pd.DataFrame(
        {
            "index": [10, 11, 12, 13, 14],
            "source_column": ["text_a", "text_a", "text_b", "text_b", "text_b"],
        }
    )
    texts = [
        "Therapy session focused on sleep and mood regulation.",
        "Mood improved after several sessions and less anxiety.",
        "Markets rallied after the earnings report.",
        "This fragment does not belong to any clear topic.",
        "Stock returns were volatile across the quarter.",
    ]
    topics = np.array([0, 0, 1, -1, 1], dtype=int)
    probs = np.array([0.91, 0.87, 0.95, np.nan, 0.89], dtype=float)
    return meta_df, texts, topics, probs


def test_fit_topic_model_raises_clear_error_when_optional_deps_missing(monkeypatch):
    monkeypatch.setattr(topic_modeling, "HAS_BERTOPIC", False)

    with pytest.raises(RuntimeError, match=r'nlpsych\[topics\]'):
        fit_topic_model(["doc one"], np.array([[1.0, 0.0, 0.0]]))


def test_build_topic_assignments_preserves_row_order_and_required_columns():
    meta_df, texts, topics, probs = _sample_topic_inputs()

    assignments = build_topic_assignments(
        meta_df,
        texts,
        topics,
        probs,
        FakeTopicModel(),
    )

    assert assignments["index"].tolist() == meta_df["index"].tolist()
    assert assignments["source_column"].tolist() == meta_df["source_column"].tolist()
    assert assignments["text"].tolist() == texts
    assert assignments["topic_id"].tolist() == topics.tolist()
    assert assignments["is_outlier"].tolist() == [False, False, False, True, False]
    assert assignments.loc[0, "topic_label"].startswith("Topic 0:")
    assert assignments.loc[3, "topic_label"] == "Outlier / noise"
    assert assignments.loc[0, "topic_probability"] == pytest.approx(0.91)
    assert assignments.columns[:7].tolist() == [
        "index",
        "source_column",
        "text",
        "topic_id",
        "topic_label",
        "topic_probability",
        "is_outlier",
    ]


def test_summarize_topics_returns_expected_columns_and_includes_noise():
    meta_df, texts, topics, probs = _sample_topic_inputs()
    assignments = build_topic_assignments(meta_df, texts, topics, probs, FakeTopicModel())

    summary = summarize_topics(FakeTopicModel(), assignments, representative_docs=2)

    assert summary.columns.tolist() == [
        "topic_id",
        "topic_label",
        "count",
        "share",
        "top_terms",
        "representative_docs",
    ]
    assert set(summary["topic_id"].tolist()) == {0, 1, -1}
    noise_row = summary.loc[summary["topic_id"].eq(-1)].iloc[0]
    assert noise_row["topic_label"] == "Outlier / noise"
    assert noise_row["count"] == 1
    assert summary.loc[summary["topic_id"].eq(0), "representative_docs"].iloc[0].startswith("Therapy session")


def test_merge_topic_assignments_attaches_topic_columns_without_reordering():
    meta_df, texts, topics, probs = _sample_topic_inputs()
    assignments = build_topic_assignments(meta_df, texts, topics, probs, FakeTopicModel())
    plot_df = pd.DataFrame(
        {
            "index": meta_df["index"],
            "source_column": meta_df["source_column"],
            "dim_1": [0.1, 0.2, 1.0, 1.2, 1.1],
            "dim_2": [0.0, 0.1, 0.9, 1.1, 1.0],
            "text": texts,
        }
    )

    merged = merge_topic_assignments(plot_df, assignments)

    assert merged["topic_id"].tolist() == topics.tolist()
    assert merged["topic_label"].tolist() == assignments["topic_label"].tolist()
    assert merged["is_outlier"].tolist() == assignments["is_outlier"].tolist()
    assert merged["dim_1"].tolist() == plot_df["dim_1"].tolist()


def test_update_topic_representation_uses_english_cleanup_defaults(monkeypatch):
    monkeypatch.setattr(topic_modeling, "HAS_BERTOPIC", True)
    monkeypatch.setattr(topic_modeling, "HAS_HDBSCAN", True)
    monkeypatch.setattr(topic_modeling, "HAS_TOPIC_UMAP", True)
    monkeypatch.setattr(topic_modeling, "ClassTfidfTransformer", FakeCTFIDF)

    model = FakeUpdatableTopicModel()
    texts = [f"Document {idx} about mood and therapy." for idx in range(25)]
    topics = [0] * 15 + [1] * 10

    updated_model = update_topic_representation(
        model,
        texts,
        topics=topics,
        top_n_words=8,
        ngram_range=(1, 2),
    )

    assert updated_model is model
    assert model.last_update is not None
    assert model.last_update["topics"] == topics
    assert model.last_update["top_n_words"] == 8
    vectorizer = model.last_update["vectorizer_model"]
    assert vectorizer.stop_words == "english"
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.max_df == 0.9
    assert vectorizer.min_df == 2
    assert vectorizer.strip_accents == "unicode"
    assert vectorizer.lowercase is True
    assert model.last_update["ctfidf_model"].reduce_frequent_words is True


def test_fit_topic_model_uses_configured_umap_reducer(monkeypatch):
    monkeypatch.setattr(topic_modeling, "HAS_BERTOPIC", True)
    monkeypatch.setattr(topic_modeling, "HAS_HDBSCAN", True)
    monkeypatch.setattr(topic_modeling, "HAS_TOPIC_UMAP", True)
    monkeypatch.setattr(topic_modeling, "BERTopic", FakeBERTopicModel)
    monkeypatch.setattr(topic_modeling, "ClassTfidfTransformer", FakeCTFIDF)
    monkeypatch.setattr(topic_modeling, "umap", type("FakeUMAPModule", (), {"UMAP": FakeReducer}))
    monkeypatch.setattr(topic_modeling, "hdbscan", type("FakeHDBSCANModule", (), {"HDBSCAN": FakeReducer}))

    topic_model, topics, probs = fit_topic_model(
        ["doc one", "doc two", "doc three", "doc four", "doc five"],
        np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.6, 0.4],
            ],
            dtype=float,
        ),
        cluster_reduce_method="umap",
        cluster_reduce_n_components=3,
        umap_n_neighbors=7,
        umap_min_dist=0.2,
        umap_metric="manhattan",
        hdbscan_min_cluster_size=2,
        hdbscan_min_samples=1,
        top_n_words=5,
        ngram_range=(1, 1),
        random_state=11,
    )

    reducer = topic_model.kwargs["umap_model"]
    assert isinstance(reducer, FakeReducer)
    assert reducer.kwargs == {
        "n_neighbors": 4,
        "n_components": 3,
        "min_dist": 0.2,
        "metric": "manhattan",
        "init": "random",
        "low_memory": True,
        "n_jobs": 1,
        "random_state": 11,
    }
    assert len(topics) == 5
    assert probs.shape == (5,)


def test_fit_topic_model_uses_pca_reducer_without_umap_dependency(monkeypatch):
    monkeypatch.setattr(topic_modeling, "HAS_BERTOPIC", True)
    monkeypatch.setattr(topic_modeling, "HAS_HDBSCAN", True)
    monkeypatch.setattr(topic_modeling, "HAS_TOPIC_UMAP", False)
    monkeypatch.setattr(topic_modeling, "BERTopic", FakeBERTopicModel)
    monkeypatch.setattr(topic_modeling, "ClassTfidfTransformer", FakeCTFIDF)
    monkeypatch.setattr(topic_modeling, "PCA", FakeReducer)
    monkeypatch.setattr(topic_modeling, "hdbscan", type("FakeHDBSCANModule", (), {"HDBSCAN": FakeReducer}))

    topic_model, topics, _probs = fit_topic_model(
        ["doc one", "doc two", "doc three", "doc four"],
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.7, 0.3, 0.0, 0.0],
            ],
            dtype=float,
        ),
        cluster_reduce_method="pca",
        cluster_reduce_n_components=3,
        hdbscan_min_cluster_size=2,
        hdbscan_min_samples=1,
        top_n_words=5,
        ngram_range=(1, 1),
        random_state=13,
    )

    reducer = topic_model.kwargs["umap_model"]
    assert isinstance(reducer, FakeReducer)
    assert reducer.kwargs == {
        "n_components": 3,
        "random_state": 13,
    }
    assert len(topics) == 4


def test_fit_topic_model_rejects_invalid_pca_n_components(monkeypatch):
    monkeypatch.setattr(topic_modeling, "HAS_BERTOPIC", True)
    monkeypatch.setattr(topic_modeling, "HAS_HDBSCAN", True)
    monkeypatch.setattr(topic_modeling, "HAS_TOPIC_UMAP", False)
    monkeypatch.setattr(topic_modeling, "BERTopic", FakeBERTopicModel)
    monkeypatch.setattr(topic_modeling, "ClassTfidfTransformer", FakeCTFIDF)
    monkeypatch.setattr(topic_modeling, "PCA", FakeReducer)
    monkeypatch.setattr(topic_modeling, "hdbscan", type("FakeHDBSCANModule", (), {"HDBSCAN": FakeReducer}))

    with pytest.raises(ValueError, match="min\\(n_samples, n_features\\)"):
        fit_topic_model(
            ["doc one", "doc two"],
            np.array([[1.0, 0.0], [0.9, 0.1]], dtype=float),
            cluster_reduce_method="pca",
            cluster_reduce_n_components=3,
            hdbscan_min_cluster_size=2,
            hdbscan_min_samples=1,
            top_n_words=5,
            ngram_range=(1, 1),
            random_state=13,
        )


def test_fit_topic_model_falls_back_to_umap_n_components_alias(monkeypatch):
    monkeypatch.setattr(topic_modeling, "HAS_BERTOPIC", True)
    monkeypatch.setattr(topic_modeling, "HAS_HDBSCAN", True)
    monkeypatch.setattr(topic_modeling, "HAS_TOPIC_UMAP", False)
    monkeypatch.setattr(topic_modeling, "BERTopic", FakeBERTopicModel)
    monkeypatch.setattr(topic_modeling, "ClassTfidfTransformer", FakeCTFIDF)
    monkeypatch.setattr(topic_modeling, "PCA", FakeReducer)
    monkeypatch.setattr(topic_modeling, "hdbscan", type("FakeHDBSCANModule", (), {"HDBSCAN": FakeReducer}))

    topic_model, _topics, _probs = fit_topic_model(
        ["doc one", "doc two", "doc three", "doc four", "doc five", "doc six"],
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
                [0.7, 0.3, 0.0, 0.0, 0.0, 0.0],
                [0.6, 0.4, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        cluster_reduce_method="pca",
        cluster_reduce_n_components=None,
        umap_n_components=6,
        hdbscan_min_cluster_size=2,
        hdbscan_min_samples=1,
        top_n_words=5,
        ngram_range=(1, 1),
    )

    reducer = topic_model.kwargs["umap_model"]
    assert reducer.kwargs["n_components"] == 6


@pytest.mark.parametrize(
    ("plot_type", "kwargs", "expected_method"),
    [
        ("barchart", {"top_n_topics": 3, "n_words": 7, "autoscale": True}, "visualize_barchart"),
        ("heatmap", {"topics": [0, 1], "n_clusters": 2, "use_ctfidf": False}, "visualize_heatmap"),
        ("hierarchy", {"top_n_topics": 4, "orientation": "bottom", "use_ctfidf": True}, "visualize_hierarchy"),
        ("term_rank", {"topics": [1], "log_scale": True}, "visualize_term_rank"),
        (
            "documents",
            {
                "docs": ["a", "b", "c"],
                "topics": [0, 1],
                "reduced_embeddings": np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]]),
                "sample": 0.5,
                "hide_annotations": True,
                "hide_document_hover": False,
            },
            "visualize_documents",
        ),
    ],
)
def test_build_topic_plot_dispatches_to_expected_method(plot_type, kwargs, expected_method):
    model = FakePlotTopicModel()

    fig = build_topic_plot(model, plot_type, **kwargs)

    assert fig["plot_name"] == expected_method
    assert model.calls[-1][0] == expected_method
    for key, expected in kwargs.items():
        actual = model.calls[-1][1][key]
        if isinstance(expected, np.ndarray):
            assert np.array_equal(actual, expected)
        else:
            assert actual == expected


def test_build_topic_plot_uses_small_topic_fallback_for_intertopic_map():
    model = FakePlotTopicModel()

    fig = build_topic_plot(
        model,
        "intertopic_map",
        topics=[0, 1, 2],
        use_ctfidf=False,
        width=640,
    )

    assert isinstance(fig, go.Figure)
    assert len(model.calls) == 0
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"


def test_build_topic_plot_resolves_top_n_topics_for_document_plots():
    model = FakePlotTopicModel()

    fig = build_topic_plot(
        model,
        "documents",
        docs=["a", "b", "c"],
        top_n_topics=2,
        embeddings=np.array([[1.0, 0.0], [0.8, 0.2], [0.2, 0.8]]),
    )

    assert fig["plot_name"] == "visualize_documents"
    assert model.calls[-1][1]["topics"] == [0, 1]


def test_build_topic_plot_converts_document_scattergl_to_scatter():
    model = FakeScatterglTopicModel()

    fig = build_topic_plot(
        model,
        "documents",
        docs=["a", "b"],
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "scatter"


def test_build_topic_plot_rejects_invalid_arg_combinations():
    model = FakePlotTopicModel()

    with pytest.raises(ValueError, match="either topics or top_n_topics"):
        build_topic_plot(model, "intertopic_map", topics=[0], top_n_topics=2)

    with pytest.raises(ValueError, match="not supported"):
        build_topic_plot(model, "term_rank", n_words=5)

    with pytest.raises(ValueError, match="docs are required"):
        build_topic_plot(model, "documents", topics=[0])

def test_resolve_projection_from_topic_match_returns_manual_projection_when_disabled():
    manual_opts = {
        "reduce_method": "tsne",
        "n_components": 2,
        "umap_n_neighbors": None,
        "umap_min_dist": None,
        "umap_metric": "cosine",
        "tsne_perplexity": 30.0,
        "tsne_learning_rate": 200.0,
        "tsne_n_iter": 1000,
        "tsne_metric": "cosine",
        "random_state": 1,
    }

    resolved = resolve_projection_from_topic_match(
        manual_opts,
        match_topic_reducer=False,
        cluster_opts={"cluster_reduce_method": "umap"},
    )

    assert resolved["reduce_method"] == "tsne"
    assert resolved["match_topic_reducer"] is False
    assert resolved["matched_cluster_opts"] is None
    assert resolved["tsne_perplexity"] == 30.0


def test_resolve_projection_from_topic_match_uses_fitted_umap_settings():
    manual_opts = {
        "reduce_method": "tsne",
        "n_components": 3,
        "umap_n_neighbors": None,
        "umap_min_dist": None,
        "umap_metric": "cosine",
        "tsne_perplexity": 30.0,
        "tsne_learning_rate": 200.0,
        "tsne_n_iter": 1000,
        "tsne_metric": "cosine",
    }
    cluster_opts = {
        "cluster_reduce_method": "umap",
        "cluster_reduce_n_components": 5,
        "umap_n_neighbors": 25,
        "umap_min_dist": 0.05,
        "umap_metric": "manhattan",
        "random_state": 42,
    }

    resolved = resolve_projection_from_topic_match(
        manual_opts,
        match_topic_reducer=True,
        cluster_opts=cluster_opts,
    )

    assert resolved["reduce_method"] == "umap"
    assert resolved["n_components"] == 3
    assert resolved["umap_n_neighbors"] == 25
    assert resolved["umap_min_dist"] == pytest.approx(0.05)
    assert resolved["umap_metric"] == "manhattan"
    assert resolved["tsne_perplexity"] is None
    assert resolved["match_topic_reducer"] is True
    assert resolved["matched_cluster_opts"] == cluster_opts


def test_store_topic_results_clears_cached_topic_plot_state():
    state = {
        "topic_plot_fig": {"old": True},
        "topic_plot_kind": "heatmap",
        "topic_plot_opts": {"width": 800},
    }

    store_topic_results(
        state,
        topic_model="model",
        topic_assignments_df="assignments",
        topic_summary_df="summary",
        cluster_opts={"cluster_reduce_method": "umap"},
        topic_repr_opts={"top_n_words": 10},
    )

    assert state["topic_model"] == "model"
    assert state["topic_assignments_df"] == "assignments"
    assert state["topic_summary_df"] == "summary"
    assert state["cluster_opts"] == {"cluster_reduce_method": "umap"}
    assert state["topic_repr_opts"] == {"top_n_words": 10}
    assert state["topic_plot_fig"] is None
    assert state["topic_plot_kind"] is None
    assert state["topic_plot_opts"] is None


def test_clear_topic_state_clears_topic_results_and_cached_plots():
    state = {
        "topic_model": "model",
        "topic_assignments_df": "assignments",
        "topic_summary_df": "summary",
        "cluster_opts": {"cluster_reduce_method": "pca"},
        "topic_repr_opts": {"top_n_words": 5},
        "topic_plot_fig": {"old": True},
        "topic_plot_kind": "barchart",
        "topic_plot_opts": {"height": 500},
    }

    clear_topic_state(state)

    assert state["topic_model"] is None
    assert state["topic_assignments_df"] is None
    assert state["topic_summary_df"] is None
    assert state["cluster_opts"] is None
    assert state["topic_repr_opts"] is None
    assert state["topic_plot_fig"] is None
    assert state["topic_plot_kind"] is None
    assert state["topic_plot_opts"] is None

def test_fit_topic_model_returns_topics_aligned_to_input_length():
    try:
        __import__("bertopic")
        __import__("hdbscan")
        __import__("umap")
    except Exception as exc:
        pytest.skip(f"topic-modeling runtime deps unavailable: {exc}")

    texts = [
        "Cats purr and nap on the sofa.",
        "Kittens chase toys and sleep indoors.",
        "Dogs bark loudly during the walk.",
        "Puppies chase balls in the yard.",
        "Stocks rallied after stronger earnings.",
        "Markets fell as bond yields increased.",
    ]
    embeddings = np.array(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.95, 0.05, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.05, 0.95, 0.00, 0.00],
            [0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.95, 0.05],
        ],
        dtype=float,
    )

    topic_model, topics, probs = fit_topic_model(
        texts,
        embeddings,
        umap_n_neighbors=2,
        umap_n_components=2,
        umap_min_dist=0.0,
        umap_metric="cosine",
        hdbscan_min_cluster_size=2,
        hdbscan_min_samples=1,
        top_n_words=5,
        ngram_range=(1, 1),
        random_state=7,
    )

    assert topic_model is not None
    assert len(topics) == len(texts)
    assert np.asarray(topics).dtype.kind in {"i", "u"}
    if probs is not None:
        assert np.asarray(probs).shape[0] == len(texts)
