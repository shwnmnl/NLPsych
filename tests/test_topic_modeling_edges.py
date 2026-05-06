import numpy as np
import pandas as pd
import pytest

import nlpsych.topic_modeling as topic_modeling
from nlpsych.topic_modeling import (
    _build_cluster_reducer,
    _document_probabilities,
    build_topic_plot,
    merge_topic_assignments,
    resolve_projection_from_topic_match,
    summarize_topics,
    update_topic_representation,
)


class FakeReducer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakePlotTopicModel:
    def __init__(self):
        self.calls = []

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

    def _record(self, name: str, **kwargs):
        self.calls.append((name, kwargs))
        return {"plot_name": name, "kwargs": kwargs}

    def visualize_term_rank(self, **kwargs):
        return self._record("visualize_term_rank", **kwargs)

    def visualize_documents(self, **kwargs):
        return self._record("visualize_documents", **kwargs)

    def visualize_hierarchy(self, **kwargs):
        return self._record("visualize_hierarchy", **kwargs)


class FakeProbabilityTopicModel:
    def get_document_info(self, docs):
        raise RuntimeError("doc info unavailable")

    def get_topic_info(self):
        return pd.DataFrame({"Topic": [0, 1, -1]})


class FakeRepresentativeDocsTopicModel:
    def get_topic_info(self):
        return pd.DataFrame({"Topic": [0, -1], "Count": [2, 1]})

    def get_topic(self, topic_id: int):
        if int(topic_id) == 0:
            return [("therapy", 0.4), ("mood", 0.3)]
        return []

    def get_representative_docs(self):
        return {0: "A single representative document string."}


class FakeUpdatableTopicModel:
    def __init__(self):
        self.last_update = None

    def update_topics(self, *args, **kwargs):
        self.last_update = {"args": args, "kwargs": kwargs}


def test_build_cluster_reducer_umap_falls_back_to_pca_for_too_few_samples(monkeypatch):
    monkeypatch.setattr(topic_modeling, "HAS_BERTOPIC", True)
    monkeypatch.setattr(topic_modeling, "HAS_TOPIC_UMAP", True)
    monkeypatch.setattr(topic_modeling, "PCA", FakeReducer)

    reducer = _build_cluster_reducer(
        cluster_reduce_method="umap",
        cluster_reduce_n_components=4,
        n_samples=3,
        n_features=8,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        umap_metric="cosine",
        random_state=9,
    )

    assert isinstance(reducer, FakeReducer)
    assert reducer.kwargs == {"n_components": 3, "random_state": 9}


def test_document_probabilities_uses_matrix_fallback_and_rowmax_for_unknown_topics():
    model = FakeProbabilityTopicModel()
    texts = ["doc a", "doc b", "doc c"]
    topics = np.array([0, 9, -1], dtype=int)
    probs = np.array(
        [
            [0.8, 0.2],
            [0.1, 0.9],
            [0.6, 0.4],
        ],
        dtype=float,
    )

    values = _document_probabilities(model, texts, topics, probs)

    assert values[0] == pytest.approx(0.8)
    assert values[1] == pytest.approx(0.9)
    assert np.isnan(values[2])


@pytest.mark.parametrize(
    "probs",
    [
        np.array([0.5, 0.2], dtype=float),
        np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float),
    ],
)
def test_document_probabilities_rejects_misaligned_probability_arrays(probs):
    model = FakeProbabilityTopicModel()
    texts = ["doc a", "doc b", "doc c"]
    topics = np.array([0, 1, 0], dtype=int)

    with pytest.raises(ValueError, match="align with topics"):
        _document_probabilities(model, texts, topics, probs)


def test_resolve_projection_from_topic_match_requires_cluster_opts_and_known_reducer():
    manual_opts = {"reduce_method": "tsne", "n_components": 2}

    with pytest.raises(ValueError, match="cluster_opts are required"):
        resolve_projection_from_topic_match(manual_opts, match_topic_reducer=True, cluster_opts=None)

    with pytest.raises(ValueError, match="requires a fitted topic reducer"):
        resolve_projection_from_topic_match(
            manual_opts,
            match_topic_reducer=True,
            cluster_opts={"cluster_reduce_method": "svd"},
        )


def test_build_topic_plot_derives_topics_for_term_rank_from_top_n_topics():
    model = FakePlotTopicModel()

    fig = build_topic_plot(
        model,
        "term_rank",
        top_n_topics=2,
        log_scale=True,
    )

    assert fig["plot_name"] == "visualize_term_rank"
    assert model.calls[-1][1]["topics"] == [0, 1]
    assert "top_n_topics" not in model.calls[-1][1]


def test_build_topic_plot_validates_numeric_and_shape_constraints():
    model = FakePlotTopicModel()

    with pytest.raises(ValueError, match="sample must be between 0 and 1"):
        build_topic_plot(model, "documents", docs=["a", "b"], sample=0.0)

    with pytest.raises(ValueError, match="reduced_embeddings must be a 2D array with exactly 2 columns"):
        build_topic_plot(
            model,
            "documents",
            docs=["a", "b"],
            reduced_embeddings=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        )

    with pytest.raises(ValueError, match="embeddings must align with docs"):
        build_topic_plot(
            model,
            "documents",
            docs=["a", "b"],
            embeddings=np.array([[0.1, 0.2]]),
        )

    with pytest.raises(ValueError, match="orientation must be either 'left' or 'bottom'"):
        build_topic_plot(model, "hierarchy", orientation="right")


def test_summarize_topics_handles_empty_frames_and_missing_required_columns():
    empty_assignments = pd.DataFrame(columns=["topic_id", "topic_label", "text"])
    summary = summarize_topics(FakeRepresentativeDocsTopicModel(), empty_assignments)

    assert summary.empty
    assert summary.columns.tolist() == [
        "topic_id",
        "topic_label",
        "count",
        "share",
        "top_terms",
        "representative_docs",
    ]

    with pytest.raises(ValueError, match="missing required columns"):
        summarize_topics(FakeRepresentativeDocsTopicModel(), pd.DataFrame({"topic_id": [0]}))


def test_summarize_topics_normalizes_string_representative_docs():
    assignments = pd.DataFrame(
        {
            "topic_id": [0, 0, -1],
            "topic_label": ["Topic 0", "Topic 0", "Outlier / noise"],
            "text": ["doc one", "doc two", "noise doc"],
        }
    )

    summary = summarize_topics(FakeRepresentativeDocsTopicModel(), assignments, representative_docs=1)

    row = summary.loc[summary["topic_id"].eq(0)].iloc[0]
    assert row["top_terms"] == "therapy, mood"
    assert row["representative_docs"] == "A single representative document string."


def test_merge_topic_assignments_detects_alignment_and_missing_columns():
    plot_df = pd.DataFrame(
        {
            "index": [1, 2],
            "source_column": ["a", "b"],
            "text": ["doc a", "doc b"],
            "dim_1": [0.1, 0.2],
            "dim_2": [0.3, 0.4],
        }
    )
    assignments_missing = pd.DataFrame(
        {
            "index": [1, 2],
            "source_column": ["a", "b"],
            "text": ["doc a", "doc b"],
            "topic_id": [0, 1],
            "topic_label": ["Topic 0", "Topic 1"],
            "topic_probability": [0.9, 0.8],
        }
    )
    assignments_bad_alignment = assignments_missing.assign(
        is_outlier=[False, False],
        text=["doc a", "different"],
    )

    with pytest.raises(ValueError, match="missing required column 'is_outlier'"):
        merge_topic_assignments(plot_df, assignments_missing)

    with pytest.raises(ValueError, match="must align on 'text'"):
        merge_topic_assignments(plot_df, assignments_bad_alignment)


def test_update_topic_representation_rejects_empty_texts_and_invalid_ngram(monkeypatch):
    monkeypatch.setattr(topic_modeling, "HAS_BERTOPIC", True)
    monkeypatch.setattr(topic_modeling, "ClassTfidfTransformer", lambda reduce_frequent_words=True: object())

    model = FakeUpdatableTopicModel()

    with pytest.raises(ValueError, match="at least one non-empty document"):
        update_topic_representation(model, [])

    with pytest.raises(ValueError, match="must not contain empty documents"):
        update_topic_representation(model, ["valid", "   "])

    with pytest.raises(ValueError, match="ngram_range must contain exactly two integers"):
        update_topic_representation(model, ["valid"], ngram_range=(1, 2, 3))
