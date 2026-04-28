from __future__ import annotations

from typing import Any, Mapping, MutableMapping


TOPIC_PLOT_STATE_KEYS = (
    "topic_plot_fig",
    "topic_plot_kind",
    "topic_plot_opts",
)

TOPIC_MODEL_STATE_KEYS = (
    "topic_model",
    "topic_assignments_df",
    "topic_summary_df",
    "cluster_opts",
    "topic_repr_opts",
)

def clear_topic_plot_state(state: MutableMapping[str, Any]) -> None:
    """Reset cached BERTopic visualization state."""
    for key in TOPIC_PLOT_STATE_KEYS:
        state[key] = None


def clear_topic_state(state: MutableMapping[str, Any]) -> None:
    """Reset fitted topic-model state and any cached topic plot."""
    for key in TOPIC_MODEL_STATE_KEYS:
        state[key] = None
    clear_topic_plot_state(state)


def store_topic_results(
    state: MutableMapping[str, Any],
    *,
    topic_model: object,
    topic_assignments_df: Any,
    topic_summary_df: Any,
    cluster_opts: Mapping[str, Any],
    topic_repr_opts: Mapping[str, Any],
) -> None:
    """Persist fitted topic outputs and invalidate cached topic plots."""
    state["topic_model"] = topic_model
    state["topic_assignments_df"] = topic_assignments_df
    state["topic_summary_df"] = topic_summary_df
    state["cluster_opts"] = dict(cluster_opts)
    state["topic_repr_opts"] = dict(topic_repr_opts)
    clear_topic_plot_state(state)
