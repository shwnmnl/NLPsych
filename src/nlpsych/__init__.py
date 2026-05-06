"""
nlpsych: Tools for text analysis, embeddings, and lightweight modeling,
with an integrated Streamlit app for interactive use.
"""

from importlib.metadata import version

# Expose top-level version
__version__ = version("nlpsych")

from .utils import get_spacy_pipeline_base, get_st_model_base
from .descriptive_stats import (
    descriptive_stats,
    build_descriptive_summary_table,
    descriptive_summary_table_to_markdown,
    descriptive_summary_table_to_latex,
)
from .embedding import (
    embed_text_columns_simple_base as embed_text_columns_simple,
    reduce_embeddings,
    build_plot_df,
    plot_projection,
)
from .topic_modeling import (
    fit_topic_model,
    build_topic_assignments,
    summarize_topics,
    build_topic_plot,
    update_topic_representation,
)
from .modeling import auto_cv_with_permutation, apply_multiple_comparisons_correction
from .report import build_report_payload

__all__ = [
    "get_spacy_pipeline_base",
    "get_st_model_base",
    "descriptive_stats",
    "build_descriptive_summary_table",
    "descriptive_summary_table_to_markdown",
    "descriptive_summary_table_to_latex",
    "embed_text_columns_simple",
    "reduce_embeddings",
    "build_plot_df",
    "plot_projection",
    "fit_topic_model",
    "build_topic_assignments",
    "summarize_topics",
    "build_topic_plot",
    "update_topic_representation",
    "auto_cv_with_permutation",
    "apply_multiple_comparisons_correction",
    "build_report_payload",
]
