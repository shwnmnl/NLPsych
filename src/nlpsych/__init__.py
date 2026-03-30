"""
nlpsych: Tools for text analysis, embeddings, and lightweight modeling,
with optional Streamlit app for interactive use.
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
    "auto_cv_with_permutation",
    "apply_multiple_comparisons_correction",
    "build_report_payload",
]
