"""
NLPsych: Tools for text analysis, embeddings, and lightweight modeling,
with optional Streamlit app for interactive use.
"""

from importlib.metadata import version

# Expose top-level version
__version__ = version("NLPsych")

from .utils import get_spacy_pipeline_base, get_st_model_base
from .descriptive_stats import spacy_descriptive_stats
from .embedding import (
    embed_text_columns_simple_base as embed_text_columns_simple,
    reduce_embeddings,
    build_plot_df,
    plot_projection,
)
from .modeling import auto_cv_with_permutation
from .report import build_report_payload

__all__ = [
    "get_spacy_pipeline_base",
    "get_st_model_base",
    "spacy_descriptive_stats",
    "embed_text_columns_simple",
    "reduce_embeddings",
    "build_plot_df",
    "plot_projection",
    "auto_cv_with_permutation",
    "build_report_payload",
]
