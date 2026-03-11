from __future__ import annotations
from typing import Iterable, Optional, Tuple, Dict, Any, Literal, Union
import numpy as np
import pandas as pd
import spacy
from collections import Counter
from nlpsych.utils import get_spacy_pipeline_base

# ===== Section: Core text stats =====
SplitOverall = Literal["combined", "per_column", "both"]


def _split_overall_payload(
    overall_obj: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]]
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Normalize overall payloads into combined and per-column components.

    Parameters
    ----------
    overall_obj : Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]]
        Output object returned by ``descriptive_stats`` for any split mode.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]
        ``(combined, per_column)`` dictionaries. Missing parts are empty dicts.
    """
    combined: Dict[str, Any] = {}
    per_column: Dict[str, Dict[str, Any]] = {}
    if not isinstance(overall_obj, dict) or not overall_obj:
        return combined, per_column

    if isinstance(overall_obj.get("combined"), dict):
        combined = dict(overall_obj["combined"])
        per_raw = overall_obj.get("per_column")
        if isinstance(per_raw, dict):
            per_column = {
                str(k): v for k, v in per_raw.items() if isinstance(v, dict)
            }
        return combined, per_column

    if overall_obj and all(isinstance(v, dict) for v in overall_obj.values()):
        per_column = {str(k): v for k, v in overall_obj.items()}
        return combined, per_column

    combined = dict(overall_obj)
    return combined, per_column


def build_descriptive_summary_table(
    stats_df: pd.DataFrame,
    overall_obj: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
    decimals: int = 3,
    include_combined: bool = True,
    include_per_column: bool = True,
) -> pd.DataFrame:
    """
    Build a publication-ready summary table from descriptive-statistics outputs.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Row-level output DataFrame from ``descriptive_stats``.
    overall_obj : Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]], default=None
        Overall aggregate output from ``descriptive_stats`` in any split mode.
    decimals : int, default=3
        Number of decimal places for floating-point columns.
    include_combined : bool, default=True
        Include an all-columns combined row.
    include_per_column : bool, default=True
        Include one row per source text column.

    Returns
    -------
    pd.DataFrame
        Tidy summary table suitable for display or export in papers/reports.
    """
    columns = [
        "Corpus",
        "N Texts",
        "Total Characters",
        "Total Words",
        "Total Sentences",
        "Mean Words / Text",
        "SD Words / Text",
        "Mean Sentence Length",
        "SD Sentence Length",
        "Mean Word Length",
        "SD Word Length",
        "Total Unique Words",
        "Lexical Diversity (TTR)",
    ]
    if not isinstance(stats_df, pd.DataFrame) or stats_df.empty:
        return pd.DataFrame(columns=columns)

    combined_overall, per_column_overall = _split_overall_payload(overall_obj)

    def _float_or_nan(value: Any) -> float:
        """
        Convert a value to float, returning NaN when conversion fails.

        Parameters
        ----------
        value : Any
            Candidate numeric value.

        Returns
        -------
        float
            Parsed float value or ``nan``.
        """
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _int_or_nan(value: Any):
        """
        Convert a value to integer, returning pandas NA when invalid.

        Parameters
        ----------
        value : Any
            Candidate integer-like value.

        Returns
        -------
        int | pandas.NA
            Parsed integer or missing marker.
        """
        if value is None:
            return pd.NA
        try:
            return int(value)
        except (TypeError, ValueError):
            return pd.NA

    def _build_row(
        label: str,
        sub_df: pd.DataFrame,
        overall_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute one summary row from row-level stats plus optional aggregates.

        Parameters
        ----------
        label : str
            Row label (for example, ``Combined`` or a source column name).
        sub_df : pd.DataFrame
            Subset of ``stats_df`` rows represented by this summary row.
        overall_stats : Optional[Dict[str, Any]], default=None
            Optional precomputed aggregate metrics for this subset.

        Returns
        -------
        Dict[str, Any]
            Dictionary aligned to the summary-table columns.
        """
        overall_stats = overall_stats if isinstance(overall_stats, dict) else {}
        total_words = int(overall_stats.get("total_words", int(sub_df["word_count"].sum())))
        total_sentences = int(overall_stats.get("total_sentences", int(sub_df["sentence_count"].sum())))
        total_chars = int(overall_stats.get("total_chars", int(sub_df["char_count"].sum())))
        total_unique_words_raw = overall_stats.get("total_unique_words")
        total_unique_words = _int_or_nan(total_unique_words_raw)

        lexical_diversity = _float_or_nan(overall_stats.get("lexical_diversity"))
        if pd.isna(lexical_diversity):
            if total_unique_words is not pd.NA and total_words > 0:
                lexical_diversity = float(total_unique_words) / float(total_words)
            else:
                lexical_diversity = _float_or_nan(sub_df["lexical_diversity"].mean())

        return {
            "Corpus": str(label),
            "N Texts": int(len(sub_df)),
            "Total Characters": total_chars,
            "Total Words": total_words,
            "Total Sentences": total_sentences,
            "Mean Words / Text": float(sub_df["word_count"].mean()),
            "SD Words / Text": float(sub_df["word_count"].std(ddof=1)) if len(sub_df) > 1 else 0.0,
            "Mean Sentence Length": _float_or_nan(overall_stats.get("avg_sentence_length"))
            if "avg_sentence_length" in overall_stats
            else float(sub_df["avg_sentence_length"].mean()),
            "SD Sentence Length": float(sub_df["avg_sentence_length"].std(ddof=1)) if len(sub_df) > 1 else 0.0,
            "Mean Word Length": _float_or_nan(overall_stats.get("avg_word_length"))
            if "avg_word_length" in overall_stats
            else float(sub_df["avg_word_length"].mean()),
            "SD Word Length": float(sub_df["avg_word_length"].std(ddof=1)) if len(sub_df) > 1 else 0.0,
            "Total Unique Words": total_unique_words,
            "Lexical Diversity (TTR)": lexical_diversity,
        }

    source_col_count = 1
    if isinstance(stats_df.index, pd.MultiIndex) and stats_df.index.nlevels >= 1:
        source_col_count = int(stats_df.index.get_level_values(0).nunique())
        source_col_count = max(1, source_col_count)

    include_combined_effective = bool(include_combined)
    if include_combined_effective and include_per_column and source_col_count == 1:
        include_combined_effective = False

    rows = []
    if include_combined_effective:
        rows.append(_build_row("Combined", stats_df, combined_overall))

    if include_per_column and isinstance(stats_df.index, pd.MultiIndex) and stats_df.index.nlevels >= 1:
        for source_column, sub_df in stats_df.groupby(level=0, sort=True):
            source_key = str(source_column)
            rows.append(_build_row(source_key, sub_df, per_column_overall.get(source_key)))

    if not rows:
        return pd.DataFrame(columns=columns)

    summary_df = pd.DataFrame(rows, columns=columns)
    int_cols = [
        "N Texts",
        "Total Characters",
        "Total Words",
        "Total Sentences",
        "Total Unique Words",
    ]
    for col in int_cols:
        summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").astype("Int64")

    float_cols = [c for c in summary_df.columns if c not in {"Corpus", *int_cols}]
    summary_df[float_cols] = summary_df[float_cols].apply(pd.to_numeric, errors="coerce").round(int(decimals))

    order = np.where(summary_df["Corpus"].astype(str).eq("Combined"), 0, 1)
    summary_df["_order"] = order
    summary_df = summary_df.sort_values(["_order", "Corpus"], kind="stable").drop(columns=["_order"]).reset_index(drop=True)
    return summary_df


def descriptive_summary_table_to_latex(
    summary_df: pd.DataFrame,
    decimals: int = 3,
) -> str:
    """
    Convert a summary table to a vertical, manuscript-friendly LaTeX table.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from ``build_descriptive_summary_table``.
    decimals : int, default=3
        Decimal precision used for floating-point values.

    Returns
    -------
    str
        LaTeX ``tabular`` string using booktabs-style horizontal rules with no
        vertical separators, in long/vertical metric layout.
    """
    latex_columns = ["Corpus", "Metric", "Value"]
    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        empty = pd.DataFrame(columns=latex_columns)
        return empty.to_latex(index=False, escape=True, column_format="lll")

    metric_columns = [c for c in summary_df.columns if c != "Corpus"]
    rows = []

    def _format_value_for_latex(value: Any) -> str:
        """
        Format a table cell value for manuscript-ready LaTeX output.

        Parameters
        ----------
        value : Any
            Raw value from the summary DataFrame.

        Returns
        -------
        str
            Rendered value string with integer grouping and fixed float precision.
        """
        if pd.isna(value):
            return ""
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{int(decimals)}f}"
        return str(value)

    for _, row in summary_df.iterrows():
        corpus = str(row.get("Corpus", ""))
        for metric in metric_columns:
            rows.append(
                {
                    "Corpus": corpus,
                    "Metric": str(metric),
                    "Value": _format_value_for_latex(row.get(metric)),
                }
            )

    latex_df = pd.DataFrame(rows, columns=latex_columns)
    return latex_df.to_latex(
        index=False,
        escape=True,
        column_format="lll",
    )

def descriptive_stats(
    *series_list: Iterable[pd.Series],
    use_lemmas_for_uniques: bool = True,
    split_overall: SplitOverall = "combined",
    nlp: Optional[spacy.Language] = None,
    n_process: int = 1,
    batch_size: int = 256,
    words_must_be_alpha: bool = True,
    drop_stopwords: bool = False,
    avg_sentence_mode: Literal["unweighted", "ratio"] = "unweighted"
) -> Tuple[pd.DataFrame, Union[Dict[str, Any], Dict[str, Dict[str, Any]]]]:
    """
    Compute per-text descriptive statistics and aggregate summaries.

    Parameters
    ----------
    *series_list : Iterable[pd.Series]
        One or more pandas Series containing text. Each non-empty string row is
        analyzed and tagged by source column and original index.
    use_lemmas_for_uniques : bool, default=True
        If True and a lemmatizer is available, unique-word counts are computed
        from lemmas; otherwise raw lowercased token text is used.
    split_overall : {"combined", "per_column", "both"}, default="combined"
        Controls whether aggregate metrics are returned across all inputs,
        per source column, or both.
    nlp : Optional[spacy.Language], default=None
        Optional preloaded spaCy pipeline. When None, the package default
        lightweight English pipeline is loaded.
    n_process : int, default=1
        Number of worker processes passed to ``nlp.pipe``.
    batch_size : int, default=256
        Batch size used by ``nlp.pipe``.
    words_must_be_alpha : bool, default=True
        If True, only alphabetic tokens count as words.
    drop_stopwords : bool, default=False
        If True, spaCy stopwords are excluded from word-level statistics.
    avg_sentence_mode : {"unweighted", "ratio"}, default="unweighted"
        Sentence-length aggregation mode.
        ``"unweighted"`` averages per-sentence counts; ``"ratio"`` uses
        total words / total sentences.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        ``stats_df``: row-level metrics indexed by ``(source_column, index)``.
        ``overall``: aggregate metrics in a dictionary, shaped according to
        ``split_overall``.
    """
    if nlp is None:
        nlp = get_spacy_pipeline_base()

    has_lemmatizer = "lemmatizer" in nlp.pipe_names
    if use_lemmas_for_uniques and not has_lemmatizer:
        use_lemmas_for_uniques = False

    rows = []
    for s in series_list:
        if not isinstance(s, pd.Series):
            raise ValueError("All inputs must be pandas Series")
        src = s.name if s.name is not None else "<unnamed>"
        for idx, text in s.dropna().items():
            if isinstance(text, str) and text.strip():
                rows.append({"text": text, "source_column": src, "index": idx})

    if not rows:
        return pd.DataFrame(), {}

    texts = [r["text"] for r in rows]

    def keep_token(t: spacy.tokens.Token) -> bool:
        """
        Decide whether a token should count toward lexical statistics.

        Parameters
        ----------
        t : spacy.tokens.Token
            Token from a parsed spaCy document.

        Returns
        -------
        bool
            True when the token passes punctuation/space/alpha/stopword rules.
        """
        if t.is_space or t.is_punct:
            return False
        if words_must_be_alpha and not t.is_alpha:
            return False
        if drop_stopwords and t.is_stop:
            return False
        return True

    stats_rows = []
    unique_sets = []
    pos_full_list = []

    # Process texts
    for doc, meta in zip(nlp.pipe(texts, batch_size=batch_size, n_process=n_process), rows):
        tokens = [t for t in doc if not t.is_space]
        word_tokens = [t for t in tokens if keep_token(t)]
        chars = len(meta["text"])
        word_count = len(word_tokens)
        sentences = list(doc.sents)
        sentence_count = len(sentences)

        # Average word length
        avg_word_length = float(np.mean([len(t.text) for t in word_tokens])) if word_count else 0.0

        # Average sentence length
        if sentence_count:
            if avg_sentence_mode == "unweighted":
                per_sent_counts = [len([t for t in sent if keep_token(t)]) for sent in sentences]
                avg_sentence_length = float(np.mean(per_sent_counts))
            else:
                total_in_sent = sum(len([t for t in sent if keep_token(t)]) for sent in sentences)
                avg_sentence_length = float(total_in_sent) / float(sentence_count) if sentence_count else 0.0
        else:
            avg_sentence_length = 0.0

        # Vocabulary set per row
        if use_lemmas_for_uniques:
            uniques = {t.lemma_.lower() for t in word_tokens if t.lemma_}
        else:
            uniques = {t.text.lower() for t in word_tokens}

        unique_words = len(uniques)
        lexical_diversity = float(unique_words / word_count) if word_count else 0.0
        pos_counts = Counter(t.pos_ for t in word_tokens)

        unique_sets.append((meta["source_column"], meta["index"], uniques))
        pos_full_list.append((meta["source_column"], meta["index"], pos_counts))

        stats_rows.append(
            {
                "source_column": meta["source_column"],
                "index": meta["index"],
                "char_count": int(chars),
                "word_count": int(word_count),
                "sentence_count": int(sentence_count),
                "avg_word_length": float(avg_word_length),
                "avg_sentence_length": float(avg_sentence_length),
                "unique_words": int(unique_words),
                "lexical_diversity": float(lexical_diversity),
                "noun_count": int(sum(1 for t in word_tokens if t.pos_ in {"NOUN", "PROPN"})),
                "verb_count": int(sum(1 for t in word_tokens if t.pos_ in {"VERB", "AUX"})),
                "adjective_count": int(sum(1 for t in word_tokens if t.pos_ == "ADJ")),
                "adverb_count": int(sum(1 for t in word_tokens if t.pos_ == "ADV")),
                "pos_full": dict(pos_counts),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df.set_index(["source_column", "index"], inplace=True)
    stats_df.sort_index(inplace=True)

    uniques_by_row = {(src, idx): u for src, idx, u in unique_sets}
    pos_by_row = {(src, idx): c for src, idx, c in pos_full_list}

    def compute_overall(index_like) -> Dict[str, Any]:
        """
        Aggregate row-level stats and lexical/POS sets for a subset of rows.

        Parameters
        ----------
        index_like
            Index selector into ``stats_df`` (typically a MultiIndex slice).

        Returns
        -------
        Dict[str, Any]
            Combined totals and means, including POS distributions and lexical
            diversity for the selected rows.
        """
        sub = stats_df.loc[index_like]
        total_words = int(sub["word_count"].sum())
        vocab_union = set()
        for key in sub.index:
            vocab_union |= uniques_by_row.get(key, set())
        pos_total = Counter()
        for key in sub.index:
            pos_total += pos_by_row.get(key, Counter())

        # Average sentence length aggregation mirrors per row choice
        if len(sub):
            if avg_sentence_mode == "unweighted":
                avg_sentence_length_agg = float(sub["avg_sentence_length"].mean())
            else:
                total_sentences = int(sub["sentence_count"].sum())
                avg_sentence_length_agg = float(total_words) / float(total_sentences) if total_sentences else 0.0
        else:
            avg_sentence_length_agg = 0.0

        return {
            "total_chars": int(sub["char_count"].sum()),
            "total_words": total_words,
            "total_sentences": int(sub["sentence_count"].sum()),
            "avg_word_length": float(sub["avg_word_length"].mean()) if len(sub) else 0.0,
            "avg_sentence_length": float(avg_sentence_length_agg),
            "total_unique_words": int(len(vocab_union)),
            "lexical_diversity": float(len(vocab_union) / total_words) if total_words else 0.0,
            "pos_distribution_basic": {
                "nouns": int(sub["noun_count"].sum()),
                "verbs": int(sub["verb_count"].sum()),
                "adjectives": int(sub["adjective_count"].sum()),
                "adverbs": int(sub["adverb_count"].sum()),
            },
            "pos_distribution_full": dict(pos_total),
        }

    if stats_df.empty:
        return stats_df, {}

    if split_overall == "combined":
        return stats_df, compute_overall(stats_df.index)
    if split_overall == "per_column":
        out = {}
        for col, subdf in stats_df.groupby(level=0):
            out[col] = compute_overall(subdf.index)
        return stats_df, out
    if split_overall == "both":
        per_col = {}
        for col, subdf in stats_df.groupby(level=0):
            per_col[col] = compute_overall(subdf.index)
        return stats_df, {"combined": compute_overall(stats_df.index), "per_column": per_col}
    return stats_df, compute_overall(stats_df.index)
