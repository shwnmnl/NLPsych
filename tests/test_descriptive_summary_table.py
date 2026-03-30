import pandas as pd
import pytest

from nlpsych.descriptive_stats import (
    build_descriptive_summary_table,
    descriptive_summary_table_to_latex,
    descriptive_summary_table_to_markdown,
)


def _single_source_stats_df() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [("note", 0), ("note", 1)],
        names=["source_column", "index"],
    )
    return pd.DataFrame(
        {
            "char_count": [50, 70],
            "word_count": [10, 14],
            "sentence_count": [2, 4],
            "avg_word_length": [4.0, 5.0],
            "avg_sentence_length": [5.0, 3.5],
            "unique_words": [5, 6],
            "lexical_diversity": [0.5, 6.0 / 14.0],
            "noun_count": [3, 4],
            "verb_count": [2, 3],
            "adjective_count": [1, 1],
            "adverb_count": [0, 1],
            "pos_full": [{}, {}],
        },
        index=index,
    )


def _single_source_overall() -> dict:
    note_overall = {
        "total_chars": 120,
        "total_words": 24,
        "total_sentences": 6,
        "avg_word_length": 4.5,
        "avg_sentence_length": 4.0,
        "total_unique_words": 9,
        "lexical_diversity": 0.375,
    }
    return {
        "combined": dict(note_overall),
        "per_column": {"note": dict(note_overall)},
    }


def test_summary_table_single_source_drops_corpus_and_adds_per_text_metrics():
    summary_df = build_descriptive_summary_table(
        stats_df=_single_source_stats_df(),
        overall_obj=_single_source_overall(),
        decimals=3,
    )

    assert list(summary_df.index) == [0]
    assert "Corpus" not in summary_df.columns
    assert summary_df.loc[0, "N Texts"] == 2
    assert summary_df.loc[0, "Total Characters"] == 120
    assert summary_df.loc[0, "Total Words"] == 24
    assert summary_df.loc[0, "Total Sentences"] == 6
    assert summary_df.loc[0, "Total Unique Words"] == 9
    assert summary_df.loc[0, "Lexical Diversity (TTR)"] == pytest.approx(0.375, abs=1e-3)
    assert summary_df.loc[0, "Mean Characters / Text"] == pytest.approx(60.0, abs=1e-3)
    assert summary_df.loc[0, "SD Characters / Text"] == pytest.approx(14.142, abs=1e-3)
    assert summary_df.loc[0, "Mean Words / Text"] == pytest.approx(12.0, abs=1e-3)
    assert summary_df.loc[0, "SD Words / Text"] == pytest.approx(2.828, abs=1e-3)
    assert summary_df.loc[0, "Mean Sentences / Text"] == pytest.approx(3.0, abs=1e-3)
    assert summary_df.loc[0, "SD Sentences / Text"] == pytest.approx(1.414, abs=1e-3)
    assert summary_df.loc[0, "Mean Unique Words / Text"] == pytest.approx(5.5, abs=1e-3)
    assert summary_df.loc[0, "SD Unique Words / Text"] == pytest.approx(0.707, abs=1e-3)
    assert summary_df.loc[0, "Mean Lexical Diversity (TTR) / Text"] == pytest.approx(0.464, abs=1e-3)
    assert summary_df.loc[0, "SD Lexical Diversity (TTR) / Text"] == pytest.approx(0.051, abs=1e-3)


def test_vertical_summary_exports_match_single_and_multi_source_layouts():
    single_summary_df = build_descriptive_summary_table(
        stats_df=_single_source_stats_df(),
        overall_obj=_single_source_overall(),
        decimals=3,
    )

    single_md = descriptive_summary_table_to_markdown(single_summary_df, decimals=3)
    single_latex = descriptive_summary_table_to_latex(single_summary_df, decimals=3)

    assert "Corpus" not in single_md
    assert "Metric" in single_md
    assert "Value" in single_md
    assert "Mean Characters / Text" in single_md
    assert "14.142" in single_md
    assert "Corpus" not in single_latex
    assert "Metric & Value" in single_latex
    assert "Mean Characters / Text" in single_latex
    assert single_latex.count("\\hline") >= 5
    assert single_latex.index("Total Unique Words") < single_latex.index("Mean Characters / Text")
    assert single_latex.index("SD Unique Words / Text") < single_latex.index("Lexical Diversity (TTR)")
    assert single_latex.index("SD Lexical Diversity (TTR) / Text") < single_latex.index("Mean Sentence Length")

    multi_summary_df = pd.DataFrame(
        [
            {"Corpus": "essay", "N Texts": 2, "Total Words": 24},
            {"Corpus": "note", "N Texts": 1, "Total Words": 11},
        ]
    )

    multi_md = descriptive_summary_table_to_markdown(multi_summary_df, decimals=3)
    multi_latex = descriptive_summary_table_to_latex(multi_summary_df, decimals=3)

    assert "Corpus" in multi_md
    assert "essay" in multi_md
    assert "note" in multi_md
    assert "Total Words" in multi_md
    assert "Corpus & Metric & Value" in multi_latex
    assert "essay & Total Words & 24" in multi_latex
