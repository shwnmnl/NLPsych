import pandas as pd
import spacy

from nlpsych.descriptive_stats import descriptive_stats


def test_descriptive_stats_computes_counts_and_combined_per_column_payloads_with_blank_pipeline():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    s1 = pd.Series(
        ["Hello world. Hello again!", "Calm minds focus."],
        index=[0, 1],
        name="note",
    )
    s2 = pd.Series(
        ["Markets move quickly."],
        index=[5],
        name="essay",
    )

    stats_df, overall = descriptive_stats(
        s1,
        s2,
        split_overall="both",
        nlp=nlp,
        use_lemmas_for_uniques=True,
    )

    assert stats_df.index.names == ["source_column", "index"]
    assert stats_df.loc[("note", 0), "word_count"] == 4
    assert stats_df.loc[("note", 0), "sentence_count"] == 2
    assert stats_df.loc[("note", 0), "unique_words"] == 3
    assert stats_df.loc[("note", 0), "lexical_diversity"] == 0.75
    assert stats_df.loc[("note", 0), "noun_count"] == 0
    assert stats_df.loc[("essay", 5), "word_count"] == 3

    assert set(overall.keys()) == {"combined", "per_column"}
    assert overall["combined"]["total_words"] == 10
    assert overall["combined"]["total_sentences"] == 4
    assert overall["combined"]["total_unique_words"] == 9
    assert overall["per_column"]["note"]["total_words"] == 7
    assert overall["per_column"]["essay"]["total_words"] == 3
    assert overall["combined"]["pos_distribution_basic"]["nouns"] == 0
