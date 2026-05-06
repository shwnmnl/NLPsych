import pandas as pd

from nlpsych.report import (
    _advanced_settings_notes,
    build_report_payload,
    interpret_model_row,
    summarize_model_row,
)


def _sample_stats_df() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [("note", 0), ("note", 1)],
        names=["source_column", "index"],
    )
    return pd.DataFrame(
        {
            "char_count": [20, 24],
            "word_count": [4, 4],
            "sentence_count": [1, 1],
            "avg_word_length": [4.0, 5.0],
            "avg_sentence_length": [4.0, 4.0],
            "unique_words": [3, 4],
            "lexical_diversity": [0.75, 1.0],
            "noun_count": [2, 2],
            "verb_count": [1, 1],
            "adjective_count": [0, 1],
            "adverb_count": [0, 0],
            "pos_full": [{}, {}],
        },
        index=index,
    )


def _sample_overall() -> dict:
    return {
        "combined": {
            "total_chars": 44,
            "total_words": 8,
            "total_sentences": 2,
            "avg_word_length": 4.5,
            "avg_sentence_length": 4.0,
            "total_unique_words": 6,
            "lexical_diversity": 0.75,
            "pos_distribution_basic": {
                "nouns": 4,
                "verbs": 2,
                "adjectives": 1,
                "adverbs": 0,
            },
        },
        "per_column": {
            "note": {
                "total_chars": 44,
                "total_words": 8,
                "total_sentences": 2,
                "avg_word_length": 4.5,
                "avg_sentence_length": 4.0,
                "total_unique_words": 6,
                "lexical_diversity": 0.75,
            }
        },
    }


def test_advanced_settings_notes_capture_nondefault_model_choices():
    model_config = {
        "feature_selection": "kbest",
        "k_best": 12,
        "reduce_method": "pca",
        "reduce_n_components": 6,
        "tune_hyperparams": True,
        "classifier_param_grid": {"C": [0.1, 1.0]},
        "reduce_components_grid": [4, 6],
        "group_col": "participant_id",
        "text_feature_mode": "separate",
        "text_cols": ["q1", "q2"],
        "n_perm": 250,
        "correction_method_label": "Holm",
        "correction_scope": "within_target",
        "target_task_overrides": {"symptoms": "regression"},
    }

    notes = _advanced_settings_notes(model_config, target="symptoms")

    assert "KBest feature selection (k=12)" in notes
    assert "PCA dimensionality reduction (n_components=6)" in notes
    assert "hyperparameter tuning with custom classifier grid, reduction grid" in notes
    assert "group-aware CV using participant_id" in notes
    assert "independent model runs per selected text column" in notes
    assert "permutation testing (250 iterations)" in notes
    assert any("multiple-comparisons correction (Holm;" in note for note in notes)
    assert "explicit task override (regression) for this target" in notes


def test_interpret_and_summarize_model_row_use_adjusted_p_values_and_negative_r2_logic():
    row = pd.Series(
        {
            "target": "symptoms",
            "task": "regression",
            "metric_name": "R2",
            "observed": -0.12,
            "p_value": 0.01,
            "p_adjusted": 0.02,
            "p_adjust_method": "bonferroni",
            "cv_folds_used": 5,
        }
    )

    bullets = interpret_model_row(row)
    summary = summarize_model_row(row, include_adjusted=True)

    assert any("worse than a constant baseline" in bullet for bullet in bullets)
    assert any("statistically reliable" in bullet and "poor predictive performance" in bullet for bullet in bullets)
    assert any("Cross validation used 5 folds." == bullet for bullet in bullets)
    assert "adjusted p=0.0200 (bonferroni)" in summary
    assert "poor predictive performance because R2 is negative" in summary


def test_build_report_payload_renders_populated_sections_and_filters_irrelevant_embed_config():
    stats_df = _sample_stats_df()
    overall_obj = _sample_overall()
    plot_df = pd.DataFrame(
        {
            "dim_1": [0.1, 0.2],
            "dim_2": [0.3, 0.4],
            "text": ["a", "b"],
        }
    )
    results_df = pd.DataFrame(
        {
            "target": ["symptoms", "wellbeing"],
            "task": ["regression", "classification"],
            "metric_name": ["R2", "ROC_AUC"],
            "observed": [0.41, 0.82],
            "p_value": [0.03, 0.01],
            "p_adjusted": [0.04, 0.02],
            "p_adjust_method": ["holm", "holm"],
            "p_adjust_scope": ["within_target", "within_target"],
            "cv_folds_used": [5, 5],
        }
    )
    stats_config = {"drop_stopwords": True, "avg_sentence_mode": "ratio"}
    embed_config = {
        "reduce_method": "pca",
        "reduce_n_components": 2,
        "tsne_metric": "euclidean",
        "umap_metric": "cosine",
    }
    model_config = {"task_mode": "regression", "n_perm": 500}

    html_report, md_report = build_report_payload(
        text_cols=["note"],
        stats_df=stats_df,
        overall_obj=overall_obj,
        plot_df=plot_df,
        results_df=results_df,
        stats_config=stats_config,
        embed_config=embed_config,
        model_config=model_config,
    )

    assert "NLPsych session report" in html_report
    assert "Summary table" in html_report
    assert "Projection computed with 2 dimensions" in html_report
    assert "Interpretation per target" in html_report
    assert "tsne_metric" not in html_report
    assert "Selected text columns: note" in md_report
    assert "drop_stopwords" in md_report
    assert "reduce_n_components" in md_report
    assert "Permutation testing gave p=0.0300 and adjusted p=0.0400 (holm)" in md_report


def test_build_report_payload_uses_empty_state_messages_when_sections_are_missing():
    html_report, md_report = build_report_payload(
        text_cols=[],
        stats_df=None,
        overall_obj=None,
        plot_df=None,
        results_df=None,
    )

    assert "No descriptive statistics were available" in html_report
    assert "No embeddings or projection were available" in html_report
    assert "No modeling results were available" in html_report
    assert "Selected text columns:" in md_report
    assert "Run the Modeling tab first." in md_report
