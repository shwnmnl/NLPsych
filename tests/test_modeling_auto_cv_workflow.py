import numpy as np
import pandas as pd
import pytest

import nlpsych.modeling as modeling
from nlpsych.modeling import auto_cv_with_permutation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class _FixedFoldSplitter:
    def __init__(self, n_splits: int):
        self.n_splits = int(n_splits)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def test_auto_cv_aligns_groups_and_passes_group_labels_through_workflow(monkeypatch):
    X = pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.8, 0.9, 1.0, 1.1],
            "f2": [1.0, 0.9, 0.2, 0.1, 0.0, -0.1],
        },
        index=[10, 11, 12, 13, 14, 15],
    )
    Y = pd.DataFrame(
        {
            "label": pd.Series(
                ["cat", "cat", "dog", "dog", None, "dog"],
                index=[10, 11, 12, 13, 14, 15],
            )
        }
    )
    groups = pd.Series(
        ["g1", "g1", "g2", "g2", "g3", None],
        index=[10, 11, 12, 13, 14, 15],
    )

    captured: dict[str, object] = {}
    splitter = _FixedFoldSplitter(n_splits=2)

    def fake_cv_iterator(task, y, cv, random_state, groups=None):
        captured["cv_task"] = task
        captured["cv_y"] = np.asarray(y)
        captured["cv_groups"] = None if groups is None else np.asarray(groups)
        captured["cv_arg"] = cv
        return splitter

    def fake_fit_and_score_cv(X, y, task, base_model, cv_splits, scorer, groups=None):
        captured["fit_X_shape"] = X.shape
        captured["fit_y"] = np.asarray(y)
        captured["fit_groups"] = None if groups is None else np.asarray(groups)
        captured["fit_cv_splits"] = cv_splits
        return [0.8, 0.9], np.asarray(y, dtype=float)

    def fake_perm_test(X, y, task, estimator, cv_splits, scorer, observed, n_perm, larger_is_better, rng, groups=None):
        captured["perm_groups"] = None if groups is None else np.asarray(groups)
        return [0.1, 0.2], 0.125

    monkeypatch.setattr(modeling, "_cv_iterator", fake_cv_iterator)
    monkeypatch.setattr(modeling, "_fit_and_score_cv", fake_fit_and_score_cv)
    monkeypatch.setattr(modeling, "_perm_test", fake_perm_test)

    results_df, preds = auto_cv_with_permutation(
        X=X,
        Y=Y,
        groups=groups,
        cv=4,
        n_permutations=2,
        random_state=3,
    )

    expected_groups = np.array(["g1", "g1", "g2", "g2"], dtype=object)
    expected_y = np.array([0, 0, 1, 1], dtype=int)

    assert captured["cv_task"] == "classification"
    assert np.array_equal(captured["cv_y"], expected_y)
    assert np.array_equal(captured["cv_groups"], expected_groups)
    assert captured["fit_X_shape"] == (4, 2)
    assert np.array_equal(captured["fit_y"], expected_y)
    assert np.array_equal(captured["fit_groups"], expected_groups)
    assert np.array_equal(captured["perm_groups"], expected_groups)
    assert captured["fit_cv_splits"] is splitter
    assert results_df.loc[0, "task"] == "classification"
    assert results_df.loc[0, "cv_folds_used"] == 2
    assert len(preds["label"]) == 4


def test_auto_cv_without_permutation_drops_permutation_and_adjustment_columns(monkeypatch):
    X = pd.DataFrame(np.arange(24, dtype=float).reshape(6, 4))
    Y = pd.DataFrame({"target": np.linspace(0.0, 1.0, 6)})

    monkeypatch.setattr(
        modeling,
        "_fit_and_score_cv",
        lambda X, y, task, base_model, cv_splits, scorer, groups=None: (
            [0.25, 0.50],
            np.linspace(0.1, 0.6, len(y)),
        ),
    )
    monkeypatch.setattr(
        modeling,
        "_perm_test",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("permutation should not run")),
    )

    results_df, preds = auto_cv_with_permutation(
        X=X,
        Y=Y,
        task_mode="regression",
        run_permutation=False,
        cv=3,
    )

    assert "target" in preds
    assert len(preds["target"]) == len(Y)
    assert "p_value" not in results_df.columns
    assert "perm_scores" not in results_df.columns
    assert "p_adjusted" not in results_df.columns
    assert "p_adjust_method" not in results_df.columns
    assert "p_fdr" not in results_df.columns


def test_auto_cv_tuning_uses_gridsearch_with_filtered_reduction_and_selector_grids(monkeypatch):
    X = pd.DataFrame(np.arange(72, dtype=float).reshape(12, 6))
    Y = pd.DataFrame({"score": np.linspace(0.0, 1.1, 12)})

    captured: dict[str, object] = {}

    def fake_fit_and_score_cv(X, y, task, base_model, cv_splits, scorer, groups=None):
        captured["base_model"] = base_model
        return [0.2, 0.3, 0.4], np.zeros(len(y), dtype=float)

    monkeypatch.setattr(modeling, "_fit_and_score_cv", fake_fit_and_score_cv)

    auto_cv_with_permutation(
        X=X,
        Y=Y,
        task_mode="regression",
        run_permutation=False,
        tune_hyperparams=True,
        regressor_param_grid={"alpha": [0.1, 1.0]},
        feature_selection="kbest",
        k_best=3,
        k_best_grid=[0, 2, 99],
        reduce_method="pca",
        reduce_n_components=4,
        reduce_components_grid=[0, 2, 99],
        cv=3,
    )

    base_model = captured["base_model"]
    assert isinstance(base_model, GridSearchCV)
    assert base_model.param_grid == {
        "model__alpha": [0.1, 1.0],
        "reducer__n_components": [2],
        "selector__k": [2],
    }


def test_auto_cv_tuning_falls_back_to_pipeline_when_inner_cv_unavailable(monkeypatch):
    X = pd.DataFrame(np.arange(40, dtype=float).reshape(10, 4))
    Y = pd.DataFrame({"score": np.linspace(0.0, 0.9, 10)})

    captured: dict[str, object] = {}

    monkeypatch.setattr(modeling, "_inner_cv", lambda task, y, random_state, groups=None: None)

    def fake_fit_and_score_cv(X, y, task, base_model, cv_splits, scorer, groups=None):
        captured["base_model"] = base_model
        return [0.3, 0.4], np.zeros(len(y), dtype=float)

    monkeypatch.setattr(modeling, "_fit_and_score_cv", fake_fit_and_score_cv)

    with pytest.warns(RuntimeWarning, match="Skipping hyperparameter tuning"):
        auto_cv_with_permutation(
            X=X,
            Y=Y,
            task_mode="regression",
            run_permutation=False,
            tune_hyperparams=True,
            regressor_param_grid={"alpha": [0.1, 1.0]},
            cv=3,
        )

    assert isinstance(captured["base_model"], Pipeline)


def test_auto_cv_returns_empty_outputs_when_target_has_single_class():
    X = pd.DataFrame(np.arange(32, dtype=float).reshape(8, 4))
    Y = pd.DataFrame({"label": ["same"] * 8})

    with pytest.warns(RuntimeWarning, match="only a single class after alignment"):
        results_df, preds = auto_cv_with_permutation(
            X=X,
            Y=Y,
            task_mode="classification",
            run_permutation=False,
            cv=3,
        )

    assert results_df.empty
    assert results_df.columns.tolist() == [
        "target",
        "task",
        "metric_name",
        "observed",
        "p_value",
        "p_fdr",
        "p_adjusted",
        "p_adjust_method",
        "cv_scores",
        "perm_scores",
        "cv_folds_used",
    ]
    assert preds == {}


def test_auto_cv_applies_integrated_multiple_comparison_correction(monkeypatch):
    X = pd.DataFrame(np.arange(64, dtype=float).reshape(16, 4))
    Y = pd.DataFrame(
        {
            "target_a": np.linspace(0.0, 1.5, 16),
            "target_b": np.linspace(1.0, 2.5, 16),
        }
    )

    perm_outcomes = iter([
        ([0.05, 0.07], 0.01),
        ([0.20, 0.25], 0.20),
    ])

    monkeypatch.setattr(
        modeling,
        "_fit_and_score_cv",
        lambda X, y, task, base_model, cv_splits, scorer, groups=None: (
            [0.3, 0.4],
            np.asarray(y, dtype=float),
        ),
    )
    monkeypatch.setattr(
        modeling,
        "_perm_test",
        lambda X, y, task, estimator, cv_splits, scorer, observed, n_perm, larger_is_better, rng, groups=None: next(perm_outcomes),
    )

    results_df, preds = auto_cv_with_permutation(
        X=X,
        Y=Y,
        task_mode="regression",
        run_permutation=True,
        n_permutations=2,
        correction_method="bonferroni",
        cv=3,
    )

    assert set(preds) == {"target_a", "target_b"}
    assert results_df["p_value"].tolist() == pytest.approx([0.01, 0.20])
    assert results_df["p_adjusted"].tolist() == pytest.approx([0.02, 0.40])
    assert results_df["p_adjust_method"].tolist() == ["bonferroni", "bonferroni"]
    assert results_df["p_adjust_scope"].tolist() == ["all_tests", "all_tests"]
    assert results_df["perm_scores"].apply(len).tolist() == [2, 2]


def test_auto_cv_classification_workflow_reports_supported_extra_metrics_and_warns_on_unsupported():
    rng = np.random.RandomState(13)
    class_zero = rng.normal(loc=-1.0, scale=0.2, size=(12, 3))
    class_one = rng.normal(loc=1.0, scale=0.2, size=(12, 3))
    X = pd.DataFrame(np.vstack([class_zero, class_one]), columns=["f1", "f2", "f3"])
    Y = pd.DataFrame({"label": [0] * 12 + [1] * 12})

    with pytest.warns(RuntimeWarning, match="Skipping unsupported metrics: bogus_metric"):
        results_df, preds = auto_cv_with_permutation(
            X=X,
            Y=Y,
            task_mode="auto",
            run_permutation=False,
            cv=4,
            report_metrics=["balanced_accuracy", "bogus_metric"],
            random_state=13,
        )

    assert len(results_df) == 1
    assert results_df.loc[0, "task"] == "classification"
    assert results_df.loc[0, "metric_name"] == "accuracy"
    assert "metric_balanced_accuracy" in results_df.columns
    assert np.isfinite(results_df.loc[0, "observed"])
    assert np.isfinite(results_df.loc[0, "metric_balanced_accuracy"])
    assert len(preds["label"]) == len(Y)
