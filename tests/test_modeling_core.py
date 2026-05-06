import numpy as np
import pandas as pd
import pytest

import nlpsych.modeling as modeling
from nlpsych.modeling import (
    _build_pipeline,
    _build_reducer,
    _build_selector,
    _fit_and_score_cv,
    _inner_cv,
    _normalize_task_mode,
    _perm_test,
    _resolve_target_task,
    apply_multiple_comparisons_correction,
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold, f_classif, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def test_normalize_task_mode_aliases_and_invalid_values():
    assert _normalize_task_mode(None) == "auto"
    assert _normalize_task_mode(" classifier ") == "classification"
    assert _normalize_task_mode("regress") == "regression"

    with pytest.raises(ValueError, match="task_mode must be one of"):
        _normalize_task_mode("clustering")


def test_resolve_target_task_uses_override_or_auto_detection():
    y = pd.Series([1.0, 2.0, 1.0, 2.0], name="symptoms")

    assert _resolve_target_task("symptoms", y, 20, "auto") == "classification"
    assert _resolve_target_task(
        "symptoms",
        y,
        20,
        "auto",
        target_task_overrides={"symptoms": "regression"},
    ) == "regression"


def test_inner_cv_returns_none_when_grouped_classification_lacks_groups_per_class():
    y = np.array([0, 0, 1, 1], dtype=int)
    groups = np.array([1, 1, 1, 2], dtype=int)

    cv = _inner_cv("classification", y, random_state=7, groups=groups)

    assert cv is None


def test_build_selector_clamps_values_and_chooses_score_functions():
    kbest = _build_selector("classification", "kbest", k_best=99, percentile=20, variance_threshold=0.0, n_features=5)
    percentile = _build_selector("regression", "percentile", k_best=10, percentile=300, variance_threshold=0.0, n_features=8)
    variance = _build_selector("classification", "variance", k_best=10, percentile=20, variance_threshold=0.25, n_features=8)

    assert isinstance(kbest, SelectKBest)
    assert kbest.k == 5
    assert kbest.score_func is f_classif

    assert isinstance(percentile, SelectPercentile)
    assert percentile.percentile == 100
    assert percentile.score_func is f_regression

    assert isinstance(variance, VarianceThreshold)
    assert variance.threshold == pytest.approx(0.25)


def test_build_reducer_warns_on_unknown_method_and_caps_components():
    with pytest.warns(RuntimeWarning, match="Unknown reduction method"):
        assert _build_reducer("tsne", n_components=2, max_components=4) is None

    with pytest.warns(RuntimeWarning, match="exceeds max=4"):
        reducer = _build_reducer("pca", n_components=99, max_components=4)

    assert isinstance(reducer, PCA)
    assert reducer.n_components == 4


def test_build_pipeline_orders_preprocessing_steps_before_model():
    pipeline = _build_pipeline(
        base_model=Ridge(alpha=1.0),
        selector=SelectKBest(score_func=f_regression, k=2),
        reducer=PCA(n_components=2),
        scale_X=True,
    )

    assert list(pipeline.named_steps) == ["scaler", "reducer", "selector", "model"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)


def test_fit_and_score_cv_raises_on_nonfinite_fold_score():
    X = np.arange(24, dtype=float).reshape(6, 4)
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    cv = KFold(n_splits=2, shuffle=True, random_state=0)

    with pytest.raises(ValueError, match="Non-finite CV score encountered"):
        _fit_and_score_cv(
            X,
            y,
            task="regression",
            base_model=Ridge(alpha=1.0),
            cv_splits=cv,
            scorer=lambda y_true, y_pred: np.nan,
        )


def test_perm_test_handles_smaller_is_better_metrics(monkeypatch):
    perm_means = iter([0.10, 0.20, 0.40])

    def fake_fit_and_score_cv(X, y, task, estimator, cv_splits, scorer, groups=None):
        return [next(perm_means)], np.zeros(len(y), dtype=float)

    monkeypatch.setattr(modeling, "_fit_and_score_cv", fake_fit_and_score_cv)

    perm_scores, p_value = _perm_test(
        X=np.zeros((5, 2), dtype=float),
        y=np.arange(5, dtype=float),
        task="regression",
        estimator=Ridge(alpha=1.0),
        cv_splits=KFold(n_splits=2),
        scorer=lambda y_true, y_pred: 0.0,
        observed=0.20,
        n_perm=3,
        larger_is_better=False,
        rng=np.random.RandomState(0),
    )

    assert perm_scores == [0.10, 0.20, 0.40]
    assert p_value == pytest.approx(0.75)


def test_apply_multiple_comparisons_correction_skips_unknown_method_cleanly():
    results_df = pd.DataFrame({"p_value": [0.01, 0.20]})

    with pytest.warns(RuntimeWarning, match="Unknown correction method"):
        corrected = apply_multiple_comparisons_correction(results_df, correction_method="definitely_not_real")

    assert corrected.columns.tolist() == ["p_value"]
    assert corrected["p_value"].tolist() == [0.01, 0.20]
