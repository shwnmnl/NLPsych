import numpy as np
import pandas as pd
import pytest

from nlpsych.modeling import _cv_iterator, auto_cv_with_permutation


def test_cv_iterator_rejects_singleton_class():
    y = np.array([0, 0, 0, 1])
    with pytest.raises(ValueError, match="at least 2 samples in each class"):
        _cv_iterator("classification", y, cv=5, random_state=42)


def test_auto_cv_respects_forced_regression_mode_for_low_cardinality_numeric_target():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=(30, 6)))
    y = pd.Series(np.tile(np.arange(5), 6)[:30], name="target")
    Y = pd.DataFrame({"target": y})

    results_df, preds = auto_cv_with_permutation(
        X=X,
        Y=Y,
        cv=3,
        n_permutations=8,
        random_state=0,
        task_mode="regression",
    )

    assert len(results_df) == 1
    assert results_df.loc[0, "task"] == "regression"
    assert "target" in preds


def test_auto_cv_target_override_can_force_regression_in_auto_mode():
    rng = np.random.RandomState(7)
    X = pd.DataFrame(rng.normal(size=(36, 5)))
    y = pd.Series(np.tile(np.arange(6), 6)[:36], name="score")
    Y = pd.DataFrame({"score": y})

    results_df, _ = auto_cv_with_permutation(
        X=X,
        Y=Y,
        cv=3,
        n_permutations=8,
        random_state=7,
        max_unique_for_class=20,
        task_mode="auto",
        target_task_overrides={"score": "regression"},
    )

    assert len(results_df) == 1
    assert results_df.loc[0, "task"] == "regression"
