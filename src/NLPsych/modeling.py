from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import warnings

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import check_random_state
from statsmodels.stats.multitest import multipletests


# ===== Section: Modeling with cross validation and permutation test =====
@dataclass
class TargetResult:
    target: str
    task: str
    metric_name: str
    observed: float
    p_value: float
    p_fdr: float
    cv_scores: List[float]
    perm_scores: List[float]
    cv_folds_used: int


def _detect_task(y: pd.Series, max_unique_for_class: int = 20) -> str:
    if y.dtype.name in {"category", "object", "bool"}:
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= max_unique_for_class:
        return "classification"
    if pd.api.types.is_float_dtype(y) and y.nunique() <= max_unique_for_class:
        return "classification"
    return "regression"


def _cv_iterator(task: str, y: np.ndarray, cv: int, random_state: int):
    n = len(y)
    if n < 2:
        raise ValueError("Not enough samples for cross validation")

    if task == "classification":
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError("Classification requires at least two classes after alignment with X")
        min_per_class = int(counts.min())
        n_splits = max(2, min(cv, min_per_class))
        if n_splits < cv:
            warnings.warn(f"Reducing CV folds from {cv} to {n_splits} due to small class counts", RuntimeWarning)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)        
    else:
        n_splits = max(2, min(cv, n))
        if n_splits < cv:
            warnings.warn(f"Reducing CV folds from {cv} to {n_splits} due to small sample size", RuntimeWarning)
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _primary_metric(task: str):
    if task == "classification":
        return "accuracy", accuracy_score, True
    return "r2", r2_score, True


def _fit_and_score_cv(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    base_model,
    cv_splits,
    scorer
) -> Tuple[List[float], np.ndarray]:
    scores: List[float] = []
    preds = np.empty_like(y, dtype=float)
    for train, test in cv_splits.split(X, y if task == "classification" else None):
        model = clone(base_model)
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        preds[test] = y_pred
        scores.append(scorer(y[test], y_pred))
    return scores, preds


def _perm_test(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    base_model,
    cv_splits,
    scorer,
    observed: float,
    n_perm: int,
    larger_is_better: bool,
    rng: np.random.RandomState
) -> Tuple[List[float], float]:
    perm_scores = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        scores, _ = _fit_and_score_cv(X, y_perm, task, base_model, cv_splits, scorer)
        perm_scores.append(np.mean(scores))
    if larger_is_better:
        p = (1.0 + np.sum(np.asarray(perm_scores) >= observed)) / (n_perm + 1.0)
    else:
        p = (1.0 + np.sum(np.asarray(perm_scores) <= observed)) / (n_perm + 1.0)
    return perm_scores, float(p)


def auto_cv_with_permutation(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    cv: int = 5,
    n_permutations: int = 200,
    random_state: int = 42,
    max_unique_for_class: int = 20,
    classifier_model=None,
    regressor_model=None,
    scale_X: bool = True
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Cross-validated evaluation with a permutation test for one or more targets.

    Parameters
    ----------
    X : DataFrame
        Feature matrix indexed like Y (rows are samples). Values must be numeric.
    Y : DataFrame
        One or more target columns to evaluate, aligned by index with X.
    cv : int
        Desired number of folds. May be reduced automatically for small n or class counts.
    n_permutations : int
        Number of label permutations for the permutation test.
    random_state : int
        Seed for reproducibility (CV shuffling and permutations).
    max_unique_for_class : int
        Threshold to treat low-cardinality numeric targets as classification.
    classifier_model
        Base estimator for classification. If None, LogisticRegression(max_iter=1000) is used.
    regressor_model
        Base estimator for regression. If None, Ridge(alpha=1.0) is used.
    scale_X : bool
        If True, standardize X before modeling.

    Returns
    -------
    results_df : DataFrame
        One row per target with observed CV score, permutation p-value, FDR correction, and CV metadata.
    cv_preds : Dict[str, np.ndarray]
        Cross-validated out-of-fold predictions per target (aligned to the samples used for that target).
    """
    if classifier_model is None:
        classifier_model = LogisticRegression(max_iter=1000,
        solver="lbfgs",
        multi_class="auto",
        n_jobs=None)
    if regressor_model is None:
        regressor_model = Ridge(alpha=1.0)

    rng = check_random_state(random_state)

    X_mat = X.values.astype(float)
    if scale_X:
        scaler = StandardScaler().fit(X_mat)
        X_mat = scaler.transform(X_mat)

    results: List[TargetResult] = []
    cv_preds: Dict[str, np.ndarray] = {}

    for target in Y.columns:
        y_raw = Y[target].dropna()
        common_idx = X.index.intersection(y_raw.index)
        y_raw = y_raw.loc[common_idx]

        if len(y_raw) < 2:
            warnings.warn(f"Skipping {target}: fewer than two samples remain after alignment", RuntimeWarning)
            continue

        X_sub = X_mat[[X.index.get_loc(i) for i in common_idx]]

        task = _detect_task(y_raw, max_unique_for_class)
        metric_name, scorer, larger_is_better = _primary_metric(task)

        if task == "classification":
            classes = np.unique(y_raw.values)
            if len(classes) < 2:
                warnings.warn(f"Skipping {target}: only a single class after alignment", RuntimeWarning)
                continue
            le = LabelEncoder()
            y = le.fit_transform(y_raw.values)
            base_model = make_pipeline(classifier_model)
        else:
            y = y_raw.values.astype(float)
            base_model = make_pipeline(regressor_model)

        cv_splits = _cv_iterator(task, y, cv, random_state)
        try:
            actual_folds = int(cv_splits.get_n_splits(X_sub, y if task == "classification" else None))
        except Exception:
            actual_folds = cv

        cv_scores, preds = _fit_and_score_cv(X_sub, y, task, base_model, cv_splits, scorer)
        observed = float(np.mean(cv_scores))

        perm_scores, p_val = _perm_test(
            X_sub, y, task, base_model, cv_splits, scorer,
            observed, int(n_permutations), larger_is_better, rng
        )

        results.append(TargetResult(
            target=str(target),
            task=task,
            metric_name=metric_name,
            observed=float(observed),
            p_value=float(p_val),
            p_fdr=np.nan,
            cv_scores=[float(s) for s in cv_scores],
            perm_scores=[float(s) for s in perm_scores],
            cv_folds_used=int(actual_folds)
        ))
        cv_preds[target] = preds

    if not results:
        return pd.DataFrame(columns=[
            "target","task","metric_name","observed","p_value","p_fdr",
            "cv_scores","perm_scores","cv_folds_used"
        ]), {}

    # FDR correction across targets
    p_vals = [r.p_value for r in results]
    _, p_fdr, _, _ = multipletests(p_vals, method="fdr_bh")
    for r, adj in zip(results, p_fdr):
        r.p_fdr = float(adj)

    results_df = pd.DataFrame([r.__dict__ for r in results])
    return results_df, cv_preds
