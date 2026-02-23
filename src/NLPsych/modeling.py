from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import warnings

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    f_classif,
    f_regression,
)
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


def _metric_functions(task: str):
    if task == "classification":
        return {
            "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
            "balanced_accuracy": lambda y_true, y_pred: balanced_accuracy_score(y_true, y_pred),
            "precision": lambda y_true, y_pred: precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": lambda y_true, y_pred: recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1": lambda y_true, y_pred: f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }
    return {
        "r2": lambda y_true, y_pred: r2_score(y_true, y_pred),
        "mae": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
        "mse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
        "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def _inner_cv(task: str, y: np.ndarray, random_state: int):
    n = len(y)
    if task == "classification":
        classes, counts = np.unique(y, return_counts=True)
        min_per_class = int(counts.min()) if len(counts) else 1
        n_splits = max(2, min(3, min_per_class))
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_splits = max(2, min(3, n))
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _build_selector(
    task: str,
    feature_selection: Optional[str],
    k_best: int,
    percentile: int,
    variance_threshold: float,
    n_features: int,
):
    if n_features < 1:
        return None
    score_func = f_classif if task == "classification" else f_regression
    if feature_selection == "kbest":
        k = min(max(1, int(k_best)), n_features)
        return SelectKBest(score_func=score_func, k=k)
    if feature_selection == "percentile":
        pct = min(max(1, int(percentile)), 100)
        return SelectPercentile(score_func=score_func, percentile=pct)
    if feature_selection == "variance":
        return VarianceThreshold(threshold=float(variance_threshold))
    return None


def _build_reducer(
    reduce_method: Optional[str],
    n_components: Optional[int],
    max_components: int,
):
    if not reduce_method or max_components < 1:
        return None
    method = str(reduce_method).lower()
    if method != "pca":
        warnings.warn(f"Unknown reduction method '{reduce_method}'. Skipping reduction.", RuntimeWarning)
        return None
    if n_components is None:
        n_use = min(50, max_components)
    else:
        try:
            n_use = int(n_components)
        except (TypeError, ValueError):
            n_use = min(50, max_components)
    if n_use < 1:
        return None
    if n_use > max_components:
        warnings.warn(
            f"Requested n_components={n_use} exceeds max={max_components}. Using {max_components}.",
            RuntimeWarning,
        )
        n_use = max_components
    return PCA(n_components=n_use)


def _build_pipeline(base_model, selector, reducer, scale_X: bool):
    steps = []
    if scale_X:
        steps.append(("scaler", StandardScaler()))
    if reducer is not None:
        steps.append(("reducer", reducer))
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("model", base_model))
    return Pipeline(steps)


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
    estimator,
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
        scores, _ = _fit_and_score_cv(X, y_perm, task, estimator, cv_splits, scorer)
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
    scale_X: bool = True,
    report_metrics: Optional[List[str]] = None,
    tune_hyperparams: bool = False,
    classifier_param_grid: Optional[Dict[str, List[Any]]] = None,
    regressor_param_grid: Optional[Dict[str, List[Any]]] = None,
    feature_selection: Optional[str] = None,
    k_best: int = 200,
    percentile: int = 20,
    variance_threshold: float = 0.0,
    reduce_method: Optional[str] = None,
    reduce_n_components: Optional[int] = None,
    reduce_components_grid: Optional[List[int]] = None,
    k_best_grid: Optional[List[int]] = None,
    percentile_grid: Optional[List[int]] = None,
    variance_threshold_grid: Optional[List[float]] = None,
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
    report_metrics : Optional[List[str]]
        Additional metric names to compute alongside the primary metric.
    tune_hyperparams : bool
        If True, run a lightweight grid search on each outer CV fold.
    classifier_param_grid : Optional[Dict[str, List[Any]]]
        Hyperparameter grid for classifier_model (unprefixed parameter names).
    regressor_param_grid : Optional[Dict[str, List[Any]]]
        Hyperparameter grid for regressor_model (unprefixed parameter names).
    feature_selection : Optional[str]
        Feature selection strategy: "kbest", "percentile", "variance", or None.
    k_best : int
        Number of top features to keep when using SelectKBest.
    percentile : int
        Percentile of features to keep when using SelectPercentile.
    variance_threshold : float
        Variance threshold for VarianceThreshold feature selection.
    reduce_method : Optional[str]
        Dimensionality reduction method for embeddings (currently supports "pca" or None).
    reduce_n_components : Optional[int]
        Number of components for reduction (applied within CV).
    reduce_components_grid : Optional[List[int]]
        Optional list of component counts to tune when hyperparameter tuning is enabled.
    k_best_grid : Optional[List[int]]
        Optional list of K values to tune when using KBest feature selection.
    percentile_grid : Optional[List[int]]
        Optional list of percentile values to tune when using Percentile feature selection.
    variance_threshold_grid : Optional[List[float]]
        Optional list of variance thresholds to tune when using VarianceThreshold selection.

    Returns
    -------
    results_df : DataFrame
        One row per target with observed CV score, permutation p-value, FDR correction, and CV metadata.
    cv_preds : Dict[str, np.ndarray]
        Cross-validated out-of-fold predictions per target (aligned to the samples used for that target).
    """
    if classifier_model is None:
        classifier_model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            n_jobs=None
        )
    if regressor_model is None:
        regressor_model = Ridge(alpha=1.0)

    rng = check_random_state(random_state)
    extra_metric_names = [m.strip().lower() for m in report_metrics or [] if str(m).strip()]

    # Keep raw features; scaling (if enabled) happens inside the CV pipeline to avoid leakage.
    X_mat = X.values.astype(float)

    results: List[TargetResult] = []
    extra_metric_rows: List[Dict[str, float]] = []
    supported_metrics = set(_metric_functions("classification")) | set(_metric_functions("regression"))
    unknown_metrics = sorted(set(extra_metric_names) - supported_metrics)
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
            base_model = classifier_model
            param_grid = classifier_param_grid or {}
        else:
            y = y_raw.values.astype(float)
            base_model = regressor_model
            param_grid = regressor_param_grid or {}

        cv_splits = _cv_iterator(task, y, cv, random_state)
        try:
            actual_folds = int(cv_splits.get_n_splits(X_sub, y if task == "classification" else None))
        except Exception:
            actual_folds = cv

        n_samples = len(y)
        n_features = int(X_sub.shape[1])
        if actual_folds > 1:
            min_train = n_samples - math.ceil(n_samples / actual_folds)
        else:
            min_train = n_samples
        max_components = max(1, min(n_features, min_train))

        reducer = _build_reducer(reduce_method, reduce_n_components, max_components)
        effective_features = int(getattr(reducer, "n_components", n_features)) if reducer is not None else n_features

        selector = _build_selector(
            task,
            feature_selection,
            k_best,
            percentile,
            variance_threshold,
            effective_features,
        )
        pipeline = _build_pipeline(base_model, selector, reducer, scale_X)

        pipeline_grid: Dict[str, List[Any]] = {}
        if param_grid:
            pipeline_grid.update({f"model__{k}": v for k, v in param_grid.items()})

        if reduce_method and reduce_components_grid:
            if reducer is None:
                warnings.warn("Reduction grid provided but reduction is disabled.", RuntimeWarning)
            else:
                comps = [int(c) for c in reduce_components_grid if isinstance(c, (int, float, np.integer, np.floating))]
                comps = [c for c in comps if 1 <= c <= max_components]
                if comps:
                    pipeline_grid["reducer__n_components"] = sorted(set(comps))
                else:
                    warnings.warn("Reduction grid empty after filtering invalid component counts.", RuntimeWarning)

        if feature_selection == "kbest" and k_best_grid:
            k_vals = [int(k) for k in k_best_grid if isinstance(k, (int, float, np.integer, np.floating))]
            k_vals = [k for k in k_vals if 1 <= k <= effective_features]
            if k_vals:
                pipeline_grid["selector__k"] = sorted(set(k_vals))
            else:
                warnings.warn("KBest grid empty after filtering invalid values.", RuntimeWarning)
        elif feature_selection == "percentile" and percentile_grid:
            p_vals = [int(p) for p in percentile_grid if isinstance(p, (int, float, np.integer, np.floating))]
            p_vals = [p for p in p_vals if 1 <= p <= 100]
            if p_vals:
                pipeline_grid["selector__percentile"] = sorted(set(p_vals))
            else:
                warnings.warn("Percentile grid empty after filtering invalid values.", RuntimeWarning)
        elif feature_selection == "variance" and variance_threshold_grid:
            v_vals = [float(v) for v in variance_threshold_grid if isinstance(v, (int, float, np.integer, np.floating))]
            if v_vals:
                pipeline_grid["selector__threshold"] = sorted(set(v_vals))
            else:
                warnings.warn("Variance threshold grid empty after filtering invalid values.", RuntimeWarning)

        if tune_hyperparams and pipeline_grid:
            inner_cv = _inner_cv(task, y, random_state)
            base_estimator = GridSearchCV(
                pipeline,
                pipeline_grid,
                cv=inner_cv,
                n_jobs=None
            )
        else:
            if tune_hyperparams and not pipeline_grid:
                warnings.warn(f"No hyperparameter grid for {target}; skipping tuning.", RuntimeWarning)
            base_estimator = pipeline

        cv_scores, preds = _fit_and_score_cv(X_sub, y, task, base_estimator, cv_splits, scorer)
        observed = float(np.mean(cv_scores))
        extra_metrics: Dict[str, float] = {}
        if extra_metric_names:
            metric_funcs = _metric_functions(task)
            for name in extra_metric_names:
                if name in metric_funcs:
                    extra_metrics[f"metric_{name}"] = float(metric_funcs[name](y, preds))

        perm_scores, p_val = _perm_test(
            X_sub, y, task, base_estimator, cv_splits, scorer,
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
        extra_metric_rows.append(extra_metrics)
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

    if unknown_metrics:
        warnings.warn(
            f"Skipping unsupported metrics: {', '.join(unknown_metrics)}",
            RuntimeWarning
        )

    rows = []
    for r, extra in zip(results, extra_metric_rows):
        row = r.__dict__.copy()
        row.update(extra)
        rows.append(row)
    results_df = pd.DataFrame(rows)
    return results_df, cv_preds
