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
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None
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
    p_adjusted: float = np.nan
    p_adjust_method: Optional[str] = None


def _detect_task(y: pd.Series, max_unique_for_class: int = 20) -> str:
    if y.dtype.name in {"category", "object", "bool"}:
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= max_unique_for_class:
        return "classification"
    if pd.api.types.is_float_dtype(y) and y.nunique() <= max_unique_for_class:
        return "classification"
    return "regression"


def _normalize_task_mode(task_mode: Optional[str]) -> str:
    if task_mode is None:
        return "auto"
    mode = str(task_mode).strip().lower()
    aliases = {
        "class": "classification",
        "classifier": "classification",
        "regress": "regression",
        "regressor": "regression",
    }
    mode = aliases.get(mode, mode)
    if mode in {"auto", "classification", "regression"}:
        return mode
    raise ValueError("task_mode must be one of: auto, classification, regression")


def _resolve_target_task(
    target: str,
    y: pd.Series,
    max_unique_for_class: int,
    task_mode: str,
    target_task_overrides: Optional[Dict[str, str]] = None,
) -> str:
    mode = task_mode
    if target_task_overrides:
        override_mode = target_task_overrides.get(str(target))
        if override_mode is not None:
            mode = override_mode
    if mode == "auto":
        return _detect_task(y, max_unique_for_class=max_unique_for_class)
    return mode


def _cv_iterator(
    task: str,
    y: np.ndarray,
    cv: int,
    random_state: int,
    groups: Optional[np.ndarray] = None,
):
    n = len(y)
    if n < 2:
        raise ValueError("Not enough samples for cross validation")

    if task == "classification":
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError("Classification requires at least two classes after alignment with X")

    if groups is not None:
        groups_arr = np.asarray(groups)
        n_groups = len(np.unique(groups_arr))
        if n_groups < 2:
            raise ValueError("Group CV requires at least two unique groups")

        if task == "classification":
            groups_per_class = []
            for cls in classes:
                cls_groups = np.unique(groups_arr[y == cls])
                groups_per_class.append(len(cls_groups))
            min_groups = int(min(groups_per_class)) if groups_per_class else 1
            if min_groups < 2:
                raise ValueError("Not enough groups per class for group-based CV")

            if StratifiedGroupKFold is not None:
                n_splits = max(2, min(cv, min_groups))
                if n_splits < cv:
                    warnings.warn(
                        f"Reducing CV folds from {cv} to {n_splits} due to limited groups per class",
                        RuntimeWarning,
                    )
                return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            warnings.warn(
                "StratifiedGroupKFold is unavailable; using GroupKFold without stratification.",
                RuntimeWarning,
            )
            n_splits = max(2, min(cv, n_groups))
            if n_splits < cv:
                warnings.warn(
                    f"Reducing CV folds from {cv} to {n_splits} due to limited group count",
                    RuntimeWarning,
                )
            return GroupKFold(n_splits=n_splits)

        n_splits = max(2, min(cv, n_groups))
        if n_splits < cv:
            warnings.warn(
                f"Reducing CV folds from {cv} to {n_splits} due to limited group count",
                RuntimeWarning,
            )
        return GroupKFold(n_splits=n_splits)

    if task == "classification":
        min_per_class = int(counts.min())
        if min_per_class < 2:
            raise ValueError(
                "Classification CV requires at least 2 samples in each class. "
                "Use regression or merge sparse classes."
            )
        n_splits = max(2, min(cv, min_per_class))
        if n_splits < cv:
            warnings.warn(f"Reducing CV folds from {cv} to {n_splits} due to small class counts", RuntimeWarning)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # R2 is the primary regression metric and requires at least 2 samples per test fold.
    max_splits_for_r2 = n // 2
    if max_splits_for_r2 < 2:
        raise ValueError(
            "Regression with R2 requires at least 4 samples after alignment "
            "to ensure each test fold has at least 2 samples."
        )
    n_splits = max(2, min(cv, max_splits_for_r2))
    if n_splits < cv:
        warnings.warn(
            f"Reducing CV folds from {cv} to {n_splits} so regression test folds have at least 2 samples",
            RuntimeWarning,
        )
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


def _inner_cv(
    task: str,
    y: np.ndarray,
    random_state: int,
    groups: Optional[np.ndarray] = None,
):
    n = len(y)
    if groups is not None:
        groups_arr = np.asarray(groups)
        n_groups = len(np.unique(groups_arr))
        if n_groups < 2:
            return None
        if task == "classification":
            classes = np.unique(y)
            groups_per_class = [len(np.unique(groups_arr[y == cls])) for cls in classes]
            min_groups = int(min(groups_per_class)) if groups_per_class else 1
            if min_groups < 2:
                return None
            if StratifiedGroupKFold is not None:
                n_splits = max(2, min(3, min_groups))
                return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            n_splits = max(2, min(3, n_groups))
            return GroupKFold(n_splits=n_splits)
        n_splits = max(2, min(3, n_groups))
        return GroupKFold(n_splits=n_splits)

    if task == "classification":
        classes, counts = np.unique(y, return_counts=True)
        min_per_class = int(counts.min()) if len(counts) else 1
        if min_per_class < 2:
            return None
        n_splits = max(2, min(3, min_per_class))
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    max_splits_for_r2 = n // 2
    if max_splits_for_r2 < 2:
        return None
    n_splits = max(2, min(3, max_splits_for_r2))
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
    scorer,
    groups: Optional[np.ndarray] = None,
) -> Tuple[List[float], np.ndarray]:
    scores: List[float] = []
    preds = np.empty_like(y, dtype=float)
    split_kwargs = {"X": X}
    if task == "classification":
        split_kwargs["y"] = y
    if groups is not None:
        split_kwargs["groups"] = groups
    for train, test in cv_splits.split(**split_kwargs):
        model = clone(base_model)
        if groups is not None and isinstance(model, GridSearchCV):
            model.fit(X[train], y[train], groups=groups[train])
        else:
            model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        preds[test] = y_pred
        fold_score = scorer(y[test], y_pred)
        if not np.isfinite(fold_score):
            raise ValueError(
                "Non-finite CV score encountered (often R2 on a fold with fewer than 2 test samples). "
                "Reduce CV folds or use more data."
            )
        scores.append(float(fold_score))
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
    rng: np.random.RandomState,
    groups: Optional[np.ndarray] = None,
) -> Tuple[List[float], float]:
    perm_scores = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        scores, _ = _fit_and_score_cv(X, y_perm, task, estimator, cv_splits, scorer, groups=groups)
        perm_scores.append(np.mean(scores))
    if larger_is_better:
        p = (1.0 + np.sum(np.asarray(perm_scores) >= observed)) / (n_perm + 1.0)
    else:
        p = (1.0 + np.sum(np.asarray(perm_scores) <= observed)) / (n_perm + 1.0)
    return perm_scores, float(p)


def _normalize_correction_method(method: Optional[str]) -> Optional[str]:
    if method is None:
        return None
    m = str(method).strip().lower()
    if m in {"none", "no", "off", "false", "0", "na", "n/a"}:
        return None
    return m


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
    correction_method: Optional[str] = "fdr_bh",
    groups: Optional[pd.Series] = None,
    task_mode: str = "auto",
    target_task_overrides: Optional[Dict[str, str]] = None,
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
    correction_method : Optional[str]
        Multiple-comparisons correction method passed to statsmodels.multipletests (e.g., "fdr_bh", "bonferroni", "holm").
        Use None or "none" to disable adjustment.
    groups : Optional[Series or array-like]
        Optional group labels aligned to X/Y. Enables GroupKFold; uses StratifiedGroupKFold for classification if available.
    task_mode : str
        Global task rule for targets: "auto", "classification", or "regression".
        Per-target overrides (if any) take precedence.
    target_task_overrides : Optional[Dict[str, str]]
        Optional mapping from target column name to "auto", "classification", or "regression".
        Use this to explicitly force task type for individual targets.

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

    group_series = None
    if groups is not None:
        if isinstance(groups, pd.Series):
            group_series = groups.reindex(X.index)
        else:
            if len(groups) != len(X):
                raise ValueError("groups must be the same length as X")
            group_series = pd.Series(groups, index=X.index)

    results: List[TargetResult] = []
    extra_metric_rows: List[Dict[str, float]] = []
    supported_metrics = set(_metric_functions("classification")) | set(_metric_functions("regression"))
    unknown_metrics = sorted(set(extra_metric_names) - supported_metrics)
    cv_preds: Dict[str, np.ndarray] = {}
    normalized_task_mode = _normalize_task_mode(task_mode)
    normalized_overrides: Dict[str, str] = {}
    if target_task_overrides:
        for target_name, mode in target_task_overrides.items():
            normalized_overrides[str(target_name)] = _normalize_task_mode(mode)

    for target in Y.columns:
        y_raw = Y[target].dropna()
        common_idx = X.index.intersection(y_raw.index)
        group_vals = None
        if group_series is not None:
            group_aligned = group_series.loc[common_idx].dropna()
            if group_aligned.empty:
                warnings.warn(
                    f"Skipping {target}: group column has no usable values after alignment",
                    RuntimeWarning,
                )
                continue
            common_idx = common_idx.intersection(group_aligned.index)
            group_vals = group_aligned.loc[common_idx].to_numpy()

        y_raw = y_raw.loc[common_idx]

        if len(y_raw) < 2:
            warnings.warn(f"Skipping {target}: fewer than two samples remain after alignment", RuntimeWarning)
            continue

        X_sub = X_mat[[X.index.get_loc(i) for i in common_idx]]

        task = _resolve_target_task(
            target=str(target),
            y=y_raw,
            max_unique_for_class=max_unique_for_class,
            task_mode=normalized_task_mode,
            target_task_overrides=normalized_overrides,
        )
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

        try:
            cv_splits = _cv_iterator(task, y, cv, random_state, groups=group_vals)
        except ValueError as exc:
            warnings.warn(f"Skipping {target}: {exc}", RuntimeWarning)
            continue
        try:
            if group_vals is not None:
                actual_folds = int(
                    cv_splits.get_n_splits(
                        X_sub,
                        y if task == "classification" else None,
                        groups=group_vals,
                    )
                )
            else:
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
            inner_cv = _inner_cv(task, y, random_state, groups=group_vals)
            if inner_cv is None:
                warnings.warn(
                    f"Skipping hyperparameter tuning for {target}: not enough groups for inner CV.",
                    RuntimeWarning,
                )
                base_estimator = pipeline
            else:
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

        try:
            cv_scores, preds = _fit_and_score_cv(
                X_sub,
                y,
                task,
                base_estimator,
                cv_splits,
                scorer,
                groups=group_vals,
            )
        except ValueError as exc:
            warnings.warn(f"Skipping {target}: {exc}", RuntimeWarning)
            continue
        observed = float(np.mean(cv_scores))
        extra_metrics: Dict[str, float] = {}
        if extra_metric_names:
            metric_funcs = _metric_functions(task)
            for name in extra_metric_names:
                if name in metric_funcs:
                    extra_metrics[f"metric_{name}"] = float(metric_funcs[name](y, preds))

        try:
            perm_scores, p_val = _perm_test(
                X_sub,
                y,
                task,
                base_estimator,
                cv_splits,
                scorer,
                observed,
                int(n_permutations),
                larger_is_better,
                rng,
                groups=group_vals,
            )
        except ValueError as exc:
            warnings.warn(f"Skipping {target}: {exc}", RuntimeWarning)
            continue

        results.append(TargetResult(
            target=str(target),
            task=task,
            metric_name=metric_name,
            observed=float(observed),
            p_value=float(p_val),
            p_fdr=np.nan,
            cv_scores=[float(s) for s in cv_scores],
            perm_scores=[float(s) for s in perm_scores],
            cv_folds_used=int(actual_folds),
            p_adjusted=np.nan,
            p_adjust_method=None,
        ))
        extra_metric_rows.append(extra_metrics)
        cv_preds[target] = preds

    if not results:
        return pd.DataFrame(columns=[
            "target","task","metric_name","observed","p_value","p_fdr","p_adjusted","p_adjust_method",
            "cv_scores","perm_scores","cv_folds_used"
        ]), {}

    # Multiple-comparisons correction across targets
    adjust_method = _normalize_correction_method(correction_method)
    p_adj = None
    if adjust_method:
        try:
            _, p_adj, _, _ = multipletests([r.p_value for r in results], method=adjust_method)
        except Exception:
            warnings.warn(
                f"Unknown correction method '{correction_method}'. Skipping adjustment.",
                RuntimeWarning,
            )
            adjust_method = None

    if adjust_method and p_adj is not None:
        for r, adj in zip(results, p_adj):
            r.p_adjusted = float(adj)
            r.p_adjust_method = adjust_method
            if adjust_method == "fdr_bh":
                r.p_fdr = float(adj)
    else:
        for r in results:
            r.p_adjusted = np.nan
            r.p_adjust_method = None

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
    for col in ("p_adjusted", "p_adjust_method", "p_fdr"):
        if col in results_df.columns and results_df[col].isna().all():
            results_df = results_df.drop(columns=[col])
    return results_df, cv_preds
