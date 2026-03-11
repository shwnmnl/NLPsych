from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

import pandas as pd
import numpy as np

def _pick_overall(overall_obj):
    """
    Normalize overall stats payloads to a single aggregate dictionary.

    Parameters
    ----------
    overall_obj
        Overall stats output that may be a plain dict or a dict containing a
        ``"combined"`` key.

    Returns
    -------
    dict
        Combined overall statistics dictionary, or an empty dict when missing.
    """
    if isinstance(overall_obj, dict) and "combined" in overall_obj:
        return overall_obj["combined"]
    return overall_obj or {}

def interpret_stats(overall: dict) -> list[str]:
    """
    Generate short natural-language interpretations of descriptive statistics.

    Parameters
    ----------
    overall : dict
        Aggregate descriptive-statistics dictionary.

    Returns
    -------
    list[str]
        Human-readable interpretation bullets derived from key metrics.
    """
    out: List[str] = []
    if not overall:
        return out
    
    ld = overall.get("lexical_diversity")
    if ld is not None:
        if ld >= 0.8:
            out.append("Lexical diversity is very high which suggests a rich vocabulary with little repetition.")
        elif ld >= 0.6:
            out.append("Lexical diversity is moderate which suggests some repetition and good variety.")
        else:
            out.append("Lexical diversity is relatively low which suggests frequent repetition of word types.")
    asl = overall.get("avg_sentence_length")
    if asl is not None:
        if asl >= 20:
            out.append("Average sentence length is long which often reflects complex structure.")
        elif asl >= 12:
            out.append("Average sentence length is mid range which often reflects balanced structure.")
        else:
            out.append("Average sentence length is short which often reflects simple direct phrasing.")
    posb = overall.get("pos_distribution_basic", {})
    total_words = overall.get("total_words", 0) or 0
    if total_words and posb:
        nouns = posb.get("nouns", 0)
        verbs = posb.get("verbs", 0)
        if nouns + verbs:
            nv_ratio = nouns / max(1, verbs)
            if nv_ratio >= 2:
                out.append("Noun to verb ratio is high which can indicate dense nominal style.")
            elif nv_ratio <= 0.7:
                out.append("Verb to noun ratio is high which can indicate action oriented style.")
    return out


def _has_nonempty(val: Any) -> bool:
    """
    Check whether a value should count as populated for reporting logic.

    Parameters
    ----------
    val : Any
        Candidate value.

    Returns
    -------
    bool
        True when the value is not None/empty after type-aware checks.
    """
    if val is None:
        return False
    if isinstance(val, (str, bytes)):
        return bool(str(val).strip())
    if isinstance(val, (list, tuple, set, dict)):
        return len(val) > 0
    return True


def _pairs_to_dict(val: Any) -> Dict[str, str]:
    """
    Convert mapping-like inputs to a string-keyed dictionary.

    Parameters
    ----------
    val : Any
        Either a dict or an iterable of 2-item pairs.

    Returns
    -------
    Dict[str, str]
        Normalized dictionary with string keys and values.
    """
    if isinstance(val, dict):
        return {str(k): str(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        out: Dict[str, str] = {}
        for item in val:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out[str(item[0])] = str(item[1])
        return out
    return {}


def _advanced_settings_notes(model_config: Optional[Dict[str, Any]], target: Optional[str] = None) -> List[str]:
    """
    Summarize advanced modeling settings into report-ready note strings.

    Parameters
    ----------
    model_config : Optional[Dict[str, Any]]
        Modeling configuration dictionary.
    target : Optional[str], default=None
        Target name used to resolve per-target task overrides.

    Returns
    -------
    List[str]
        Descriptive notes for non-default/advanced settings.
    """
    if not isinstance(model_config, dict):
        return []

    notes: List[str] = []

    feature_selection = str(model_config.get("feature_selection", "")).strip().lower()
    if feature_selection and feature_selection != "none":
        if feature_selection == "kbest":
            notes.append(f"KBest feature selection (k={model_config.get('k_best')})")
        elif feature_selection == "percentile":
            notes.append(f"Percentile feature selection ({model_config.get('percentile')}%)")
        elif feature_selection in {"variance threshold", "variance"}:
            notes.append(f"variance-threshold feature selection ({model_config.get('variance_threshold')})")
        else:
            notes.append(f"feature selection ({feature_selection})")

    reduce_method = model_config.get("reduce_method")
    if isinstance(reduce_method, str) and reduce_method.strip():
        method = reduce_method.strip().lower()
        if method == "pca":
            n_components = model_config.get("reduce_n_components")
            if n_components is not None:
                notes.append(f"PCA dimensionality reduction (n_components={n_components})")
            else:
                notes.append("PCA dimensionality reduction")
        else:
            notes.append(f"{reduce_method} dimensionality reduction")

    if bool(model_config.get("tune_hyperparams")):
        grid_parts: List[str] = []
        if _has_nonempty(model_config.get("classifier_param_grid")):
            grid_parts.append("classifier grid")
        if _has_nonempty(model_config.get("regressor_param_grid")):
            grid_parts.append("regressor grid")
        if _has_nonempty(model_config.get("reduce_components_grid")):
            grid_parts.append("reduction grid")
        if _has_nonempty(model_config.get("k_best_grid")) or _has_nonempty(model_config.get("percentile_grid")) or _has_nonempty(model_config.get("variance_threshold_grid")):
            grid_parts.append("feature-selection grid")
        if grid_parts:
            notes.append(f"hyperparameter tuning with custom {', '.join(grid_parts)}")
        else:
            notes.append("light-grid hyperparameter tuning")

    group_col = model_config.get("group_col")
    if isinstance(group_col, str) and group_col.strip():
        notes.append(f"group-aware CV using {group_col}")

    target_name = str(target) if target is not None else None
    override_map = _pairs_to_dict(model_config.get("target_task_overrides"))
    if target_name and target_name in override_map:
        notes.append(f"explicit task override ({override_map[target_name]}) for this target")
    else:
        task_mode = str(model_config.get("task_mode", "")).strip().lower()
        if task_mode in {"classification", "regression"}:
            notes.append(f"forced task mode ({task_mode})")

    return notes


def interpret_model_row(row: pd.Series, model_config: Optional[Dict[str, Any]] = None) -> list[str]:
    """
    Create interpretation bullets for one modeling-results row.

    Parameters
    ----------
    row : pd.Series
        One row from the modeling results table.
    model_config : Optional[Dict[str, Any]], default=None
        Optional modeling configuration used to mention advanced settings.

    Returns
    -------
    list[str]
        Interpretation bullets describing model quality and significance.
    """
    out: List[str] = []
    metric = str(row.get("metric_name", "")).upper()
    score_raw = row.get("observed", np.nan)
    score = float(score_raw) if pd.notnull(score_raw) else np.nan
    p_raw = row.get("p_value", np.nan)
    p = float(p_raw) if pd.notnull(p_raw) else np.nan
    p_adjusted = row.get("p_adjusted", np.nan)
    p_adjust_method = row.get("p_adjust_method", None)
    if (p_adjust_method is None or p_adjust_method == "" or pd.isna(p_adjust_method)):
        legacy_fdr = row.get("p_fdr", np.nan)
        if pd.notnull(legacy_fdr):
            p_adjusted = legacy_fdr if pd.isna(p_adjusted) else p_adjusted
            p_adjust_method = "fdr_bh"
    folds = int(row.get("cv_folds_used", 0)) if pd.notnull(row.get("cv_folds_used", np.nan)) else None
    task = row.get("task", "classification")

    if pd.isna(score):
        out.append(
            f"{metric} is undefined for at least one fold (often due to too few test samples). "
            "Consider fewer CV folds."
        )
    else:
        if task == "classification":
            if score >= 0.80:
                out.append(f"{metric} indicates strong discrimination.")
            elif score >= 0.60:
                out.append(f"{metric} indicates moderate discrimination.")
            else:
                out.append(f"{metric} indicates weak discrimination.")
        else:
            if score >= 0.50:
                out.append(f"{metric} indicates strong fit.")
            elif score >= 0.20:
                out.append(f"{metric} indicates moderate fit.")
            elif score >= 0.00:
                out.append(f"{metric} indicates weak fit.")
            else:
                out.append(f"{metric} is negative which is worse than a constant baseline and suggests poor generalization.")

    method_label = None
    if p_adjust_method:
        method_map = {
            "fdr_bh": "FDR (Benjamini–Hochberg)",
            "fdr_by": "FDR (Benjamini–Yekutieli)",
            "bonferroni": "Bonferroni",
            "holm": "Holm",
            "sidak": "Sidak",
            "hochberg": "Hochberg",
            "hommel": "Hommel",
        }
        method_label = method_map.get(str(p_adjust_method).lower(), str(p_adjust_method))

    if pd.isna(p):
        out.append("Permutation test result is unavailable.")
    elif method_label and pd.notnull(p_adjusted):
        if float(p_adjusted) < 0.05:
            if task == "regression" and metric == "R2" and pd.notnull(score) and score < 0:
                out.append(
                    f"Permutation test with {method_label} correction is statistically reliable, "
                    "but negative R2 indicates poor predictive performance."
                )
            else:
                out.append(f"Permutation test with {method_label} correction indicates a statistically reliable effect.")
        elif p < 0.05:
            out.append(f"Permutation test is nominally significant but not after {method_label} correction.")
        else:
            out.append("Permutation test does not indicate statistical significance.")
    else:
        if p < 0.05:
            if task == "regression" and metric == "R2" and pd.notnull(score) and score < 0:
                out.append(
                    "Permutation test is nominally significant, but negative R2 indicates poor predictive performance."
                )
            else:
                out.append("Permutation test is nominally significant (no multiple-comparisons correction).")
        else:
            out.append("Permutation test does not indicate statistical significance.")

    if folds is not None and folds > 0:
        out.append(f"Cross validation used {folds} folds.")

    target = str(row.get("target", "target"))
    advanced_notes = _advanced_settings_notes(model_config, target=target)
    if advanced_notes:
        out.append(f"Advanced settings used: {'; '.join(advanced_notes)}.")

    return out


def summarize_model_row(
    row: pd.Series,
    include_adjusted: bool = True,
    model_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a narrative one-paragraph summary for one model result.

    Parameters
    ----------
    row : pd.Series
        One row from the modeling results table.
    include_adjusted : bool, default=True
        Whether to prefer adjusted p-values in the sentence when available.
    model_config : Optional[Dict[str, Any]], default=None
        Optional modeling configuration used to append advanced-setting notes.

    Returns
    -------
    str
        Human-readable summary sentence block for reporting.
    """
    target = str(row.get("target", "target"))
    task = str(row.get("task", "classification"))
    metric = str(row.get("metric_name", "score")).upper()
    score_raw = row.get("observed", np.nan)
    score = float(score_raw) if pd.notnull(score_raw) else np.nan
    p_raw = row.get("p_value", np.nan)
    p = float(p_raw) if pd.notnull(p_raw) else np.nan
    folds_val = row.get("cv_folds_used", np.nan)
    folds = int(folds_val) if pd.notnull(folds_val) else None

    p_adjusted = row.get("p_adjusted", np.nan)
    p_adjust_method = row.get("p_adjust_method", None)
    if (p_adjust_method is None or p_adjust_method == "" or pd.isna(p_adjust_method)):
        legacy_fdr = row.get("p_fdr", np.nan)
        if pd.notnull(legacy_fdr):
            p_adjusted = legacy_fdr if pd.isna(p_adjusted) else p_adjusted
            p_adjust_method = "fdr_bh"

    task_label = "classification" if task == "classification" else "regression"
    parts: List[str] = [f"For target {target}, we ran a {task_label} model"]
    if folds is not None and folds > 0:
        parts[-1] += f" with {folds}-fold cross validation"
    parts[-1] += "."

    if pd.notnull(score):
        parts.append(f"The observed {metric} was {score:.3f}.")
    else:
        parts.append(f"The observed {metric} was undefined in at least one fold.")

    if pd.notnull(p):
        use_adjusted = bool(include_adjusted and p_adjust_method and pd.notnull(p_adjusted))
        if use_adjusted:
            if float(p_adjusted) < 0.05:
                if task == "regression" and metric == "R2" and pd.notnull(score) and score < 0:
                    finding = (
                        "suggesting a statistically reliable difference from permutations, "
                        "but with poor predictive performance because R2 is negative"
                    )
                else:
                    finding = "suggesting a statistically reliable effect after multiple-comparisons correction"
            elif p < 0.05:
                finding = "showing nominal significance that did not survive multiple-comparisons correction"
            else:
                finding = "indicating no statistically reliable effect"
            parts.append(
                f"Permutation testing gave p={p:.4f} and adjusted p={float(p_adjusted):.4f} ({p_adjust_method}), {finding}."
            )
        else:
            if p < 0.05:
                if task == "regression" and metric == "R2" and pd.notnull(score) and score < 0:
                    finding = (
                        "suggesting a statistically reliable difference from permutations, "
                        "but with poor predictive performance because R2 is negative"
                    )
                else:
                    finding = "suggesting a statistically reliable effect"
            else:
                finding = "indicating no statistically reliable effect"
            parts.append(f"Permutation testing gave p={p:.4f}, {finding}.")
    else:
        parts.append("Permutation testing was unavailable.")

    advanced_notes = _advanced_settings_notes(model_config, target=target)
    if advanced_notes:
        parts.append(f"Advanced settings used: {'; '.join(advanced_notes)}.")

    return " ".join(parts)

def _to_html_table(df: pd.DataFrame, index: bool = False) -> str:
    """
    Render a DataFrame as HTML with report table styling.

    Parameters
    ----------
    df : pd.DataFrame
        Table to render.
    index : bool, default=False
        Whether to include the DataFrame index in output.

    Returns
    -------
    str
        HTML table markup.
    """
    return df.to_html(index=index, classes="table", border=0, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x)

def _df_to_markdown_safe(df: pd.DataFrame, index: bool = False) -> str:
    """
    Convert a DataFrame to Markdown with a plain-text fallback.

    Parameters
    ----------
    df : pd.DataFrame
        Table to serialize.
    index : bool, default=False
        Whether to include the DataFrame index in output.

    Returns
    -------
    str
        Markdown table when available, otherwise ``to_string`` output.
    """
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)

def _stringify_config_value(val: Any) -> str:
    """
    Convert configuration values to compact display strings.

    Parameters
    ----------
    val : Any
        Raw configuration value.

    Returns
    -------
    str
        Stringified representation suitable for report tables.
    """
    if val is None:
        return "None"
    if isinstance(val, (list, tuple, set)):
        return ", ".join(str(v) for v in val)
    if isinstance(val, dict):
        return ", ".join(f"{k}={v}" for k, v in val.items())
    return str(val)

def _config_to_df(config: Optional[Dict[str, Any]], exclude_keys: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Convert a configuration dictionary into a two-column report DataFrame.

    Parameters
    ----------
    config : Optional[Dict[str, Any]]
        Configuration mapping to render.
    exclude_keys : Optional[List[str]], default=None
        Keys to omit from the rendered output.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with ``Setting``/``Value`` columns, or ``None`` when empty.
    """
    if not config:
        return None
    exclude = set(exclude_keys or [])
    rows = []
    for k in sorted(config.keys()):
        if k in exclude:
            continue
        v = config[k]
        if v is None:
            continue
        if isinstance(v, (list, tuple, set)) and not v:
            continue
        rows.append([k, _stringify_config_value(v)])
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["Setting", "Value"])

def build_report_payload(
    text_cols: list[str],
    stats_df: Optional[pd.DataFrame],
    overall_obj: Optional[dict],
    plot_df: Optional[pd.DataFrame],
    results_df: Optional[pd.DataFrame],
    stats_config: Optional[Dict[str, Any]] = None,
    embed_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Build full HTML and Markdown session reports from analysis outputs.

    Parameters
    ----------
    text_cols : list[str]
        Text columns selected in the session.
    stats_df : Optional[pd.DataFrame]
        Row-level descriptive statistics table.
    overall_obj : Optional[dict]
        Aggregate descriptive statistics payload.
    plot_df : Optional[pd.DataFrame]
        Embedding/projection table used to infer projection availability.
    results_df : Optional[pd.DataFrame]
        Modeling results table.
    stats_config : Optional[Dict[str, Any]], default=None
        Descriptive-stats configuration for report display.
    embed_config : Optional[Dict[str, Any]], default=None
        Embedding/projection configuration for report display.
    model_config : Optional[Dict[str, Any]], default=None
        Modeling configuration for report display and interpretations.

    Returns
    -------
    Tuple[str, str]
        ``(html_report, md_report)`` versions of the same report content.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Prepare sections
    stats_overall = _pick_overall(overall_obj) if overall_obj else {}
    stats_rows = []
    if stats_overall:
        stats_rows = [
            ["Total characters", stats_overall.get("total_chars", "NA")],
            ["Total words", stats_overall.get("total_words", "NA")],
            ["Total sentences", stats_overall.get("total_sentences", "NA")],
            ["Average word length", round(float(stats_overall.get("avg_word_length", 0.0)), 3)],
            ["Average sentence length", round(float(stats_overall.get("avg_sentence_length", 0.0)), 3)],
            ["Total unique words", stats_overall.get("total_unique_words", "NA")],
            ["Lexical diversity", round(float(stats_overall.get("lexical_diversity", 0.0)), 3)],
        ]
    stats_table_df = pd.DataFrame(stats_rows, columns=["Metric", "Value"]) if stats_rows else None
    stats_interps = interpret_stats(stats_overall) if stats_overall else []
    stats_cfg_df = _config_to_df(stats_config)

    # Embeddings quick note
    emb_note = None
    if plot_df is not None and len(plot_df):
        dims = 3 if "dim_3" in plot_df.columns else 2
        emb_note = f"Projection computed with {dims} dimensions. Downloadable files are available from the Embeddings tab."
    embed_method = None
    if isinstance(embed_config, dict):
        embed_method = embed_config.get("reduce_method")
    if embed_method == "pca":
        emb_exclude = ["umap_metric", "tsne_metric"]
    elif embed_method == "umap":
        emb_exclude = ["tsne_metric"]
    elif embed_method == "tsne":
        emb_exclude = ["umap_metric"]
    else:
        emb_exclude = []
    emb_cfg_df = _config_to_df(embed_config, exclude_keys=emb_exclude)

    # Modeling table and interpretations
    model_table_df = None
    model_interps: List[Tuple[str, List[str], str]] = []
    if results_df is not None and len(results_df):
        show_cols = ["target", "task", "metric_name", "observed", "p_value", "cv_folds_used"]
        p_adjust_available = "p_adjusted" in results_df.columns and results_df["p_adjusted"].notna().any()
        p_fdr_available = "p_fdr" in results_df.columns and results_df["p_fdr"].notna().any()
        if p_adjust_available:
            show_cols.insert(5, "p_adjusted")
        elif p_fdr_available:
            show_cols.insert(5, "p_fdr")
        if "p_adjust_method" in results_df.columns and results_df["p_adjust_method"].notna().any():
            show_cols.insert(6, "p_adjust_method")
        extra_cols = [c for c in results_df.columns if c.startswith("metric_") and c != "metric_name"]
        show_cols.extend(extra_cols)
        show_cols = [c for c in show_cols if c in results_df.columns]
        model_table_df = results_df[show_cols].copy()
        model_table_df["observed"] = model_table_df["observed"].astype(float).round(3)

        if "p_value" in model_table_df:
            model_table_df["p_value"] = model_table_df["p_value"].astype(float).round(4)
        if "p_adjusted" in model_table_df:
            model_table_df["p_adjusted"] = model_table_df["p_adjusted"].astype(float).round(4)
        if "p_fdr" in model_table_df:
            model_table_df["p_fdr"] = model_table_df["p_fdr"].astype(float).round(4)
        include_adjusted_in_writeup = len(results_df) > 1
        for _, row in results_df.iterrows():
            tgt = str(row.get("target", "unknown"))
            bullets = interpret_model_row(row, model_config=model_config)
            summary_sentence = summarize_model_row(
                row,
                include_adjusted=include_adjusted_in_writeup,
                model_config=model_config,
            )
            model_interps.append((tgt, bullets, summary_sentence))
    model_cfg_df = _config_to_df(model_config)

    # HTML with light CSS
    css = """<style>
    .report { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; line-height: 1.5; color: #111; }
    .header { margin-bottom: 1rem; }
    .title { font-size: 1.75rem; font-weight: 700; margin: 0; }
    .subtitle { color: #555; margin: .25rem 0 0 0; }
    .card { background: #fff; border: 1px solid #eee; border-radius: 12px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .section-title { font-size: 1.2rem; font-weight: 700; margin: 0 0 8px 0; }
    .table { border-collapse: collapse; width: 100%; }
    .table th, .table td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #f0f0f0; }
    .table th { background: #fafafa; font-weight: 600; }
    .muted { color: #666; }
    ul { margin: .25rem 0 .5rem 1.2rem; }
    li { margin: .15rem 0; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f4f6f8; font-size: .875rem; margin-right: 6px; }
    </style>"""

    html_parts: List[str] = []
    html_parts.append('<div class="report">')
    html_parts.append('<div class="header">')
    html_parts.append(f'<p class="title">NLPsych session report</p>')
    html_parts.append(f'<p class="subtitle">Generated at {ts}</p>')
    html_parts.append('</div>')

    # Overview card
    html_parts.append('<div class="card">')
    html_parts.append('<div class="section-title">Overview</div>')
    html_parts.append(f'<p>Selected text columns</p>')
    html_parts.append("".join([f'<span class="pill">{c}</span>' for c in text_cols]) or '<p class="muted">None</p>')
    html_parts.append('</div>')

    # Stats card
    html_parts.append('<div class="card">')
    html_parts.append('<div class="section-title">Descriptive statistics</div>')
    if stats_table_df is None:
        html_parts.append('<p class="muted">No descriptive statistics were available. Run the Descriptive stats tab first.</p>')
    else:
        html_parts.append(_to_html_table(stats_table_df, index=False))
        if stats_interps:
            html_parts.append('<p><strong>Interpretation</strong></p>')
            html_parts.append('<ul>')
            for s in stats_interps:
                html_parts.append(f'<li>{s}</li>')
            html_parts.append('</ul>')
        if stats_cfg_df is not None:
            html_parts.append('<p><strong>Configuration</strong></p>')
            html_parts.append(_to_html_table(stats_cfg_df, index=False))
    html_parts.append('</div>')

    # Embeddings card
    html_parts.append('<div class="card">')
    html_parts.append('<div class="section-title">Embeddings and projection</div>')
    if emb_note is None:
        html_parts.append('<p class="muted">No embeddings or projection were available. Run the Embeddings tab first.</p>')
    else:
        html_parts.append(f'<p>{emb_note}</p>')
        if emb_cfg_df is not None:
            html_parts.append('<p><strong>Configuration</strong></p>')
            html_parts.append(_to_html_table(emb_cfg_df, index=False))
    html_parts.append('</div>')

    # Modeling card
    html_parts.append('<div class="card">')
    html_parts.append('<div class="section-title">Modeling results</div>')
    if model_table_df is None:
        html_parts.append('<p class="muted">No modeling results were available. Run the Modeling tab first.</p>')
    else:
        html_parts.append(_to_html_table(model_table_df, index=False))
        if model_cfg_df is not None:
            html_parts.append('<p><strong>Configuration</strong></p>')
            html_parts.append(_to_html_table(model_cfg_df, index=False))
        if model_interps:
            html_parts.append('<p><strong>Interpretation per target</strong></p>')
            html_parts.append('<ul>')
            for tgt, bullets, summary_sentence in model_interps:
                html_parts.append(f'<li><strong>{tgt}</strong>')
                if bullets:
                    html_parts.append('<ul>')
                    for b in bullets:
                        html_parts.append(f'<li>{b}</li>')
                    html_parts.append('</ul>')
                html_parts.append(f'<p><em>Example write-up sentence: {summary_sentence}</em></p>')
                html_parts.append('</li>')
            html_parts.append('</ul>')
    html_parts.append('</div>')

    html_parts.append('</div>')  # end report

    html_report = css + "\n" + "\n".join(html_parts) 

    # Markdown version for users who prefer plain text
    md_parts: List[str] = []
    md_parts.append("# NLPsych session report")
    md_parts.append(f"Generated at {ts}")
    md_parts.append(f"Selected text columns: {', '.join(text_cols)}\n")

    md_parts.append("## Descriptive statistics")
    if stats_table_df is None:
        md_parts.append("No descriptive statistics were available. Run the Descriptive stats tab first.")
    else:
        md_parts.append(stats_table_df.to_markdown(index=False))
        if stats_interps:
            md_parts.append("Interpretation")
            for s in stats_interps:
                md_parts.append(f"* {s}")
        if stats_cfg_df is not None:
            md_parts.append("Configuration")
            md_parts.append(_df_to_markdown_safe(stats_cfg_df, index=False))

    md_parts.append("\n## Embeddings and projection")
    if emb_note is None:
        md_parts.append("No embeddings or projection were available. Run the Embeddings tab first.")
    else:
        md_parts.append(emb_note)
        if emb_cfg_df is not None:
            md_parts.append("Configuration")
            md_parts.append(_df_to_markdown_safe(emb_cfg_df, index=False))
    md_parts.append("\n## Modeling results")
    if model_table_df is None:
        md_parts.append("No modeling results were available. Run the Modeling tab first.")
    else:
        md_parts.append(_df_to_markdown_safe(model_table_df, index=False))
        if model_cfg_df is not None:
            md_parts.append("Configuration")
            md_parts.append(_df_to_markdown_safe(model_cfg_df, index=False))
        md_parts.append("Interpretation per target")
        for tgt, bullets, summary_sentence in model_interps:
            md_parts.append(f"* {tgt}")
            for b in bullets:
                md_parts.append(f"  * {b}")
            md_parts.append(f"  * *Example write-up sentence: {summary_sentence}*")

    md_report = "\n\n".join(md_parts)
    return html_report, md_report
