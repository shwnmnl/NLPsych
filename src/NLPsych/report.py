from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Tuple 

import pandas as pd
import numpy as np

def _pick_overall(overall_obj):
    if isinstance(overall_obj, dict) and "combined" in overall_obj:
        return overall_obj["combined"]
    return overall_obj or {}

def interpret_stats(overall: dict) -> list[str]:
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

def interpret_model_row(row: pd.Series) -> list[str]:
    out: List[str] = []
    metric = str(row.get("metric_name", "")).upper()
    score = float(row.get("observed", 0.0))
    p = float(row.get("p_value", 1.0))
    p_fdr = float(row.get("p_fdr", 1.0))
    folds = int(row.get("cv_folds_used", 0)) if pd.notnull(row.get("cv_folds_used", np.nan)) else None
    task = row.get("task", "classification")

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

    if p_fdr < 0.05:
        out.append("Permutation test with FDR correction indicates a statistically reliable effect.")
    elif p < 0.05:
        out.append("Permutation test is nominally significant but not after FDR correction.")
    else:
        out.append("Permutation test does not indicate statistical significance.")

    if folds is not None and folds > 0:
        out.append(f"Cross validation used {folds} folds.")
    return out

def _to_html_table(df: pd.DataFrame, index: bool = False) -> str:
    return df.to_html(index=index, classes="table", border=0, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x)

def _df_to_markdown_safe(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)

def build_report_payload(
    text_cols: list[str],
    stats_df: Optional[pd.DataFrame],
    overall_obj: Optional[dict],
    plot_df: Optional[pd.DataFrame],
    results_df: Optional[pd.DataFrame]
) -> Tuple[str, str]:
    """Returns HTML and Markdown versions of the same report."""
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

    # Embeddings quick note
    emb_note = None
    if plot_df is not None and len(plot_df):
        dims = 3 if "dim_3" in plot_df.columns else 2
        emb_note = f"Projection computed with {dims} dimensions. Downloadable files are available from the Embeddings tab."

    # Modeling table and interpretations
    model_table_df = None
    model_interps: List[Tuple[str, List[str]]] = []
    if results_df is not None and len(results_df):
        show_cols = ["target", "task", "metric_name", "observed", "p_value", "p_fdr", "cv_folds_used"]
        show_cols = [c for c in show_cols if c in results_df.columns]
        model_table_df = results_df[show_cols].copy()
        model_table_df["observed"] = model_table_df["observed"].astype(float).round(3)

        if "p_value" in model_table_df:
            model_table_df["p_value"] = model_table_df["p_value"].astype(float).round(4)
        if "p_fdr" in model_table_df:
            model_table_df["p_fdr"] = model_table_df["p_fdr"].astype(float).round(4)
        for _, row in results_df.iterrows():
            tgt = str(row.get("target", "unknown"))
            bullets = interpret_model_row(row)
            model_interps.append((tgt, bullets))

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
    html_parts.append('</div>')

    # Embeddings card
    html_parts.append('<div class="card">')
    html_parts.append('<div class="section-title">Embeddings and projection</div>')
    if emb_note is None:
        html_parts.append('<p class="muted">No embeddings or projection were available. Run the Embeddings tab first.</p>')
    else:
        html_parts.append(f'<p>{emb_note}</p>')
    html_parts.append('</div>')

    # Modeling card
    html_parts.append('<div class="card">')
    html_parts.append('<div class="section-title">Modeling results</div>')
    if model_table_df is None:
        html_parts.append('<p class="muted">No modeling results were available. Run the Modeling tab first.</p>')
    else:
        html_parts.append(_to_html_table(model_table_df, index=False))
        if model_interps:
            html_parts.append('<p><strong>Interpretation per target</strong></p>')
            html_parts.append('<ul>')
            for tgt, bullets in model_interps:
                html_parts.append(f'<li><strong>{tgt}</strong>')
                if bullets:
                    html_parts.append('<ul>')
                    for b in bullets:
                        html_parts.append(f'<li>{b}</li>')
                    html_parts.append('</ul>')
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

    md_parts.append("\n## Embeddings and projection")
    if emb_note is None:
        md_parts.append("No embeddings or projection were available. Run the Embeddings tab first.")
    else:
        md_parts.append(emb_note)
    md_parts.append("\n## Modeling results")
    if model_table_df is None:
        md_parts.append("No modeling results were available. Run the Modeling tab first.")
    else:
        md_parts.append(_df_to_markdown_safe(model_table_df, index=False))
        md_parts.append("Interpretation per target")
        for tgt, bullets in model_interps:
            md_parts.append(f"* {tgt}")
            for b in bullets:
                md_parts.append(f"  * {b}")

    md_report = "\n\n".join(md_parts)
    return html_report, md_report

