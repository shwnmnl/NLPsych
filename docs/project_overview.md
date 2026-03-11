# NLPsych Documentation

> Natural Language Psychometrics (NLPsych) is a hybrid Python library + Streamlit app for turning free-text datasets into descriptive statistics, semantic embeddings, lightweight predictive models, and shareable reports.

## Table of Contents
1. [Overview](#overview)
2. [Orientation: Using Text Embeddings for Statistical Analysis](#orientation-using-text-embeddings-for-statistical-analysis)
3. [Feature Highlights](#feature-highlights)
4. [Architecture](#architecture)
5. [Installation](#installation)
6. [Quickstart (Library)](#quickstart-library)
7. [Streamlit App](#streamlit-app)
8. [Module Reference](#module-reference)
9. [Data Flow: Text → Report](#data-flow-text-report)
10. [Testing & Quality](#testing--quality)
11. [Support & Contributing](#support--contributing)

## Overview

NLPsych bundles spaCy-driven descriptive statistics, SentenceTransformer embeddings, classical scikit-learn models, and a report generator behind a single API. A packaged Streamlit UI (`nlpsych_app`) exposes the same primitives to non-technical users.

- **Source**: `src/nlpsych` (library) and `src/nlpsych_app` (Streamlit front-end)
- **Entry points**: Import `nlpsych` in Python, launch installed UI with `nlpsych-app`, or run source UI with `streamlit run streamlit_app.py`
- **Use cases**: Rapid exploratory text analytics, psychometric prototyping, teaching demos, or lightweight NLP quality assurance

## Orientation: Using Text Embeddings for Statistical Analysis

This walkthrough is for non-coders and non-ML users. NLPsych allows you to convert text into numerical representations and use those representations in statistical or machine learning models. Before doing so, it is important to understand what these methods are doing and how to use them in a scientifically responsible way.

### 1. What a text embedding is

A text embedding is a numerical vector that represents the meaning of a piece of text.

The embedding is learned from large text corpora and captures statistical patterns in how words and phrases are used. Texts with similar meanings tend to be mapped to nearby points in a high dimensional space.

Important implications:

* Embeddings reflect usage patterns in language
* They approximate semantic similarity
* They are learned rather than rule based
* They do not encode logical truth or causal structure

Similarity between embedded texts means that they tend to appear in similar contexts. It does not imply that they are scientifically related in a causal or mechanistic sense.

### 2. What Embeddings Preserve and What They Lose

Embeddings compress textual information into a fixed length vector.

They tend to preserve:

* General semantic meaning
* Topical similarity
* Broad contextual relationships

They may lose:

* Syntax
* Negation
* Rare terminology
* Subtle distinctions in phrasing

Small wording changes can sometimes produce large changes in representation. Domain specific terminology may also be poorly represented if it is uncommon in general language corpora.

### 3. Embeddings as Predictors

Once generated, embeddings can be treated as predictors in a statistical model.

Examples include:

* Document classification
* Predicting outcomes from written reports
* Clustering texts into groups
* Testing whether language use differs across conditions

Embedding vectors typically contain hundreds or thousands of numerical features. This often creates a setting where the number of predictors is large relative to the number of observations.

In such cases:

* Overfitting becomes a serious risk
* Regularization may be required
* Dimensionality reduction may be helpful
* Validation becomes essential

### 4. Cross Validation

Cross validation estimates how well a model is expected to perform on new data.

A typical procedure:

1. Split the data into several folds
2. Train the model on some folds
3. Evaluate it on the remaining fold
4. Repeat across folds

This produces a distribution of performance values rather than a single estimate.

Performance reported from a single train test split can be unstable when the number of predictors is large. Cross validation provides a more reliable estimate of expected generalization performance.

### 5. Hyperparameter Tuning

Most machine learning models contain settings that influence their behavior. Examples include:

* Regularization strength
* Number of retained components after dimensionality reduction
* Model complexity
* Number of clusters

Choosing these settings based on the same data used to estimate performance can produce optimistic results.

A safer approach uses nested cross validation:

Outer loop
Estimates performance

Inner loop
Selects model settings

Final reported performance comes from the outer loop only.

### 6. Permutation Testing

Permutation testing provides a simple nonparametric way to test whether text contains predictive information about an outcome.

Rather than relying on distributional assumptions, permutation tests evaluate whether the observed relationship between predictors and outcome is stronger than would be expected by chance.

A typical procedure:

Fit a model using the true outcome values

Measure predictive performance using cross validation

Randomly shuffle the outcome values

Refit the same model using the shuffled outcomes

Recompute predictive performance

Repeat many times

This produces a null distribution representing performance expected if there were no real relationship between predictors and outcome.

The observed performance can then be compared to this distribution to obtain a p value for predictive signal.

## Feature Highlights

- **Descriptive statistics**: Token/character counts, lexical diversity, POS breakdowns, and customizable averaging modes powered by spaCy.
- **Embeddings**: Plug-and-play SentenceTransformer encoders (MiniLM, MPNet, multilingual variants, or custom names) plus PCA/UMAP/t-SNE reduction and Plotly visualizations.
- **Modeling**: Auto-detection plus explicit task overrides (`task_mode`, `target_task_overrides`), CV/permutation testing, and Benjamini–Hochberg FDR correction via `auto_cv_with_permutation`.
- **Reports**: `build_report_payload` assembles HTML + Markdown summaries with per-target interpretations and example write-up sentences.
- **App workflow**: Upload CSV/TSV/XLSX (or use demo data), pick text columns, compute stats, explore embeddings, run models, and export reports without touching code.

## Architecture

```
streamlit_app.py      # Source/dev multipage router (Main + Docs)
src/
├── nlpsych/          # Core analytics library (importable)
│   ├── descriptive_stats.py
│   ├── embedding.py
│   ├── modeling.py
│   ├── report.py
│   └── utils.py
├── nlpsych_app/      # Streamlit UI built on top of the library
│   ├── app.py
│   ├── launch.py     # Exposed as the `nlpsych-app` console script
│   └── pages/
│       └── Docs.py
└── tests/            # Pytest suite for critical utilities
```

`README.md` provides a lightweight intro, while this document dives deeper into internals and workflows.

## Installation

### Library only

```bash
pip install nlpsych
```

This installs the `nlpsych` package with dependencies such as pandas, spaCy 3.8, scikit-learn, statsmodels, SentenceTransformers, UMAP, Plotly, and Tabulate.
Use lowercase in commands and imports (`nlpsych`, `nlpsych_app`).

### Streamlit app

```bash
pip install "nlpsych[app]"
nlpsych-app
```

The `[app]` extra adds Streamlit and wires up the `nlpsych-app` console entry point (`src/nlpsych_app/launch.py`). The launcher ensures Streamlit exists, points to the packaged `.streamlit` config directory, and runs `streamlit run` on the packaged `nlpsych_app/app.py`.

### Local development

```bash
git clone https://github.com/shwnmnl/nlpsych.git
cd nlpsych
pip install -e ".[app,dev]"
```

- `.[app,dev]` gives you the UI plus formatting/testing tools (`ruff`, `black`, `pytest`).
- Run the app from source with `streamlit run streamlit_app.py`.
- For Streamlit Community Cloud, set the main file path to `streamlit_app.py`.

## Quickstart (Library)

```python
import pandas as pd
from nlpsych.descriptive_stats import descriptive_stats
from nlpsych.embedding import embed_text_columns_simple, reduce_embeddings, build_plot_df
from nlpsych.modeling import auto_cv_with_permutation
from nlpsych.report import build_report_payload

df = pd.DataFrame({"note": [
    "Patient denies chest pain. Vitals stable.",
    "Lexical diversity matters for stylistic profiling."
]})

# 1. Descriptive statistics
stats_df, overall = descriptive_stats(df["note"], split_overall="both")

# 2. Embeddings + projection
meta_df, emb, texts = embed_text_columns_simple([df["note"]])
Z = reduce_embeddings(emb, method="pca", n_components=2)
plot_df = build_plot_df(Z, meta_df, texts)

# 3. Modeling (if you have targets)
X = meta_df.join(pd.DataFrame(emb)).groupby("index").mean()
Y = pd.DataFrame({"target": [0, 1]})
results_df, preds = auto_cv_with_permutation(X, Y)

# 4. Report
html_report, md_report = build_report_payload(
    text_cols=["note"],
    stats_df=stats_df,
    overall_obj=overall,
    plot_df=plot_df,
    results_df=results_df
)
```

## Streamlit App

`streamlit_app.py` defines source/dev multipage routing, and `src/nlpsych_app/app.py` layers an opinionated UX on top of the primitives:

1. **Upload data**: CSV/TSV/XLSX or toggle demo data seeded with two text columns and a binary target.
2. **Tabs workflow**:
   - Overview: dataset summary and selected text columns.
   - Descriptive stats: toggles for alphabetic filtering, stopword removal, lemma-based vocab, and averaging mode; caches spaCy pipeline.
   - Embeddings: SentenceTransformer selector (preset list + custom option), choice of PCA/UMAP/t-SNE, and 2D/3D plotting with Plotly.
   - Modeling: pick non-text targets, set CV/permutation controls, choose task mode (auto/classification/regression), optionally override task per target, reuse cached embeddings, and view permutation histograms.
   - Report: combine previous outputs into HTML or Markdown, ready for download.
3. **Downloads**: CSVs for per-row stats, projection coordinates, embeddings (with metadata), modeling tables, plus `.npy` arrays and HTML/Markdown reports.

State is cached via `st.session_state` and Streamlit caching decorators to decouple heavy computations from UI interactions.

## Module Reference

### `nlpsych.descriptive_stats`

| Function | Purpose |
| --- | --- |
| `descriptive_stats(*series)` | Iterates over pandas Series, processes text with a cached spaCy pipeline, and returns per-row stats plus aggregated overall metrics. Options cover lemma vs. surface form vocabularies, alphabetic filtering, stopword removal, and sentence averaging mode (`unweighted` vs `ratio`). |

Outputs:
- `stats_df`: MultiIndex (`source_column`, `index`) with char/word counts, POS tallies, and lexical diversity.
- `overall`: Combined or per-column dictionaries formed by the helper `_pick_overall`.

### `nlpsych.embedding`

| Function | Purpose |
| --- | --- |
| `embed_text_columns_simple(series_list, model_name, normalize)` | Collects non-empty rows, encodes them with SentenceTransformers via `get_st_model_base`, and returns (`meta_df`, `embeddings`, `texts`). |
| `reduce_embeddings(embeddings, method, n_components, metric)` | Wrapper over PCA/UMAP/t-SNE with sane defaults (e.g., auto-adjusted t-SNE perplexity, UMAP fallback). |
| `build_plot_df(Z, meta, texts)` | Combines reduced coordinates with metadata for plotting/export. |
| `plot_projection(plot_df, n_components, color_by, point_size)` | Generates Plotly 2D or 3D scatter plots for Streamlit and notebooks. |

### `nlpsych.modeling`

`auto_cv_with_permutation` orchestrates the full modeling loop:

1. Resolves task per target using `task_mode` plus optional `target_task_overrides`; in `auto`, `_detect_task` treats categorical/low-cardinality numeric targets as classification.
2. Builds CV iterators (StratifiedKFold vs. KFold, group-aware variants when groups are provided) with safeguards for tiny datasets and sparse classes.
3. Fits classification/regression pipelines (default LogisticRegression/Ridge, with optional scaling, feature selection, PCA, and hyperparameter search).
4. Computes chosen metrics, stores per-fold predictions, and runs permutation tests.
5. Applies multiple-testing correction (default `fdr_bh`) and packages results as a tidy DataFrame + prediction dict.

Key helpers: `_fit_and_score_cv`, `_perm_test`, and the `TargetResult` dataclass.

### `nlpsych.report`

- `build_report_payload(text_cols, stats_df, overall_obj, plot_df, results_df)` returns parallel HTML/Markdown reports.
- Includes interpretation helpers (`interpret_stats`, `interpret_model_row`, `summarize_model_row`) that translate outputs into narrative bullets plus an example write-up sentence.
- HTML output ships with lightweight inline CSS for cards/tables; Markdown output favors `pandas.DataFrame.to_markdown`.

### `nlpsych.utils`

| Helper | Description |
| --- | --- |
| `get_spacy_pipeline_base(allow_download=True)` | Loads `en_core_web_sm` without NER/parser, guaranteeing a sentencizer. If missing, optionally downloads; otherwise falls back to a blank English pipeline. Cached via `functools.lru_cache`. |
| `get_st_model_base(model_name, allow_download=True)` | Thin wrapper around `SentenceTransformer(model_name)` with error messaging when downloads are disallowed. |

### `nlpsych_app`

- `app.py`: Streamlit layout (tabbed workflow, caching, download helpers, demo data, Plotly charts).
- `launch.py`: Console entry point that checks for Streamlit, points `STREAMLIT_CONFIG_DIR` at packaged defaults, and shells out to `streamlit run` on `src/nlpsych_app/app.py`.

## Data Flow: Text → Report

1. **Ingest**: CSV columns are selected as `series_list`.
2. **Descriptive stats**: `descriptive_stats` yields per-row + overall aggregates; cached in `st.session_state`.
3. **Embeddings**: SentenceTransformer encodes rows → `reduce_embeddings` condenses dimensions → Plotly visualizes `build_plot_df`.
4. **Modeling**: Embeddings are averaged per original row, joined with numeric targets, and passed to `auto_cv_with_permutation`.
5. **Report assembly**: `build_report_payload` stitches together whichever components the user ran (stats, embeddings note, modeling table) and emits HTML/Markdown for download or sharing.

This modular flow allows you to reuse outputs elsewhere (e.g., feed embeddings into downstream ML, export reports to collaborators, or call functions directly inside notebooks).

## Testing & Quality

- **Unit tests** (`tests/`):
  - `test_import.py`: guards package importability and exposed `__version__`.
  - `test_spacy_utils.py`: verifies `get_spacy_pipeline_base` returns a spaCy `Language` with a sentencizer even when downloads are disabled.
  - `test_modeling_task_control.py`: verifies forced task-mode behavior (`task_mode`, `target_task_overrides`) and small-class CV guardrails.
- **Suggested commands**:

```bash
pytest
```

Add `-k module_name -vv` for focused runs. Future coverage could extend further into embedding/report rendering smoke tests with small fixtures.

## Support & Contributing

- Issues & feature requests: [GitHub Issues](https://github.com/shwnmnl/nlpsych/issues)
- Pull requests: welcome! Please describe the feature/fix, add tests when practical, and keep documentation updated (including this page).
- Questions or collaboration ideas: open an issue or reach out to the maintainer listed in `pyproject.toml`.

Happy analyzing!
