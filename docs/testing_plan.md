# Testing Plan

## Current State

The repository has working pytest coverage for:

- package import smoke tests
- spaCy pipeline fallback behavior
- descriptive summary table exports
- selected modeling task-control behavior
- topic-modeling helpers and topic-state storage

The repository does not yet have broad automated coverage for:

- embedding generation and reduction
- report generation/export paths
- most of the Streamlit app flow
- app launch behavior

## Local Workflow

Install contributor tooling in the active environment:

```bash
pip install -e ".[dev]"
```

Run the full test suite:

```bash
python -m pytest -q
```

Generate a coverage report:

```bash
python -m pytest --cov --cov-report=term-missing
```

## Coverage Priorities

### 1. Embedding Pipeline

File: `src/nlpsych/embedding.py`

Highest-value cases:

- embedding generation returns aligned metadata, text, and matrix outputs
- empty or all-missing text inputs fail clearly
- dimensionality reduction paths behave deterministically with fixed seeds
- cached/local model loading errors surface clearly

### 2. Report Generation

File: `src/nlpsych/report.py`

Highest-value cases:

- report sections render when optional inputs are missing
- markdown/HTML/text export helpers preserve expected headings and tables
- topic/modeling/descriptive sections compose without crashing

### 3. Modeling Core

File: `src/nlpsych/modeling.py`

Current tests cover only a small part of the module. Expand into:

- automatic task detection edge cases
- regression vs classification metric selection
- invalid CV/task configurations
- permutation output structure and shape guarantees
- multiple-comparison correction edge cases beyond the current happy path

### 4. Descriptive Statistics

File: `src/nlpsych/descriptive_stats.py`

Current tests emphasize summary-table formatting. Add direct coverage for:

- raw text statistics computation
- multi-column aggregation
- missing/blank text handling
- token/POS count behavior under the blank spaCy fallback pipeline

### 5. Streamlit App Logic

Files:

- `src/nlpsych_app/app.py`
- `src/nlpsych_app/launch.py`
- `src/nlpsych_app/pages/Docs.py`

Do not start with browser automation. Start with importable helpers and pure logic:

- file-loading helpers
- state reset/storage helpers
- plot option routing
- launch entry-point smoke tests

## Suggested Sequence

1. Run the new coverage baseline and save the report output.
2. Add tests for `embedding.py`.
3. Add tests for `report.py`.
4. Expand `modeling.py` and `descriptive_stats.py`.
5. Add lightweight smoke tests for app/launch logic.
6. After that, wire coverage into CI and consider a minimum threshold.
