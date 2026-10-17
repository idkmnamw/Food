# Local food-price surge prediction in Malaysia

Six readable notebooks prepare PriceCatcher data and evaluate next-week warnings.
The target is a **10% increase in the observed local weekly median**, not proof of
within-shop inflation, consumer benefit, or causal effects.

## Project layout

```text
notebooks/     01 integration → 02 quality → 03 cleaning → 04 weekly aggregation
               → 05 features/labels → 06 modelling/evaluation
src/           shared settings and small data/model helpers
scripts/       reproducible notebook execution and synthetic validation
tests/         checks for time leakage, missing weeks, units, and model inputs
docs/          methodological changes and validation evidence
data/          cached source files and frozen lookups (ignored by Git)
outputs/       tables, models, plots, executed notebooks, SHAP cache (ignored)
```

## Install and run

Use Python 3.12 (the tested version). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_notebooks.py
```

Alternatively open `notebooks/` in VS Code or Jupyter with the `.venv` interpreter
and run each notebook in numerical order. The runner saves executed copies under
`outputs/executed_notebooks/`, leaving source notebooks free of stale results.
The first run downloads 18 monthly files; later runs reuse source snapshots.
Only `storage.data.gov.my` is needed for data, alongside the Python package registry.

**Basket:** the uploaded notebooks' `pricecatcher_config.py` was missing. The
replacement explicitly lists 27 official item codes in the shared configuration;
it is not guaranteed to reproduce the original basket. See the readable
[basket reference](docs/BASKET.md) and inspect `outputs/01/basket.csv`.
To restore an exact basket, provide root-level `basket.csv` with columns
`item_code,project_group` **before the first run**. Use new data/output directories
when changing basket/source snapshots; never reuse stale frozen metadata.

All dates, coverage rules, model features, fold boundaries, and proposed performance
criteria live in `src/pricecatcher_config.py`. Do not change these after inspecting
final test scores to improve the reported result. An altered research design needs
a new untouched holdout for confirmatory claims.

Environment variables support portable data locations and an optional local mirror:

| Variable | Purpose |
| --- | --- |
| `PRICECATCHER_DATA_DIR` | Cached sources and lookup snapshots; default `data/` |
| `PRICECATCHER_OUTPUT_DIR` | Generated research artifacts; default `outputs/` |
| `PRICECATCHER_SOURCE_DIR` | Optional mirror of official Parquet filenames |

Keep synthetic mirrors separate from official sources. Large data and executable
model artifacts are not committed. Only load joblib artifacts you trust.

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/validate_notebooks.py --work-dir /tmp/food-validation-new
```

The synthetic command requires a new empty directory. It executes all six notebooks
on deterministic fixture data, including gaps, invalid records, and a group with no
surge events. **Its scores must never be used as FYP findings.** See
[validation evidence](docs/VALIDATION.md) for the actual checks performed.
The recorded real-data results are summarised in [RESULTS.md](docs/RESULTS.md).

## Evaluation design and outputs

Notebook 06 compares logistic regression, XGBoost, and LightGBM using three
expanding-window folds within 2025. Each fold relearns item screening, imputation,
encoding, and class weighting from its training observations. Highest mean
average precision selects the model. Validation (January–March 2026) selects its
F1 alert cutoff; April–June 2026 is the final holdout. Boundary-crossing targets
are excluded. Fixed settings avoid a large opaque hyperparameter search.

Important outputs under `outputs/06/`:

- `cv_leaderboard.csv`, `validation_results.csv`, `test_results.csv`:
  comparative results, including no-surge and current-change baselines.
- `test_metrics_by_food_group.csv` and district/state tables: support counts and
  disaggregated performance. AP is undefined for groups without positive cases.
- `test_week_bootstrap_intervals.csv`: uncertainty from resampling prediction weeks.
- `selected_pipeline.joblib` and `model_selection.json`: preprocessing, model,
  supported items, frozen alert threshold, and data fingerprint.
- `latest_forecasts.parquet` and `shap_cache.json`: cached predictions and up to
  200 offline explanations; uncomputed explanations are explicitly marked.
- `isolation_forest.joblib` and `test_anomaly_review.parquet`: separate unsupervised
  anomaly evidence, not supervised surge labels or validated anomaly accuracy.

The proposed F1 ≥ 0.50 and AP/prevalence ≥ 1.5 criteria are **provisional** and
must be justified against warning costs in the report. The code records whether
they were met; it does not guarantee success.

## Research limits

Monitoring composition can change the observed median without comparable
within-premise increases. Notebook 05 compares observation-weighted,
equal-premise, and matched-premise outcomes. Sparse district records are filtered,
not filled with invented prices; excluded areas must be shown as insufficient
coverage. Model numeric imputation is fitted on training data only.

Only 18 months are studied. Few events in stable foods, short test duration,
changing metadata, possible source errors, and publication delays limit conclusions.
Weather, exchange rates, fuel costs, imports, and interventions are not modelled.
SHAP is associational; class-weighted scores are not established calibrated risks.

A dashboard and stakeholder evaluation remain separate work. Offline explanation
caching does not establish UI latency or prove that warnings improve affordability.
See [methodological changes](docs/METHODOLOGY.md) for the rationale and report guidance.
