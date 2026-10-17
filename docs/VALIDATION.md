# Validation evidence

The repository was checked in this Linux cloud environment with Python 3.12.
Pinned package versions are in `requirements.txt`.

## Executed checks

| Check | Outcome |
| --- | --- |
| All six notebook schemas and Python code cells | Valid; source notebooks contain no stale outputs |
| Ten targeted automated tests | Passed |
| All six notebooks on a separate deterministic synthetic fixture | Completed successfully |
| All six notebooks on official PriceCatcher data | Completed successfully |
| Cached repeat of Notebook 01 | Completed; frozen basket and source reuse accepted |
| Dependency consistency (`pip check`) | Passed |
| Saved selected pipeline reload | Predictions matched the in-memory model |
| SHAP log-odds reconstruction | Passed for the selected real-data XGBoost model |
| Separate tree SHAP regression checks | Passed for XGBoost and LightGBM |
| Offline anomaly detector and explanation cache | Saved successfully |

Tests cover exact 10% boundary rounding, missing calendar weeks, invariance of
earlier features to future prices, fold-specific item screening, protected target
horizons, egg units, composition effects, undefined AP, validation threshold
selection, missing inputs, unseen categories, and tree explanation additivity.

## Where to inspect evidence

- Real executed notebooks: `outputs/executed_notebooks/`.
- Real numerical results and limitations: [RESULTS.md](RESULTS.md).
- Real machine-readable model results: `outputs/06/test_results.csv`.
- Synthetic executed notebooks and results from this check:
  `/tmp/food-validation-isolated/` (temporary, not committed and not academic evidence).

The earlier original notebooks were not executable alone because their shared
configuration and datasets were not attached. The replacement pipeline downloaded
official source files after `storage.data.gov.my` access became available. Its
explicit basket is documented; equivalence to the original missing basket is not
claimed.

The source notebook outputs are deliberately cleared. Executed copies hold the
actual tables, plots, execution counts, and errors if a later run fails. Some
nonfatal notebook runtime/progress-widget/cache warnings occurred in this restricted
cloud account; the reported notebook runs completed without code-cell errors.
The saved install instructions were also exercised in an isolated virtual
environment; dependency checks, all ten tests, and all six synthetic notebook runs
passed there. The real-data pipeline artifact loaded and reproduced 100 saved
predictions with the expected feature-file checksum.

## What passing does not establish

These checks do not guarantee an absence of every possible bug, prove that the
framework is optimal, or establish future model accuracy. The real-data F1 target
was not met. No dashboard usability, serving latency, independent anomaly labels,
external driver contribution, or affordability impact was evaluated. The tests
establish a working, reproducible research pipeline for the documented scope.
