# Changes from the uploaded notebooks

The uploaded notebooks remain available as attachments. The repository contains
rewritten versions with no saved local-machine paths or old numerical outputs.

| Topic | Revised approach | Why |
| --- | --- | --- |
| Missing shared configuration | Portable paths and an explicit replacement basket | Original notebook imports could not run alone |
| Global item exclusions in Notebook 03 | Preserve selected items through feature creation; screen inside each training fold | Full-2025 screening must not influence earlier backtests |
| Six suspected source-price errors | Retain valid positive observations; flag and export training evidence | Errors were unconfirmed; fixed positive-price retention avoids retrospective hand deletion |
| Outlier bounds | Training-only diagnostics, excluded from all model inputs and eligibility rules | A later bound cannot change earlier-fold predictions |
| Weekly aggregation | Preserve the primary median; add equal-premise and matched-premise diagnostics | Quantify monitoring/composition sensitivity |
| Sparse observations | Exact calendar grids; no price filling; documented coverage/history eligibility | A previous observed row need not be the previous week |
| Label definition | Fixed next-calendar-week rise ≥10% | Objective, reproducible classification target |
| Evaluation | Three expanding windows, then validation and final test | Demonstrate temporal generalisation with horizon protection |
| Benchmarks | No-surge, current-change rule, logistic regression, XGBoost, LightGBM | Evaluate incremental predictive value rather than declare integration novel |
| Decision cutoff | Maximum validation F1; never test-optimised | Keep test assessment independent of tuning |
| Uncertainty | Whole-week bootstrap and subgroup support tables | Avoid treating correlated district rows as independent samples |
| SHAP | Offline capped cache, feature-level contribution grouping and additivity check | Concrete explanation artifacts without per-click computation |
| Isolation Forest | Separate training-fitted detector with a fixed 1% review-budget setting | Unsupervised anomaly detection is distinct from surge classification |

## What may be claimed in the dissertation

Objective 1 can now refer to constructing and validating a labelled local weekly
monitoring dataset with an explicit 10% threshold and documented coverage rules.
Objective 2 can compare predictive models with expanding-window AP and independent
holdout precision, recall, and F1. Objective 3 can provide associational SHAP
explanations and anomaly review outputs; deployment/usability still needs evaluation.

AP here means scikit-learn's `average_precision_score`, not trapezoidal PR-AUC.
Report that convention explicitly. A constant-score baseline's AP equals event
prevalence when positive and negative cases are present. No-positive groups have
undefined AP; their support counts matter more than an apparent perfect accuracy.

The F1 and AP-lift targets are proposed, not derived from a stakeholder study.
Explain why those targets are useful, and report unmet criteria instead of adjusting
the target after seeing results. The fixed model comparison is not exhaustive and
does not establish superiority over all existing food forecasting systems.

## Population and time interpretation

Each row identifies an item, state, district, and premise type. Current-week
statistics are available only after that week's records have been collected.
Historical evaluation requires valid observations in both the current and target
weeks; future target coverage is used solely to determine whether a historical
outcome is measurable. Live forecast eligibility uses only current coverage and
past history. Report the retrospective filtering and possible selection bias.

Holdout predictions may use earlier observed holdout prices as lagged inputs: this
simulates successive one-week forecasts, not a simultaneous three-month forecast.
Labels and feature preprocessing are never updated from holdout outcomes.

The weekly median is not a consumption-weighted CPI, a household expenditure
index, or a fixed-premise index. Matching premises can reduce composition effects
but introduces its own coverage selection. No aggregation alternative is ground
truth. Do not select an alternative because it produces a better final test score.

Eighteen months of observations do not establish annual seasonal robustness,
causal effects, or affordability benefits. An actionable-warning study would need
named stakeholders, feasible responses, false-alert tolerance, and evidence that
warnings influence useful decisions. State those as limitations or future work.
