# Real-data evaluation snapshot

These are real PriceCatcher results from this prepared environment, not synthetic
fixture scores. They use the documented **replacement basket**, not a verified
reproduction of the missing original shared configuration.

## Data and population

- Source: official HTTPS PriceCatcher monthly Parquets, January 2025–June 2026.
- Integrated observations: 5,581,612; 515 distinct observation dates.
- Basket: 27 explicit item codes; 16 states/federal territories and 169 state–district pairs.
- Invalid prices and duplicate daily observation keys: zero in the integrated data.
- The target is next week's observed local median increase of at least 10%.
- Final supported test population: 62,516 item–district–premise-type weekly rows,
  12 prediction weeks, 9,009 positive events (14.41%). Unsupported item codes are
  excluded using training-only screening; this is not universal district coverage.

## Frozen evaluation

Three expanding-window folds selected **XGBoost** by mean AP, before final test
inspection. Validation selected its score cutoff (0.611645). All other
candidate models and baselines were assessed under the same temporal protocol.

| Model | Selected by backtests | AP | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| LogisticRegression | No | 0.299 | 0.330 | 0.354 | 0.342 |
| XGBoost | Yes | 0.329 | 0.336 | 0.413 | 0.371 |
| LightGBM | No | 0.326 | 0.331 | 0.444 | 0.379 |
| AlwaysNoSurge | No | 0.144 | 0.000 | 0.000 | 0.000 |
| CurrentChangeRule | No | 0.161 | 0.219 | 0.221 | 0.220 |

The selected model's AP is 2.28 times test prevalence. Its F1 **does not meet**
the provisional 0.50 target; the AP-lift criterion of 1.5 is met. No later tuning
was performed to improve these final test numbers. LightGBM's higher test F1
does not change the model selected using backtests.

Whole-week bootstrap 95% intervals for the selected model are approximately
AP 0.294–0.374 and F1 0.351–0.386. These describe uncertainty within this short
holdout, not generalisation to independent future years.

## Important limitations revealed by the run

- Cooking oil and sugar have **zero positive test events**. Their AP is undefined;
  this is not evidence of successful surge detection.
- Chicken recall is approximately 0.006 at the global validation-selected cutoff.
  The overall model therefore does not establish useful warning performance for
  every food group. Any future group-specific threshold design needs separate
  development data and a fresh final assessment.
- Training label disagreement is about 4.95% for equal-premise medians and 5.08%
  for matched-premise comparisons. Differences in monitoring composition remain
  an important uncertainty in interpreting observed-median surges.
- The final model is trained through December 2025. It is an evaluation artifact,
  not a continually retrained deployment or a calibrated probability service.

## Explanation and anomaly artifacts

The run saved 5,708 current-week forecasts for 2026-06-22, with SHAP explanations
precomputed for the top 200 scores. Explanation additivity was checked
against model log-odds; loading the saved pipeline reproduced its predictions.
Rows without a cached explanation are marked explicitly. Isolation Forest was
trained on training features and saved separately; its accuracy cannot be
established without independent anomaly labels.

## Reproduction

See `outputs/01/source_manifest.json` for source and lookup SHA-256 checksums,
`outputs/01/basket.csv` for selected items, and `outputs/06/model_selection.json`
for the feature-file fingerprint and frozen settings. Full executed notebooks
are in `outputs/executed_notebooks/`. Large outputs are ignored by Git; this
small report records the observed results without embedding raw data or models.
