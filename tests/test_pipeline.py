"""Targeted checks for the research assumptions most likely to corrupt results."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pricecatcher_config as cfg
from data_utils import aggregate_weeks, make_features, item_screen, normalise_units, time_masks, surge_event
from model_utils import metrics, choose_threshold, make_model, model_inputs


def daily_example():
    rows = []
    for week in pd.date_range("2025-01-06", periods=12, freq="W-MON"):
        for premise in [1, 2]:
            for day in [0, 2]:
                rows.append({"date": week + pd.Timedelta(days=day), "item_code": 1,
                    "state": "Test", "district": "Test", "premise_type": "Shop",
                    "premise_code": premise, "price": 10.0, "item": "TEST",
                    "unit": "1kg", "project_group": "Chicken", "is_suspicious_price": False})
    return pd.DataFrame(rows)


class PipelineTests(unittest.TestCase):
    def test_exact_ten_percent_increase_is_a_surge(self):
        changes = pd.Series([(3.3 / 3 - 1) * 100, 9.999, 10.001, np.nan])
        self.assertEqual(surge_event(changes).tolist(), [True, False, True, False])

    def test_missing_calendar_week_is_not_previous_observation(self):
        weekly = aggregate_weeks(daily_example())
        missing_week = pd.Timestamp("2025-02-03")
        weekly = weekly.loc[weekly["week_start"].ne(missing_week)]
        frame = make_features(weekly).set_index("week_start")
        self.assertTrue(pd.isna(frame.loc["2025-02-10", "lag_1w_price"]))
        self.assertTrue(pd.isna(frame.loc["2025-01-27", "surge_label"]))

    def test_future_price_cannot_change_earlier_features(self):
        weekly = aggregate_weeks(daily_example())
        original = make_features(weekly)
        changed = weekly.copy()
        changed.loc[changed["week_start"].ge("2025-02-17"), "weekly_median_price"] *= 10
        updated = make_features(changed)
        before = original["week_start"].lt("2025-02-17")
        pd.testing.assert_frame_equal(original.loc[before, cfg.FEATURES], updated.loc[before, cfg.FEATURES])
        self.assertNotEqual(original.loc[original["week_start"].eq("2025-02-10"), "surge_label"].iloc[0],
                            updated.loc[updated["week_start"].eq("2025-02-10"), "surge_label"].iloc[0])

    def test_item_screen_ignores_future_counts(self):
        counts = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-01", "2026-01-01"]),
                               "item_code": [1, 2, 2], "records": [1000, 1, 999999]})
        screen = item_screen(counts, "2025-12-31").set_index("item_code")
        self.assertTrue(screen.loc[1, "include"])
        self.assertFalse(screen.loc[2, "include"])

    def test_target_horizon_does_not_cross_cutoff(self):
        frame = make_features(aggregate_weeks(daily_example()))
        training, evaluation = time_masks(frame, "2025-02-16", "2025-03-30")
        self.assertTrue((frame.loc[training, "next_week_end"] <= pd.Timestamp("2025-02-16")).all())
        self.assertTrue((frame.loc[evaluation, "week_end"] > pd.Timestamp("2025-02-16")).all())
        self.assertFalse((training & evaluation).any())

    def test_unit_conversion_and_rejection(self):
        frame = pd.DataFrame({"project_group": ["Eggs", "Eggs", "Chicken"],
                              "unit": ["30 biji", "10 biji", "1kg"], "price": [15.0, 5.0, 9.0]})
        np.testing.assert_allclose(normalise_units(frame)["price"], [5, 5, 9])
        frame.loc[0, "unit"] = "unknown"
        with self.assertRaises(ValueError):
            normalise_units(frame)

    def test_aggregation_detects_composition_effect(self):
        daily = daily_example()
        # Same shop prices, changing observation frequency: primary median changes.
        daily["price"] = daily["premise_code"].map({1: 10.0, 2: 20.0})
        first = daily.loc[daily["date"].lt("2025-01-13")]
        first = pd.concat([first, first.loc[first["premise_code"].eq(1)]], ignore_index=True)
        following = daily.loc[daily["date"].ge("2025-01-13") & daily["date"].lt("2025-01-20")]
        following = pd.concat([following, following.loc[following["premise_code"].eq(2)]], ignore_index=True)
        weekly = aggregate_weeks(pd.concat([first, following], ignore_index=True))
        self.assertEqual(weekly.iloc[0]["common_panel_change_pct"], 0.0)
        self.assertEqual(weekly.iloc[0]["equal_premise_median"], weekly.iloc[1]["equal_premise_median"])
        self.assertNotEqual(weekly.iloc[0]["weekly_median_price"], weekly.iloc[1]["weekly_median_price"])

    def test_undefined_ap_and_threshold_selection(self):
        result = metrics([0, 0], [0.1, 0.2])
        self.assertTrue(np.isnan(result["average_precision"]))
        self.assertEqual(result["f1"], 0.0)
        threshold = choose_threshold([0, 1, 1], [0.1, 0.4, 0.9])
        self.assertEqual(threshold, 0.4)

    def test_all_models_impute_and_handle_unseen_categories(self):
        frame = make_features(aggregate_weeks(daily_example()))
        frame["surge_label"] = [0, 1] * 6
        X = model_inputs(frame)
        for name in ["LogisticRegression", "XGBoost", "LightGBM"]:
            pipeline = make_model(name, frame["surge_label"])
            pipeline.fit(X, frame["surge_label"].astype(int))
            unseen = X.iloc[-2:].copy()
            unseen["district"] = "Never seen during training"
            probability = pipeline.predict_proba(unseen)[:, 1]
            self.assertTrue(np.isfinite(probability).all())

    def test_tree_shap_reconstructs_model_margin(self):
        import shap
        frame = make_features(aggregate_weeks(daily_example()))
        frame = pd.concat([frame] * 20, ignore_index=True)
        frame["weekly_median_price"] = np.random.default_rng(42).uniform(1, 20, len(frame))
        labels = frame["weekly_median_price"].gt(10).astype(int)
        X = model_inputs(frame)
        for name in ["XGBoost", "LightGBM"]:
            pipeline = make_model(name, labels)
            pipeline.fit(X, labels)
            matrix = pipeline.named_steps["preprocess"].transform(X.iloc[:5])
            matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
            estimator = pipeline.named_steps["model"]
            explanation = shap.TreeExplainer(estimator)(matrix)
            if name == "XGBoost":
                margin = estimator.predict(matrix, output_margin=True)
            else:
                margin = estimator.predict(matrix, raw_score=True)
            np.testing.assert_allclose(explanation.base_values + explanation.values.sum(axis=1),
                                       margin, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
