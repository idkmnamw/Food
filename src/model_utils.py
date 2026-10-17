"""Readable model factories and evaluation helpers used by Notebook 06."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pricecatcher_config as cfg


def model_inputs(frame):
    """Only explicitly listed inputs; no outcomes or future coverage can enter X."""
    result = frame[cfg.FEATURES].copy()
    for name in cfg.CAT_FEATURES:
        result[name] = result[name].astype("string").fillna("Unknown").astype(str)
    result[cfg.NUM_FEATURES] = result[cfg.NUM_FEATURES].astype(float)
    if np.isinf(result[cfg.NUM_FEATURES].to_numpy()).any():
        raise ValueError("Infinite model input found.")
    return result


def make_model(name, y):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler(with_mean=False)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, cfg.NUM_FEATURES),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cfg.CAT_FEATURES),
    ], sparse_threshold=1.0)
    positives = int(np.asarray(y).sum())
    if positives == 0 or positives == len(y):
        raise ValueError("Training needs both classes; report this fold as unsupported.")
    weight = (len(y) - positives) / positives
    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1500, random_state=cfg.SEED),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=weight,
            tree_method="hist", eval_metric="logloss", n_jobs=cfg.N_JOBS, random_state=cfg.SEED),
        "LightGBM": LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=weight,
            verbosity=-1, n_jobs=cfg.N_JOBS, random_state=cfg.SEED),
    }
    return Pipeline([("preprocess", preprocessing), ("model", models[name])])


def metrics(y, probabilities, threshold=0.5):
    y, probabilities = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    if not len(y) or not np.isfinite(probabilities).all():
        raise ValueError("Evaluation needs non-empty labels and finite scores.")
    predicted = probabilities >= threshold
    prevalence = float(y.mean())
    ap = float(average_precision_score(y, probabilities)) if y.sum() else np.nan
    return {"rows": len(y), "positives": int(y.sum()), "prevalence": prevalence,
        "average_precision": ap, "ap_lift": ap / prevalence if prevalence else np.nan,
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "alerts": int(predicted.sum()), "threshold": float(threshold)}


def choose_threshold(y, probabilities):
    """Choose maximum F1 on validation only; highest cutoff wins equal-F1 ties."""
    if np.unique(y).size != 2:
        raise ValueError("Validation needs both classes to select an F1 threshold.")
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    candidates = np.flatnonzero(np.isclose(f1, f1.max()))
    return float(thresholds[candidates[-1]])


def grouped_metrics(frame, probabilities, threshold, columns):
    result = frame[columns + ["surge_label"]].copy()
    result["probability"] = probabilities
    rows = []
    for key, part in result.groupby(columns, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(columns, key))
        row.update(metrics(part["surge_label"], part["probability"], threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def week_bootstrap(frame, probabilities, threshold, repeats=200):
    """Resample whole prediction weeks, preserving within-week district dependence.

    This interval describes this short test period, not independent future years.
    """
    weeks = frame["week_start"].to_numpy()
    unique = np.unique(weeks)
    if len(unique) < 2:
        raise ValueError("At least two test weeks are required for week bootstrap.")
    index = [np.flatnonzero(weeks == week) for week in unique]
    labels = frame["surge_label"].to_numpy(dtype=int)
    rng = np.random.default_rng(cfg.SEED)
    rows = []
    for _ in range(repeats):
        sample = np.concatenate([index[i] for i in rng.integers(0, len(index), len(index))])
        rows.append(metrics(labels[sample], np.asarray(probabilities)[sample], threshold))
    estimates = pd.DataFrame(rows)
    return estimates[["average_precision", "f1", "precision", "recall"]].quantile([0.025, 0.975])
