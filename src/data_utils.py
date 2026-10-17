"""Small reusable operations shared by the data notebooks and tests."""
import hashlib
import json
import shutil
import urllib.request
import numpy as np
import pandas as pd
import pricecatcher_config as cfg


def fetch_file(source, destination):
    """Cache source bytes without disabling TLS verification or overwriting snapshots."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        if str(source).startswith("https://"):
            with urllib.request.urlopen(source, timeout=180) as response, open(temporary, "wb") as output:
                shutil.copyfileobj(response, output)
        else:
            shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_basket(items):
    """Resolve explicit item codes once; later runs reuse the frozen basket."""
    custom = cfg.ROOT / "basket.csv"
    if custom.exists():
        basket = pd.read_csv(custom, usecols=["item_code", "project_group"])
    else:
        basket = pd.DataFrame([
            {"item_code": code, "project_group": group}
            for group, codes in cfg.BASKET_CODES.items() for code in codes
        ])
    if basket.empty or basket["item_code"].duplicated().any() or basket.isna().any().any():
        raise ValueError("Basket must have unique codes and non-missing groups.")
    basket = basket.merge(items, on="item_code", how="left", validate="one_to_one", indicator=True)
    if not basket["_merge"].eq("both").all():
        raise ValueError("Basket contains codes absent from the lookup snapshot.")
    return basket.drop(columns="_merge")


def normalise_units(frame):
    """Convert only documented 30-egg packs to ten-egg equivalents."""
    frame = frame.copy()
    eggs = frame["project_group"].eq("Eggs")
    unit = frame["unit"].astype("string").str.strip().str.lower()
    if (eggs & ~unit.isin(["10 biji", "30 biji"])).any():
        raise ValueError("Unexpected egg unit: review lookup rather than guessing a conversion.")
    frame.loc[eggs & unit.eq("30 biji"), "price"] /= 3.0
    frame.loc[eggs, "unit"] = "10 biji"
    return frame


def surge_event(change_pct, threshold=None):
    """Include exact-threshold changes despite binary floating-point roundoff.

    For example, (3.30 / 3.00 - 1) * 100 can be slightly below 10.
    The tolerance is 1e-10 percentage points, not a substantive price tolerance.
    """
    threshold = cfg.SURGE_THRESHOLD_PCT if threshold is None else threshold
    return change_pct.ge(threshold) | np.isclose(change_pct, threshold, rtol=0, atol=1e-10)


def item_screen(counts, cutoff):
    """Learn item eligibility using records available by this training cutoff."""
    before = counts.loc[counts["date"] <= pd.Timestamp(cutoff)]
    result = before.groupby("item_code")["records"].sum()
    result = result.reindex(counts["item_code"].unique(), fill_value=0).rename("records").reset_index()
    positive = result.loc[result["records"] > 0, "records"]
    threshold = positive.median() * cfg.LOW_COVERAGE_RATIO if len(positive) else 0.0
    result["cutoff"] = float(threshold)
    result["include"] = (result["records"] > 0) & (result["records"] >= threshold)
    return result


def aggregate_weeks(daily):
    """Primary median weights daily observations; alternatives support auditing."""
    daily = daily.copy()
    daily["week_start"] = daily["date"] - pd.to_timedelta(daily["date"].dt.dayofweek, unit="D")
    keys = cfg.SERIES_KEYS + ["week_start"]
    weekly = daily.groupby(keys, observed=True).agg(
        weekly_median_price=("price", "median"), weekly_std_price=("price", "std"),
        weekly_min_price=("price", "min"), weekly_max_price=("price", "max"),
        record_count=("price", "size"), premise_count=("premise_code", "nunique"),
        observed_days=("date", "nunique"), suspicious_record_count=("is_suspicious_price", "sum"),
    ).reset_index()
    premise = daily.groupby(keys + ["premise_code"], observed=True)["price"].median().rename("premise_median").reset_index()
    alternative = premise.groupby(keys, observed=True)["premise_median"].median().rename("equal_premise_median").reset_index()
    weekly = weekly.merge(alternative, on=keys, validate="one_to_one")
    # Join only identical premises in immediately consecutive calendar weeks.
    following = premise.copy()
    following["week_start"] -= pd.Timedelta(days=7)
    matched = premise.merge(following, on=keys + ["premise_code"], suffixes=("", "_next"), validate="one_to_one")
    common = matched.groupby(keys, observed=True).agg(
        common_premises=("premise_code", "size"),
        common_current_price=("premise_median", "median"),
        common_next_price=("premise_median_next", "median"),
    ).reset_index()
    common["common_panel_change_pct"] = (common["common_next_price"] / common["common_current_price"] - 1) * 100
    weekly = weekly.merge(common, on=keys, how="left", validate="one_to_one")
    metadata = daily[cfg.SERIES_KEYS + ["project_group", "item", "unit"]].drop_duplicates()
    if metadata.duplicated(cfg.SERIES_KEYS).any():
        raise ValueError("Inconsistent names or units inside a series.")
    return weekly.merge(metadata, on=cfg.SERIES_KEYS, validate="many_to_one")


def make_features(weekly):
    """Reindex gaps before shifting: one row back must mean one calendar week."""
    parts = []
    for _, series in weekly.groupby(cfg.SERIES_KEYS, sort=False, observed=True):
        grid = series.set_index("week_start").sort_index()
        calendar = pd.date_range(grid.index.min(), grid.index.max(), freq="W-MON")
        grid = grid.reindex(calendar)
        price = grid["weekly_median_price"]
        past = price.shift(1)
        for lag in [1, 2, 4]:
            grid[f"lag_{lag}w_price"] = price.shift(lag)
        for window in [4, 8]:
            grid[f"rolling_mean_{window}w"] = past.rolling(window, min_periods=2).mean()
            grid[f"rolling_std_{window}w"] = past.rolling(window, min_periods=2).std()
        grid["history_count_4w"] = past.rolling(4, min_periods=1).count()
        grid["weekly_price_change_pct"] = (price / price.shift(1) - 1) * 100
        for source, target in [
            ("weekly_median_price", "next_week_price"), ("record_count", "next_week_record_count"),
            ("premise_count", "next_week_premise_count"), ("equal_premise_median", "next_equal_premise_median"),
        ]:
            grid[target] = grid[source].shift(-1)
        grid["next_week_change_pct"] = (grid["next_week_price"] / price - 1) * 100
        grid = grid.loc[price.notna()].copy()
        grid.index.name = "week_start"
        parts.append(grid.reset_index())
    if not parts:
        raise ValueError("No weekly series available.")
    frame = pd.concat(parts, ignore_index=True)
    # Reindexing may have promoted numeric identifiers to floats. Codes are categories.
    frame["item_code"] = frame["item_code"].astype("int64")
    frame["week_end"] = frame["week_start"] + pd.Timedelta(days=6)
    frame["next_week_end"] = frame["week_start"] + pd.Timedelta(days=13)
    frame["month"] = frame["week_start"].dt.month
    angle = 2 * np.pi * frame["week_start"].dt.isocalendar().week.astype(float) / 52.1775
    frame["week_sin"], frame["week_cos"] = np.sin(angle), np.cos(angle)
    frame["current_coverage_ok"] = (frame["record_count"] >= cfg.MIN_RECORDS) & (frame["premise_count"] >= cfg.MIN_PREMISES)
    frame["target_coverage_ok"] = (frame["next_week_record_count"] >= cfg.MIN_RECORDS) & (frame["next_week_premise_count"] >= cfg.MIN_PREMISES)
    frame["history_ok"] = frame["lag_1w_price"].notna() & (frame["history_count_4w"] >= cfg.MIN_HISTORY_4W)
    valid = frame["current_coverage_ok"] & frame["target_coverage_ok"] & frame["next_week_price"].notna()
    frame["surge_label"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[valid, "surge_label"] = surge_event(frame.loc[valid, "next_week_change_pct"]).astype(int)
    return frame


def time_masks(frame, train_end, evaluation_end):
    """Cutoffs refer to availability of the full target week, not just row date."""
    train_end, evaluation_end = pd.Timestamp(train_end), pd.Timestamp(evaluation_end)
    usable = frame["surge_label"].notna() & frame["history_ok"]
    train = usable & (frame["next_week_end"] <= train_end)
    evaluation = usable & (frame["week_end"] > train_end) & (frame["next_week_end"] <= evaluation_end)
    return train, evaluation
