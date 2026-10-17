"""Shared research settings. Change these before running notebooks 01 -> 06."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PRICECATCHER_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("PRICECATCHER_OUTPUT_DIR", ROOT / "outputs"))
# Optional local mirror containing lookup_item.parquet, lookup_premise.parquet,
# and pricecatcher_YYYY-MM.parquet. Never mix synthetic and official sources.
SOURCE_DIR = os.environ.get("PRICECATCHER_SOURCE_DIR")
BASE_URL = "https://storage.data.gov.my/pricecatcher"
START_MONTH, END_MONTH = "2025-01", "2026-06"
TRAIN_END = "2025-12-31"
VALIDATION_END = "2026-03-31"
CV_WINDOWS = [
    ("2025-06-30", "2025-08-31"),
    ("2025-08-31", "2025-10-31"),
    ("2025-10-31", "2025-12-31"),
]
SURGE_THRESHOLD_PCT = 10.0
MIN_RECORDS, MIN_PREMISES, MIN_HISTORY_4W = 2, 1, 2
LOW_COVERAGE_RATIO = 0.05
REMOVE_BORONG = True
SEED, N_JOBS = 42, 2
# Provisional research acceptance criteria, not claims of achieved performance.
# Justify these against stakeholder warning costs in the dissertation.
TARGET_F1, TARGET_AP_LIFT = 0.50, 1.50

SERIES_KEYS = ["item_code", "state", "district", "premise_type"]
CAT_FEATURES = SERIES_KEYS + ["project_group"]
NUM_FEATURES = [
    "weekly_median_price", "lag_1w_price", "lag_2w_price", "lag_4w_price",
    "weekly_price_change_pct", "rolling_mean_4w", "rolling_std_4w",
    "rolling_mean_8w", "rolling_std_8w", "history_count_4w",
    "weekly_std_price", "record_count", "premise_count", "observed_days",
    "month", "week_sin", "week_cos",
]
FEATURES = CAT_FEATURES + NUM_FEATURES

# Replacement 27-code basket: the original shared config was not supplied.
# Codes and descriptions were checked against the official item lookup.
# Inspect outputs/01/basket.csv before interpreting results. This is not a claim
# to reproduce the original basket exactly. A custom basket.csv at the repo root
# (columns item_code, project_group) takes precedence over these rules.
BASKET_CODES = {
    "Chicken": [1, 2],
    "Eggs": [118, 119, 120, 1109, 1110, 1111],
    "Rice": [992, 1581, 1582, 1583],
    "Flour": [917, 1593],
    "Sugar": [1589],
    "Cooking Oil": [918],
    "Vegetables": [93, 94, 104, 105, 113, 114, 1396, 1458, 1556, 1558, 1559],
}

def stage(number):
    path = OUTPUT_DIR / f"{number:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def source_file(name):
    return str(Path(SOURCE_DIR) / name) if SOURCE_DIR else f"{BASE_URL}/{name}"
