"""Execute notebooks with a separate deterministic SYNTHETIC fixture.

Not evidence of real-world accuracy. Outputs never enter the normal data folders.
Run: python scripts/validate_notebooks.py --work-dir /tmp/food-validation
"""
import argparse
import os
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pricecatcher_config as cfg


def create_fixture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    groups = ["Chicken", "Eggs", "Rice", "Flour", "Sugar", "Cooking Oil", "Vegetables"]
    codes = [cfg.BASKET_CODES[group][0] for group in groups]
    items = pd.DataFrame([
        {"item_code": code, "item": f"SYNTHETIC {group} {code}",
         "unit": "30 biji" if group == "Eggs" else "1kg"}
        for group, group_codes in cfg.BASKET_CODES.items() for code in group_codes
    ])
    items.to_parquet(directory / "lookup_item.parquet", index=False)
    premises = []
    for district in range(8):
        for shop in range(3):
            premises.append({"premise_code": 100 + district * 3 + shop,
                "premise": f"SYNTHETIC shop {district}-{shop}", "state": "SYNTHETIC STATE",
                "district": f"SYNTHETIC DISTRICT {district}", "premise_type": "Shop"})
    pd.DataFrame(premises).to_parquet(directory / "lookup_premise.parquet", index=False)
    weeks = pd.date_range("2024-12-30", "2026-06-29", freq="W-MON")
    rows = []
    for code, group in zip(codes, groups):
        for district in range(8):
            price = 5.0 + codes.index(code)
            previous_jump = False
            for number, week in enumerate(weeks):
                if rng.random() < 0.035:
                    continue  # Real gaps exercise exact-week lags.
                # Synthetic dependence allows models to train; it is not an economic claim.
                jump = rng.random() < (0.35 if previous_jump else 0.10)
                change = 0.18 if jump else (-0.12 if previous_jump else rng.normal(0, 0.015))
                if group == "Sugar":
                    change = 0.0  # Deliberately unsupported positive-event subgroup.
                price = max(1.0, min(50.0, price * (1 + change)))
                previous_jump = jump
                shops = [0] if rng.random() < 0.12 else [0, 1, 2]
                for shop in shops:
                    for day in [0, 2, 4]:
                        date = week + pd.Timedelta(days=day)
                        if pd.Timestamp("2025-01-01") <= date <= pd.Timestamp("2026-06-30"):
                            value = round(price * (1 + 0.015 * shop), 2)
                            if group == "Eggs":
                                value *= 3
                            rows.append({"date": date, "item_code": code,
                                "premise_code": 100 + district * 3 + shop, "price": value})
    data = pd.DataFrame(rows)
    # Exercise invalid-price counting without corrupting the rest of the fixture.
    data.loc[data.index[10], "price"] = 0.0
    data.loc[data.index[100], "price"] = np.nan
    for month, part in data.groupby(data["date"].dt.to_period("M")):
        part.to_parquet(directory / f"pricecatcher_{month}.parquet", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    arguments = parser.parse_args()
    work = arguments.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        raise SystemExit("Use a new empty work directory to avoid stale validation results.")
    source = work / "synthetic_source"
    create_fixture(source)
    os.environ.update({"PRICECATCHER_SOURCE_DIR": str(source), "PRICECATCHER_DATA_DIR": str(work / "data"),
                       "PRICECATCHER_OUTPUT_DIR": str(work / "outputs")})
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    # The standard python3 kernelspec uses PATH to find this virtual environment.
    executed_dir = work / "executed_notebooks"
    executed_dir.mkdir(parents=True)
    completed = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        print("Executing", path.name, flush=True)
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        try:
            NotebookClient(notebook, timeout=900, kernel_name="python3",
                           resources={"metadata": {"path": str(ROOT)}}).execute()
        finally:
            nbformat.write(notebook, executed_dir / path.name)
        completed.append(path.name)
        print("Passed", path.name, flush=True)
    (work / "validation_result.json").write_text(json.dumps({
        "data": "SYNTHETIC ONLY — not FYP performance evidence", "completed": completed}, indent=2))
    print("All six notebooks executed. Synthetic outputs:", work, flush=True)


if __name__ == "__main__":
    main()
