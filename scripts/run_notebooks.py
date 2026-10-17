"""Execute the research notebooks, retaining separate executed copies and logs.

Run from the repository root: python scripts/run_notebooks.py
Use --start 6 to rerun modelling after deliberate, documented changes.
"""
import argparse
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pricecatcher_config as cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--end", type=int, default=6, choices=range(1, 7))
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("start must not exceed end")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    # Keep caches inside the project when the account home is read-only.
    for name, folder in [("IPYTHONDIR", "ipython"), ("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "cache")]:
        location = cfg.DATA_DIR / ".runtime" / folder
        location.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(location))
    destination = cfg.OUTPUT_DIR / "executed_notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        if not args.start <= int(path.name[:2]) <= args.end:
            continue
        print("Executing", path.name, flush=True)
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        try:
            NotebookClient(notebook, timeout=3600, kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}}).execute()
        finally:
            nbformat.write(notebook, destination / path.name)
        print("Passed", path.name, flush=True)


if __name__ == "__main__":
    main()
