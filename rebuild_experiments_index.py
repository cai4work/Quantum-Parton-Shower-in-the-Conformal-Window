"""Rebuild experiments.csv without running a simulation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from Parameters import PARAMS
from result_storage import rebuild_experiments_index, resolve_results_root


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild experiments.csv from experiment-* directories that contain "
            "parameters.json, ordered by experiment creation time. Completed "
            "simulation runs are not required."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help=(
            "Results directory to scan. Relative paths are resolved from this "
            "script's directory. The configured RESULTS_DIR is used by default."
        ),
    )
    return parser.parse_args()


def _resolve_results_dir(value: Path | None) -> Path:
    if value is None:
        return resolve_results_root(BASE_DIR, PARAMS)
    if not value.is_absolute():
        value = BASE_DIR / value
    return value.resolve()


def main() -> None:
    args = parse_args()
    results_dir = _resolve_results_dir(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    index_path = rebuild_experiments_index(results_dir)
    with index_path.open(newline="", encoding="utf-8") as handle:
        experiment_count = sum(1 for _ in csv.DictReader(handle))

    print(f"Updated {index_path}")
    print(f"Indexed experiments: {experiment_count}")


if __name__ == "__main__":
    main()
