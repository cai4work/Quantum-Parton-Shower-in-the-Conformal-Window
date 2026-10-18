"""Shared simulation API, result storage, readers, and index maintenance."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import importlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import secrets
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from Parameters import BASELINE_PARAMETERS


BASE_DIR = Path(__file__).resolve().parent


# --- Result formats, experiment identity, storage, and data discovery ---

SCHEMA_VERSION = "3.1"
RESULTS_DIRNAME = "results"
EXPERIMENT_PREFIX = "experiment-"
SHORT_HASH_LENGTH = 10

MODE_DIRECTORIES = {
    "quantum": "qvps",
    "qvps": "qvps",
    "classical": "classic",
    "classic": "classic",
}

# Only parameters that can change the simulated distribution belong in the
# experiment identity. Execution and output controls intentionally do not.
EXPERIMENT_PARAMETER_KEYS = (
    "N_CHARGE",
    "ALPHA_D",
    "Q_MIN",
    "Q_MAX",
    "t2_0",
    "E0",
    "LAMBDA_D",
    "M_Q",
    "R_STEP",
    "OVERESTIMATE_FACTOR",
    "QUBITS_PER_FERMION",
    "MIXING_STRENGTH",
    "INITIAL_PHYSICAL_FLAVOR",
    "CLASSICAL_FLAVOR_SCHEME",
    "ENABLE_Q_TO_Q_G",
    "ENABLE_Q_TO_Q_DP",
    "ENABLE_G_TO_GG",
    "ENABLE_G_TO_QQ",
    "ENABLE_DP_TO_QQ",
    "MAX_STEPS",
    "MAX_PAIRS",
    "N_SHOTS",
)

# Retain the shared export for readers; no additional record parameters remain.
EXPERIMENT_RECORD_PARAMETER_KEYS = ()

# Keep the cross-experiment CSV compact. Parameters omitted here still remain
# part of the experiment identity and the complete parameters.json snapshot.
EXPERIMENT_INDEX_PARAMETER_KEYS = (
    "N_CHARGE",
    "ALPHA_D",
    "Q_MIN",
    "Q_MAX",
    "t2_0",
    "E0",
    "LAMBDA_D",
    "M_Q",
    "MIXING_STRENGTH",
    "MAX_STEPS",
    "MAX_PAIRS",
    "N_SHOTS",
)

RUN_FIELDS = (
    "run_id",
    "mode",
    "n_shots",
    "valid_events",
    "valid_rate",
    "mean_n_ph",
    "mean_n_gl",
    "p_mass_state_0",
    "wall_time_s",
    "started_at",
    "finished_at",
    "random_seed",
    "dropped_events",
    "events_path",
    "particles_path",
    "emissions_path",
    "summary_path",
)

EXPERIMENT_INDEX_FIELDS = (
    "experiment_id",
    "created_at",
    *EXPERIMENT_INDEX_PARAMETER_KEYS,
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _canonical_value(value: Any) -> Any:
    """Normalize values so numerically equal inputs hash identically."""
    value = _json_value(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Experiment parameters must be finite, got {value!r}")
        return format(numeric, ".17g")
    if isinstance(value, Mapping):
        return {key: _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"Unsupported experiment parameter value: {value!r}")


def select_experiment_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in EXPERIMENT_PARAMETER_KEYS if key not in params]
    if missing:
        raise KeyError(f"Missing experiment parameters: {', '.join(missing)}")
    return {key: _json_value(params[key]) for key in EXPERIMENT_PARAMETER_KEYS}


def parameter_fingerprint(parameters: Mapping[str, Any]) -> str:
    canonical = {
        key: _canonical_value(value)
        for key, value in sorted(parameters.items())
    }
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_random_seed(configured_seed: Any) -> int:
    """Return a recorded root seed, generating one when configuration is None."""
    if configured_seed is None:
        return secrets.randbits(64)
    seed = int(configured_seed)
    if seed < 0:
        raise ValueError(f"RANDOM_SEED must be non-negative or None, got {seed}")
    return seed


def resolve_results_root(base_dir: Path, params: Mapping[str, Any]) -> Path:
    configured = params.get("RESULTS_DIR")
    if configured is None:
        return (base_dir / RESULTS_DIRNAME).resolve()
    candidate = Path(str(configured))
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _configured_results_root(base_dir: Path) -> Path:
    """Use the shared configuration for discovery without making it mandatory."""
    try:
        from Parameters import BASELINE_PARAMETERS as PARAMS
    except (ImportError, ModuleNotFoundError):
        return base_dir / RESULTS_DIRNAME
    return resolve_results_root(base_dir, PARAMS)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(row.get(key, "")) for key in fieldnames})
    os.replace(temporary, path)


def _ensure_experiment(
    results_root: Path,
    params: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    identity_parameters = select_experiment_parameters(params)
    stored_parameters = {
        **identity_parameters,
        **{
            key: _json_value(params.get(key))
            for key in EXPERIMENT_RECORD_PARAMETER_KEYS
        },
    }
    fingerprint = parameter_fingerprint(identity_parameters)
    hash_length = SHORT_HASH_LENGTH

    while hash_length <= len(fingerprint):
        experiment_id = f"{EXPERIMENT_PREFIX}{fingerprint[:hash_length]}"
        experiment_dir = results_root / experiment_id
        parameters_path = experiment_dir / "parameters.json"
        if not experiment_dir.exists():
            experiment_dir.mkdir(parents=True)
            document = {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": experiment_id,
                "parameter_fingerprint": fingerprint,
                "created_at": _now(),
                "parameters": stored_parameters,
            }
            _atomic_json(parameters_path, document)
            return experiment_dir, document

        if parameters_path.exists():
            with parameters_path.open(encoding="utf-8") as handle:
                document = json.load(handle)
            if document.get("parameter_fingerprint") == fingerprint:
                recorded = document.setdefault("parameters", {})
                changed = document.get("schema_version") != SCHEMA_VERSION
                if changed:
                    document["schema_version"] = SCHEMA_VERSION
                for key in EXPERIMENT_RECORD_PARAMETER_KEYS:
                    if key not in recorded:
                        recorded[key] = stored_parameters[key]
                        changed = True
                if changed:
                    _atomic_json(parameters_path, document)
                return experiment_dir, document
        hash_length += 2

    raise RuntimeError("Unable to allocate a unique experiment directory")


def _next_run_directory(experiment_dir: Path, mode: str) -> tuple[str, Path]:
    method = MODE_DIRECTORIES.get(mode.lower())
    if method is None:
        raise ValueError(f"Unsupported simulation mode: {mode!r}")
    method_dir = experiment_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in method_dir.glob("run-*"):
        suffix = path.name.removeprefix("run-")
        if path.is_dir() and suffix.isdigit():
            existing.append(int(suffix))
    number = max(existing, default=0) + 1
    run_id = f"run-{number:03d}"
    run_dir = method_dir / run_id
    run_dir.mkdir()
    return run_id, run_dir


def _parse_flavors(encoded: str) -> dict[int, int]:
    flavors: dict[int, int] = {}
    if not encoded:
        return flavors
    for item in encoded.split("|"):
        particle_id, flavor = item.split(":", 1)
        flavors[int(particle_id)] = int(flavor)
    return flavors


def _build_tables(
    experiment_id: str,
    run_id: str,
    method: str,
    n_ph: Sequence[int],
    n_gl: Sequence[int],
    mass_state: Sequence[int],
    final_particles: Sequence[str],
    final_flavors: Sequence[str] | None,
    final_emissions: Sequence[str] | None,
    e0: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_count = len(mass_state)
    if not (len(n_ph) == len(n_gl) == len(final_particles) == row_count):
        raise ValueError("Event result columns have inconsistent lengths")
    if final_flavors is None:
        final_flavors = [""] * row_count
    if len(final_flavors) != row_count:
        raise ValueError("final_flavors has an inconsistent length")
    if final_emissions is None:
        final_emissions = [""] * row_count
    if len(final_emissions) != row_count:
        raise ValueError("final_emissions has an inconsistent length")

    particle_rows: list[dict[str, Any]] = []
    emission_rows: list[dict[str, Any]] = []
    final_counts: list[Counter[str]] = []
    for event_id, (encoded_particles, encoded_flavors, encoded_emissions) in enumerate(
        zip(final_particles, final_flavors, final_emissions), start=1
    ):
        flavors = _parse_flavors(encoded_flavors)
        counts: Counter[str] = Counter()
        if encoded_particles:
            for item in encoded_particles.split("|"):
                parts = item.split(":", 4)
                if len(parts) != 5:
                    raise ValueError(
                        f"Invalid final particle in event {event_id}: {item!r}"
                    )
                particle_id, particle_type, x, theta_x, theta_y = parts
                particle_id_int = int(particle_id)
                energy_fraction = float(x)
                counts[particle_type] += 1
                particle_rows.append({
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "mode": method,
                    "event_id": event_id,
                    "particle_id": particle_id_int,
                    "particle_type": particle_type,
                    "energy_fraction": energy_fraction,
                    "energy": energy_fraction * e0,
                    "theta_x": float(theta_x),
                    "theta_y": float(theta_y),
                    "flavor": flavors.get(particle_id_int),
                })
        if encoded_emissions:
            for item in encoded_emissions.split("|"):
                parts = item.split(":", 7)
                if len(parts) != 8:
                    raise ValueError(
                        f"Invalid accepted emission in event {event_id}: {item!r}"
                    )
                (
                    emission_index,
                    step_index,
                    channel,
                    parent_id,
                    parent_type,
                    t2,
                    z,
                    parent_energy_fraction,
                ) = parts
                emission_rows.append({
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "mode": method,
                    "event_id": event_id,
                    "emission_index": int(emission_index),
                    "step_index": int(step_index),
                    "channel": channel,
                    "parent_id": int(parent_id),
                    "parent_type": parent_type,
                    "t2": float(t2),
                    "z": float(z),
                    "parent_energy_fraction": float(parent_energy_fraction),
                })
        final_counts.append(counts)

    events = pd.DataFrame({
        "experiment_id": experiment_id,
        "run_id": run_id,
        "mode": method,
        "event_id": np.arange(1, row_count + 1, dtype=np.int64),
        "n_ph": np.asarray(n_ph, dtype=np.int64),
        "n_gl": np.asarray(n_gl, dtype=np.int64),
        "mass_state": np.asarray(mass_state, dtype=np.int64),
        "n_final_particles": [sum(counts.values()) for counts in final_counts],
        "n_final_chi": [counts["chi"] for counts in final_counts],
        "n_final_g": [counts["g"] for counts in final_counts],
        "n_final_gamma_d": [counts["gamma_d"] for counts in final_counts],
        # Retained in the event table so old string-based analyses can migrate
        # independently from the storage layout.
        "final_particles": list(final_particles),
        "final_flavors": list(final_flavors),
    })

    particle_columns = (
        "experiment_id",
        "run_id",
        "mode",
        "event_id",
        "particle_id",
        "particle_type",
        "energy_fraction",
        "energy",
        "theta_x",
        "theta_y",
        "flavor",
    )
    particles = pd.DataFrame(particle_rows, columns=particle_columns)
    if not particles.empty:
        particles["flavor"] = particles["flavor"].astype("Int64")
    emission_columns = (
        "experiment_id",
        "run_id",
        "mode",
        "event_id",
        "emission_index",
        "step_index",
        "channel",
        "parent_id",
        "parent_type",
        "t2",
        "z",
        "parent_energy_fraction",
    )
    emissions = pd.DataFrame(emission_rows, columns=emission_columns)
    emissions = emissions.astype({
        "experiment_id": "string",
        "run_id": "string",
        "mode": "string",
        "event_id": "int64",
        "emission_index": "int64",
        "step_index": "int64",
        "channel": "string",
        "parent_id": "int64",
        "parent_type": "string",
        "t2": "float64",
        "z": "float64",
        "parent_energy_fraction": "float64",
    })
    return events, particles, emissions


def _mass_probabilities(mass_state: Sequence[int]) -> dict[str, float]:
    values, counts = np.unique(np.asarray(mass_state, dtype=np.int64), return_counts=True)
    total = max(int(counts.sum()), 1)
    return {str(int(value)): float(count / total) for value, count in zip(values, counts)}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metadata_strings(values: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key): str(_json_value(value))
        for key, value in values.items()
        if value is not None
    }


def load_run_metadata(data_path: str | os.PathLike[str]) -> dict[str, str]:
    """Load flattened run metadata for a Parquet run or a legacy event CSV."""
    path = Path(data_path)
    if path.suffix.lower() == ".csv":
        metadata: dict[str, str] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if not stripped.startswith("#"):
                    break
                payload = stripped[1:].strip()
                if "," in payload:
                    key, value = payload.split(",", 1)
                    metadata[key.strip()] = value.strip()
        return metadata

    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported event data format: {path}")

    metadata: dict[str, Any] = {}
    if len(path.parents) >= 3:
        parameters_path = path.parents[2] / "parameters.json"
        if parameters_path.exists():
            with parameters_path.open(encoding="utf-8") as handle:
                parameter_document = json.load(handle)
            metadata.update(parameter_document.get("parameters", {}))

    summary_path = path.with_name("summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary beside {path}: {summary_path}")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    stored_metadata = summary.get("metadata", {})
    if stored_metadata:
        metadata.update(stored_metadata)
    else:
        # Schema 2.x stored complete run metadata only in the compatibility
        # CSV. Use it as a metadata sidecar while reading event rows from
        # Parquet so existing runs remain analyzable after the migration.
        legacy_metadata_path = path.with_name("events.csv")
        if legacy_metadata_path.exists():
            metadata.update(load_run_metadata(legacy_metadata_path))
    for key in (
        "schema_version",
        "experiment_id",
        "run_id",
        "random_seed",
        "requested_events",
        "valid_events",
        "dropped_events",
        "valid_rate",
        "wall_time_s",
    ):
        if key in summary:
            metadata.setdefault(key, summary[key])
    metadata.setdefault("N_SHOTS", summary.get("requested_events"))
    for key, value in summary.get("execution", {}).items():
        metadata.setdefault(key.upper(), value)

    stored_mode = str(metadata.get("mode", summary.get("mode", ""))).lower()
    mode = {"qvps": "quantum", "classic": "classical"}.get(stored_mode, stored_mode)
    if mode:
        metadata["mode"] = mode
    metadata.setdefault("method", summary.get("mode"))
    return _metadata_strings(metadata)


def load_event_table(
    data_path: str | os.PathLike[str],
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read event rows from Parquet, with legacy CSV support for old results."""
    path = Path(data_path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path, columns=list(columns) if columns else None)
        except OSError:
            # The first schema-2.1 Dev runs contain malformed Parquet page
            # statistics under current PyArrow. Their sibling CSV is retained
            # solely as a recovery source; all schema-3 runs read Parquet.
            legacy_path = path.with_name("events.csv")
            if not legacy_path.exists():
                raise
            return pd.read_csv(
                legacy_path,
                comment="#",
                usecols=list(columns) if columns else None,
            )
    if suffix == ".csv":
        return pd.read_csv(path, comment="#", usecols=list(columns) if columns else None)
    raise ValueError(f"Unsupported event data format: {path}")


def _update_runs_index(experiment_dir: Path, row: Mapping[str, Any]) -> None:
    path = experiment_dir / "runs.csv"
    rows = _read_csv_rows(path)
    rows.append({key: _json_value(row.get(key, "")) for key in RUN_FIELDS})
    rows.sort(key=lambda item: (item.get("mode", ""), item.get("run_id", "")))
    _atomic_csv(path, RUN_FIELDS, rows)


def rebuild_experiments_index(results_root: Path) -> Path:
    indexed_rows: list[tuple[str | None, dict[str, Any]]] = []
    for experiment_dir in sorted(results_root.glob(f"{EXPERIMENT_PREFIX}*")):
        parameters_path = experiment_dir / "parameters.json"
        if not experiment_dir.is_dir() or not parameters_path.exists():
            continue
        with parameters_path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        parameters = document.get("parameters", {})
        created_at = document.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            created_at = None
        row: dict[str, Any] = {
            "experiment_id": document.get("experiment_id", experiment_dir.name),
            "created_at": created_at or "",
            **{key: parameters.get(key, "") for key in EXPERIMENT_INDEX_PARAMETER_KEYS},
        }
        indexed_rows.append((created_at, row))
    indexed_rows.sort(
        key=lambda item: (
            item[0] is None,
            item[0] or "",
            str(item[1]["experiment_id"]),
        )
    )
    rows = [row for _, row in indexed_rows]
    index_path = results_root / "experiments.csv"
    _atomic_csv(index_path, EXPERIMENT_INDEX_FIELDS, rows)
    return index_path


def save_simulation_results(
    *,
    base_dir: Path,
    params: Mapping[str, Any],
    mode: str,
    simulation_metadata: Mapping[str, Any],
    n_ph: Sequence[int],
    n_gl: Sequence[int],
    mass_state: Sequence[int],
    final_particles: Sequence[str],
    final_flavors: Sequence[str] | None,
    dropped_events: int,
    wall_time_s: float,
    random_seed: int,
    started_at: str,
    final_emissions: Sequence[str] | None = None,
) -> dict[str, Path | str]:
    """Save one completed simulation run and refresh both index levels."""
    results_root = resolve_results_root(base_dir, params)
    results_root.mkdir(parents=True, exist_ok=True)
    experiment_dir, parameter_document = _ensure_experiment(results_root, params)
    method = MODE_DIRECTORIES.get(mode.lower())
    if method is None:
        raise ValueError(f"Unsupported simulation mode: {mode!r}")
    run_id, run_dir = _next_run_directory(experiment_dir, method)
    finished_at = _now()

    events, particles, emissions = _build_tables(
        parameter_document["experiment_id"],
        run_id,
        method,
        n_ph,
        n_gl,
        mass_state,
        final_particles,
        final_flavors,
        final_emissions,
        float(params["E0"]),
    )
    events_path = run_dir / "events.parquet"
    particles_path = run_dir / "particles.parquet"
    emissions_path = run_dir / "emissions.parquet"
    summary_path = run_dir / "summary.json"
    events.to_parquet(events_path, index=False)
    particles.to_parquet(particles_path, index=False)
    emissions.to_parquet(emissions_path, index=False)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": parameter_document["experiment_id"],
        "run_id": run_id,
        "method": method,
        "random_seed": random_seed,
        **dict(simulation_metadata),
    }
    valid_events = len(events)
    requested_events = int(params["N_SHOTS"])
    probabilities = _mass_probabilities(mass_state)
    mean_n_ph = float(events["n_ph"].mean())
    mean_n_gl = float(events["n_gl"].mean())
    p_mass_state_0 = float(probabilities.get("0", 0.0))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": parameter_document["experiment_id"],
        "run_id": run_id,
        "mode": method,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "random_seed": random_seed,
        "requested_events": requested_events,
        "valid_events": valid_events,
        "dropped_events": int(dropped_events),
        "valid_rate": valid_events / requested_events if requested_events else 0.0,
        "wall_time_s": float(wall_time_s),
        "metadata": _json_value(metadata),
        "execution": {
            "n_workers": int(params["N_WORKERS"]),
            "progress_interval": int(params["PROGRESS_INTERVAL"]),
        },
        "metrics": {
            "mean_n_ph": mean_n_ph,
            "mean_n_gl": mean_n_gl,
            "p_mass_state_0": p_mass_state_0,
            "mass_state_probabilities": probabilities,
        },
        "emission_table": {
            "row_grain": "one accepted splitting",
            "t2_semantics": (
                "parent evolution virtuality after the current R_STEP update "
                "and immediately before daughter virtualities are assigned"
            ),
            "step_index_semantics": "one-based global shower evolution step",
        },
        "files": {
            "events": "events.parquet",
            "particles": "particles.parquet",
            "emissions": "emissions.parquet",
        },
    }
    _atomic_json(summary_path, summary)

    relative_run = run_dir.relative_to(experiment_dir).as_posix()
    run_row = {
        "run_id": run_id,
        "mode": method,
        "n_shots": requested_events,
        "valid_events": valid_events,
        "valid_rate": summary["valid_rate"],
        "mean_n_ph": mean_n_ph,
        "mean_n_gl": mean_n_gl,
        "p_mass_state_0": p_mass_state_0,
        "wall_time_s": float(wall_time_s),
        "started_at": started_at,
        "finished_at": finished_at,
        "random_seed": random_seed,
        "dropped_events": int(dropped_events),
        "events_path": f"{relative_run}/events.parquet",
        "particles_path": f"{relative_run}/particles.parquet",
        "emissions_path": f"{relative_run}/emissions.parquet",
        "summary_path": f"{relative_run}/summary.json",
    }
    _update_runs_index(experiment_dir, run_row)
    index_path = rebuild_experiments_index(results_root)
    return {
        "experiment_id": parameter_document["experiment_id"],
        "run_id": run_id,
        "experiment_dir": experiment_dir,
        "run_dir": run_dir,
        "events_path": events_path,
        "particles_path": particles_path,
        "emissions_path": emissions_path,
        "summary_path": summary_path,
        "index_path": index_path,
    }


def _latest_run_data_in_experiment(experiment_dir: Path, method: str) -> Path | None:
    method_dir = experiment_dir / method
    candidates = [
        (int(path.name.removeprefix("run-")), path / "events.parquet")
        for path in method_dir.glob("run-*")
        if (
            path.is_dir()
            and path.name.removeprefix("run-").isdigit()
            and (path / "events.parquet").exists()
            and (path / "summary.json").exists()
        )
    ] if method_dir.exists() else []
    return max(candidates, default=(0, None))[1]


def _experiment_recency(experiment_dir: Path) -> tuple[str, float]:
    runs = _read_csv_rows(experiment_dir / "runs.csv")
    updated = max((row.get("finished_at", "") for row in runs), default="")
    return updated, experiment_dir.stat().st_mtime


def discover_latest_pair(base_dir: Path) -> tuple[Path, Path, Path | None]:
    """Return latest Classic/QVPS event tables from one experiment."""
    results_root = _configured_results_root(base_dir)
    candidates = []
    if results_root.exists():
        for experiment_dir in results_root.glob(f"{EXPERIMENT_PREFIX}*"):
            qvps = _latest_run_data_in_experiment(experiment_dir, "qvps")
            classic = _latest_run_data_in_experiment(experiment_dir, "classic")
            if qvps is not None and classic is not None:
                candidates.append((experiment_dir, classic, qvps))
    if candidates:
        experiment_dir, classic, qvps = max(
            candidates, key=lambda item: _experiment_recency(item[0])
        )
        return classic, qvps, experiment_dir

    classic = _latest_legacy_csv(base_dir, "Classic_data_")
    qvps = _latest_legacy_csv(base_dir, "QVPS_CW_data_")
    return classic, qvps, None


def _resolve_experiment_directory(base_dir: Path, experiment: str) -> Path:
    value = experiment.strip().lower()
    if value.startswith(EXPERIMENT_PREFIX):
        short_hash = value.removeprefix(EXPERIMENT_PREFIX)
    else:
        short_hash = value
    if not re.fullmatch(r"[0-9a-f]{10,64}", short_hash):
        raise ValueError(
            "Experiment must be a 10-64 character hexadecimal hash or an "
            f"{EXPERIMENT_PREFIX}<hash> ID, got {experiment!r}"
        )

    experiment_id = f"{EXPERIMENT_PREFIX}{short_hash}"
    experiment_dir = _configured_results_root(base_dir) / experiment_id
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment not found: {experiment_id}")
    return experiment_dir


def discover_pair_for_experiment(
    base_dir: Path,
    experiment: str,
) -> tuple[Path, Path, Path]:
    """Return the latest Classic/QVPS runs for one experiment hash or ID."""
    experiment_dir = _resolve_experiment_directory(base_dir, experiment)
    experiment_id = experiment_dir.name

    qvps = _latest_run_data_in_experiment(experiment_dir, "qvps")
    classic = _latest_run_data_in_experiment(experiment_dir, "classic")
    missing = [
        method
        for method, path in (("QVPS", qvps), ("Classic", classic))
        if path is None
    ]
    if missing:
        raise FileNotFoundError(
            f"{experiment_id} has no completed {' or '.join(missing)} run"
        )
    return classic, qvps, experiment_dir


def discover_mode_for_experiment(
    base_dir: Path,
    experiment: str,
    method: str,
) -> tuple[Path, Path]:
    """Return the latest run for one method in a selected experiment."""
    normalized_method = MODE_DIRECTORIES.get(method.lower())
    if normalized_method is None:
        raise ValueError(f"Unsupported simulation mode: {method!r}")
    experiment_dir = _resolve_experiment_directory(base_dir, experiment)
    data_path = _latest_run_data_in_experiment(experiment_dir, normalized_method)
    if data_path is None:
        display_name = "QVPS" if normalized_method == "qvps" else "Classic"
        raise FileNotFoundError(
            f"{experiment_dir.name} has no completed {display_name} run"
        )
    return data_path, experiment_dir


def _latest_legacy_csv(base_dir: Path, prefix: str) -> Path:
    numbered: list[tuple[int, Path]] = []
    for path in base_dir.glob(f"{prefix}*.csv"):
        suffix = path.stem[len(prefix):]
        if suffix.isdigit():
            numbered.append((int(suffix), path.resolve()))
    if not numbered:
        raise FileNotFoundError(f"No structured runs or legacy {prefix}*.csv files found")
    return max(numbered)[1]


def discover_latest_mode_with_experiment(
    base_dir: Path,
    method: str,
) -> tuple[Path, Path | None]:
    """Return the latest run for one method and its structured experiment."""
    method = MODE_DIRECTORIES.get(method.lower(), method.lower())
    results_root = _configured_results_root(base_dir)
    candidates: list[tuple[Path, Path]] = []
    if results_root.exists():
        for experiment_dir in results_root.glob(f"{EXPERIMENT_PREFIX}*"):
            data_path = _latest_run_data_in_experiment(experiment_dir, method)
            if data_path is not None:
                candidates.append((experiment_dir, data_path))
    if candidates:
        experiment_dir, data_path = max(
            candidates, key=lambda item: _experiment_recency(item[0])
        )
        return data_path, experiment_dir
    prefix = "QVPS_CW_data_" if method == "qvps" else "Classic_data_"
    return _latest_legacy_csv(base_dir, prefix), None


def discover_latest_mode(base_dir: Path, method: str) -> Path:
    """Return the latest run for one method."""
    data_path, _experiment_dir = discover_latest_mode_with_experiment(base_dir, method)
    return data_path


def discover_latest_runs_by_experiment(base_dir: Path, method: str) -> list[Path]:
    method = MODE_DIRECTORIES.get(method.lower(), method.lower())
    results_root = _configured_results_root(base_dir)
    discovered: list[Path] = []
    if results_root.exists():
        for experiment_dir in sorted(results_root.glob(f"{EXPERIMENT_PREFIX}*")):
            data_path = _latest_run_data_in_experiment(experiment_dir, method)
            if data_path is not None:
                discovered.append(data_path)
    if discovered:
        return discovered
    prefix = "QVPS_CW_data_" if method == "qvps" else "Classic_data_"
    legacy = []
    for path in base_dir.glob(f"{prefix}*.csv"):
        suffix = path.stem[len(prefix):]
        if suffix.isdigit():
            legacy.append((int(suffix), path.resolve()))
    if not legacy:
        raise FileNotFoundError(f"No structured runs or legacy {prefix}*.csv files found")
    return [path for _, path in sorted(legacy)]


# --- Simulation configuration and execution ---

_SIMULATOR_MODULES = {"qvps": "QVPS_CW", "classic": "Classic"}
_worker_simulator = None
_worker_parameters = None


def baseline_input_parameters():
    """Read legacy input overrides without mutating the default baseline.

    This is an input-layer helper for scripts. Programmatic calls with explicit
    parameters do not implicitly read process environment variables.
    """
    params = dict(BASELINE_PARAMETERS)
    for key in ("Q_MIN", "Q_MAX", "M_Q", "MIXING_STRENGTH"):
        value = os.environ.get(f"QVPS_{key}")
        if value is not None:
            params[key] = float(value)
    for key in ("N_SHOTS", "MAX_STEPS"):
        value = os.environ.get(f"QVPS_{key}")
        if value is not None:
            params[key] = int(value)
    return params


def normalize_method(method):
    try:
        return MODE_DIRECTORIES[method.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"Unknown simulation method: {method!r}") from exc


def mass_cutoff_steps(parameters):
    """Return the grid steps needed to reach the physical mass cutoff."""
    t2_0 = float(parameters["t2_0"])
    r_step = float(parameters["R_STEP"])
    mass = float(parameters["M_Q"])
    if not math.isfinite(t2_0) or t2_0 <= 0.0:
        raise ValueError("t2_0 must be finite and positive")
    if not math.isfinite(r_step) or not 0.0 < r_step < 1.0:
        raise ValueError("R_STEP must satisfy 0 < R_STEP < 1")
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("M_Q must be finite and positive")
    log_ratio = math.log(4.0) + 2.0 * math.log(mass) - math.log(t2_0)
    return max(0, int(math.ceil(log_ratio / math.log(r_step))))


def prepare_parameters(parameters=None, *, shots=None, workers=None, random_seed=None,
                       results_dir=None):
    """Validate explicit step limits, defaulting to the baseline configuration."""
    overrides = dict(parameters or {})
    unknown = overrides.keys() - BASELINE_PARAMETERS.keys()
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    params = {**BASELINE_PARAMETERS, **overrides}
    if shots is not None:
        params["N_SHOTS"] = shots
    if workers is not None:
        params["N_WORKERS"] = workers
    if random_seed is not None:
        params["RANDOM_SEED"] = random_seed
    if results_dir is not None:
        params["RESULTS_DIR"] = str(results_dir)
    if params["N_WORKERS"] is None:
        params["N_WORKERS"] = max(1, (os.cpu_count() or 1) // 2)
    for key, minimum in (("N_SHOTS", 1), ("N_WORKERS", 1),
                         ("PROGRESS_INTERVAL", 1), ("MAX_STEPS", 0), ("MAX_PAIRS", 0)):
        value = params[key]
        if isinstance(value, bool) or int(value) != value or int(value) < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
        params[key] = int(value)
    for key in ("t2_0", "E0", "LAMBDA_D", "M_Q"):
        if not math.isfinite(float(params[key])) or float(params[key]) <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0.0 < float(params["R_STEP"]) < 1.0:
        raise ValueError("R_STEP must satisfy 0 < R_STEP < 1")
    return params


def experiment_directory(parameters, base_dir=BASE_DIR):
    """Locate an experiment using the existing identity and directory rules."""
    fingerprint = parameter_fingerprint(select_experiment_parameters(parameters))
    root = resolve_results_root(Path(base_dir), parameters)
    for candidate in sorted(root.glob("experiment-*")):
        path = candidate / "parameters.json"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        if document.get("parameter_fingerprint") == fingerprint:
            return candidate
    return root / f"experiment-{fingerprint[:10]}"


def completed_run(experiment_dir, method, *, random_seed=None, requested_events=None):
    """Return the latest completed, nonempty run with consistent event counts."""
    candidates = []
    for run_dir in (Path(experiment_dir) / normalize_method(method)).glob("run-*"):
        summary_path = run_dir / "summary.json"
        events_path = run_dir / "events.parquet"
        if not summary_path.is_file() or not events_path.is_file() or events_path.stat().st_size == 0:
            continue
        try:
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            valid = int(summary["valid_events"])
            dropped = int(summary["dropped_events"])
            requested = int(summary["requested_events"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if (summary.get("status") == "completed" and valid > 0 and dropped >= 0
                and valid + dropped == requested):
            if random_seed is not None and summary.get("random_seed") != int(random_seed):
                continue
            if requested_events is not None and requested != int(requested_events):
                continue
            candidates.append((str(summary.get("finished_at", "")), run_dir.name, run_dir, summary))
    if not candidates:
        return None
    _, _, run_dir, summary = max(candidates, key=lambda item: item[:2])
    return {"run_dir": run_dir, "summary": summary}


def has_completed_run(experiment_dir, method):
    return completed_run(experiment_dir, method) is not None


def require_simulator(method):
    method = normalize_method(method)
    if method == "qvps" and importlib.util.find_spec("qiskit") is None:
        raise ModuleNotFoundError("Qiskit is required for QVPS; run in the qsimu Conda environment")
    return importlib.import_module(_SIMULATOR_MODULES[method])


def _initialize_worker(method, parameters):
    global _worker_simulator, _worker_parameters
    _worker_simulator = require_simulator(method)
    _worker_parameters = dict(parameters)
    # Matrix-logarithm setup can consume NumPy RNG state on its first call.
    # Finish it before simulate_event sets each event's independent seed.
    _worker_simulator.CWKinematics12(_worker_parameters)


def _run_worker(seed):
    return _worker_simulator.simulate_event(_worker_parameters, seed)


def _collect_events(method, parameters, root_seed, progress):
    seeds = [s.generate_state(1)[0] for s in np.random.SeedSequence(root_seed).spawn(parameters["N_SHOTS"])]
    events = []
    dropped = 0
    start = time.time()

    def consume(iterator):
        nonlocal dropped
        for done, event in enumerate(iterator, start=1):
            if event is None:
                dropped += 1
            else:
                events.append(event)
            if progress and (done % parameters["PROGRESS_INTERVAL"] == 0 or done == len(seeds)):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(seeds) - done) / rate if rate > 0 else 0.0
                print(f"  {done:>6d}/{len(seeds)} ({elapsed:.1f}s, {rate:.0f} shots/s, ETA {eta:.0f}s)")

    if parameters["N_WORKERS"] == 1:
        simulator = require_simulator(method)
        consume(simulator.simulate_event(parameters, seed) for seed in seeds)
    else:
        with mp.Pool(parameters["N_WORKERS"], initializer=_initialize_worker,
                     initargs=(method, parameters)) as pool:
            consume(pool.imap_unordered(_run_worker, seeds, chunksize=32))
    return events, dropped, time.time() - start


def _encode_particles(particles):
    return "|".join(f"{pid}:{kind}:{x:.17g}:{tx:.17g}:{ty:.17g}"
                    for pid, kind, x, tx, ty in particles)


def _encode_emissions(emissions):
    return "|".join(
        f"{index}:{step}:{channel}:{pid}:{kind}:{t2:.17g}:{z:.17g}:{x:.17g}"
        for index, (step, channel, pid, kind, t2, z, x) in enumerate(emissions, start=1)
    )


def run_simulation(method, parameters=None, *, shots=None, workers=None,
                   random_seed=None, results_dir=None, reuse_existing=True,
                   save=True, progress=True, dry_run=False, base_dir=BASE_DIR):
    """Run one effective configuration and return counts, paths, and reuse status.

    With save=False, return physical event records in ``events`` as well.
    Entry-point scripts using multiple workers must have a __main__ guard.
    """
    method = normalize_method(method)
    params = prepare_parameters(parameters, shots=shots, workers=workers,
                                random_seed=random_seed, results_dir=results_dir)
    directory = experiment_directory(params, base_dir)
    previous = (completed_run(directory, method, random_seed=params.get("RANDOM_SEED"),
                              requested_events=params["N_SHOTS"])
                if save and reuse_existing else None)
    if previous is not None:
        summary = previous["summary"]
        if progress:
            print(f"{method}: reusing {previous['run_dir']}")
        return {"reused": True, "parameters": params, "experiment_dir": directory,
                "run_dir": previous["run_dir"], "experiment_id": summary["experiment_id"],
                "run_id": summary["run_id"], "valid_events": summary["valid_events"],
                "dropped_events": summary["dropped_events"], "random_seed": summary["random_seed"]}
    if dry_run:
        if progress:
            print(f"{method}: would run MAX_STEPS={params['MAX_STEPS']}, shots={params['N_SHOTS']}, experiment={directory.name}")
        return {"reused": False, "dry_run": True, "parameters": params, "experiment_dir": directory}
    simulator = require_simulator(method)
    if method == "qvps" and (params["QUBITS_PER_FERMION"] != 4 or not 1 <= params["N_CHARGE"] <= 16):
        raise ValueError("The current QVPS flavor encoding requires 4 qubits and 1..16 charges")
    kin = simulator.CWKinematics12(params)
    root_seed = resolve_random_seed(params.get("RANDOM_SEED"))
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    if progress:
        print(f"{method}: {params['N_SHOTS']} events, {params['N_WORKERS']} workers, MAX_STEPS={params['MAX_STEPS']}")
    events, dropped, wall = _collect_events(method, params, root_seed, progress)
    result = {"reused": False, "parameters": params, "experiment_dir": directory,
              "valid_events": len(events), "dropped_events": dropped,
              "random_seed": root_seed, "wall_time_s": wall}
    if not save:
        result["events"] = events
    if save and events:
        metadata_fn = _qvps_metadata if method == "qvps" else _classic_metadata
        result.update(save_simulation_results(
            base_dir=Path(base_dir), params=params, mode=method,
            simulation_metadata=dict(metadata_fn(params, kin, len(events), dropped, wall)),
            n_ph=[event.n_ph for event in events], n_gl=[event.n_gl for event in events],
            mass_state=[event.mass_state for event in events],
            final_particles=[_encode_particles(event.final_particles) for event in events],
            final_flavors=(["|".join(f"{pid}:{flavor}" for pid, flavor in event.final_flavors)
                            for event in events] if method == "classic" else None),
            final_emissions=[_encode_emissions(event.final_emissions) for event in events],
            dropped_events=dropped, wall_time_s=wall, random_seed=root_seed, started_at=started_at,
        ))
    if progress:
        print(f"  Valid: {len(events)}, Dropped: {dropped}, Wall time: {wall:.1f}s")
        if "run_dir" in result:
            print(f"  Data saved: {result['run_dir']}")
    return result


def _qvps_metadata(params, kin, n_valid, n_fail, wall):
    return [
        ("mode", "quantum"),
        ("N_SHOTS", params["N_SHOTS"]),
        ("valid_events", n_valid),
        ("dropped_events", n_fail),
        ("wall_time_s", f"{wall:.6f}"),
        ("N_WORKERS", params["N_WORKERS"]),
        ("PROGRESS_INTERVAL", params["PROGRESS_INTERVAL"]),
        ("N_C", kin.N_C),
        ("N_F", kin.N_F),
        ("N_CHARGE", kin.N_CHARGE),
        ("MIXING_STRENGTH", kin.mixing_strength),
        ("initial_physical_flavor", params.get("INITIAL_PHYSICAL_FLAVOR", 0)),
        ("ALPHA_D", kin.alpha_D),
        ("Q_MIN", params["Q_MIN"]),
        ("Q_MAX", params["Q_MAX"]),
        ("t2_0", kin.t2_0),
        ("E0", params["E0"]),
        ("LAMBDA_D", kin.lambda_d),
        ("M_Q", kin.M_q),
        ("t2_MIN", kin.t2_MIN),
        ("R_STEP", params["R_STEP"]),
        ("OVERESTIMATE_FACTOR", params["OVERESTIMATE_FACTOR"]),
        ("MAX_STEPS", params["MAX_STEPS"]),
        ("MAX_PAIRS", params["MAX_PAIRS"]),
        ("ENABLE_Q_TO_Q_G", params["ENABLE_Q_TO_Q_G"]),
        ("ENABLE_Q_TO_Q_DP", params["ENABLE_Q_TO_Q_DP"]),
        ("ENABLE_G_TO_GG", params["ENABLE_G_TO_GG"]),
        ("ENABLE_G_TO_QQ", params["ENABLE_G_TO_QQ"]),
        ("ENABLE_DP_TO_QQ", params["ENABLE_DP_TO_QQ"]),
    ]


def _classic_metadata(params, kin, n_valid, n_fail, wall):
    return [
        ("mode", "classical"),
        ("classical_flavor_model", params.get(
            "CLASSICAL_FLAVOR_SCHEME", "paper_physical_markov"
        )),
        ("initial_physical_flavor", params.get("INITIAL_PHYSICAL_FLAVOR", 0)),
        ("physical_coupling_matrix", "U_diag_Q_Udagger"),
        ("mass_state_semantics", (
            "physical_flavor_of_initial_fermion"
            if params.get("CLASSICAL_FLAVOR_SCHEME", "paper_physical_markov")
            == "paper_physical_markov"
            else "mixing_matrix_measurement_of_initial_interaction_eigenstate"
        )),
        ("sudakov_scheme", "stepwise_exp_integrated_rate"),
        ("qiskit_required", False),
        ("N_SHOTS", params["N_SHOTS"]),
        ("valid_events", n_valid),
        ("dropped_events", n_fail),
        ("wall_time_s", f"{wall:.6f}"),
        ("N_WORKERS", params["N_WORKERS"]),
        ("PROGRESS_INTERVAL", params["PROGRESS_INTERVAL"]),
        ("N_C", kin.N_C),
        ("N_F", kin.N_F),
        ("N_CHARGE", kin.N_CHARGE),
        ("MIXING_STRENGTH", kin.mixing_strength),
        ("ALPHA_D", kin.alpha_D),
        ("Q_MIN", params["Q_MIN"]),
        ("Q_MAX", params["Q_MAX"]),
        ("t2_0", kin.t2_0),
        ("E0", params["E0"]),
        ("LAMBDA_D", kin.lambda_d),
        ("M_Q", kin.M_q),
        ("t2_MIN", kin.t2_MIN),
        ("R_STEP", params["R_STEP"]),
        ("OVERESTIMATE_FACTOR", params["OVERESTIMATE_FACTOR"]),
        ("MAX_STEPS", params["MAX_STEPS"]),
        ("MAX_PAIRS", params["MAX_PAIRS"]),
        ("ENABLE_Q_TO_Q_G", params["ENABLE_Q_TO_Q_G"]),
        ("ENABLE_Q_TO_Q_DP", params["ENABLE_Q_TO_Q_DP"]),
        ("ENABLE_G_TO_GG", params["ENABLE_G_TO_GG"]),
        ("ENABLE_G_TO_QQ", params["ENABLE_G_TO_QQ"]),
        ("ENABLE_DP_TO_QQ", params["ENABLE_DP_TO_QQ"]),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run the CW baseline or rebuild its experiment index")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--method", choices=("qvps", "classic", "both"), default="both")
    action.add_argument("--rebuild-index", action="store_true",
                        help="Rebuild experiments.csv from parameters.json without running simulations")
    parser.add_argument("--shots", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.rebuild_index:
        if args.rerun or any(value is not None for value in
                             (args.shots, args.workers, args.seed, args.max_steps)):
            parser.error("--rebuild-index cannot be combined with simulation execution options")
        params = dict(BASELINE_PARAMETERS)
        if args.results_dir is not None:
            params["RESULTS_DIR"] = str(args.results_dir)
        results_root = resolve_results_root(BASE_DIR, params)
        if not results_root.is_dir():
            raise SystemExit(f"Results directory not found: {results_root}")
        if args.dry_run:
            print(f"Would rebuild {results_root / 'experiments.csv'}")
            return
        index_path = rebuild_experiments_index(results_root)
        with index_path.open(newline="", encoding="utf-8") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        print(f"Updated {index_path}")
        print(f"Indexed experiments: {count}")
        return
    overrides = baseline_input_parameters()
    if args.max_steps is not None:
        overrides["MAX_STEPS"] = args.max_steps
    params = prepare_parameters(overrides, shots=args.shots, workers=args.workers,
                                random_seed=args.seed, results_dir=args.results_dir)
    methods = ("classic", "qvps") if args.method == "both" else (args.method,)
    for method in methods:
        if args.dry_run:
            print(f"{method}: MAX_STEPS={params['MAX_STEPS']}, shots={params['N_SHOTS']}, experiment={experiment_directory(params)}")
        else:
            run_simulation(method, params, reuse_existing=not args.rerun)


if __name__ == "__main__":
    main()
