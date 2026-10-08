"""Compute grouped M_Q scans of the QVPS EEC from Parquet event data."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from result_storage import (
    EXPERIMENT_PARAMETER_KEYS,
    EXPERIMENT_RECORD_PARAMETER_KEYS,
    discover_latest_runs_by_experiment,
    discover_mode_for_experiment,
    load_event_table,
    load_run_metadata,
)

OUTPUT_DIRECTORY = "fig"
DEFAULT_PLOT_STEM = "QVPS_CW_eec_mq"
MIN_MQ_DATASETS = 3


plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)
GROUP_PARAMETER_KEYS = tuple(
    key
    for key in (*EXPERIMENT_PARAMETER_KEYS, *EXPERIMENT_RECORD_PARAMETER_KEYS)
    if key != "M_Q"
)


def _resolve_path(path, base_dir):
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(base_dir, path)


def _read_e0(metadata, data_path, e0_override=None):
    if "E0" in metadata:
        source = f"E0 metadata in {data_path}"
        raw_e0 = metadata["E0"]
    elif e0_override is not None:
        source = "--e0"
        raw_e0 = e0_override
    else:
        raise ValueError(f"Missing E0 metadata in {data_path}; provide --e0 manually.")
    try:
        e0 = float(raw_e0)
    except ValueError as exc:
        raise ValueError(f"Invalid {source}: {raw_e0!r}") from exc
    if e0 <= 0.0:
        raise ValueError(f"E0 must be positive from {source}, got {e0}")
    return e0


def _read_m_q(metadata, data_path):
    if "M_Q" not in metadata:
        raise ValueError(f"Missing M_Q metadata in {data_path}.")
    try:
        m_q = float(metadata["M_Q"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid M_Q metadata in {data_path}: {metadata['M_Q']!r}"
        ) from exc
    if not np.isfinite(m_q) or m_q < 0.0:
        raise ValueError(
            f"M_Q must be finite and non-negative in {data_path}, got {m_q}"
        )
    return m_q


def _normalize_group_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value

    text = str(value).strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if np.isfinite(number):
        return format(number, ".17g")
    return text


def _parameter_group(metadata):
    """Return a stable signature for every experiment parameter except M_Q."""
    payload = [
        (key, _normalize_group_value(metadata.get(key)))
        for key in GROUP_PARAMETER_KEYS
    ]
    signature = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    group_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:10]
    return signature, group_id


def group_eec_inputs(input_paths, e0_override=None):
    """Group inputs that differ only in M_Q without reading their event tables."""
    grouped = {}
    rejected = []
    for data_path in input_paths:
        try:
            metadata = load_run_metadata(data_path)
            metadata["E0"] = _read_e0(metadata, data_path, e0_override)
            m_q = _read_m_q(metadata, data_path)
        except (FileNotFoundError, OSError, ValueError) as exc:
            rejected.append((data_path, str(exc)))
            continue

        signature, group_id = _parameter_group(metadata)
        group = grouped.setdefault(signature, {
            "group_id": group_id,
            "inputs": [],
        })
        group["inputs"].append({
            "path": data_path,
            "metadata": metadata,
            "m_q": m_q,
        })

    groups = []
    for group in grouped.values():
        group["inputs"].sort(key=lambda item: (item["m_q"], str(item["path"])))
        group["distinct_m_q"] = len({item["m_q"] for item in group["inputs"]})
        groups.append(group)
    groups.sort(key=lambda group: group["group_id"])
    return groups, rejected


def _group_plot_path(filename, base_dir, group_id, group_count):
    output_dir = Path(base_dir) / OUTPUT_DIRECTORY
    if filename:
        requested_name = Path(filename).name
        stem = Path(requested_name).stem or DEFAULT_PLOT_STEM
    else:
        stem = DEFAULT_PLOT_STEM
    if filename and group_count == 1:
        output_name = f"{stem}.pdf"
    else:
        output_name = f"{stem}_{group_id}.pdf"
    return str(output_dir / output_name)


def _parse_final_particles(value, data_path, event_id):
    particles = []
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return particles
    for item in str(value).split("|"):
        fields = item.split(":")
        if len(fields) != 5:
            raise ValueError(
                f"Invalid final_particles entry in {data_path}, event {event_id}: {item!r}"
            )
        _, _, x, theta_x, theta_y = fields
        particles.append((float(x), float(theta_x), float(theta_y)))
    return particles


def load_eec_events(data_path, e0_override=None):
    metadata = load_run_metadata(data_path)
    e0 = _read_e0(metadata, data_path, e0_override)
    events = load_event_table(data_path)
    required = ("n_ph", "n_gl", "mass_state", "final_particles")
    missing = [key for key in required if key not in events.columns]
    if missing:
        raise ValueError(
            f"Missing required EEC columns in {data_path}: {', '.join(missing)}. "
            "Rerun QVPS_CW.py to generate current Parquet event data."
        )

    results = []
    for row_number, row in enumerate(events.to_dict("records"), start=1):
        event_id = row.get("event_id", row_number)
        particles = _parse_final_particles(row["final_particles"], data_path, event_id)
        eec_rl, eec_w = compute_eec_from_particles(particles, e0)
        results.append((
            int(row["n_ph"]),
            int(row["n_gl"]),
            int(row["mass_state"]),
            eec_rl,
            eec_w,
        ))

    n_fail = int(metadata.get("dropped_events", 0))
    wall_time = float(metadata.get("wall_time_s", 0.0))
    return results, n_fail, wall_time, e0, metadata


def compute_eec_from_particles(particles, e0):
    final = []
    for p in particles:
        if hasattr(p, "x"):
            x, theta_x, theta_y = p.x, p.theta_x, p.theta_y
        else:
            x, theta_x, theta_y = p
        if x > 1e-10:
            final.append((x, theta_x, theta_y))
    eec_rl = []
    eec_w = []
    if len(final) >= 2:
        for i in range(len(final)):
            xi, txi, tyi = final[i]
            for j in range(i + 1, len(final)):
                xj, txj, tyj = final[j]
                rl = np.sqrt((txi - txj) ** 2 + (tyi - tyj) ** 2)
                if rl < 1e-15:
                    continue
                eec_rl.append(rl)
                eec_w.append(2.0 * xi * xj * e0 ** 2)
    return np.array(eec_rl, dtype=np.float64), np.array(eec_w, dtype=np.float64)


def collect_eec_data(results, n_fail):
    if not results:
        raise RuntimeError("No valid EEC events.")

    rl_parts = [r[3] for r in results if len(r[3]) > 0]
    w_parts = [r[4] for r in results if len(r[4]) > 0]
    eec_rl = np.concatenate(rl_parts) if rl_parts else np.array([], dtype=np.float64)
    eec_w = np.concatenate(w_parts) if w_parts else np.array([], dtype=np.float64)

    n_evt = len(results)
    print(f"Valid events: {n_evt}, dropped: {n_fail}, EEC points: {len(eec_rl)}")
    return eec_rl, eec_w, n_evt


def _metadata_label(metadata, data_path, e0):
    m_q = metadata.get("M_Q")
    if m_q is not None:
        return rf"$M_Q={float(m_q):g}$"
    return rf"$E_0={e0:g}$, {os.path.basename(data_path)}"


def _sort_dataset_key(dataset):
    metadata = dataset["metadata"]

    def _float_or_default(key, default):
        try:
            return float(metadata[key])
        except (KeyError, TypeError, ValueError):
            return default

    return (
        _float_or_default("E0", float("inf")),
        _float_or_default("M_Q", float("inf")),
        _float_or_default("LAMBDA_D", float("inf")),
        os.path.basename(dataset["path"]),
    )


def plot_eec(datasets, plot_path):
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    plotted = False

    rl_limits = []
    for dataset in datasets:
        eec_rl = dataset["eec_rl"]
        if len(eec_rl) > 0:
            rl_pos = eec_rl[eec_rl > 1e-10]
            if len(rl_pos) > 0:
                rl_limits.append((float(rl_pos.min()), float(rl_pos.max())))

    if rl_limits:
        rl_min = min(item[0] for item in rl_limits)
        rl_max = max(item[1] for item in rl_limits)
        if rl_max > rl_min:
            log_bins = np.logspace(np.log10(rl_min), np.log10(rl_max), 25)
            bin_c = np.sqrt(log_bins[:-1] * log_bins[1:])
            bin_w = log_bins[1:] - log_bins[:-1]
            colors = plt.cm.tab10(
                np.linspace(0, 1, max(len(datasets), 2))
            )[:len(datasets)]
            markers = ("o", "s", "^", "D", "v", "P", "X")

            for idx, (color, dataset) in enumerate(zip(colors, datasets)):
                eec_rl = dataset["eec_rl"]
                eec_w = dataset["eec_w"]
                if len(eec_rl) == 0 or len(eec_w) == 0 or dataset["n_evt"] <= 0:
                    continue
                mask = eec_rl > 1e-10
                rl_pos = eec_rl[mask]
                w_pos = eec_w[mask]
                if len(rl_pos) <= 1:
                    continue
                hist, _edges = np.histogram(rl_pos, bins=log_bins, weights=w_pos)
                density = hist / (dataset["n_evt"] * bin_w)
                valid = density > 0
                if valid.any():
                    ax.loglog(
                        bin_c[valid],
                        density[valid],
                        marker=markers[idx % len(markers)],
                        color=color,
                        lw=1.5,
                        markersize=3,
                        label=_metadata_label(
                            dataset["metadata"], dataset["path"], dataset["e0"]
                        ),
                    )
                    plotted = True

    ax.set_xlabel(r"$R_L$")
    ax.set_ylabel(r"$d\Sigma/dR_L$")
    ax.set_title(r"QVPS EEC with only $M_Q$ varied")
    if plotted:
        ax.legend(fontsize=18, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    output_dir = os.path.dirname(os.path.abspath(plot_path))
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"EEC plot saved: {plot_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=(
            "Group all completed QVPS experiments by parameters other than M_Q "
            "and plot eligible M_Q scans."
        )
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help=(
            "Optional experiment hash or full experiment-<hash> ID. When omitted "
            "and --input is not used, all experiments with a completed QVPS run "
            "are scanned."
        ),
    )
    parser.add_argument(
        "--input", "--input-csv", dest="inputs", nargs="+", default=None,
        help="Explicit Parquet path(s); legacy CSV is also accepted.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help=(
            "Output PDF filename inside fig. Multiple eligible groups append "
            "their parameter-group hash."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Compute EEC without writing a plot.",
    )
    parser.add_argument(
        "--e0",
        type=float,
        default=None,
        help="Initial absolute energy to use if input metadata has no E0.",
    )
    args = parser.parse_args()

    try:
        if args.inputs:
            input_paths = [_resolve_path(path, base_dir) for path in args.inputs]
        elif args.experiment:
            input_path, _experiment_dir = discover_mode_for_experiment(
                Path(base_dir), args.experiment, "qvps"
            )
            input_paths = [input_path]
        else:
            input_paths = discover_latest_runs_by_experiment(Path(base_dir), "qvps")
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    groups, rejected = group_eec_inputs(input_paths, args.e0)
    print(
        f"Scanned QVPS datasets: {len(input_paths)}; "
        f"parameter groups: {len(groups)}; rejected inputs: {len(rejected)}"
    )
    for data_path, reason in rejected:
        print(f"Skipping unreadable input {data_path}: {reason}")

    eligible_groups = [
        group for group in groups
        if group["distinct_m_q"] >= MIN_MQ_DATASETS
    ]
    ineligible_groups = [
        group for group in groups
        if group["distinct_m_q"] < MIN_MQ_DATASETS
    ]
    if ineligible_groups:
        print(
            f"Skipping {len(ineligible_groups)} parameter group(s) with fewer than "
            f"{MIN_MQ_DATASETS} distinct M_Q datasets."
        )

    if not eligible_groups:
        print(
            "No parameter group contains at least "
            f"{MIN_MQ_DATASETS} datasets with distinct M_Q values; no PDF written."
        )
        return

    for group in eligible_groups:
        unique_inputs = []
        seen_m_q = set()
        for item in group["inputs"]:
            if item["m_q"] in seen_m_q:
                print(
                    f"Skipping duplicate M_Q={item['m_q']:g} input in parameter "
                    f"group {group['group_id']}: {item['path']}"
                )
                continue
            seen_m_q.add(item["m_q"])
            unique_inputs.append(item)

        m_q_values = ", ".join(f"{item['m_q']:g}" for item in unique_inputs)
        print(
            f"Processing parameter group {group['group_id']} with "
            f"M_Q values: {m_q_values}"
        )
        datasets = []
        for item in unique_inputs:
            input_path = item["path"]
            print(f"Reading QVPS event data: {input_path}")
            results, n_fail, _wall_time, e0, metadata = load_eec_events(
                input_path, args.e0
            )
            print(f"Using absolute initial energy E0 = {e0:g}")
            eec_rl, eec_w, n_evt = collect_eec_data(results, n_fail)
            datasets.append({
                "path": input_path,
                "metadata": metadata,
                "e0": e0,
                "eec_rl": eec_rl,
                "eec_w": eec_w,
                "n_evt": n_evt,
            })
        datasets.sort(key=_sort_dataset_key)

        if not args.no_plot:
            plot_path = _group_plot_path(
                args.plot, base_dir, group["group_id"], len(eligible_groups)
            )
            plot_eec(datasets, plot_path)


if __name__ == "__main__":
    main()
