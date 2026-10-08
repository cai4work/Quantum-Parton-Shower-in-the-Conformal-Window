# Quantum Parton Shower in the Conformal Window

This repository includes two dark-sector parton showers in the conformal
window. `QVPS_CW.py` retains coherent flavor evolution in a Qiskit statevector
simulation; `Classic.py` provides a classical flavor evolution baseline. Both
use the same physics settings and save structured results for comparison.

## Requirements

Use Python 3.10 and install the dependencies:

python -m pip install -r requirements.txt
## Run

Edit `Parameters.py` to set the physics parameters, event count, worker
count, and output location. Its current defaults generate 100 events and evolve
from 100 GeV^2 toward 0.0001 GeV^2.

python QVPS_CW.py
python Classic.py
The two runs use the same experiment directory when their experiment settings,
including `N_SHOTS`, match. Statevector runs can require substantial memory and
time; choose `N_WORKERS` to fit the machine before increasing `N_SHOTS`.

## Files

- `Parameters.py` — shared simulation settings.
- `QVPS_CW.py` — quantum-veto shower.
- `Classic.py` — classical baseline.
- `FlavorMixing.py` — flavor-basis mixing utilities.
- `result_storage.py` — experiment directories, simulated data, and summaries.
- `eec.py` — EEC analysis for QVPS mass scans.
- `rebuild_experiments_index.py` — rebuild `experiments.csv` from experiment directories.

## Results

Runs are saved under `results/experiment-*/{qvps,classic}/run-*/`. Each run
contains `events.parquet`, `particles.parquet`, `emissions.parquet`, and
`summary.json`.

## References

- [QVPS in Conformal Window]
