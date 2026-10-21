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
count, and output location.

```
python simulation_api.py --method qvps
python simulation_api.py --method classic
```

`--method both`runs Classic followed by QVPS and is the default method.

## Files

- `Parameters.py` — shared simulation settings.
- `simulation_api.py` — command-line entry point, shared execution API, result storage and readers, and experiment index maintenance.
- `QVPS_CW.py` — quantum-veto shower.
- `Classic.py` — classical baseline.
- `FlavorMixing.py` — flavor-basis mixing utilities.

## Results

Runs are saved under `results/experiment-*/{qvps,classic}/run-*/`. Each run
contains `events.parquet`, `particles.parquet`, `emissions.parquet`, and
`summary.json`.

## Citation

```bibtex
@article{Cai:2026qod,
    author = {Cai, Nongqiu and Deng, Xu and Li, Jinmian and Long, Bingwei and Yang, Shuo},
    title = {Quantum Veto Parton Shower Algorithm in the Conformal Window},
    eprint = {2610.25189},
    archivePrefix = {arXiv},
    primaryClass = {hep-ph},
    month = {10},
    year = {2026},
    url = {https://arxiv.org/abs/2610.25189}
}
```

