"""Shared parameter settings for the CW QVPS and Classic simulations."""

import os


def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value is not None else default


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value is not None else default


PARAMS = {
    # Gauge group parameters
    "N_C": 3,
    "N_F": 12,

    # U(1)_D dark photon parameters
    "N_CHARGE": 12,
    "ALPHA_D": 0.2,
    "Q_MIN": _env_float("QVPS_Q_MIN", 0.5),
    "Q_MAX": _env_float("QVPS_Q_MAX", 2.0),

    # Energy scales
    "t2_0": 100,
    "E0": 50,
    "LAMBDA_D": 0.2,
    "M_Q": _env_float("QVPS_M_Q", 0.005),

    # Evolution step parameters
    "R_STEP": 0.995,
    # Optional nominal endpoint for the evolution variable squared.  When set,
    # each simulation entry point derives MAX_STEPS from t2_0, R_STEP, and this
    # value.  None preserves the explicit MAX_STEPS fallback below.
    "T2_END": _env_float("QVPS_T2_END", 0.0001),
    "OVERESTIMATE_FACTOR": 1.3,

    # Qubit encoding
    "QUBITS_PER_FERMION": 4,
    # Unitary interpolation: 0 gives identity and 1 gives the normalized DFT.
    "MIXING_STRENGTH": _env_float("QVPS_MIXING_STRENGTH", 1.0),

    # Classical flavor algorithm. ``paper_physical_markov`` reproduces the
    # conventional physical-flavor shower used as the classical baseline in
    # arXiv:2204.12500. ``interaction_eigenstate_mixture`` keeps the previous
    # behavior, which samples one interaction eigenstate per event.
    "CLASSICAL_FLAVOR_SCHEME": "paper_physical_markov",
    "INITIAL_PHYSICAL_FLAVOR": 0,

    # Splitting channel switches
    "ENABLE_Q_TO_Q_G": True,
    "ENABLE_Q_TO_Q_DP": True,
    "ENABLE_G_TO_GG": True,
    "ENABLE_G_TO_QQ": False,
    "ENABLE_DP_TO_QQ": False,

    # Shower cutoff parameters
    "MAX_STEPS": 10000,
    "MAX_PAIRS": 3,

    # Statistics run parameters
    "N_SHOTS": _env_int("QVPS_N_SHOTS", 100),
    "N_WORKERS": max(1, os.cpu_count() // 2),
    "PROGRESS_INTERVAL": 500,

    # Reproducibility and structured Parquet output settings.
    "RANDOM_SEED": None,
    "RESULTS_DIR": None,
}
