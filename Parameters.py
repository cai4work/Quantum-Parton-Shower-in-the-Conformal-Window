"""Default baseline parameters only; execution lives in simulation_api.py."""

BASELINE_PARAMETERS = {
    # Gauge group parameters
    "N_C": 3,
    "N_F": 12,

    # U(1)_D dark photon parameters
    "N_CHARGE": 12,
    "ALPHA_D": 0.2,
    "Q_MIN": 0.5,
    "Q_MAX": 2.0,

    # Energy scales
    "t2_0": 100,
    "E0": 50,
    "LAMBDA_D": 0.2,
    "M_Q": 0.005,

    # Evolution step parameters
    "R_STEP": 0.995,
    # Upper bound; the physical mass cutoff can stop evolution earlier.
    "MAX_STEPS": 10000,
    "OVERESTIMATE_FACTOR": 1.3,

    # Qubit encoding
    "QUBITS_PER_FERMION": 4,
    # Unitary interpolation: 0 gives identity and 1 gives the normalized DFT.
    "MIXING_STRENGTH": 1.0,

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
    "MAX_PAIRS": 3,

    # Statistics run parameters
    "N_SHOTS": 100000,
    "N_WORKERS": None,
    "PROGRESS_INTERVAL": 500,

    # Reproducibility and structured Parquet output settings.
    "RANDOM_SEED": None,
    "RESULTS_DIR": None,
}
