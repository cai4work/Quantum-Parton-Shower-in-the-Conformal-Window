"""Shared flavor-basis mixing matrices for the quantum and classical showers."""

from functools import lru_cache

import numpy as np
from scipy.linalg import expm, logm


def _validate_mixing_strength(mixing_strength):
    strength = float(mixing_strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError(
            "MIXING_STRENGTH must satisfy 0 <= value <= 1, got "
            f"{mixing_strength!r}."
        )
    return strength


@lru_cache(maxsize=None)
def _cached_dft_geodesic(n_flavor, mixing_strength):
    """Return the unitary geodesic from identity to the normalized DFT."""
    n_flavor = int(n_flavor)
    if n_flavor < 1:
        raise ValueError(f"n_flavor must be positive, got {n_flavor}.")

    strength = _validate_mixing_strength(mixing_strength)
    identity = np.eye(n_flavor, dtype=complex)
    if strength == 0.0:
        identity.setflags(write=False)
        return identity

    indices = np.arange(n_flavor)
    dft = np.exp(
        2j * np.pi * np.outer(indices, indices) / n_flavor
    ) / np.sqrt(n_flavor)
    if strength == 1.0:
        dft.setflags(write=False)
        return dft

    # If DFT = exp(i H), exp(i strength H) remains unitary for every strength.
    # Hermitian symmetrization removes only floating-point noise from logm.
    generator = -1j * logm(dft)
    generator = 0.5 * (generator + generator.conj().T)
    unitary = expm(1j * strength * generator)
    unitary.setflags(write=False)
    return unitary


def build_flavor_mixing_matrix(n_flavor, mixing_strength):
    """Build U(strength): identity at 0 and normalized DFT at 1."""
    strength = _validate_mixing_strength(mixing_strength)
    return _cached_dft_geodesic(int(n_flavor), strength).copy()
