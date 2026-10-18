"""
SU(3) x U(1)_D quantum parton shower with 12 charges and finite quark mass.
Based on arXiv:2310.19881 (QVPS) and arXiv:2502.18566 (conformal-window shower).
"""

import math
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.special import lambertw

from FlavorMixing import build_flavor_mixing_matrix


@dataclass
class EventResult:
    """One quantum event; individual particles' flavor amplitudes are not measured."""

    n_ph: int
    n_gl: int
    mass_state: int
    final_particles: list[tuple[int, str, float, float, float]]
    final_emissions: list[tuple[int, str, int, str, float, float, float]]
    final_flavors: list[tuple[int, int]] | None = None


# --- 1. Kinematics, couplings, and proposal overestimates ---

class CWKinematics12:
    """
    Kinematics and quantum-veto inputs for the conformal-window shower.

    Provides two-loop SU(3) running, charge-dependent U(1)_D couplings,
    splitting kernels, proposal densities, and the flavor mixing matrix.
    The evolution variable t2 denotes tilde q^2, not invariant mass squared.
    """

    def __init__(self, parameters):
        self.params = dict(parameters)
        N_C = self.params["N_C"]
        N_F = self.params["N_F"]

        self.N_C = N_C
        self.N_F = N_F
        self.N_CHARGE = self.params["N_CHARGE"]
        self.t2_0 = self.params["t2_0"]
        self.lambda_d = self.params["LAMBDA_D"]
        self.alpha_D = self.params["ALPHA_D"]
        self.r_step = self.params["R_STEP"]

        self.LAMBDA2 = self.lambda_d ** 2
        self.neg_ln_r = -np.log(self.r_step)

        self.M_q = self.params["M_Q"]
        self.MQ2 = self.M_q ** 2
        self.t2_MIN = 4.0 * self.MQ2

        # SU(3) group factors.
        self.C_A = float(N_C)
        self.C_F = (N_C ** 2 - 1) / (2.0 * N_C)
        self.T_F = 0.5

        # Two-loop beta-function coefficients.
        self.beta_0 = (
            11 * self.C_A / 3 - 4 * self.T_F * N_F / 3
        ) / (4 * np.pi)
        self.beta_1 = (
            34 * self.C_A ** 2 / 3
            - (20 * self.C_A / 3 + 4 * self.C_F) * self.T_F * N_F
        ) / (4 * np.pi) ** 2

        self.alpha_star = -self.beta_0 / self.beta_1
        self.gamma_anom = -self.beta_0 ** 2 / self.beta_1

        # U(1)_D charges in the interaction eigenbasis.
        self.Q_charges = np.linspace(
            self.params["Q_MIN"], self.params["Q_MAX"], self.N_CHARGE
        )
        self.q2_charges = self.Q_charges ** 2
        self.q2_sum = float(np.sum(self.q2_charges))
        self.q2_max = float(np.max(self.q2_charges))

        # Coupling bounds used by the proposal densities.
        ovf = self.params["OVERESTIMATE_FACTOR"]
        self.ALPHA_S_OVER = self.alpha_star * ovf
        self.ALPHA_D_OVER = self.alpha_D * self.q2_max * ovf
        self.ALPHA_D_SUM_OVER = self.alpha_D * self.q2_sum * ovf

        # Map the interaction eigenbasis to the physical/mass basis.
        N = self.N_CHARGE
        self.mixing_strength = float(self.params["MIXING_STRENGTH"])
        self.mixing_matrix = build_flavor_mixing_matrix(
            N, self.mixing_strength
        )

        # Embed into the 16-dimensional four-qubit flavor register.
        self.mixing_unitary_16 = np.eye(16, dtype=complex)
        self.mixing_unitary_16[:N, :N] = self.mixing_matrix

    # --- Couplings ---

    def alpha_s(self, mu2):
        """Return the two-loop SU(3) coupling alpha_s(mu2) via Lambert W."""
        if mu2 <= 0.0:
            return self.alpha_star
        v = (1.0 / np.e) * (mu2 / self.LAMBDA2) ** self.gamma_anom
        W = float(np.real(lambertw(v, k=0)))
        return max(self.alpha_star / (1.0 + W), 0.0)

    def alpha_d_charge(self, charge_idx):
        """Return the fixed U(1)_D effective coupling alpha_D * Q_k^2."""
        return self.alpha_D * self.q2_charges[charge_idx]

    def alpha_d_sum(self):
        """Sum the fixed U(1)_D couplings over all charge eigenstates."""
        return self.alpha_D * self.q2_sum

    # --- Renormalization scales ---

    def mu2_chi_to_chi_g(self, t2, z):
        """Scale for chi -> chi g; chi -> chi gamma_D uses the same prescription."""
        return z * (1.0 - z) ** 2 * t2

    def mu2_g_to_chi_chibar(self, t2, z):
        """Scale for g -> chi chibar; gamma_D -> chi chibar uses the same prescription."""
        return z ** 2 * (1.0 - z) ** 2 * t2

    def mu2_g_to_gg(self, t2, z):
        """g -> gg scale: k_perp^2 with k_perp=z(1-z)sqrt(t2)."""
        return z ** 2 * (1.0 - z) ** 2 * t2

    # --- Altarelli-Parisi splitting kernels ---

    def p_ff(self, z):
        return (1.0 + z ** 2) / (1.0 - z)

    def p_fA(self, z):
        return z ** 2 + (1.0 - z) ** 2

    def p_gg(self, z):
        return z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z)

    # --- Kinematic limits on z ---

    def z_limits(self, t2):
        """Return the implemented mass-cutoff bounds for the energy fraction z."""
        if t2 <= self.t2_MIN:
            return 0.5, 0.5
        z_min = np.sqrt(self.MQ2 / t2)
        return z_min, 1.0 - z_min

    # --- Proposal overestimates: densities at z and their z integrals ---

    def gover_chi_g_at_z(self, t2, z):
        return (
            self.neg_ln_r * self.ALPHA_S_OVER * self.C_F
            / (2 * np.pi) * 2.0 / (1.0 - z)
        )

    def gover_chi_g_tot(self, t2):
        z_min, z_max = self.z_limits(t2)
        if z_min >= z_max:
            return 0.0
        return (
            self.neg_ln_r * self.ALPHA_S_OVER * self.C_F
            / np.pi * np.log((1 - z_min) / z_min)
        )

    def gover_chi_gd_at_z(self, t2, z):
        return (
            self.neg_ln_r * self.ALPHA_D_OVER
            / (2 * np.pi) * 2.0 / (1.0 - z)
        )

    def gover_chi_gd_tot(self, t2):
        z_min, z_max = self.z_limits(t2)
        if z_min >= z_max:
            return 0.0
        return (
            self.neg_ln_r * self.ALPHA_D_OVER
            / np.pi * np.log((1 - z_min) / z_min)
        )

    def gover_gg_at_z(self, t2, z):
        return (
            self.neg_ln_r * self.ALPHA_S_OVER * self.C_A
            / np.pi * (1.0 / z + 1.0 / (1 - z))
        )

    def gover_gg_tot(self, t2):
        z_min, z_max = self.z_limits(t2)
        if z_min >= z_max:
            return 0.0
        return (
            self.neg_ln_r * self.ALPHA_S_OVER * self.C_A
            / np.pi * 2.0 * np.log((1 - z_min) / z_min)
        )

    def gover_g_qq_at_z(self, t2, z):
        return self.neg_ln_r * self.ALPHA_S_OVER * self.N_F * self.T_F / np.pi

    def gover_g_qq_tot(self, t2):
        z_min, z_max = self.z_limits(t2)
        if z_min >= z_max:
            return 0.0
        return (
            self.neg_ln_r * self.ALPHA_S_OVER * self.N_F * self.T_F
            / np.pi * (z_max - z_min)
        )

    def gover_gd_qq_at_z(self, t2, z):
        return self.neg_ln_r * self.ALPHA_D_SUM_OVER / np.pi

    def gover_gd_qq_tot(self, t2):
        z_min, z_max = self.z_limits(t2)
        if z_min >= z_max:
            return 0.0
        return self.neg_ln_r * self.ALPHA_D_SUM_OVER / np.pi * (z_max - z_min)

    # --- Sampling z from proposal densities ---

    def sample_z_pole(self, t2):
        """Sample z from the single-pole density proportional to 1 / (1 - z)."""
        z_min, z_max = self.z_limits(t2)
        if z_min >= z_max:
            return 0.5
        u = np.random.uniform()
        z = 1.0 - (1.0 - z_min) ** (1 - u) * z_min ** u
        return float(np.clip(z, z_min + 1e-10, z_max - 1e-10))

    def sample_z_gg(self, t2):
        """Sample z from the two-pole density 1 / z + 1 / (1 - z)."""
        z_min, z_max = self.z_limits(t2)
        if z_min >= z_max:
            return 0.5
        u = np.random.uniform()
        if np.random.uniform() < 0.5:
            z = 1.0 - (1.0 - z_min) ** (1 - u) * z_min ** u
        else:
            z = z_min ** (1 - u) * (1.0 - z_min) ** u
        return float(np.clip(z, z_min + 1e-10, z_max - 1e-10))

    def sample_z_flat(self, t2):
        """Sample z uniformly within the current kinematic bounds."""
        z_min, z_max = self.z_limits(t2)
        if z_min >= z_max:
            return 0.5
        return float(np.random.uniform(z_min, z_max))

    # --- Quantum-veto rotation angles ---

    @staticmethod
    def theta_hat(x):
        """Map acceptance probability x to an RY angle with sin^2(theta/2) = x."""
        x = float(np.clip(x, 0.0, 1.0 - 1e-12))
        return 2.0 * np.arctan(np.sqrt(x / (1.0 - x)))

    def angle_chi_gluon(self, t2, z, g_tot):
        """Build the flavor-independent veto angle using total proposal weight g_tot."""
        mu2_chi_to_chi_g = self.mu2_chi_to_chi_g(t2, z)
        g_true = (
            self.neg_ln_r * self.alpha_s(mu2_chi_to_chi_g) * self.C_F
            / (2 * np.pi) * self.p_ff(z)
        )
        denom = max(self.gover_chi_g_at_z(t2, z) / g_tot, 1e-15)
        return self.theta_hat(g_true / denom)

    def angles_chi_photon(self, t2, z, g_tot):
        """Return one chi -> chi gamma_D veto angle per charge eigenstate.

        g_tot is the sum of proposal weights over all active particles.
        """
        denom = max(self.gover_chi_gd_at_z(t2, z) / g_tot, 1e-15)
        thetas = np.empty(self.N_CHARGE)
        for k in range(self.N_CHARGE):
            g_k = (self.neg_ln_r * self.alpha_d_charge(k)
                   / (2 * np.pi) * self.p_ff(z))
            thetas[k] = self.theta_hat(g_k / denom)
        return thetas

    def angle_gg(self, t2, z, g_tot):
        mu2_g_to_gg = self.mu2_g_to_gg(t2, z)
        g_true = (
            self.neg_ln_r * self.alpha_s(mu2_g_to_gg) * self.C_A
            / (2 * np.pi) * self.p_gg(z)
        )
        denom = max(self.gover_gg_at_z(t2, z) / g_tot, 1e-15)
        return self.theta_hat(g_true / denom)

    def angle_g_qq(self, t2, z, g_tot):
        mu2_g_to_chi_chibar = self.mu2_g_to_chi_chibar(t2, z)
        g_true = (
            self.neg_ln_r * self.N_F * self.alpha_s(mu2_g_to_chi_chibar)
            * self.T_F / (2 * np.pi) * self.p_fA(z)
        )
        denom = max(self.gover_g_qq_at_z(t2, z) / g_tot, 1e-15)
        return self.theta_hat(g_true / denom)

    def angle_gd_qq(self, t2, z, g_tot):
        g_true = (self.neg_ln_r * self.alpha_d_sum()
                  / (2 * np.pi) * self.p_fA(z))
        denom = max(self.gover_gd_qq_at_z(t2, z) / g_tot, 1e-15)
        return self.theta_hat(g_true / denom)

    # --- Entangled flavor coefficients for boson-to-fermion-pair splittings ---

    def entangle_phi_gluon(self):
        return np.pi / 2.0

    def entangle_coeffs_photon(self):
        """Return signed pair amplitudes c_k = Q_k / sqrt(sum_j Q_j^2)."""
        norm = np.sqrt(self.q2_sum)
        return self.Q_charges / norm


# --- 2. Quantum-state management with Qiskit Statevector ---

class QuantumStateManager12:
    """Manage flavor registers and one reusable emission qubit.

    Each fermion occupies QPF consecutive qubits starting at qubit_start.
    The emission qubit follows all flavor registers, at n_flavor_qubits.
    Statevector indices use Qiskit's least-significant-qubit convention.
    """

    def __init__(self, parameters, initial_flavor_state=None):
        self.params = dict(parameters)
        self.QPF = int(self.params["QUBITS_PER_FERMION"])
        n_charge = self.params["N_CHARGE"]
        self.n_flavor_qubits = self.QPF

        if initial_flavor_state is None:
            mixing_matrix = build_flavor_mixing_matrix(
                n_charge, self.params["MIXING_STRENGTH"]
            )
            initial_physical_flavor = int(
                self.params.get("INITIAL_PHYSICAL_FLAVOR", 0)
            )
            if not 0 <= initial_physical_flavor < n_charge:
                raise ValueError(
                    "INITIAL_PHYSICAL_FLAVOR must satisfy "
                    f"0 <= flavor < {n_charge}"
                )
            initial_flavor_state = (
                mixing_matrix.conj().T[:, initial_physical_flavor]
            )

        initial_flavor_state = np.asarray(initial_flavor_state, dtype=complex)
        if initial_flavor_state.shape != (n_charge,):
            raise ValueError(
                "initial_flavor_state must have shape "
                f"({n_charge},), got {initial_flavor_state.shape}"
            )
        norm = float(np.linalg.norm(initial_flavor_state))
        if not np.isfinite(norm) or norm < 1e-15:
            raise ValueError("initial_flavor_state must have a finite nonzero norm")

        dim = 2 ** (self.n_flavor_qubits + 1)
        self.state_data = np.zeros(dim, dtype=complex)
        self.state_data[:n_charge] = initial_flavor_state / norm

    def emission_idx(self):
        """Return the emission-qubit index after the current flavor registers."""
        return self.n_flavor_qubits

    def sv_evolve(self, data, qc):
        norm = float(np.linalg.norm(data))
        if norm > 1e-12:
            data = data / norm
        return np.array(Statevector(data).evolve(qc).data)

    def sv_measure_reset_emission(self, data, e_idx):
        """Measure the emission qubit, project the state, and reset it to zero.

        Return (outcome, projected_state); the reset preserves the flavor bits.
        """
        dim = len(data)
        idx = np.arange(dim)
        e_bits = (idx >> e_idx) & 1

        p1 = float(np.clip(np.sum(np.abs(data) ** 2 * e_bits), 0.0, 1.0))
        e_result = int(np.random.choice([0, 1], p=[1.0 - p1, p1]))

        new_data = data * (e_bits == e_result)
        norm = float(np.linalg.norm(new_data))
        if norm > 1e-12:
            new_data = new_data / norm

        if e_result == 1:
            new_data = new_data[idx ^ (1 << e_idx)]

        return e_result, new_data

    # Flavor-independent veto: apply RY(theta) to the emission qubit.

    def try_boson_veto(self, theta):
        e_idx = self.emission_idx()
        qc = QuantumCircuit(self.n_flavor_qubits + 1)
        qc.ry(theta, e_idx)

        self.state_data = self.sv_evolve(self.state_data, qc)
        outcome, self.state_data = self.sv_measure_reset_emission(
            self.state_data, e_idx
        )
        return outcome

    # Flavor-dependent veto: rotate the emission qubit conditional on flavor.

    @staticmethod
    def _build_flavor_veto_unitary(thetas, n_charge):
        """Build a 32x32 unitary applying RY(thetas[k]) for flavor k.

        The local basis contains four flavor qubits and one emission qubit.
        Indices k and k + 16 share flavor k and differ in the emission bit.
        """
        U = np.eye(32, dtype=complex)
        for k in range(n_charge):
            c = np.cos(thetas[k] / 2.0)
            s = np.sin(thetas[k] / 2.0)
            i0 = k
            i1 = k + 16
            U[i0, i0] = c
            U[i0, i1] = -s
            U[i1, i0] = s
            U[i1, i1] = c
        return U

    def try_fermion_veto_multi(self, fermion_start, thetas):
        e_idx = self.emission_idx()
        n_total = self.n_flavor_qubits + 1
        qc = QuantumCircuit(n_total)

        U_veto = self._build_flavor_veto_unitary(thetas, self.params["N_CHARGE"])

        target_qubits = list(range(fermion_start, fermion_start + self.QPF))
        target_qubits.append(e_idx)
        qc.unitary(U_veto, target_qubits, label="FlavorVeto")

        self.state_data = self.sv_evolve(self.state_data, qc)
        outcome, self.state_data = self.sv_measure_reset_emission(
            self.state_data, e_idx
        )
        return outcome

    # Expand the statevector to add an entangled fermion pair.

    def sv_expand_with_fermion_pair_12(self, data, n_flavor_qubits, coeffs):
        """Insert two flavor registers in sum_k coeffs[k] |k>_1 |k>_2.

        Existing flavor bits keep their positions; the emission bit moves
        above the two new registers.
        """
        n_new = 2 * self.QPF
        old_dim = len(data)
        new_dim = old_dim * (2 ** n_new)
        new_data = np.zeros(new_dim, dtype=complex)

        fmask = (1 << n_flavor_qubits) - 1
        n_charge = self.params["N_CHARGE"]

        for i in range(old_dim):
            amp = data[i]
            if abs(amp) < 1e-15:
                continue
            fv = i & fmask
            ev = (i >> n_flavor_qubits) & 1
            ec = ev << (n_flavor_qubits + n_new)

            for k in range(n_charge):
                pair_bits = (
                    (k << n_flavor_qubits)
                    | (k << (n_flavor_qubits + self.QPF))
                )
                new_data[fv | pair_bits | ec] += coeffs[k] * amp

        return new_data

    # Pair-production interfaces.

    def apply_boson_split_gluon(self):
        n_charge = self.params["N_CHARGE"]
        coeffs = np.ones(n_charge) / np.sqrt(n_charge)
        return self._do_pair_production(coeffs)

    def apply_boson_split_photon(self, coeffs):
        return self._do_pair_production(coeffs)

    def _do_pair_production(self, coeffs):
        old_n_flavor_qubits = self.n_flavor_qubits
        self.state_data = self.sv_expand_with_fermion_pair_12(
            self.state_data, old_n_flavor_qubits, coeffs
        )
        self.n_flavor_qubits += 2 * self.QPF
        return old_n_flavor_qubits, old_n_flavor_qubits + self.QPF

    # Final measurement in the physical/mass basis.

    def measure_q0_mass_basis(self, mixing_unitary_16):
        """Rotate the initial fermion's register and sample its physical flavor."""
        n_total = self.n_flavor_qubits + 1
        qc = QuantumCircuit(n_total)
        qc.unitary(mixing_unitary_16, [0, 1, 2, 3], label="Mixing12")

        rotated = self.sv_evolve(self.state_data, qc)

        N = self.params["N_CHARGE"]
        flavor_mask = (1 << self.QPF) - 1
        idx = np.arange(len(rotated))
        probs = np.zeros(N)
        for m in range(N):
            mask = (idx & flavor_mask) == m
            probs[m] = float(np.sum(np.abs(rotated[mask]) ** 2))

        total = probs.sum()
        if total > 1e-12:
            probs /= total
        else:
            probs = np.ones(N) / N

        return int(np.random.choice(N, p=probs))


# --- 3. Particle ---

class Particle:
    """Classical kinematic record linked to a quantum flavor register.

    t2 is the angular evolution variable tilde q^2. The energy fraction x
    gives E = x * E0. theta_x and theta_y are transverse angular components
    relative to the initial axis. qubit_start locates a chi flavor register;
    bosons have no flavor register.
    """
    _LABELS = {"chi": "χ", "g": "g", "gamma_d": "γ_D"}

    def __init__(self, pid, ptype, t2, qubit_start=None, energy_fraction=1.0,
                 theta_x=0.0, theta_y=0.0):
        self.id = pid
        self.type = ptype
        self.t2 = t2
        self.qubit_start = qubit_start
        self.x = energy_fraction
        self.theta_x = theta_x
        self.theta_y = theta_y

    def __repr__(self):
        return f"[{self._LABELS.get(self.type, self.type)}-{self.id} E={self.x:.3f}]"


# --- 4. Quantum shower: candidate selection, veto, and accepted splittings ---

class CWShower12:
    """Evolve one event using proposal weights and quantum-veto measurements."""

    def __init__(self, parameters):
        self.params = dict(parameters)
        self.kin = CWKinematics12(self.params)
        self.initial_physical_flavor = int(
            self.params.get("INITIAL_PHYSICAL_FLAVOR", 0)
        )
        if not 0 <= self.initial_physical_flavor < self.kin.N_CHARGE:
            raise ValueError(
                "INITIAL_PHYSICAL_FLAVOR must satisfy "
                f"0 <= flavor < {self.kin.N_CHARGE}"
            )
        initial_interaction_state = self.kin.mixing_matrix.conj().T[
            :, self.initial_physical_flavor
        ]
        self.qsm = QuantumStateManager12(self.params, initial_interaction_state)
        self.next_id = 1
        self.emissions = []

        self.particles = [
            Particle(
                self.next_id,
                "chi",
                t2=self.kin.t2_0,
                qubit_start=0,
                energy_fraction=1.0,
            )
        ]
        self.next_id += 1

    def _particle_weight(self, p):
        """Sum the enabled channels' integrated proposal weights for particle p."""
        if p.type == "chi":
            return (
                (self.kin.gover_chi_g_tot(p.t2)
                 if self.params["ENABLE_Q_TO_Q_G"] else 0.0)
                + (self.kin.gover_chi_gd_tot(p.t2)
                   if self.params["ENABLE_Q_TO_Q_DP"] else 0.0)
            )
        elif p.type == "g":
            return (
                (self.kin.gover_gg_tot(p.t2)
                 if self.params["ENABLE_G_TO_GG"] else 0.0)
                + (self.kin.gover_g_qq_tot(p.t2)
                   if self.params["ENABLE_G_TO_QQ"] else 0.0)
            )
        elif p.type == "gamma_d":
            return (
                self.kin.gover_gd_qq_tot(p.t2)
                if self.params["ENABLE_DP_TO_QQ"] else 0.0
            )
        return 0.0

    def _split_2d(self, t2, z, x_parent):
        """Return (dx_ell, dy_ell, dx_m, dy_m) for a two-body splitting.

        The ell daughter carries energy fraction z; the m daughter carries
        1 - z. Offsets are added to the parent's angular components.
        With t2 = tilde q^2, the opening angle is sqrt(t2) / E_parent.
        """
        E_parent = x_parent * self.params["E0"]
        if E_parent < 1e-15:
            return 0.0, 0.0, 0.0, 0.0
        q = np.sqrt(max(t2, 0.0))
        phi = np.random.uniform(0, 2 * np.pi)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        d_ell = (1 - z) * q / E_parent
        d_m = z * q / E_parent
        return (
            d_ell * cos_phi,
            d_ell * sin_phi,
            -d_m * cos_phi,
            -d_m * sin_phi,
        )

    # --- Accepted-splitting kinematics and channel handling ---

    def _apply_radiation_split(self, candidate, daughter_type, z):
        """Update the retained particle and append its radiated daughter.

        The caller records the emission and advances next_id after this update.
        Flavor registers are unchanged by these radiation splittings.
        """
        parent_t2, parent_energy_fraction = candidate.t2, candidate.x
        dx_ell, dy_ell, dx_m, dy_m = self._split_2d(
            parent_t2, z, parent_energy_fraction
        )
        parent_theta_x, parent_theta_y = candidate.theta_x, candidate.theta_y
        candidate.t2 = z ** 2 * parent_t2
        candidate.x = parent_energy_fraction * z
        candidate.theta_x = parent_theta_x + dx_ell
        candidate.theta_y = parent_theta_y + dy_ell
        self.particles.append(
            Particle(
                self.next_id,
                daughter_type,
                t2=(1 - z) ** 2 * parent_t2,
                energy_fraction=parent_energy_fraction * (1 - z),
                theta_x=parent_theta_x + dx_m,
                theta_y=parent_theta_y + dy_m,
            )
        )

    def _record_emission(self, step_index, channel, parent_id, parent_type,
                         t2, z, parent_energy_fraction):
        """Append an accepted emission using the parent's pre-split kinematics."""
        self.emissions.append((
            step_index,
            channel,
            parent_id,
            parent_type,
            t2,
            z,
            parent_energy_fraction,
        ))

    def _handle_chi_split(self, candidate, total_proposal_weight, step_index):
        """Choose a chi radiation channel and apply its quantum veto."""
        kin = self.kin
        gluon_weight = (
            kin.gover_chi_g_tot(candidate.t2)
            if self.params["ENABLE_Q_TO_Q_G"] else 0.0
        )
        photon_weight = (
            kin.gover_chi_gd_tot(candidate.t2)
            if self.params["ENABLE_Q_TO_Q_DP"] else 0.0
        )
        channel_weight = gluon_weight + photon_weight
        if channel_weight < 1e-15:
            return None

        if np.random.uniform() < gluon_weight / channel_weight:
            # chi -> chi + g: the SU(3) veto is independent of flavor.
            z = kin.sample_z_pole(candidate.t2)
            theta = kin.angle_chi_gluon(candidate.t2, z, total_proposal_weight)
            outcome = self.qsm.try_boson_veto(theta)
            if outcome == 1:
                parent_t2, parent_energy_fraction = candidate.t2, candidate.x
                self._apply_radiation_split(candidate, "g", z)
                self._record_emission(
                    step_index, "chi_to_chi_g", candidate.id, "chi",
                    parent_t2, z, parent_energy_fraction,
                )
                self.next_id += 1
                return "gluon"
        else:
            # chi -> chi + gamma_D: the U(1)_D veto depends on flavor.
            z = kin.sample_z_pole(candidate.t2)
            thetas = kin.angles_chi_photon(candidate.t2, z, total_proposal_weight)
            outcome = self.qsm.try_fermion_veto_multi(
                candidate.qubit_start, thetas
            )
            if outcome == 1:
                parent_t2, parent_energy_fraction = candidate.t2, candidate.x
                self._apply_radiation_split(candidate, "gamma_d", z)
                self._record_emission(
                    step_index, "chi_to_chi_gamma_d", candidate.id, "chi",
                    parent_t2, z, parent_energy_fraction,
                )
                self.next_id += 1
                return "photon"

        return None

    def _handle_gluon_split(self, candidate, total_proposal_weight, step_index):
        """Choose g -> gg or g -> chi chi_bar and apply the quantum veto."""
        kin = self.kin
        gg_weight = (
            kin.gover_gg_tot(candidate.t2)
            if self.params["ENABLE_G_TO_GG"] else 0.0
        )
        pair_weight = (
            kin.gover_g_qq_tot(candidate.t2)
            if self.params["ENABLE_G_TO_QQ"] else 0.0
        )
        channel_weight = gg_weight + pair_weight
        if channel_weight < 1e-15:
            return None

        if np.random.uniform() < gg_weight / channel_weight:
            # g -> g + g.
            z = kin.sample_z_gg(candidate.t2)
            theta = kin.angle_gg(candidate.t2, z, total_proposal_weight)
            outcome = self.qsm.try_boson_veto(theta)
            if outcome == 1:
                parent_t2, parent_energy_fraction = candidate.t2, candidate.x
                self._apply_radiation_split(candidate, "g", z)
                self._record_emission(
                    step_index, "g_to_gg", candidate.id, "g",
                    parent_t2, z, parent_energy_fraction,
                )
                self.next_id += 1
                return "gg"
        else:
            if not self.params["ENABLE_G_TO_QQ"]:
                return None
            # g -> chi + chi_bar.
            z = kin.sample_z_flat(candidate.t2)
            theta = kin.angle_g_qq(candidate.t2, z, total_proposal_weight)
            outcome = self.qsm.try_boson_veto(theta)
            if outcome == 1:
                qubit1_start, qubit2_start = self.qsm.apply_boson_split_gluon()
                parent_t2, parent_energy_fraction = candidate.t2, candidate.x
                self.particles = [p for p in self.particles if p is not candidate]
                self.particles.append(
                    Particle(
                        self.next_id,
                        "chi",
                        t2=z ** 2 * parent_t2,
                        qubit_start=qubit1_start,
                        energy_fraction=parent_energy_fraction * z,
                    )
                )
                self.particles.append(
                    Particle(
                        self.next_id + 1,
                        "chi",
                        t2=(1 - z) ** 2 * parent_t2,
                        qubit_start=qubit2_start,
                        energy_fraction=parent_energy_fraction * (1 - z),
                    )
                )
                self._record_emission(
                    step_index, "g_to_chi_chi", candidate.id, "g",
                    parent_t2, z, parent_energy_fraction,
                )
                self.next_id += 2
                return "pair"

        return None

    def _handle_photon_split(self, candidate, total_proposal_weight, step_index):
        """Apply the gamma_D -> chi chi_bar quantum veto and pair-state update."""
        if not self.params["ENABLE_DP_TO_QQ"]:
            return None
        kin = self.kin
        z = kin.sample_z_flat(candidate.t2)
        theta = kin.angle_gd_qq(candidate.t2, z, total_proposal_weight)
        outcome = self.qsm.try_boson_veto(theta)

        if outcome == 1:
            coeffs = kin.entangle_coeffs_photon()
            qubit1_start, qubit2_start = self.qsm.apply_boson_split_photon(coeffs)
            parent_t2, parent_energy_fraction = candidate.t2, candidate.x
            self.particles = [p for p in self.particles if p is not candidate]
            self.particles.append(
                Particle(
                    self.next_id,
                    "chi",
                    t2=z ** 2 * parent_t2,
                    qubit_start=qubit1_start,
                    energy_fraction=parent_energy_fraction * z,
                )
            )
            self.particles.append(
                Particle(
                    self.next_id + 1,
                    "chi",
                    t2=(1 - z) ** 2 * parent_t2,
                    qubit_start=qubit2_start,
                    energy_fraction=parent_energy_fraction * (1 - z),
                )
            )
            self._record_emission(
                step_index, "gamma_d_to_chi_chi", candidate.id, "gamma_d",
                parent_t2, z, parent_energy_fraction,
            )
            self.next_id += 2
            return "pair"

        return None

    def run_shot(self, max_steps=None, max_pairs=None):
        """Return (n_photon, n_gluon, final_mass_state) for one event.

        The counters track accepted chi radiation into photons and gluons.
        final_mass_state measures the initial fermion in the physical basis.
        Return None if accepted pair production exceeds max_pairs.
        """
        if max_steps is None:
            max_steps = self.params["MAX_STEPS"]
        if max_pairs is None:
            max_pairs = self.params["MAX_PAIRS"]

        n_photon = 0
        n_gluon = 0
        n_pairs = 0
        kin = self.kin

        for step_index in range(1, max_steps + 1):
            # 1. Lower each active particle's scale and apply the mass cutoff.
            for p in self.particles:
                if p.t2 > kin.t2_MIN:
                    p.t2 *= kin.r_step

            active = [p for p in self.particles if p.t2 > kin.t2_MIN]
            if not active:
                break

            # 2. Build integrated proposal weights for candidate selection.
            weights = np.array([self._particle_weight(p) for p in active])
            weights = np.maximum(weights, 0.0)
            total_proposal_weight = float(weights.sum())
            if total_proposal_weight < 1e-15:
                continue

            # 3. Choose a particle; its handler selects a channel and applies veto.
            candidate = active[int(np.random.choice(
                len(active), p=weights / total_proposal_weight
            ))]

            # 4. Count accepted splittings and enforce the pair-production limit.
            if candidate.type == "chi":
                result = self._handle_chi_split(
                    candidate, total_proposal_weight, step_index
                )
                if result == "gluon":
                    n_gluon += 1
                elif result == "photon":
                    n_photon += 1

            elif candidate.type == "g":
                result = self._handle_gluon_split(
                    candidate, total_proposal_weight, step_index
                )
                if result == "pair":
                    n_pairs += 1
                    if n_pairs > max_pairs:
                        return None

            elif candidate.type == "gamma_d":
                result = self._handle_photon_split(
                    candidate, total_proposal_weight, step_index
                )
                if result == "pair":
                    n_pairs += 1
                    if n_pairs > max_pairs:
                        return None

        # 5. Measure the initial fermion after the complete quantum evolution.
        final_mass_state = self.qsm.measure_q0_mass_basis(
            self.kin.mixing_unitary_16
        )
        return n_photon, n_gluon, final_mass_state


def simulate_event(parameters, seed):
    """Evolve one event with explicit parameters and a per-event seed."""
    np.random.seed(seed)
    shower = CWShower12(parameters)
    result = shower.run_shot()
    if result is None:
        return None
    n_ph, n_gl, mass_state = result
    return EventResult(
        n_ph=n_ph,
        n_gl=n_gl,
        mass_state=mass_state,
        final_particles=[
            (p.id, p.type, p.x, p.theta_x, p.theta_y)
            for p in shower.particles
        ],
        final_emissions=list(shower.emissions),
        final_flavors=None,
    )
