"""
SU(3) x U(1)_D classical parton shower with 12 charges and finite quark mass.
Based on arXiv:2310.19881 (QVPS) and arXiv:2502.18566 (conformal-window shower).
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import lambertw

from FlavorMixing import build_flavor_mixing_matrix


@dataclass
class EventResult:
    """One classical event, including discrete final-particle flavor labels."""

    n_ph: int
    n_gl: int
    mass_state: int
    final_particles: list[tuple[int, str, float, float, float]]
    final_emissions: list[tuple[int, str, int, str, float, float, float]]
    final_flavors: list[tuple[int, int]] | None = None


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(24)

PAPER_PHYSICAL_MARKOV = "paper_physical_markov"
INTERACTION_EIGENSTATE_MIXTURE = "interaction_eigenstate_mixture"


# --- 1. Kinematics, couplings, integrated rates, and rejection sampling ---

class CWKinematics12:
    """
    Kinematics and classical splitting rates for the conformal-window shower.

    Provides two-loop SU(3) running, flavor-dependent U(1)_D couplings,
    splitting kernels, z rejection sampling, and the flavor mixing matrix.
    The evolution variable t2 denotes tilde q^2, not invariant mass squared.
    """

    def __init__(self, parameters):
        self.params = dict(parameters)
        N_C = self.params["N_C"]
        N_F = self.params["N_F"]

        self.N_C = N_C
        self.N_F = N_F
        self.N_CHARGE = self.params["N_CHARGE"]
        if self.N_F != self.N_CHARGE:
            raise ValueError(
                "The classical flavor model requires N_F == N_CHARGE, got "
                f"N_F={self.N_F}, N_CHARGE={self.N_CHARGE}."
            )
        self.t2_0 = self.params["t2_0"]
        self.lambda_d = self.params["LAMBDA_D"]
        self.alpha_D = self.params["ALPHA_D"]
        self.r_step = self.params["R_STEP"]
        if not 0.0 < self.r_step < 1.0:
            raise ValueError(
                f"R_STEP must satisfy 0 < R_STEP < 1, got {self.r_step}."
            )

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

        # Coupling bounds used by the z rejection proposals.
        ovf = self.params["OVERESTIMATE_FACTOR"]
        if ovf < 1.0:
            raise ValueError(
                "OVERESTIMATE_FACTOR must be at least 1 for rejection "
                f"sampling, got {ovf}."
            )
        self.ALPHA_S_OVER = self.alpha_star * ovf
        self.ALPHA_D_OVER = self.alpha_D * self.q2_max * ovf
        self.ALPHA_D_SUM_OVER = self.alpha_D * self.q2_sum * ovf

        # U maps the interaction eigenbasis to the physical/mass basis.
        N = self.N_CHARGE
        self.mixing_strength = float(self.params["MIXING_STRENGTH"])
        self.mixing_matrix = build_flavor_mixing_matrix(
            N, self.mixing_strength
        )

        # G_phys = U diag(Q_k) U_dagger. The classical shower uses
        # alpha_D * |G_phys[j, i]|^2 as the chi_i -> chi_j gamma_D weight.
        self.physical_charge_matrix = (
            self.mixing_matrix
            @ np.diag(self.Q_charges)
            @ self.mixing_matrix.conj().T
        )
        self.physical_transition_alpha = (
            self.alpha_D * np.abs(self.physical_charge_matrix) ** 2
        )
        self.physical_total_alpha = np.sum(
            self.physical_transition_alpha, axis=0
        )

    # --- Couplings ---

    def alpha_s(self, mu2):
        """Return the two-loop SU(3) coupling alpha_s(mu2) via Lambert W."""
        values = np.asarray(mu2, dtype=float)
        safe_values = np.maximum(values, np.finfo(float).tiny)
        v = (1.0 / np.e) * (safe_values / self.LAMBDA2) ** self.gamma_anom
        W = np.real(lambertw(v, k=0))
        result = np.maximum(self.alpha_star / (1.0 + W), 0.0)
        result = np.where(values <= 0.0, self.alpha_star, result)
        if values.ndim == 0:
            return float(result)
        return np.asarray(result, dtype=float)

    def alpha_d_charge(self, charge_idx):
        """Return the fixed U(1)_D effective coupling alpha_D * Q_k^2."""
        return self.alpha_D * self.q2_charges[charge_idx]

    def alpha_d_sum(self):
        """Sum the fixed U(1)_D couplings over all charge eigenstates."""
        return self.alpha_D * self.q2_sum

    def alpha_d_total_physical(self, source_flavor):
        """Return sum_j alpha_D * |G_ji|^2 for source physical flavor i."""
        return float(self.physical_total_alpha[source_flavor])

    def sample_physical_transition(self, source_flavor):
        """Sample final physical flavor j for chi_i -> chi_j gamma_D."""
        weights = np.asarray(
            self.physical_transition_alpha[:, source_flavor], dtype=float
        )
        total = float(weights.sum())
        if total <= 1e-15:
            return int(source_flavor)
        return int(np.random.choice(self.N_CHARGE, p=weights / total))

    def sample_physical_pair_flavors(self):
        """Sample (chi_flavor, anti_flavor) with weights |G_ij|^2."""
        weights = np.asarray(self.physical_transition_alpha, dtype=float)
        flat = weights.ravel()
        total = float(flat.sum())
        if total <= 1e-15:
            flavor = int(np.random.randint(self.N_CHARGE))
            return flavor, flavor
        matrix_index = int(np.random.choice(flat.size, p=flat / total))
        anti_flavor, chi_flavor = np.unravel_index(
            matrix_index, weights.shape
        )
        return int(chi_flavor), int(anti_flavor)

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

    # --- Proposal overestimates for z rejection sampling ---

    def gover_chi_g_at_z(self, t2, z):
        return (
            self.neg_ln_r * self.ALPHA_S_OVER * self.C_F
            / (2 * np.pi) * 2.0 / (1.0 - z)
        )

    def gover_chi_gd_at_z(self, t2, z):
        return (
            self.neg_ln_r * self.ALPHA_D_OVER
            / (2 * np.pi) * 2.0 / (1.0 - z)
        )

    def gover_gg_at_z(self, t2, z):
        return (
            self.neg_ln_r * self.ALPHA_S_OVER * self.C_A
            / np.pi * (1.0 / z + 1.0 / (1 - z))
        )

    def gover_g_qq_at_z(self, t2, z):
        return self.neg_ln_r * self.ALPHA_S_OVER * self.N_F * self.T_F / np.pi

    def gover_gd_qq_at_z(self, t2, z):
        return self.neg_ln_r * self.ALPHA_D_SUM_OVER / np.pi

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

    # --- True integrated splitting rates and stepwise Sudakov factors ---

    @staticmethod
    def _primitive_p_ff(z):
        return -0.5 * z ** 2 - z - 2.0 * np.log1p(-z)

    @staticmethod
    def _primitive_p_fA(z):
        return (2.0 / 3.0) * z ** 3 - z ** 2 + z

    @staticmethod
    def _primitive_p_gg(z):
        return (-np.log1p(-z) + np.log(z) - 2.0 * z
                + 0.5 * z ** 2 - z ** 3 / 3.0)

    def _integrated_kernel(self, t2, primitive):
        z_min, z_max = self.z_limits(t2)
        if z_min >= z_max:
            return 0.0
        return float(primitive(z_max) - primitive(z_min))

    def true_chi_g_at_z(self, t2, z):
        mu2_chi_to_chi_g = self.mu2_chi_to_chi_g(t2, z)
        return (self.neg_ln_r * self.alpha_s(mu2_chi_to_chi_g) * self.C_F
                / (2 * np.pi) * self.p_ff(z))

    def true_chi_gd_at_z(self, t2, z, flavor):
        return (self.neg_ln_r * self.alpha_d_charge(flavor)
                / (2 * np.pi) * self.p_ff(z))

    def true_chi_gd_physical_at_z(self, t2, z, source_flavor):
        return (
            self.neg_ln_r * self.alpha_d_total_physical(source_flavor)
            / (2 * np.pi) * self.p_ff(z)
        )

    def true_gg_at_z(self, t2, z):
        mu2_g_to_gg = self.mu2_g_to_gg(t2, z)
        return (self.neg_ln_r * self.alpha_s(mu2_g_to_gg) * self.C_A
                / (2 * np.pi) * self.p_gg(z))

    def true_g_qq_at_z(self, t2, z):
        mu2_g_to_chi_chibar = self.mu2_g_to_chi_chibar(t2, z)
        return (
            self.neg_ln_r * self.N_F * self.alpha_s(mu2_g_to_chi_chibar)
            * self.T_F / (2 * np.pi) * self.p_fA(z)
        )

    def true_gd_qq_at_z(self, t2, z):
        return (self.neg_ln_r * self.alpha_d_sum()
                / (2 * np.pi) * self.p_fA(z))

    def rate_chi_g(self, t2):
        """Return the z-integrated chi -> chi g intensity for one step."""
        z_min, z_max = self.z_limits(t2)
        if z_min >= z_max:
            return 0.0
        # y = -ln(1-z) removes the pole: P_ff(z) dz = (1 + z^2) dy.
        y_min = -np.log1p(-z_min)
        y_max = -np.log1p(-z_max)
        half = 0.5 * (y_max - y_min)
        mid = 0.5 * (y_max + y_min)
        y = mid + half * _GL_NODES
        z = 1.0 - np.exp(-y)
        values = (
            self.neg_ln_r * self.alpha_s(self.mu2_chi_to_chi_g(t2, z)) * self.C_F
            / (2 * np.pi) * (1.0 + z ** 2)
        )
        return max(float(half * np.dot(_GL_WEIGHTS, values)), 0.0)

    def rate_chi_gd(self, t2, flavor):
        """Integrate chi_k -> chi_k gamma_D with the fixed U(1)_D coupling."""
        kernel = self._integrated_kernel(t2, self._primitive_p_ff)
        return max(
            self.neg_ln_r * self.alpha_d_charge(flavor)
            / (2 * np.pi) * kernel,
            0.0,
        )

    def rate_chi_gd_physical(self, t2, source_flavor):
        """Integrate sum_j chi_i -> chi_j gamma_D using physical-basis G_ji."""
        kernel = self._integrated_kernel(t2, self._primitive_p_ff)
        return max(
            self.neg_ln_r * self.alpha_d_total_physical(source_flavor)
            / (2 * np.pi) * kernel,
            0.0,
        )

    def rate_gg(self, t2):
        """Return the one-step g -> gg intensity using mu_gg^2 = k_perp^2."""
        z_min, z_max = self.z_limits(t2)
        if z_min >= z_max:
            return 0.0
        # y = log(z/(1-z)) removes both endpoint poles of P_gg.
        y_min = np.log(z_min / (1.0 - z_min))
        y_max = np.log(z_max / (1.0 - z_max))
        half = 0.5 * (y_max - y_min)
        mid = 0.5 * (y_max + y_min)
        y = mid + half * _GL_NODES
        z = 1.0 / (1.0 + np.exp(-y))
        transformed_kernel = (
            z ** 2 + (1.0 - z) ** 2 + z ** 2 * (1.0 - z) ** 2
        )
        values = (
            self.neg_ln_r * self.alpha_s(self.mu2_g_to_gg(t2, z)) * self.C_A
            / (2 * np.pi) * transformed_kernel
        )
        return max(float(half * np.dot(_GL_WEIGHTS, values)), 0.0)

    def rate_g_qq(self, t2):
        """Return the one-step g -> chi chi_bar intensity using QVPS mu_fA^2."""
        z_min, z_max = self.z_limits(t2)
        if z_min >= z_max:
            return 0.0
        half = 0.5 * (z_max - z_min)
        mid = 0.5 * (z_max + z_min)
        z = mid + half * _GL_NODES
        values = (
            self.neg_ln_r * self.N_F * self.alpha_s(self.mu2_g_to_chi_chibar(t2, z))
            * self.T_F / (2 * np.pi) * self.p_fA(z)
        )
        return max(float(half * np.dot(_GL_WEIGHTS, values)), 0.0)

    def rate_gd_qq(self, t2):
        kernel = self._integrated_kernel(t2, self._primitive_p_fA)
        return max(
            self.neg_ln_r * self.alpha_d_sum()
            / (2 * np.pi) * kernel,
            0.0,
        )

    @staticmethod
    def sudakov_no_emission(total_rate):
        """Return the one-step no-splitting probability exp(-sum_c lambda_c)."""
        total_rate = max(float(total_rate), 0.0)
        return float(np.exp(-total_rate))

    def _sample_true_z(self, t2, proposal, true_density, over_density, label):
        """Reject overestimate proposals until z follows the true density."""
        for _ in range(10000):
            z = proposal(t2)
            over = float(over_density(t2, z))
            true = float(true_density(t2, z))
            if over <= 0.0:
                raise RuntimeError(f"Non-positive overestimate for {label} at t2={t2:g}")
            ratio = true / over
            if ratio > 1.0 + 1e-10:
                raise RuntimeError(
                    f"Overestimate violated for {label}: true/over={ratio:.6g}, "
                    f"t2={t2:g}, z={z:g}. Increase OVERESTIMATE_FACTOR."
                )
            if np.random.uniform() < np.clip(ratio, 0.0, 1.0):
                return z
        raise RuntimeError(f"Unable to sample z for {label} after 10000 proposals")

    def sample_true_z_chi_g(self, t2):
        return self._sample_true_z(
            t2, self.sample_z_pole, self.true_chi_g_at_z,
            self.gover_chi_g_at_z, "chi->chi g",
        )

    def sample_true_z_chi_gd(self, t2, flavor):
        return self._sample_true_z(
            t2, self.sample_z_pole,
            lambda scale, z: self.true_chi_gd_at_z(scale, z, flavor),
            self.gover_chi_gd_at_z, "chi->chi gamma_D",
        )

    def sample_true_z_chi_gd_physical(self, t2, source_flavor):
        return self._sample_true_z(
            t2, self.sample_z_pole,
            lambda scale, z: self.true_chi_gd_physical_at_z(
                scale, z, source_flavor
            ),
            self.gover_chi_gd_at_z, "chi_i->chi_j gamma_D",
        )

    def sample_true_z_gg(self, t2):
        return self._sample_true_z(
            t2, self.sample_z_gg, self.true_gg_at_z,
            self.gover_gg_at_z, "g->gg",
        )

    def sample_true_z_g_qq(self, t2):
        return self._sample_true_z(
            t2, self.sample_z_flat, self.true_g_qq_at_z,
            self.gover_g_qq_at_z, "g->chi chi_bar",
        )

    def sample_true_z_gd_qq(self, t2):
        return self._sample_true_z(
            t2, self.sample_z_flat, self.true_gd_qq_at_z,
            self.gover_gd_qq_at_z, "gamma_D->chi chi_bar",
        )

    def sample_photon_pair_flavor(self):
        weights = np.array([
            self.alpha_d_charge(k)
            for k in range(self.N_CHARGE)
        ], dtype=float)
        weights = np.maximum(weights, 0.0)
        total = float(weights.sum())
        if total <= 1e-15:
            weights = np.ones(self.N_CHARGE, dtype=float) / self.N_CHARGE
        else:
            weights /= total
        return int(np.random.choice(self.N_CHARGE, p=weights))


# --- 2. Particle ---

class Particle:
    """Classical particle kinematics and a discrete flavor label.

    t2 is the angular evolution variable tilde q^2. The energy fraction x
    gives E = x * E0. theta_x and theta_y are transverse angular components
    relative to the initial axis. For chi particles, flavor is a physical
    flavor or an interaction eigenstate index, as selected by flavor_scheme.
    """
    _LABELS = {"chi": "χ", "g": "g", "gamma_d": "γ_D"}

    def __init__(self, pid, ptype, t2, flavor=None, energy_fraction=1.0,
                 theta_x=0.0, theta_y=0.0):
        self.id = pid
        self.type = ptype
        self.t2 = t2
        self.flavor = flavor
        if self.type == "chi" and self.flavor is None:
            raise ValueError("Classical chi particles require a flavor label")
        self.x = energy_fraction
        self.theta_x = theta_x
        self.theta_y = theta_y

    def __repr__(self):
        flavor = f" k={self.flavor}" if self.flavor is not None else ""
        return (f"[{self._LABELS.get(self.type, self.type)}-{self.id}"
                f"{flavor} E={self.x:.3f}]")


# --- 3. Classical shower: Sudakov decision and flavor-channel sampling ---

class CWShower12:
    """Evolve one event using integrated rates and discrete flavor transitions."""

    def __init__(self, parameters):
        self.params = dict(parameters)
        self.kin = CWKinematics12(self.params)
        self.next_id = 1
        self.emissions = []
        self.flavor_scheme = self.params.get(
            "CLASSICAL_FLAVOR_SCHEME", PAPER_PHYSICAL_MARKOV
        )
        allowed_schemes = {
            PAPER_PHYSICAL_MARKOV,
            INTERACTION_EIGENSTATE_MIXTURE,
        }
        if self.flavor_scheme not in allowed_schemes:
            raise ValueError(
                "CLASSICAL_FLAVOR_SCHEME must be one of "
                f"{sorted(allowed_schemes)}, got {self.flavor_scheme!r}"
            )

        if self.flavor_scheme == PAPER_PHYSICAL_MARKOV:
            self.initial_flavor = int(self.params.get("INITIAL_PHYSICAL_FLAVOR", 0))
            if not 0 <= self.initial_flavor < self.kin.N_CHARGE:
                raise ValueError(
                    "INITIAL_PHYSICAL_FLAVOR must satisfy "
                    f"0 <= flavor < {self.kin.N_CHARGE}"
                )
        else:
            self.initial_flavor = int(np.random.randint(self.kin.N_CHARGE))

        self.particles = [
            Particle(
                self.next_id,
                "chi",
                t2=self.kin.t2_0,
                flavor=self.initial_flavor,
                energy_fraction=1.0,
            )
        ]
        self.next_id += 1

    def _channel_entries(self, p):
        """Return enabled (channel, integrated_rate) entries with positive rate."""
        entries = []
        if p.type == "chi":
            if self.params["ENABLE_Q_TO_Q_G"]:
                entries.append(("chi_g", self.kin.rate_chi_g(p.t2)))
            if self.params["ENABLE_Q_TO_Q_DP"]:
                if self.flavor_scheme == PAPER_PHYSICAL_MARKOV:
                    rate = self.kin.rate_chi_gd_physical(p.t2, p.flavor)
                else:
                    rate = self.kin.rate_chi_gd(p.t2, p.flavor)
                entries.append(("chi_gd", rate))
        elif p.type == "g":
            if self.params["ENABLE_G_TO_GG"]:
                entries.append(("g_gg", self.kin.rate_gg(p.t2)))
            if self.params["ENABLE_G_TO_QQ"]:
                entries.append(("g_qq", self.kin.rate_g_qq(p.t2)))
        elif p.type == "gamma_d":
            if self.params["ENABLE_DP_TO_QQ"]:
                entries.append(("gd_qq", self.kin.rate_gd_qq(p.t2)))
        return [(channel, rate) for channel, rate in entries if rate > 0.0]

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

        Advance next_id here; the caller records the pre-split kinematics
        and applies any flavor transition.
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
                t2=(1.0 - z) ** 2 * parent_t2,
                energy_fraction=parent_energy_fraction * (1.0 - z),
                theta_x=parent_theta_x + dx_m,
                theta_y=parent_theta_y + dy_m,
            )
        )
        self.next_id += 1

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

    def _handle_chi_channel(self, candidate, channel, step_index):
        """Apply an already selected chi channel and its flavor transition."""
        if channel == "chi_g":
            parent_t2, parent_energy_fraction = candidate.t2, candidate.x
            z = self.kin.sample_true_z_chi_g(candidate.t2)
            self._apply_radiation_split(candidate, "g", z)
            self._record_emission(
                step_index, "chi_to_chi_g", candidate.id, "chi",
                parent_t2, z, parent_energy_fraction,
            )
            return "gluon"
        if channel == "chi_gd":
            parent_t2, parent_energy_fraction = candidate.t2, candidate.x
            if self.flavor_scheme == PAPER_PHYSICAL_MARKOV:
                source_flavor = candidate.flavor
                z = self.kin.sample_true_z_chi_gd_physical(
                    candidate.t2, source_flavor
                )
                target_flavor = self.kin.sample_physical_transition(
                    source_flavor
                )
            else:
                z = self.kin.sample_true_z_chi_gd(candidate.t2, candidate.flavor)
                target_flavor = candidate.flavor
            self._apply_radiation_split(candidate, "gamma_d", z)
            candidate.flavor = target_flavor
            self._record_emission(
                step_index, "chi_to_chi_gamma_d", candidate.id, "chi",
                parent_t2, z, parent_energy_fraction,
            )
            return "photon"
        raise ValueError(f"Unsupported chi channel: {channel}")

    def _apply_pair_split(self, candidate, z, flavor, second_flavor=None):
        """Replace a gauge boson with a chi pair carrying the given flavors."""
        if second_flavor is None:
            second_flavor = flavor
        parent_t2, parent_energy_fraction = candidate.t2, candidate.x
        dx_ell, dy_ell, dx_m, dy_m = self._split_2d(
            parent_t2, z, parent_energy_fraction
        )
        parent_theta_x, parent_theta_y = candidate.theta_x, candidate.theta_y
        self.particles = [p for p in self.particles if p is not candidate]
        self.particles.append(
            Particle(
                self.next_id,
                "chi",
                t2=z ** 2 * parent_t2,
                flavor=flavor,
                energy_fraction=parent_energy_fraction * z,
                theta_x=parent_theta_x + dx_ell,
                theta_y=parent_theta_y + dy_ell,
            )
        )
        self.particles.append(
            Particle(
                self.next_id + 1,
                "chi",
                t2=(1.0 - z) ** 2 * parent_t2,
                flavor=second_flavor,
                energy_fraction=parent_energy_fraction * (1.0 - z),
                theta_x=parent_theta_x + dx_m,
                theta_y=parent_theta_y + dy_m,
            )
        )
        self.next_id += 2

    def _handle_gluon_channel(self, candidate, channel, step_index):
        """Apply an already selected g -> gg or g -> chi chi_bar channel."""
        if channel == "g_gg":
            parent_id, parent_t2, parent_energy_fraction = (
                candidate.id, candidate.t2, candidate.x
            )
            z = self.kin.sample_true_z_gg(candidate.t2)
            self._apply_radiation_split(candidate, "g", z)
            self._record_emission(
                step_index, "g_to_gg", parent_id, "g",
                parent_t2, z, parent_energy_fraction,
            )
            return "gg"
        if channel == "g_qq":
            parent_id, parent_t2, parent_energy_fraction = (
                candidate.id, candidate.t2, candidate.x
            )
            z = self.kin.sample_true_z_g_qq(candidate.t2)
            flavor = int(np.random.randint(self.kin.N_CHARGE))
            self._apply_pair_split(candidate, z, flavor)
            self._record_emission(
                step_index, "g_to_chi_chi", parent_id, "g",
                parent_t2, z, parent_energy_fraction,
            )
            return "pair"
        raise ValueError(f"Unsupported gluon channel: {channel}")

    def _handle_photon_channel(self, candidate, channel, step_index):
        """Apply gamma_D -> chi chi_bar with the configured flavor scheme."""
        if channel != "gd_qq":
            raise ValueError(f"Unsupported dark-photon channel: {channel}")
        parent_id, parent_t2, parent_energy_fraction = (
            candidate.id, candidate.t2, candidate.x
        )
        z = self.kin.sample_true_z_gd_qq(candidate.t2)
        if self.flavor_scheme == PAPER_PHYSICAL_MARKOV:
            flavor, second_flavor = self.kin.sample_physical_pair_flavors()
        else:
            flavor = self.kin.sample_photon_pair_flavor()
            second_flavor = flavor
        self._apply_pair_split(candidate, z, flavor, second_flavor)
        self._record_emission(
            step_index, "gamma_d_to_chi_chi", parent_id, "gamma_d",
            parent_t2, z, parent_energy_fraction,
        )
        return "pair"

    def run_shot(self, max_steps=None, max_pairs=None):
        """Return (n_photon, n_gluon, final_mass_state) for one event.

        The counters track accepted chi radiation into photons and gluons.
        final_mass_state follows the configured classical flavor scheme.
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

            # 2. Collect true integrated rates for all particle-channel pairs.
            entries = []
            for particle in active:
                for channel, rate in self._channel_entries(particle):
                    entries.append((particle, channel, rate))

            if not entries:
                continue

            rates = np.array([entry[2] for entry in entries], dtype=float)
            rates = np.maximum(rates, 0.0)
            total_rate = float(rates.sum())
            if total_rate <= 0.0:
                continue

            # 3. Decide whether this step emits, then choose a particle-channel pair.
            delta_step = kin.sudakov_no_emission(total_rate)
            p_emit = 1.0 - delta_step
            if np.random.uniform() >= p_emit:
                continue

            chosen = int(np.random.choice(len(entries), p=rates / total_rate))
            candidate, channel, _ = entries[chosen]

            # 4. Apply the splitting and enforce the pair-production limit.
            if candidate.type == "chi":
                result = self._handle_chi_channel(candidate, channel, step_index)
                if result == "gluon":
                    n_gluon += 1
                elif result == "photon":
                    n_photon += 1

            elif candidate.type == "g":
                result = self._handle_gluon_channel(
                    candidate, channel, step_index
                )
                if result == "pair":
                    n_pairs += 1
                    if n_pairs > max_pairs:
                        return None

            elif candidate.type == "gamma_d":
                result = self._handle_photon_channel(
                    candidate, channel, step_index
                )
                if result == "pair":
                    n_pairs += 1
                    if n_pairs > max_pairs:
                        return None

        # 5. Obtain the initial fermion's final flavor using the selected scheme.
        if self.flavor_scheme == PAPER_PHYSICAL_MARKOV:
            # Particle id=1 is the original physical-flavor fermion. Radiation
            # keeps that object and updates its Markov flavor label in place.
            original = next(p for p in self.particles if p.id == 1)
            final_mass_state = int(original.flavor)
        else:
            mass_probabilities = np.abs(
                self.kin.mixing_matrix[:, self.initial_flavor]
            ) ** 2
            mass_probabilities /= mass_probabilities.sum()
            final_mass_state = int(np.random.choice(
                self.kin.N_CHARGE, p=mass_probabilities
            ))
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
        final_flavors=[(p.id, int(p.flavor)) for p in shower.particles if p.type == "chi"],
    )
