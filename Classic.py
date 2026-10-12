"""
SU(3) × U(1)_D 共形窗口经典 Parton Shower — 12 种电荷、有限夸克质量。

基于 arXiv:2310.19881 (QVPS) 与 arXiv:2502.18566 (共形窗口簇射)。
"""

import math
import time
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.special import lambertw

from FlavorMixing import build_flavor_mixing_matrix
from Parameters import PARAMS
from result_storage import resolve_random_seed, save_simulation_results


def _configure_max_steps_from_t2_end():
    """Derive the discrete evolution length from the requested nominal scale."""
    t2_end = PARAMS.get("T2_END")
    PARAMS["T2_END_GRID"] = None
    if t2_end is None:
        return

    t2_0 = float(PARAMS["t2_0"])
    t2_end = float(t2_end)
    r_step = float(PARAMS["R_STEP"])
    t2_min = 4.0 * float(PARAMS["M_Q"]) ** 2

    if not 0.0 < r_step < 1.0:
        raise ValueError("R_STEP must satisfy 0 < R_STEP < 1")
    if not t2_min <= t2_end <= t2_0:
        raise ValueError(
            f"T2_END must satisfy {t2_min:g} <= T2_END <= {t2_0:g}"
        )

    max_steps = int(math.ceil(math.log(t2_end / t2_0) / math.log(r_step)))
    PARAMS["MAX_STEPS"] = max_steps
    PARAMS["T2_END_GRID"] = t2_0 * r_step ** max_steps


_configure_max_steps_from_t2_end()


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(24)

PAPER_PHYSICAL_MARKOV = "paper_physical_markov"
INTERACTION_EIGENSTATE_MIXTURE = "interaction_eigenstate_mixture"


# --- 1. CWKinematics12 — 运动学、耦合常数、拒绝采样 ---

class CWKinematics12:
    """
    物理运动学: SU(3) 两环跑动耦合, U(1)_D 味道依赖耦合,
    Altarelli-Parisi 分裂函数, z 拒绝采样的过高估计分布, 可调味道混合矩阵.
    """

    def __init__(self):
        N_C = PARAMS["N_C"]
        N_F = PARAMS["N_F"]

        self.N_C = N_C
        self.N_F = N_F
        self.N_CHARGE = PARAMS["N_CHARGE"]
        if self.N_F != self.N_CHARGE:
            raise ValueError(
                "The classical flavor model requires N_F == N_CHARGE, got "
                f"N_F={self.N_F}, N_CHARGE={self.N_CHARGE}."
            )
        self.t2_0 = PARAMS["t2_0"]
        self.lambda_d = PARAMS["LAMBDA_D"]
        self.alpha_D = PARAMS["ALPHA_D"]
        self.r_step = PARAMS["R_STEP"]
        if not 0.0 < self.r_step < 1.0:
            raise ValueError(
                f"R_STEP must satisfy 0 < R_STEP < 1, got {self.r_step}."
            )

        self.LAMBDA2 = self.lambda_d ** 2
        self.neg_ln_r = -np.log(self.r_step)

        self.M_q = PARAMS["M_Q"]
        self.MQ2 = self.M_q ** 2
        self.t2_MIN = 4.0 * self.MQ2

        # SU(3) 群论因子
        self.C_A = float(N_C)
        self.C_F = (N_C ** 2 - 1) / (2.0 * N_C)
        self.T_F = 0.5

        # 两环 β 函数系数
        self.beta_0 = (
            11 * self.C_A / 3 - 4 * self.T_F * N_F / 3
        ) / (4 * np.pi)
        self.beta_1 = (
            34 * self.C_A ** 2 / 3
            - (20 * self.C_A / 3 + 4 * self.C_F) * self.T_F * N_F
        ) / (4 * np.pi) ** 2

        self.alpha_star = -self.beta_0 / self.beta_1
        self.gamma_anom = -self.beta_0 ** 2 / self.beta_1

        # U(1)_D 电荷列表
        self.Q_charges = np.linspace(PARAMS["Q_MIN"], PARAMS["Q_MAX"],
                                     self.N_CHARGE)
        self.q2_charges = self.Q_charges ** 2
        self.q2_sum = float(np.sum(self.q2_charges))
        self.q2_max = float(np.max(self.q2_charges))

        # 过高估计耦合
        ovf = PARAMS["OVERESTIMATE_FACTOR"]
        if ovf < 1.0:
            raise ValueError(
                "OVERESTIMATE_FACTOR must be at least 1 for rejection "
                f"sampling, got {ovf}."
            )
        self.ALPHA_S_OVER = self.alpha_star * ovf
        self.ALPHA_D_OVER = self.alpha_D * self.q2_max * ovf
        self.ALPHA_D_SUM_OVER = self.alpha_D * self.q2_sum * ovf

        # U 把 interaction eigenbasis 映射到 physical/mass flavor basis。
        N = self.N_CHARGE
        self.mixing_strength = float(PARAMS["MIXING_STRENGTH"])
        self.mixing_matrix = build_flavor_mixing_matrix(
            N, self.mixing_strength
        )

        # G_phys = U diag(Q_k) U†。传统经典 shower 在截面层面使用
        # alpha_D |(G_phys)_ji|² 作为 χ_i→χ_j gamma_D 的 flavor 权重。
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

    # --- 耦合常数 ---

    def alpha_s(self, mu2):
        """SU(3) Lambert W 两环跑动耦合 α_s(μ²)"""
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
        """固定 U(1)_D 有效耦合 alpha_D Q_k^2。"""
        return self.alpha_D * self.q2_charges[charge_idx]

    def alpha_d_sum(self):
        """固定 U(1)_D 耦合对所有电荷本征态求和。"""
        return self.alpha_D * self.q2_sum

    def alpha_d_total_physical(self, source_flavor):
        """Σ_j alpha_D |G_ji|²，传统 physical-flavor shower 的总耦合。"""
        return float(self.physical_total_alpha[source_flavor])

    def sample_physical_transition(self, source_flavor):
        """按 |G_ji|² 抽取一次 χ_i→χ_j gamma_D 的末态 flavor j。"""
        weights = np.asarray(
            self.physical_transition_alpha[:, source_flavor], dtype=float
        )
        total = float(weights.sum())
        if total <= 1e-15:
            return int(source_flavor)
        return int(np.random.choice(self.N_CHARGE, p=weights / total))

    def sample_physical_pair_flavors(self):
        """按 |G_ij|² 抽取 gamma_D→χ_j χbar_i 的两个 physical flavors。"""
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

    # --- 重整化标度 ---

    def mu2_chi_to_chi_g(self, t2, z):
        """Scale for chi -> chi g; chi -> chi gamma_D uses the same prescription."""
        return z * (1.0 - z) ** 2 * t2

    def mu2_g_to_chi_chibar(self, t2, z):
        """Scale for g -> chi chibar; gamma_D -> chi chibar uses the same prescription."""
        return z ** 2 * (1.0 - z) ** 2 * t2

    def mu2_g_to_gg(self, t2, z):
        """g -> gg scale: k_perp^2 with k_perp=z(1-z)sqrt(t2)."""
        return z ** 2 * (1.0 - z) ** 2 * t2

    # --- Altarelli-Parisi 分裂函数 ---

    def p_ff(self, z):
        return (1.0 + z ** 2) / (1.0 - z)

    def p_fA(self, z):
        return z ** 2 + (1.0 - z) ** 2

    def p_gg(self, z):
        return z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z)

    # --- z 运动学限制 ---

    def z_limits(self, t2):
        if t2 <= self.t2_MIN:
            return 0.5, 0.5
        zh = np.sqrt(self.MQ2 / t2)
        return zh, 1.0 - zh

    # --- 过高估计分布 ---

    def gover_chi_g_at_z(self, t2, z):
        return self.neg_ln_r * self.ALPHA_S_OVER * self.C_F / (2 * np.pi) * 2.0 / (1.0 - z)

    def gover_chi_gd_at_z(self, t2, z):
        return self.neg_ln_r * self.ALPHA_D_OVER / (2 * np.pi) * 2.0 / (1.0 - z)

    def gover_gg_at_z(self, t2, z):
        return self.neg_ln_r * self.ALPHA_S_OVER * self.C_A / np.pi * (1.0 / z + 1.0 / (1 - z))

    def gover_g_qq_at_z(self, t2, z):
        return self.neg_ln_r * self.ALPHA_S_OVER * self.N_F * self.T_F / np.pi

    def gover_gd_qq_at_z(self, t2, z):
        return self.neg_ln_r * self.ALPHA_D_SUM_OVER / np.pi

    # --- z 采样 ---

    def sample_z_pole(self, t2):
        """从 1/(1-z) 极点分布采样"""
        zn, zx = self.z_limits(t2)
        if zn >= zx:
            return 0.5
        u = np.random.uniform()
        z = 1.0 - (1.0 - zn) ** (1 - u) * zn ** u
        return float(np.clip(z, zn + 1e-10, zx - 1e-10))

    def sample_z_gg(self, t2):
        """从 1/z + 1/(1-z) 双极点分布采样"""
        zn, zx = self.z_limits(t2)
        if zn >= zx:
            return 0.5
        u = np.random.uniform()
        if np.random.uniform() < 0.5:
            z = 1.0 - (1.0 - zn) ** (1 - u) * zn ** u
        else:
            z = zn ** (1 - u) * (1.0 - zn) ** u
        return float(np.clip(z, zn + 1e-10, zx - 1e-10))

    def sample_z_flat(self, t2):
        zn, zx = self.z_limits(t2)
        if zn >= zx:
            return 0.5
        return float(np.random.uniform(zn, zx))

    # --- 真实积分分裂率与逐步 Sudakov 因子 ---

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
        zn, zx = self.z_limits(t2)
        if zn >= zx:
            return 0.0
        return float(primitive(zx) - primitive(zn))

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
        return (self.neg_ln_r * self.N_F * self.alpha_s(mu2_g_to_chi_chibar) * self.T_F
                / (2 * np.pi) * self.p_fA(z))

    def true_gd_qq_at_z(self, t2, z):
        return (self.neg_ln_r * self.alpha_d_sum()
                / (2 * np.pi) * self.p_fA(z))

    def rate_chi_g(self, t2):
        """该步 χ→χg 的 z 积分强度 λ。"""
        zn, zx = self.z_limits(t2)
        if zn >= zx:
            return 0.0
        # y=-ln(1-z) 消去 P_ff 的端点极点：P_ff(z)dz=(1+z²)dy。
        y_min = -np.log1p(-zn)
        y_max = -np.log1p(-zx)
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
        """该步 χ_k→χ_kγ_D 的 z 积分强度 λ；当前 U(1)_D 耦合不跑动。"""
        kernel = self._integrated_kernel(t2, self._primitive_p_ff)
        return max(
            self.neg_ln_r * self.alpha_d_charge(flavor)
            / (2 * np.pi) * kernel,
            0.0,
        )

    def rate_chi_gd_physical(self, t2, source_flavor):
        """Σ_j χ_i→χ_jγ_D 的总积分强度，使用 physical-basis G_ji。"""
        kernel = self._integrated_kernel(t2, self._primitive_p_ff)
        return max(
            self.neg_ln_r * self.alpha_d_total_physical(source_flavor)
            / (2 * np.pi) * kernel,
            0.0,
        )

    def rate_gg(self, t2):
        """该步 g→gg 的 z 积分强度，使用 mu_gg^2=k_perp^2。"""
        zn, zx = self.z_limits(t2)
        if zn >= zx:
            return 0.0
        # y=log(z/(1-z)) 同时消去 P_gg 的两个端点极点。
        y_min = np.log(zn / (1.0 - zn))
        y_max = np.log(zx / (1.0 - zx))
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
        """该步 g→chi chi_bar 的 z 积分强度，使用 QVPS mu_fA^2。"""
        zn, zx = self.z_limits(t2)
        if zn >= zx:
            return 0.0
        half = 0.5 * (zx - zn)
        mid = 0.5 * (zx + zn)
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
        """一步内无分裂概率 Δ=exp(-Σλ_c)。"""
        total_rate = max(float(total_rate), 0.0)
        return float(np.exp(-total_rate))

    def _sample_true_z(self, t2, proposal, true_density, over_density, label):
        """从 overestimate proposal 反复拒绝抽样，得到真实 z 分布。"""
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


# --- 3. CWShower12 — 共形窗口经典簇射主循环 ---

class CWShower12:

    def __init__(self):
        self.kin = CWKinematics12()
        self.next_id = 1
        self.emissions = []
        self.flavor_scheme = PARAMS.get(
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
            self.initial_flavor = int(PARAMS.get("INITIAL_PHYSICAL_FLAVOR", 0))
            if not 0 <= self.initial_flavor < self.kin.N_CHARGE:
                raise ValueError(
                    "INITIAL_PHYSICAL_FLAVOR must satisfy "
                    f"0 <= flavor < {self.kin.N_CHARGE}"
                )
        else:
            self.initial_flavor = int(np.random.randint(self.kin.N_CHARGE))

        self.particles = [
            Particle(self.next_id, "chi",
                     t2=self.kin.t2_0, flavor=self.initial_flavor,
                     energy_fraction=1.0)
        ]
        self.next_id += 1

    def _channel_entries(self, p):
        """返回该粒子在当前一步的 (channel, integrated rate) 列表。"""
        entries = []
        if p.type == "chi":
            if PARAMS["ENABLE_Q_TO_Q_G"]:
                entries.append(("chi_g", self.kin.rate_chi_g(p.t2)))
            if PARAMS["ENABLE_Q_TO_Q_DP"]:
                if self.flavor_scheme == PAPER_PHYSICAL_MARKOV:
                    rate = self.kin.rate_chi_gd_physical(p.t2, p.flavor)
                else:
                    rate = self.kin.rate_chi_gd(p.t2, p.flavor)
                entries.append(("chi_gd", rate))
        elif p.type == "g":
            if PARAMS["ENABLE_G_TO_GG"]:
                entries.append(("g_gg", self.kin.rate_gg(p.t2)))
            if PARAMS["ENABLE_G_TO_QQ"]:
                entries.append(("g_qq", self.kin.rate_g_qq(p.t2)))
        elif p.type == "gamma_d":
            if PARAMS["ENABLE_DP_TO_QQ"]:
                entries.append(("gd_qq", self.kin.rate_gd_qq(p.t2)))
        return [(channel, rate) for channel, rate in entries if rate > 0.0]

    @staticmethod
    def _split_2d(t2, z, x_parent):
        """Eq. 65-68 splitting kinematics as two transverse angular offsets."""
        E_parent = x_parent * PARAMS["E0"]
        if E_parent < 1e-15:
            return 0.0, 0.0, 0.0, 0.0
        q = np.sqrt(max(t2, 0.0))
        phi = np.random.uniform(0, 2 * np.pi)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        d_ell = (1 - z) * q / E_parent
        d_m = z * q / E_parent
        return (d_ell * cos_phi, d_ell * sin_phi,
                -d_m * cos_phi, -d_m * sin_phi)

    # --- 分裂通道处理 ---

    def _apply_radiation_split(self, cand, daughter_type, z):
        """应用 χ→χV 或 g→gg 的二体运动学。"""
        t2o, xo = cand.t2, cand.x
        dx_ell, dy_ell, dx_m, dy_m = self._split_2d(t2o, z, xo)
        tx, ty = cand.theta_x, cand.theta_y
        cand.t2 = z ** 2 * t2o
        cand.x = xo * z
        cand.theta_x = tx + dx_ell
        cand.theta_y = ty + dy_ell
        self.particles.append(
            Particle(
                self.next_id,
                daughter_type,
                t2=(1.0 - z) ** 2 * t2o,
                energy_fraction=xo * (1.0 - z),
                theta_x=tx + dx_m,
                theta_y=ty + dy_m,
            )
        )
        self.next_id += 1

    def _record_emission(self, step_index, channel, parent_id, parent_type,
                         t2, z, parent_energy_fraction):
        self.emissions.append((
            step_index,
            channel,
            parent_id,
            parent_type,
            t2,
            z,
            parent_energy_fraction,
        ))

    def _handle_chi_channel(self, cand, channel, step_index):
        if channel == "chi_g":
            t2o, xo = cand.t2, cand.x
            z = self.kin.sample_true_z_chi_g(cand.t2)
            self._apply_radiation_split(cand, "g", z)
            self._record_emission(
                step_index, "chi_to_chi_g", cand.id, "chi", t2o, z, xo
            )
            return "gluon"
        if channel == "chi_gd":
            t2o, xo = cand.t2, cand.x
            if self.flavor_scheme == PAPER_PHYSICAL_MARKOV:
                source_flavor = cand.flavor
                z = self.kin.sample_true_z_chi_gd_physical(
                    cand.t2, source_flavor
                )
                target_flavor = self.kin.sample_physical_transition(
                    source_flavor
                )
            else:
                z = self.kin.sample_true_z_chi_gd(cand.t2, cand.flavor)
                target_flavor = cand.flavor
            self._apply_radiation_split(cand, "gamma_d", z)
            cand.flavor = target_flavor
            self._record_emission(
                step_index, "chi_to_chi_gamma_d", cand.id, "chi", t2o, z, xo
            )
            return "photon"
        raise ValueError(f"Unsupported chi channel: {channel}")

    def _apply_pair_split(self, cand, z, flavor, second_flavor=None):
        """把规范玻色子替换为带指定 flavor 标签的经典 χχ̄ 对。"""
        if second_flavor is None:
            second_flavor = flavor
        t2o, xo = cand.t2, cand.x
        dx_ell, dy_ell, dx_m, dy_m = self._split_2d(t2o, z, xo)
        tx, ty = cand.theta_x, cand.theta_y
        self.particles = [p for p in self.particles if p is not cand]
        self.particles.append(
            Particle(
                self.next_id,
                "chi",
                t2=z ** 2 * t2o,
                flavor=flavor,
                energy_fraction=xo * z,
                theta_x=tx + dx_ell,
                theta_y=ty + dy_ell,
            )
        )
        self.particles.append(
            Particle(
                self.next_id + 1,
                "chi",
                t2=(1.0 - z) ** 2 * t2o,
                flavor=second_flavor,
                energy_fraction=xo * (1.0 - z),
                theta_x=tx + dx_m,
                theta_y=ty + dy_m,
            )
        )
        self.next_id += 2

    def _handle_gluon_channel(self, cand, channel, step_index):
        if channel == "g_gg":
            parent_id, t2o, xo = cand.id, cand.t2, cand.x
            z = self.kin.sample_true_z_gg(cand.t2)
            self._apply_radiation_split(cand, "g", z)
            self._record_emission(
                step_index, "g_to_gg", parent_id, "g", t2o, z, xo
            )
            return "gg"
        if channel == "g_qq":
            parent_id, t2o, xo = cand.id, cand.t2, cand.x
            z = self.kin.sample_true_z_g_qq(cand.t2)
            flavor = int(np.random.randint(self.kin.N_CHARGE))
            self._apply_pair_split(cand, z, flavor)
            self._record_emission(
                step_index, "g_to_chi_chi", parent_id, "g", t2o, z, xo
            )
            return "pair"
        raise ValueError(f"Unsupported gluon channel: {channel}")

    def _handle_photon_channel(self, cand, channel, step_index):
        if channel != "gd_qq":
            raise ValueError(f"Unsupported dark-photon channel: {channel}")
        parent_id, t2o, xo = cand.id, cand.t2, cand.x
        z = self.kin.sample_true_z_gd_qq(cand.t2)
        if self.flavor_scheme == PAPER_PHYSICAL_MARKOV:
            flavor, second_flavor = self.kin.sample_physical_pair_flavors()
        else:
            flavor = self.kin.sample_photon_pair_flavor()
            second_flavor = flavor
        self._apply_pair_split(cand, z, flavor, second_flavor)
        self._record_emission(
            step_index, "gamma_d_to_chi_chi", parent_id, "gamma_d", t2o, z, xo
        )
        return "pair"

    def run_shot(self, max_steps=None, max_pairs=None):
        if max_steps is None:
            max_steps = PARAMS["MAX_STEPS"]
        if max_pairs is None:
            max_pairs = PARAMS["MAX_PAIRS"]

        n_photon = 0
        n_gluon = 0
        n_pairs = 0
        K = self.kin

        for step_index in range(1, max_steps + 1):
            for p in self.particles:
                if p.t2 > K.t2_MIN:
                    p.t2 *= K.r_step

            active = [p for p in self.particles if p.t2 > K.t2_MIN]
            if not active:
                break

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

            delta_step = K.sudakov_no_emission(total_rate)
            p_emit = 1.0 - delta_step
            if np.random.uniform() >= p_emit:
                continue

            chosen = int(np.random.choice(len(entries), p=rates / total_rate))
            cand, channel, _ = entries[chosen]

            if cand.type == "chi":
                result = self._handle_chi_channel(cand, channel, step_index)
                if result == "gluon":
                    n_gluon += 1
                elif result == "photon":
                    n_photon += 1

            elif cand.type == "g":
                result = self._handle_gluon_channel(cand, channel, step_index)
                if result == "pair":
                    n_pairs += 1
                    if n_pairs > max_pairs:
                        return None

            elif cand.type == "gamma_d":
                result = self._handle_photon_channel(cand, channel, step_index)
                if result == "pair":
                    n_pairs += 1
                    if n_pairs > max_pairs:
                        return None

        if self.flavor_scheme == PAPER_PHYSICAL_MARKOV:
            # Particle id=1 is the original physical-flavor fermion.  Radiation
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


# --- 4. 统计运行 ---

def _encode_final_particles(particles):
    return "|".join(
        f"{p.id}:{p.type}:{p.x:.17g}:{p.theta_x:.17g}:{p.theta_y:.17g}"
        for p in particles
    )


def _encode_final_flavors(particles):
    """附加经典 flavor 信息，不改变 final_particles 的既有编码。"""
    return "|".join(
        f"{p.id}:{p.flavor}"
        for p in particles
        if p.type == "chi"
    )


def _encode_emissions(emissions):
    return "|".join(
        f"{emission_index}:{step_index}:{channel}:{parent_id}:{parent_type}:"
        f"{t2:.17g}:{z:.17g}:{parent_x:.17g}"
        for emission_index, (
            step_index, channel, parent_id, parent_type, t2, z, parent_x
        ) in enumerate(emissions, start=1)
    )


def _run_single_shot(seed):
    np.random.seed(seed)
    shower = CWShower12()
    result = shower.run_shot()
    if result is None:
        return None
    n_ph, n_gl, mass_state = result
    return (
        n_ph,
        n_gl,
        mass_state,
        _encode_final_particles(shower.particles),
        _encode_final_flavors(shower.particles),
        _encode_emissions(shower.emissions),
    )


def _print_header(kin):
    print("=" * 70)
    print("  SU(3) x U(1)_D CW Pure-Classical Parton Shower (finite Mq)")
    print("=" * 70)
    print(f"  SU(3): N_C={kin.N_C}, N_F={kin.N_F}")
    print(f"    alpha* = {kin.alpha_star:.4f},  gamma = {kin.gamma_anom:.4f}")
    print(f"  U(1)_D: alpha_D={kin.alpha_D}, N_charge={kin.N_CHARGE}")
    print(f"    Q = [{', '.join(f'{q:.2f}' for q in kin.Q_charges)}]")
    print(f"    alpha_D*Q2_max = {kin.alpha_D * kin.q2_max:.4f}")
    flavor_scheme = PARAMS.get(
        "CLASSICAL_FLAVOR_SCHEME", PAPER_PHYSICAL_MARKOV
    )
    print(f"  Classical flavor scheme: {flavor_scheme}")
    if flavor_scheme == PAPER_PHYSICAL_MARKOV:
        initial_flavor = PARAMS.get("INITIAL_PHYSICAL_FLAVOR", 0)
        print(f"    initial physical flavor = {initial_flavor}")
    print(f"  M_q = {kin.M_q}  (M_q/Lambda = {kin.M_q/kin.lambda_d:.3f})")
    print(f"  t2_MIN = {kin.t2_MIN:.2e}")
    print("  Emission decision: P(split) = 1 - exp(-sum integrated rates)")
    print(f"  Shots = {PARAMS['N_SHOTS']},  Workers = {PARAMS['N_WORKERS']}")
    print("=" * 70)


def _run_parallel(n_shots, n_workers, random_seed):
    ss = np.random.SeedSequence(random_seed)
    seeds = [s.generate_state(1)[0] for s in ss.spawn(n_shots)]

    t0 = time.time()
    results = []
    n_fail = 0
    done = 0
    progress = PARAMS["PROGRESS_INTERVAL"]

    with mp.Pool(processes=n_workers) as pool:
        for res in pool.imap_unordered(_run_single_shot, seeds, chunksize=32):
            done += 1
            if res is None:
                n_fail += 1
            else:
                results.append(res)
            if done % progress == 0 or done == n_shots:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_shots - done) / rate if rate > 0 else 0
                print(f"  {done:>6d}/{n_shots}  "
                      f"({elapsed:.1f}s, {rate:.0f} shots/s, ETA {eta:.0f}s)")

    wall = time.time() - t0
    return results, n_fail, wall


def _simulation_parameters(kin, n_valid, n_fail, wall):
    return [
        ("mode", "classical"),
        ("classical_flavor_model", PARAMS.get(
            "CLASSICAL_FLAVOR_SCHEME", PAPER_PHYSICAL_MARKOV
        )),
        ("initial_physical_flavor", PARAMS.get("INITIAL_PHYSICAL_FLAVOR", 0)),
        ("physical_coupling_matrix", "U_diag_Q_Udagger"),
        ("mass_state_semantics", (
            "physical_flavor_of_initial_fermion"
            if PARAMS.get("CLASSICAL_FLAVOR_SCHEME", PAPER_PHYSICAL_MARKOV)
            == PAPER_PHYSICAL_MARKOV
            else "mixing_matrix_measurement_of_initial_interaction_eigenstate"
        )),
        ("sudakov_scheme", "stepwise_exp_integrated_rate"),
        ("qiskit_required", False),
        ("N_SHOTS", PARAMS["N_SHOTS"]),
        ("valid_events", n_valid),
        ("dropped_events", n_fail),
        ("wall_time_s", f"{wall:.6f}"),
        ("N_WORKERS", PARAMS["N_WORKERS"]),
        ("PROGRESS_INTERVAL", PARAMS["PROGRESS_INTERVAL"]),
        ("N_C", kin.N_C),
        ("N_F", kin.N_F),
        ("N_CHARGE", kin.N_CHARGE),
        ("MIXING_STRENGTH", kin.mixing_strength),
        ("ALPHA_D", kin.alpha_D),
        ("Q_MIN", PARAMS["Q_MIN"]),
        ("Q_MAX", PARAMS["Q_MAX"]),
        ("t2_0", kin.t2_0),
        ("E0", PARAMS["E0"]),
        ("LAMBDA_D", kin.lambda_d),
        ("M_Q", kin.M_q),
        ("t2_MIN", kin.t2_MIN),
        ("R_STEP", PARAMS["R_STEP"]),
        ("T2_END", PARAMS.get("T2_END")),
        ("T2_END_GRID", PARAMS.get("T2_END_GRID")),
        ("OVERESTIMATE_FACTOR", PARAMS["OVERESTIMATE_FACTOR"]),
        ("MAX_STEPS", PARAMS["MAX_STEPS"]),
        ("MAX_PAIRS", PARAMS["MAX_PAIRS"]),
        ("ENABLE_Q_TO_Q_G", PARAMS["ENABLE_Q_TO_Q_G"]),
        ("ENABLE_Q_TO_Q_DP", PARAMS["ENABLE_Q_TO_Q_DP"]),
        ("ENABLE_G_TO_GG", PARAMS["ENABLE_G_TO_GG"]),
        ("ENABLE_G_TO_QQ", PARAMS["ENABLE_G_TO_QQ"]),
        ("ENABLE_DP_TO_QQ", PARAMS["ENABLE_DP_TO_QQ"]),
    ]


def run_statistics():
    kin = CWKinematics12()
    _print_header(kin)

    n_shots = PARAMS["N_SHOTS"]
    n_workers = PARAMS["N_WORKERS"]
    random_seed = resolve_random_seed(PARAMS.get("RANDOM_SEED"))
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    results, n_fail, wall = _run_parallel(n_shots, n_workers, random_seed)

    if not results:
        print("No valid events")
        return

    n_ph = np.array([r[0] for r in results], dtype=int)
    n_gl = np.array([r[1] for r in results], dtype=int)
    mass_state = np.array([r[2] for r in results], dtype=int)
    final_particles = [r[3] for r in results]
    final_flavors = [r[4] for r in results]
    final_emissions = [r[5] for r in results]

    print(f"\n  Valid: {len(results)},  Dropped: {n_fail}")
    print(f"  Wall time: {wall:.1f}s  ({len(results)/wall:.0f} valid shots/s)")

    saved = save_simulation_results(
        base_dir=Path(__file__).resolve().parent,
        params=PARAMS,
        mode="classical",
        simulation_metadata=dict(
            _simulation_parameters(kin, len(mass_state), n_fail, wall)
        ),
        n_ph=n_ph,
        n_gl=n_gl,
        mass_state=mass_state,
        final_particles=final_particles,
        final_flavors=final_flavors,
        final_emissions=final_emissions,
        dropped_events=n_fail,
        wall_time_s=wall,
        random_seed=random_seed,
        started_at=started_at,
    )
    print(f"  Experiment: {saved['experiment_id']}")
    print(f"  Run: {saved['run_id']}")
    print(f"  Data saved: {saved['run_dir']}")
    print(f"  Experiment index: {saved['index_path']}")

    print("\n" + "=" * 70)
    print("  Done")
    print("=" * 70)


if __name__ == "__main__":
    run_statistics()
