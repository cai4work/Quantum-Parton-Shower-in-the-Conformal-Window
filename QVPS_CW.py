"""
SU(3) × U(1)_D 共形窗口量子 Parton Shower — 12 种电荷, 有限夸克质量 (Qiskit)

基于 arXiv:2310.19881 (QVPS), arXiv:2502.18566 (共形窗口簇射)
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


QuantumCircuit = None
Statevector = None


def _require_qiskit():
    global QuantumCircuit, Statevector
    if QuantumCircuit is not None and Statevector is not None:
        return
    try:
        from qiskit import QuantumCircuit as _QuantumCircuit
        from qiskit.quantum_info import Statevector as _Statevector
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Qiskit is required for QVPS_CW.py simulations. "
            "Plotting can still import CWKinematics12 without Qiskit."
        ) from exc
    QuantumCircuit = _QuantumCircuit
    Statevector = _Statevector


# --- 1. CWKinematics12 — 运动学、耦合常数、过高估计 ---

class CWKinematics12:
    """
    物理运动学: SU(3) 两环跑动耦合, U(1)_D 味道依赖耦合,
    Altarelli-Parisi 分裂函数, veto 过高估计分布, 可调味道混合矩阵.
    """

    def __init__(self):
        N_C = PARAMS["N_C"]
        N_F = PARAMS["N_F"]

        self.N_C = N_C
        self.N_F = N_F
        self.N_CHARGE = PARAMS["N_CHARGE"]
        self.t2_0 = PARAMS["t2_0"]
        self.lambda_d = PARAMS["LAMBDA_D"]
        self.alpha_D = PARAMS["ALPHA_D"]
        self.r_step = PARAMS["R_STEP"]

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
        self.ALPHA_S_OVER = self.alpha_star * ovf
        self.ALPHA_D_OVER = self.alpha_D * self.q2_max * ovf
        self.ALPHA_D_SUM_OVER = self.alpha_D * self.q2_sum * ovf

        # interaction eigenbasis -> physical/mass basis mixing matrix.
        N = self.N_CHARGE
        self.mixing_strength = float(PARAMS["MIXING_STRENGTH"])
        self.mixing_matrix = build_flavor_mixing_matrix(
            N, self.mixing_strength
        )

        # Embed into the 16-dimensional four-qubit flavor register.
        self.mixing_unitary_16 = np.eye(16, dtype=complex)
        self.mixing_unitary_16[:N, :N] = self.mixing_matrix

    # --- 耦合常数 ---

    def alpha_s(self, mu2):
        """SU(3) Lambert W 两环跑动耦合 α_s(μ²)"""
        if mu2 <= 0.0:
            return self.alpha_star
        v = (1.0 / np.e) * (mu2 / self.LAMBDA2) ** self.gamma_anom
        W = float(np.real(lambertw(v, k=0)))
        return max(self.alpha_star / (1.0 + W), 0.0)

    def alpha_d_charge(self, charge_idx):
        """固定 U(1)_D 有效耦合 alpha_D Q_k^2。"""
        return self.alpha_D * self.q2_charges[charge_idx]

    def alpha_d_sum(self):
        """固定 U(1)_D 耦合对所有电荷本征态求和。"""
        return self.alpha_D * self.q2_sum

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

    def gover_chi_g_tot(self, t2):
        zn, zx = self.z_limits(t2)
        if zn >= zx:
            return 0.0
        return self.neg_ln_r * self.ALPHA_S_OVER * self.C_F / np.pi * np.log((1 - zn) / zn)

    def gover_chi_gd_at_z(self, t2, z):
        return self.neg_ln_r * self.ALPHA_D_OVER / (2 * np.pi) * 2.0 / (1.0 - z)

    def gover_chi_gd_tot(self, t2):
        zn, zx = self.z_limits(t2)
        if zn >= zx:
            return 0.0
        return self.neg_ln_r * self.ALPHA_D_OVER / np.pi * np.log((1 - zn) / zn)

    def gover_gg_at_z(self, t2, z):
        return self.neg_ln_r * self.ALPHA_S_OVER * self.C_A / np.pi * (1.0 / z + 1.0 / (1 - z))

    def gover_gg_tot(self, t2):
        zn, zx = self.z_limits(t2)
        if zn >= zx:
            return 0.0
        return self.neg_ln_r * self.ALPHA_S_OVER * self.C_A / np.pi * 2.0 * np.log((1 - zn) / zn)

    def gover_g_qq_at_z(self, t2, z):
        return self.neg_ln_r * self.ALPHA_S_OVER * self.N_F * self.T_F / np.pi

    def gover_g_qq_tot(self, t2):
        zn, zx = self.z_limits(t2)
        if zn >= zx:
            return 0.0
        return self.neg_ln_r * self.ALPHA_S_OVER * self.N_F * self.T_F / np.pi * (zx - zn)

    def gover_gd_qq_at_z(self, t2, z):
        return self.neg_ln_r * self.ALPHA_D_SUM_OVER / np.pi

    def gover_gd_qq_tot(self, t2):
        zn, zx = self.z_limits(t2)
        if zn >= zx:
            return 0.0
        return self.neg_ln_r * self.ALPHA_D_SUM_OVER / np.pi * (zx - zn)

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

    # --- Veto 旋转角 ---

    @staticmethod
    def theta_hat(x):
        """接受概率 x → RY 旋转角 θ, 使 sin²(θ/2) = x"""
        x = float(np.clip(x, 0.0, 1.0 - 1e-12))
        return 2.0 * np.arctan(np.sqrt(x / (1.0 - x)))

    def angle_chi_gluon(self, t2, z, g_tot):
        mu2_chi_to_chi_g = self.mu2_chi_to_chi_g(t2, z)
        g_true = self.neg_ln_r * self.alpha_s(mu2_chi_to_chi_g) * self.C_F / (2 * np.pi) * self.p_ff(z)
        denom = max(self.gover_chi_g_at_z(t2, z) / g_tot, 1e-15)
        return self.theta_hat(g_true / denom)

    def angles_chi_photon(self, t2, z, g_tot):
        """χ→χγ_D: 返回 N_CHARGE 个角, 每种本征态耦合不同 → 干涉源"""
        denom = max(self.gover_chi_gd_at_z(t2, z) / g_tot, 1e-15)
        thetas = np.empty(self.N_CHARGE)
        for k in range(self.N_CHARGE):
            g_k = (self.neg_ln_r * self.alpha_d_charge(k)
                   / (2 * np.pi) * self.p_ff(z))
            thetas[k] = self.theta_hat(g_k / denom)
        return thetas

    def angle_gg(self, t2, z, g_tot):
        mu2_g_to_gg = self.mu2_g_to_gg(t2, z)
        g_true = self.neg_ln_r * self.alpha_s(mu2_g_to_gg) * self.C_A / (2 * np.pi) * self.p_gg(z)
        denom = max(self.gover_gg_at_z(t2, z) / g_tot, 1e-15)
        return self.theta_hat(g_true / denom)

    def angle_g_qq(self, t2, z, g_tot):
        mu2_g_to_chi_chibar = self.mu2_g_to_chi_chibar(t2, z)
        g_true = self.neg_ln_r * self.N_F * self.alpha_s(mu2_g_to_chi_chibar) * self.T_F / (2 * np.pi) * self.p_fA(z)
        denom = max(self.gover_g_qq_at_z(t2, z) / g_tot, 1e-15)
        return self.theta_hat(g_true / denom)

    def angle_gd_qq(self, t2, z, g_tot):
        g_true = (self.neg_ln_r * self.alpha_d_sum()
                  / (2 * np.pi) * self.p_fA(z))
        denom = max(self.gover_gd_qq_at_z(t2, z) / g_tot, 1e-15)
        return self.theta_hat(g_true / denom)

    # --- 纠缠系数 (玻色子 → 费米子对) ---

    def entangle_phi_gluon(self):
        return np.pi / 2.0

    def entangle_coeffs_photon(self):
        """γ_D→χχ̄ 纠缠系数: c_k = √(α_D·Q_k²) / √(Σ α_D·Q_k²)"""
        norm = np.sqrt(self.q2_sum)
        return self.Q_charges / norm


# --- 2. QuantumStateManager12 — Qiskit Statevector 量子态管理 ---

class QuantumStateManager12:

    QPF = PARAMS["QUBITS_PER_FERMION"]

    def __init__(self, initial_flavor_state=None):
        _require_qiskit()
        n_charge = PARAMS["N_CHARGE"]
        self.n_flavor_qubits = self.QPF

        if initial_flavor_state is None:
            mixing_matrix = build_flavor_mixing_matrix(
                n_charge, PARAMS["MIXING_STRENGTH"]
            )
            initial_physical_flavor = int(
                PARAMS.get("INITIAL_PHYSICAL_FLAVOR", 0)
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
        return self.n_flavor_qubits

    def sv_evolve(self, data, qc):
        norm = float(np.linalg.norm(data))
        if norm > 1e-12:
            data = data / norm
        return np.array(Statevector(data).evolve(qc).data)

    def sv_measure_reset_emission(self, data, e_idx):
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

    # 味道无关 veto: qc.ry(θ, e_idx)

    def try_boson_veto(self, theta):
        e_idx = self.emission_idx()
        qc = QuantumCircuit(self.n_flavor_qubits + 1)
        qc.ry(theta, e_idx)

        self.state_data = self.sv_evolve(self.state_data, qc)
        outcome, self.state_data = self.sv_measure_reset_emission(
            self.state_data, e_idx
        )
        return outcome

    # 味道依赖 veto: qc.unitary(U_32×32) → 干涉

    @staticmethod
    def _build_flavor_veto_unitary(thetas, n_charge):
        """构造 32×32 分块对角酉矩阵: 味道 k 的发射比特施加 RY(θ_k)"""
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

        U_veto = self._build_flavor_veto_unitary(thetas, PARAMS["N_CHARGE"])

        target_qubits = list(range(fermion_start, fermion_start + self.QPF))
        target_qubits.append(e_idx)
        qc.unitary(U_veto, target_qubits, label="FlavorVeto")

        self.state_data = self.sv_evolve(self.state_data, qc)
        outcome, self.state_data = self.sv_measure_reset_emission(
            self.state_data, e_idx
        )
        return outcome

    # 扩展态向量: 添加纠缠费米子对

    def sv_expand_with_fermion_pair_12(self, data, n_flavor_qubits, coeffs):
        """扩展态向量, 添加纠缠对 Σ_k coeffs[k] |k⟩₁|k⟩₂"""
        n_new = 2 * self.QPF
        old_dim = len(data)
        new_dim = old_dim * (2 ** n_new)
        new_data = np.zeros(new_dim, dtype=complex)

        fmask = (1 << n_flavor_qubits) - 1
        n_charge = PARAMS["N_CHARGE"]

        for i in range(old_dim):
            amp = data[i]
            if abs(amp) < 1e-15:
                continue
            fv = i & fmask
            ev = (i >> n_flavor_qubits) & 1
            ec = ev << (n_flavor_qubits + n_new)

            for k in range(n_charge):
                pair_bits = (k << n_flavor_qubits) | (k << (n_flavor_qubits + self.QPF))
                new_data[fv | pair_bits | ec] += coeffs[k] * amp

        return new_data

    # 对产生接口

    def apply_boson_split_gluon(self):
        n_charge = PARAMS["N_CHARGE"]
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

    # 质量基底测量

    def measure_q0_mass_basis(self, mixing_unitary_16):
        n_total = self.n_flavor_qubits + 1
        qc = QuantumCircuit(n_total)
        qc.unitary(mixing_unitary_16, [0, 1, 2, 3], label="Mixing12")

        rotated = self.sv_evolve(self.state_data, qc)

        N = PARAMS["N_CHARGE"]
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


# --- 4. CWShower12 — 共形窗口量子簇射主循环 ---

class CWShower12:

    def __init__(self):
        self.kin = CWKinematics12()
        self.initial_physical_flavor = int(
            PARAMS.get("INITIAL_PHYSICAL_FLAVOR", 0)
        )
        if not 0 <= self.initial_physical_flavor < self.kin.N_CHARGE:
            raise ValueError(
                "INITIAL_PHYSICAL_FLAVOR must satisfy "
                f"0 <= flavor < {self.kin.N_CHARGE}"
            )
        initial_interaction_state = self.kin.mixing_matrix.conj().T[
            :, self.initial_physical_flavor
        ]
        self.qsm = QuantumStateManager12(initial_interaction_state)
        self.next_id = 1
        self.emissions = []

        self.particles = [
            Particle(self.next_id, "chi",
                     t2=self.kin.t2_0, qubit_start=0,
                     energy_fraction=1.0)
        ]
        self.next_id += 1

    def _particle_weight(self, p):
        if p.type == "chi":
            return (
                (self.kin.gover_chi_g_tot(p.t2)  if PARAMS["ENABLE_Q_TO_Q_G"]  else 0.0)
              + (self.kin.gover_chi_gd_tot(p.t2) if PARAMS["ENABLE_Q_TO_Q_DP"] else 0.0)
            )
        elif p.type == "g":
            return (
                (self.kin.gover_gg_tot(p.t2)   if PARAMS["ENABLE_G_TO_GG"] else 0.0)
              + (self.kin.gover_g_qq_tot(p.t2) if PARAMS["ENABLE_G_TO_QQ"] else 0.0)
            )
        elif p.type == "gamma_d":
            return self.kin.gover_gd_qq_tot(p.t2) if PARAMS["ENABLE_DP_TO_QQ"] else 0.0
        return 0.0

    @staticmethod
    def _split_2d(t2, z, x_parent):
        """eq.65-68 分裂运动学: 返回两个子粒子的 2D 角偏移"""
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

    def _handle_chi_split(self, cand, g_tot, step_index):
        K = self.kin
        w_g = K.gover_chi_g_tot(cand.t2)  if PARAMS["ENABLE_Q_TO_Q_G"]  else 0.0
        w_p = K.gover_chi_gd_tot(cand.t2) if PARAMS["ENABLE_Q_TO_Q_DP"] else 0.0
        w_sum = w_g + w_p
        if w_sum < 1e-15:
            return None

        if np.random.uniform() < w_g / w_sum:
            # χ → χ + g  (SU(3), 味道无关)
            z = K.sample_z_pole(cand.t2)
            theta = K.angle_chi_gluon(cand.t2, z, g_tot)
            outcome = self.qsm.try_boson_veto(theta)
            if outcome == 1:
                t2o, xo = cand.t2, cand.x
                dx_ell, dy_ell, dx_m, dy_m = self._split_2d(t2o, z, xo)
                tx, ty = cand.theta_x, cand.theta_y
                cand.t2 = z ** 2 * t2o
                cand.x = xo * z
                cand.theta_x = tx + dx_ell
                cand.theta_y = ty + dy_ell
                self.particles.append(
                    Particle(self.next_id, "g",
                             t2=(1 - z) ** 2 * t2o,
                             energy_fraction=xo * (1 - z),
                             theta_x=tx + dx_m, theta_y=ty + dy_m)
                )
                self._record_emission(
                    step_index, "chi_to_chi_g", cand.id, "chi", t2o, z, xo
                )
                self.next_id += 1
                return "gluon"
        else:
            # χ → χ + γ_D  (U(1), 味道依赖 → 干涉)
            z = K.sample_z_pole(cand.t2)
            thetas = K.angles_chi_photon(cand.t2, z, g_tot)
            outcome = self.qsm.try_fermion_veto_multi(cand.qubit_start, thetas)
            if outcome == 1:
                t2o, xo = cand.t2, cand.x
                dx_ell, dy_ell, dx_m, dy_m = self._split_2d(t2o, z, xo)
                tx, ty = cand.theta_x, cand.theta_y
                cand.t2 = z ** 2 * t2o
                cand.x = xo * z
                cand.theta_x = tx + dx_ell
                cand.theta_y = ty + dy_ell
                self.particles.append(
                    Particle(self.next_id, "gamma_d",
                             t2=(1 - z) ** 2 * t2o,
                             energy_fraction=xo * (1 - z),
                             theta_x=tx + dx_m, theta_y=ty + dy_m)
                )
                self._record_emission(
                    step_index, "chi_to_chi_gamma_d", cand.id, "chi", t2o, z, xo
                )
                self.next_id += 1
                return "photon"

        return None

    def _handle_gluon_split(self, cand, g_tot, step_index):
        K = self.kin
        w_gg = K.gover_gg_tot(cand.t2)   if PARAMS["ENABLE_G_TO_GG"] else 0.0
        w_qq = K.gover_g_qq_tot(cand.t2) if PARAMS["ENABLE_G_TO_QQ"] else 0.0
        w_sum = w_gg + w_qq
        if w_sum < 1e-15:
            return None

        if np.random.uniform() < w_gg / w_sum:
            # g → g + g
            z = K.sample_z_gg(cand.t2)
            theta = K.angle_gg(cand.t2, z, g_tot)
            outcome = self.qsm.try_boson_veto(theta)
            if outcome == 1:
                t2o, xo = cand.t2, cand.x
                dx_ell, dy_ell, dx_m, dy_m = self._split_2d(t2o, z, xo)
                tx, ty = cand.theta_x, cand.theta_y
                cand.t2 = z ** 2 * t2o
                cand.x = xo * z
                cand.theta_x = tx + dx_ell
                cand.theta_y = ty + dy_ell
                self.particles.append(
                    Particle(self.next_id, "g",
                             t2=(1 - z) ** 2 * t2o,
                             energy_fraction=xo * (1 - z),
                             theta_x=tx + dx_m, theta_y=ty + dy_m)
                )
                self._record_emission(
                    step_index, "g_to_gg", cand.id, "g", t2o, z, xo
                )
                self.next_id += 1
                return "gg"
        else:
            if not PARAMS["ENABLE_G_TO_QQ"]:
                return None
            # g → χ + χ̄
            z = K.sample_z_flat(cand.t2)
            theta = K.angle_g_qq(cand.t2, z, g_tot)
            outcome = self.qsm.try_boson_veto(theta)
            if outcome == 1:
                qubit1_start, qubit2_start = self.qsm.apply_boson_split_gluon()
                t2o, xo = cand.t2, cand.x
                self.particles = [p for p in self.particles if p is not cand]
                self.particles.append(
                    Particle(self.next_id, "chi",
                             t2=z ** 2 * t2o, qubit_start=qubit1_start,
                             energy_fraction=xo * z)
                )
                self.particles.append(
                    Particle(self.next_id + 1, "chi",
                             t2=(1 - z) ** 2 * t2o, qubit_start=qubit2_start,
                             energy_fraction=xo * (1 - z))
                )
                self._record_emission(
                    step_index, "g_to_chi_chi", cand.id, "g", t2o, z, xo
                )
                self.next_id += 2
                return "pair"

        return None

    def _handle_photon_split(self, cand, g_tot, step_index):
        if not PARAMS["ENABLE_DP_TO_QQ"]:
            return None
        K = self.kin
        z = K.sample_z_flat(cand.t2)
        theta = K.angle_gd_qq(cand.t2, z, g_tot)
        outcome = self.qsm.try_boson_veto(theta)

        if outcome == 1:
            coeffs = K.entangle_coeffs_photon()
            qubit1_start, qubit2_start = self.qsm.apply_boson_split_photon(coeffs)
            t2o, xo = cand.t2, cand.x
            self.particles = [p for p in self.particles if p is not cand]
            self.particles.append(
                Particle(self.next_id, "chi",
                         t2=z ** 2 * t2o, qubit_start=qubit1_start,
                         energy_fraction=xo * z)
            )
            self.particles.append(
                Particle(self.next_id + 1, "chi",
                         t2=(1 - z) ** 2 * t2o, qubit_start=qubit2_start,
                         energy_fraction=xo * (1 - z))
            )
            self._record_emission(
                step_index, "gamma_d_to_chi_chi", cand.id, "gamma_d", t2o, z, xo
            )
            self.next_id += 2
            return "pair"

        return None

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

            weights = np.array([self._particle_weight(p) for p in active])
            weights = np.maximum(weights, 0.0)
            g_tot = float(weights.sum())
            if g_tot < 1e-15:
                continue

            cand = active[int(np.random.choice(len(active), p=weights / g_tot))]

            if cand.type == "chi":
                result = self._handle_chi_split(cand, g_tot, step_index)
                if result == "gluon":
                    n_gluon += 1
                elif result == "photon":
                    n_photon += 1

            elif cand.type == "g":
                result = self._handle_gluon_split(cand, g_tot, step_index)
                if result == "pair":
                    n_pairs += 1
                    if n_pairs > max_pairs:
                        return None

            elif cand.type == "gamma_d":
                result = self._handle_photon_split(cand, g_tot, step_index)
                if result == "pair":
                    n_pairs += 1
                    if n_pairs > max_pairs:
                        return None

        final_mass_state = self.qsm.measure_q0_mass_basis(
            self.kin.mixing_unitary_16
        )
        return n_photon, n_gluon, final_mass_state


# --- 5. 统计运行 ---

def _encode_final_particles(particles):
    return "|".join(
        f"{p.id}:{p.type}:{p.x:.17g}:{p.theta_x:.17g}:{p.theta_y:.17g}"
        for p in particles
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
        _encode_emissions(shower.emissions),
    )


def _print_header(kin):
    print("=" * 70)
    print("  SU(3) x U(1)_D CW Quantum Parton Shower (Qiskit, finite Mq)")
    print("=" * 70)
    print(f"  SU(3): N_C={kin.N_C}, N_F={kin.N_F}")
    print(f"    alpha* = {kin.alpha_star:.4f},  gamma = {kin.gamma_anom:.4f}")
    print(f"  U(1)_D: alpha_D={kin.alpha_D}, N_charge={kin.N_CHARGE}")
    print(f"    Q = [{', '.join(f'{q:.2f}' for q in kin.Q_charges)}]")
    print(f"    alpha_D*Q2_max = {kin.alpha_D * kin.q2_max:.4f}")
    print(f"  M_q = {kin.M_q}  (M_q/Lambda = {kin.M_q/kin.lambda_d:.3f})")
    print(f"  T2_MIN = {kin.t2_MIN:.2e}")
    print(f"  Qubits/fermion = {PARAMS['QUBITS_PER_FERMION']}")
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
        ("mode", "quantum"),
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
        ("initial_physical_flavor", PARAMS.get("INITIAL_PHYSICAL_FLAVOR", 0)),
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
    final_emissions = [r[4] for r in results]

    print(f"\n  Valid: {len(results)},  Dropped: {n_fail}")
    print(f"  Wall time: {wall:.1f}s  ({len(results)/wall:.0f} valid shots/s)")

    saved = save_simulation_results(
        base_dir=Path(__file__).resolve().parent,
        params=PARAMS,
        mode="quantum",
        simulation_metadata=dict(
            _simulation_parameters(kin, len(mass_state), n_fail, wall)
        ),
        n_ph=n_ph,
        n_gl=n_gl,
        mass_state=mass_state,
        final_particles=final_particles,
        final_flavors=None,
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
