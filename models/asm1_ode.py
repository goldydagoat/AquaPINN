"""
Activated Sludge Model No. 1 (ASM1) ODE Implementation.

This module implements the benchmark Activated Sludge Model No. 1 (Henze et al., 1987; IWA 2000)
for biological wastewater treatment processes in a continuous stirred-tank reactor (CSTR).
It computes kinetic reaction rates, stoichiometric transformations, and solves the stiff
differential equations using SciPy's solve_ivp.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ASM1 State Variable Names and Indices
STATE_NAMES: List[str] = [
    "S_I",   # 0: Inert soluble organic matter (g COD / m^3)
    "S_S",   # 1: Readily biodegradable substrate (g COD / m^3)
    "X_I",   # 2: Inert particulate organic matter (g COD / m^3)
    "X_S",   # 3: Slowly biodegradable substrate (g COD / m^3)
    "X_BH",  # 4: Active heterotrophic biomass (g COD / m^3)
    "X_BA",  # 5: Active autotrophic biomass (g COD / m^3)
    "X_P",   # 6: Particulate decay products (g COD / m^3)
    "S_O",   # 7: Dissolved oxygen (g O2 / m^3)
    "S_NO",  # 8: Nitrate and nitrite nitrogen (g N / m^3)
    "S_NH",  # 9: Ammonium and ammonia nitrogen (g N / m^3)
    "S_ND",  # 10: Soluble biodegradable organic nitrogen (g N / m^3)
    "X_ND",  # 11: Particulate biodegradable organic nitrogen (g N / m^3)
    "S_ALK", # 12: Alkalinity (mol HCO3- / m^3)
]

STATE_UNITS: Dict[str, str] = {
    "S_I": "g COD/m³",
    "S_S": "g COD/m³",
    "X_I": "g COD/m³",
    "X_S": "g COD/m³",
    "X_BH": "g COD/m³",
    "X_BA": "g COD/m³",
    "X_P": "g COD/m³",
    "S_O": "g O₂/m³",
    "S_NO": "g N/m³",
    "S_NH": "g N/m³",
    "S_ND": "g N/m³",
    "X_ND": "g N/m³",
    "S_ALK": "mol HCO₃⁻/m³",
}

NUM_STATES = len(STATE_NAMES)


@dataclass
class ASM1Parameters:
    """Stoichiometric and kinetic parameters for ASM1 at neutral temperature (20 °C)."""

    # Stoichiometric parameters
    Y_H: float = 0.67       # Heterotrophic yield (g cell COD / g COD oxidized)
    Y_A: float = 0.24       # Autotrophic yield (g cell COD / g N oxidized)
    f_P: float = 0.08       # Fraction of biomass yielding particulate products
    i_XB: float = 0.086     # Mass of N per mass COD in biomass (g N / g COD)
    i_XP: float = 0.060     # Mass of N per mass COD in decay products (g N / g COD)

    # Kinetic parameters - Heterotrophs
    mu_H: float = 6.0       # Max heterotrophic specific growth rate (1/d)
    K_S: float = 20.0       # Readily biodegradable substrate half-velocity constant (g COD / m^3)
    K_OH: float = 0.20      # Heterotrophic oxygen affinity constant (g O2 / m^3)
    K_NO: float = 0.50      # Nitrate affinity constant for denitrification (g NO3-N / m^3)
    b_H: float = 0.62       # Heterotrophic decay coefficient (1/d)
    eta_g: float = 0.80     # Anoxic growth reduction factor for heterotrophs (-)

    # Kinetic parameters - Autotrophs (Nitrification)
    mu_A: float = 0.80      # Max autotrophic specific growth rate (1/d)
    K_NH: float = 1.0       # Ammonium affinity constant for nitrifiers (g NH4-N / m^3)
    K_OA: float = 0.40      # Autotrophic oxygen affinity constant (g O2 / m^3)
    b_A: float = 0.15       # Autotrophic decay coefficient (1/d)

    # Hydrolysis & Ammonification
    k_h: float = 3.0        # Maximum specific hydrolysis rate (1/d)
    K_X: float = 0.03       # Half-velocity constant for slowly biodegradable COD (g COD / g COD)
    eta_h: float = 0.40     # Anoxic hydrolysis reduction factor (-)
    k_a: float = 0.08       # Ammonification rate constant (m^3 / (g COD * d))

    # Physical / Environmental constants
    S_O_sat: float = 8.5    # Saturated dissolved oxygen concentration at 20 °C (g O2 / m^3)
    temp_celsius: float = 20.0 # Operating temperature (°C)

    def apply_temperature_correction(self, temp: float) -> "ASM1Parameters":
        """Return a new ASM1Parameters instance scaled to the target temperature using Arrhenius scaling."""
        theta_muH = 1.072
        theta_bH = 1.072
        theta_muA = 1.072
        theta_bA = 1.072
        theta_kh = 1.072
        theta_ka = 1.072
        dt = temp - 20.0

        return ASM1Parameters(
            Y_H=self.Y_H,
            Y_A=self.Y_A,
            f_P=self.f_P,
            i_XB=self.i_XB,
            i_XP=self.i_XP,
            mu_H=self.mu_H * (theta_muH ** dt),
            K_S=self.K_S,
            K_OH=self.K_OH,
            K_NO=self.K_NO,
            b_H=self.b_H * (theta_bH ** dt),
            eta_g=self.eta_g,
            mu_A=self.mu_A * (theta_muA ** dt),
            K_NH=self.K_NH,
            K_OA=self.K_OA,
            b_A=self.b_A * (theta_bA ** dt),
            k_h=self.k_h * (theta_kh ** dt),
            K_X=self.K_X,
            eta_h=self.eta_h,
            k_a=self.k_a * (theta_ka ** dt),
            S_O_sat=self.S_O_sat,
            temp_celsius=temp,
        )


def get_default_influent() -> np.ndarray:
    """Return typical domestic municipal wastewater influent concentrations (g/m^3 or mol/m^3)."""
    # [S_I, S_S, X_I, X_S, X_BH, X_BA, X_P, S_O, S_NO, S_NH, S_ND, X_ND, S_ALK]
    return np.array([
        30.0,   # S_I: Soluble inert COD
        150.0,  # S_S: Readily biodegradable COD
        50.0,   # X_I: Particulate inert COD
        200.0,  # X_S: Slowly biodegradable COD
        30.0,   # X_BH: Heterotrophic biomass in feed
        0.0,    # X_BA: Autotrophic biomass in feed
        0.0,    # X_P: Inert decay products
        0.0,    # S_O: Dissolved oxygen (anoxic/anaerobic raw feed)
        0.0,    # S_NO: Nitrate
        35.0,   # S_NH: Ammonium
        10.0,   # S_ND: Soluble organic nitrogen
        15.0,   # X_ND: Particulate organic nitrogen
        5.0,    # S_ALK: Alkalinity (mol/m^3)
    ], dtype=np.float64)


def get_default_initial_conditions() -> np.ndarray:
    """Return typical steady-state aeration basin initial concentrations (g/m^3)."""
    return np.array([
        30.0,    # S_I
        5.0,     # S_S
        1100.0,  # X_I
        60.0,    # X_S
        2500.0,  # X_BH (active sludge heterotrophic biomass)
        150.0,   # X_BA (nitrifying biomass)
        450.0,   # X_P
        2.2,     # S_O (aerobic tank DO level)
        12.0,    # S_NO (nitrate from nitrification)
        1.5,     # S_NH (low effluent ammonium under good nitrification)
        1.0,     # S_ND
        4.0,     # X_ND
        4.0,     # S_ALK
    ], dtype=np.float64)


class ASM1Simulator:
    """
    CSTR Activated Sludge Bioreactor Simulator driven by ASM1 kinetics.
    
    Supports dynamic influent, oxygen mass transfer (aeration KLa control),
    hydraulic retention time (HRT), and reaction stoichiometric transformations.
    """

    def __init__(
        self,
        params: Optional[ASM1Parameters] = None,
        volume: float = 1000.0,  # m^3
        q_in: float = 1000.0,    # m^3 / day (HRT = V/Q = 1.0 day = 24h)
        kla: float = 120.0,      # Aeration coefficient (1/day) ~ 5.0 1/hr
    ) -> None:
        self.params = params or ASM1Parameters()
        self.volume = max(volume, 1.0)
        self.q_in = max(q_in, 0.0)
        self.kla = max(kla, 0.0)

    @property
    def hrt(self) -> float:
        """Hydraulic Retention Time in days (HRT = V / Q)."""
        return self.volume / max(self.q_in, 1e-6)

    @property
    def dilution_rate(self) -> float:
        """Dilution rate D = Q / V (1/day)."""
        return self.q_in / self.volume

    def compute_process_rates(
        self, y: np.ndarray, params: Optional[ASM1Parameters] = None
    ) -> np.ndarray:
        """
        Compute the 8 ASM1 kinetic process rates (rho_1 to rho_8).
        
        y: State vector of 13 biochemical concentrations
        Returns: np.ndarray of shape (8,)
        """
        p = params or self.params
        
        # Enforce non-negative states in kinetic Monod terms to prevent division/log anomalies
        c = np.maximum(y, 0.0)
        
        S_I, S_S, X_I, X_S, X_BH, X_BA, X_P, S_O, S_NO, S_NH, S_ND, X_ND, S_ALK = c

        rho = np.zeros(8, dtype=np.float64)

        # 1. Aerobic growth of heterotrophs
        monod_ss = S_S / (p.K_S + S_S + 1e-9)
        monod_so_h = S_O / (p.K_OH + S_O + 1e-9)
        rho[0] = p.mu_H * monod_ss * monod_so_h * X_BH

        # 2. Anoxic growth of heterotrophs (Denitrification)
        inhibition_so_h = p.K_OH / (p.K_OH + S_O + 1e-9)
        monod_sno = S_NO / (p.K_NO + S_NO + 1e-9)
        rho[1] = p.mu_H * p.eta_g * monod_ss * inhibition_so_h * monod_sno * X_BH

        # 3. Aerobic growth of autotrophs (Nitrification)
        monod_snh = S_NH / (p.K_NH + S_NH + 1e-9)
        monod_so_a = S_O / (p.K_OA + S_O + 1e-9)
        rho[2] = p.mu_A * monod_snh * monod_so_a * X_BA

        # 4. Decay of heterotrophs
        rho[3] = p.b_H * X_BH

        # 5. Decay of autotrophs
        rho[4] = p.b_A * X_BA

        # 6. Ammonification of soluble organic nitrogen
        rho[5] = p.k_a * S_ND * X_BH

        # 7. Hydrolysis of particulate organics
        ratio_xs_xbh = X_S / (X_BH + 1e-9)
        monod_xs = ratio_xs_xbh / (p.K_X + ratio_xs_xbh + 1e-9)
        hydrolysis_factor = monod_so_h + p.eta_h * inhibition_so_h * monod_sno
        rho[6] = p.k_h * monod_xs * hydrolysis_factor * X_BH

        # 8. Hydrolysis of particulate organic nitrogen
        ratio_xnd_xs = X_ND / (X_S + 1e-9)
        rho[7] = rho[6] * ratio_xnd_xs

        return rho

    def compute_reaction_rates(
        self, y: np.ndarray, params: Optional[ASM1Parameters] = None
    ) -> np.ndarray:
        """
        Compute net biochemical reaction rate vector r = S * rho (shape: 13,).
        """
        p = params or self.params
        rho = self.compute_process_rates(y, p)
        r = np.zeros(NUM_STATES, dtype=np.float64)

        # 0: S_I (Inert soluble organic matter - unreactive)
        r[0] = 0.0

        # 1: S_S (Readily biodegradable substrate)
        r[1] = - (1.0 / p.Y_H) * rho[0] - (1.0 / p.Y_H) * rho[1] + rho[6]

        # 2: X_I (Inert particulate organic matter - unreactive)
        r[2] = 0.0

        # 3: X_S (Slowly biodegradable substrate)
        r[3] = (1.0 - p.f_P) * rho[3] + (1.0 - p.f_P) * rho[4] - rho[6]

        # 4: X_BH (Active heterotrophic biomass)
        r[4] = rho[0] + rho[1] - rho[3]

        # 5: X_BA (Active autotrophic biomass)
        r[5] = rho[2] - rho[4]

        # 6: X_P (Particulate decay products)
        r[6] = p.f_P * rho[3] + p.f_P * rho[4]

        # 7: S_O (Dissolved oxygen reaction sink)
        r[7] = - ((1.0 - p.Y_H) / p.Y_H) * rho[0] - ((4.57 - p.Y_A) / p.Y_A) * rho[2]

        # 8: S_NO (Nitrate and nitrite nitrogen)
        r[8] = - ((1.0 - p.Y_H) / (2.86 * p.Y_H)) * rho[1] + (1.0 / p.Y_A) * rho[2]

        # 9: S_NH (Ammonium and ammonia nitrogen)
        r[9] = (
            - p.i_XB * rho[0]
            - p.i_XB * rho[1]
            - (p.i_XB + 1.0 / p.Y_A) * rho[2]
            + rho[5]
        )

        # 10: S_ND (Soluble biodegradable organic nitrogen)
        r[10] = - rho[5] + rho[7]

        # 11: X_ND (Particulate biodegradable organic nitrogen)
        r[11] = (
            (p.i_XB - p.f_P * p.i_XP) * rho[3]
            + (p.i_XB - p.f_P * p.i_XP) * rho[4]
            - rho[7]
        )

        # 12: S_ALK (Alkalinity)
        r[12] = (
            - (p.i_XB / 14.0) * rho[0]
            + ((1.0 - p.Y_H) / (14.0 * 2.86 * p.Y_H) - p.i_XB / 14.0) * rho[1]
            - (p.i_XB / 14.0 + 1.0 / (7.0 * p.Y_A)) * rho[2]
            + (1.0 / 14.0) * rho[5]
        )

        return r

    def reactor_derivatives(
        self,
        t: float,
        y: np.ndarray,
        influent: np.ndarray,
        kla: float,
        params: Optional[ASM1Parameters] = None,
    ) -> np.ndarray:
        """
        Evaluate CSTR differential equations dy/dt.
        
        Mass balance: dy/dt = D * (y_in - y) + r(y) + [aeration_transfer]
        """
        p = params or self.params
        d = self.dilution_rate
        r = self.compute_reaction_rates(y, p)

        # Dilution and hydraulic transport
        dydt = d * (influent - y) + r

        # Aeration mass transfer for Dissolved Oxygen: KLa * (S_O_sat - S_O)
        so_current = max(y[7], 0.0)
        aeration_rate = kla * (p.S_O_sat - so_current)
        dydt[7] += aeration_rate

        return dydt

    def simulate(
        self,
        t_span: Tuple[float, float] = (0.0, 5.0),  # days
        num_points: int = 500,
        y0: Optional[np.ndarray] = None,
        influent: Optional[np.ndarray] = None,
        kla: Optional[float] = None,
        method: str = "Radau",
    ) -> pd.DataFrame:
        """
        Integrate the ASM1 ODEs over t_span using SciPy solve_ivp.
        
        Returns a rich pandas DataFrame with concentrations and diagnostic metrics.
        """
        if y0 is None:
            y0 = get_default_initial_conditions()
        if influent is None:
            influent = get_default_influent()
        if kla is None:
            kla = self.kla

        t_eval = np.linspace(t_span[0], t_span[1], num_points)

        # Stiff solvers: Radau or BDF handle fast Monod growth/decay transitions best
        sol = solve_ivp(
            fun=lambda t, y: self.reactor_derivatives(t, y, influent, kla, self.params),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
            rtol=1e-5,
            atol=1e-6,
        )

        if not sol.success:
            # Fallback to RK45 if stiff solver encountered a step issue
            sol = solve_ivp(
                fun=lambda t, y: self.reactor_derivatives(t, y, influent, kla, self.params),
                t_span=t_span,
                y0=y0,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-4,
                atol=1e-5,
            )

        data = {"time": sol.t}
        for idx, name in enumerate(STATE_NAMES):
            data[name] = np.maximum(sol.y[idx], 0.0)

        df = pd.DataFrame(data)

        # Compute key aggregated wastewater metrics
        df["COD_soluble"] = df["S_I"] + df["S_S"]
        df["COD_particulate"] = df["X_I"] + df["X_S"] + df["X_BH"] + df["X_BA"] + df["X_P"]
        df["COD_total"] = df["COD_soluble"] + df["COD_particulate"]
        df["TN"] = (
            df["S_NO"]
            + df["S_NH"]
            + df["S_ND"]
            + df["X_ND"]
            + self.params.i_XB * (df["X_BH"] + df["X_BA"])
            + self.params.i_XP * df["X_P"]
        )
        df["TIN"] = df["S_NH"] + df["S_NO"]  # Total Inorganic Nitrogen

        # Removal efficiencies based on influent
        cod_in_total = influent[0] + influent[1] + influent[2] + influent[3]
        s_s_in = influent[1]
        nh_in = influent[9]

        df["COD_removal_pct"] = np.clip(100.0 * (1.0 - df["COD_total"] / max(cod_in_total, 1e-3)), 0.0, 100.0)
        df["SS_removal_pct"] = np.clip(100.0 * (1.0 - df["S_S"] / max(s_s_in, 1e-3)), 0.0, 100.0)
        df["NH_removal_pct"] = np.clip(100.0 * (1.0 - df["S_NH"] / max(nh_in, 1e-3)), 0.0, 100.0)

        return df


def simulate_asm1(
    days: float = 5.0,
    num_points: int = 250,
    q_in: float = 1000.0,
    volume: float = 1000.0,
    kla: float = 120.0,
    influent_override: Optional[Dict[str, float]] = None,
    temp_celsius: float = 20.0,
    initial_conditions: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Convenience functional API to run an ASM1 simulation run.
    """
    params = ASM1Parameters().apply_temperature_correction(temp_celsius)
    simulator = ASM1Simulator(params=params, volume=volume, q_in=q_in, kla=kla)

    influent = get_default_influent()
    if influent_override:
        for name, val in influent_override.items():
            if name in STATE_NAMES:
                influent[STATE_NAMES.index(name)] = float(val)

    return simulator.simulate(
        t_span=(0.0, days),
        num_points=num_points,
        y0=initial_conditions,
        influent=influent,
        kla=kla,
    )
