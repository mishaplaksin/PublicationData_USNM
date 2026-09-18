#!/usr/bin/env python3
"""
Faithful Python port of the BLS (bilayer sonophore) ODE used in the NICE model,
transcribed from `H&H Simulations/NICE RS_FS_LTS/Sonophore.m` in this repository
(Plaksin, Shoham & Kimmel, Phys. Rev. X 4, 011004 (2014)).

Purpose here: provide the quantitative target that the coarse-grained MD in
../src/cgmd.c is compared against, and to report the *gas content* of the
intramembrane space implied by the model at its own operating point.

State vector x = [Z, dZ/dt, n_a]
  Z    : leaflet-centre displacement from flat configuration [m]
  n_a  : moles of air inside the intramembrane cavity [mol]
"""

import argparse
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ----------------------------------------------------------------------------
# Parameters, verbatim from Sonophore.m / LTS_RS_FS_combined_model.m
# ----------------------------------------------------------------------------
AR = 1.0e5          # attraction/repulsion coefficient between leaflets [Pa]
DELTA0 = 2.0e-9     # membrane (leaflet pair) thickness [m]
MUS = 0.035         # membrane dynamic viscosity [Pa s]
M_REP = 5.0         # repulsion exponent [-]
N_ATT = 3.3         # attraction exponent [-]
DA = 3.0e-9         # air diffusion coefficient in water [m^2/s]
RHO = 1028.0        # water density [kg/m^3]
MUL = 0.7e-3        # water dynamic viscosity [Pa s]
CA = 0.62           # air saturation concentration in water [mol/m^3]
P0 = 101325.0       # static pressure [Pa]
TEMP_K = 273.15 + 36.0
RG = 8.314          # gas constant [m^3 Pa /(K mol)]
ZETA = 0.5e-9       # water/membrane boundary-layer length [m]
KS = 0.12           # area compression modulus [N/m]
A_BLS = 32.0e-9     # leaflet boundary radius [m]
EPS0 = 8.8541e-12   # vacuum permittivity [F/m]
DELTA_OLD = 1.4e-9  # constant gap length in the molecular-force term [m]

# Cell-specific initial gaps (LTS, RS, FS) from LTS_RS_FS_combined_model.m
DELTA_N = {"LTS": 1.303e-9, "RS": 1.2553e-9, "FS": 1.2567e-9}
# Initial membrane potentials [V] and resting capacitance [F/m^2]
VM0 = {"LTS": -0.0539633, "RS": -0.0719107, "FS": -0.0713872}
CM0 = 1.0e-2

# PRX 2014 ultrasound stimulus
F_US = 0.69e6       # [Hz]
PA_US = 320.608e3   # [Pa]

NAV = 6.02214076e23


def cavity_volume(Z, delta, a=A_BLS):
    """Intramembrane cavity volume, as implied by the Pin expression in Sonophore.m."""
    return np.pi * a**2 * delta * (1.0 + Z / (3.0 * delta) * ((Z / a) ** 2 + 3.0))


def p_in(n_a, Z, delta, a=A_BLS):
    return n_a * RG * TEMP_K / cavity_volume(Z, delta, a)


def p_molecular(Z, delta, a=A_BLS):
    """Leaflet attraction/repulsion pressure (the four 'Par' terms of Sonophore.m)."""
    R = (a**2 + Z**2) / (2.0 * Z)
    d2 = 2.0 * Z + delta
    c = 2.0 * (Z**2 + a**2)
    p1 = -(AR * DELTA_OLD**M_REP) / (c * (2.0 - M_REP)) * (delta ** (2.0 - M_REP) - d2 ** (2.0 - M_REP))
    p2 = -(AR * DELTA_OLD**M_REP) * (2.0 * R - d2) / (c * (1.0 - M_REP)) * (
        delta ** (1.0 - M_REP) - d2 ** (1.0 - M_REP))
    p3 = (AR * DELTA_OLD**N_ATT) / (c * (2.0 - N_ATT)) * (delta ** (2.0 - N_ATT) - d2 ** (2.0 - N_ATT))
    p4 = (AR * DELTA_OLD**N_ATT) * (2.0 * R - d2) / (c * (1.0 - N_ATT)) * (
        delta ** (1.0 - N_ATT) - d2 ** (1.0 - N_ATT))
    return p1 + p2 + p3 + p4


def p_tension(Z, a=A_BLS):
    """Ps: equivalent pressure of the elastic membrane tension."""
    return 4.0 * KS * Z**3 / (a**2 * (a**2 + Z**2))


def p_electrostatic(Q, Z, a=A_BLS):
    """Pat: electrostatic attraction between the charged leaflets."""
    return -(Q**2) / (2.0 * EPS0) * a**2 / (a**2 + Z**2)


def rhs(t, x, delta, Q, a, pa, omega):
    Z, dZ, n_a = x
    Pin = p_in(n_a, Z, delta, a)
    R = (a**2 + Z**2) / (2.0 * Z)
    Ps = p_tension(Z, a)
    Par = p_molecular(Z, delta, a)
    Pat = p_electrostatic(Q, Z, a)
    Pac = pa * np.sin(omega * t)
    absR = abs(R)
    damping = 4.0 / absR * dZ * (3.0 * DELTA0 * MUS / absR + MUL)
    d2Z = (1.0 / (RHO * absR)) * (Pin + Par + Pat - P0 + Pac - Ps - damping) - 3.0 / (2.0 * R) * dZ**2
    dn = (2.0 * np.pi * (a**2 + Z**2) * DA / ZETA) * (CA - Pin / (P0 / CA))
    return [dZ, d2Z, dn]


def run(cell="RS", a=A_BLS, pa=PA_US, f=F_US, n_cycles=40, delta=None, Q=None):
    delta = DELTA_N[cell] if delta is None else delta
    Q = CM0 * VM0[cell] if Q is None else Q
    omega = 2.0 * np.pi * f
    n_a0 = P0 * np.pi * a**2 * delta / (RG * TEMP_K)
    x0 = [1.0e-16, 0.0, n_a0]
    t_end = n_cycles / f
    t_eval = np.linspace(0.0, t_end, int(n_cycles * 400) + 1)
    sol = solve_ivp(rhs, (0.0, t_end), x0, t_eval=t_eval, method="LSODA",
                    args=(delta, Q, a, pa, omega), rtol=1e-9, atol=[1e-18, 1e-9, 1e-34],
                    max_step=0.05 / f)
    return sol, dict(cell=cell, a=a, pa=pa, f=f, delta=delta, Q=Q, n_a0=n_a0)


def quasistatic_Z(p_applied, delta, Q, a, n_a=None, gas="frozen"):
    """
    Equilibrium leaflet displacement Z for a *constant* applied acoustic pressure.
    gas='frozen'  : gas content fixed at its resting value n_a0
    gas='henry'   : gas content re-equilibrated so that Pin = P0 (Henry saturation)
    gas='none'    : no gas at all (Pin = 0)
    """
    n_a0 = P0 * np.pi * a**2 * delta / (RG * TEMP_K) if n_a is None else n_a

    def net(Z):
        if gas == "none":
            Pin = 0.0
        elif gas == "henry":
            Pin = P0
        else:
            Pin = p_in(n_a0, Z, delta, a)
        return Pin + p_molecular(Z, delta, a) + p_electrostatic(Q, Z, a) - P0 + p_applied - p_tension(Z, a)

    lo, hi = 1e-12, 60e-9
    f_lo, f_hi = net(lo), net(hi)
    if f_lo * f_hi > 0:
        return np.nan
    return brentq(net, lo, hi, xtol=1e-15, rtol=1e-12)


def capacitance(Z, delta, a=A_BLS, cm0=CM0):
    """Membrane capacitance of a deflected BLS (Cm expression from Sonophore_and_BlackM.m)."""
    Z = np.atleast_1d(np.asarray(Z, dtype=float))
    out = np.where(
        Z > 1e-12,
        (delta * cm0) / a**2 * (Z + (a**2 - Z**2 - Z * delta) / (2.0 * np.maximum(Z, 1e-30))
                                * np.log((2.0 * np.maximum(Z, 1e-30) + delta) / delta)),
        cm0,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../runs/bls_reference.npz")
    ap.add_argument("--json", default="../runs/bls_reference.json")
    args = ap.parse_args()

    report = {}

    # ---- 1. PRX 2014 operating point, a = 32 nm -----------------------------
    sol32, meta32 = run(cell="RS", a=32e-9, n_cycles=40)
    Z32 = sol32.y[0]
    n32 = sol32.y[2]
    report["prx_a32nm"] = dict(
        a_nm=32.0, f_MHz=F_US / 1e6, pa_kPa=PA_US / 1e3,
        delta_nm=meta32["delta"] * 1e9,
        Zmax_nm=float(Z32.max() * 1e9),
        Zmin_nm=float(Z32.min() * 1e9),
        n_gas_molecules_initial=float(meta32["n_a0"] * NAV),
        n_gas_molecules_max=float(n32.max() * NAV),
        n_gas_molecules_final=float(n32[-1] * NAV),
        Cm_over_Cm0_min=float(capacitance(Z32.max(), meta32["delta"], 32e-9)[0] / CM0),
    )

    # ---- 2. MD-accessible patch radius, a = 8 nm ----------------------------
    sol8, meta8 = run(cell="RS", a=8e-9, n_cycles=40)
    Z8 = sol8.y[0]
    report["md_scale_a8nm"] = dict(
        a_nm=8.0, Zmax_nm=float(Z8.max() * 1e9),
        n_gas_molecules_initial=float(meta8["n_a0"] * NAV),
        n_gas_molecules_max=float(sol8.y[2].max() * NAV),
    )

    # ---- 3. Quasi-static Z(P) for both radii, several gas assumptions -------
    # Sign convention follows Sonophore.m: the acoustic term enters the leaflet
    # force balance additively, so POSITIVE values here denote rarefaction
    # (leaflet-separating) pressure.
    pressures = np.linspace(0.0, 0.8e6, 41)
    qs = {}
    for a in (32e-9, 16e-9, 8e-9):
        for gas in ("frozen", "henry", "none"):
            key = f"a{a*1e9:.0f}nm_{gas}"
            qs[key] = [float(quasistatic_Z(p, DELTA_N["RS"], CM0 * VM0["RS"], a, gas=gas))
                       for p in pressures]
    report["quasistatic_pressures_MPa"] = (pressures / 1e6).tolist()
    report["quasistatic_Z_nm"] = {k: (np.array(v) * 1e9).tolist() for k, v in qs.items()}

    # ---- 4. Pressure-term budget at rest and at peak rarefaction ------------
    for label, Z in (("rest", 1e-12), ("Z=1nm", 1e-9), ("Z=5nm", 5e-9), ("Z=10nm", 10e-9)):
        d = DELTA_N["RS"]
        Q = CM0 * VM0["RS"]
        n_a0 = P0 * np.pi * (32e-9) ** 2 * d / (RG * TEMP_K)
        report[f"budget_{label}"] = dict(
            Pin_kPa=float(p_in(n_a0, Z, d, 32e-9) / 1e3),
            Par_kPa=float(p_molecular(Z, d, 32e-9) / 1e3),
            Pat_kPa=float(p_electrostatic(Q, Z, 32e-9) / 1e3),
            Ps_kPa=float(p_tension(Z, 32e-9) / 1e3),
            Pac_amp_kPa=PA_US / 1e3,
        )

    np.savez_compressed(args.out, t32=sol32.t, Z32=Z32, n32=n32,
                        t8=sol8.t, Z8=Z8, n8=sol8.y[2],
                        qs_P=pressures, **{f"qs_{k}": np.array(v) for k, v in qs.items()})
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
