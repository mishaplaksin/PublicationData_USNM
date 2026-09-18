#!/usr/bin/env python3
"""
Analytic bridge to the acoustic timescale.

MD reaches tens of nanoseconds; one 0.69 MHz half-cycle is 725 ns. This script
covers the gap with classical nucleation theory and a slit-cavity energy
balance, using only measured interfacial tensions -- no coarse-grained mapping
enters, so these numbers are independent of the MD unit conversion.

Two geometries are considered for a cavity opening inside the hydrophobic core:

1. Spherical cavity in a hydrocarbon medium (standard CNT):
       r*     = 2 gamma / dP
       dG*    = 16 pi gamma^3 / (3 dP^2)
2. Slit cavity of thickness d between the two leaflets (the BLS geometry): a
   disc of radius R creates two hydrocarbon/vapour interfaces and gains dP per
   unit volume,
       W(R)   = 2 pi R^2 gamma - dP pi R^2 d
   which decreases with R only if dP > 2 gamma / d. Below that threshold no
   disc of any radius is favourable, so the cavity cannot grow at all.

gamma is the hydrocarbon/vapour tension, 20-30 mN/m (hexadecane 27.5 mN/m at
20 C, decane 23.4 mN/m); 25 mN/m is used as the central value. Using the
hydrocarbon/water tension instead (~50 mN/m) roughly doubles gamma and so
raises the threshold pressures by 2x and the barriers by 8x.
"""
import numpy as np

KB = 1.380649e-23
T = 273.15 + 36.0
KT = KB * T

# PRX 2014 operating point, from this repository's NICE code
F_US = 0.69e6
PA_US = 320.608e3
A_BLS = 32e-9
DELTA = 1.2553e-9


def cnt_sphere(gamma, dP):
    r_star = 2 * gamma / dP
    dG = 16 * np.pi * gamma**3 / (3 * dP**2)
    return r_star, dG


def dP_for_barrier(gamma, barrier_kT):
    """Rarefaction needed for the spherical barrier to equal barrier_kT."""
    return np.sqrt(16 * np.pi * gamma**3 / (3 * barrier_kT * KT))


def dP_slit(gamma, d):
    """Rarefaction below which no slit cavity of any radius can grow."""
    return 2 * gamma / d


def report(gamma):
    print(f"\n=== hydrocarbon/vapour tension gamma = {gamma*1e3:.1f} mN/m ===")
    r, dG = cnt_sphere(gamma, PA_US)
    print(f"At the PRX drive amplitude {PA_US/1e3:.1f} kPa:")
    print(f"  critical cavity radius r*        = {r*1e9:9.1f} nm   "
          f"({r/A_BLS:.1f} x the BLS patch radius a = 32 nm)")
    print(f"  nucleation barrier dG*           = {dG/KT:9.3e} kT")
    print(f"  -> attempts needed ~ exp(dG*/kT) = 10^{dG/KT/np.log(10):.3e}")

    print(f"Rarefaction required to reach a given barrier (spherical):")
    for b in (60, 100):
        print(f"  dG* = {b:3d} kT  ->  dP = {dP_for_barrier(gamma, b)/1e6:8.1f} MPa"
              f"   ({dP_for_barrier(gamma, b)/PA_US:6.0f} x PRX)")

    ps = dP_slit(gamma, DELTA)
    print(f"Slit cavity of thickness delta = {DELTA*1e9:.3f} nm:")
    print(f"  growth threshold dP = 2 gamma/delta = {ps/1e6:8.1f} MPa"
          f"   ({ps/PA_US:6.0f} x PRX)")
    print(f"  at the PRX amplitude, W(R) grows as R^2: no cavity of any radius grows")

    # pressure at which the critical radius equals the BLS patch radius
    p_a = 2 * gamma / A_BLS
    print(f"Rarefaction for r* = a = 32 nm      = {p_a/1e6:8.2f} MPa"
          f"   ({p_a/PA_US:6.1f} x PRX)")
    # and for r* equal to half the intramembrane gap
    p_d = 2 * gamma / (DELTA / 2)
    print(f"Rarefaction for r* = delta/2        = {p_d/1e6:8.1f} MPa"
          f"   ({p_d/PA_US:6.0f} x PRX)")
    return dict(gamma_mNm=gamma * 1e3,
                r_star_at_PRX_nm=r * 1e9, dG_at_PRX_kT=dG / KT,
                dP_barrier60kT_MPa=dP_for_barrier(gamma, 60) / 1e6,
                dP_slit_MPa=ps / 1e6,
                dP_rstar_eq_a_MPa=p_a / 1e6)


def bls_configuration_energy(gamma):
    """
    Interfacial cost of the cavity the BLS model *assumes* exists, relative to a
    flat bilayer whose leaflets are in contact.

    At rest the model's cavity is a disc of radius a and thickness delta; at
    displacement Z it is bounded by two spherical caps of base radius a and
    height Z. Either way two hydrocarbon/vapour interfaces must exist that a
    flat, contacting bilayer does not have.
    """
    rows = []
    # resting configuration: flat disc, two faces of area pi a^2
    A_rest = 2 * np.pi * A_BLS**2
    rows.append(("rest (gap = delta)", A_rest * 1e18, gamma * A_rest / KT))
    # expanded configurations: two spherical caps of base radius a, height Z
    for Z_nm in (1.0, 4.0, 8.8):
        Z = Z_nm * 1e-9
        Rc = (A_BLS**2 + Z**2) / (2 * Z)
        A = 2 * (2 * np.pi * Rc * Z)          # two caps
        rows.append((f"Z = {Z_nm:.1f} nm", A * 1e18, gamma * A / KT))
    return rows


def main():
    print(__doc__)
    print(f"kT at 36 C = {KT:.3e} J;  one 0.69 MHz half-cycle = {0.5/F_US*1e9:.0f} ns")
    out = [report(g) for g in (0.020, 0.025, 0.030, 0.050)]

    print("\n=== energy of the cavity the BLS model assumes, vs a flat bilayer ===")
    print(f"(gamma = 25 mN/m, a = 32 nm; two hydrocarbon/vapour interfaces)")
    print(f"{'configuration':>22} {'interface area [nm^2]':>23} {'energy above flat [kT]':>24}")
    for name, area_nm2, e_kT in bls_configuration_energy(0.025):
        print(f"{name:>22} {area_nm2:23.0f} {e_kT:24.3e}")
    print(f"collapse pressure driving the gap shut, 2 gamma/delta = "
          f"{2*0.025/DELTA/1e6:.1f} MPa, vs the model's A_R = 0.1 MPa")

    print("\n--- summary ------------------------------------------------------")
    print(f"{'gamma [mN/m]':>13} {'r* at PRX [nm]':>16} {'dG* at PRX [kT]':>17} "
          f"{'dP for 60kT [MPa]':>19} {'slit dP [MPa]':>15}")
    for o in out:
        print(f"{o['gamma_mNm']:13.1f} {o['r_star_at_PRX_nm']:16.1f} "
              f"{o['dG_at_PRX_kT']:17.2e} {o['dP_barrier60kT_MPa']:19.1f} "
              f"{o['dP_slit_MPa']:15.1f}")

    import json
    with open("../runs/nucleation.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
