#!/usr/bin/env python3
"""
Analyse the interleaflet separation windows and compare the measured
intramembrane separation pressure with the BLS model of Plaksin et al.

MD observable
-------------
Each window holds the leaflet-leaflet COM separation at dz with a stiff
harmonic restraint, under a semi-isotropic barostat at zero external pressure.
The mean restraint force <F> is then the force needed to hold an intramembrane
gap open, so

    P_sep(d) = <F> / (Lx Ly)          d = <dz> - dz_0

is the external rarefaction pressure required to sustain a planar
intramembrane void of thickness d.  Integrating gives the work of adhesion

    W_adh = int_0^inf P_sep dd        [energy / area]

BLS counterpart
---------------
In the BLS force balance (Sonophore.m) the rarefaction pressure needed to hold
the leaflets at displacement Z, neglecting the stretching term Ps (the MD gap
is laterally uniform, so no area strain is involved), is

    P_coh(Z) = P0 - Pin(Z) - Par(Z) - Pat(Z)

with the gap opening by 2Z.  The two curves are directly comparable.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A
import bls_ode as B


def dump_diagnostics(dump_path, frac=0.5):
    """How the membrane accommodates the imposed separation: tail-tip gap,
    solvent admitted into the hydrophobic core, and any dry void."""
    if not dump_path or not os.path.exists(dump_path):
        return {}
    n, nlip, nbl, types, frames = A.read_dump(dump_path)
    use = frames[int(len(frames) * frac):]
    tipgap, nw_core, voidvol = [], [], []
    for t, L, pos in use:
        zc = L[2] / 2
        hz = pos[0:nlip * nbl:nbl, 2] - zc
        hz -= L[2] * np.round(hz / L[2])
        tips = pos[nbl - 1:nlip * nbl:nbl, 2] - zc
        tips -= L[2] * np.round(tips / L[2])
        up = hz > 0
        if up.any() and (~up).any():
            tipgap.append(float(tips[up].mean() - tips[~up].mean()))
        zw = pos[types == 0, 2] - zc
        zw -= L[2] * np.round(zw / L[2])
        nw_core.append(int((np.abs(zw) < 1.5).sum()))
        mask, shape, cv = A.void_grid(pos, L, spacing=0.6, r_void=1.15)
        voidvol.append(mask.sum() * cv)
    return dict(tip_gap_star=float(np.mean(tipgap)) if tipgap else None,
                n_solvent_in_core=float(np.mean(nw_core)),
                void_volume_star=float(np.mean(voidvol)),
                nframes=len(use))


def window_result(path, frac_equil=0.4, kcom=400.0):
    log = A.read_log(path)
    n = len(log["step"])
    s = slice(int(n * frac_equil), None)
    F = log["comF"][s]
    area = (log["Lx"][s] ** 2)
    # <dz> recovered from the restraint force
    return dict(
        path=path,
        F_mean=float(F.mean()), F_err=A.block_error(F),
        area=float(area.mean()),
        P_sep=float((F / area).mean()), P_sep_err=A.block_error(F / area),
        Lz=float(log["Lz"][s].mean()),
        gap=float(log["gap"][s].mean()), gap_err=A.block_error(log["gap"][s]),
        T=float(log["T"][s].mean()),
        pzz=float(log["pzz"][s].mean()),
        nsamples=int(n - int(n * frac_equil)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory with dzXX.XX.log windows")
    ap.add_argument("--kcom", type=float, default=400.0)
    ap.add_argument("--rho-star", type=float, required=True, help="CG liquid density")
    ap.add_argument("--gamma-star-water", type=float, required=True)
    ap.add_argument("--gamma-star-tail", type=float, required=True)
    ap.add_argument("--core-thickness-star", type=float, required=True,
                    help="CG hydrophobic core thickness [sigma]")
    ap.add_argument("--core-thickness-nm", type=float, default=2.8,
                    help="real acyl-core thickness to match [nm]")
    ap.add_argument("--gamma-tail-SI", type=float, default=0.025,
                    help="hydrocarbon/vapour tension [N/m]")
    ap.add_argument("--out", default="../runs/parz_result.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "dz*.log")))
    if not files:
        sys.exit(f"no windows found in {args.dir}")
    res = [window_result(f, kcom=args.kcom) for f in files]
    for r in res:
        r.update(dump_diagnostics(r["path"][:-4] + ".dump"))
    dz_target = [float(os.path.basename(f)[2:-4]) for f in files]
    for r, d in zip(res, dz_target):
        r["dz_target"] = d
        r["dz_mean"] = d - r["F_mean"] / args.kcom

    dz = np.array([r["dz_mean"] for r in res])
    P = np.array([r["P_sep"] for r in res])
    Perr = np.array([r["P_sep_err"] for r in res])
    order = np.argsort(dz)
    dz, P, Perr = dz[order], P[order], Perr[order]
    res = [res[i] for i in order]

    # reference separation: the window with the smallest |P_sep|, i.e. the
    # unstressed bilayer, taken as d = 0
    dz0 = float(np.interp(0.0, P[np.argsort(P)], dz[np.argsort(P)])) if (P.min() < 0 < P.max()) else float(dz[0])
    d = dz - dz0

    # ---- unit mapping -------------------------------------------------------
    # length: match the CG hydrophobic core thickness to a real acyl core
    sigma_nm = args.core_thickness_nm / args.core_thickness_star
    # pressure: anchor to the tail/vapour tension that sets the adhesion
    #   gamma_SI = gamma* eps/sigma^2  ->  eps = gamma_SI sigma^2 / gamma*
    sigma_m = sigma_nm * 1e-9
    eps_J = args.gamma_tail_SI * sigma_m**2 / args.gamma_star_tail
    p_unit_Pa = eps_J / sigma_m**3            # = gamma_tail_SI /(gamma*_tail sigma)
    # cross-check mapping anchored on water instead
    units_w = A.Units(args.rho_star, args.gamma_star_water, n_water=4,
                      gamma_SI=0.0700, T_star=1.0, T_SI=309.15)

    P_MPa = P * p_unit_Pa / 1e6
    Perr_MPa = Perr * p_unit_Pa / 1e6
    d_nm = d * sigma_nm

    # work of adhesion: integrate the positive branch
    pos = d >= 0
    W_star = float(np.trapezoid(np.clip(P[pos], 0, None), d[pos]))
    W_SI_mNm = W_star * (eps_J / sigma_m**2) * 1e3

    # ---- BLS counterpart ----------------------------------------------------
    delta = B.DELTA_N["RS"]
    Q = B.CM0 * B.VM0["RS"]
    Z_nm = np.linspace(0.02, max(4.0, d_nm.max() / 2 + 0.5), 200)
    Z = Z_nm * 1e-9
    n_a0 = B.P0 * np.pi * B.A_BLS**2 * delta / (B.RG * B.TEMP_K)
    P_coh = np.array([B.P0 - B.p_in(n_a0, z, delta, B.A_BLS)
                      - B.p_molecular(z, delta, B.A_BLS)
                      - B.p_electrostatic(Q, z, B.A_BLS) for z in Z])
    W_bls_mNm = float(np.trapezoid(np.clip(P_coh, 0, None), 2 * Z) * 1e3)

    # BLS prediction with no gas at all, for reference
    P_coh_nogas = np.array([B.P0 - B.p_molecular(z, delta, B.A_BLS)
                            - B.p_electrostatic(Q, z, B.A_BLS) for z in Z])

    # pressure required by MD to open the gap to the BLS's predicted opening
    out = dict(
        mapping=dict(sigma_nm=sigma_nm, eps_J=eps_J,
                     pressure_unit_MPa=p_unit_Pa / 1e6,
                     gamma_star_tail=args.gamma_star_tail,
                     gamma_star_water=args.gamma_star_water,
                     gamma_tail_SI_mNm=args.gamma_tail_SI * 1e3,
                     core_thickness_star=args.core_thickness_star,
                     core_thickness_nm=args.core_thickness_nm,
                     water_anchored=units_w.summary()),
        dz0_star=dz0,
        windows=res,
        d_nm=d_nm.tolist(),
        P_sep_MPa=P_MPa.tolist(),
        P_sep_MPa_err=Perr_MPa.tolist(),
        W_adh_mN_per_m=W_SI_mNm,
        W_adh_reduced=W_star,
        bls=dict(Z_nm=Z_nm.tolist(),
                 P_coh_MPa=(P_coh / 1e6).tolist(),
                 P_coh_nogas_MPa=(P_coh_nogas / 1e6).tolist(),
                 W_adh_mN_per_m=W_bls_mNm,
                 PA_PRX_kPa=B.PA_US / 1e3),
    )
    # MD pressure at the gap the BLS opens at the PRX amplitude
    d_bls_at_PRX_nm = 2 * 7.788      # from bls_ode quasi-static, a = 32 nm
    if d_nm.max() > 0.2:
        out["P_MD_at_d0p5nm_MPa"] = float(np.interp(0.5, d_nm, P_MPa))
        out["P_MD_at_d1nm_MPa"] = float(np.interp(1.0, d_nm, P_MPa))
        out["P_MD_at_d2nm_MPa"] = float(np.interp(2.0, d_nm, P_MPa))
    out["bls"]["P_coh_at_Z0p5nm_kPa"] = float(np.interp(0.5, Z_nm, P_coh) / 1e3)
    out["bls"]["P_coh_at_Z1nm_kPa"] = float(np.interp(1.0, Z_nm, P_coh) / 1e3)
    out["bls"]["d_at_PRX_amplitude_nm"] = d_bls_at_PRX_nm

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"mapping: sigma = {sigma_nm:.3f} nm, pressure unit = {p_unit_Pa/1e6:.2f} MPa")
    print(f"reference separation dz0 = {dz0:.3f} sigma")
    print(f"{'d [nm]':>9} {'P_sep [MPa]':>14} {'+-':>8} {'tipgap[nm]':>11} "
          f"{'core solv':>10} {'void[nm3]':>10}")
    for r, dd, pp, ee in zip(res, d_nm, P_MPa, Perr_MPa):
        tg = r.get("tip_gap_star")
        tg = f"{tg*sigma_nm:11.3f}" if tg is not None else f"{'-':>11}"
        vv = r.get("void_volume_star")
        vv = f"{vv*sigma_nm**3:10.3f}" if vv is not None else f"{'-':>10}"
        print(f"{dd:9.3f} {pp:14.3f} {ee:8.3f} {tg} "
              f"{r.get('n_solvent_in_core', float('nan')):10.1f} {vv}")
    print(f"\nMD  work of adhesion : {W_SI_mNm:8.2f} mN/m")
    print(f"BLS work of adhesion : {W_bls_mNm:8.4f} mN/m   (A_R = 1e5 Pa model)")
    print(f"ratio MD/BLS         : {W_SI_mNm / W_bls_mNm:8.1f}")
    print(f"\nBLS P_coh at Z=1 nm  : {out['bls']['P_coh_at_Z1nm_kPa']:.1f} kPa"
          f"   (PRX drive amplitude {B.PA_US/1e3:.1f} kPa)")
    if "P_MD_at_d2nm_MPa" in out:
        print(f"MD  P_sep at d=2 nm  : {out['P_MD_at_d2nm_MPa']:.1f} MPa "
              f"= {out['P_MD_at_d2nm_MPa']*1e3:.0f} kPa")


if __name__ == "__main__":
    main()
