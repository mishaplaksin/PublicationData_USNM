#!/usr/bin/env python3
"""
Analyse the rarefaction ladder: for each imposed external normal pressure,
report the achieved mean pressure, the intramembrane gap, how much solvent
entered the hydrophobic core, and the largest void and where it sits
(membrane core vs bulk solvent).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A


def analyse_run(log_path, dump_path, p_unit_Pa, sigma_nm, frac=0.5):
    lg = A.read_log(log_path)
    n = len(lg["step"])
    s = slice(int(n * frac), None)
    out = dict(
        log=os.path.basename(log_path),
        pzz_target_star=float(os.path.basename(log_path)[2:-4]),
        pzz_mean_star=float(lg["pzz"][s].mean()),
        pzz_err_star=A.block_error(lg["pzz"][s]),
        plat_mean_star=float(0.5 * (lg["pxx"][s] + lg["pyy"][s]).mean()),
        gap_star=float(lg["gap"][s].mean()),
        gap_err_star=A.block_error(lg["gap"][s]),
        Lz_star=float(lg["Lz"][s].mean()),
        apl_star=float(lg["apl"][s].mean()),
        T=float(lg["T"][s].mean()),
        steps=int(lg["step"][-1]),
    )
    out["pzz_mean_MPa"] = out["pzz_mean_star"] * p_unit_Pa / 1e6
    # rarefaction relative to ambient: P0 - P_target
    P0 = 101325.0
    out["rarefaction_MPa"] = (P0 - out["pzz_target_star"] * p_unit_Pa) / 1e6
    out["rarefaction_achieved_MPa"] = (P0 - out["pzz_mean_star"] * p_unit_Pa) / 1e6
    out["gap_nm"] = out["gap_star"] * sigma_nm

    if dump_path and os.path.exists(dump_path):
        nb, nlip, nbl, types, frames = A.read_dump(dump_path)
        use = frames[len(frames) // 2:]
        nw_core, voidvol, voidz, nvoid_in_core = [], [], [], []
        for t, L, pos in use:
            zc = L[2] / 2
            zw = pos[types == 0, 2] - zc
            zw -= L[2] * np.round(zw / L[2])
            nw_core.append(int((np.abs(zw) < 1.5).sum()))
            mask, shape, cv = A.void_grid(pos, L, spacing=0.6, r_void=1.15)
            voidvol.append(mask.sum() * cv)
            if mask.any():
                zi = np.array(np.nonzero(mask))[2] * (L[2] / shape[2]) - zc
                zi -= L[2] * np.round(zi / L[2])
                voidz.append(float(np.abs(zi).mean()))
                nvoid_in_core.append(float((np.abs(zi) < 1.5).mean()))
            else:
                voidz.append(np.nan); nvoid_in_core.append(np.nan)
        out.update(
            n_solvent_in_core=float(np.mean(nw_core)),
            void_volume_star=float(np.mean(voidvol)),
            void_volume_nm3=float(np.mean(voidvol)) * sigma_nm**3,
            void_mean_absz_star=float(np.nanmean(voidz)) if np.isfinite(voidz).any() else None,
            void_fraction_in_core=float(np.nanmean(nvoid_in_core)) if np.isfinite(nvoid_in_core).any() else None,
            nframes=len(use),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--p-unit-MPa", type=float, required=True)
    ap.add_argument("--sigma-nm", type=float, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for lp in sorted(glob.glob(os.path.join(args.dir, "pn*.log"))):
        dp = lp[:-4] + ".dump"
        rows.append(analyse_run(lp, dp, args.p_unit_MPa * 1e6, args.sigma_nm))
    rows.sort(key=lambda r: -r["pzz_target_star"])

    hdr = f"{'rarefaction':>12} {'P_zz achieved':>16} {'gap [nm]':>14} {'core solvent':>13} {'void [nm^3]':>12} {'void in core':>12}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['rarefaction_MPa']:9.3f} MPa "
              f"{r['pzz_mean_MPa']:9.3f}±{r['pzz_err_star']*args.p_unit_MPa:5.3f} MPa "
              f"{r['gap_nm']:8.3f}±{r['gap_err_star']*args.sigma_nm:5.3f} "
              f"{r.get('n_solvent_in_core', float('nan')):12.1f} "
              f"{r.get('void_volume_nm3', float('nan')):12.3f} "
              f"{str(r.get('void_fraction_in_core')):>12.12}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(dict(p_unit_MPa=args.p_unit_MPa, sigma_nm=args.sigma_nm, runs=rows), fh, indent=2)
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
