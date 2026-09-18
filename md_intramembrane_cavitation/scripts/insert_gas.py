#!/usr/bin/env python3
"""
Insert hydrophobic "gas" beads into the intramembrane space of an equilibrated
bilayer configuration, to test the BLS premise of a gas-filled cavity between
the leaflets.

The BLS resting gas content follows from the model's own expression
n_a0 = P0 pi a^2 delta /(Rg T), i.e. a surface density of

    n/A = P0 delta /(Rg T) = 0.0298 air molecules per nm^2

so a patch of a few hundred nm^2 holds only a handful of molecules. Larger,
deliberately supersaturated fillings are also supported, to ask whether even a
*pre-formed* pocket is mechanically stable.

Beads are placed near the midplane; residual overlaps are relieved by running
cgmd with --softstart.
"""
import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recenter import load, save

P0 = 101325.0
RG = 8.314
TEMP_K = 273.15 + 36.0
NAV = 6.02214076e23
DELTA_RS = 1.2553e-9


def henry_count(area_nm2, delta_m=DELTA_RS):
    """BLS resting air content for a patch of this area, in molecules."""
    per_nm2 = P0 * delta_m / (RG * TEMP_K) * NAV * 1e-18
    return per_nm2 * area_nm2, per_nm2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--ngas", type=int, required=True)
    ap.add_argument("--radius", type=float, default=-1.0,
                    help="place within this radius of the patch centre [sigma]; "
                         "negative = anywhere in the periodic midplane")
    ap.add_argument("--zspread", type=float, default=0.3)
    ap.add_argument("--sigma-nm", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    n, nlip, nbl, L, types, pos = load(args.inp)
    rng = np.random.default_rng(args.seed)
    zc = L[2] / 2

    area_nm2 = L[0] * L[1] * args.sigma_nm**2
    nh, per_nm2 = henry_count(area_nm2)
    print(f"patch area {L[0]*L[1]:.1f} sigma^2 = {area_nm2:.1f} nm^2")
    print(f"BLS resting air content: {per_nm2:.4f} molecules/nm^2 -> {nh:.2f} molecules "
          f"for this patch")
    print(f"inserting {args.ngas} gas beads "
          f"({args.ngas/max(nh,1e-9):.1f}x the model's own content)")

    new = []
    for _ in range(args.ngas):
        if args.radius > 0:
            r = args.radius * np.sqrt(rng.random())
            th = 2 * np.pi * rng.random()
            x, y = L[0] / 2 + r * np.cos(th), L[1] / 2 + r * np.sin(th)
        else:
            x, y = rng.random() * L[0], rng.random() * L[1]
        z = zc + rng.normal(0, args.zspread)
        new.append([x % L[0], y % L[1], z % L[2]])

    pos = np.vstack([pos, np.array(new)])
    types = np.concatenate([types, np.full(args.ngas, 3, dtype=int)])
    save(args.out, len(types), nlip, nbl, L, types, pos)
    print(f"wrote {args.out}: N={len(types)} (added {args.ngas} type-3 beads)")


if __name__ == "__main__":
    main()
