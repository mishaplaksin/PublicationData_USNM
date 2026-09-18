#!/usr/bin/env python3
"""Recentre a cgmd configuration so that the bilayer midplane sits at Lz/2.

The Langevin thermostat does not conserve momentum, so a membrane slowly
diffuses along z over a long equilibration; cgmd's leaflet bookkeeping assumes
the membrane is centred, so this is applied between stages.
"""
import argparse
import numpy as np


def load(path):
    with open(path) as fh:
        n, nlip, nbl, Lx, Ly, Lz = fh.readline().split()
        n, nlip, nbl = int(n), int(nlip), int(nbl)
        L = np.array([float(Lx), float(Ly), float(Lz)])
        data = np.loadtxt(fh)
    return n, nlip, nbl, L, data[:, 0].astype(int), data[:, 1:4]


def save(path, n, nlip, nbl, L, types, pos):
    with open(path, "w") as fh:
        fh.write(f"{n} {nlip} {nbl} {L[0]:.10f} {L[1]:.10f} {L[2]:.10f}\n")
        for t, p in zip(types, pos):
            fh.write(f"{int(t)} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    args = ap.parse_args()
    n, nlip, nbl, L, types, pos = load(args.inp)

    # circular mean of the tail-bead z distribution locates the midplane robustly
    tails = pos[np.isin(types, [2]), 2]
    ang = 2 * np.pi * tails / L[2]
    mid = L[2] * np.angle(np.exp(1j * ang).mean()) / (2 * np.pi)
    shift = L[2] / 2 - mid
    pos[:, 2] = np.mod(pos[:, 2] + shift, L[2])
    save(args.out, n, nlip, nbl, L, types, pos)

    tails2 = pos[np.isin(types, [2]), 2]
    print(f"midplane was at z={mid:.3f}, shifted by {shift:+.3f}; "
          f"tail-bead z range now {tails2.min():.2f}..{tails2.max():.2f}, Lz={L[2]:.3f}")


if __name__ == "__main__":
    main()
