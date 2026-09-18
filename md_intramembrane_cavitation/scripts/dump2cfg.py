#!/usr/bin/env python3
"""Recover a cgmd configuration from a frame of a binary dump.

Useful to pick up an equilibrated state from a run that was stopped before it
wrote its final configuration. Velocities are not stored in the dump; cgmd
re-draws them from the Maxwell distribution on startup, so only positions and
the box are needed.
"""
import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--frame", type=int, default=-1)
    args = ap.parse_args()

    n, nlip, nbl, types, frames = A.read_dump(args.dump)
    t, L, pos = frames[args.frame]
    pos = np.mod(pos, L)
    with open(args.out, "w") as fh:
        fh.write(f"{n} {nlip} {nbl} {L[0]:.10f} {L[1]:.10f} {L[2]:.10f}\n")
        for ty, p in zip(types, pos):
            fh.write(f"{int(ty)} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
    print(f"frame {args.frame} (t={t:.1f} tau) of {len(frames)} -> {args.out}: "
          f"N={n} nlip={nlip} nbl={nbl} L=({L[0]:.3f},{L[1]:.3f},{L[2]:.3f})")


if __name__ == "__main__":
    main()
