#!/usr/bin/env python3
"""
Build starting configurations for cgmd.

Layouts
-------
bilayer : explicit-solvent lipid bilayer centred at Lz/2, optional hydrophobic
          gas beads placed in the bilayer midplane (the "intramembrane space").
liquid  : pure solvent box (for the equation-of-state / surface-tension
          calibration runs).
slab    : solvent slab with two free interfaces (surface-tension calibration).

Bead types: 0 = W (solvent), 1 = H (lipid head), 2 = T (lipid tail), 3 = G (gas)
Lipids are stored first as consecutive triples (H, T, T); cgmd relies on that.
"""
import argparse
import numpy as np


def write_cfg(path, types, pos, L, nlip, nbl=3):
    with open(path, "w") as fh:
        fh.write(f"{len(types)} {nlip} {nbl} {L[0]:.10f} {L[1]:.10f} {L[2]:.10f}\n")
        for t, p in zip(types, pos):
            fh.write(f"{int(t)} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


def build_bilayer(nx, ny, apl_target, lz, rho_w, n_gas, gas_radius, seed=1, ntail=3):
    """nx, ny = lipids per side per leaflet."""
    rng = np.random.default_rng(seed)
    spacing = np.sqrt(apl_target)
    Lx, Ly = nx * spacing, ny * spacing
    zc = lz / 2.0
    bond = 1.0
    # leaflet geometry: head outermost, two tails pointing to the midplane
    # midplane half-gap: start the terminal tail beads ~0.5 sigma from z = zc
    gap_half = 0.55
    nbl = 1 + ntail
    pos, types = [], []
    for leaf in (+1, -1):
        for i in range(nx):
            for j in range(ny):
                x = (i + 0.5) * spacing + rng.normal(0, 0.05)
                y = (j + 0.5) * spacing + rng.normal(0, 0.05)
                z_tip = zc + leaf * gap_half
                # chain written head-first: head is the outermost bead
                chain = [[x, y, z_tip + leaf * bond * k] for k in range(nbl)][::-1]
                pos += chain
                types += [1] + [2] * ntail
    nlip = 2 * nx * ny
    membrane_half = gap_half + ntail * bond + 0.6  # keep solvent out of the core

    # gas beads in the intramembrane space, within gas_radius of the patch centre
    gas_pos = []
    if n_gas > 0:
        r = gas_radius * np.sqrt(rng.random(n_gas))
        th = 2 * np.pi * rng.random(n_gas)
        gas_pos = np.column_stack([Lx / 2 + r * np.cos(th),
                                   Ly / 2 + r * np.sin(th),
                                   zc + rng.normal(0, 0.25, n_gas)])
        pos += gas_pos.tolist()
        types += [3] * n_gas

    # solvent on both sides of the membrane, on a dimension-matched lattice with
    # overlap rejection against the lipids/gas already placed
    existing = np.array(pos)
    slabs = [(0.0, zc - membrane_half), (zc + membrane_half, lz)]
    dmin = 0.85
    for z0, z1 in slabs:
        thickness = z1 - z0
        if thickness <= dmin:
            continue
        vol = Lx * Ly * thickness
        count = int(round(rho_w * vol))
        # lattice spacing isotropic-ish: n_i proportional to L_i
        scale = (count / vol) ** (1 / 3)
        nax = max(1, int(round(Lx * scale)))
        nay = max(1, int(round(Ly * scale)))
        naz = max(1, int(np.ceil(count / (nax * nay))))
        sites = []
        for a in range(nax):
            for b in range(nay):
                for c in range(naz):
                    sites.append([(a + 0.5) / nax * Lx, (b + 0.5) / nay * Ly,
                                  z0 + (c + 0.5) / naz * thickness])
        sites = np.array(sites) + rng.normal(0, 0.05, (len(sites), 3))
        # reject sites that clash with lipid/gas beads (periodic in x, y)
        keep = np.ones(len(sites), dtype=bool)
        if len(existing):
            for k, sp in enumerate(sites):
                d = existing - sp
                d[:, 0] -= Lx * np.round(d[:, 0] / Lx)
                d[:, 1] -= Ly * np.round(d[:, 1] / Ly)
                if (np.einsum("ij,ij->i", d, d) < dmin * dmin).any():
                    keep[k] = False
        sites = sites[keep][:count]
        pos += sites.tolist()
        types += [0] * len(sites)
    return np.array(types), np.array(pos), (Lx, Ly, lz), nlip, nbl


def build_liquid(n, rho, seed=1):
    rng = np.random.default_rng(seed)
    L = (n / rho) ** (1 / 3)
    m = int(np.ceil(n ** (1 / 3)))
    pos, placed = [], 0
    for i in range(m):
        for j in range(m):
            for k in range(m):
                if placed >= n:
                    break
                pos.append([(i + 0.5) / m * L + rng.normal(0, 0.05),
                            (j + 0.5) / m * L + rng.normal(0, 0.05),
                            (k + 0.5) / m * L + rng.normal(0, 0.05)])
                placed += 1
    return np.zeros(n, dtype=int), np.array(pos), (L, L, L), 0, 3


def build_slab(n, rho, lz_factor=3.0, seed=1):
    """Liquid slab of density rho occupying the middle of a box lz_factor taller."""
    rng = np.random.default_rng(seed)
    Lxy = (n / rho / lz_factor) ** (1 / 3) * np.sqrt(lz_factor)
    # choose cubic-ish slab: Lx=Ly=L, slab thickness t, box Lz = lz_factor*t
    L = (n / rho) ** (1 / 3)
    t = L
    Lz = lz_factor * t
    m = int(np.ceil(n ** (1 / 3)))
    pos, placed = [], 0
    z0 = (Lz - t) / 2
    for i in range(m):
        for j in range(m):
            for k in range(m):
                if placed >= n:
                    break
                pos.append([(i + 0.5) / m * L + rng.normal(0, 0.05),
                            (j + 0.5) / m * L + rng.normal(0, 0.05),
                            z0 + (k + 0.5) / m * t + rng.normal(0, 0.05)])
                placed += 1
    return np.zeros(n, dtype=int), np.array(pos), (L, L, Lz), 0, 3


def build_tslab(n, rho, lz_factor=3.0, with_water=False, rho_w=0.85, seed=1):
    """Slab of pure tail beads, in vacuum or immersed in solvent.
    Used to measure the tail/vapour and tail/water interfacial tensions that
    set the interleaflet work of adhesion."""
    rng = np.random.default_rng(seed)
    L = (n / rho) ** (1 / 3)
    t = L
    Lz = lz_factor * t
    m = int(np.ceil(n ** (1 / 3)))
    pos, types, placed = [], [], 0
    z0 = (Lz - t) / 2
    for i in range(m):
        for j in range(m):
            for k in range(m):
                if placed >= n:
                    break
                pos.append([(i + 0.5) / m * L + rng.normal(0, 0.05),
                            (j + 0.5) / m * L + rng.normal(0, 0.05),
                            z0 + (k + 0.5) / m * t + rng.normal(0, 0.05)])
                types.append(2)
                placed += 1
    if with_water:
        dmin = 0.9
        existing = np.array(pos)
        for za, zb in ((0.0, z0), (z0 + t, Lz)):
            thick = zb - za
            if thick <= dmin:
                continue
            count = int(round(rho_w * L * L * thick))
            scale = (count / (L * L * thick)) ** (1 / 3)
            nax = max(1, int(round(L * scale))); nay = max(1, int(round(L * scale)))
            naz = max(1, int(np.ceil(count / (nax * nay))))
            sites = np.array([[(a + 0.5) / nax * L, (b + 0.5) / nay * L,
                               za + (c + 0.5) / naz * thick]
                              for a in range(nax) for b in range(nay) for c in range(naz)])
            sites = sites + rng.normal(0, 0.05, sites.shape)
            keep = np.ones(len(sites), dtype=bool)
            for k2, sp in enumerate(sites):
                d = existing - sp
                d[:, 0] -= L * np.round(d[:, 0] / L)
                d[:, 1] -= L * np.round(d[:, 1] / L)
                if (np.einsum("ij,ij->i", d, d) < dmin * dmin).any():
                    keep[k2] = False
            sites = sites[keep][:count]
            pos += sites.tolist(); types += [0] * len(sites)
    return np.array(types), np.array(pos), (L, L, Lz), 0, 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout", choices=["bilayer", "liquid", "slab", "tslab", "tslab_water"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--nx", type=int, default=18)
    ap.add_argument("--ny", type=int, default=18)
    ap.add_argument("--apl", type=float, default=1.30, help="target area per lipid [sigma^2]")
    ap.add_argument("--lz", type=float, default=26.0)
    ap.add_argument("--rhow", type=float, default=0.80)
    ap.add_argument("--ngas", type=int, default=0)
    ap.add_argument("--gasradius", type=float, default=6.0)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--rho", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ntail", type=int, default=3, help="tail beads per lipid")
    args = ap.parse_args()

    if args.layout == "bilayer":
        t, p, L, nlip, nbl = build_bilayer(args.nx, args.ny, args.apl, args.lz,
                                           args.rhow, args.ngas, args.gasradius,
                                           args.seed, args.ntail)
    elif args.layout == "liquid":
        t, p, L, nlip, nbl = build_liquid(args.n, args.rho, args.seed)
    elif args.layout == "slab":
        t, p, L, nlip, nbl = build_slab(args.n, args.rho, seed=args.seed)
    elif args.layout == "tslab":
        t, p, L, nlip, nbl = build_tslab(args.n, args.rho, seed=args.seed)
    else:
        t, p, L, nlip, nbl = build_tslab(args.n, args.rho, with_water=True,
                                         rho_w=args.rhow, seed=args.seed)

    p[:, 0] = np.mod(p[:, 0], L[0])
    p[:, 1] = np.mod(p[:, 1], L[1])
    p[:, 2] = np.mod(p[:, 2], L[2])
    write_cfg(args.out, t, p, L, nlip, nbl)
    counts = {k: int((t == v).sum()) for k, v in (("W", 0), ("H", 1), ("T", 2), ("G", 3))}
    print(f"wrote {args.out}: N={len(t)} nlip={nlip} nbl={nbl} L=({L[0]:.2f},{L[1]:.2f},{L[2]:.2f}) {counts}")


if __name__ == "__main__":
    main()
