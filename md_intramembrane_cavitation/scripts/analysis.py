#!/usr/bin/env python3
"""
Analysis helpers for the cgmd runs: trajectory reading, density profiles,
surface tension, void (cavity) detection and the reduced-unit -> SI mapping.
"""
import numpy as np

W, H, T, G = 0, 1, 2, 3


# --------------------------------------------------------------- trajectory
def read_dump(path):
    """Yield (t, L, positions) frames from a cgmd binary dump."""
    with open(path, "rb") as fh:
        n, nlip, nbl = np.frombuffer(fh.read(12), dtype=np.int32)
        types = np.frombuffer(fh.read(4 * n), dtype=np.int32).copy()
        frames = []
        while True:
            hdr = fh.read(16)
            if len(hdr) < 16:
                break
            t, Lx, Ly, Lz = np.frombuffer(hdr, dtype=np.float32)
            buf = fh.read(12 * n)
            if len(buf) < 12 * n:
                break
            pos = np.frombuffer(buf, dtype=np.float32).reshape(n, 3).copy()
            frames.append((float(t), np.array([Lx, Ly, Lz], dtype=float), pos))
    return int(n), int(nlip), int(nbl), types, frames


def read_log(path):
    d = np.loadtxt(path)
    cols = "step time epot T pxx pyy pzz Lx Lz gap comF apl".split()
    return {c: d[:, i] for i, c in enumerate(cols)}


# ------------------------------------------------------------------ physics
def surface_tension(log, frac=0.5):
    """gamma per interface, for a slab with two interfaces normal to z."""
    n = len(log["step"]); s = slice(int(n * frac), None)
    g = 0.5 * log["Lz"][s] * (log["pzz"][s] - 0.5 * (log["pxx"][s] + log["pyy"][s]))
    return g.mean(), block_error(g)


def block_error(x, nblocks=8):
    x = np.asarray(x)
    if len(x) < nblocks * 2:
        return float(x.std() / max(np.sqrt(len(x)), 1))
    b = np.array_split(x, nblocks)
    m = np.array([bb.mean() for bb in b])
    return float(m.std(ddof=1) / np.sqrt(nblocks))


def density_profile(types, frames, L, species, nbins=120, frac=0.5):
    """Number-density profile along z, averaged over the last (1-frac) frames."""
    use = frames[int(len(frames) * frac):]
    edges = np.linspace(0, L[2], nbins + 1)
    prof = np.zeros(nbins)
    sel = np.isin(types, species)
    for _, Lf, pos in use:
        z = np.mod(pos[sel, 2], Lf[2])
        h, _ = np.histogram(z, bins=np.linspace(0, Lf[2], nbins + 1))
        prof += h
    prof /= len(use)
    binvol = L[0] * L[1] * (L[2] / nbins)
    return 0.5 * (edges[1:] + edges[:-1]), prof / binvol


def liquid_density(types, frames, L, frac=0.5, species=(W,)):
    """Coexistence liquid density: mean density in the middle half of the slab."""
    z, rho = density_profile(types, frames, L, species, frac=frac)
    mid = (z > 0.35 * L[2]) & (z < 0.65 * L[2])
    return rho[mid].mean()


def void_grid(pos, L, types=None, spacing=0.5, r_void=1.1, exclude=()):
    """
    Grid-based void detection. A grid point counts as void when no bead centre
    lies within r_void of it. Returns (mask, grid shape, cell volume).
    Periodic in x, y, z.
    """
    if types is not None and len(exclude):
        keep = ~np.isin(types, exclude)
        pos = pos[keep]
    nx, ny, nz = [max(4, int(round(L[i] / spacing))) for i in range(3)]
    dx, dy, dz = L[0] / nx, L[1] / ny, L[2] / nz
    occ = np.zeros((nx, ny, nz), dtype=bool)
    # stamp a sphere of radius r_void around every bead
    rx, ry, rz = int(np.ceil(r_void / dx)), int(np.ceil(r_void / dy)), int(np.ceil(r_void / dz))
    ox = (np.arange(-rx, rx + 1) * dx)[:, None, None]
    oy = (np.arange(-ry, ry + 1) * dy)[None, :, None]
    oz = (np.arange(-rz, rz + 1) * dz)[None, None, :]
    stencil = (ox**2 + oy**2 + oz**2) <= r_void**2
    sidx = np.array(np.nonzero(stencil)).T - np.array([rx, ry, rz])
    ix = np.floor(pos[:, 0] / dx).astype(int) % nx
    iy = np.floor(pos[:, 1] / dy).astype(int) % ny
    iz = np.floor(pos[:, 2] / dz).astype(int) % nz
    for sx, sy, sz in sidx:
        occ[(ix + sx) % nx, (iy + sy) % ny, (iz + sz) % nz] = True
    return ~occ, (nx, ny, nz), dx * dy * dz


def largest_void_cluster(mask):
    """Largest connected void cluster (6-connectivity, periodic). Returns
    (size_in_cells, list_of_indices)."""
    nx, ny, nz = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best, best_cells = 0, None
    idxs = np.array(np.nonzero(mask)).T
    for start in map(tuple, idxs):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        cells = []
        while stack:
            c = stack.pop()
            cells.append(c)
            x, y, z = c
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nb = ((x + d[0]) % nx, (y + d[1]) % ny, (z + d[2]) % nz)
                if mask[nb] and not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        if len(cells) > best:
            best, best_cells = len(cells), cells
    return best, best_cells


# ----------------------------------------------------------- unit mapping
class Units:
    """
    Reduced (sigma, eps) -> SI mapping.

    sigma is fixed by the coarse-graining level: one solvent bead stands for
    n_water water molecules, so sigma^3 / rho*_liquid = n_water * v_water.

    eps is fixed by matching the simulated liquid/vapour surface tension to
    that of water at the temperature of interest:  gamma_SI = gamma* eps/sigma^2.
    """
    V_WATER_NM3 = 0.03006          # molecular volume of liquid water [nm^3]

    def __init__(self, rho_star, gamma_star, n_water=4, gamma_SI=0.0700, T_star=1.0,
                 T_SI=309.15, mass_amu=None):
        self.rho_star = rho_star
        self.gamma_star = gamma_star
        self.n_water = n_water
        self.gamma_SI = gamma_SI
        self.T_star = T_star
        self.T_SI = T_SI
        # length
        v_bead_nm3 = n_water * self.V_WATER_NM3
        self.sigma_nm = (v_bead_nm3 * rho_star) ** (1 / 3)
        self.sigma_m = self.sigma_nm * 1e-9
        # energy from surface tension
        self.eps_J = gamma_SI * self.sigma_m**2 / gamma_star
        # what temperature the simulated T* corresponds to under that eps
        self.kB = 1.380649e-23
        self.T_implied = T_star * self.eps_J / self.kB
        # alternative: energy scale from temperature matching
        self.eps_J_from_T = self.kB * T_SI / T_star
        # pressure and time units (surface-tension-based eps)
        self.p_unit_Pa = self.eps_J / self.sigma_m**3
        self.p_unit_Pa_fromT = self.eps_J_from_T / self.sigma_m**3
        self.mass_amu = mass_amu if mass_amu else n_water * 18.015
        m_kg = self.mass_amu * 1.66053906660e-27
        self.tau_s = self.sigma_m * np.sqrt(m_kg / self.eps_J)
        self.tau_s_fromT = self.sigma_m * np.sqrt(m_kg / self.eps_J_from_T)

    def P(self, p_star, from_T=False):
        """reduced pressure -> Pa"""
        return p_star * (self.p_unit_Pa_fromT if from_T else self.p_unit_Pa)

    def gamma(self, g_star):
        """reduced surface tension -> N/m"""
        return g_star * self.eps_J / self.sigma_m**2

    def length_nm(self, l_star):
        return l_star * self.sigma_nm

    def summary(self):
        return dict(
            sigma_nm=self.sigma_nm,
            eps_J=self.eps_J,
            eps_kJ_per_mol=self.eps_J * 6.02214076e23 / 1e3,
            rho_star=self.rho_star,
            gamma_star=self.gamma_star,
            pressure_unit_MPa=self.p_unit_Pa / 1e6,
            pressure_unit_MPa_from_T=self.p_unit_Pa_fromT / 1e6,
            tau_ps=self.tau_s * 1e12,
            tau_ps_from_T=self.tau_s_fromT * 1e12,
            T_implied_K=self.T_implied,
            T_target_K=self.T_SI,
        )
