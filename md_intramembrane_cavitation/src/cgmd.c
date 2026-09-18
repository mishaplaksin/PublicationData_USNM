/*
 * cgmd.c -- minimal coarse-grained molecular dynamics engine for testing
 *           intramembrane cavitation in a lipid bilayer under acoustic load.
 *
 * Model: Cooke-Deserno-style implicit-chain lipids (H-T-T) with EXPLICIT
 *        solvent, plus an optional hydrophobic "gas" species.
 *
 *   V_rep(r) = 4 eps [ (b/r)^12 - (b/r)^6 ] + eps ,  r < 2^(1/6) b
 *   V_att(r) = -eps_att                            ,  r < rc = 2^(1/6) b
 *            = -eps_att cos^2[ pi (r-rc) / (2 wc) ],  rc <= r < rc + wc
 *   bonds   : FENE (k=30, r0=1.5) in addition to V_rep
 *   bending : harmonic on the 1-3 distance
 *
 * Reduced units: sigma = eps = m = 1.  Integrator: BAOAB Langevin.
 *
 * Special features used by the intramembrane-cavitation experiments:
 *   --pin          : clamp z of head beads with r_xy > a_pin (BLS rim condition)
 *   --comrestr     : harmonic restraint on the leaflet-leaflet COM-z separation,
 *                    reporting the mean restraint force -> interleaflet
 *                    separation pressure Par(Z)
 *   --baro         : semi-isotropic Berendsen barostat (lateral / normal targets)
 *   --stretchz     : one-shot affine z-strain, to impose a rarefaction
 *
 * Build: make
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define W_TYPE 0
#define H_TYPE 1
#define T_TYPE 2
#define G_TYPE 3
#define NTYPE 4

static const double TWO_POW_1_6 = 1.1224620483093730;

/* ------------------------------------------------------------------ state */
typedef struct {
    int n;                  /* number of beads */
    int nlip;               /* number of lipids (beads 0..nbl*nlip-1) */
    int nbl;                /* beads per lipid: 1 head + (nbl-1) tails */
    double *x, *y, *z;
    double *vx, *vy, *vz;
    double *fx, *fy, *fz;
    int *type;
    double Lx, Ly, Lz;

    /* pair tables */
    double b[NTYPE][NTYPE];        /* repulsion diameter */
    double eatt[NTYPE][NTYPE];     /* attraction depth */
    double wc;                     /* attraction tail width */
    double rc_max;                 /* global cutoff */

    /* bonded */
    double k_fene, r0_fene, k_bend, r13_0;

    /* thermostat / integrator */
    double kT, dt, gamma;

    /* neighbour list */
    int *nlist;             /* flattened */
    int *nn;                /* neighbours per bead */
    int maxnb;
    double skin;
    double *x0, *y0, *z0;   /* positions at last rebuild */

    /* pinning */
    int pin_on;
    double a_pin, k_pin;
    int *pinned;            /* per-bead flag */
    double *zref;           /* reference z (scaled affinely with Lz) */

    /* COM-z restraint between leaflets */
    int com_on;
    double k_com, dz_target;
    int *grp;               /* 0 = none, 1 = lower leaflet tails, 2 = upper */
    int ngrp1, ngrp2;
    double com_force;       /* instantaneous restraint force (z), group2 side */

    /* virial accumulators */
    double vir_xx, vir_yy, vir_zz;

    uint64_t rng;
} Sys;

/* ------------------------------------------------------------------- rng */
static inline uint64_t xorshift64(uint64_t *s) {
    uint64_t v = *s;
    v ^= v << 13; v ^= v >> 7; v ^= v << 17;
    *s = v; return v;
}
static inline double uniform(uint64_t *s) {
    return (double)(xorshift64(s) >> 11) * (1.0 / 9007199254740992.0);
}
static double gauss(uint64_t *s) {
    /* Box-Muller with cached second value */
    static int have = 0; static double cached = 0.0;
    if (have) { have = 0; return cached; }
    double u1, u2;
    do { u1 = uniform(s); } while (u1 <= 1e-15);
    u2 = uniform(s);
    double r = sqrt(-2.0 * log(u1));
    cached = r * sin(2.0 * M_PI * u2); have = 1;
    return r * cos(2.0 * M_PI * u2);
}

/* Pool of unit normals, drawn by random index in the hot loop.  With 2^22
 * entries and an independent index stream per thread this is statistically
 * indistinguishable from fresh normals for our observables, and ~20x cheaper
 * than Box-Muller per draw. */
#define GPOOL_BITS 22
#define GPOOL_SIZE (1u << GPOOL_BITS)
#define GPOOL_MASK (GPOOL_SIZE - 1u)
static float *gpool = NULL;

static void init_gpool(uint64_t seed) {
    uint64_t r = seed ? seed : 7777777ULL;
    gpool = malloc((size_t)GPOOL_SIZE * sizeof(float));
    double mean = 0.0, var = 0.0;
    for (unsigned i = 0; i < GPOOL_SIZE; i++) { gpool[i] = (float)gauss(&r); mean += gpool[i]; }
    mean /= GPOOL_SIZE;
    for (unsigned i = 0; i < GPOOL_SIZE; i++) { gpool[i] -= (float)mean; var += (double)gpool[i]*gpool[i]; }
    var = sqrt(var / GPOOL_SIZE);
    for (unsigned i = 0; i < GPOOL_SIZE; i++) gpool[i] /= (float)var;
}

static inline double gpool_draw(uint64_t *st) {
    return gpool[xorshift64(st) & GPOOL_MASK];
}

/* -------------------------------------------------------------- utilities */
static inline double minimg(double d, double L) {
    /* displacements handed to this routine are always < 1.5 L */
    double h = 0.5 * L;
    if (d >  h) { d -= L; if (d >  h) d -= L; }
    else if (d < -h) { d += L; if (d < -h) d += L; }
    return d;
}

/* ------------------------------------------------- tabulated pair forces */
#define NTAB 8192
static double tab_e[NTYPE][NTYPE][NTAB];
static double tab_f[NTYPE][NTYPE][NTAB];   /* fmag = -dV/dr / r */
static double tab_dr2, tab_inv_dr2, tab_r2max;

static void pair_analytic(double r2, double bb, double ea, double wc,
                          double *e, double *fmag) {
    double r = sqrt(r2);
    double rcut = TWO_POW_1_6 * bb;
    double ee = 0.0, ff = 0.0;
    if (r < rcut) {
        double sr2 = bb * bb / r2;
        double sr6 = sr2 * sr2 * sr2;
        double sr12 = sr6 * sr6;
        ee += 4.0 * (sr12 - sr6) + 1.0;
        ff += 24.0 * (2.0 * sr12 - sr6) / r2;
    }
    if (ea > 0.0) {
        if (r < rcut) {
            ee += -ea;
        } else {
            double arg = M_PI * (r - rcut) / (2.0 * wc);
            if (arg < 0.5 * M_PI) {
                double c = cos(arg), sn = sin(arg);
                ee += -ea * c * c;
                ff += -(ea * c * sn * M_PI / wc) / r;
            }
        }
    }
    *e = ee; *fmag = ff;
}

static void build_tables(const double b[NTYPE][NTYPE], const double eatt[NTYPE][NTYPE],
                         double wc, double rc_max) {
    tab_r2max = rc_max * rc_max;
    tab_dr2 = tab_r2max / (NTAB - 1);
    tab_inv_dr2 = 1.0 / tab_dr2;
    for (int a = 0; a < NTYPE; a++)
        for (int c = 0; c < NTYPE; c++)
            for (int k = 0; k < NTAB; k++) {
                double r2 = k * tab_dr2;
                if (r2 < 0.04) r2 = 0.04;          /* table floor; guarded at use */
                pair_analytic(r2, b[a][c], eatt[a][c], wc, &tab_e[a][c][k], &tab_f[a][c][k]);
            }
}

static inline void pair_lookup(int ti, int tj, double r2, double *e, double *fmag) {
    double u = r2 * tab_inv_dr2;
    int k = (int)u;
    double frac = u - k;
    const double *te = tab_e[ti][tj], *tf = tab_f[ti][tj];
    *e = te[k] + frac * (te[k + 1] - te[k]);
    *fmag = tf[k] + frac * (tf[k + 1] - tf[k]);
}

static void wrap(Sys *s) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < s->n; i++) {
        while (s->x[i] <  0)      s->x[i] += s->Lx;
        while (s->x[i] >= s->Lx)  s->x[i] -= s->Lx;
        while (s->y[i] <  0)      s->y[i] += s->Ly;
        while (s->y[i] >= s->Ly)  s->y[i] -= s->Ly;
        while (s->z[i] <  0)      s->z[i] += s->Lz;
        while (s->z[i] >= s->Lz)  s->z[i] -= s->Lz;
    }
}

/* ------------------------------------------------------- neighbour list */
static void build_nlist(Sys *s) {
    double rl = s->rc_max + s->skin;
    int mx = (int)(s->Lx / rl); if (mx < 3) mx = 3;
    int my = (int)(s->Ly / rl); if (my < 3) my = 3;
    int mz = (int)(s->Lz / rl); if (mz < 3) mz = 3;
    double cx = s->Lx / mx, cy = s->Ly / my, cz = s->Lz / mz;
    int ncell = mx * my * mz;
    int *head = malloc(ncell * sizeof(int));
    int *next = malloc(s->n * sizeof(int));
    for (int c = 0; c < ncell; c++) head[c] = -1;
    for (int i = 0; i < s->n; i++) {
        int ix = (int)(s->x[i] / cx); if (ix >= mx) ix = mx - 1; if (ix < 0) ix = 0;
        int iy = (int)(s->y[i] / cy); if (iy >= my) iy = my - 1; if (iy < 0) iy = 0;
        int iz = (int)(s->z[i] / cz); if (iz >= mz) iz = mz - 1; if (iz < 0) iz = 0;
        int c = (ix * my + iy) * mz + iz;
        next[i] = head[c]; head[c] = i;
    }
    double rl2 = rl * rl;
    /* FULL neighbour lists: every pair appears in both partners' lists, so the
       force loop writes only to bead i and needs no atomics. */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < s->n; i++) {
        s->nn[i] = 0;
        int ix = (int)(s->x[i] / cx); if (ix >= mx) ix = mx - 1; if (ix < 0) ix = 0;
        int iy = (int)(s->y[i] / cy); if (iy >= my) iy = my - 1; if (iy < 0) iy = 0;
        int iz = (int)(s->z[i] / cz); if (iz >= mz) iz = mz - 1; if (iz < 0) iz = 0;
        int cnt = 0;
        for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++) {
            int jx = (ix + dx + mx) % mx, jy = (iy + dy + my) % my, jz = (iz + dz + mz) % mz;
            int c2 = (jx * my + jy) * mz + jz;
            for (int j = head[c2]; j >= 0; j = next[j]) {
                if (j == i) continue;
                double dxx = minimg(s->x[i] - s->x[j], s->Lx);
                double dyy = minimg(s->y[i] - s->y[j], s->Ly);
                double dzz = minimg(s->z[i] - s->z[j], s->Lz);
                if (dxx*dxx + dyy*dyy + dzz*dzz > rl2) continue;
                if (cnt >= s->maxnb) { fprintf(stderr, "neighbour overflow\n"); exit(1); }
                s->nlist[(size_t)i * s->maxnb + cnt++] = j;
            }
        }
        s->nn[i] = cnt;
    }
    free(head); free(next);
    memcpy(s->x0, s->x, s->n * sizeof(double));
    memcpy(s->y0, s->y, s->n * sizeof(double));
    memcpy(s->z0, s->z, s->n * sizeof(double));
}

static int need_rebuild(Sys *s) {
    double lim = 0.25 * s->skin; double lim2 = lim * lim;
    for (int i = 0; i < s->n; i++) {
        double dx = minimg(s->x[i] - s->x0[i], s->Lx);
        double dy = minimg(s->y[i] - s->y0[i], s->Ly);
        double dz = minimg(s->z[i] - s->z0[i], s->Lz);
        if (dx*dx + dy*dy + dz*dz > lim2) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------- forces */
static int bonded_pair(const Sys *s, int i, int j) {
    /* beads nbl*k .. nbl*k+nbl-1 form one lipid chain; all intra-lipid pairs are
       excluded from the pair loop and handled by the bonded terms instead */
    int nl = s->nbl * s->nlip;
    if (i >= nl || j >= nl) return 0;
    return (i / s->nbl == j / s->nbl);
}

static double compute_forces(Sys *s) {
    double epot = 0.0;
    for (int i = 0; i < s->n; i++) { s->fx[i] = s->fy[i] = s->fz[i] = 0.0; }
    s->vir_xx = s->vir_yy = s->vir_zz = 0.0;

    /* ---- non-bonded (full lists: each pair visited twice; halve at the end) */
    double pvxx = 0, pvyy = 0, pvzz = 0, pep = 0;
    double rc2 = s->rc_max * s->rc_max;
#pragma omp parallel for schedule(dynamic, 64) reduction(+:pvxx,pvyy,pvzz,pep)
    for (int i = 0; i < s->n; i++) {
        double xi = s->x[i], yi = s->y[i], zi = s->z[i];
        int ti = s->type[i];
        int nb = s->nn[i];
        const int *lst = &s->nlist[(size_t)i * s->maxnb];
        double fxi = 0, fyi = 0, fzi = 0;
        for (int k = 0; k < nb; k++) {
            int j = lst[k];
            if (bonded_pair(s, i, j)) continue;
            double dx = minimg(xi - s->x[j], s->Lx);
            double dy = minimg(yi - s->y[j], s->Ly);
            double dz = minimg(zi - s->z[j], s->Lz);
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > rc2) continue;
            int tj = s->type[j];
            double e, fmag;
            if (r2 < 0.04) {                /* far inside the core: use analytic form */
                pair_analytic(0.04, s->b[ti][tj], s->eatt[ti][tj], s->wc, &e, &fmag);
            } else {
                pair_lookup(ti, tj, r2, &e, &fmag);
            }
            pep += e;
            if (fmag != 0.0) {
                double fxv = fmag * dx, fyv = fmag * dy, fzv = fmag * dz;
                fxi += fxv; fyi += fyv; fzi += fzv;
                pvxx += fxv * dx; pvyy += fyv * dy; pvzz += fzv * dz;
            }
        }
        s->fx[i] = fxi; s->fy[i] = fyi; s->fz[i] = fzi;
    }
    epot += 0.5 * pep;
    s->vir_xx += 0.5 * pvxx; s->vir_yy += 0.5 * pvyy; s->vir_zz += 0.5 * pvzz;

    /* ---- bonded: FENE + intra-lipid repulsion + bending ---- */
    for (int l = 0; l < s->nlip; l++) {
        int base = l * s->nbl;
        /* chain bonds */
        for (int p = 0; p < s->nbl - 1; p++) {
            int i = base + p, j = base + p + 1;
            double dx = minimg(s->x[i]-s->x[j], s->Lx);
            double dy = minimg(s->y[i]-s->y[j], s->Ly);
            double dz = minimg(s->z[i]-s->z[j], s->Lz);
            double r2 = dx*dx+dy*dy+dz*dz, r = sqrt(r2);
            double r0 = s->r0_fene;
            if (r >= 0.999 * r0) { r = 0.999 * r0; r2 = r * r; }
            double q = r / r0;
            epot += -0.5 * s->k_fene * r0 * r0 * log(1.0 - q*q);
            double fmag = -(s->k_fene * r / (1.0 - q*q)) / r;
            double bb = s->b[s->type[i]][s->type[j]];
            double rcut = TWO_POW_1_6 * bb;
            if (r < rcut) {
                double q2 = bb*bb/r2, sr6 = q2*q2*q2, sr12 = sr6*sr6;
                epot += 4.0*(sr12-sr6)+1.0;
                fmag += 24.0*(2.0*sr12-sr6)/r2;
            }
            double fxv = fmag*dx, fyv = fmag*dy, fzv = fmag*dz;
            s->fx[i] += fxv; s->fy[i] += fyv; s->fz[i] += fzv;
            s->fx[j] -= fxv; s->fy[j] -= fyv; s->fz[j] -= fzv;
            s->vir_xx += fxv*dx; s->vir_yy += fyv*dy; s->vir_zz += fzv*dz;
        }
        /* bending: harmonic on every 1-3 distance along the chain */
        for (int p = 0; p + 2 < s->nbl; p++) {
            int i = base + p, j = base + p + 2;
            double dx = minimg(s->x[i]-s->x[j], s->Lx);
            double dy = minimg(s->y[i]-s->y[j], s->Ly);
            double dz = minimg(s->z[i]-s->z[j], s->Lz);
            double r2 = dx*dx+dy*dy+dz*dz, r = sqrt(r2);
            double dr = r - s->r13_0;
            epot += 0.5 * s->k_bend * dr * dr;
            double fmag = -s->k_bend * dr / r;
            double bb = s->b[s->type[i]][s->type[j]];
            double rcut = TWO_POW_1_6 * bb;
            if (r < rcut) {
                double q2 = bb*bb/r2, sr6 = q2*q2*q2, sr12 = sr6*sr6;
                epot += 4.0*(sr12-sr6)+1.0;
                fmag += 24.0*(2.0*sr12-sr6)/r2;
            }
            double fxv = fmag*dx, fyv = fmag*dy, fzv = fmag*dz;
            s->fx[i] += fxv; s->fy[i] += fyv; s->fz[i] += fzv;
            s->fx[j] -= fxv; s->fy[j] -= fyv; s->fz[j] -= fzv;
            s->vir_xx += fxv*dx; s->vir_yy += fyv*dy; s->vir_zz += fzv*dz;
        }
        /* non-adjacent intra-chain pairs (excluded above) still need repulsion */
        for (int p = 0; p + 3 < s->nbl; p++) {
            for (int q = p + 3; q < s->nbl; q++) {
                int i = base + p, j = base + q;
                double dx = minimg(s->x[i]-s->x[j], s->Lx);
                double dy = minimg(s->y[i]-s->y[j], s->Ly);
                double dz = minimg(s->z[i]-s->z[j], s->Lz);
                double r2 = dx*dx+dy*dy+dz*dz, r = sqrt(r2);
                double bb = s->b[s->type[i]][s->type[j]];
                double rcut = TWO_POW_1_6 * bb;
                if (r >= rcut) continue;
                double q2 = bb*bb/r2, sr6 = q2*q2*q2, sr12 = sr6*sr6;
                epot += 4.0*(sr12-sr6)+1.0;
                double fmag = 24.0*(2.0*sr12-sr6)/r2;
                double fxv = fmag*dx, fyv = fmag*dy, fzv = fmag*dz;
                s->fx[i] += fxv; s->fy[i] += fyv; s->fz[i] += fzv;
                s->fx[j] -= fxv; s->fy[j] -= fyv; s->fz[j] -= fzv;
                s->vir_xx += fxv*dx; s->vir_yy += fyv*dy; s->vir_zz += fzv*dz;
            }
        }
    }

    /* ---- rim pinning (BLS clamped-boundary condition) ---- */
    if (s->pin_on) {
        for (int i = 0; i < s->n; i++) {
            if (!s->pinned[i]) continue;
            double dz = minimg(s->z[i] - s->zref[i], s->Lz);
            epot += 0.5 * s->k_pin * dz * dz;
            s->fz[i] += -s->k_pin * dz;
        }
    }

    /* ---- leaflet-leaflet COM-z restraint ---- */
    s->com_force = 0.0;
    if (s->com_on) {
        /* unwrapped z relative to box centre keeps the two groups identifiable */
        double z1 = 0, z2 = 0;
        for (int i = 0; i < s->n; i++) {
            if (s->grp[i] == 1) z1 += s->z[i];
            else if (s->grp[i] == 2) z2 += s->z[i];
        }
        z1 /= s->ngrp1; z2 /= s->ngrp2;
        double dz = z2 - z1;
        double diff = dz - s->dz_target;
        epot += 0.5 * s->k_com * diff * diff;
        double F = -s->k_com * diff;      /* force on group 2 in +z */
        s->com_force = F;
        for (int i = 0; i < s->n; i++) {
            if (s->grp[i] == 2) s->fz[i] += F / s->ngrp2;
            else if (s->grp[i] == 1) s->fz[i] -= F / s->ngrp1;
        }
    }
    return epot;
}

/* -------------------------------------------------------- measurements */
static void pressure_tensor(Sys *s, double *pxx, double *pyy, double *pzz) {
    double kxx = 0, kyy = 0, kzz = 0;
    for (int i = 0; i < s->n; i++) {
        kxx += s->vx[i]*s->vx[i]; kyy += s->vy[i]*s->vy[i]; kzz += s->vz[i]*s->vz[i];
    }
    double V = s->Lx * s->Ly * s->Lz;
    *pxx = (kxx + s->vir_xx) / V;
    *pyy = (kyy + s->vir_yy) / V;
    *pzz = (kzz + s->vir_zz) / V;
}

static double kinetic(Sys *s) {
    double k = 0;
    for (int i = 0; i < s->n; i++) k += s->vx[i]*s->vx[i]+s->vy[i]*s->vy[i]+s->vz[i]*s->vz[i];
    return 0.5 * k;
}

/* leaflet inner-surface separation near the patch centre */
static void leaflet_gap(Sys *s, double rmax, double *gap, int *n_up, int *n_lo) {
    double xc = 0.5*s->Lx, yc = 0.5*s->Ly, zc = 0.5*s->Lz;
    double zu = 0, zl = 0; int nu = 0, nl = 0;
    for (int l = 0; l < s->nlip; l++) {
        int tip = l*s->nbl + s->nbl - 1;        /* terminal tail bead */
        double dx = minimg(s->x[tip]-xc, s->Lx);
        double dy = minimg(s->y[tip]-yc, s->Ly);
        if (dx*dx + dy*dy > rmax*rmax) continue;
        double dz = minimg(s->z[tip]-zc, s->Lz);
        /* leaflet identity from the head bead's side */
        double hz = minimg(s->z[l*s->nbl]-zc, s->Lz);
        if (hz > 0) { zu += dz; nu++; } else { zl += dz; nl++; }
    }
    *n_up = nu; *n_lo = nl;
    *gap = (nu && nl) ? (zu/nu - zl/nl) : 0.0;
}

/* ------------------------------------------------------------------ io */
static void read_config(Sys *s, const char *fn) {
    FILE *f = fopen(fn, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", fn); exit(1); }
    if (fscanf(f, "%d %d %d %lf %lf %lf", &s->n, &s->nlip, &s->nbl, &s->Lx, &s->Ly, &s->Lz) != 6) {
        fprintf(stderr, "bad header (expect: N nlip nbl Lx Ly Lz)\n"); exit(1);
    }
    if (s->nbl < 2) s->nbl = 2;
    s->x = malloc(s->n*sizeof(double)); s->y = malloc(s->n*sizeof(double)); s->z = malloc(s->n*sizeof(double));
    s->vx = calloc(s->n, sizeof(double)); s->vy = calloc(s->n, sizeof(double)); s->vz = calloc(s->n, sizeof(double));
    s->fx = calloc(s->n, sizeof(double)); s->fy = calloc(s->n, sizeof(double)); s->fz = calloc(s->n, sizeof(double));
    s->type = malloc(s->n*sizeof(int));
    for (int i = 0; i < s->n; i++) {
        if (fscanf(f, "%d %lf %lf %lf", &s->type[i], &s->x[i], &s->y[i], &s->z[i]) != 4) {
            fprintf(stderr, "bad line %d\n", i); exit(1);
        }
    }
    fclose(f);
}

static void write_config(Sys *s, const char *fn) {
    FILE *f = fopen(fn, "w");
    fprintf(f, "%d %d %d %.10f %.10f %.10f\n", s->n, s->nlip, s->nbl, s->Lx, s->Ly, s->Lz);
    for (int i = 0; i < s->n; i++)
        fprintf(f, "%d %.6f %.6f %.6f\n", s->type[i], s->x[i], s->y[i], s->z[i]);
    fclose(f);
}

static void dump_frame(Sys *s, FILE *f, double t) {
    float hdr[4] = {(float)t, (float)s->Lx, (float)s->Ly, (float)s->Lz};
    fwrite(hdr, sizeof(float), 4, f);
    for (int i = 0; i < s->n; i++) {
        float p[3] = {(float)s->x[i], (float)s->y[i], (float)s->z[i]};
        fwrite(p, sizeof(float), 3, f);
    }
}

/* ---------------------------------------------------------------- main */
int main(int argc, char **argv) {
    Sys s; memset(&s, 0, sizeof(s));
    const char *in = NULL, *out = "final.cfg", *logf = "run.log", *dumpf = NULL;
    long nsteps = 100000; long log_every = 500, dump_every = 0;
    double kT = 1.0, dt = 0.005, gamma = 1.0, wc = 1.3;
    int baro = 0; double p_lat = 0.0, p_norm = 0.0, tau_p = 5.0, kappa = 0.05;
    int pin = 0; double a_pin = 12.0, k_pin = 10.0;
    int com = 0; double k_com = 200.0, dz_target = -1.0;
    double stretchz = 0.0;
    unsigned long seed = 12345;
    double gap_rmax = 4.0;
    long softstart = 0; double fcap = 50.0; long ckpt_every = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in")) in = argv[++i];
        else if (!strcmp(argv[i], "--out")) out = argv[++i];
        else if (!strcmp(argv[i], "--log")) logf = argv[++i];
        else if (!strcmp(argv[i], "--dump")) dumpf = argv[++i];
        else if (!strcmp(argv[i], "--steps")) nsteps = atol(argv[++i]);
        else if (!strcmp(argv[i], "--logevery")) log_every = atol(argv[++i]);
        else if (!strcmp(argv[i], "--dumpevery")) dump_every = atol(argv[++i]);
        else if (!strcmp(argv[i], "--kT")) kT = atof(argv[++i]);
        else if (!strcmp(argv[i], "--dt")) dt = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gamma")) gamma = atof(argv[++i]);
        else if (!strcmp(argv[i], "--wc")) wc = atof(argv[++i]);
        else if (!strcmp(argv[i], "--baro")) baro = 1;
        else if (!strcmp(argv[i], "--plat")) p_lat = atof(argv[++i]);
        else if (!strcmp(argv[i], "--pnorm")) p_norm = atof(argv[++i]);
        else if (!strcmp(argv[i], "--taup")) tau_p = atof(argv[++i]);
        else if (!strcmp(argv[i], "--kappa")) kappa = atof(argv[++i]);
        else if (!strcmp(argv[i], "--pin")) pin = 1;
        else if (!strcmp(argv[i], "--apin")) a_pin = atof(argv[++i]);
        else if (!strcmp(argv[i], "--kpin")) k_pin = atof(argv[++i]);
        else if (!strcmp(argv[i], "--comrestr")) com = 1;
        else if (!strcmp(argv[i], "--kcom")) k_com = atof(argv[++i]);
        else if (!strcmp(argv[i], "--dztarget")) dz_target = atof(argv[++i]);
        else if (!strcmp(argv[i], "--stretchz")) stretchz = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed")) seed = strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--gaprmax")) gap_rmax = atof(argv[++i]);
        else if (!strcmp(argv[i], "--softstart")) softstart = atol(argv[++i]);
        else if (!strcmp(argv[i], "--fcap")) fcap = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ckptevery")) ckpt_every = atol(argv[++i]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
    }
    if (!in) { fprintf(stderr, "need --in config\n"); return 1; }

    read_config(&s, in);
    s.kT = kT; s.dt = dt; s.gamma = gamma; s.wc = wc;
    s.rng = seed ? seed : 88172645463325252ULL;

    /* pair tables */
    double bmat[NTYPE][NTYPE] = {
        /*        W     H     T     G   */
        /* W */ {1.00, 0.95, 1.00, 1.00},
        /* H */ {0.95, 0.95, 0.95, 0.95},
        /* T */ {1.00, 0.95, 1.00, 0.90},
        /* G */ {1.00, 0.95, 0.90, 0.90},
    };
    double emat[NTYPE][NTYPE] = {
        /*        W     H     T     G   */
        /* W */ {1.00, 1.00, 0.00, 0.00},
        /* H */ {1.00, 0.00, 0.00, 0.00},
        /* T */ {0.00, 0.00, 1.00, 0.60},
        /* G */ {0.00, 0.00, 0.60, 0.10},
    };
    memcpy(s.b, bmat, sizeof(bmat));
    memcpy(s.eatt, emat, sizeof(emat));
    double bmax = 1.0;
    s.rc_max = TWO_POW_1_6 * bmax + s.wc;
    build_tables(s.b, s.eatt, s.wc, s.rc_max);

    s.k_fene = 30.0; s.r0_fene = 1.5; s.k_bend = 10.0; s.r13_0 = 2.0;

    s.skin = 0.4;
    s.maxnb = 240;
    s.nlist = malloc((size_t)s.n * s.maxnb * sizeof(int));
    s.nn = malloc(s.n * sizeof(int));
    s.x0 = malloc(s.n*sizeof(double)); s.y0 = malloc(s.n*sizeof(double)); s.z0 = malloc(s.n*sizeof(double));

    /* optional one-shot affine z-strain (imposes a rarefaction) */
    if (stretchz != 0.0) {
        double f = 1.0 + stretchz;
        for (int i = 0; i < s.n; i++) s.z[i] *= f;
        s.Lz *= f;
    }

    /* pinning setup: head beads outside a_pin of the patch centre */
    s.pinned = calloc(s.n, sizeof(int));
    s.zref = calloc(s.n, sizeof(double));
    s.pin_on = pin; s.a_pin = a_pin; s.k_pin = k_pin;
    if (pin) {
        double xc = 0.5*s.Lx, yc = 0.5*s.Ly;
        int npin = 0;
        for (int l = 0; l < s.nlip; l++) {
            int h = l*s.nbl;
            double dx = minimg(s.x[h]-xc, s.Lx), dy = minimg(s.y[h]-yc, s.Ly);
            if (dx*dx + dy*dy > a_pin*a_pin) { s.pinned[h] = 1; s.zref[h] = s.z[h]; npin++; }
        }
        fprintf(stderr, "pinned %d head beads (a_pin=%.2f)\n", npin, a_pin);
    }

    /* COM restraint groups: terminal tail beads of each leaflet */
    s.grp = calloc(s.n, sizeof(int));
    s.com_on = com; s.k_com = k_com;
    {
        double zc = 0.5*s.Lz; int n1 = 0, n2 = 0;
        for (int l = 0; l < s.nlip; l++) {
            /* leaflet identity from the head bead's side of the midplane; the
               whole chain joins that leaflet's group, so the restraint
               coordinate is a genuine leaflet-leaflet separation rather than a
               tail-tip separation that lipid tilt could absorb */
            double hz = minimg(s.z[l*s.nbl]-zc, s.Lz);
            int g = (hz > 0) ? 2 : 1;
            for (int p = 0; p < s.nbl; p++) {
                s.grp[l*s.nbl + p] = g;
                if (g == 2) n2++; else n1++;
            }
        }
        s.ngrp1 = n1; s.ngrp2 = n2;
        fprintf(stderr, "leaflet groups: %d / %d beads\n", n1, n2);
    }
    if (com) {
        if (dz_target < 0) {
            double z1 = 0, z2 = 0;
            for (int i = 0; i < s.n; i++) { if (s.grp[i]==1) z1 += s.z[i]; else if (s.grp[i]==2) z2 += s.z[i]; }
            dz_target = z2/s.ngrp2 - z1/s.ngrp1;
            fprintf(stderr, "COM restraint: using current separation %.4f\n", dz_target);
        }
        s.dz_target = dz_target;
    }

    init_gpool(seed ^ 0x9e3779b97f4a7c15ULL);

    /* initial velocities */
    for (int i = 0; i < s.n; i++) {
        s.vx[i] = sqrt(kT) * gauss(&s.rng);
        s.vy[i] = sqrt(kT) * gauss(&s.rng);
        s.vz[i] = sqrt(kT) * gauss(&s.rng);
    }

    wrap(&s);
    build_nlist(&s);
    compute_forces(&s);

    FILE *lf = fopen(logf, "w");
    fprintf(lf, "# step time Epot/N T Pxx Pyy Pzz Lx Lz gap comF areaperlipid\n");
    FILE *df = dumpf ? fopen(dumpf, "wb") : NULL;
    if (df) {
        int32_t hdr[3] = {s.n, s.nlip, s.nbl};
        fwrite(hdr, sizeof(int32_t), 3, df);
        fwrite(s.type, sizeof(int), s.n, df);
    }

    double c1 = exp(-gamma * dt);
    double c2 = sqrt(kT * (1.0 - c1*c1));

    /* per-thread random streams for the O step */
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    uint64_t *tstate = malloc(nthreads * sizeof(uint64_t));
    for (int t = 0; t < nthreads; t++) tstate[t] = (seed ? seed : 1) * 2654435761ULL + 7919ULL * (t + 1);

    for (long step = 0; step <= nsteps; step++) {
        /* BAOAB: B + A fused, then O, then A -- all parallel */
#pragma omp parallel
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            uint64_t st = tstate[tid];
#pragma omp for schedule(static) nowait
            for (int i = 0; i < s.n; i++) {
                double vxi = s.vx[i] + 0.5*dt*s.fx[i];
                double vyi = s.vy[i] + 0.5*dt*s.fy[i];
                double vzi = s.vz[i] + 0.5*dt*s.fz[i];
                s.x[i] += 0.5*dt*vxi; s.y[i] += 0.5*dt*vyi; s.z[i] += 0.5*dt*vzi;
                /* O */
                vxi = c1*vxi + c2*gpool_draw(&st);
                vyi = c1*vyi + c2*gpool_draw(&st);
                vzi = c1*vzi + c2*gpool_draw(&st);
                s.vx[i] = vxi; s.vy[i] = vyi; s.vz[i] = vzi;
                s.x[i] += 0.5*dt*vxi; s.y[i] += 0.5*dt*vyi; s.z[i] += 0.5*dt*vzi;
            }
            tstate[tid] = st;
        }

        wrap(&s);
        if (step % 4 == 0 && need_rebuild(&s)) build_nlist(&s);
        double epot = compute_forces(&s);
        if (step < softstart) {
            /* soft start: cap forces and velocities so that a freshly built
               configuration with residual overlaps can relax without blowing up */
#pragma omp parallel for schedule(static)
            for (int i = 0; i < s.n; i++) {
                if (s.fx[i] >  fcap) s.fx[i] =  fcap;
                if (s.fx[i] < -fcap) s.fx[i] = -fcap;
                if (s.fy[i] >  fcap) s.fy[i] =  fcap;
                if (s.fy[i] < -fcap) s.fy[i] = -fcap;
                if (s.fz[i] >  fcap) s.fz[i] =  fcap;
                if (s.fz[i] < -fcap) s.fz[i] = -fcap;
                double vmax = 3.0 * sqrt(s.kT);
                if (s.vx[i] >  vmax) s.vx[i] =  vmax;
                if (s.vx[i] < -vmax) s.vx[i] = -vmax;
                if (s.vy[i] >  vmax) s.vy[i] =  vmax;
                if (s.vy[i] < -vmax) s.vy[i] = -vmax;
                if (s.vz[i] >  vmax) s.vz[i] =  vmax;
                if (s.vz[i] < -vmax) s.vz[i] = -vmax;
            }
        }

        /* B: velocity half kick */
#pragma omp parallel for schedule(static)
        for (int i = 0; i < s.n; i++) {
            s.vx[i] += 0.5*dt*s.fx[i]; s.vy[i] += 0.5*dt*s.fy[i]; s.vz[i] += 0.5*dt*s.fz[i];
        }

        double pxx, pyy, pzz;
        pressure_tensor(&s, &pxx, &pyy, &pzz);

        /* semi-isotropic Berendsen barostat */
        if (baro) {
            double plat_now = 0.5*(pxx+pyy);
            double mu_l = 1.0 - (dt/tau_p)*kappa*(p_lat - plat_now)/3.0;
            double mu_n = 1.0 - (dt/tau_p)*kappa*(p_norm - pzz)/3.0;
            if (mu_l < 0.99999) mu_l = 0.99999; if (mu_l > 1.00001) mu_l = 1.00001;
            if (mu_n < 0.99999) mu_n = 0.99999; if (mu_n > 1.00001) mu_n = 1.00001;
            for (int i = 0; i < s.n; i++) { s.x[i] *= mu_l; s.y[i] *= mu_l; s.z[i] *= mu_n; }
            for (int i = 0; i < s.n; i++) if (s.pinned[i]) s.zref[i] *= mu_n;
            s.Lx *= mu_l; s.Ly *= mu_l; s.Lz *= mu_n;
            /* dz_target is an absolute separation: deliberately NOT rescaled,
               so the imposed intramembrane gap is held while the box relaxes */
        }

        if (log_every && step % log_every == 0) {
            double gap; int nu, nl;
            leaflet_gap(&s, gap_rmax, &gap, &nu, &nl);
            double T = 2.0*kinetic(&s)/(3.0*s.n);
            double apl = (s.nlip > 0) ? (2.0*s.Lx*s.Ly/s.nlip) : 0.0;
            fprintf(lf, "%ld %.4f %.6f %.4f %.6f %.6f %.6f %.4f %.4f %.5f %.6f %.5f\n",
                    step, step*dt, epot/s.n, T, pxx, pyy, pzz, s.Lx, s.Lz, gap, s.com_force, apl);
            fflush(lf);
        }
        if (df && dump_every && step % dump_every == 0) dump_frame(&s, df, step*dt);
        if (ckpt_every && step > 0 && step % ckpt_every == 0) write_config(&s, out);
    }

    fclose(lf); if (df) fclose(df);
    write_config(&s, out);
    return 0;
}
