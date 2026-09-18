#!/usr/bin/env python3
"""Figures for the intramembrane-cavitation MD study."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A
import bls_ode as B

# validated categorical slots 1-3 (all-pairs safe) + recessive ink
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
C4 = "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.labelcolor": INK, "text.color": INK,
    "grid.color": "#e3e2dd", "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0,
})
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")


def fig_bls_reference():
    """BLS/NICE reference: leaflet displacement and cavity gas content at the
    PRX 2014 operating point, for the published patch radius and the MD one."""
    d = np.load(os.path.join(RUNS, "bls_reference.npz"))
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))

    # (a) Z(t) at a = 32 nm
    t_us = d["t32"] * 1e6
    m = t_us < 6
    ax[0].plot(t_us[m], d["Z32"][m] * 1e9, color=C1)
    ax[0].set_xlabel("time [µs]"); ax[0].set_ylabel("leaflet displacement Z [nm]")
    ax[0].set_title("(a) BLS at PRX conditions\n0.69 MHz, 320.6 kPa, a = 32 nm", loc="left")
    ax[0].grid(axis="y")
    ax[0].annotate(f"peak {d['Z32'].max()*1e9:.1f} nm", xy=(0.97, 0.9),
                   xycoords="axes fraction", ha="right", color=INK2, fontsize=8)

    # (b) gas content in molecules
    ax[1].plot(t_us[m], d["n32"][m] * B.NAV, color=C2, label="a = 32 nm")
    ax[1].plot(d["t8"][d["t8"] * 1e6 < 6] * 1e6,
               d["n8"][d["t8"] * 1e6 < 6] * B.NAV, color=C3, label="a = 8 nm (MD scale)")
    ax[1].set_xlabel("time [µs]"); ax[1].set_ylabel("air content of cavity [molecules]")
    ax[1].set_title("(b) intramembrane gas content", loc="left")
    ax[1].set_yscale("log"); ax[1].grid(axis="y"); ax[1].legend()

    # (c) quasi-static Z(P) for three radii
    P = d["qs_P"] / 1e6
    for a, c in ((32, C1), (16, C2), (8, C3)):
        key = f"qs_a{a}nm_frozen"
        ax[2].plot(P, d[key] * 1e9, color=c, label=f"a = {a} nm")
    ax[2].axvline(B.PA_US / 1e6, color=MUTED, ls="--", lw=1)
    ax[2].annotate("PRX amplitude\n320.6 kPa", xy=(B.PA_US / 1e6 + 0.02, 0.8),
                   color=INK2, fontsize=7.5, va="bottom")
    ax[2].set_xlabel("rarefaction pressure [MPa]")
    ax[2].set_ylabel("equilibrium Z [nm]")
    ax[2].set_title("(c) quasi-static BLS response", loc="left")
    ax[2].grid(axis="y"); ax[2].legend()

    fig.tight_layout()
    out = os.path.join(FIGS, "fig1_bls_reference.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out)


def fig_membrane_structure(dump, log, tag="fig2_membrane"):
    n, nlip, nbl, types, frames = A.read_dump(dump)
    lg = A.read_log(log)
    t, L, pos = frames[-1]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))

    for name, sp, c in (("solvent", [0], C1), ("head", [1], C2), ("tail", [2], C3)):
        z, rho = A.density_profile(types, frames, L, sp, nbins=90, frac=0.6)
        ax[0].plot(z - L[2] / 2, rho, color=c, label=name)
    ax[0].set_xlabel("z − z$_{mid}$ [σ]"); ax[0].set_ylabel("number density [σ$^{-3}$]")
    ax[0].set_title("(a) equilibrated bilayer", loc="left")
    ax[0].legend(); ax[0].grid(axis="y")

    half = slice(len(lg["step"]) // 2, None)
    ax[1].plot(lg["time"], lg["apl"], color=C1)
    ax[1].set_xlabel("time [τ]"); ax[1].set_ylabel("area per lipid [σ²]")
    ax[1].set_title(f"(b) area per lipid → {lg['apl'][half].mean():.3f} σ²", loc="left")
    ax[1].grid(axis="y")

    ax[2].plot(lg["time"], lg["pzz"], color=C2, lw=1, label="P$_{zz}$")
    ax[2].plot(lg["time"], 0.5 * (lg["pxx"] + lg["pyy"]), color=C1, lw=1, label="P$_{lat}$")
    ax[2].axhline(0, color=MUTED, lw=0.8)
    ax[2].set_xlabel("time [τ]"); ax[2].set_ylabel("pressure [ε/σ³]")
    ax[2].set_title("(c) barostat: tensionless state", loc="left")
    ax[2].legend(); ax[2].grid(axis="y")

    fig.tight_layout()
    out = os.path.join(FIGS, f"{tag}.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out)


def fig_parz(result_json):
    with open(result_json) as fh:
        r = json.load(fh)
    d = np.array(r["d_nm"]); P = np.array(r["P_sep_MPa"]); E = np.array(r["P_sep_MPa_err"])
    bz = np.array(r["bls"]["Z_nm"]); bp = np.array(r["bls"]["P_coh_MPa"])
    bpn = np.array(r["bls"]["P_coh_nogas_MPa"])

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # (a) linear
    m = d >= -0.05
    ax[0].errorbar(d[m], P[m], yerr=E[m], color=C1, marker="o", ms=4.5, lw=2,
                   capsize=2, label="MD (this work)")
    ax[0].plot(2 * bz, bp, color=C2, label="BLS model (A$_R$ = 10$^5$ Pa)")
    ax[0].axhline(r["bls"]["PA_PRX_kPa"] / 1e3, color=MUTED, ls="--", lw=1)
    ax[0].annotate("PRX drive 320.6 kPa", xy=(0.98, 0.06), xycoords="axes fraction",
                   ha="right", color=INK2, fontsize=7.5)
    ax[0].set_xlabel("intramembrane gap opening d = 2Z [nm]")
    ax[0].set_ylabel("required rarefaction pressure [MPa]")
    ax[0].set_title("(a) pressure needed to hold the leaflets apart", loc="left")
    ax[0].set_xlim(-0.03, max(1.2, d[m].max() * 1.1))
    ax[0].grid(axis="y")
    h, l = ax[0].get_legend_handles_labels()
    order = [l.index("MD (this work)"), l.index("BLS model (A$_R$ = 10$^5$ Pa)")]
    ax[0].legend([h[i] for i in order], [l[i] for i in order])

    # (b) log, showing the scale separation
    pos = P > 0
    ax[1].semilogy(d[pos], P[pos], color=C1, marker="o", ms=4.5, label="MD (this work)")
    ax[1].semilogy(2 * bz, np.clip(bp, 1e-6, None), color=C2, label="BLS, Henry gas")
    ax[1].semilogy(2 * bz, np.clip(bpn, 1e-6, None), color=C3, ls=":",
                   label="BLS, gas-free")
    ax[1].axhline(r["bls"]["PA_PRX_kPa"] / 1e3, color=MUTED, ls="--", lw=1)
    ax[1].annotate("PRX drive", xy=(0.98, 0.32 / 1), xycoords=("axes fraction", "data"),
                   ha="right", va="bottom", color=INK2, fontsize=7.5)
    ax[1].set_xlabel("intramembrane gap opening d = 2Z [nm]")
    ax[1].set_ylabel("required rarefaction pressure [MPa]")
    # analytic slit-cavity thresholds, independent of the CG mapping
    for val, lab, st in ((39.8, "2γ$_{oil/vap}$/δ = 40 MPa", "-"),
                         (79.7, "2γ$_{oil/water}$/δ = 80 MPa", "--")):
        ax[1].axhline(val, color=MUTED, lw=0.9, ls=st)
        ax[1].annotate(lab, xy=(0.02, val), xycoords=("axes fraction", "data"),
                       va="bottom", fontsize=7, color=INK2)
    ax[1].set_title("(b) same data, log scale, with analytic thresholds", loc="left")
    ax[1].grid(axis="y"); ax[1].legend(loc="lower right", fontsize=7.5)

    fig.tight_layout()
    out = os.path.join(FIGS, "fig3_separation_pressure.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out)


def fig_snapshots(entries, tag="fig4_snapshots"):
    """x-z projections of the bilayer under different imposed rarefactions."""
    fig, axes = plt.subplots(1, len(entries), figsize=(3.0 * len(entries), 3.4),
                             sharey=True)
    if len(entries) == 1:
        axes = [axes]
    for ax, (dump, label) in zip(axes, entries):
        n, nlip, nbl, types, frames = A.read_dump(dump)
        t, L, pos = frames[-1]
        zc = L[2] / 2
        z = pos[:, 2] - zc
        z -= L[2] * np.round(z / L[2])
        # thin slab in y so the projection is readable
        sl = np.abs(pos[:, 1] - L[1] / 2) < 0.18 * L[1]
        for sp, c, sz, lab in ((0, C1, 5, "solvent"), (1, C2, 9, "head"),
                               (2, C3, 7, "tail"), (3, C4, 26, "gas")):
            m = sl & (types == sp)
            if m.sum():
                ax.scatter(pos[m, 0], z[m], s=sz, c=c, linewidths=0, label=lab)
        ax.set_xlabel("x [σ]")
        ax.set_title(label, loc="left", fontsize=9)
        ax.set_xlim(0, L[0]); ax.set_ylim(-7, 7)
        ax.grid(False)
    axes[0].set_ylabel("z − z$_{mid}$ [σ]")
    axes[-1].legend(loc="upper right", markerscale=1.6, fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIGS, f"{tag}.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out)


def fig_ladder(result_json):
    with open(result_json) as fh:
        r = json.load(fh)
    runs = sorted(r["runs"], key=lambda x: x["rarefaction_MPa"])
    P = np.array([x["rarefaction_MPa"] for x in runs])
    gap = np.array([x["gap_nm"] for x in runs])
    gaperr = np.array([x["gap_err_star"] * r["sigma_nm"] for x in runs])
    core = np.array([x.get("n_solvent_in_core", np.nan) for x in runs])
    void = np.array([x.get("void_volume_nm3", np.nan) for x in runs])

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax[0].errorbar(P, gap, yerr=gaperr, color=C1, marker="o", ms=5, capsize=2,
                   label="MD intramembrane gap")
    # BLS prediction over the same range, a = 32 nm and the MD patch size
    d = np.load(os.path.join(RUNS, "bls_reference.npz"))
    Pq = d["qs_P"] / 1e6
    ax[0].plot(Pq, 2 * d["qs_a32nm_frozen"] * 1e9 + gap[0], color=C2,
               label="BLS, a = 32 nm")
    ax[0].plot(Pq, 2 * d["qs_a8nm_frozen"] * 1e9 + gap[0], color=C3, ls="--",
               label="BLS, a = 8 nm")
    ax[0].axvline(B.PA_US / 1e6, color=MUTED, ls=":", lw=1)
    ax[0].annotate("PRX 320.6 kPa", xy=(B.PA_US / 1e6, 0.92), xycoords=("data", "axes fraction"),
                   rotation=90, fontsize=7, color=INK2, ha="right", va="top")
    ax[0].set_xscale("symlog", linthresh=0.3)
    ax[0].set_xlabel("imposed rarefaction [MPa]")
    ax[0].set_ylabel("intramembrane gap [nm]")
    ax[0].set_title("(a) gap vs imposed rarefaction", loc="left")
    ax[0].grid(axis="y"); ax[0].legend(fontsize=7.5)

    ax2 = ax[1]
    ax2.plot(P, void, color=C1, marker="o", ms=5, label="void volume [nm³]")
    ax2.plot(P, core, color=C2, marker="s", ms=5, label="solvent beads in core")
    ax2.set_xscale("symlog", linthresh=0.3)
    ax2.set_xlabel("imposed rarefaction [MPa]")
    ax2.set_ylabel("count / volume")
    ax2.set_title("(b) void formation and core hydration", loc="left")
    ax2.axvline(B.PA_US / 1e6, color=MUTED, ls=":", lw=1)
    ax2.grid(axis="y"); ax2.legend(fontsize=7.5)

    fig.tight_layout()
    out = os.path.join(FIGS, "fig5_ladder.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "bls"):
        fig_bls_reference()
    if which in ("all", "membrane") and os.path.exists(os.path.join(RUNS, "bl4_eq.dump")):
        fig_membrane_structure(os.path.join(RUNS, "bl4_eq.dump"),
                               os.path.join(RUNS, "bl4_eq.log"))
    if which in ("all", "parz") and os.path.exists(os.path.join(RUNS, "parz_result.json")):
        fig_parz(os.path.join(RUNS, "parz_result.json"))
    if which in ("all", "ladder") and os.path.exists(os.path.join(RUNS, "ladder12_result.json")):
        fig_ladder(os.path.join(RUNS, "ladder12_result.json"))
    if which in ("all", "snap"):
        ents = []
        for pn, lab in (("0.00387", "control (0 MPa)"), ("-0.01004", "0.32 MPa (PRX)"),
                        ("-0.13272", "3 MPa"), ("-1.36899", "30 MPa")):
            f = os.path.join(RUNS, "ladder12", f"pn{pn}.dump")
            if os.path.exists(f):
                ents.append((f, lab))
        if ents:
            fig_snapshots(ents)
