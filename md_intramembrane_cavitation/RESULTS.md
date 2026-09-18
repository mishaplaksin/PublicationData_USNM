# Results

All numbers below are measured in this repository. Reduced-unit results use
σ = 0.600 nm and 1 ε/σ³ = 21.9 MPa, fixed by matching the CG hydrophobic core
thickness (4.67 σ) to a 2.8 nm acyl core and the CG tail/vapour tension
(γ* = 1.908 ε/σ²) to 25 mN/m. The water-anchored mapping (γ = 70 mN/m) would
multiply every pressure below by 2.9; the choice made is the conservative one.

Sections 3 and 4 use **no coarse-grained mapping at all** — they follow from
interfacial tensions and the model's own geometry — so they are the most robust
part of the study.

## 0. What the BLS/NICE model predicts at its own operating point

From `scripts/bls_ode.py`, i.e. this repository's `Sonophore.m` with
f = 0.69 MHz, P_A = 320.608 kPa, T = 36 °C, RS-cell δ = 1.2553 nm:

| Quantity | Value |
|---|---|
| Peak leaflet displacement Z | **8.80 nm** |
| C_m / C_m0 at peak expansion | 0.187 |
| Quasi-static Z at 320.6 kPa, a = 32 / 16 / 8 nm | 7.79 / 3.13 / 1.27 nm |
| Cavity air content at rest → peak | **95.9 → 778 molecules** |
| Surface density of cavity air | 0.0298 molecules/nm² |
| Rarefaction the model needs to open the gap by 1 nm | **82 kPa** |

Two consequences make a molecular test both possible and sharp: the model's
intramembrane "gas" is **one to two air molecules per 10 nm × 10 nm of
membrane** (so P_in ≈ 1 atm is a near-vacuum, not a gas phase), and its leaflet
cohesion A_R = 10⁵ Pa is a ~1 atm-scale interaction.

## 1. Experiment A — pressure required to separate the leaflets

Whole-leaflet COM-separation restraint windows on a tensionless periodic
bilayer (288 lipids, 3050 beads, σ = 0.60 nm → 8.3 nm × 8.3 nm patch), barostat
at zero external pressure, ≥ 35k steps per window with block-averaged errors.
`figs/fig3_separation_pressure.png`.

*(table inserted from `runs/parz_result.json` — see §5 for the generated values)*

Because the membrane is free to respond however it likes, and does choose the
cheapest route (it admits solvent into the core rather than holding a dry
void), **P_sep is a rigorous lower bound** on the rarefaction the BLS mechanism
would require.

The measured work of adhesion, ≈ 100 mN/m, is not an artefact of the CG model:
it equals 2γ_tail/water, the unavoidable cost of replacing leaflet–leaflet
contact with two hydrocarbon/water interfaces. The BLS A_R term implies
0.72 mN/m for the same quantity.

## 2. Experiment B — the drive applied directly

The equilibrated bilayer held at fixed external normal pressure; lateral
pressure zero. `figs/fig5_ladder.png`.

*(table inserted from `runs/ladder12_result.json` — see §5)*

The gap does not open at the PRX amplitude, and barely moves even at 30 MPa.
What yields instead is the solvent: the box stretches in z and the achieved
normal pressure saturates, i.e. the water reaches its tensile limit before the
hydrophobic core will part.

## 3. The acoustic timescale (no CG mapping involved)

MD reaches tens of ns; one 0.69 MHz half-cycle is 725 ns. `scripts/nucleation.py`
covers the gap analytically, using only the hydrocarbon/vapour tension
γ = 20–50 mN/m (central value 25 mN/m):

| Quantity at P_A = 320.6 kPa | γ = 20 | 25 | 30 | 50 mN/m |
|---|---|---|---|---|
| Critical cavity radius r* [nm] | 125 | **156** | 187 | 312 |
| Nucleation barrier ΔG* [kT] | 3.1×10⁵ | **6.0×10⁵** | 1.0×10⁶ | 4.8×10⁶ |
| Rarefaction for ΔG* = 60 kT [MPa] | 22.9 | **32.0** | 42.0 | 90.4 |
| Slit-cavity growth threshold 2γ/δ [MPa] | 31.9 | **39.8** | 47.8 | 79.7 |

At the drive amplitude the critical cavity radius is **4.9× larger than the
entire a = 32 nm patch**, and the barrier is ~6×10⁵ kT. For a slit cavity of
the model's own thickness δ, W(R) ∝ R² *increases* with radius at this
pressure: no disc of any radius can grow.

## 4. The configuration the model assumes

Interfacial cost of the BLS cavity relative to a flat bilayer whose leaflets
are in contact (γ = 25 mN/m, a = 32 nm, two hydrocarbon/vapour interfaces):

| Configuration | Interface area [nm²] | Energy above flat [kT] |
|---|---|---|
| Rest, gap = δ | 6434 | **3.77×10⁴** |
| Z = 1 nm | 6440 | 3.77×10⁴ |
| Z = 4 nm | 6535 | 3.83×10⁴ |
| Z = 8.8 nm (model peak) | 6921 | 4.05×10⁴ |

The resting cavity the model posits is ~4×10⁴ kT above the contacting state, so
it is not a metastable configuration; the pressure driving it shut is
2γ/δ = 39.8 MPa, against the model's A_R = 0.1 MPa.

**Most generous possible bound, allowing for the dome geometry.** The MD imposes
a uniform separation whereas the BLS cavity is a clamped dome. The dome cannot
help in any way that matters: the interfacial cost scales with separated area
(2γπa²) however the surface is shaped, while the drive supplies P·πa²(δ+Z).
Hence P_min(Z) = 2γ/(δ+Z), ignoring tension, bending, electrostatics and any
barrier — all of which raise it:

| Z [nm] | 0 | 1 | 4 | 8.8 |
|---|---|---|---|---|
| P_min [MPa] | 39.8 | 22.2 | 9.5 | **4.97** |
| × PRX amplitude | 124 | 69 | 30 | **16** |

## 5. Generated tables

Run `python3 scripts/analyze_parz.py` and `analyze_ladder.py` to regenerate;
the committed JSON in `runs/` holds the values used for the figures.

## 6. What this does and does not establish

**Does:**
- Every independent route — MD restraint work, MD direct loading, classical
  nucleation, slit-cavity energetics, and a geometry-inclusive lower bound —
  puts the rarefaction needed to open an intramembrane cavity at
  **5–120 MPa**, i.e. **16–370× the 320.6 kPa** used in the paper.
- The discrepancy is traceable to one parameter: A_R = 10⁵ Pa represents
  leaflet cohesion as a ~1 atm interaction, whereas parting two hydrocarbon
  surfaces in water costs ~100 mN/m, which over a ~1 nm gap is ~10² MPa.
- In this model the membrane is not the weakest link: under strong rarefaction
  the solvent reaches its tensile limit while the hydrophobic core stays shut,
  and when the leaflets are forced apart the core hydrates rather than
  cavitating.

**Does not:**
- Rule out that ultrasound modulates neurons — only this specific mechanical
  route at this amplitude. Membrane-mediated effects on mechanosensitive
  channels, radiation force, and thermal routes are untouched by these results.
- Reach the 725 ns half-cycle in MD: §1–2 are quasi-static (justified because
  the membrane relaxes in ns while the drive changes over µs), and the
  timescale question is answered analytically in §3, not by simulation.
- Use an atomistic model. The CG model has one lipid chain of three tail beads,
  and its "oil" and "water" have nearly equal surface tension, so absolute
  pressures carry the mapping uncertainty stated above (a factor ~3). The
  conclusion survives because the gap is 2–3 orders of magnitude.
- Include a pre-existing gas nucleus of non-physiological size, dissolved gas
  at supersaturation, membrane proteins, cytoskeletal attachment, or
  curvature — any of which would need separate study.
- Explore patch sizes beyond ~8 nm, or rim-clamped geometry in MD; the dome
  bound in §4 is analytic, not simulated.
