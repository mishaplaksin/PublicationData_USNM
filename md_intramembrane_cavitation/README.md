# Molecular-dynamics test of intramembrane cavitation at the PRX 2014 operating point

A coarse-grained molecular-dynamics study asking whether a lipid bilayer, driven
under the conditions of

> M. Plaksin, S. Shoham & E. Kimmel, *Intramembrane cavitation as a predictive
> bio-piezoelectric mechanism for ultrasonic brain stimulation*,
> Phys. Rev. X **4**, 011004 (2014)

opens an intramembrane cavity, and how the pressure required to separate the two
leaflets compares with what the BLS/NICE model assumes.

Everything here is self-contained: a purpose-written CG MD engine (C, OpenMP), a
faithful Python port of the repository's own BLS ODE, the run drivers and the
analysis. No external MD package is required.

---

## 1. Why the experiment had to be designed this way

A 0.69 MHz acoustic period is 1.45 µs. Atomistic or coarse-grained MD on this
hardware (4 CPU cores, no GPU) reaches tens of nanoseconds, i.e. ~10⁻² of one
acoustic cycle, and the published BLS radius a = 32 nm needs a ≳ 80 nm box.
Directly integrating the membrane through an acoustic cycle at the published
patch size is therefore not possible — not on this machine, and not in any
published study to date.

Two facts about the acoustic field make a rigorous test possible anyway:

1. **The field is spatially uniform on the simulation scale.** At 0.69 MHz the
   wavelength in water is ≈ 2.1 mm, ~10⁵ times the box. The acoustic field can
   only enter a box this small as a spatially uniform, time-varying pressure.
2. **The drive is quasi-static for the membrane.** The membrane's own mechanical
   relaxation is nanoseconds; the drive changes over microseconds. Each acoustic
   half-cycle is therefore a *constant* rarefaction as far as the membrane is
   concerned, so holding the membrane at a fixed rarefaction pressure and asking
   what it does is the physically correct reduction — not a shortcut.

The BLS model also does **not** claim de-novo bubble nucleation: it posits a
pre-existing intramembrane space of thickness δ ≈ 1.26 nm that expands. The
sharp, answerable question is therefore *mechanical*:

> What external rarefaction pressure is required to separate the two leaflets by
> a given amount, and how does it compare with the 320.6 kPa used in the paper?

That question is answered here in two independent ways (§4, §5).

## 2. What the BLS model itself implies

`scripts/bls_ode.py` is a transcription of `Sonophore.m` from
`H&H Simulations.7z` in this repository, with the parameters taken from
`LTS_RS_FS_combined_model.m`: **f = 0.69 MHz, P_A = 320.608 kPa**, T = 36 °C,
a = 32 nm, δ_RS = 1.2553 nm, δ₀ = 2 nm, δ_old = 1.4 nm, A_R = 10⁵ Pa, m = 5,
n = 3.3, k_s = 0.12 N/m, μ_s = 0.035 Pa·s, μ_l = 0.7 mPa·s, ρ = 1028 kg/m³,
C_a = 0.62 mol/m³, D_a = 3·10⁻⁹ m²/s, ζ = 0.5 nm, Q = C_m0·V_m0.

Integrating it reproduces the published behaviour (fig. `figs/fig1_bls_reference.png`):
peak leaflet displacement Z ≈ 8.8 nm, with C_m/C_m0 falling to 0.19 at peak
expansion. Two quantitative consequences are worth stating because they set up
the MD:

- **Gas content.** n_a⁰ = P₀·πa²δ/(R T) is **96 air molecules** at rest, rising
  to ≈ 780 at peak expansion. Per unit area that is 0.030 molecules/nm², i.e. a
  10 nm × 10 nm patch of membrane contains **one to two air molecules** in its
  intramembrane space. The model's "gas" is far too dilute to be a gas *phase*;
  P_in ≈ 1 atm is a near-vacuum on the molecular scale.
- **Cohesion scale.** A_R = 10⁵ Pa ≈ 1 atm sets the leaflet attraction, and the
  rarefaction the model needs to open the gap by 1 nm is ≈ 82 kPa. Hydrophobic
  adhesion between hydrocarbon surfaces in water is 2γ ≈ 40–50 mN/m, which over
  a ~1 nm separation is tens of MPa. The MD measures which is right.

## 3. Model, engine and unit mapping

**Engine** (`src/cgmd.c`, ~700 lines): Cooke–Deserno-style CG lipids with
*explicit* solvent and an optional hydrophobic gas species.

- Pair interactions, all species: `V_rep = 4ε[(b/r)¹²−(b/r)⁶]+ε` for r < 2^(1/6)b,
  plus a cos² attractive tail of width w_c = 1.3 σ and depth ε_att.
- ε_att: W–W = 1, W–H = 1, T–T = 1, T–G = 0.6, G–G = 0.1; **W–T = W–G = 0**, so
  the hydrophobic effect and gas insolubility in water are built in, not fitted.
- Lipids: one head + three tail beads, FENE bonds (k = 30, r₀ = 1.5 σ) plus
  harmonic 1–3 stiffening (k = 10, r₁₃ = 2 σ).
- BAOAB Langevin at kT = 1.0 ε, dt = 0.005 τ, γ = 1/τ; full Verlet neighbour
  lists; tabulated pair forces; semi-isotropic Berendsen barostat with
  independent lateral and normal targets.
- Purpose-built features for this study: a whole-leaflet COM-z separation
  restraint that reports its own mean force (§4); a cylindrical rim clamp
  emulating the BLS pinned boundary; one-shot affine z-strain.

**Validation.** T = 1.000 ± 0.01 with isotropic pressure in bulk liquid; the
bilayer stays intact with a dry core (solvent density at the midplane
0.005 σ⁻³ vs 0.90 σ⁻³ for tails) and is fluid (lipid lateral
D ≈ 0.014 σ²/τ).

**Unit mapping.** Length is set by the membrane itself, energy by the
interfacial tension that governs interleaflet adhesion:

| Anchor | Measurement | Result |
|---|---|---|
| Hydrophobic core thickness | CG core 4.67 σ (tail ρ > 50 %) ↔ 2.8 nm acyl core | **σ = 0.600 nm** |
| (head-peak separation, for reference) | 5.69 σ = 3.4 nm | — |
| Tail/vapour tension | γ*_tail = 1.908 ± 0.052 ε/σ² ↔ 25 mN/m | ε = 4.71·10⁻²¹ J, **1 ε/σ³ = 21.9 MPa** |
| Water/vapour tension | γ*_water = 1.856 ± 0.047 ε/σ² ↔ 70 mN/m | 1 ε/σ³ = 62.9 MPa |

The model gives its oil and its water nearly the same surface tension (both have
ε = 1), so it cannot be anchored on both simultaneously. Interleaflet adhesion is
set by the hydrocarbon interface, so the **tail anchor (21.9 MPa) is used
throughout and is the conservative choice**; the water anchor would raise every
reported pressure by 2.9×.

Consequences worth stating plainly, since they are the honest cost of coarse
graining: one bead occupies 0.242 nm³ (≈ 8 water molecules, mass ≈ 145 u), the
time unit is τ ≈ 4.3 ps so dt ≈ 21 fs, and kT = 1.0 ε corresponds to 341 K
rather than the target 309 K — a 10 % temperature mismatch. None of these
affect the conclusions, which rest on a 2–3 order-of-magnitude separation and,
in §3–4 of `RESULTS.md`, on no mapping at all.

## 4. Experiment A — pressure required to separate the leaflets

A tensionless periodic bilayer is held with a stiff harmonic restraint on the
**whole-leaflet** COM-z separation (not tail tips, which lipid tilt could absorb),
under a semi-isotropic barostat at zero external pressure. The mean restraint
force per unit area,

    P_sep(d) = ⟨F⟩ / (L_x L_y),   d = ⟨Δz⟩ − Δz₀,

is the external rarefaction pressure required to sustain an intramembrane gap of
opening d. Because the membrane is free to respond however it likes — and it
does choose the cheapest route, admitting solvent into the core rather than
holding a dry void — **P_sep is a rigorous lower bound** on the rarefaction the
BLS mechanism would need.

The BLS counterpart, at the same gap opening and excluding the stretching term
Ps (the MD separation is laterally uniform, so no area strain is involved), is

    P_coh(Z) = P₀ − P_in(Z) − Par(Z) − Pat(Z),   d = 2Z.

## 5. Experiment B — the drive applied directly

The equilibrated bilayer is held at a fixed external normal pressure equal to
P₀ − P_A with P_A stepped over 0 (control), **0.3206 MPa (the PRX amplitude)**,
1, 3, 10 and 30 MPa, lateral pressure at zero. For each, the achieved mean
pressure, the intramembrane gap, the solvent that entered the core, and the
largest void and its location (membrane core vs bulk solvent) are measured. This
is the acoustic rarefaction half-cycle applied as it acts on a box this size.

## 6. Reproducing

```bash
make -C src
python3 scripts/bls_ode.py                       # BLS reference + gas content
python3 scripts/build_system.py bilayer --out runs/bl.cfg --nx 12 --ny 12 \
        --apl 1.28 --lz 20 --rhow 0.88 --ntail 3
src/cgmd --in runs/bl.cfg --out runs/bl_eq.cfg --log runs/bl_eq.log \
        --steps 120000 --kT 1.0 --softstart 3000 --baro --plat 0 --pnorm 0
python3 scripts/recenter.py runs/bl_eq.cfg runs/bl_c.cfg
bash scripts/run_parz.sh  runs/bl_c.cfg runs/parz  20000 40000 1500 <targets>
bash scripts/run_ladder.sh runs/bl_c.cfg runs/ladder 80000 <reduced pressures>
python3 scripts/analyze_parz.py   --dir runs/parz  ...
python3 scripts/analyze_ladder.py --dir runs/ladder ...
python3 scripts/figures.py
```

## 7. Files

| Path | Contents |
|---|---|
| `src/cgmd.c` | CG MD engine |
| `scripts/bls_ode.py` | BLS/NICE ODE port (from this repo's `Sonophore.m`) |
| `scripts/build_system.py` | bilayer / liquid / slab / tail-slab builders |
| `scripts/run_parz.sh`, `run_ladder.sh` | campaign drivers |
| `scripts/analysis.py` | trajectory, profiles, γ, void detection, unit mapping |
| `scripts/analyze_parz.py`, `analyze_ladder.py` | experiment analyses |
| `scripts/figures.py` | figures |
| `RESULTS.md` | measured numbers and what they do and do not establish |
