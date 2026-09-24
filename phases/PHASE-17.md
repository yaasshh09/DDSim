# Phase 17: Quantum correction

**Target: 3 to 4 weeks. Docs: 01-physics (Where drift-diffusion breaks down), 06-constants.**

At an inverted surface, classical statistics put the electron density peak
right at the oxide. Quantum confinement pushes it about a nanometer into the
silicon. That makes the oxide look thicker, lowers the inversion capacitance
and raises Vth, and the effect grows as the oxide gets thinner. DEVSIM has a
density gradient example. This phase does density gradient too, and then does
something DEVSIM doesn't: it solves Schrodinger-Poisson with the layer 0
solver. That's the first real link in the stack's chain of layers.

## Scope

1. **Schrodinger-Poisson on the 1D MOS capacitor.** Electron density comes from
   the bound states of the potential well at the interface instead of from
   Boltzmann statistics. Poisson and Schrodinger are iterated to self
   consistency, with a predictor-corrector for Poisson so it doesn't
   oscillate. Two valley families, with longitudinal and transverse masses
   from `docs/06-constants.md`. Likely (unverified): m_l = 0.916 and
   m_t = 0.19 electron masses. They go in with a source.
2. **The eigen solver is AtomSIM's.** `atomsim.numerics.radial_solver.solve_radial`
   with l = 0 solves -(1/2 m) u'' + V u = E u with u = 0 at both ends, for
   any potential, in Hartree atomic units, on a uniform grid. That's exactly
   the problem of an electron against a hard oxide wall. The seam is a small
   adapter that:
   - converts eV and cm to Hartree and bohr,
   - interpolates the graded DDSim mesh onto the uniform grid and back, and
   - is tested on its own.

   DDSim never reaches into AtomSIM internals, and AtomSIM stays a dependency
   of this module only.
3. **Density gradient** as the model that works inside the drift-diffusion
   solve for the 2D MOSFET. It adds a quantum potential equation per carrier.
   Its gamma parameter is set by matching Schrodinger-Poisson on the 1D MOS
   capacitor. That's calibration of one layer against another, not fitting to
   a reference tool, and it gets a decision row that says exactly that.
4. The API gains a quantum correction flag, and a lesson: "why thin oxides
   look thicker than they are".

## Analytic limits, write these tests first

- **Infinite triangular well.** With V = qFx and a hard wall at x = 0, the
  levels are E_n = (hbar^2 / 2m)^(1/3) (qF)^(2/3) a_n. Here a_n are the
  magnitudes of the zeros of the Airy function: 2.33811, 4.08795, 5.52056 for
  the first three. Through the adapter, within 1e-4.
- **Infinite square well.** E_n = n^2 pi^2 hbar^2 / (2 m L^2), within 1e-5.
- **The classical limit.** At high temperature or in a wide, shallow well,
  the Schrodinger-Poisson density approaches the Boltzmann density within
  2 percent, away from the wall.
- **The centroid moves.** In strong inversion, the charge centroid sits
  between 0.5 and 2 nm from the interface, against zero for classical
  statistics. This is a trend check with a physical window, not a fitted
  number.
- **The inversion C-V drops.** On a 2 nm oxide the quantum-corrected
  inversion capacitance sits below the classical one, and the gap shrinks as
  the oxide thickens.

## Acceptance criteria

- All of the above, shown failing first.
- Benchmark 20: DEVSIM's density gradient on the 1D MOS capacitor against
  DDSim's density gradient, with the same gamma, within 3 percent on C-V.
- The README's stack section updated with the chain as code. AtomSIM's solver
  produces the levels, and DDSim uses them.

## Honest limits

This is confinement only. Tunnelling through the oxide is still absent, and
`docs/01-physics.md` keeps saying so.
