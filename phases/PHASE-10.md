# Phase 10: Small-signal AC at any frequency, then noise

**Target: 2 to 3 weeks. Docs: 02-numerics (Small-signal AC), 01-physics (Noise), 04-validation.**

Today `extract/cv.py` does the omega to zero limit only, plus the textbook
approximation for high frequency, where the minority carriers are frozen by
hand. This phase solves the coupled system at a real frequency. The high
frequency C-V then comes out of generation and recombination, not out of an
approximation, which fits the project's rule that effects have to emerge.

## Scope

1. **The mass matrix.** With unknowns (psi, n, p), node interleaved as in
   `discretize/coupled.py`, the time derivative only touches the continuity
   rows, and it's diagonal:

       M[3i+1, 3i+1] = volume_i      (the n row, dn/dt)
       M[3i+2, 3i+2] = volume_i      (the p row, dp/dt)

   in scaled time units t_0 = x_0^2 / D_0. Poisson rows get nothing. The sign
   comes from `F_n = R vol + (dn/dt) vol - div Jn` and
   `F_p = div Jp + R vol + (dp/dt) vol`. The derivation goes into
   `docs/02-numerics.md` before the code, together with a test that checks
   M against a finite difference of a time-stepped residual.
2. **The 1D MOS capacitor on the coupled system.** Today it's Poisson only,
   with Boltzmann densities substituted in. It needs continuity in silicon,
   Poisson only in the oxide, and a substrate contact. The 2D MOSFET already
   has that region split, so this reuses it rather than writing it again.
3. **`extract/ac.py`.** Solve `(J + i omega M) x = b` per frequency using the
   complex LU from SciPy. This gives the full admittance matrix
   Y_ij = dI_i/dV_j for every pair of contacts, and from it G, C, gm, gds,
   Cgg, Cgd, Cgs, and f_T = gm / (2 pi Cgg).
4. **Noise, by the impedance field method.** This means diffusion noise
   sources for electrons and holes, generation-recombination noise from SRH,
   and the Green's function taken from the same adjoint solve (`J^T`, which
   `splu` already gives through `trans='T'`). The output is the spectral
   density of the open circuit voltage noise and the short circuit current
   noise at each contact.
5. The API gains an AC sweep: a frequency list, a C-V at a chosen frequency,
   and Y parameters for the MOSFET. The client draws them. It doesn't compute
   them.

## Analytic limits, write these tests first

- **The DC limit is exact.** At omega = 0, Y must equal the DC conductance
  matrix from a central difference of the DC sweep to 1e-6, and C must equal
  the quasi-static `extract/cv.py` result to 1e-6.
- **High frequency C-V emerges.** At a frequency well above the inverse
  generation time, the MOS capacitor C-V must match the existing frozen
  minority result to 2 percent. At a frequency well below it, it must match
  the quasi-static curve to 1e-4.
- **Long p+n diode** (n side much longer than L_p, low injection):
  Y = G0 sqrt(1 + i omega tau_p) with G0 = I/V_T. Magnitude and phase within
  2 percent over three decades of omega tau_p centered on 1.
- **Short base diode**: the low-frequency diffusion capacitance is
  G0 W^2 / (3 D), which is two thirds of G0 times the transit time
  W^2/(2D). Within 3 percent.
- **Transconductance**: gm from AC at low frequency equals dId/dVg from a
  central difference of two DC solves to 1e-4. These are two independent
  routes to the same number.
- **Nyquist.** A uniformly doped bar with two ohmic contacts at equilibrium
  must give an open circuit voltage noise density of 4 kT Re(Z), within
  1 percent. It's a thermodynamic identity, so a discretization that breaks
  it has a bug.
- **Shot noise in a forward diode.** Likely (unverified) target:
  S_I = 2q(I + 2 I_s) at low frequency. The formula gets checked against its
  source before it's asserted.

## Acceptance criteria

- Everything above passes, shown failing first.
- DEVSIM `solve(type="ac")` golden curves: benchmark 11 (MOS capacitor C
  against frequency) and 12 (the p+n diode admittance against frequency),
  within 3 percent.
- DEVSIM `solve(type="noise")` on the resistor bar: benchmark 13, within
  5 percent.
- A dated row in `docs/07-decisions.md` for the high frequency approximation
  in `cv.py`: it stays as the quick path, and the frequency solve is now the
  reference it gets checked against.

## Why this comes before transient

Phase 11 checks its transient solver by pushing a small sinusoid through it
and comparing the result with Y(omega) from this phase. Doing it in the other
order would leave the transient solver with no independent check.
