# Phase 14: Schottky contacts and avalanche breakdown

**Target: 3 weeks. Docs: 01-physics (Schottky contacts, Impact ionization), 02-numerics (Bias continuation), 06-constants, 05-pitfalls.**

Two things `docs/01-physics.md` lists as out of scope. The Schottky contact
is a new boundary condition and conditions about as well as an ohmic one.
Avalanche is the hardest numerics on the roadmap before 3D, because impact
ionization is positive feedback: near breakdown the I-V goes vertical, and a
voltage-stepped continuation can't cross it.

## Scope

1. **Thermionic emission contacts.** At a Schottky contact, the normal
   electron current is J_n = q v_n (n - n_0B), with
   n_0B = Nc exp(-phi_Bn / V_T) and recombination velocity
   v_n = A_n* T^2 / (q Nc), and the same form for holes. psi is Dirichlet at
   the metal work function. The sign convention is fixed by the tests below,
   not by the formula as typed here.
2. **Image force barrier lowering** as a flag, off by default:
   Delta_phi = sqrt(q E / (4 pi eps_s)), with E the field at the contact.
3. **Impact ionization.** Generation G_ii = (alpha_n |J_n| + alpha_p |J_p|)/q,
   with alpha = a exp(-b / E_par), and E_par the field along the current on
   each edge, the same edge-wise choice Caughey-Thomas already makes. It goes
   in with its full Jacobian, including dG/dpsi through E_par.
4. **Continuation through the vertical.** Current-driven sweeps using the
   contact current source from Phase 11. Arclength continuation in (V, I) is
   the fallback if the current drive isn't enough. Either way, the driver
   lives in `solve/continuation.py`, which SPICE reuses.
5. Parameters go into `docs/06-constants.md` from a cited source before
   anything uses them. That covers A* for electrons and holes, and a and b for
   both carriers. Likely (unverified): van Overstraeten and de Man 1970 is the
   usual silicon source for a and b, with two field ranges for holes. I don't
   know A* for silicon precisely enough to write it here. Look it up in
   Sze's table of effective Richardson constants.

## Analytic limits, write these tests first

- **Thermionic emission limit.** With mobility large enough that diffusion
  isn't limiting, forward J = A* T^2 exp(-phi_B/V_T) (exp(V/V_T) - 1)
  within 2 percent.
- **Thermionic emission diffusion.** Across mobility, J follows
  q Nc v_R / (1 + v_R/v_D) exp(-phi_B/V_T) (exp(V/V_T) - 1), the combined
  Crowell-Sze form. Within 5 percent at both ends and in between.
- **Barrier height from C-V.** The 1/C^2 intercept, taken from the Phase 10
  small-signal solve, gives V_bi = phi_B - V_T ln(Nc/Nd). Within 2 percent.
- **Nothing ionizes at low field.** Multiplication minus one is below 1e-12
  in a forward-biased diode.
- **Breakdown matches the ionization integral.** On a p+n diode, compute the
  electron ionization integral from the solved reverse bias field,
  integral of alpha_n exp(-integral of (alpha_n - alpha_p)) dx, and find
  where it reaches 1. The drift-diffusion breakdown voltage, taken by
  extrapolating 1/M to zero, must agree within 2 percent. Both come from the
  same alpha, so a disagreement is a coupling or Jacobian bug.
- **Breakdown against doping**, as a trend: V_B falls with doping. Likely
  (unverified): the universal Sze-Gibbons fit for a one-sided abrupt silicon
  junction is V_B = 60 (Eg/1.1)^1.5 (N/1e16)^-0.75 V. Within 15 percent,
  since the fit's ionization data aren't the model's.

## Acceptance criteria

- All of the above, shown failing first.
- DEVSIM goldens, with DEVSIM given the same user-written equations:
  benchmark 16 (the Schottky diode I-V) and benchmark 17 (the p+n diode
  breakdown I-V through the knee), within 5 percent.
- `docs/05-pitfalls.md` gets whatever broke along the way. Positive feedback
  terms have a history of hiding sign errors behind damping, and the rule in
  CLAUDE.md stands: check signs before adding damping.

## Stretch

MOSFET substrate current from impact ionization near the drain. That's the
standard hot-carrier indicator, and it emerges without anything new once
the model is in place.
