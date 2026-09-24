# Phase 13: Bipolar transistor

**Target: 2 weeks. Docs: 01-physics (What emerges), 04-validation.**

It's the first new device since the MOSFET, and it follows the same rule:
beta, the Early effect, the Kirk effect and high injection roll-off all have
to come out of the physics. None of them may be entered as parameters.

## Scope

1. `device/bjt.py`: a 1D NPN for the analytic checks and a 2D NPN with an
   emitter, base and collector contact, gaussian profiles and a buried
   collector. It's built from the same builder pieces as the MOSFET, so
   there's no second copy of the doping or mesh code.
2. **Sweeps.**
   - The Gummel plot: I_C and I_B against V_BE at V_CB = 0.
   - Output curves: I_C against V_CE at fixed I_B. The base is driven by a
     current, using the contact source from Phase 11.
   - f_T against I_C, from the Phase 10 AC solve.
3. `extract/bjt.py`: beta against I_C, the ideality of I_C and I_B, the
   Early voltage, and peak f_T.
4. Error bars on all of it from Phase 12.
5. The API and a lesson: "why the gain falls at both ends".

## Analytic limits, write these tests first

All of these are in low injection, on the 1D NPN.

- **Gummel's collector current**:
  I_C = q A exp(V_BE/V_T) / integral over the neutral base of
  N_A / (D_n n_i^2) dx, using the code's own D_n and n_i. It's exact when
  there's no recombination in the base. Within 2 percent.
- **Ideality of I_C** is 1.00 +/- 0.01 over the low injection range.
- **Reciprocity**: alpha_F I_ES = alpha_R I_CS from the Ebers-Moll
  parameters, extracted by two separate sweeps. It's a property of linear
  transport, so it holds to 1e-3.
- **Early voltage**: V_A = Q_B / C_BC, with Q_B = q times the integral of N_A
  over the neutral base, and C_BC from the Phase 10 solve. Within 10 percent.
- **Base transit time**: W_B^2 / (2 D_n) against the diffusion capacitance
  from AC, divided by G0. Within 5 percent on a uniform base.

## What must emerge, and is checked only as a trend

- beta falls at low current, where recombination in the emitter-base
  depletion region takes over I_B (ideality near 2).
- beta falls at high current, from high injection in the base and then base
  push-out (the Kirk effect). f_T peaks and then falls against I_C.

If any of these fail to show up, the physics is wrong. Adding a term to
force them in is not allowed.

## Acceptance criteria

- The analytic tests pass, shown failing first.
- DEVSIM golden for the 2D NPN Gummel plot and one output curve:
  benchmark 15, 5 percent on log I_C and log I_B.
- A scoreboard feature row for the BJT. It's parity: DEVSIM can build one if
  you script it.
