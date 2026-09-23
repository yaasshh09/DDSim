# Phase 2: Scharfetter-Gummel + Gummel iteration

**Target: 2 to 3 weeks. Docs: 02-numerics (SG, Gummel), 01-physics (recombination), 05-pitfalls.**

Transport comes in, and this phase produces the first real device
characteristic.

## Scope

1. `physics/bernoulli.py` already exists. Now it gets used.
2. `discretize/continuity.py`, the SG edge flux assembly for both carriers.
   The derivation and exact sign convention are in `docs/02-numerics.md`. Get
   this right before anything else.
3. `physics/recombination.py`, SRH with the Scharfetter doping dependent
   lifetime
4. `solve/gummel.py`, the decoupled cycle: Poisson, then n, then p, repeat
5. `solve/continuation.py`, a **generic** bias ramp driver that takes a solve
   callback. Keep it separate from the device code, because SPICE reuses it
   as is.
6. `extract/iv.py`, I-V sweeps and terminal current

## Acceptance criteria

- **Current continuity**: with recombination off, Jn + Jp is constant across
  all nodes to 1e-6 relative. This is the main gate for the phase. Don't move
  on until it passes.
- Current flows the right way under forward bias, with an explicit test.
- **Shockley diode**: I-V over -1 V to 0.5 V, with an ideality factor near 2
  at low bias trending toward 1 at moderate bias. The crossover has to emerge
  on its own, not be fitted.
- The saturation current within 10 percent of the analytic value from the
  configured lifetimes and diffusion lengths
- Gummel converges at 0.5 V forward bias. Above roughly 0.6 V it's
  **expected** to degrade. Record the bias where it fails. That failure is the
  motivation for Phase 3, not a bug. (It turned out not to fail at all, only
  to slow down without limit. Phase 3 has the numbers.)
- The terminal current sum is zero to 1e-8 relative
- The continuation driver handles a failed step by halving and retrying, and
  logs the event

## Do not

- Add full Newton. That's Phase 3.
- Fight Gummel's slowdown at high injection.
- Add field dependent mobility.

## Definition of done

A committed log scale I-V plot of a real PN diode from your own solver, with
the extracted ideality factor marked on it.
