# Phase 2: Scharfetter-Gummel + Gummel iteration

**Target: 2 to 3 weeks. Docs: 02-numerics (SG, Gummel), 01-physics (recombination), 05-pitfalls.**

Transport enters. This phase produces the first real device characteristic.

## Scope

1. `physics/bernoulli.py` already exists. Now use it.
2. `discretize/continuity.py`, SG edge flux assembly for both carriers.
   Derivation and exact sign convention in `docs/02-numerics.md`. Get this right
   before anything else.
3. `physics/recombination.py`, SRH with Scharfetter doping-dependent lifetime
4. `solve/gummel.py`, decoupled cycle: Poisson, then n, then p, repeat
5. `solve/continuation.py`, **generic** bias ramp driver taking a solve callback.
   Write it decoupled from device code. It gets reused verbatim in SPICE.
6. `extract/iv.py`, I-V sweeps and terminal current

## Acceptance criteria

- **Current continuity**: with recombination disabled, Jn + Jp is constant across
  all nodes to 1e-6 relative. This is the primary gate for this phase. Do not
  proceed until it passes.
- Current flows the correct direction under forward bias. Explicit test.
- **Shockley diode**: I-V over -1 V to 0.5 V. Ideality factor near 2 at low bias,
  trending toward 1 at moderate bias. The crossover must emerge, not be fitted.
- Saturation current within 10 percent of the analytic value from configured
  lifetimes and diffusion lengths
- Gummel converges at 0.5 V forward bias. Above roughly 0.6 V it is **expected**
  to degrade. Document the bias at which it fails. That failure is the
  motivation for Phase 3, not a bug.
- Terminal current sum is zero to 1e-8 relative
- Continuation driver handles a failed step by halving and retrying, with a
  logged event

## Do not

- Add full Newton. That is Phase 3.
- Fight Gummel's high-injection failure.
- Add field-dependent mobility.

## Definition of done

A committed log-scale I-V plot of a real PN diode from your own solver, with the
extracted ideality factor annotated on it.
