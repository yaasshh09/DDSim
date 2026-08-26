# Phase 5: MOSFET and short-channel effects

**Target: 3 to 4 weeks. Docs: all of 01-physics (mobility, statistics), 04-validation.**

The headline result. Everything before this was infrastructure.

## Scope

1. `device/mosfet.py`, full 2D NMOS: gate stack, source/drain gaussian profiles,
   channel, substrate contact
2. **Fermi-Dirac statistics** with Joyce-Dixon inversion, required in the 1e20
   source/drain regions. If you keep Boltzmann, document the resulting error
   explicitly in `PROGRESS.md`. Do not leave it implicit.
3. **Caughey-Thomas field-dependent mobility**, using the field component
   parallel to the mesh edge, not the field magnitude. Produces velocity
   saturation.
4. **Lombardi surface mobility.** Without it, inversion-layer mobility is too
   high by 2 to 3x and Id is correspondingly wrong. Not optional.
5. Graded meshing: resolve the local Debye length at every junction. At 1e18,
   L_D is 4.1 nm.
6. `extract/params.py`: Vth (constant-current and linear extrapolation methods),
   subthreshold slope, DIBL, transconductance
7. Config-driven gate length sweep, 1 um down to 50 nm

## Acceptance criteria

Emergent, none fitted:

- **Subthreshold slope at or above 59.5 mV/decade.** Below that is physically
  impossible at 300 K and means a bug. Primary sanity gate.
- **Vth roll-off**: threshold voltage decreases monotonically as Lg shrinks
- **DIBL**: Vth extracted at Vd = 0.05 V exceeds Vth at Vd = 1.0 V, and the gap
  widens as Lg shrinks. Report DIBL in mV/V.
- **Velocity saturation**: at short Lg, saturation current scales roughly
  linearly with overdrive rather than quadratically. Fit the exponent across the
  Lg sweep and show it moving from 2 toward 1.
- DEVSIM regression #6, #7, #8 pass at stated tolerances
- Vth vs Lg curve matches DEVSIM's trend and is within 10 percent

## Deliverable plot

Vth versus Lg from 1 um to 50 nm, your solver against DEVSIM. This single plot is
the argument that the project worked. Nothing in it was fitted; it came out of
Poisson plus two continuity equations plus a doping profile.

## After acceptance criteria pass

The frontend. It is no longer an optional appendix to this phase; it is
`phases/PHASE-7.md`, with its own scope and its own acceptance criteria. The one
thing worth knowing here is that Phase 7 wants a per-iteration callback on
`newton_solve` so a browser can watch the residual fall live. Do not build it
early and do not let it change a solved number.

## Honest limits, state these in the README

Below roughly 50 nm, drift-diffusion breaks down. Carriers become quasi-ballistic
and local field no longer determines local velocity. Velocity overshoot is
invisible to the model by construction. Quantum confinement in the inversion
layer is not modeled. Stating these limits accurately is worth more than
extending the sweep to 20 nm and reporting numbers you cannot defend.

## Definition of done

The Lg sweep plot in the README, all nine DEVSIM regressions green in CI, and a
short section in the README explaining which effects emerged and why.
