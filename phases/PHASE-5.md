# Phase 5: MOSFET and short-channel effects

**Target: 3 to 4 weeks. Docs: all of 01-physics (mobility, statistics), 04-validation.**

The headline result. Everything before this was infrastructure.

## Scope

1. `device/mosfet.py`, a full 2D NMOS: gate stack, gaussian source and drain
   profiles, channel, substrate contact
2. **Fermi-Dirac statistics** with the Joyce-Dixon inversion, needed in the
   1e20 source and drain. If Boltzmann stays, write down the error it causes
   in `docs/07-decisions.md`. Don't leave it implicit.
3. **Caughey-Thomas field dependent mobility**, using the field component
   along the mesh edge, not the field magnitude. This produces velocity
   saturation.
4. **Lombardi surface mobility.** Without it the inversion layer mobility is
   two to three times too high and Id is off by the same factor. Not optional.
5. Graded meshing that resolves the local Debye length at every junction. At
   1e18, L_D is 4.1 nm.
6. `extract/params.py`: Vth (constant current and linear extrapolation),
   subthreshold slope, DIBL, transconductance
7. A config driven gate length sweep, 1 um down to 50 nm

## Acceptance criteria

All emergent, none fitted:

- **Subthreshold slope at or above 59.5 mV/decade.** Below that is physically
  impossible at 300 K and means a bug. This is the main sanity gate.
- **Vth roll-off**: the threshold voltage falls steadily as Lg shrinks
- **DIBL**: Vth at Vd = 0.05 V is above Vth at Vd = 1.0 V, and the gap widens
  as Lg shrinks. Report DIBL in mV/V.
- **Velocity saturation**: at short Lg, the saturation current scales roughly
  linearly with overdrive instead of quadratically. Fit the exponent across
  the Lg sweep and show it moving from 2 toward 1.
- DEVSIM regressions #6, #7 and #8 pass at their stated tolerances
- The Vth vs Lg curve follows DEVSIM's trend and sits within 10 percent

## Deliverable plot

Vth against Lg from 1 um to 50 nm, this solver against DEVSIM. That one plot
is the argument that the project worked. Nothing in it was fitted. It came out
of Poisson plus two continuity equations plus a doping profile.

## After acceptance criteria pass

The frontend. It's no longer an optional appendix to this phase; it's
`phases/PHASE-7.md`, with its own scope and acceptance criteria. The one thing
worth knowing here is that Phase 7 wants a per iteration callback on
`newton_solve`, so a browser can watch the residual fall live. Don't build it
early, and don't let it change a solved number.

## Honest limits, state these in the README

Below roughly 50 nm, drift-diffusion breaks down. Carriers go quasi-ballistic
and the local field no longer sets the local velocity. Velocity overshoot
can't show up in the model by construction, and quantum confinement in the
inversion layer isn't modelled. Stating these limits accurately is worth more
than stretching the sweep to 20 nm and reporting numbers you can't defend.

## Definition of done

The Lg sweep plot in the README, all nine DEVSIM regressions green in CI, and
a short README section explaining which effects emerged and why.
