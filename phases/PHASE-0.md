# Phase 0: Foundation

**Target: 1 week. Docs: 02-numerics (scaling, Bernoulli), 03-architecture, 06-constants.**

No physics yet. Build the substrate that everything else stands on. Do not skip
ahead; every later phase assumes these are correct and tested.

## Scope

1. `core/constants.py` mirroring `docs/06-constants.md`, T as a parameter
2. `core/scaling.py`, de Mari scale factors, `to_scaled` / `to_physical`
3. `core/field.py`, the Field type with unit, scaling state, mesh location
4. `physics/bernoulli.py`, `B(x)` and `dB/dx`, all branches
5. `solve/linear.py`, splu wrapper with symbolic factorization caching
6. `mesh/mesh1d.py`, uniform and graded, node/edge/cell accessors
7. pytest scaffold, CI on push, coverage reporting

## Acceptance criteria

- `B(0) == 1.0` exactly
- `B(-x) == B(x) + x` to 1e-14 relative, swept over [-100, 100] with points
  straddling every branch threshold
- No discontinuity larger than 1e-13 at any branch boundary
- `dB/dx` matches complex-step differentiation to 1e-13
- `to_physical(to_scaled(x)) == x` to 1e-14 for every unit type
- Field arithmetic raises on scaling-state mismatch, and on mesh-location mismatch
- Graded mesh generator produces monotonic spacing with a specified minimum
  spacing at a specified location
- CI green

## Do not

- Write any Poisson or continuity code
- Add 2D meshing
- Add a mobility model beyond a constant stub

## Definition of done

`pytest` green, CI badge live, and you can construct a 200 node graded 1D mesh
with 1 nm spacing at x = 0.5 um and confirm the Field type refuses to add a
scaled potential to a physical one.
