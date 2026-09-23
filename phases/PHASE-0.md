# Phase 0: Foundation

**Target: 1 week. Docs: 02-numerics (scaling, Bernoulli), 03-architecture, 06-constants.**

No physics yet. This builds the ground everything else stands on. Don't skip
ahead: every later phase assumes these pieces are correct and tested.

## Scope

1. `core/constants.py`, mirroring `docs/06-constants.md`, with T as a
   parameter
2. `core/scaling.py`, the de Mari scale factors, `to_scaled` / `to_physical`
3. `core/field.py`, the Field type with unit, scaling state and mesh location
4. `physics/bernoulli.py`, `B(x)` and `dB/dx`, all branches
5. `solve/linear.py`, a splu wrapper with symbolic factorization caching
   (later dropped, because SciPy can't do it; see Phase 3)
6. `mesh/mesh1d.py`, uniform and graded, with node, edge and cell accessors
7. A pytest scaffold, CI on push, and coverage reporting

## Acceptance criteria

- `B(0) == 1.0` exactly
- `B(-x) == B(x) + x` to 1e-14 relative, swept over [-100, 100] with points on
  both sides of every branch threshold
- No discontinuity bigger than 1e-13 at any branch boundary
- `dB/dx` matches complex step differentiation to 1e-13
- `to_physical(to_scaled(x)) == x` to 1e-14 for every unit type
- Field arithmetic raises on a scaling state mismatch and on a mesh location
  mismatch
- The graded mesh generator gives monotonic spacing, with a chosen minimum
  spacing at a chosen location
- CI green

## Do not

- Write any Poisson or continuity code
- Add 2D meshing
- Add a mobility model beyond a constant stub

## Definition of done

`pytest` is green, the CI badge is live, and you can build a 200 node graded
1D mesh with 1 nm spacing at x = 0.5 um and watch the Field type refuse to add
a scaled potential to a physical one.
