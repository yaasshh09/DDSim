# Phase 16: 3D

**Target: 4 to 6 weeks. Docs: 02-numerics (Linear algebra, Mesh), 03-architecture (Performance).**

DEVSIM solves in 3D. This is the largest single gap on the scoreboard and the
most expensive one to close, which is why it comes this late. It needs
unstructured meshes from Phase 15. It also needs an iterative linear solver:
a direct LU on a 3D Jacobian fills in far more than on a 2D one, and memory
runs out long before the mesh gets interesting.

## Scope

1. **3D tensor meshes first.** Build them as a 2D mesh extruded along z. This
   reuses `Mesh1D` for the third axis, the same way `mesh2d.py` reuses it for
   two.
2. **Then tetrahedral meshes from Gmsh**, with a Voronoi dual in 3D. The 3D
   Delaunay condition isn't enough on its own to keep every edge coupling
   positive. The quality check has to refuse a negative coupling in 3D just
   as it does in 2D.
3. **Iterative linear solves.** GMRES with an incomplete LU preconditioner
   (`scipy.sparse.linalg.gmres` and `spilu`, both already installed), inside
   an inexact Newton with Eisenstat-Walker forcing. Direct `splu` stays the
   default for 1D and 2D, where it's faster and exact.
4. Assembly cost measured before any of it is optimized. If assembly
   dominates, the fix gets its own decision row. CLAUDE.md asks that the
   Jacobian assembly stay readable, and speed doesn't override that without a
   measured reason.
5. The browser client gets a slice view of 3D fields: a plane chosen in the
   client, cut on the server. The client still does no physics.

## Identities and limits, write these tests first

- **Extrusion is exact.** A 2D benchmark extruded along z with reflecting
  side walls gives the same current per unit width as the 2D solve, to 1e-10
  relative. This is the test the phase rests on.
- **Iterative agrees with direct.** On every 2D benchmark, GMRES-ILU and
  `splu` land on the same converged state within the Newton tolerance.
- **3D capacitor.** A parallel plate with a large area-to-gap ratio gives
  eps A / d within its fringing estimate. A cube in a grounded box converges
  to a fixed value under refinement.
- **Narrow-width effect emerges.** A MOSFET whose width shrinks toward its
  depletion depth shows Vth moving with W. The direction depends on the edge
  geometry, and it's checked as a trend with the direction explained in the
  test.

## Acceptance criteria

- All of the above, shown failing first.
- Benchmark 19: a 3D p+n diode on one Gmsh tetrahedral mesh loaded into both
  tools, within 2 percent on log I. Likely (unverified): DEVSIM's Gmsh import
  handles 3D tetrahedra. Check that against its documentation before building
  the benchmark around it.
- The scoreboard's speed axis gets a 3D row: wall time and peak memory, both
  tools, the same mesh.
- The 3D row on the capability matrix moves to yes.

## Honest limits

Python assembly in 3D will be slow next to C++. The scoreboard reports that
number and doesn't hide it behind the 2D speed results.
