# Phase 15: Unstructured 2D meshes and cylindrical coordinates

**Target: 3 to 4 weeks. Docs: 02-numerics (Mesh), 03-architecture (The Device object), 05-pitfalls (obtuse triangles).**

The tensor mesh was the right call for Phase 4, where every device was a
rectangle. It stops being right when:

- a device isn't a rectangle,
- refinement from Phase 12 has to stay local, or
- the same mesh has to go into DDSim and DEVSIM so the comparison is exact.

The assembly already runs over an edge list, and
`tests/unit/test_edge_list_assembly.py` shows the edge order doesn't matter,
so the discretization is ready for this. Only the mesh is new.

## Scope

1. **Triangle meshes with a Voronoi dual.** Box integration on a Delaunay
   triangulation, `scipy.spatial.Delaunay` for generation (already installed),
   and `mesh/quality.py`, which already exists for this day, refusing any mesh
   with a negative edge coupling. A mesh has to be Delaunay and boundary
   conforming: an obtuse triangle on a boundary gives a negative dual area
   even when the interior is Delaunay.
2. **A Gmsh importer**, for the `.msh` 4.1 ASCII format, handling physical
   groups as regions and contacts. It's a small reader written here, not a new
   dependency.
3. **Local refinement** by longest-edge bisection, driven by the Phase 12
   indicator, with a Delaunay flip pass after each round.
4. **Cylindrical coordinates** (r, z) as a mesh option. Dual volumes and edge
   couplings pick up the 2 pi r weight. DEVSIM has this, so it's a parity row.
5. The drawing editor accepts polygons as well as rectangles. The builder
   meshes them, and the client still only draws.

## Analytic limits and identities, write these tests first

- **The tensor mesh is a special case.** Split every rectangle of an existing
  tensor mesh along one diagonal. The diagonal's Voronoi coupling is exactly
  zero, because the two angles opposite it sum to pi, and the dual areas equal
  the tensor ones. So every existing benchmark must come out the same on
  the triangulated mesh, to 1e-12 relative. This is the test the phase rests
  on.
- **Convergence on irregular meshes.** A randomly perturbed Delaunay mesh of
  benchmark 1 converges to the tensor mesh's Richardson value, at the order
  the order studies predict.
- **Coaxial capacitor** in cylindrical coordinates:
  C = 2 pi eps L / ln(r2/r1), within 1e-3 on a refined mesh.
- **Local refinement saves nodes.** Reaching tolerance takes fewer nodes than
  the tensor refinement of Phase 12 on the MOSFET benchmarks. Measured and
  reported either way.
- **A negative coupling is refused.** A deliberately obtuse boundary triangle
  must be refused with a message, not solved.

## Acceptance criteria

- The identities and limits above, shown failing first.
- Benchmark 18: the 2D MOSFET on one Gmsh mesh file, loaded into both DDSim
  and DEVSIM (`create_gmsh_mesh`). Since the mesh is identical, the tolerance
  tightens to 1 percent on log Id.
- A cylindrical p+n junction breakdown, reusing Phase 14, showing breakdown
  at a lower voltage than the planar junction because of curvature. It's
  checked as a trend.
- The scoreboard's Gmsh, unstructured and cylindrical rows move to yes.
