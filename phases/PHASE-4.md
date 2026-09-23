# Phase 4: 2D and the MOS capacitor

**Target: 3 weeks. Docs: 02-numerics (mesh), 03-architecture, 01-physics (MOS BC).**

One more dimension. The physics doesn't change; the discretization and the
mesh quality requirements do.

## Scope

1. `mesh/mesh2d.py`, box integration finite volume.
   **Start with a structured tensor product mesh.** A rectangular MOS
   capacitor doesn't need unstructured meshing, and a structured mesh can't
   produce obtuse triangles. Only move to Delaunay when the geometry demands
   it.
2. `mesh/quality.py`, obtuse triangle detection and a dual area positivity
   check. Refuse to build a mesh that fails.
3. Insulator regions: Poisson only, no continuity equations in the oxide
4. The Si/SiO2 interface: continuity of the normal displacement D, not of E,
   with an optional fixed interface charge Q_f
5. The MOS gate boundary condition with the work function difference
6. `extract/cv.py`, small signal AC. Linearize around the DC solution and
   solve the complex system at frequency omega. It reuses the Phase 3
   Jacobian directly and comes to about 60 lines. Don't time step.
7. `device/mos_cap.py`

## Acceptance criteria

- Every 1D test still green. The 2D code path has to reproduce 1D results on
  a single column mesh to 1e-10.
- The mesh quality checker rejects a deliberately obtuse mesh
- No negative carrier densities anywhere in 2D
- **Ideal MOS C-V**, in all three regimes:
  - Accumulation capacitance equals `eps_ox/t_ox` to under 1 percent
  - The depletion minimum matches the maximum depletion width calculation
  - Flatband voltage matches the work function difference to under 20 mV
  - Threshold voltage matches the textbook expression to under 20 mV
- DEVSIM regressions #4 and #5 (both oxide thicknesses) pass at 2 percent
- The current continuity invariant holds in 2D

## Do not

- Build the frontend. Still not time.
- Add adaptive mesh refinement.

## Definition of done

A C-V curve from your own solver overlaid on DEVSIM's, committed to the
README, with the three regimes labelled.
