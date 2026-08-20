# Phase 4: 2D and the MOS capacitor

**Target: 3 weeks. Docs: 02-numerics (mesh), 03-architecture, 01-physics (MOS BC).**

Dimensionality increases. The physics does not change; the discretization and the
mesh quality requirements do.

## Scope

1. `mesh/mesh2d.py`, box-integration finite volume.
   **Start with a structured tensor-product mesh.** A rectangular MOS capacitor
   does not need unstructured meshing and structured meshes cannot produce obtuse
   triangles. Move to Delaunay only when geometry demands it.
2. `mesh/quality.py`, obtuse triangle detection, dual area positivity check.
   Refuse to build a mesh that fails.
3. Insulator regions: Poisson only, no continuity equations in oxide
4. Si/SiO2 interface: continuity of normal displacement D, not of E. Optional
   fixed interface charge Q_f.
5. MOS gate boundary condition with work function difference
6. `extract/cv.py`, small-signal AC. Linearize around the DC solution and solve
   the complex system at frequency omega. Reuses the Phase 3 Jacobian directly.
   Roughly 60 lines. Do not time-step.
7. `device/mos_cap.py`

## Acceptance criteria

- All 1D tests still green. The 2D code path must reproduce 1D results on a
  single-column mesh to 1e-10.
- Mesh quality checker rejects a deliberately obtuse mesh
- No negative carrier densities anywhere in 2D
- **Ideal MOS C-V**, all three regimes:
  - Accumulation capacitance equals `eps_ox/t_ox` to under 1 percent
  - Depletion minimum matches max depletion width calculation
  - Flatband voltage matches work function difference to under 20 mV
  - Threshold voltage matches the textbook expression to under 20 mV
- DEVSIM regression #4 and #5 (both oxide thicknesses) pass at 2 percent
- Current continuity invariant holds in 2D

## Do not

- Build the frontend. Still not time.
- Add adaptive mesh refinement.

## Definition of done

A C-V curve from your own solver overlaid on DEVSIM's, committed to the README,
with the three regimes annotated.
