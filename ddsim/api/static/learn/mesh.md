---
title: The mesh
summary: The device chopped into a finite set of points, where those points go, and what the node counts and spacings trade off.
docs: 02-numerics.md#Mesh; 05-pitfalls.md#Specific traps; 07-decisions.md#Physics decisions log
---

## In plain words

A computer can't solve for the potential at every one of the infinitely many
points inside a device. So the device gets chopped into a finite set of
points called nodes, and the equations are written between neighbouring
nodes. That set of points is the mesh. Everything the solver reports lives on
it, which is why the profile plot is really a join the dots of the numbers at
the nodes.

The points aren't spread evenly. Far from a junction the potential and the
carrier densities change slowly, and a few widely spaced nodes describe them
fine. At a junction they change by many powers of ten over a few nanometres.
The nodes have to bunch up there, or the solver smears the depletion region
into something that isn't the device. So the mesh is graded: finest at the
junction and steadily coarser moving away from it.

The mesh knobs set that trade. The node counts (`n_nodes` on the diode, and
the column and row counts on the two 2D devices) decide how many points
there are in total. More points means more accuracy and a slower solve. The
minimum spacings (`h_min` on the diode, `h_min_x` and `h_min_y` on the
MOSFET) decide how fine the mesh gets at the junction or the silicon surface.
With the node count fixed, a smaller minimum spacing leaves fewer nodes for
the rest of the device, so the cells have to grow faster away from it. If
neighbouring cells would differ by more than a factor of 1.5, the device
builder refuses the mesh and tells you why, instead of handing back one whose
sudden jumps would look like physics. The diode defaults are 201 nodes and a
1 nm spacing at the junction.

## In more depth

The equations are discretized by box integration, a finite volume method.
Every node owns a dual cell, the region closer to it than to any neighbour.
In 1D that's the interval between the midpoints of its two edges, with a half
cell at each end. Each equation gets integrated over that cell, and the
divergence turns into a sum of fluxes through the cell's faces. For node $k$
with volume $V_k$, and each edge $e$ from $k$ to a neighbour $j$ with length
$h_e$ and face $f_e$, the scaled Poisson row is

$$\sum_{e} \varepsilon_{r,e}\, f_e\, \frac{\psi_k - \psi_j}{h_e} = (p_k - n_k + N_k)\, V_k$$

and the electron continuity row says the current leaving the cell through its
faces equals what recombines inside it, $\sum_e J_{n,e}\, f_e = R_k V_k$,
with $J_{n,e}$ the Scharfetter-Gummel flux on the edge. Each edge carries one
flux shared by the two cells it separates, so what leaves one cell enters the
next exactly. That's where discrete current conservation comes from. The
permittivity rides on the face, so an edge that crosses from silicon into
oxide carries the right displacement. A carrier flux uses only the
semiconductor share of the face, so nothing flows through an insulator. In 1D
the face is 1. In 2D a simulation is per unit depth, so the face is a length
and the volume is an area.

The 2D mesh is a tensor product of two graded 1D axes. Every x position pairs
with every y position, so the cells are rectangles. A horizontal edge crosses
a vertical face whose length comes from the y axis dual grid, and the other
way round for a vertical edge. The docs sketch a boundary conforming Delaunay
triangulation for 2D. I went with the tensor mesh instead, because every
device here is rectangular, and a rectangle can't produce the obtuse triangle
that gives a negative dual face, a negative conductance, and carrier
densities going negative for no physical reason. A triangle quality check is
already there for the day unstructured meshing arrives, and it refuses an
obtuse mesh.

Each graded axis hits its node count, its length and its minimum spacing
exactly and solves for the growth ratio, which it refuses above 1.5. The
spacing at a junction should resolve the local Debye length, $h < L_D / 2$,
and at $10^{18}$ cm$^{-3}$ the Debye length is 4.09 nm. One trap is specific
to the MOSFET. Its silicon and oxide axes are built separately and then
joined, so the 1.5 guard never sees the Si/SiO2 seam. Halve `h_min_y` on its
own and that seam opens to a ratio of 2, then 4, then 8, with no complaint,
and the drain current moves in a way that looks like convergence. Refine
`n_oxide` along with it.
