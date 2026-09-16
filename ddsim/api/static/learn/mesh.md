---
title: The mesh
summary: The device cut into a finite set of points, where the points go, and what the node counts and spacings trade.
docs: 02-numerics.md#Mesh; 05-pitfalls.md#Specific traps; 07-decisions.md#Physics decisions log
---

## In plain words

A computer cannot solve for the potential at every one of the infinitely many
points inside a device, so the device is cut into a finite set of points,
called nodes, and the equations are written between neighbouring nodes. The
mesh is that set of points. Everything the solver reports lives on it, which
is why the profile plot is really a join the dots of numbers at the nodes.

The points are not spread evenly. Far from a junction the potential and the
carrier densities change slowly and a few widely spaced nodes describe them
well. At a junction they change by many decades over a few nanometres, and
the nodes have to bunch up there or the solver smears the depletion region
into something that is not the device. So the mesh is graded: finest at the
junction and growing steadily coarser away from it.

The mesh knobs set that trade. The node counts, `n_nodes` on the diode and the
column and row counts on the two 2D devices, decide how many points there are
in total, and more points means more accuracy and a slower solve. The minimum
spacings, `h_min` on the diode and `h_min_x` and `h_min_y` on the MOSFET, decide
how fine the mesh is at the junction or the silicon surface. With the node
count fixed, a smaller minimum spacing leaves fewer nodes for the rest of the
device, so the cells have to grow faster away from it. If neighbouring cells
would differ by more than a factor of 1.5, the device builder refuses the mesh
and says so, rather than handing back one whose sudden jumps would look like
physics. On the diode the defaults are 201 nodes and a 1 nm spacing at the
junction.

## In more depth

The equations are discretized by box integration, a finite volume method.
Every node owns a dual cell, the region closer to it than to any neighbour: in
1D the interval between the midpoints of its two edges, with a half cell at
each end. Each equation is integrated over that cell, and the divergence
becomes a sum of fluxes through the cell's faces. For node $k$ with volume
$V_k$, and each edge $e$ from $k$ to a neighbour $j$ with length $h_e$ and
face $f_e$, the scaled Poisson row is

$$\sum_{e} \varepsilon_{r,e}\, f_e\, \frac{\psi_k - \psi_j}{h_e} = (p_k - n_k + N_k)\, V_k$$

and the electron continuity row says the current leaving the cell through its
faces equals what recombines inside it, $\sum_e J_{n,e}\, f_e = R_k V_k$, with
$J_{n,e}$ the Scharfetter-Gummel flux on the edge. Each edge carries one flux
shared by the two cells it separates, so what leaves one cell enters the next
exactly, and that is where the discrete current conservation comes from. The
permittivity rides on the face, so an edge that crosses from silicon into
oxide carries the right displacement, while a carrier flux uses only the
semiconductor share of the face and nothing flows through an insulator. In
1D the face is 1. In 2D a simulation is per unit depth, so the face is a
length and the volume an area.

The 2D mesh is a tensor product of two graded 1D axes: every x position is
paired with every y position, so the cells are rectangles. For a horizontal
edge the face it crosses is vertical and its length comes from the y axis
dual grid, and the other way round for a vertical edge. The docs sketch a
boundary conforming Delaunay triangulation for 2D. The project chose the
tensor mesh instead, because every device here is rectangular and a
rectangle cannot produce the obtuse triangle that gives a negative dual face,
a negative conductance, and carrier densities going negative for no physical
reason. A triangle quality check exists for the day unstructured meshing
arrives, and it refuses an obtuse mesh.

Each graded axis honours its node count, its length and its minimum spacing
exactly and solves for the growth ratio, which it refuses above 1.5. The
spacing at a junction should resolve the local Debye length,
$h < L_D / 2$, and at $10^{18}$ cm$^{-3}$ the Debye length is 4.09 nm. One
trap is specific to the MOSFET: its silicon and oxide axes are built
separately and joined, so the 1.5 guard never sees the Si/SiO2 seam. Halving
`h_min_y` alone opens that seam to a ratio of 2, then 4, then 8, with no
complaint, and the drain current moves in a way that reads like convergence.
Refine `n_oxide` along with it.
