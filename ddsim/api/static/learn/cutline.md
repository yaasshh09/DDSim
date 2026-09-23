---
title: The cutline
summary: Dragging a line across a 2D device to see the band diagram along it, and what the browser does and doesn't compute to draw it.
docs: 01-physics.md#Notation
---

## In plain words

A 2D device is hard to read from a picture alone. A band diagram is much
easier, but it's a plot along one line, and on a MOSFET there are lots of
lines worth looking at: straight down through the middle of the channel, say,
or along the surface from source to drain. The cutline lets you pick.

Press on the 2D picture, drag, and let go. The page draws the band diagram
along the line you drew: the conduction and valence band edges in blue, and
the electron and hole quasi-Fermi levels in green and red, all in eV. The
horizontal axis is distance along your line, in cm. A vertical line through
the channel of a switched on MOSFET shows the bands bending down at the
surface. That's the inversion layer. A line along the surface shows the
barrier electrons have to climb from source to drain, and stepping the point
slider along a transfer curve shows the gate pulling that barrier down.

## In more depth

The band edges and quasi-Fermi levels are computed on the server from the
solved state, in the same step that computes psi, n and p, and they arrive
together. The browser computes no energy.

What it does do is geometry. The line you drag is recorded in the picture's
own coordinates, which are fractional mesh indices, since the picture is
drawn one pixel per node. It then gets sampled at 200 evenly spaced steps,
and at each step all four energies are bilinearly interpolated from the four
surrounding nodes. The distance axis comes from the true mesh coordinates of
each sample, so it's a real length even though the picture it was drawn on
is spaced by index.

Two things follow. A cutline can't show anything finer than the mesh, because
between nodes it's a straight blend. And since the samples are even in index
rather than in distance, the plotted points crowd together where the mesh is
fine, which is exactly where the bands change fastest.

An oxide node has no silicon band edges, and the server sends it as NaN. Any
sample that touches one is NaN too, so the plot leaves a gap there instead of
drawing a band through an insulator. On a line drawn down from the gate, the
bands start at the first sample below the silicon surface.
