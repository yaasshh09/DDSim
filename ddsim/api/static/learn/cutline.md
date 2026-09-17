---
title: The cutline
summary: Dragging a line across a 2D device to see the band diagram along it, and what the browser does and does not compute to draw it.
docs: 01-physics.md#Notation
---

## In plain words

A 2D device is hard to read from an image alone. A band diagram is much easier,
but a band diagram is a plot along a line, and on a MOSFET there are many lines
worth looking at: straight down through the middle of the channel, or along
the surface from source to drain. The cutline lets you choose one.

Press on the 2D image, drag, and let go. The page draws the band diagram along
the line you drew: the conduction band edge and the valence band edge in blue,
and the electron and hole quasi-Fermi levels in green and red, all in eV. The
horizontal axis is distance along your line, in cm. A vertical line through the
channel of an on MOSFET shows the bands bending down at the surface, which is
the inversion layer. A line along the surface shows the barrier electrons
have to cross from source to drain, and stepping the point slider along a
transfer curve shows the gate pulling it down.

## In more depth

The band edges and quasi-Fermi levels are computed on the server from the
solved state, in the same step that computes psi, n and p, and they arrive
with them. The browser computes no energy.

What it does do is geometry. The line you drag is recorded in the image's own
coordinates, which are fractional mesh indices, because the image is drawn one
pixel per node. It is then sampled at 200 evenly spaced steps, and at each step
each of the four energies is bilinearly interpolated from the four surrounding
nodes. The
distance axis is worked out from the true mesh coordinates of each sample, so
it is a real length even though the image it was drawn on is spaced by index.

Two consequences follow. A cutline cannot show structure finer than the mesh,
because between nodes it is a straight blend. And since the samples are even in
index rather than in distance, the plotted points crowd together where the mesh
is fine, which is where the bands change fastest.

An oxide node has no silicon band edges, and the server sends it as NaN. A
sample that touches one is NaN too, and the plot leaves a gap there rather
than drawing a band through an insulator. On a line drawn down from the gate
the bands begin at the first sample below the silicon surface.
