---
title: The profile plot
summary: Reading psi, n and p along a 1D device, choosing the bias point with the slider, and what the 2D image does and does not show.
docs: 01-physics.md#Notation
---

## In plain words

The profile plot shows the inside of the device at one solved bias point. It
fills in once a sweep finishes, at the last point, and the `point` slider then
picks any other point the sweep reached. The note says which voltage you are
looking at. The slider stays off for a sweep that failed or was cancelled,
because that job has no finished result to read the states from.

On a 1D device, the diode, the horizontal axis is position along the bar, in
cm. The blue line is psi, the electrostatic potential in V, read on the left
axis. The green line is the electron density n and the red line the hole
density p, both in cm^-3 and sharing a log axis on the right, because they span
many decades. A junction shows up as a step in psi, with n and p crossing over.

On a 2D device, the MOS capacitor or the MOSFET, the plot becomes an image of
psi alone, low potential in blue through to high potential in yellow, with the
gate at the top and the body at the bottom. It has no axes and no colour scale,
so it shows the shape of the potential rather than values. n and p are sent
for a 2D device but not drawn.

## In more depth

The arrays arrive over HTTP only when a point is asked for, as float32 in
physical units with the mesh coordinates alongside them, and nothing in the
browser computes a physical quantity from them. The 1D plot uses those
coordinates, so a graded mesh is drawn at true positions and the junction is
not squashed.

The 2D image does not. Each node becomes one pixel of a small picture, one
column per mesh column and one row per mesh row, and that picture is stretched
to fill the canvas. So rows, and columns too, are evenly spaced by index and
not by position. That is simply how the picture is built: nothing in the
drawing code looks at the mesh coordinates, so nothing places a row at its true
height.

The consequence is worth knowing, because the mesh is graded on purpose. On the
default MOSFET, 101 rows cover 1 um of silicon and 33 rows cover 20 nm of
oxide, so the oxide is about 2 percent of the device's thickness and about a
quarter of the image's height. The rows packed against the silicon surface,
where the inversion layer lives, are drawn just as tall as the coarse rows deep
in the body. That magnifies exactly the places the mesh was refined for, and
it distorts every distance and slope you might try to read off the picture.
