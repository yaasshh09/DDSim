---
title: The profile plot
summary: Reading psi, n and p along a 1D device, picking the bias point with the slider, and what the 2D image does and doesn't show.
docs: 01-physics.md#Notation
---

## In plain words

The profile plot shows the inside of the device at one solved bias point. It
fills in once a sweep finishes, at the last point, and the `point` slider
then picks any other point the sweep reached. The note tells you which
voltage you're looking at. The slider stays off for a sweep that failed or
got cancelled, because that job has no finished result to read from.

On a 1D device (the diode) the horizontal axis is position along the bar, in
cm. The blue line is psi, the electrostatic potential in V, read on the left
axis. The green line is the electron density n and the red line is the hole
density p, both in cm^-3, sharing a log axis on the right because they span
so many powers of ten. A junction shows up as a step in psi with n and p
crossing over. Tick `bands` and the same plot shows the band diagram instead:
band edges in blue and quasi-Fermi levels in green and red, all in eV.

On a 2D device (the MOS capacitor or the MOSFET) the plot turns into a
picture of psi alone, low potential in blue through to high in yellow, with
the gate at the top and the body at the bottom. It has no axes and no colour
scale, so it shows the shape of the potential, not values. n and p get sent
for a 2D device but aren't drawn as a picture. Drag across the picture to
draw a cutline, and the band diagram along it appears underneath.

## In more depth

The arrays only arrive over HTTP when you ask for a point, as float32 in
physical units with the mesh coordinates alongside. Nothing in the browser
computes a physical quantity from them. The 1D plot uses those coordinates,
so a graded mesh gets drawn at its true positions and the junction isn't
squashed.

The 2D image doesn't. Each node becomes one pixel of a small picture, one
column per mesh column and one row per mesh row, and that picture gets
stretched to fill the canvas. So rows and columns are evenly spaced by index,
not by position. That's just how the picture is built: nothing in the drawing
code looks at the mesh coordinates, so nothing puts a row at its true height.

This matters because the mesh is graded on purpose. On the default MOSFET,
101 rows cover 1 um of silicon and 33 rows cover 20 nm of oxide. The oxide is
about 2 percent of the device's thickness but about a quarter of the
picture's height. The rows packed against the silicon surface, where the
inversion layer lives, get drawn just as tall as the coarse rows deep in the
body. That magnifies exactly the places the mesh was refined for, and it
distorts every distance and slope you might try to read off the picture.
