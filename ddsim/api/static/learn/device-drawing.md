---
title: Drawing a 2D device
summary: Rectangles of silicon and oxide, rectangles of doping and electrodes along straight lines, how the mesh follows them, why a drawing gets refused, and what a result on a device you drew does and does not tell you.
docs: 01-physics.md#Doping range; 02-numerics.md#Mesh
---

## In plain words

Pick `drawing` as the device kind. It starts as the benchmark MOSFET, drawn,
and everything on it can be changed. x runs across the device and y up it, in
cm, with the gate side on top.

A drawing is three lists. **Blocks** are rectangles of silicon or oxide. A
later block is painted over an earlier one, so a trench is an oxide block
drawn over the silicon. Together they must cover a rectangle whose lower left
corner is at x = 0, y = 0, with no gaps: an undrawn gap would be vacuum, and
this solver has no material for it. **Implants** are rectangles of doping,
`p` or `n`, and where two overlap they add up. A `uniform` implant is its
peak inside the rectangle and nothing outside. A `gaussian` one is the shape
of a real implant: the peak holds inside, falls off below and above it with a
spread of `straggle`, and creeps past its left and right edges over
`lateral`, reaching half the peak at the edge. **Electrodes** are straight
lines, across or up. An `ohmic` one sits on silicon; a `gate` sits on oxide
and has a work function. Its name is what the sweep's contact box takes.

You can type the numbers into the rows, or drag on the picture to add a
rectangle of whatever the box next to it says. A drag that ends near an edge
already drawn snaps onto it. The picture stretches the two axes separately so
a thin oxide stays visible; the numbers are the drawing, not the picture.

Only rectangles. The mesh is a grid of lines that each run the whole width or
height of the device, so a slanted or curved edge would cut cells in two, and
a cell that is part oxide and part silicon has no single permittivity.

Some drawings are refused before anything is solved, and the refusal names
the part and the reason. A silicon island no ohmic contact touches floats:
nothing sets its Fermi level. A gate touching silicon is a Schottky contact,
which this solver does not model. An ohmic contact on oxide has no carriers
to hold. Two edges closer than `h_min`, or a rectangle with fewer than two
mesh lines inside it, is smaller than the mesh can resolve. A mesh over the
node budget the page states is refused, because a 2D solve slows faster than
its node count grows. A doping outside 1e14 to 1e19 cm^-3, or up to 1e20
with `degenerate` on, is outside the range the models are used over.

Surface scattering, the `surface` model, reads the field across a flat
Si/SiO2 interface. A drawing with an oxide wall standing beside silicon has a
second direction, and that model refuses it rather than read the wrong field.

A result on a device you drew is the validated solver's answer on a structure
nothing has checked, and the page says so under the curve.

## In more depth

Every edge you draw becomes a mesh line in its axis, and the spacing is
graded towards every doping edge and every Si/SiO2 interface, where the
potential and the carriers change fastest, to `h_min_x` across and `h_min_y`
up. One growth rate serves the whole axis, the rule the 1D builder uses, so
two nearby edges meet at matching cell sizes. The spacing aimed for is

$$h(x) = h_{min} + g\,d(x)$$

with $d$ the distance to the nearest graded edge and $g$ solved so the axis
holds `nx` (or `ny`) nodes. Each span between two drawn lines takes a whole
number of cells, so the spacing at an edge is near $h_{min}$ rather than
exact. An implant edge flush with the device boundary is not graded towards,
because the implant carries on past the drawing.

A Gaussian implant is separable. Across, it is the window blurred by a
Gaussian, $\tfrac{1}{2}[\mathrm{erfc}((x - x_1)/L) - \mathrm{erfc}((x - x_0)/L)]$
with $L$ = `lateral`; up, it is the peak inside and
$\exp(-d^2/2\sigma^2)$ at a distance $d$ outside, with $\sigma$ = `straggle`.
That is the profile the benchmark nmos is built from, and drawn as that
device the doping agrees with it to the last bit on the source side.

The benchmark MOS capacitor and MOSFET drawn this way reproduce the
constructors: the C-V to 2e-4 and the drain current to 1.5 percent above
1e-7 A/cm, which is the size of the change each device shows when its own
mesh is refined. The difference is the mesh, not the physics.
