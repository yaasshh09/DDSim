---
title: Building a 1D device
summary: Doped regions stacked left to right with a contact at each end, how the mesh grades toward every junction, why a stack gets refused, and what a result on a device you built can and can't tell you.
docs: 01-physics.md#Doping range; 02-numerics.md#Mesh
---

## In plain words

Pick `stack` as the device kind and the page gives you a list of regions,
left contact first. Each region is a slab of silicon with a dopant (`p` for
acceptors, `n` for donors), a length in cm and a doping in cm^-3. Add as many
as you like. A pn diode is two regions. A pin diode puts a lightly doped one
between two heavy ones. An npn is three. The left contact is called `left`
and the right one `right`, and those are the names the sweep uses.

Wherever the doping changes between two regions there's a junction. The mesh
puts its finest spacing there, `h_min`, and grows away from it, because
that's where the potential and the carriers change fastest. `n_nodes` gets
shared between all the junctions, so a stack with lots of them wants more
nodes.

Some stacks get refused before anything is solved, and the refusal tells you
why. A region shorter than the mesh can put points inside. A doping outside
1e14 to 1e19 cm^-3, the range this device's models are meant for. A stack
where the doping never changes, which has no junction and so nothing to see.
There's no intrinsic dopant, so draw the i of a pin as a 1e14 region.

`save device` writes what's on the form to a file, and `load device` reads
one back, so you can hand a device to someone else. The file is the device
part of the request the page sends, so it's exactly what your page solved.

A device you built gets labelled as one. The solver is checked against an
independent simulator on its benchmark devices. Your structure hasn't been
checked by anything, so its curve is the validated solver's answer on a
structure nobody has verified.

## In more depth

Nothing new gets solved for a stack. It's the pn diode's recipe with more
junctions: the same Van Roosbroeck system, the same Scharfetter-Gummel
fluxes, ohmic contacts at both ends. The net doping is constant in each
region, $N_d - N_a = \pm N$, and a node sitting exactly on a junction takes
the region to its right.

The mesh aims for a spacing that grows linearly with distance $d$ from the
nearest junction,

$$h(d) = h_{min} + g\,d$$

which is geometric growth by a factor $1 + g$ per cell. A stretch of length
$D$ running away from a junction then wants $\ln(1 + g D / h_{min}) / g$
cells, and $g$ is solved so the stretches add up to `n_nodes`. Each stretch
is laid out geometrically from $h_{min}$, and neighbouring cells can never
differ by more than a factor of 1.5, the same limit the diode's mesh keeps.
With one junction this is exactly the pn diode's mesh, which is why the
Phase 2 diode drawn as a stack gives the diode's current to the last bit.

The doping range comes from the statistics. The 1D devices use Boltzmann
statistics, and at 1e19 the Fermi level is already 3.2 mV off the
Fermi-Dirac answer, with the Einstein ratio 12 percent off. Going higher
needs the degenerate path, which the MOSFET has and the stack doesn't.
