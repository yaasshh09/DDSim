---
title: Building a 1D device
summary: Doped regions stacked left to right with a contact at each end, how the mesh is graded to every junction, why a stack gets refused, and what a result on a device you built does and does not tell you.
docs: 01-physics.md#Doping range; 02-numerics.md#Mesh
---

## In plain words

Pick `stack` as the device kind and the page gives you a list of regions,
left contact first. Each region is a slab of silicon with a dopant, `p` for
acceptors or `n` for donors, a length in cm and a doping in cm^-3. Add as
many as you like. A pn diode is two regions, a pin diode puts a lightly
doped one between two heavy ones, and an npn is three. The left contact is
called `left` and the right one `right`, and those are the names the sweep
takes.

Wherever the doping changes between two regions there is a junction, and
the mesh puts its finest spacing there, `h_min`, and grows away from it,
because that is where the potential and the carriers change fastest.
`n_nodes` is shared between all the junctions, so a stack with many of them
wants more nodes.

Some stacks are refused before anything is solved, and the refusal says
why. A region shorter than the mesh can put points inside. A doping outside
1e14 to 1e19 cm^-3, the range the models this device is solved with are used
over. A stack where the doping never changes, which has no junction and so
nothing to see. There is no intrinsic dopant: draw the i of a pin as a
1e14 region.

`save device` writes what is on the form to a file, and `load device` reads
one back, so you can hand a device to someone else. The file is the device
part of the request the page sends, so it is exactly what your page solved.

A device you built is labelled as one. The solver is checked against an
independent simulator on its benchmark devices. Your structure has not been
checked by anything, so its curve is the validated solver's answer on a
structure nobody has checked.

## In more depth

Nothing new is solved for a stack. It is the pn diode's recipe with more
junctions: the same Van Roosbroeck system, the same Scharfetter-Gummel
fluxes, ohmic contacts at both ends. The net doping is constant in each
region, $N_d - N_a = \pm N$, and a node sitting exactly on a junction takes
the region to its right.

The mesh aims for a spacing that grows linearly with distance $d$ from the
nearest junction,

$$h(d) = h_{min} + g\,d$$

which is geometric growth by a factor $1 + g$ per cell. A stretch of length
$D$ running away from a junction then wants $\ln(1 + g D / h_{min}) / g$
cells, and $g$ is solved so that the stretches add up to `n_nodes`. Each
stretch is laid geometric from $h_{min}$, and neighbouring cells may never
differ by more than a factor of 1.5, the same limit the diode's mesh keeps.
With one junction this is exactly the pn diode's mesh, which is why the
Phase 2 diode drawn as a stack gives the diode's current to the last bit.

The doping range comes from the statistics. The 1D devices use Boltzmann
statistics, and at 1e19 the Fermi level is already 3.2 mV from the
Fermi-Dirac answer with the Einstein ratio 12 percent off. Higher than that
needs the degenerate path, which the MOSFET has and the stack does not.
