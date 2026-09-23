---
title: Depletion
summary: The region around a junction that gets swept almost empty of carriers, and why reverse bias widens it.
docs: 01-physics.md#The Van Roosbroeck system; 01-physics.md#What emerges, and must not be hardcoded; 06-constants.md#Derived, useful for sanity checks
---

## In plain words

Put p type and n type silicon side by side and the carriers near the
boundary don't stay put. Electrons from the n side spill across and fill holes
on the p side. Soon the region around the junction has almost no free
carriers left. What remains are the fixed dopant ions, positive on the n side
and negative on the p side. That charged, carrier free layer is the depletion
region, and its charge is what builds the step in psi across the junction.

On the profile plot it's the stretch where n and p both plunge by many powers
of ten at once. Reverse bias pulls the two sides further apart in voltage.
The only way the junction can hold a bigger step is to uncover more ions, so
the region gets wider. Forward bias does the opposite and narrows it. The
same thing happens under the gate of a MOS capacitor: a positive gate voltage
pushes holes away from the oxide and leaves a depleted layer underneath.

## In more depth

The textbook estimate is the depletion approximation. Assume the region is
completely empty of carriers, with sharp edges, and fully neutral outside.
Integrating Poisson twice across an abrupt junction gives

$$W = \sqrt{\frac{2\varepsilon\,(V_{bi} - V)}{q}\,\frac{N_a + N_d}{N_a N_d}}$$

with $V$ positive for forward bias and, in the Boltzmann limit, the built in
potential

$$V_{bi} = V_T \ln\!\left(\frac{N_a N_d}{n_i^2}\right)$$

For the default diode, $10^{16}$ on both sides, $V_{bi} = 0.7143$ V and
$W \approx 430$ nm at zero bias, split evenly between the two sides. $W$ grows
as the square root of $V_{bi} - V$.

The simulator doesn't use this approximation anywhere. It solves Poisson,

$$\nabla \cdot (\varepsilon \nabla \psi) = -q\,(p - n + N_d - N_a)$$

together with the two continuity equations, with the real carrier densities
in the charge. The depletion edges come out smooth instead of sharp, with
carrier tails reaching a few Debye lengths into the region. Both the built in
potential and the width are outputs of the solve, not inputs. The formula
above is something the tests check the solution against. The solver never
evaluates it.
