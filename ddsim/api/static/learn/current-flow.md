---
title: Current flow, drift and diffusion
summary: The two ways carriers move, how the continuity equations keep count of them, and why the total current is the same everywhere.
docs: 01-physics.md#The Van Roosbroeck system; 02-numerics.md#Scharfetter-Gummel discretization; 02-numerics.md#Current conservation
---

## In plain words

Carriers move for two reasons. An electric field pushes them: electrons
against the field, holes along it. That is drift. And carriers spread out from
where they are crowded to where they are sparse, the way ink spreads in water,
with no field needed at all. That is diffusion. Every current in the device is
some mix of the two, and in a junction at rest they are both large and cancel
exactly, which is why an unbiased diode carries no current.

Carriers are also counted. Whatever flows into a small region either flows out
again or is lost to recombination there, and electrons and holes each have
their own ledger. Add the electron and hole currents together, though, and
the recombination drops out: in steady state the total current entering any
region equals the total leaving it. In a diode that means the same current
crosses every slice of the device, carried by holes at one end and electrons at
the other. Where the page draws current streamlines, they follow that total
current and show the path it takes through the device.

## In more depth

The electron and hole current densities, with $E = -\nabla\psi$, are

$$J_n = q\,\mu_n\,n\,E + q\,D_n \nabla n, \qquad J_p = q\,\mu_p\,p\,E - q\,D_p \nabla p$$

The first term of each is drift and the second diffusion. The signs differ
because diffusion carries each carrier down its own density gradient and the
two carry opposite charge.

The continuity equations keep the count:

$$\frac{\partial n}{\partial t} = \frac{1}{q}\nabla\cdot J_n - R + G, \qquad \frac{\partial p}{\partial t} = -\frac{1}{q}\nabla\cdot J_p - R + G$$

Everything here is steady state, so the time derivatives are zero, and $G = 0$.
That leaves $\nabla\cdot J_n = qR$ and $\nabla\cdot J_p = -qR$, and adding them

$$\nabla\cdot(J_n + J_p) = 0$$

so the total current is conserved whatever the recombination does. With
Poisson's equation these make up the Van Roosbroeck system the solver works
on.

The solver does not difference $n$ directly across an edge, which fails when
$n$ changes by decades over one mesh spacing. It assumes the current and
the field are constant along each edge and integrates the relation exactly, the
Scharfetter-Gummel scheme:

$$J_{n,\,i+1/2} = \frac{q D_n}{h}\left(B(X)\,n_{i+1} - B(-X)\,n_i\right), \qquad X = \frac{\psi_{i+1} - \psi_i}{V_T}$$

with $B$ the Bernoulli function and $h$ the edge length, written here in the
Boltzmann limit. With Fermi-Dirac statistics on, $X$ is taken across each
carrier's own degeneracy corrected potential instead of $\psi$, and the form is unchanged.
Because each edge carries one
flux shared by the two nodes it joins, the discrete current is conserved to machine
precision. In a one dimensional diode with recombination off, $J_n + J_p$ is
identical at every node, and that is the strongest single check the project
runs.
