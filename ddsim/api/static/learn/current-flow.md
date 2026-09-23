---
title: Current flow, drift and diffusion
summary: The two ways carriers move, how the continuity equations keep count of them, and why the total current is the same everywhere.
docs: 01-physics.md#The Van Roosbroeck system; 02-numerics.md#Scharfetter-Gummel discretization; 02-numerics.md#Current conservation
---

## In plain words

Carriers move for two reasons. An electric field pushes them, electrons
against the field and holes along it. That's drift. They also spread out from
where they're crowded to where they're sparse, the way a drop of ink spreads
in water, with no field needed. That's diffusion. Every current in the device
is some mix of the two. In a junction at rest both are large and they cancel
exactly, which is why an unbiased diode carries no current.

Carriers also get counted. Whatever flows into a small region either flows
back out or gets lost to recombination there, and electrons and holes each
keep their own tally. Add the electron and hole currents together, though,
and recombination drops out: in steady state the total current going into any
region equals the total coming out. In a diode that means the same current
crosses every slice of the device, carried by holes at one end and electrons
at the other. The current streamlines follow that total and show the path it
takes.

## In more depth

With $E = -\nabla\psi$, the electron and hole current densities are

$$J_n = q\,\mu_n\,n\,E + q\,D_n \nabla n, \qquad J_p = q\,\mu_p\,p\,E - q\,D_p \nabla p$$

The first term of each is drift and the second is diffusion. The signs differ
because diffusion carries each carrier down its own density gradient and the
two carry opposite charge.

The continuity equations keep the count:

$$\frac{\partial n}{\partial t} = \frac{1}{q}\nabla\cdot J_n - R + G, \qquad \frac{\partial p}{\partial t} = -\frac{1}{q}\nabla\cdot J_p - R + G$$

Everything here is steady state, so the time derivatives vanish, and $G = 0$.
That leaves $\nabla\cdot J_n = qR$ and $\nabla\cdot J_p = -qR$. Add them:

$$\nabla\cdot(J_n + J_p) = 0$$

So total current is conserved whatever recombination does. Together with
Poisson's equation these make up the Van Roosbroeck system the solver works
on.

The solver doesn't difference $n$ directly across an edge. That fails when
$n$ changes by decades over one mesh spacing. Instead it assumes the current
and field are constant along each edge and integrates the relation exactly.
That's the Scharfetter-Gummel scheme:

$$J_{n,\,i+1/2} = \frac{q D_n}{h}\left(B(X)\,n_{i+1} - B(-X)\,n_i\right), \qquad X = \frac{\psi_{i+1} - \psi_i}{V_T}$$

with $B$ the Bernoulli function and $h$ the edge length, written here in the
Boltzmann limit. With Fermi-Dirac statistics on, $X$ is taken across each
carrier's own degeneracy corrected potential instead of $\psi$, and the form
doesn't change. Each edge carries one flux shared by the two nodes it joins,
so the discrete current is conserved to machine precision. In a one
dimensional diode with recombination off, $J_n + J_p$ is identical at every
node. That's the strongest single check the project runs.
