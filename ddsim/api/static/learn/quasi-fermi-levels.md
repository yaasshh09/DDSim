---
title: Quasi-Fermi levels, phi_n and phi_p
summary: One Fermi level per carrier once the device is biased, and the slope that drives current.
docs: 01-physics.md#Carrier statistics; 01-physics.md#The Van Roosbroeck system; 01-physics.md#Ohmic contacts
---

## In plain words

In a device left alone, electrons and holes share a single Fermi level, a flat
line that says how full the available states are. Nothing flows, because there
is no slope for anything to roll down. Apply a bias and that single level is no
longer enough to describe what is happening, so each carrier gets its own:
phi_n for electrons and phi_p for holes.

Two rules make them easy to read. Where a quasi-Fermi level is flat, that
carrier carries no current, however many of them there are. Where the two
levels split apart, the device is out of equilibrium: a forward biased junction
has more electrons and holes than equilibrium allows and they recombine, a
reverse biased one has fewer and it generates them. At an ohmic contact both
levels are pinned to the voltage applied to that contact.

## In more depth

In the Boltzmann limit the quasi-Fermi potentials are defined through the
densities:

$$n = n_i\,e^{(\psi - \phi_n)/V_T}, \qquad p = n_i\,e^{(\phi_p - \psi)/V_T}$$

so $\phi_n = \psi - V_T \ln(n/n_i)$ and $\phi_p = \psi + V_T \ln(p/n_i)$. In
equilibrium $\phi_n = \phi_p = 0$ everywhere, and at an ohmic contact
$\phi_n = \phi_p = V_{applied}$. Multiplying the two densities gives

$$n\,p = n_i^2\,e^{(\phi_p - \phi_n)/V_T}$$

so the split between the levels measures how far the product is from
equilibrium, and it is exactly what the SRH rate responds to.

Putting the Boltzmann relations and $D_n = \mu_n V_T$ into the drift diffusion
currents, the drift and diffusion terms combine into one:

$$J_n = -q\,\mu_n\, n\,\nabla\phi_n, \qquad J_p = -q\,\mu_p\, p\,\nabla\phi_p$$

Current is a density times the gradient of that carrier's quasi-Fermi
potential. A flat $\phi_n$ means zero $J_n$, and a large current through a
region with few carriers needs a steep slope, which is why the levels drop
sharply where the density is low.

With Fermi-Dirac statistics switched on, the solver keeps the same form by
writing each density as the Boltzmann expression times a degeneracy factor.
The quasi-Fermi potential then has to be read under the same statistics the
state was solved with: reading a degenerate state with the Boltzmann formula
misplaces it by 30.5 mV at $10^{20}$ cm$^{-3}$.
