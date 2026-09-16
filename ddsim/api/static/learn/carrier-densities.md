---
title: Carrier densities, n and p
summary: How many free electrons and holes sit at each point of the device, and why the plot of them is logarithmic.
docs: 01-physics.md#Carrier statistics; 06-constants.md#The n_i problem, read this; 06-constants.md#n_i is not consistent with Nc, Nv and Eg, and that is deliberate
---

## In plain words

Current in silicon is carried by two kinds of particle. Electrons are the free
negative charges, and their density at each point is n. Holes are the gaps an
electron leaves behind in the bonds between atoms, they behave like free
positive charges, and their density is p. Doping decides which one wins: n type
silicon is crowded with electrons and has almost no holes, p type is the
reverse.

The two densities are tied together. Push one up and the other falls in
proportion, so in a region with a hundred million billion electrons per cubic
centimetre there are only about a thousand holes. Across one device the numbers
can easily run over twenty decades, from the heavily doped source of a MOSFET
to the minority carriers next to it. No straight axis can show both ends at
once, which is why the profile plot draws n and p on a logarithmic one. A step
of one grid line there is a factor of ten.

## In more depth

In the Boltzmann limit each density follows the potential and its own
quasi-Fermi potential:

$$n = n_i\,e^{(\psi - \phi_n)/V_T}, \qquad p = n_i\,e^{(\phi_p - \psi)/V_T}$$

with $V_T = kT/q = 25.85$ mV at 300 K. In equilibrium $\phi_n = \phi_p = 0$,
so multiplying the two gives the mass action law

$$n\,p = n_i^2$$

which the test suite checks directly. It is why the minority density falls as
the majority rises: at $N_d = 10^{16}$ cm$^{-3}$, $n \approx 10^{16}$ and
$p \approx 10^{4}$ cm$^{-3}$. Under the Fermi-Dirac statistics the nMOSFET
uses, each density carries a degeneracy factor and the equilibrium product
becomes $n_i^2\,\gamma_n \gamma_p$, which only departs from $n_i^2$ in heavily
doped material.

The intrinsic density is $n_i = 1.0 \times 10^{10}$ cm$^{-3}$ at 300 K, and
that number is anchored rather than derived. Textbooks quote $9.65 \times
10^{9}$, $1.0 \times 10^{10}$ and $1.45 \times 10^{10}$, and $10^{10}$ matches
the worked examples the analytic tests are checked against. It cannot also be
computed from the table values of $N_c$, $N_v$ and $E_g$:

$$\sqrt{N_c N_v}\;e^{-E_g/(2V_T)} = 1.0757 \times 10^{10}\ \text{cm}^{-3}$$

is 7.6 percent higher, 16 percent in $n_i^2$. Both are measured and they cannot
both be primary, so the solver pins $n_i$ at 300 K to $10^{10}$ and takes only
its temperature dependence from the physics.
