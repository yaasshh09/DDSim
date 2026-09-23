---
title: Scaled units
summary: Why the solver divides every quantity by a reference value before it starts, and what those reference values are.
docs: 02-numerics.md#Scaling (de Mari); 02-numerics.md#Why this problem is hard; 06-constants.md#Derived, useful for sanity checks
---

## In plain words

In everyday device units, the numbers inside one device are wildly different
sizes. A voltage is around 1. A length is around a ten thousandth of a
centimetre. A doping can be a hundred million billion per cubic centimetre,
and the minority carrier density in the same device can be far below 1. A
computer keeps about sixteen digits per number. Add something the size of ten
to the minus five to something the size of ten to the twenty and the small
one simply vanishes. Do that inside a big system of equations and you can't
trust the answer anymore.

So before solving anything, the simulator divides each kind of quantity by a
reference value of the same kind. Voltages get divided by the thermal
voltage, densities by the intrinsic carrier density, lengths by a Debye
length, and so on. The equations come out with no physical constants left and
most numbers close to 1, and that's what lets the solver reach the precision
it reports. Everything the page shows you gets turned back into volts, cm and
cm^-3 on the way out. The residual plot is the exception: its numbers are
still scaled and have no units.

## In more depth

This is de Mari scaling. For silicon at 300 K the code sets the reference
quantities to

$$\psi_0 = V_T = kT/q = 25.852\ \text{mV}$$

$$C_0 = n_i = 10^{10}\ \text{cm}^{-3}$$

$$x_0 = L_D = \sqrt{\varepsilon_{Si} V_T / (q\, C_0)} = 40.9\ \mu\text{m}$$

$$D_0 = \max(D_n, D_p) = V_T\,\mu_n = 36.6\ \text{cm}^2/\text{s}$$

$$\mu_0 = D_0 / V_T = 1417\ \text{cm}^2/(\text{V s})$$

$$t_0 = x_0^2 / D_0 = 4.56 \times 10^{-7}\ \text{s}$$

$$J_0 = q\, D_0\, C_0 / x_0 = 1.44 \times 10^{-5}\ \text{A/cm}^2$$

$$R_0 = D_0\, C_0 / x_0^2 = 2.19 \times 10^{16}\ \text{cm}^{-3}\text{s}^{-1}$$

The docs allow $C_0$ to be the largest net doping instead of $n_i$, but every
device in this project uses $n_i$. With these, Poisson in silicon becomes

$$\nabla^2 \psi = -(p - n + N)$$

and steady state continuity becomes $\nabla \cdot J_n = R$ and
$\nabla \cdot J_p = -R$. The Scharfetter-Gummel flux on an edge of scaled
length $h$ loses $V_T$ from its Bernoulli argument, because $\psi$ is already
measured in thermal voltages.

Scaling doesn't remove the stiffness. It just stops the units from making it
worse. One volt is 38.7 in scaled potential, and $e^{38.7}$ is about
$6 \times 10^{16}$, so the densities still span some twenty five decades
across a device. A $10^{16}$ cm$^{-3}$ doping is $10^6$ in scaled units. That
spread is why the convergence tests measure a density change relative to the
density itself, floored at $n_i$, which is 1 in these units. Every array in
the code carries in its type whether it's scaled or physical, because mixing
the two is the most common silent bug in a drift diffusion code.
