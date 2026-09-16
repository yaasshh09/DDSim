---
title: The electrostatic potential, psi
summary: The voltage inside the device at every point, and the quantity everything else is measured against.
docs: 01-physics.md#Notation; 01-physics.md#The Van Roosbroeck system; 02-numerics.md#Why this problem is hard; 02-numerics.md#Scaling (de Mari)
---

## In plain words

Inside a semiconductor the voltage is not the same everywhere. Where the
material is doped with donors it sits higher, where it is doped with acceptors
it sits lower, and a junction between the two has a step in it. That voltage,
point by point, is psi. It matters because electrons drift towards higher psi
and holes towards lower psi, so the shape of psi tells you where carriers pile
up, where they are pushed out, and which way current wants to flow. The blue
line on the profile plot is psi across the device you just solved.

In this simulator psi is measured from the intrinsic level: psi is zero where
the material behaves like undoped silicon, positive towards n type and
negative towards p type.

## In more depth

psi is fixed by Poisson's equation, which says the curvature of the potential
is set by the net charge:

$$\nabla \cdot (\varepsilon \nabla \psi) = -q\,(p - n + N_d^+ - N_a^-)$$

The electric field is $E = -\nabla\psi$, so a slope in psi is a field and a
step in psi across a junction is the built in potential. The carriers follow
psi through Boltzmann statistics, $n = n_i\,e^{(\psi - \phi_n)/V_T}$ and
$p = n_i\,e^{(\phi_p - \psi)/V_T}$, and in equilibrium both quasi-Fermi
potentials are zero.

Numerically, the solver never works with psi in volts. It divides by the
thermal voltage $V_T = kT/q$, about 25.9 mV at 300 K, so psi becomes a number
of order ten across a whole device. That is the de Mari scaling. Densities are
divided by a reference concentration and lengths by a Debye length in the same
way, so Poisson loses its constants and becomes
$\nabla^2 \psi = -(p - n + N)$. The densities still span some twenty five
decades across one device, which is what makes the problem stiff, but no row
of the Jacobian mixes volts with cm$^{-3}$, and that is what keeps the linear
solve conditioned well enough to trust.
