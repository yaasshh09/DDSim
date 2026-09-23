---
title: The electrostatic potential, psi
summary: The voltage at every point inside the device. Everything else gets measured against it.
docs: 01-physics.md#Notation; 01-physics.md#The Van Roosbroeck system; 02-numerics.md#Why this problem is hard; 02-numerics.md#Scaling (de Mari)
---

## In plain words

The voltage inside a semiconductor isn't the same everywhere. It sits higher
where the silicon is doped with donors, lower where it's doped with acceptors,
and it steps from one to the other across a junction. That voltage, point by
point, is psi.

Why care? Electrons drift toward higher psi and holes drift toward lower psi.
So once you know the shape of psi, you know where carriers pile up, where
they get pushed out, and which way current wants to flow. The blue line on
the profile plot is psi across the device you just solved.

Here psi is measured from the intrinsic level. It's zero where the silicon
acts undoped, positive on the n type side and negative on the p type side.

## In more depth

Poisson's equation fixes psi. The curvature of the potential is set by the
net charge:

$$\nabla \cdot (\varepsilon \nabla \psi) = -q\,(p - n + N_d - N_a)$$

Every dopant is taken as ionized. At 300 K and moderate doping that's good to
under one percent.

The electric field is $E = -\nabla\psi$, so a slope in psi is a field and a
step in psi across a junction is the built in potential. The carriers follow
psi. In the Boltzmann limit $n = n_i\,e^{(\psi - \phi_n)/V_T}$ and
$p = n_i\,e^{(\phi_p - \psi)/V_T}$, and in equilibrium both quasi-Fermi
potentials are zero.

The solver never works with psi in volts. It divides by the thermal voltage
$V_T = kT/q$, about 25.9 mV at 300 K, so a one volt step in psi becomes a
step of about 39. That's the de Mari scaling. Densities get divided by a
reference concentration and lengths by a Debye length, so Poisson loses its
constants and becomes $\nabla^2 \psi = -(p - n + N)$. The densities still
span some twenty five decades across one device, which is what makes the
problem stiff. But no row of the Jacobian mixes volts with cm$^{-3}$, and
that keeps the linear solve well enough conditioned to trust.
