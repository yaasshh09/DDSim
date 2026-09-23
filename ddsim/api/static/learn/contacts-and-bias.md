---
title: Contacts and bias
summary: What a contact voltage means, which contact the sweep takes over, and the boundary conditions an ohmic contact imposes.
docs: 01-physics.md#Ohmic contacts
---

## In plain words

A contact is where a wire meets the device. Each device names its own. The
diode has `anode` and `cathode`, the MOS capacitor has `gate` and `body`, and
the MOSFET has `source`, `drain`, `gate` and `body`. The voltage knobs set
each one in volts, all measured against the same ground.

During a sweep, the contact picked in the sweep's contact box steps through
the voltage list and its own knob gets ignored. Every other contact stays at
its knob's value the whole time. That's how a transfer curve gets its drain
bias: `drain_voltage` holds the drain steady while the gate moves.

There are two kinds of contact. An ohmic contact touches silicon and pins it:
the potential and the electron and hole densities all get fixed to what
neutral silicon at that doping and bias would have. A gate sits on oxide and
fixes only the potential, since no current can flow through an insulator. A
contact can also be a plate covering many nodes, like the body along the
whole bottom edge, and then every node under it sits at the same voltage.
Any boundary that isn't a contact is reflecting, so no current and no field
cross it.

## In more depth

At an ohmic contact node the silicon is taken as neutral and in thermal
equilibrium:

$$p - n + N_d - N_a = 0, \qquad n p = n_i^2$$

That pair fixes $n$ and $p$, and then

$$\psi_{contact} = V_{applied} + V_T\,\mathrm{asinh}\left(\frac{N}{2 n_i}\right)$$

with both quasi-Fermi potentials equal to the applied voltage,
$\phi_n = \phi_p = V_{applied}$. The asinh form matters. The log form
$V_T \ln(N/n_i)$ breaks where the net doping is near zero or negative.

Those are the Boltzmann limit forms. On a device solved with Fermi-Dirac
statistics, which the MOSFET is by default, the contact densities and
potential come from the degenerate relation instead. The interior uses the
same statistics, and docs/07-decisions.md records why they have to match: a
Fermi-Dirac contact on a Boltzmann interior would cram the whole 30.5 mV
correction at 1e20 cm^-3 into a layer one node wide.

A gate is a single Dirichlet condition on $\psi$,
$\psi_{gate} = V_{gate} - \Phi_{MS}$, with no continuity rows at all. Its
reported current is exactly zero, because an ideal insulator passes no DC
current. Contact currents are read from the continuity residual at the
contact nodes, summed over a plate, and they add to zero across all contacts.
