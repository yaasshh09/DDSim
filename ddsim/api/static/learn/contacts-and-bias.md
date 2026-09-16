---
title: Contacts and bias
summary: What setting a contact voltage means, which contact the sweep overrides, and the boundary conditions an ohmic contact imposes.
docs: 01-physics.md#Ohmic contacts
---

## In plain words

A contact is where a wire meets the device. Each device names its own: the
diode has `anode` and `cathode`, the MOS capacitor has `gate` and `body`, and
the MOSFET has `source`, `drain`, `gate` and `body`. The voltage knobs set each
one, in volts, all measured against the same ground.

When a sweep runs, the contact named in the sweep's contact box is walked
through the voltage list, and its own knob is ignored. Every other contact
stays at its knob's value for the whole sweep. That is how a transfer curve
gets its drain bias: `drain_voltage` holds the drain while the gate moves.

There are two kinds of contact. An ohmic contact touches silicon and fixes it
there: the potential, and the electron and hole densities, all pinned to what
neutral silicon at that doping and bias would have. A gate sits on oxide and
fixes the potential only, since there are no carriers in metal on an insulator
and no current can flow into it. A contact can also be a plate covering many
nodes, like the body along the whole bottom edge, and then every node it covers
sits at the same voltage. Every boundary that is not a contact is reflecting:
no current and no field cross it.

## In more depth

At an ohmic contact node the silicon is taken as neutral and in thermal
equilibrium:

$$p - n + N_d - N_a = 0, \qquad n p = n_i^2$$

The pair fixes $n$ and $p$, and then

$$\psi_{contact} = V_{applied} + V_T\,\mathrm{asinh}\left(\frac{N}{2 n_i}\right)$$

with both quasi-Fermi potentials equal to the applied voltage,
$\phi_n = \phi_p = V_{applied}$. The asinh form matters: the log form
$V_T \ln(N/n_i)$ breaks where the net doping is near zero or negative.

Those are the Boltzmann limit forms. On a device solved with Fermi-Dirac
statistics, which the MOSFET is by default, the contact densities and
potential come from the degenerate relation instead. The interior uses the
same statistics, and docs/07-decisions.md records why the two have to match: a
Fermi-Dirac contact on a Boltzmann interior would put the whole 30.5 mV
correction at 1e20 cm^-3 into a layer one node wide.

A gate is a single Dirichlet condition on $\psi$,
$\psi_{gate} = V_{gate} - \Phi_{MS}$, with no continuity rows at all. Its
reported current is exactly zero, because an ideal insulator passes no DC
current. Contact currents are read from the continuity residual at the contact
nodes, summed over a plate, and they add to zero across all contacts.
