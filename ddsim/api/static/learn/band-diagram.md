---
title: The band diagram
summary: The conduction and valence band edges drawn across the device. They bend wherever psi does.
docs: 01-physics.md#Notation; 01-physics.md#Carrier statistics; 06-constants.md#n_i is not consistent with Nc, Nv and Eg, and that is deliberate
---

## In plain words

A band diagram plots electron energy against position. The top line, Ec, is
the conduction band edge, the lowest energy a free electron can have. The
bottom line, Ev, is the valence band edge, the highest energy a hole can sit
at (holes like to be low on this picture). Between them is the band gap,
about 1.12 eV wide in silicon, where no free carrier can live.

The two edges move together, and they trace psi flipped upside down. Where
the potential rises, the bands fall, because the plot shows an electron's
energy and an electron has negative charge. A pn junction shows up as the
bands stepping down from the p side to the n side. A MOS capacitor in
inversion shows them bending down at the oxide. Electrons roll downhill on Ec
and holes float uphill on Ev. The quasi-Fermi levels drawn alongside tell you
how full each band is: the closer the electron level sits to Ec, the more
electrons there are.

## In more depth

Energies are in eV, with the equilibrium Fermi level at zero. The intrinsic
level is just the potential with its sign flipped,

$$E_i = -q\psi$$

and the band edges sit a fixed distance on either side of it, set by the
effective densities of states and the intrinsic density:

$$E_c = E_i + kT\ln\!\left(\frac{N_c}{n_i}\right), \qquad E_v = E_i - kT\ln\!\left(\frac{N_v}{n_i}\right)$$

These come from writing the Boltzmann relation $n = n_i\,e^{(\psi - \phi_n)/V_T}$
against $n = N_c\,e^{(E_F - E_c)/kT}$. With $N_c = 2.86 \times 10^{19}$ and
$N_v = 3.10 \times 10^{19}$ cm$^{-3}$ at 300 K, $E_c$ is 0.563 eV above $E_i$
and $E_v$ is 0.565 eV below it, so the intrinsic level isn't exactly midgap.
The quasi-Fermi levels go on the same axis as $E_{Fn} = -q\phi_n$ and
$E_{Fp} = -q\phi_p$.

The drawn gap is

$$E_c - E_v = kT\ln\!\left(\frac{N_c N_v}{n_i^2}\right) = 1.1279\ \text{eV}$$

about 3.8 meV wider than the $E_g = 1.1241$ eV the Varshni formula gives at
300 K. That's not a drawing error. $n_i$ is anchored at $10^{10}$ cm$^{-3}$,
while $N_c$, $N_v$ and $E_g$ would imply $1.0757 \times 10^{10}$, and the band
edges have to be placed from one or the other. I place them from the
anchored $n_i$, the same number every density in the solve uses. That way the
edges agree with the carriers drawn next to them, and the gap soaks up the
7.6 percent inconsistency instead.
