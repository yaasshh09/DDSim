---
title: Recombination and generation
summary: Electrons and holes meeting and cancelling out, or being created in pairs, and why that sets a diode's current.
docs: 01-physics.md#Recombination; 01-physics.md#Shockley-Read-Hall; 01-physics.md#Auger; 06-constants.md#SRH lifetimes; 06-constants.md#Auger coefficients
---

## In plain words

An electron that drops into a hole is gone, and so is the hole. That is
recombination. The reverse, a pair created out of thermal energy, is
generation. In equilibrium the two balance exactly and nothing changes. Bias
breaks the balance.

This is what sets the current of a diode. Forward bias pushes electrons into
the p side and holes into the n side, where they are the minority and do not
last: they recombine. Every carrier lost that way has to be replaced from the
contacts, and that steady supply is the forward current. Under reverse bias
the junction holds fewer carriers than equilibrium wants, pairs are generated
in the depletion region, and they are swept out as the small leakage current.

Most recombination in silicon goes through defects, single trap levels in the
gap that catch an electron and then a hole. That process, Shockley-Read-Hall,
is always on in this simulator. The auger switch adds a second process that
only matters when carrier densities are very high: an electron and hole
recombine and hand their energy to a third carrier. It is off by default
because below high injection it changes nothing measurable.

## In more depth

The Shockley-Read-Hall rate for a trap at midgap is

$$R_{SRH} = \frac{n p - n_i^2}{\tau_p\,(n + n_1) + \tau_n\,(p + p_1)}, \qquad n_1 = p_1 = n_i$$

positive for net recombination and negative for net generation. The numerator
vanishes exactly at equilibrium, which is what keeps a device at zero bias from
carrying a current with no cause. Note that $\tau_p$ pairs with $n$: in strongly
n type material the rate reduces to the excess hole density over $\tau_p$, the
minority carrier lifetime.

The lifetimes depend on doping through the Scharfetter relation,

$$\tau = \tau_{min} + \frac{\tau_{max} - \tau_{min}}{1 + (N_{total}/N_{ref})^{\gamma}}$$

with $\tau_{max} = 10^{-5}$ s for electrons and $3 \times 10^{-6}$ s for holes,
$\tau_{min} = 0$, $N_{ref} = 5 \times 10^{16}$ cm$^{-3}$ and $\gamma = 1$. The
relation wants the total doping $N_a + N_d$, but a device only stores the net
doping, so the solver uses $|N_d - N_a|$. The two agree everywhere except in
compensated material.

With auger on, a band to band term is added to SRH:

$$R_{Auger} = (C_n\,n + C_p\,p)\,(n p - n_i^2)$$

with $C_n = 2.8 \times 10^{-31}$ and $C_p = 9.9 \times 10^{-32}$ cm$^6$/s. It
is cubic in the densities where SRH is roughly linear, which is why it only
takes over at high injection and in heavily doped regions.

One approximation to know about: SRH takes its equilibrium product from
$n_i^2$. Under Fermi-Dirac statistics, which the nMOSFET uses, the true
equilibrium product is $n_i^2\,\gamma_n \gamma_p$, smaller than $n_i^2$ in
material doped near $10^{20}$ cm$^{-3}$, so the model generates a small
spurious rate there at rest. It is sized in the decisions log and sits about
six decades below anything this project claims.

Generation from light is not modelled: $G = 0$ everywhere.
