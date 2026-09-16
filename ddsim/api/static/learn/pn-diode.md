---
title: The PN diode
summary: A p type region meeting an n type region, the step in potential that forms between them, and what forward and reverse bias do to it.
docs: 01-physics.md#The Van Roosbroeck system
---

## In plain words

Put p type silicon, full of holes, next to n type silicon, full of electrons.
Near the boundary the electrons spill into the p side and the holes into the n
side, and they leave behind ionized dopants that nothing neutralizes. That thin
slab of exposed charge is the depletion region, and the charge sets up a step
in the potential called the built in potential. At equilibrium the step is
exactly big enough to stop any net flow. For the default 1e16 by 1e16 diode it
is about 0.714 V.

Forward bias means making the p side, the `anode`, more positive than the n
side, the `cathode`. That lowers the step, carriers pour over it, and the
current rises roughly tenfold for every 60 mV. Reverse bias raises the step and
widens the depletion region, and only a tiny leakage current flows.

The default diode is 1 um long with the junction in the middle, meshed with
201 nodes that crowd towards the junction, down to 1 nm apart there, because
that is where everything changes fastest. A current on the curve is positive
when it flows from the contact into the device, so a forward biased anode
reads positive, in A/cm^2.

## In more depth

The simulator solves the full Van Roosbroeck system on this bar: Poisson's
equation for $\psi$ and a continuity equation for each carrier, with every
dopant taken as ionized. Nothing about a diode is written into it.

The textbook results are limits the solve is checked against. In the Boltzmann
limit the built in potential is

$$V_{bi} = V_T \ln\left(\frac{N_a N_d}{n_i^2}\right)$$

which with $n_i = 10^{10}$ cm$^{-3}$ gives 0.7143 V at the defaults. The zero
bias depletion width from the depletion approximation,
$W = \sqrt{2\varepsilon (V_{bi} - V)(1/N_a + 1/N_d)/q}$, is about 430 nm
there.

The ideal diode law is

$$J = J_s \left(e^{V/(n V_T)} - 1\right)$$

with an ideality factor $n$ of 1 when diffusion of injected minority carriers
dominates and 2 when recombination inside the depletion region does. A
simulated curve departs from a single exponential wherever both mechanisms
matter at once. With the documented lifetimes the 1e16 diode is diffusion
limited, with an ideality of 1 above about 50 mV. The crossover shows up on a
1e18 device, where the measured peak ideality is 1.79 at 0.16 V rather than 2,
because the depletion region narrows under forward bias and some diffusion
current is still mixed in. A solver reporting exactly 2 would be reporting the
formula rather than the device.

Under reverse bias the ideal law flattens at $-J_s$, while the simulated
current also carries SRH generation in the depletion region, which grows as the
region widens. At high forward bias the injected carriers stop being a small
perturbation and the low injection picture behind the law no longer holds.

The 1D diodes are tier 4 benchmarks against DEVSIM, and they agree to within
0.440 percent.
