---
title: Surface scattering
summary: Carriers pressed against the oxide scatter off the interface and move more slowly than they would in the bulk.
docs: 01-physics.md#Mobility models; 06-constants.md#Lombardi surface mobility, enhanced form
---

## In plain words

The channel of a MOSFET is a very thin sheet of electrons held against the
silicon and oxide interface by the gate. That interface is not a perfect
mirror. It is rough on the scale of atoms, and the crystal vibrations there
are different from those deep in the silicon, so carriers in the channel
collide more often than the same carriers would in the bulk. The harder the
gate pulls them against the interface, the more they feel it.

The `surface` switch adds this effect. Leaving it out makes the channel
mobility two to three times too high, and the drain current too high by the
same factor. It needs a two dimensional device with an oxide on it, since it
reads the field pushing carriers towards the interface, and it is off by
default.

## In more depth

The model is the enhanced Lombardi form, combined with the bulk mobility from
the mobility knob by Matthiessen's rule, which adds scattering rates:

$$\frac{1}{\mu} = \frac{1}{\mu_{bulk}} + \frac{1}{\mu_{ac}} + \frac{1}{\mu_{sr}}$$

The acoustic phonon term and the surface roughness term both depend on
$E_\perp$, the magnitude of the field normal to the interface in V/cm:

$$\mu_{ac} = \frac{B}{E_\perp} + \frac{C_{ac}\,N^{\tau}}{E_\perp^{1/3}\,(T/300)^{\kappa}}$$

$$\mu_{sr} = \delta\,E_\perp^{-\gamma}, \qquad \gamma = A + \alpha\,(n + p)\,N^{-\eta}$$

with $N$ the doping magnitude and $n + p$ the local carrier density. The 1988
Lombardi model fixes $\gamma = 2$. The enhanced form lets it grow with carrier
density, so a denser inversion layer sees a rougher interface. The parameters
are the ones DEVSIM ships, so the MOSFET regressions compare the same model:

| | $B$ | $C_{ac}$ | $\tau$ | $\delta$ | $A$ | $\alpha$ | $\eta$ | $\kappa$ |
|---|---|---|---|---|---|---|---|---|
| electrons | $3.61 \times 10^{7}$ | $1.70 \times 10^{4}$ | 0.0233 | $3.58 \times 10^{18}$ | 2.58 | $6.85 \times 10^{-21}$ | 0.0767 | 1.7 |
| holes | $1.51 \times 10^{7}$ | $4.18 \times 10^{3}$ | 0.0119 | $4.10 \times 10^{15}$ | 2.18 | $7.82 \times 10^{-21}$ | 0.123 | 0.9 |

Both surface terms grow without bound as $E_\perp$ falls, so wherever the
normal field is weak their reciprocals vanish and the bulk mobility comes back
untouched.
That is why no surface layer thickness has to be chosen. Two floors keep the
arithmetic finite: $E_\perp$ is held at or above $10^2$ V/cm, as DEVSIM holds
it, and the doping the model reads is held at or above $n_i$. The correction is
applied at semiconductor nodes only, since an insulator has no mobility to
correct. When velocity saturation is also on, this correction comes first, at
the nodes, and Caughey-Thomas is applied to the result on the edges.

Unlike Caughey-Thomas, this correction is not inside the Newton Jacobian. The
normal field on a channel edge lives on the vertical edges beside it, which the
edge based assembly cannot reach, so the surface mobility is frozen during each
Newton solve and refreshed between solves until it stops moving. At that fixed
point the frozen mobility is the one the answer implies, so the converged state
solves the full equations; what the freezing costs is convergence rate, not
accuracy.
