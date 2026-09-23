---
title: Fermi-Dirac statistics
summary: Why the simple exponential law for carrier density breaks in heavily doped silicon, and what the degenerate switch does about it.
docs: 01-physics.md#Carrier statistics; 01-physics.md#Einstein relation; 07-decisions.md#Physics decisions log
---

## In plain words

Electrons follow a strict rule: no two of them can share the same state.
When carriers are scarce the rule barely matters, since empty states vastly
outnumber electrons, and the density follows a simple exponential law called
the Boltzmann approximation. Cram enough carriers into one place, though, and
the lowest states fill up. Each new electron has to sit a little higher in
energy than the last, and the simple law starts to overcount how many
carriers a given Fermi level holds.

In silicon that kicks in once the doping gets close to a few times ten to the
nineteen per cubic centimetre. The nMOSFET's source and drain are doped well
past that. Silicon doped that heavily is called degenerate. The `degenerate`
switch, which only the nMOSFET has, turns on the Fermi-Dirac correction, and
it's on by default for that device. Turn it off and the source and drain
Fermi levels land about 30 mV in the wrong place.

## In more depth

The exact density is a Fermi-Dirac integral with no closed form:

$$n = N_c\,\frac{2}{\sqrt{\pi}}\,F_{1/2}(\eta), \qquad \eta = \frac{E_F - E_c}{kT}$$

and likewise for holes with $N_v$. As $\eta \to -\infty$ it reduces to the
Boltzmann form $n = N_c\,e^{\eta}$. The solver needs the inverse, the Fermi
level for a given density, and takes it from the Joyce-Dixon series in
$u = n/N_c$:

$$\eta \approx \ln u + A_1 u + A_2 u^2 + A_3 u^3 + A_4 u^4$$

$$A_1 = \tfrac{1}{\sqrt 8} = 3.53553 \times 10^{-1}, \quad A_2 = \tfrac{3}{16} - \tfrac{\sqrt 3}{9} = -4.95009 \times 10^{-3}, \quad A_3 = 1.48386 \times 10^{-4}, \quad A_4 = -4.42563 \times 10^{-6}$$

$A_2$ is negative and enters with a plus sign. The series is a fit, and I
don't trust it past $u = 8$.

The solver doesn't swap the Boltzmann relation for this. Written straight
from $N_c$ it would imply an intrinsic density of $1.0757 \times 10^{10}$
cm$^{-3}$ against the anchored $10^{10}$. So it keeps

$$n = n_i\,e^{(\psi - \phi_n)/V_T}\,\gamma_n, \qquad \gamma_n = e^{-(A_1 u + A_2 u^2 + A_3 u^3 + A_4 u^4)}$$

and the mirror image for holes. $\gamma = 1$ exactly at zero density, so
nothing nondegenerate moves. At $10^{20}$ cm$^{-3}$, where $u = 3.5$, the
Fermi level correction is 30.5 mV and Boltzmann overcounts the density by a
factor of 3.26. At $10^{19}$ it's 3.2 mV, and at $10^{18}$ only 0.3 mV.

Degeneracy also changes the ratio of diffusivity to mobility, the generalized
Einstein relation. Differentiating the same series gives it without a second
approximation:

$$\frac{D}{\mu V_T} = u\,\frac{d\eta}{du} = 1 + A_1 u + 2A_2 u^2 + 3A_3 u^3 + 4A_4 u^4$$

That's about 2.1 at $10^{20}$ cm$^{-3}$. Incomplete ionization still isn't
modelled, so every dopant counts as ionized even here.
