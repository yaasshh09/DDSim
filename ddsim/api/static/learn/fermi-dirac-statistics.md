---
title: Fermi-Dirac statistics
summary: Why the simple exponential law for carrier density fails in heavily doped silicon, and what the degenerate switch does about it.
docs: 01-physics.md#Carrier statistics; 01-physics.md#Einstein relation; 07-decisions.md#Physics decisions log
---

## In plain words

Electrons obey a rule that no two of them can share the same state. When
carriers are scarce that rule hardly matters, because there are far more empty
states than electrons, and the density follows a simple exponential law, the
Boltzmann approximation. Pack enough carriers into a region and the lowest
states start to fill. Each new electron has to sit higher in energy than the
last, and the simple law begins to overcount how many carriers a given Fermi
level holds.

In silicon that happens once the doping approaches the effective density of
states of the band, a few times ten to the nineteen per cubic centimetre. The
source and drain of the nMOSFET are doped well past that, and silicon doped
that heavily is called degenerate. The `degenerate` switch, which only the nMOSFET
has, turns on the Fermi-Dirac correction. It is on by default for that
device. Turned off, the source and drain Fermi levels come out about 30 mV in
the wrong place.

## In more depth

The exact density is a Fermi-Dirac integral with no closed form:

$$n = N_c\,\frac{2}{\sqrt{\pi}}\,F_{1/2}(\eta), \qquad \eta = \frac{E_F - E_c}{kT}$$

and likewise for holes with $N_v$. For $\eta \to -\infty$ it reduces to the
Boltzmann form $n = N_c\,e^{\eta}$. The solver needs the inverse, the Fermi
level for a given density, and takes it from the Joyce-Dixon series in
$u = n/N_c$:

$$\eta \approx \ln u + A_1 u + A_2 u^2 + A_3 u^3 + A_4 u^4$$

$$A_1 = \tfrac{1}{\sqrt 8} = 3.53553 \times 10^{-1}, \quad A_2 = \tfrac{3}{16} - \tfrac{\sqrt 3}{9} = -4.95009 \times 10^{-3}, \quad A_3 = 1.48386 \times 10^{-4}, \quad A_4 = -4.42563 \times 10^{-6}$$

$A_2$ is negative and enters with a plus sign. The series is a fit and is not
trusted past $u = 8$.

The solver does not replace the Boltzmann relation with this. Written
directly from $N_c$ it would imply an intrinsic density of
$1.0757 \times 10^{10}$ cm$^{-3}$ against the anchored $10^{10}$. Instead it
keeps

$$n = n_i\,e^{(\psi - \phi_n)/V_T}\,\gamma_n, \qquad \gamma_n = e^{-(A_1 u + A_2 u^2 + A_3 u^3 + A_4 u^4)}$$

and the mirror image for holes. $\gamma = 1$ exactly at zero density, so
nothing nondegenerate moves. At $10^{20}$ cm$^{-3}$, where $u = 3.5$, the
Fermi level correction is 30.5 mV and Boltzmann overcounts the density by a
factor of 3.26; at $10^{19}$ it is 3.2 mV, and at $10^{18}$ only 0.3 mV.

Degeneracy also changes the ratio of diffusivity to mobility, the generalized
Einstein relation. Differentiating the same series gives it without a second
approximation:

$$\frac{D}{\mu V_T} = u\,\frac{d\eta}{du} = 1 + A_1 u + 2A_2 u^2 + 3A_3 u^3 + 4A_4 u^4$$

which is about 2.1 at $10^{20}$ cm$^{-3}$. Incomplete ionization is still
not modelled, so every dopant counts as ionized even here.
