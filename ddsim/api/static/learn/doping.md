---
title: Doping
summary: The impurity atoms that make silicon n type or p type, and what each doping knob sets.
docs: 01-physics.md#Notation; 01-physics.md#Incomplete ionization; 01-physics.md#The Van Roosbroeck system
---

## In plain words

Pure silicon has very few free carriers. Doping fixes that by swapping a tiny
fraction of the silicon atoms for impurities. A donor, such as phosphorus, has
one more outer electron than silicon needs and gives it away, leaving a fixed
positive ion behind. An acceptor, such as boron, has one fewer and takes an
electron from a neighbouring bond, which leaves a mobile hole and a fixed
negative ion. Silicon with more donors than acceptors is n type and conducts
mostly with electrons; more acceptors makes it p type, conducting mostly with
holes.

Each device on the page sets its doping differently. The pn diode has an
acceptor level `Na` on the left, a donor level `Nd` on the right, and an abrupt
step between them. The MOS capacitor has one uniform substrate doping, signed:
negative means p type. The nMOSFET has a p type substrate with an n type source
and drain implanted into its surface. Their peak concentration is `sd_peak`, the
depth where the implant has fallen to the substrate level is `x_j`, and
`lateral_diffusion` is how far the source and drain creep sideways under the
edge of the gate, which makes the real channel shorter than the gate.

## In more depth

What Poisson's equation needs is the net doping at each point,

$$N = N_d - N_a$$

positive in n type material and negative in p type, and that single signed
number is what the solver stores. The charge term in Poisson is
$q\,(p - n + N_d - N_a)$, which assumes every dopant is ionized. That is the
full ionization assumption, and at 300 K and moderate doping it is good to
under one percent. The model it leaves out is

$$N_d^+ = \frac{N_d}{1 + g_d\,e^{(E_F - E_d)/kT}}, \qquad N_a^- = \frac{N_a}{1 + g_a\,e^{(E_a - E_F)/kT}}$$

with $g_d = 2$ and $g_a = 4$. It shows where the assumption weakens: the
ionized fraction falls once the Fermi level approaches the dopant level, which
is what happens at low temperature and in heavily doped material. The solver
does not include it.

The nMOSFET implant is written in closed form so that its junctions land where
the knobs say. Across the depth it is a Gaussian centred on the silicon
surface, and along the channel it is an erfc edge at the gate mask:

$$N_{sd}(x, y) = N_{peak}\cdot\tfrac{1}{2}\,\mathrm{erfc}\!\left(\frac{x - x_{mask}}{\ell}\right)\cdot e^{-(t_{si} - y)^2/(2\sigma^2)}$$

where $N_{peak}$ is `sd_peak`, $x_{mask}$ is the gate mask edge and $y$ runs
up from the body contact to the surface at $t_{si}$. The width
$\sigma = x_j / \sqrt{2\ln(N_{peak}/N_a)}$ puts the vertical crossing
with the substrate at depth $x_j$, and $\ell$ is chosen from
`lateral_diffusion` in the same way. The drain is the mirror image of the source,
and both are added to the uniform substrate doping.
