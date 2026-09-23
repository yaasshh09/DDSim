---
title: Doping
summary: The impurity atoms that make silicon n type or p type, and what each doping knob sets.
docs: 01-physics.md#Notation; 01-physics.md#Incomplete ionization; 01-physics.md#The Van Roosbroeck system
---

## In plain words

Pure silicon has hardly any free carriers. Doping fixes that by swapping a
tiny fraction of the silicon atoms for impurities. A donor like phosphorus
has one more outer electron than silicon needs, so it gives that electron
away and leaves a fixed positive ion behind. An acceptor like boron has one
too few. It steals an electron from a neighbouring bond, which leaves a
mobile hole and a fixed negative ion. More donors than acceptors makes the
silicon n type, and it conducts mostly with electrons. More acceptors makes
it p type, and it conducts mostly with holes.

Each device here sets its doping its own way. The pn diode has acceptors `Na`
on the left, donors `Nd` on the right, and an abrupt step between them. The
MOS capacitor has one uniform substrate doping with a sign: negative means p
type. The nMOSFET has a p type substrate with an n type source and drain
implanted into the surface. `sd_peak` is their peak concentration. `x_j` is
the depth where the implant has faded to the substrate level. And
`lateral_diffusion` is how far the source and drain creep sideways under the
gate, which makes the real channel shorter than the gate you drew.

## In more depth

Poisson's equation needs the net doping at each point,

$$N = N_d - N_a$$

positive in n type material and negative in p type. That single signed number
is what the solver stores. The charge term in Poisson is
$q\,(p - n + N_d - N_a)$, which assumes every dopant is ionized. That's the
full ionization assumption, and at 300 K and moderate doping it's good to
under one percent. The model it leaves out is

$$N_d^+ = \frac{N_d}{1 + g_d\,e^{(E_F - E_d)/kT}}, \qquad N_a^- = \frac{N_a}{1 + g_a\,e^{(E_a - E_F)/kT}}$$

with $g_d = 2$ and $g_a = 4$. It shows where the assumption breaks down: the
ionized fraction drops once the Fermi level gets near the dopant level, which
happens at low temperature and in heavily doped material. The solver doesn't
include it.

I wrote the nMOSFET implant in closed form so its junctions land exactly
where the knobs say. With depth it's a Gaussian centred on the silicon
surface, and along the channel it's an erfc edge at the gate mask:

$$N_{sd}(x, y) = N_{peak}\cdot\tfrac{1}{2}\,\mathrm{erfc}\!\left(\frac{x - x_{mask}}{\ell}\right)\cdot e^{-(t_{si} - y)^2/(2\sigma^2)}$$

Here $N_{peak}$ is `sd_peak`, $x_{mask}$ is the gate mask edge, and $y$ runs
up from the body contact to the surface at $t_{si}$. The width
$\sigma = x_j / \sqrt{2\ln(N_{peak}/N_a)}$ puts the vertical crossing with the
substrate at depth $x_j$, and $\ell$ comes from `lateral_diffusion` the same
way. The drain mirrors the source, and both sit on top of the uniform
substrate doping.
