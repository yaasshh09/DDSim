---
title: The bipolar transistor
summary: An npn where a small base current lets a much bigger collector current through, and where the gain, the Early voltage and the speed all come out of the same drift diffusion as everything else.
docs: decisions.md#Physics decisions log
---

## In plain words

An npn is two pn junctions back to back, sharing a thin p layer in the middle,
the base. Forward bias the emitter junction and the emitter pours electrons
into the base. The base is so thin that almost all of them diffuse across
before they recombine, and the reverse biased collector junction sweeps them
out the other side. That's the collector current. The base only has to supply
the few holes that go back into the emitter or meet an electron on the way,
so a small base current controls a big collector current. Their ratio is the
gain, beta.

There are two devices. `npn` is one dimensional: emitter, base and collector
in a row, with the base contact in the middle of the base. That contact only
holds the holes, so electrons pass straight through it, the way they pass
under a real base contact that sits off to the side. `npn_2d` is the real
shape: an emitter diffused into the top on the left, a base implant across
the whole top, the base contact on the right of the surface, and the
collector underneath on top of a heavily doped buried layer.

To get a Gummel plot, run a `transfer` sweep with the base as the contact.
The collector current is what gets plotted, and the collector voltage knob
sets V_CE. Put it at 1 V or so first. At its default of 0 the collector
junction is forward biased as hard as the emitter one, the transistor is in
saturation, and the collector current comes out negative. For output curves, run an `iv` sweep on the collector with the base
voltage held.

## In more depth

Nothing about the gain is entered anywhere. Beta comes from the ratio of the
emitter's and the base's Gummel numbers, the integrals of doping over each
neutral region divided by the diffusivity. On the 1D device with no
recombination, the collector current matches Gummel's formula
$I_C = q n_i^2 e^{V_{BE}/V_T} / \int p / D_n \, dx$ over the base to 2.6e-4.

Beta falls at both ends of the current range. At low current,
Shockley-Read-Hall recombination in the emitter junction's depletion region
adds a base current that rises only as $e^{V_{BE}/2V_T}$, so it wins at low
bias. At high current the electrons in the base outnumber its doping, which is
high injection, and the collector current stops rising as fast as the base
current. The same high current is what makes f_T peak and then fall.

The Early effect is the collector junction's depletion eating into the base as
V_CB rises. The base gets thinner and the collector current climbs. Its size,
$V_A = Q_B / C_{BC}$, the base charge over the collector junction capacitance,
is checked against a separate AC solve to within 10 percent.

On the 2D device the gain is in the thousands, not the hundreds. That's
because I use Boltzmann statistics with no bandgap narrowing, so a $10^{20}$
emitter barely takes any holes. Bandgap narrowing would bring it down, and
it isn't in the model yet.
