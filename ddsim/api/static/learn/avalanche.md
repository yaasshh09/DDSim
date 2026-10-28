---
title: Avalanche breakdown
summary: Why a reverse biased junction suddenly lets current through, and how the solver gets there without being told where the edge is.
docs: decisions.md#Physics decisions log
---

## In plain words

Reverse bias a pn junction and almost nothing flows, just the few carriers
the silicon makes on its own. Keep pushing and the field in the junction gets
so strong that a carrier crossing it picks up enough energy to knock an
electron out of a bond when it hits something. Now there are two carriers,
and both of them get sped up and knock out more. That chain reaction is the
avalanche. Past a certain voltage it feeds itself and the current shoots up
by thousands of times over a few tens of millivolts.

The `impact` knob turns this on. Nothing tells the solver where breakdown
is. It only knows how often a carrier ionizes at each field strength, and the
breakdown voltage falls out of that. Heavier doping squeezes the junction,
raises the field, and brings breakdown down.

Try a pn diode with Na at 1e19 and Nd at 1e17, and sweep the anode from 0 to
about -13 V. The current starts to climb steeply near the end.

## In more depth

Each carrier ionizes at a rate $\alpha = a e^{-b/E}$ per centimetre of travel,
with van Overstraeten and de Man's $a$ and $b$ for silicon, and holes on two
field ranges. On every edge the generation is
$G = \alpha_n |J_n| + \alpha_p |J_p|$, divided by $q$, using that edge's own
field and currents, and half of it goes to each end. Newton gets its full
derivative, including how $\alpha$ moves with the potential through $E$.

Breakdown is where the ionization integral
$\int \alpha_n e^{-\int (\alpha_n - \alpha_p)} dx$ reaches 1. I check the
solver against that: the multiplication from the full solve, extrapolated to
where $1/M$ hits zero, lands within 0.8 percent of the voltage where the
integral reaches 1, on the same field. Against DEVSIM given the same
equations, the current through the knee agrees within 0.52 percent.

The sweep in the app steps the voltage, so it can stall right at the knee,
where the current is steepest. Smaller voltage steps help.
