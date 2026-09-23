---
title: The MOS capacitor
summary: A gate on oxide on p type silicon, and the three regimes a gate voltage drives the surface through.
docs: 01-physics.md#MOS gate
---

## In plain words

A MOS capacitor is a metal gate, a thin layer of oxide, and a slab of p type
silicon with a contact on the bottom called the body. Here the oxide is a
perfect insulator, so no current ever flows. What the gate voltage does is
shuffle charge around at the silicon surface, and it goes through three
stages.

Make the gate negative enough and holes get pulled up against the oxide.
That's accumulation. The device acts like a plain capacitor whose plates are
the gate and the hole layer, with only the oxide between them. Raise the gate
and the holes get pushed away, leaving a layer of bare negative acceptors.
That's depletion, and the capacitance drops because the charge now sits
deeper. Raise it further and electrons collect in a thin layer right at the
surface. That's inversion: the surface of p type silicon has flipped to n
type.

`t_ox` is the oxide thickness, 10 nm by default. Thinner oxide means more
capacitance. `work_function` is the gate metal's work function, 4.05 eV by
default for n+ polysilicon, and it slides the whole curve along the voltage
axis. `t_si` is how deep the silicon goes, and `width` is how wide the device
is drawn. Nothing changes across the width, so capacitance is reported per
unit area.

## In more depth

The gate isn't a semiconductor node. The oxide carries Poisson's equation
only, and the gate is a fixed potential with the work function difference
folded in:

$$\psi_{gate} = V_{gate} - \Phi_{MS}$$

At the silicon to oxide interface it's the normal component of the
displacement $\varepsilon E$ that's continuous, not the field itself.

Flatband is the gate voltage where the silicon bands stay flat all the way to
the surface. With no fixed interface charge in this model it's just
$\Phi_{MS}$, which for an n+ poly gate on 1e16 p type silicon is $-0.9192$ V.
A wrong $\Phi_{MS}$ slides the curve without changing its shape, so all three
regimes would still look right.

In accumulation the capacitance approaches the oxide capacitance

$$C_{ox} = \frac{\varepsilon_{ox}}{t_{ox}}$$

which for the default 10 nm oxide, with $\varepsilon_{ox} = 3.9\,\varepsilon_0$,
is $3.45 \times 10^{-7}$ F/cm$^2$. It gets there slowly. The accumulation
layer has a finite thickness, so $C_{ox}$ sits in series with a large but
finite silicon capacitance. docs/07-decisions.md records the curve 1.7
percent below $C_{ox}$ at 2.6 V below flatband, and within 1 percent only
around 5 V below.

This is the ideal capacitor, with no fixed interface charge and no poly
depletion. The body is a plate across the whole bottom edge, not a point,
because a point would leave the rest of that edge reflecting, and that's a
different device. The default mesh is 3 columns by 125 rows, graded hard
toward the surface in the silicon and uniform in the oxide, where the
potential is a straight line. The two MOS capacitor benchmarks agree with
DEVSIM on charge to 0.440 percent with a 5 nm oxide and 0.053 percent with a
20 nm one.
