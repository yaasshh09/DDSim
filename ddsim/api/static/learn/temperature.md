---
title: Temperature
summary: How hot the silicon is, and why a few tens of degrees change the leakage by orders of magnitude.
docs: physics.md#Carrier statistics
---

## In plain words

`T` is the temperature of the whole device, in kelvin. 300 K is room
temperature, about 27 degrees Celsius. The range runs from 200 K, colder
than any freezer, to 500 K, hotter than a chip is ever allowed to run.

Heat shakes electrons loose from the bonds that hold them, so hotter silicon
has many more free carriers of both kinds. That matters most where there
should be almost none: a reverse biased diode leaks far more current, and a
transistor that's meant to be off lets more through. The switch also gets
softer. At room temperature a transistor can't turn off faster than about 60
mV of gate per tenfold drop in current, and that limit grows in step with
the temperature.

## In more depth

The intrinsic density follows the bandgap and the band edge densities:

$$n_i(T) \propto T^{3/2} \exp\left(-\frac{E_g(T)}{2 k T}\right)$$

with $E_g(T)$ from Varshni's fit. Going from 300 K to 400 K raises $n_i$
from 1e10 to about 5e12 cm$^{-3}$, so the $n_i^2$ that sets a diode's
saturation current climbs by more than five orders of magnitude.

The thermal voltage $kT/q$ scales with $T$, and with it the subthreshold
swing floor, $\ln(10)\,kT/q$. A real device sits above the floor by the body
factor $1 + C_d / C_{ox}$. The test suite checks the solved 1 um nmos against
that formula, with $C_d$ read off the solved depletion width, at 250, 300,
350 and 400 K.

Saturation velocity and the Arora and Lombardi mobility models take $T$ too.
SRH lifetimes and the constant mobility model stay at their 300 K values,
and decisions.md says why. The temperature is uniform: there's no self
heating. A 400 K diode agrees with DEVSIM, given the same $n_i$, to 0.06
percent.
