---
title: The Schottky diode
summary: A metal on n type silicon, where the current is electrons hopping over a fixed barrier at the metal instead of crossing a pn junction.
docs: decisions.md#Physics decisions log
---

## In plain words

Put a metal on lightly doped n type silicon and the electrons near the surface
see a step up in energy, the barrier, before they can get into the metal. The
barrier knob sets how tall that step is. At zero bias the few electrons with
enough energy to jump it go both ways equally, so no current flows.

Forward bias the metal and the silicon side of the barrier drops, so far more
electrons make the jump into the metal and the current climbs by about ten
times for every 60 mV. Reverse bias it and the silicon side rises out of
reach, so only the trickle coming the other way, out of the metal, is left.
That trickle barely depends on how hard you reverse bias it, which is why the
reverse current goes flat.

There's no junction and no minority carriers doing the work, so a Schottky
diode switches fast and turns on at a lower voltage than a pn diode does.

The anode is the metal. Run an `iv` sweep on it.

## In more depth

At the metal, psi is pinned so that the electron density there in
equilibrium is $n_{0B} = N_c e^{-\phi_B / V_T}$, and neither carrier is
pinned. Instead each one flows into the metal at
$J = q v (c - c_0)$ with $v = A^* T^2 / (q N_c)$, a recombination velocity of
about $2.2 \times 10^6$ cm/s for electrons. With fast diffusion that is the
thermionic emission law $J = A^* T^2 e^{-\phi_B/V_T}(e^{V/V_T} - 1)$. When
diffusion through the depletion region is slow too, the two limits combine as
$J = q N_c v / (1 + v/v_D)\, e^{-\phi_B/V_T}(e^{V/V_T} - 1)$, the Crowell and
Sze form, and I check it against $v_D$ taken from the solved potential across
four decades of mobility.

The `image_force` knob turns on barrier lowering. An electron near a metal is
pulled toward its own image charge, which shaves
$\Delta\phi = \sqrt{q E / (4 \pi \varepsilon_s)}$ off the top of the barrier,
with $E$ the field at the contact. More reverse bias means more field and a
lower barrier, so with it on the reverse current keeps creeping up instead of
going flat.

The reverse capacitance follows the depletion width, and the intercept of
$1/C^2$ against bias gives back $\phi_B - V_T \ln(N_c / N_d)$, the built in
voltage, to better than a part in a million.
