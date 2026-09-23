---
title: The pn junction at rest
summary: Put p and n silicon side by side, apply nothing, and watch a voltage appear that no battery put there.
claims: built_in_potential_matches_the_textbook_formula; depletion_width_matches_the_depletion_approximation; the_fermi_level_is_flat_at_equilibrium; the_lightly_doped_side_takes_the_potential_drop
device: {"kind": "pn_diode", "parameters": {}}
sweep: {"kind": "iv", "contact": "anode", "voltages": [0.0]}
---
## Steps

### Solve it at rest

This is the default diode: 1 um of silicon, $10^{16}$ acceptors on the left,
$10^{16}$ donors on the right, and both contacts at 0 V. The sweep has one
point, 0 V, so there's no bias at all. Press solve.

On the profile plot, follow psi from left to right. It's flat, then it climbs
in the middle, then it's flat again. Now follow n and p. Each one drops by
many powers of ten across that same stretch.

### Look at the bands

Tick "bands" above the profile plot. The conduction and valence band edges
bend through the middle, and the two quasi-Fermi levels sit right on top of
each other as one flat line.

### Dope one side harder

set: {"device": {"Na": 1e18}}

The p side now has a hundred times more acceptors. Press solve, and watch
where psi does its climbing now.

## What to look for

- How tall the step in psi is from one end to the other. On the default
  device it's about 0.714 V.
- The stretch in the middle where n and p are both far below the doping.
  That's the depletion region, about 0.43 um wide here.
- A Fermi level that stays flat however much the bands bend.
- After the last step, a taller step in psi that sits almost entirely on the
  lightly doped n side.

## What you saw

Electrons diffuse from the n side, where there are plenty, into the p side,
where there are hardly any. Holes do the same in the other direction. Every
carrier that leaves uncovers a fixed dopant ion, so a layer of negative
acceptor ions builds up on the p side and positive donor ions on the n side.
Their field pushes back against the diffusion until the two balance exactly.
At that point no net current flows, and the step in psi that holds the
balance is the built-in potential:

$$V_{bi} = V_T \ln\!\left(\frac{N_a N_d}{n_i^2}\right)$$

The solver never evaluates this formula. It solves Poisson and the two
continuity equations, and the step comes out within half a percent of the
formula, both at $10^{16}$ on both sides and at $10^{18}$ against $10^{16}$.

The depletion approximation pretends the middle is completely empty of
carriers with sharp edges, which gives

$$W = \sqrt{\frac{2\varepsilon V_{bi}}{q}\,\frac{N_a + N_d}{N_a N_d}}$$

The solved region has soft edges, with carrier tails a few Debye lengths
long, and its width read off the field lands 1.2 percent inside the formula.

The Fermi level is flat because nothing's flowing. In equilibrium the drift
and diffusion of each carrier cancel at every point, and that's the same
thing as saying the quasi-Fermi level is flat. The bands bend by exactly
$qV_{bi}$ around it.

The charge on the two sides has to balance, $N_a x_p = N_d x_n$. So the
heavily doped side only needs a thin layer of ions, the light side takes
nearly all of the width, and most of the potential drop goes with it. At
$10^{18}$ against $10^{16}$ the depletion approximation puts 99 percent of it
on the n side. The solve puts 96 percent there. The heavy side's depleted
layer is only 4 nm thick, the same as its Debye length, and at that scale a
sharp depletion edge isn't a good picture: the carrier tail there holds about
37 mV on its own. That's the one-sided junction, and it's why the lightly
doped side of a real diode sets its capacitance and its breakdown.

Checked by tests/analytic/test_lesson_claims.py on this lesson's own device.
