---
title: The pn junction at rest
summary: Put p and n silicon side by side, apply nothing, and watch a voltage appear that no battery put there.
claims: built_in_potential_matches_the_textbook_formula; depletion_width_matches_the_depletion_approximation; the_fermi_level_is_flat_at_equilibrium; the_lightly_doped_side_takes_the_potential_drop
device: {"kind": "pn_diode", "parameters": {}}
sweep: {"kind": "iv", "contact": "anode", "voltages": [0.0]}
---
## Steps

### Solve it at rest

The device is the default diode: 1 um of silicon, $10^{16}$ acceptors on the
left, $10^{16}$ donors on the right, both contacts at 0 V. The sweep has a
single point, 0 V, so this is a solve with no bias at all. Press solve.

On the profile plot, follow psi from left to right. It is flat, then it climbs
in the middle, then it is flat again. Now follow n and p: each one falls by
many decades across the same stretch.

### Look at the bands

Tick "bands" above the profile plot. The conduction and valence band edges
bend through the middle, and the two quasi-Fermi levels lie on top of each
other as one flat line.

### Dope one side harder

set: {"device": {"Na": 1e18}}

The p side now has a hundred times more acceptors. Press solve, and watch
where psi does its climbing now.

## What to look for

- The height of the step in psi from one end to the other. On the default
  device it is about 0.714 V.
- The stretch in the middle where n and p are both far below the doping. That
  is the depletion region, about 0.43 um wide here.
- A Fermi level that stays flat however much the bands bend.
- After the last step, a taller step in psi that sits almost entirely on the
  lightly doped n side.

## What you saw

Electrons diffuse from the n side, where they are plentiful, into the p side,
where they are scarce. Holes do the same the other way. Each carrier that
leaves uncovers a fixed dopant ion, so a layer of negative acceptor ions builds
up on the p side and positive donor ions on the n side. Their field pushes
back on the diffusion until the two balance exactly. At that balance no net
current flows, and the step in psi that holds it there is the built-in
potential:

$$V_{bi} = V_T \ln\!\left(\frac{N_a N_d}{n_i^2}\right)$$

The solver never evaluates this formula. It solves Poisson and the two
continuity equations, and the step comes out of the solve within half a
percent of the formula, both at $10^{16}$ on both sides and at $10^{18}$
against $10^{16}$.

The depletion approximation treats the middle as completely empty of carriers
with sharp edges, which gives

$$W = \sqrt{\frac{2\varepsilon V_{bi}}{q}\,\frac{N_a + N_d}{N_a N_d}}$$

The solved region has soft edges, with carrier tails a few Debye lengths long,
and its width read off the field lands 1.2 percent inside the formula.

The Fermi level is flat because nothing is flowing. In equilibrium the drift
and diffusion of each carrier cancel at every point, and that is the same
statement as a flat quasi-Fermi level. The bands bend by exactly $qV_{bi}$
around it.

The charge on the two sides has to balance, $N_a x_p = N_d x_n$, so the
heavily doped side needs only a thin layer of ions and the light side takes
nearly all of the width, and most of the potential drop goes with it. At
$10^{18}$ against $10^{16}$ the depletion approximation puts 99 percent of it
on the n side. The solve puts 96 percent there, because the heavy side's
depleted layer is only 4 nm thick, the same as its Debye length, and at that
scale a sharp depletion edge is not a good picture: the carrier tail there
holds about 37 mV on its own. That is the one-sided junction, and it is why
the lightly doped side of a real diode sets its capacitance and its breakdown.

Checked by tests/analytic/test_lesson_claims.py on this lesson's own device.
