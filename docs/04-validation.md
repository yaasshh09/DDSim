# Validation

The whole project's credibility rests on this file. "I wrote a device
simulator" can't be checked. "It matches DEVSIM on ten benchmark devices, in
CI, on every push" can.

## Four tiers

1. **Unit** - single functions against closed form values
2. **Analytic** - full solves against textbook closed form device physics
3. **Invariant** - properties that have to hold in any correct solution
4. **Regression** - golden data from DEVSIM, committed to the repo

All four run in CI, and a failure in any of them fails the build.

Phase 8 adds a fifth, the scoreboard in Tier 5 below. Tiers 1 to 4 ask
whether DDSim is right. Tier 5 asks whether it's better than DEVSIM, and it
measures that instead of asserting it.

## Tier 1: unit tests

**Bernoulli.** The gateway test, written first, in Phase 0.

- `B(0) == 1.0` exactly
- `B(-x) == B(x) + x` to 1e-14 relative, swept over x in [-100, 100],
  including values on both sides of every branch threshold
- Continuity across each branch boundary: no jump bigger than 1e-13
- `B(x) -> -x` for x < -80, exactly
- For large positive x, `B(x)` follows x*exp(-x) all the way down and
  underflows to 0 instead of overflowing
- Both B and dB/dx against an 80 digit reference, and dB/dx against complex
  step differentiation where complex step is exact (0.1 <= |x| <= 300, see
  docs/02-numerics.md for why not closer in)

**Statistics.**

- `n * p == n_i^2` at equilibrium, at every doping level
- Joyce-Dixon against tabulated F_{1/2} values, under 1 percent up to
  n/Nc = 4
- Boltzmann and Fermi-Dirac agree to 1 percent when n/Nc < 0.01

**Scaling.** `to_physical(to_scaled(x)) == x` to 1e-14, for every unit type.

**Mobility.** Each model against its published curve at reference points:
Arora at 1e16 and 1e18 cm^-3, Caughey-Thomas approaching v_sat at 1e5 V/cm.

## Tier 2: analytic device tests

Each of these is a full solve compared against a closed form result. The
tolerances are starting points, to tighten as the code improves.

**Ohmic resistor.** A uniformly doped bar between two ohmic contacts has to
obey

    J = sigma * V / L,      sigma = q * (mu_n * n + mu_p * p)

Unlike everything else in this tier, this isn't a limit the solver gets to
miss by a percent. With constant mobility and Boltzmann statistics the Van
Roosbroeck system reduces to Ohm's law exactly, and Scharfetter-Gummel is
exact for a constant field, so the agreement is at solver tolerance and
doesn't improve under refinement.

Measured relative error, across net doping from -1e16 to 1e18 cm^-3,
including the near intrinsic cases where the hole term carries a quarter of
the current:

| Bias | Worst relative error |
|---|---|
| 1e-4 V | 4.3e-12 |
| 1e-2 V | 3.4e-10 |
| 0.1 V | 3.1e-9 |

The growth with bias comes from the Gummel convergence tolerance, not the
discretization. Refinement confirms it: at 1e16 and 0.01 V the error is
1.4e-10 on 11 nodes and 2.0e-10 on 401, flat to the digit that matters. The
test's gate is 1e-7, two orders of headroom over the worst case.

Run this one first. With a single number it pins the Einstein relation, the
drift sign, the contact conditions, the unit scaling and the terminal current
extraction, and if it's wrong nothing further down this list is worth
reading. See `tests/analytic/test_ohmic_resistor.py`.

**Debye length.** Solve nonlinear Poisson on a doping step. The potential
decays with the local Debye length. Fit the decay and compare, under 1
percent.

**Built-in potential.** Abrupt PN junction at equilibrium.

    V_bi = V_T * ln(Na * Nd / n_i^2)

Under 0.5 percent. This expression is itself an approximation, and it degrades
above 1e18 cm^-3 where degeneracy kicks in, so test at 1e16 / 1e16 where it's
clean.

**Depletion width.** Abrupt junction under reverse bias.

    W = sqrt(2 * eps * (V_bi - V) / q * (1/Na + 1/Nd))

Read W off the simulated field profile, under 3 percent. The depletion
approximation has known error at its edges, so don't chase anything tighter.

**Shockley diode equation.** Forward bias, low injection.

    I = I_s * (exp(V / (n * V_T)) - 1)

Fit the ideality factor n over 0.1 V to 0.4 V. It should sit near 2
(recombination dominated) at low bias and move toward 1 (diffusion dominated)
at moderate bias. That crossover showing up on its own is the real test, more
than any tolerance.

Also check I_s against the analytic saturation current from the diffusion
lengths and lifetimes you set. Under 10 percent is respectable.

**Ideal MOS capacitor C-V.** All three regimes: accumulation, depletion and
inversion.

- C_ox in accumulation, from eps_ox / t_ox, under 1 percent
- Minimum capacitance in depletion, from the maximum depletion width
- Flatband voltage from the work function difference
- Threshold voltage against the textbook expression, under 20 mV

**Subthreshold slope.** A long channel MOSFET's Id-Vg on a log scale, with the
slope fitted in the subthreshold region.

    SS = V_T * ln(10) = 59.5 mV/decade at 300K

It has to be at or above 59.5. **If the simulator ever produces a value below
that, there's a bug.** It's a thermodynamic floor for a thermionic device, not
a fitting parameter, and it's the best sanity check in the whole project.

## Tier 3: invariants

Cheap, run on every solve in debug mode, and they catch the most bugs per
line of code.

**Current continuity.** In 1D steady state, with one carrier type or with
recombination off, Jn + Jp is identical at every node. Assert a max relative
deviation under 1e-6. It's the strongest single check available, and it
catches SG sign errors, boundary condition errors and assembly errors alike.

The 1e-6 gate only holds above roughly 0.25 V forward, and the reason is
arithmetic, not physics. Jn is the difference of two edge terms of size
(Dn/h)*n, and near equilibrium they cancel almost completely, so the relative
spread is machine epsilon times the ratio of a flux term to the surviving
current. On the reference diode: 3.3e-10 at 0.5 V, 1.3e-8 at 0.4 V, 2.8e-7 at
0.3 V, 1.5e-5 at 0.2 V, 6.6e-4 at 0.1 V. Divide each by eps times that flux to
current ratio and you get between 0.4 and 1.2 across seven decades of bias.
That's what running out of digits in a subtraction looks like, not what a
broken scheme looks like. Assert the 1e-6 gate at 0.3 V and above, and assert
the cancellation model separately at every bias, reverse included. A gate
that quietly fails at low bias trains you to ignore it.

**Charge neutrality in the bulk.** Far from any junction, |p - n + N| / N is
under 1e-6.

**np = n_i^2 at zero bias.** Everywhere, at every doping, under 1e-8
relative.

**Positivity.** n > 0 and p > 0 at every node, always. A negative density
means the M-matrix property is broken. In 1D that means a sign error. In 2D it
usually means obtuse triangles and negative dual areas. Don't paper over it
by clamping.

**Terminal current sum.** The currents into all contacts sum to zero, under
1e-8 relative to the largest terminal current. This catches boundary
condition errors. Take the terminal current from the continuity residual at
the contact node, not from the edge flux next to it. Written that way, the
recombination in the contact half cell cancels between the two carriers and
the sum is exactly zero instead of roughly zero.

Same low bias caveat as current continuity, for the same reason. It holds
from 0.3 V up (5.6e-9, 2.6e-10, 2.0e-12) and reaches 2.5e-5 relative at -1 V,
where the leftover in absolute terms is 1e-13 A/cm^2 against an arithmetic
floor of 6.3e-13. Bound the low bias case against the floor, not a relative
tolerance.

**Mirror symmetry.** Build the device back to front, solve it, and every
terminal current has to come back the same. Potentials reflect and change
sign, and so do current densities. It's nearly free, and it catches any
asymmetry accidentally baked into the mesh, the assembly or the contact
handling. Gate at 1e-12 relative; measured 9e-16. One caveat worth knowing
before it costs you an afternoon: a step doping profile is right continuous,
so a node sitting exactly on the junction takes the n side value one way
round and the p side value the other. That shifts the metallurgical junction
by one cell, and on a short base diode one cell of base width is worth about
0.1 percent of the current. Put the junction between nodes, or expect that
offset and test for it.

**Gummel and Newton agree.** Same device, same bias, both paths. The solutions
have to agree to solver tolerance. If they don't, one of the two Jacobians is
wrong.

## Tier 4: regression against DEVSIM

DEVSIM is open source, free, and solves exactly this system, so it's the
ground truth. Install it, script the same devices, commit the output curves as
`data/golden/*.csv`, and diff against them in CI.

Benchmark set:

| # | Device | Sweep | Target |
|---|---|---|---|
| 1 | PN diode 1e16/1e16 | I-V, -1 to 0.7 V | 2% on log I |
| 2 | PN diode 1e18/1e16 | I-V forward | 3% |
| 3 | P+N diode 1e20/1e15 | I-V (tests degeneracy) | 5% |
| 4 | MOS cap, 5 nm oxide | C-V, -2 to 2 V | 2% |
| 5 | MOS cap, 20 nm oxide | C-V | 2% |
| 6 | NMOS Lg = 1 um | Id-Vg, Id-Vd | 5% |
| 7 | NMOS Lg = 180 nm | Id-Vg, Id-Vd | 5% |
| 8 | NMOS Lg = 65 nm | Id-Vg (DIBL) | 8% |
| 9 | Lg sweep 1 um to 50 nm | Vth vs Lg | trend + 10% |
| 10 | The same Lg sweep, full Phase 5 stack | Id-Vg, Vth vs Lg, DIBL | 10% |

**Benchmarks 9 and 10 are the same five devices twice.** Benchmark 9 runs
both codes at Boltzmann statistics and constant mobility, which makes a
disagreement there a statement about the 2D transport, the geometry and the
electrostatics and nothing else. Benchmark 10 runs both at the Phase 5 stack,
Fermi-Dirac by Joyce-Dixon with Arora inside Lombardi inside Caughey-Thomas,
and it's the only benchmark in this tier where either mobility model or the
statistics meets an implementation that isn't ddsim's. Having both tells a
mobility disagreement apart from a geometry one: a residual that shows up in
both is the device, and one that only shows up in 10 is a model.

**Match the models before comparing numbers.** DEVSIM's defaults for n_i, the
mobility model and the lifetime have to be set explicitly to match yours, or
the comparison means nothing. Silicon n_i at 300 K is quoted as 9.65e9, 1.0e10
and 1.45e10 depending on the source. Pick one, set it explicitly in both
tools, and write the choice down in `docs/06-constants.md`.

The DEVSIM generation scripts live in `tests/regression/devsim_gen/`, so the
golden data can be reproduced instead of being a mystery.

### Planned benchmarks, Phases 10 to 18

None of these exist yet. Each one is built by the phase that names it, with
the same rule as 1 to 10: match the models in both tools before comparing
numbers.

| # | Phase | Device | Sweep | Target |
|---|---|---|---|---|
| 11 | 10 | MOS cap, 5 nm oxide | C against frequency, DEVSIM `ac` | 3% |
| 12 | 10 | Long p+n diode | Y against frequency, DEVSIM `ac` | 3% |
| 13 | 10 | Uniform resistor bar | voltage noise, DEVSIM `noise` | 5% |
| 14 | 11 | Long p+n diode | reverse recovery, DEVSIM `transient_bdf2` | 5% |
| 15 | 13 | 2D NPN | Gummel plot and one output curve | 5% on log I |
| 16 | 14 | Schottky diode | forward and reverse I-V | 5% |
| 17 | 14 | p+n diode | breakdown I-V through the knee | 5% |
| 18 | 15 | 2D NMOS on one shared Gmsh mesh | Id-Vg | 1% on log Id |
| 19 | 16 | 3D p+n diode on one shared Gmsh mesh | I-V | 2% on log I |
| 20 | 17 | MOS cap, density gradient | C-V | 3% |
| 21 | 18 | Abrupt heterojunction | I-V | 5% |

## Tier 5: the scoreboard

Built in Phase 8, specified in `phases/PHASE-8.md`, with the axes defined in
`phases/ROADMAP.md`. In short:

- **Robustness**: 200 seeded cold start cases, each run through DDSim,
  DEVSIM's stock ramp and my best hand-written DEVSIM ramp. A case counts only
  if it converged, its terminal currents balance, and it agrees with a fine
  step reference. A converged flag on its own isn't evidence.
- **Accuracy**: error against node count on benchmarks 1 to 10, against each
  tool's own Richardson extrapolation.
- **Error control**: the share of reported results that carry an error
  estimate.
- **Speed**: median wall time per converged bias point, one thread each.
- **Features**: a capability matrix where every yes cites a test id for DDSim
  or an API name for DEVSIM.

The data lives in `data/scoreboard/` and is generated by hand, like the golden
data, since DEVSIM isn't in CI. `tests/regression/test_scoreboard.py` does run
in CI. It fails when a DDSim number gets worse than the last committed one,
when a cited test doesn't exist, or when the README table disagrees with the
CSVs.

## Convergence order studies

Refine the mesh by factors of 2 and confirm the error against an analytic
solution falls at the expected rate. Scharfetter-Gummel is formally first
order where fields are strong and second order in smooth regions, so seeing
roughly first order at a junction is correct, not a bug.

Plot error against h on log-log axes and put the plot in the README. It's a
strong sign you understand what you built.

## CI

GitHub Actions on every push:

1. pytest, all four tiers
2. The regression diff against golden data, failing on any tolerance
   violation
3. The README plots, which are drawn by tests, so every run regenerates them
   from the current code
4. A badge

Don't skip step 3. Stale plots in a README are worse than no plots, because
the first thing anyone technical asks is whether the plot matches the current
code.
