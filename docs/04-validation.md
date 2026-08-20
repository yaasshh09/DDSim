# Validation

The project's entire credibility rests on this file. "I wrote a device simulator"
is unverifiable. "It matches DEVSIM to 2 percent on nine benchmark devices, in
CI, on every push" is a checkable claim.

## Four tiers

1. **Unit** - single functions against closed-form values
2. **Analytic** - full solves against textbook closed-form device physics
3. **Invariant** - properties that must hold in any correct solution
4. **Regression** - golden data from DEVSIM, committed to the repo

Run all four in CI. Fail the build on any tier.

## Tier 1: unit tests

**Bernoulli.** The gateway test. Write it first, in Phase 0.

- `B(0) == 1.0` exactly
- `B(-x) == B(x) + x` to 1e-14 relative, swept over x in [-100, 100] including
  values straddling every branch threshold
- Continuity across each branch boundary: no jump larger than 1e-13
- `B(x) -> -x` for x < -80, exact
- `B(x) == 0.0` for x > 80
- `dB/dx` against complex-step differentiation to 1e-13. Complex step is exact to
  machine precision and will catch algebra errors finite differences hide.

**Statistics.**

- `n * p == n_i^2` at equilibrium, all doping levels
- Joyce-Dixon against tabulated F_{1/2} values, under 1 percent to n/Nc = 4
- Boltzmann and Fermi-Dirac agree to 1 percent when n/Nc < 0.01

**Scaling.** `to_physical(to_scaled(x)) == x` to 1e-14, for every unit type.

**Mobility.** Each model against its published curve at reference points. Arora
at 1e16 and 1e18 cm^-3. Caughey-Thomas approaching v_sat at 1e5 V/cm.

## Tier 2: analytic device tests

Each of these is a full solve compared to a closed-form result. Tolerances are
starting points, tighten as the code improves.

**Debye length.** Solve nonlinear Poisson on a doping step. The potential decays
with the local Debye length. Fit the decay, compare. Under 1 percent.

**Built-in potential.** Abrupt PN junction, equilibrium.

    V_bi = V_T * ln(Na * Nd / n_i^2)

Under 0.5 percent. Note this expression is itself an approximation and degrades
above 1e18 cm^-3 where degeneracy matters. Test at 1e16 / 1e16 where it is clean.

**Depletion width.** Abrupt junction under reverse bias.

    W = sqrt(2 * eps * (V_bi - V) / q * (1/Na + 1/Nd))

Extract W from the simulated field profile. Under 3 percent. The depletion
approximation has known error at the edges, so do not chase tighter than that.

**Shockley diode equation.** Forward bias, low injection.

    I = I_s * (exp(V / (n * V_T)) - 1)

Fit the ideality factor n over 0.1 V to 0.4 V. Should be near 2 (recombination
dominated) at low bias and approach 1 (diffusion dominated) at moderate bias.
That crossover appearing on its own is the real test, more than any tolerance.

Also verify I_s against the analytic saturation current from the diffusion
lengths and lifetimes you configured. Under 10 percent is respectable.

**Ideal MOS capacitor C-V.** All three regimes: accumulation, depletion,
inversion.

- C_ox in accumulation, from eps_ox / t_ox, under 1 percent
- Minimum capacitance in depletion, from max depletion width
- Flatband voltage from the work function difference
- Threshold voltage against the textbook expression, under 20 mV

**Subthreshold slope.** Long channel MOSFET, Id-Vg on log scale. Fit the slope in
the subthreshold region.

    SS = V_T * ln(10) = 59.5 mV/decade at 300K

Must be at or above 59.5. **If your simulator ever produces a value below this,
you have a bug.** It is a thermodynamic floor for a thermionic device, not a
fitting parameter. This is the single best sanity check in the whole project.

## Tier 3: invariants

Cheap, run on every solve in debug mode, catch the most bugs per line of code.

**Current continuity.** In 1D steady state with a single carrier type or with
recombination disabled, Jn + Jp is identical at every node. Assert max relative
deviation under 1e-6. This is the strongest single check available and it catches
SG sign errors, boundary condition errors, and assembly errors alike.

**Charge neutrality in the bulk.** Far from any junction, |p - n + N| / N under
1e-6.

**np = n_i^2 at zero bias.** Everywhere, all doping. Under 1e-8 relative.

**Positivity.** n > 0 and p > 0 at every node, always. A negative density means
the M-matrix property is broken. In 1D that means a sign error. In 2D it usually
means obtuse triangles and negative dual areas. Do not paper over it by clamping.

**Terminal current sum.** Sum of currents into all contacts is zero. Under 1e-8
relative to the largest terminal current. Catches boundary condition errors.

**Gummel and Newton agree.** Same device, same bias, both paths. Solutions must
agree to solver tolerance. If they disagree, one of the two Jacobians is wrong.

## Tier 4: regression against DEVSIM

DEVSIM is open source, free, and solves exactly this system. It is the ground
truth. Install it, script the same nine devices, commit the output curves as
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

**Match the models before comparing numbers.** DEVSIM's defaults for n_i,
mobility model, and lifetime must be set explicitly to match yours or the
comparison is meaningless. Silicon n_i at 300K is quoted as 9.65e9, 1.0e10, and
1.45e10 in different sources. Pick one, set it explicitly in both tools, and
document the choice in `docs/06-constants.md`.

Store the DEVSIM generation scripts in `tests/regression/devsim_gen/` so the
golden data is reproducible rather than mysterious.

## Convergence order studies

Refine the mesh by factors of 2 and confirm the error against an analytic
solution decreases at the expected rate. Scharfetter-Gummel is formally first
order in the presence of strong fields and second order in smooth regions.
Observing roughly first order at a junction is correct, not a bug.

Plot error versus h on log-log. Put the plot in the README. It is a strong signal
that you understand what you built.

## CI

GitHub Actions on every push:

1. pytest, all four tiers
2. Regression diff against golden data, fail on tolerance violation
3. Regenerate README plots so they never go stale
4. Badge

Do not skip step 3. Stale plots in a README are worse than no plots, because the
first thing anyone technical does is ask whether the plot matches the current
code.
