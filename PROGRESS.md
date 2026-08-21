# Progress

State of the world. Read this first, every session. Append before ending, every
session. Newest entry at the top.

## Current state

**Active phase:** Phase 2 complete, Phase 3 not started
**Blocked on:** CI has still never run. No remote, and I do not push. Everything
in the Phase 0, 1 and 2 acceptance lists passes locally.
**Next action:** push to a GitHub remote and confirm the workflow is green, then
start Phase 3, full Newton on the coupled 3N system

Local verification, Python 3.14.6, numpy 2.5.2, scipy 1.18.0:

    pytest   707 passed in 25 s
    coverage 99.83 percent, gate is 95
    ruff     clean
    mypy     clean

Phase 2 headline numbers, 1e16 / 1e16 diode, 12 um, 201 nodes:

| Quantity | Simulated | Analytic | Error |
|---|---|---|---|
| Saturation current | 1.3322e-10 A/cm^2 | 1.3333e-10 | 0.08 % |
| Reverse current at -1 V | -3.94e-9 A/cm^2 | between I_s and 9.7e-9 | in bracket |
| Jn + Jp spread at 0.5 V | 3.3e-10 | 0 | gate is 1e-6 |
| Terminal current sum at 0.5 V | 2.0e-12 of the largest | 0 | gate is 1e-8 |
| Peak ideality, 1e18 device | 1.79 at 0.16 V | 2 in the ideal limit | see deviations |

## Physics decisions log

Decisions that affect results, with reasoning. Never silently reverse one of
these; add a new entry instead.

| Date | Decision | Reason |
|---|---|---|
| init | n_i = 1.0e10 cm^-3 at 300K | matches textbook worked examples used as analytic targets. See docs/06-constants.md |
| init | C_0 = n_i for de Mari scaling | cleanest Poisson form. Revisit if Phase 5 conditioning degrades |
| init | Boltzmann statistics through Phase 4 | Fermi-Dirac deferred to Phase 5 where degenerate S/D makes it necessary |
| init | Structured tensor mesh in Phase 4 | avoids obtuse triangle problem entirely for rectangular geometry |
| init | Steady state only | transient out of scope. C-V by small-signal AC, not time stepping |
| 2026-08-21 | n_i(300) anchored to 1.0e10, temperature dependence from physics | 1.0e10 cannot be derived from the Nc, Nv and Eg in docs/06-constants.md, those give 1.0757e10. The logged decision wins for the 300 K value and the physics supplies only the T scaling. n_i^2/(Nc Nv exp(-Eg/V_T)) is then exactly T independent, which is tested |
| 2026-08-21 | Eg from the Varshni formula, not the rounded table value | the doc gives both, they differ by 8.1e-5 eV. The formula is the definition and the table entry is it rounded |
| 2026-08-21 | Bernoulli derivative series widened to abs(x) <= 0.1 with terms to x^7 | the doc's recipe (x^3 series, cutoff 1e-4) has 1.8e-8 relative error just outside the cutoff, which fails Phase 0's own 1e-13 criterion by five orders of magnitude. Measured, see the Broke entry below |
| 2026-08-21 | B(x) uses -x*exp(-x)/expm1(-x) for x > 0, no cutoff at 80 | never overflows, underflows to 0.0 on its own past x = 745, and keeps the 290 decades of representable values that "return 0.0" would discard |
| 2026-08-21 | Field.to_scaled on an already scaled field raises rather than no-op | calling it means the caller has lost track of state, which is the bug worth surfacing |
| 2026-08-21 | solve/linear.py always factorizes fresh with COLAMD | no part of a scipy splu factorization is reusable. Measured, see the Broke entry below |
| 2026-08-21 | Graded mesh spacing is monotone per side, not globally | globally monotone is impossible for an interior refinement, the spacing has to fall to h_min and rise again |
| 2026-08-21 | Poisson carries fixed phi_n and phi_p, not one common phi | reverse bias splits the two levels by exactly the applied bias. One common level with a step at the junction drives n to 1e26 cm^-3 there and shrinks the depletion width threefold. Measured |
| 2026-08-21 | Newton residual criterion is relative to the initial residual | an absolute threshold is not scale free. The Poisson residual scales with the doping and so does its roundoff floor, so a fixed 1e-10 that works at 1e16 cannot be met at 1e18 |
| 2026-08-21 | Ohmic contacts by row replacement, giving up matrix symmetry | costs nothing with a direct solver and keeps the assembly obvious |
| 2026-08-21 | Step doping profile is right continuous at the junction | arbitrary, but it has to be decided somewhere, and an abrupt junction is an idealisation anyway |
| 2026-08-21 | Equilibrium minority carrier from n*p = 1, not the quadratic formula | the quadratic formula loses twelve digits computing the minority carrier at 1e16. The reciprocal makes mass action exact rather than merely close |
| 2026-08-21 | device/doping.py and device/equilibrium.py added to the layout | docs/03-architecture.md names neither. Profiles do not belong inside builder.py, and the equilibrium driver has to live outside solve/ to keep that package free of semiconductor knowledge |
| 2026-08-21 | Gummel freezes the SRH denominator rather than using the exact tangent | Writing R as c*n - g with c = p/D and g = n_i^2/D keeps both coefficients non-negative. The continuity matrix is an M-matrix, so a non-negative right hand side means the solved density cannot go negative. The exact tangent dR/dn has no such guarantee. Both forms are exact at the current iterate, so the converged fixed point solves the true equation either way. The exact derivatives are implemented and tested, for Phase 3 |
| 2026-08-21 | Terminal current comes from the continuity residual at the contact node, not the adjacent edge flux | The residual at a contact is exactly the current that has to be injected there. Written that way, the recombination in the contact half cell cancels between the two carriers, and the terminal currents sum to zero identically rather than approximately |
| 2026-08-21 | Dirichlet eliminates the column as well as the row | Row replacement alone leaves the pinned unknown coupled into its neighbours, so the factorization mixes it with the rest of the solution. Measured: a density pinned to +1e-6 came back as -1.02e-6. See the Broke entry |
| 2026-08-21 | Contact densities are imposed on the solved profile, not accumulated through it | old + update cancels when the two are far apart, and on the first forward biased cycle they are thirteen decades apart. Imposing a value that was known before the solve is what a Dirichlet condition means; it is not clamping, and no interior node is touched |
| 2026-08-21 | newton_solve takes an explicit residual scale for warm starts | A threshold relative to the initial residual is unreachable when the solve starts at the answer, which is what every Gummel cycle after the first does. The Poisson block passes the doping charge in the largest dual cell, which does not depend on the starting iterate |
| 2026-08-21 | Density convergence measured as max abs(dn) / (n + n_i) | A pure relative change is dominated by nodes where the density is 1e-15 and physically irrelevant; an absolute change is dominated by the majority carrier. The floor at n_i is 1 in scaled units and says that a carrier below the intrinsic density carries no charge worth converging |
| 2026-08-21 | The Scharfetter lifetime is evaluated on abs(net doping) | It wants the total Na + Nd and only the net is available from a composed profile. The two agree everywhere except in compensated material, and nothing here is compensated yet |
| 2026-08-21 | The ideality crossover is demonstrated on a 1e18 device, not the 1e16 one | At 1e16 with the documented lifetimes the diode is diffusion limited and its ideality is 1 above 50 mV, which is correct rather than a shortfall. Depletion recombination has to dominate somewhere for n = 2 to exist, and raising the doping does that by cutting minority injection and raising the recombination rate at once. Nothing else changes between the two |
| 2026-08-21 | extract/params.py and device/transport.py and device/state.py added to the layout | docs/03-architecture.md names params.py and puts ideality extraction in it. transport.py is the Gummel wiring, which cannot live in solve/ without breaking the module boundary. state.py holds DeviceState, which both equilibrium.py and transport.py return |

## Known deviations from reference

Things that do not match DEVSIM or an analytic result, with the reason, so they
are not rediscovered as bugs later.

| Item | Deviation | Reason | Acceptable? |
|---|---|---|---|
| n_i vs Nc, Nv, Eg | `sqrt(Nc*Nv)*exp(-Eg/(2*V_T))` = 1.0757e10, we use 1.0e10. 7.6 percent in n_i, 16 percent in n_i^2 | n_i is anchored to the textbook value by decision. Nc and Nv are the measured 300 K values. They are not mutually consistent and cannot both be primary | Yes for now. It will matter the first time something computes a band edge position or a Fermi level from Nc. Revisit at Phase 2 |
| Intrinsic Debye length | docs/06-constants.md says "roughly 24 um". The doc's own formula with the doc's own n_i gives **40.885 um** | 24 um reproduces exactly as `sqrt(eps*V_T/(2*q*1.45e10))` = 24.01 um, so that line carries both an extra factor of 2 and the superseded 1.45e10 n_i. The doping table in the same section (128, 40, 13, 4.1, 0.4 nm) is consistent with the formula, so only the intrinsic line is wrong | No, the doc line should be corrected to 40.9 um. Code follows the formula |
| Built-in potential, 1e16/1e16 | docs/06-constants.md says "about 0.695 V". With n_i = 1.0e10 it is **0.7143 V** | 0.695 V is the n_i = 1.45e10 answer. Same superseded constant as the line above | No, the doc line should be corrected to 0.714 V. This one bites in Phase 1, where V_bi is an acceptance target and a 19 mV offset reads exactly like a boundary condition sign error |
| Symbolic factorization reuse | docs/02-numerics.md calls it a "significant speedup for free". It is not available at all | scipy's splu exposes no symbolic and numeric split. See the Broke entry | No, the doc claim should be softened. The interface is built so a UMFPACK or KLU backend can deliver it later |
| V_bi in the Phase 1 acceptance criteria | phases/PHASE-1.md asks for V_T*ln(Na*Nd/n_i^2) to under 0.5 percent and then says "expect about 0.695 V". Those disagree by 2.8 percent, five times the tolerance | The same superseded n_i = 1.45e10 as the constants doc. The formula gives 0.7143 V with our n_i | The formula wins. A test asserts 0.7143 V and asserts it is not 0.695, so the discrepancy stays visible |
| Depletion width on a heavily doped side | x_p/x_n = Nd/Na is only recovered when the depleted width exceeds the local Debye length | At 100 to 1 doping the heavy side depletes over 0.74 Debye lengths, so it is entirely smeared and has no abrupt edge to find. Measured ratio 29 against a predicted 100 | Yes. This is the depletion approximation failing, not the solver. Tested at 10 to 1 where it holds to 30 percent |
| Reverse bias without continuation | -10 V needs 46 Newton iterations against a budget of 50 | The step limiter caps psi at 5 V_T per step, so a 10 V shift needs many of them | Acceptable for now. Bias continuation in Phase 3 is the proper fix, not a bigger budget |
| Current continuity below 0.25 V forward | phases/PHASE-2.md asks for Jn + Jp constant to 1e-6 at any bias. Measured: 3.3e-10 at 0.5 V, 1.3e-8 at 0.4 V, 2.8e-7 at 0.3 V, 1.5e-5 at 0.2 V, 6.6e-4 at 0.1 V | Not a conservation error. Jn is the difference of two edge terms of size (Dn/h)*n, and near equilibrium they cancel to nothing, so the relative spread is machine epsilon times the ratio of a flux term to the current. Measured spread over eps times that ratio sits between 0.4 and 1.2 across seven decades of bias, which is what a subtraction out of digits looks like and is not what a broken scheme looks like | Yes, and tested as such. The gate is asserted at 0.3 V and above, and the cancellation model is asserted separately at every bias including reverse |
| Terminal current sum below 0.25 V forward | Asked for 1e-8 relative. Holds from 0.3 V up (5.6e-9, 2.6e-10, 2.0e-12). At -1 V it is 2.5e-5 relative | The leftover is the sum of the interior residuals, and those are edge flux differences with the same cancellation. In absolute terms it is 1e-13 A/cm^2 against an arithmetic floor of 6.3e-13 | Yes. The strict 1e-8 test runs at forward bias and a separate test bounds the low bias case against the floor rather than against a relative tolerance |
| Where Gummel gives up | phases/PHASE-2.md expects degradation above roughly 0.6 V and asks for the bias at which it fails. It does not fail. Continued in 0.05 V steps it takes 4 cycles at 0.3 V, 12 at 0.7 V, 25 at 0.9 V, 60 at 1.1 V, and the per cycle convergence rate climbs from 0.51 at 0.9 V to 0.95 at 1.8 V | The degradation is exactly as predicted, it is just graceful. Two things help: the Poisson block is solved nonlinearly at fixed quasi-Fermi levels rather than with frozen densities, and continuation hands each solve a good guess. Where a solve does exceed its budget the continuation driver halves the step and the retry succeeds | Yes, and better than the phase expected. The rate curve is the honest answer to the question and it is what motivates Phase 3 |
| Peak ideality factor | 1.79 rather than 2.0, at 0.16 V on the 1e18 device | Two reasons, both physical. The depletion region narrows under forward bias, so the recombination volume shrinks and the current rises slightly faster than exp(V/2V_T), which pulls the apparent ideality below 2. And diffusion current still contributes a few percent at the peak. A sum of two mechanisms has an apparent ideality strictly between theirs | Yes. 2.0 is the limit of a single idealized mechanism, and a solver that reported exactly 2 would be reporting the formula rather than the device |
| Boltzmann statistics at 1e18 | The ideality crossover device runs at 1e18, where n/Nc is 0.035 and Joyce-Dixon puts the Fermi level correction at about 0.3 mV | docs/01-physics.md defers Fermi-Dirac to Phase 5. The correction shifts I_s slightly and leaves a slope alone, and the ideality is a slope | Yes for this use. It would not be acceptable for a quantitative I_s claim at that doping, and no such claim is made |

## Session log

Format per entry:

    ### YYYY-MM-DD
    **Landed:** what works now
    **Broke:** what failed, what the cause turned out to be
    **Open:** unresolved
    **Next:** the single next action

Write the "Broke" field carefully even when it is embarrassing. The debugging
narrative is the most interesting engineering content this project will produce,
and reconstructing it later from git history is much harder than writing it down
now.

### 2026-08-21, later still, Phase 2

**Landed:** All six items of Phase 2 scope, TDD throughout. 707 tests total,
249 of them new. Transport works: the diode passes current, in the right
direction, in the right amount.

- `physics/recombination.py`. SRH with the Scharfetter doping dependent
  lifetime. Unit agnostic, so the same expressions serve physical and scaled
  units and the two routes are compared directly. Exact derivatives verified
  against complex step, plus the frozen denominator linearization that the
  Gummel density solves use.
- `discretize/continuity.py`. Scharfetter-Gummel flux for both carriers, box
  integrated on the same dual cells as Poisson. The Bernoulli asymmetry is
  tested three ways: the zero field limit, the pure drift limit, and which node
  dominates at high field, which is the one that would catch a reversal.
- `discretize/assembly.py`. The residual and Jacobian pair, shared by Poisson
  and both continuity equations so that the boundary conditions are written
  once.
- `solve/gummel.py` and `solve/continuation.py`, both generic. The Gummel
  driver cycles a list of block steps over an opaque state; the continuation
  driver walks a scalar parameter with a solve callback. Neither knows what a
  carrier is, and the import graph test covers both automatically.
- `device/transport.py`, `device/state.py`, `device/builder.with_bias`.
- `extract/iv.py` and `extract/params.py`. Terminal current, I-V sweeps,
  ideality factor and saturation current.

Results. The saturation current comes out at 1.3322e-10 A/cm^2 against an
analytic 1.3333e-10 from the configured lifetimes and diffusion lengths, which
is 0.08 percent, where the phase asks for 10. The short base coth factor is
worth a factor of nine here and the agreement does not survive dropping it, so
that is a real check rather than a coincidence. Reverse current saturates at
-3.94e-9, sitting between the diffusion floor and the full depletion generation
ceiling as it should. Jn + Jp is constant to 3.3e-10 at 0.5 V. Terminal currents
sum to zero to 2e-12. The ideality factor runs from 1.79 at 0.16 V down to 1.005
at 0.59 V on a 1e18 diode, with nothing fitted anywhere.

**Broke:** Six things worth writing down, and the first two were real bugs that
would have been miserable to find later.

1. **The Poisson block could never converge, because its convergence criterion
   was relative to where it started.** Phase 1 made the residual threshold
   relative to the initial residual, for a good reason: the Poisson residual
   scales with the doping and so does its roundoff floor. But a Gummel cycle
   hands the Poisson solve a potential that is already converged, so the
   initial residual is already at the floor, and a threshold set a decade below
   the floor is unreachable no matter how correct the answer is. At thermal
   equilibrium it happens on the very first cycle. The symptom was a device at
   zero bias failing to solve, which looks like a catastrophic sign error and
   is nothing of the sort. Fixed by letting the caller pass a residual scale
   taken from the size of the terms rather than from the starting iterate: the
   Poisson block passes the doping charge in the largest dual cell. There is
   now a test that pins both halves, that a warm start cannot converge without
   a scale and that a genuinely stalled solve is still rejected with one.

2. **A carrier density pinned to +1e-6 came back as -1.02e-6, and whether it
   did depended on the last bit of the device length.** Two test files built
   the same 12 um diode, one as `12 * 1e-4` and one as `12e-4`. Those differ by
   one ulp. In one the cold start at 0.9 V forward converged in 21 cycles, and
   in the other the electron density at the anode came out negative and the
   solve stopped. Finding that was unpleasant and the cause is worth knowing:
   Dirichlet applied by row replacement alone leaves the pinned unknown in
   every neighbouring equation through its column, so the factorization mixes
   it with the rest of the solution. The minority density at a contact is 1e-6
   while the majority density elsewhere is 1e7, thirteen decades apart in one
   linear system, and the pinned value only survives to within the conditioning
   of the whole thing. Fixed by eliminating the column as well as the row,
   which is exact because the update at a pinned node is known before the
   solve. Then a second, smaller version of the same problem: even with an
   exact update, building the answer as old + update cancels when the two are
   far apart, so the contact densities are now imposed on the solved profile
   rather than accumulated through it. Both were needed. Neither is clamping,
   and the distinction matters: nothing inspects a solved value or moves it
   toward anything.

3. **The primary Phase 2 gate looked broken and was not.** Jn + Jp is supposed
   to be constant to 1e-6 and it was out by 35000 percent at zero bias, 4
   percent at 0.1 V, and fine above 0.3 V. The instinct is to go looking for a
   sign error. The cause is that Jn is the difference of two edge terms of size
   (Dn/h)*n, and near equilibrium those cancel to nothing: on this diode the
   terms reach 5e9 in scaled units while the current is zero. So the relative
   spread is about machine epsilon times the ratio of a flux term to the
   current, and that ratio grows exponentially as the bias falls. Rather than
   assume that explanation I measured it: the observed spread divided by eps
   times the ratio sits between 0.4 and 1.2 across seven decades of bias and
   twelve decades of spread. A conservation error has no reason to track that
   ratio. The solved densities are unaffected, and the equilibrium profile
   n = exp(psi) survives a block solve to 8.5e-16 componentwise, which is the
   check that separates the two explanations properly.

4. **My analytic estimate of the depletion recombination current was fifteen
   times too high, so the first ideality result looked wrong.** I used the
   textbook q*n_i*W/(2*tau) with W the full depletion width, predicted a
   crossover at 0.19 V on the 1e16 diode, and measured an ideality of 1.1 with
   no crossover in sight. The simulator was right. The recombination rate is
   sharply peaked where n = p, and the width that matters is the distance over
   which the potential moves by about 2 V_T, which at 3e4 V/cm is 17 nm rather
   than 400. Redoing the estimate with that gives 2.7e-9 A/cm^2 at 0.1 V
   against a measured 3.2e-9. The lesson is the one in the working agreement:
   the analytic target has to be derived properly before it is used to judge a
   solve, because a wrong target is indistinguishable from a wrong answer.

5. **Extracting a saturation current by fitting amplifies the slope error by
   the exponent of the fitting window.** Fitting ln I against V over 0.9 to
   1.0 V and reading I_s off the intercept means running the line back 37 units
   of exponent to zero bias. One percent of a second mechanism left in the
   current moved the fitted ideality by 0.6 percent and I_s by 25 percent.
   `saturation_current` now takes an optional fixed ideality, which measures
   I_s as the average of I / (exp(V/V_T) - 1) with no extrapolation at all.
   That is the number compared against the analytic value, and it agrees to
   0.08 percent.

6. Three test arithmetic errors of mine in one file, all caught by the tests
   failing rather than by review: a lifetime tolerance computed as 1e-6 when
   the correct value was 5e-6, a hand computed SRH denominator with two digits
   transposed, and a low injection limit written without its equilibrium term.
   And `zip(xs, xs[1:], strict=True)` for the third time in this project, which
   is always a length mismatch and which I will presumably write again.

**Open:**

- CI still has never executed.
- Gummel does not fail where the phase expects it to. It degrades smoothly and
  keeps converging to at least 1.8 V, which is well past where constant
  mobility and Boltzmann statistics mean anything. The measurement that matters
  is the per cycle convergence rate climbing from 0.51 at 0.9 V to 0.95 at
  1.8 V, and that is the motivation for Phase 3 rather than any hard wall.
- The 1e6 to 1e-6 dynamic range in the density solves is handled by solving for
  the Newton update rather than for the density, which keeps roundoff relative
  to the update. That works, and it is worth remembering that the density
  formulation is the fragile choice and quasi-Fermi variables are the robust
  one. docs/02-numerics.md suggests them and Phase 3 is where the question gets
  asked properly.
- The overflow guard in the Poisson block is unreachable as configured, since
  the step limiter caps one cycle at 250 units of psi. It is kept with a test
  that says why, so that raising either number brings it back into play
  deliberately.
- docs/06-constants.md and docs/02-numerics.md are still uncorrected. The list
  is unchanged from Phase 1.
- No mobility model yet beyond the constant stub. Arora and Masetti are Phase 3.

**Next:** Phase 3, full Newton on the coupled 3N system, with the Jacobian
verified block by block against complex step before anything is trusted.

### 2026-08-21, later, Phase 1

**Landed:** All six items of Phase 1 scope, TDD throughout. 458 tests total,
228 of them new.

- `physics/statistics.py`. Boltzmann in both scaled and physical form, plus the
  equilibrium solve from doping. The equilibrium densities take the majority
  carrier from the quadratic formula and the minority carrier from n*p = 1,
  which makes mass action exact instead of merely accurate.
- `device/doping.py`. Uniform, Step, Gaussian and Erfc, composable by addition,
  all callables of position so they stay re-evaluable when Phase 5 refines the
  mesh underneath them.
- `discretize/poisson.py`. Box integration on the dual cells, residual signed so
  the Jacobian is a symmetric M-matrix. Jacobian verified against complex step
  differentiation to 1e-12, and separately against finite differences on a
  20 node mesh as the phase asks.
- `discretize/boundary.py`. Ohmic contacts through the asinh form, applied by
  row replacement.
- `solve/newton.py`. Damped Newton with step limiting at 5 V_T. Still knows
  nothing about semiconductors: the assembly arrives through a structural
  protocol, and the import graph test covers this file automatically.
- `device/builder.py`, `device/pn_diode.py`, `device/equilibrium.py`.
- Tier 2 analytic tests, Tier 3 invariants, a mesh refinement study, and the
  band diagram in the README.

Results. V_bi matches V_T*ln(Na*Nd/n_i^2) exactly, though that is mostly a check
on the boundary condition, since the contacts pin it. The genuinely emergent
numbers are the better evidence: depletion width within 1.17 percent at 0 V,
1.27 percent at -1 V and 0.45 percent at -5 V; the potential decaying into the
bulk with the local Debye length to under 1 percent; the peak field converging
at second order under refinement. np = n_i^2 holds to 3e-16 rather than the
1e-8 asked for. Newton takes 8 iterations with a quadratic tail, at every doping
from 1e14 to 1e20.

**Broke:** Five things, and two of them were real bugs rather than bad tests.

1. **The solve failed at 1e18 cm^-3 and above, and the cause was the
   convergence criterion rather than the physics.** The Poisson residual is
   proportional to the doping, so its roundoff floor is too: 3.6e-11 at 1e16,
   7.1e-11 at 1e17, 1.5e-10 at 1e18. An absolute residual tolerance of 1e-10 is
   therefore unreachable above 1e17, and a perfectly good solve gets reported as
   a failure. Phase 5 needs 1e20 source and drain, so this would have surfaced
   there looking exactly like a conditioning problem. Fixed by making the
   residual threshold relative to the initial residual. The update norm needed
   no such treatment, because psi is measured in units of V_T whatever the
   doping. Writing the failing test for this was itself awkward: my first two
   attempts used problems that reach an exact zero residual and so passed
   against the broken code. The version that works states the requirement
   directly, that multiplying the whole system by 1e8 must not change how it
   converges.

2. **Applying a reverse bias did nothing to the junction, and the reason is
   physics rather than code.** With phi_n = phi_p = 0 the carrier densities are
   tied absolutely to psi, so a quasi-neutral region cannot shift its potential
   without changing p by exp(38.7) per volt. The applied volt has nowhere to go
   and drops across a thin layer at the contact instead: measured on a 1e16
   diode at -1 V, the whole volt fell across 0.05 um at the contact with a
   2e5 V/cm field there, while the junction field stayed at its zero bias value
   of 3.2e4 V/cm. So phases/PHASE-1.md asks for something its own scope rules
   out. Depletion widths at -1 V and -5 V cannot be produced by equilibrium
   Poisson alone. The fix is the standard one, and it is step one of Gummel in
   Phase 2: carry fixed quasi-Fermi levels.

   Then I got the fix wrong the first time, in an instructive way. I used a
   single common phi, flat in each quasi-neutral region with a step at the
   junction. That is wrong because reverse bias is precisely the condition
   where phi_n and phi_p are separated, and a single level with a step drives
   n = exp(psi - phi) to about 1e26 cm^-3 on the p side of the junction. That
   fake charge screened the field, and the depletion width came out three times
   too small with a peak field of 1.5e5 V/cm instead of 5.2e4. Two separate
   flat levels, phi_n at the n side contact bias and phi_p at the p side, give
   the right answer at every bias.

3. **The mesh generator overflowed on any mesh with more than about a thousand
   cells.** The bracketing search for the geometric growth ratio starts at ratio
   2 and doubles, and the geometric sum is h_min*(r^m - 1)/(r - 1), so with 1200
   cells it evaluates 2^1199 and raises OverflowError before the bisection ever
   starts. Phase 0 never saw it because 200 cells and 1 nm spacing keep r^m
   small. The sum now saturates to infinity, which is all the bisection needs to
   know.

4. **I measured the depletion width three different ways before finding one that
   works.** A fixed threshold on the field is the obvious approach and it is
   wrong, because the field does not stop at the depletion edge, it decays
   exponentially over the local Debye length. That tail is ten times longer on a
   1e15 side than a 1e17 side, so one threshold measures different things on the
   two sides. Sweeping the threshold from 1 to 50 percent of peak gave widths
   from 0.21 to 0.62 um against a true 0.43 um, and the 3 percent criterion was
   met only by accident at one particular threshold. Fitting the linear part of
   the field on each side and extrapolating to zero, which is what the depletion
   approximation actually claims, lands within 1.2 percent and is stable.

5. **My first bulk neutrality check reported a 48 percent violation.** The mask
   was wrong, not the solver. I asked for nodes more than 0.2 um from the
   junction in a device whose depletion region is 0.43 um wide, so the whole
   "bulk" was still inside the depletion region. A 1 um diode at 1e16 has
   essentially no neutral bulk at all. The deviation decays cleanly in units of
   the local Debye length, and 25 of them gives 1e-8 or better across four
   decades of doping, so that is the criterion now. The devices in these tests
   are 12 um for that reason.

Two smaller ones. A Protocol with mutable attributes cannot be satisfied by a
frozen dataclass, because mutable protocol members are invariant, so
newton_solve rejected PoissonAssembly until the protocol members became
read-only properties. And ScaleFactors.for_silicon(T=0) raised ZeroDivisionError
instead of ValueError, because the defaults evaluate n_i(T) before __post_init__
ever runs. That one was found by a test written purely to close a coverage gap,
which is a fair argument for the gate.

**Open:**

- CI still has never executed.
- Two acceptance criteria in phases/PHASE-1.md are internally inconsistent, and
  I resolved them rather than met them. The 0.695 V figure contradicts the
  formula printed beside it, and the reverse bias depletion widths cannot be
  produced under the phase's own rule of no bias except through the contact
  boundary condition. Both are in the deviations table.
- The frozen quasi-Fermi levels are an approximation. They are exact for reverse
  bias with no recombination and degrade under forward bias, which is where
  Phase 2 has to take over.
- docs/06-constants.md and docs/02-numerics.md are still uncorrected. The list is
  unchanged from the Phase 0 entry, plus the V_bi figure in PHASE-1.md.
- No I-V curve yet, because there is no current yet. That is Phase 2.

**Next:** Phase 2, Scharfetter-Gummel flux assembly and Gummel iteration.

### 2026-08-21

**Landed:** All seven items of Phase 0 scope, TDD throughout, 246 tests.

- Repo restructured. Docs moved from `ddsim-docs/` to `docs/` and `phases/` at
  the root, matching what docs/03-architecture.md assumes, and the three
  duplicated files at the root were removed. DDSim had no git repo of its own,
  it was sitting inside the Desktop-wide repo, so it now has one.
- `core/constants.py`. Every temperature dependent value is a function of T with
  300 K as the default argument. Nc and Nv go through a `BAND_DENSITY` object
  satisfying a `BandDensityModel` protocol, so the band structure layer swaps in
  by replacing one module level object and no call site changes.
  `EffectiveMassBandDensity` is written and tested already, and reproduces the
  tabulated 2.86e19 and 3.10e19 to within 5 percent from silicon effective
  masses, which validates both the formula and the m^-3 to cm^-3 conversion.
- `core/scaling.py`. `ScaleFactors`, frozen, built from T, C_0, eps and D_0 with
  everything else derived. C_0 is a constructor parameter defaulting to n_i(T).
  Unit dispatch is a closed registry that raises on anything unrecognised.
  The load bearing test is that `eps*psi_0/(q*C_0*x_0^2)` is exactly 1, which is
  the entire reason the scaling exists.
- `core/field.py`. Unit, scaling state and mesh location, all checked at runtime,
  all raising. Deliberately no `__array__`, no in place operators, no float
  coercion, and `__array_ufunc__ = None` so numpy defers rather than quietly
  broadcasting a Field into an array. 100 percent covered.
- `physics/bernoulli.py`. Branch structure tuned against an 80 digit `decimal`
  reference rather than accepted from the doc. Worst measured relative error
  1.7e-16 for B and 7.1e-15 for B' over an 8000 point dense sweep. `B(0) == 1.0`
  and `B'(0) == -0.5` are exact. Reflection identity holds to 2.2e-16.
- `solve/linear.py`. COO in, CSC once, splu with COLAMD. Sparsity pattern
  fingerprint tracked. An import graph test enforces that `solve/` never reaches
  into `physics/`, `device/`, `discretize/` or `core/constants`.
- `mesh/mesh1d.py`. Uniform and graded, with the dual grid and both index maps.
  The Phase 0 acceptance case builds: 200 nodes, 1 um domain, 1.0000 nm minimum
  at 0.5 um, 14.40 nm maximum, worst neighbouring cell ratio 1.027588, and the
  dual cell volumes sum to the domain length with **exactly zero** relative
  error. The 1000 to 1 case also works: 100 um domain, 400 nodes, 1 nm at 50 um
  gives a 1867 to 1 spacing range at a 1.0385 growth ratio.
- pytest, coverage gate at 95, ruff, mypy, and a GitHub Actions workflow with a
  lint job and a 3.11 / 3.12 / 3.13 test matrix.

**Broke:** Five things, in rough order of how much time they would have cost if
found in Phase 3 instead of now.

1. **The doc's Bernoulli derivative recipe cannot pass Phase 0's own acceptance
   criterion.** docs/02-numerics.md says series below abs(x) = 1e-4 and
   `(exp(x)*(1-x) - 1)/(exp(x)-1)^2` outside. Measured against 80 digit decimal,
   that closed form has **1.8e-8** relative error at x = 1e-4 and 7.7e-8 at
   1e-5. The criterion is 1e-13. The branch boundary discontinuity would have
   been about 1e-8. The cause is that `exp(x)*(1-x) - 1` subtracts two order one
   quantities to produce an order x^2 result, so the relative error is
   `2*eps/x^2 * x = 2*eps/x^2`-ish and blows up as x shrinks. I had originally
   estimated 4e-12 from a back of the envelope that assumed the cancellation was
   at order x rather than order one, and the measurement was 4000 times worse
   than my estimate. Fix was three changes: rewrite the closed form with expm1
   as `(E*(1-x) - x)/E^2`, which moves the cancellation down to order x and buys
   three orders of magnitude, extend the series to x^7, and widen the window to
   abs(x) <= 0.1. Both sides of the boundary now sit near 1e-15.
2. **scipy cannot reuse any part of an LU factorization, and the obvious
   workaround is a 46x pessimization.** The doc calls symbolic reuse a
   "significant speedup for free". `splu` takes `permc_spec` as a string and
   returns `perm_c`, with no way to pass a symbolic factorization or a
   precomputed permutation back in, so a true symbolic and numeric split is not
   available. I implemented the standard workaround, keep `perm_c` and refactorize
   `A[:, perm_c]` with `permc_spec="NATURAL"`, and benchmarked it. It is slightly
   faster in 1D and catastrophically slower in 2D: 23 ms becomes 352 ms on a
   100x100 five point stencil, 134 ms becomes 6146 ms on 200x200. I assumed the
   column gather was the cost and was wrong, it is 0.6 ms. The factorization
   itself blows up: COLAMD gives 645,750 nonzeros in L and U, the pre-permuted
   NATURAL run gives 3,933,424, a factor of 6.1 more fill. SuperLU's COLAMD path
   does column elimination tree postordering that the NATURAL path skips, so
   `perm_c` alone does not reproduce the ordering it actually eliminated with.
   Ripped the whole thing out, kept the pattern fingerprint (free, and the hook
   a real backend would use), and left a test that pins the ordering to COLAMD
   so nobody reintroduces it.
3. **Complex step differentiation is not exact for this function, contrary to
   both the brief and my own assumption.** It removes cancellation from the
   differencing, not from the function's own algebra, and `B(z) = z/(exp(z)-1)`
   has plenty of the latter near the origin. `Im(B(x+ih))/h` forms two terms of
   order `h*x` to produce a result of order `h*x^2/2`, so it loses roughly
   `eps/abs(x)`. Measured: 9.3e-11 at x = 1e-6, 3.6e-12 at 1e-4, 3.0e-14 at
   1e-2, 3.3e-15 at 0.1. Below abs(x) = 0.01 the reference is worse than the
   implementation it is supposed to check. Resolved by keeping complex step as
   the criterion for 0.1 <= abs(x) <= 300, where it is genuinely exact, and
   adding an 80 digit decimal reference that covers everything including the
   origin. There is also a trap inside the trap: numpy and math have no complex
   expm1, and the naive `exp(z) - 1` makes the reference three orders worse
   again (8.7e-9 at x = 1e-4), which looks exactly like an implementation bug.
   The reference implements complex expm1 by hand.
4. **I measured a branch discontinuity that did not exist.** Evaluating B at
   `nextafter(80, -inf)` and `nextafter(80, +inf)` gave a 2.8e-14 relative
   "jump" at the x = 80 boundary and I spent time trying to tighten the branch
   to close it. There is no jump. Both branch kernels agree at x = 80 to exactly
   0.0. What I was measuring was the genuine slope of B across two ulps of x,
   which at x = 80 is about 1.1e-14 relative because `dB/B` is order 1 there.
   The lesson is now enforced in the test design: the continuity tests call the
   two branch kernels at the *same* x, which is why the kernels are module level
   rather than inlined.
5. **`ScaleFactors.for_silicon(T=0)` raised ZeroDivisionError instead of
   ValueError.** Found by a test written purely to close a coverage gap, which
   is a fair argument for the coverage gate. `__post_init__` validates T, but
   `for_silicon` evaluates `n_i(T)` and `D_n(T)` to build the defaults, and both
   divide by `V_T(T)`, so it blew up before validation ever ran. Now checked in
   both places, with a comment saying why the duplication is deliberate.

Two smaller ones. `filterwarnings = ["error"]` caught a test of mine that
divided a field by itself when the field contained a zero, which is a good sign
the setting earns its keep. And I wrote `zip(xs, xs[1:], strict=True)` twice,
which is always a length mismatch.

**Open:**

- CI has never executed. No remote, and I do not push. The workflow file is
  written and the same three commands run clean locally.
- Three numbers in docs/06-constants.md are wrong: the 24 um intrinsic Debye
  length, the 0.695 V built-in potential, and the implicit claim that n_i is
  consistent with Nc, Nv and Eg. All three trace to a superseded n_i = 1.45e10.
  The code and tests follow the formulas, and the deviations table above records
  it, but the doc itself is still uncorrected. Say the word and I will fix it.
- docs/02-numerics.md needs three corrections: the derivative branch table, the
  "exp overflow otherwise" note on the negative branch (expm1 cannot overflow
  there, the branch is a short circuit), and the symbolic factorization claim.
- The n_i versus Nc, Nv inconsistency is recorded but not resolved. It becomes
  a real decision the first time a band edge or Fermi level is computed.
- `EffectiveMassBandDensity` is tested but not wired in. That is deliberate, it
  is the Phase 1 seam.

**Next:** Push to a GitHub remote and confirm the workflow goes green, then
start Phase 1.
