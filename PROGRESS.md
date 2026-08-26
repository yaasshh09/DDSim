# Progress

State of the world. Read this first, every session. Append before ending, every
session. Newest entry at the top.

## Current state

**Active phase:** Phase 4, in progress. Stages 1 and 2 of the plan land: the
assemblies are dimension free, Mesh2D and the quality gate exist, and the
sparse solver carries complex numbers. Phase 3 remains complete on every
criterion that can be checked here, items 1 to 7, with 8 dropped for a reason.
**Blocked on:** nothing. Tier 4 now runs. DEVSIM 2.11 is installed in
`.venv-devsim`, `data/golden/` holds five curves, and benchmarks 1 to 5 of
docs/04-validation.md all pass at their stated tolerance. The Phase 3
acceptance criterion that could not be met in this working copy, "Diode #1 and
#2 in the DEVSIM regression set pass at stated tolerance", is met. Benchmarks
6 to 9 are the MOSFETs and have no golden data, which is a Phase 4 gap rather
than an unverified claim.
**Next action:** Phase 4 stage 3, regions and the Si/SiO2 interface, then the
MOS capacitor and C-V. The small signal AC solve reuses the DC Jacobian Phase 3
built, which docs/02-numerics.md puts at roughly sixty lines; the one thing
that blocked it, a float64 only linear solver, is now fixed.

Phase 3 scope, against phases/PHASE-3.md:

| # | Item | State |
|---|---|---|
| 1 | Full 3N Jacobian for (psi, n, p) | done |
| 2 | Node interleaved ordering | done |
| 3 | Jacobian verification against complex step | done, all nine blocks |
| 4 | Damping, psi limited to 5 V_T per step | done |
| 5 | Hybrid Gummel to Newton driver with fallback | done |
| 6 | Auger recombination | done |
| 7 | Doping dependent mobility, Arora | done |
| 8 | Symbolic factorization reuse | dropped, see deviations |

Phase 3 headline numbers. 1e16 / 1e16 diode, 12 um, 201 nodes, cold from the
Poisson guess with no continuation at all:

| Bias | Newton steps | Final residual | Steps limited | Negative densities |
|---|---|---|---|---|
| 0.6 V | 4 | 2.8e-15 | 0 | none |
| 0.8 V | 8 | 6.2e-16 | 0 | none |
| 1.0 V | 9 | 1.2e-16 | 0 | none |
| 1.2 V | 11 | 1.6e-16 | 1 | none |

Jacobian verification on a 20 node mesh. Worst relative disagreement per block
against complex step differentiation, against a criterion of 1e-10:

| State | Worst block | Error |
|---|---|---|
| Diode at equilibrium | dF_p/dpsi | 3.5e-14 |
| Diode perturbed in psi, n and p at once | dF_p/dpsi | 1.9e-15 |
| Uniform bar, X = 0 on every edge | all nine | 0.0 |

Local verification, Python 3.14.6, numpy 2.5.2, scipy 1.18.0:

    pytest   965 passed in 8 s
    coverage 99 percent, gate is 95
    ruff     clean
    mypy     clean

Gummel against Newton, warm started along 0.05 V continuation steps with a
2000 cycle budget. This is the comparison the phase was really asking for:

| Bias | Gummel cycles | Newton steps |
|---|---|---|
| 0.1 V | 3 | 5 |
| 0.5 V | 5 | 5 |
| 1.0 V | 46 | 4 |
| 1.5 V | 224 | 4 |
| 2.0 V | 466 | 4 |

Remote verification, run 32516425400 on 21ecf00, all four jobs green:

    lint          ruff and mypy on 3.13
    test (3.11)   full suite
    test (3.12)   full suite
    test (3.13)   full suite

Local verification, Python 3.14.6, numpy 2.5.2, scipy 1.18.0:

    pytest   767 passed in 16 s
    coverage 99.84 percent, gate is 95
    ruff     clean
    mypy     clean

Phase 2 headline numbers, 1e16 / 1e16 diode, 12 um, 201 nodes:

| Quantity | Simulated | Analytic | Error |
|---|---|---|---|
| Saturation current | 1.3322e-10 A/cm^2 | 1.330 to 1.353e-10 | 0.1 to 1.6 % |
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
| 2026-08-21 | newton_solve gives up once the residual is frozen to the last bit and the update is already inside tolerance | Both criteria still have to pass and the result is still converged=False, so the outcome is unchanged and only the iteration count moves. The check runs after the convergence test, so a warm start that arrives with a frozen residual and a zero update is still reported as the success it is. A frozen residual under a still moving iterate is deliberately not caught: that is a solver taking real steps that happen not to help, which is a different failure and gets the whole budget. Verified bit for bit inert on psi, n, p and the terminal currents at seven biases |
| 2026-08-21 | extract/params.py and device/transport.py and device/state.py added to the layout | docs/03-architecture.md names params.py and puts ideality extraction in it. transport.py is the Gummel wiring, which cannot live in solve/ without breaking the module boundary. state.py holds DeviceState, which both equilibrium.py and transport.py return |
| 2026-08-22 | The coupled dF_psi/dpsi carries no Boltzmann charge term | In Phase 1 n and p are functions of psi, so the diagonal picks up (n + p)*volume, which is what makes that matrix an M-matrix. Coupled, they are separate unknowns and the same physics lives in dF_psi/dn = +volume and dF_psi/dp = -volume. Carrying it in both places is the most natural way to get this wrong, because the Phase 1 Jacobian is sitting right there to copy from, and Newton would still converge, slower, to the right answer. The block verification catches it and nothing else does |
| 2026-08-22 | The coupled Jacobian uses the exact SRH tangent, not the Gummel frozen denominator | The frozen form exists to keep the continuity matrix an M-matrix so a solved density cannot go negative. The coupled matrix is not an M-matrix under any linearization, so that guarantee is not available to give up. Positivity comes from damping psi instead. Measured: the two give residual histories identical to three significant figures on a 1e16 diode at 0.8 and 1.0 V, so the exact tangent is not bought for convergence, it is bought for being right |
| 2026-08-22 | The coupled system is row scaled by equation family before the solve | One residual threshold cannot serve a charge and a current. The Poisson rows are order N*volume and the continuity rows are order (D/h)*n, six decades apart on a 1e16 device, so a single max abs(F) is set by the larger and declares Poisson converged a million times above its own floor. Each family is divided by the largest single term that goes into it. It is a diagonal left preconditioner, so it changes no answer, and it is applied outside the assembly so the Jacobian the block verification checks is the unweighted one |
| 2026-08-22 | The coupled update is measured per family, absolutely on psi and relative on the densities | max abs(dx) over the raw vector cannot fall below 1e-9 when n is 1e6 in scaled units, so no threshold suits both. docs/02-numerics.md already asks for the carrier change as max abs(dn)/(n + n_i). The floor at n_i matters as much as the ratio: without it the measure is dominated by nodes where the density is 1e-15 and carries no charge |
| 2026-08-22 | Only psi is damped in the coupled Newton, and the psi sub-vector is scaled rather than clipped entry by entry | Both docs/02-numerics.md and docs/05-pitfalls.md prescribe capping psi and taking the full n and p updates. One factor over the whole vector would be set entirely by the density update and would freeze psi. Within psi the direction is preserved, so the rotation is only between psi and the densities and only while the cap is active |
| 2026-08-22 | physics/bernoulli.py grew a complex branch, in production rather than in the test tree | phases/PHASE-3.md makes complex step verification a permanent CI requirement, so a residual that survives a complex argument is a property the code has to have, not test scaffolding. The real path is untouched: one dtype test per call and every array on it is still float64 |
| 2026-08-22 | solve/newton.py takes a limit callable and an update_norm callable rather than learning about components | Both are the same shape of problem: a scalar rule that cannot express a system whose unknowns differ by decades. Injecting them keeps solve/ free of semiconductor knowledge, which the import graph test enforces |
| 2026-08-22 | Phase 3 scope item 8, symbolic factorization reuse, is dropped rather than attempted | Phase 0 already measured it. scipy exposes no symbolic and numeric split, and the standard workaround gives 6.1x fill and a 46x slowdown in 2D. Nothing has changed since. The interface still lets a UMFPACK or KLU backend deliver it later |
| 2026-08-22 | The row scale is measured at every iterate, not frozen at the starting guess | "Does not depend on the starting iterate" means it must not depend on how converged the start is. It does not mean freezing it. The terms a residual is built from are a property of the state, and on a forward biased junction they grow with the injected density: 28 times between guess and answer on a 1e16 diode at 1 V, 660000 times on a 1e20 / 1e14 one. The threshold itself never moves, at residual_rtol against a scale of one; what is re-measured is the question |
| 2026-08-22 | The Poisson term scale counts the carriers, not only the doping | The residual carries -(p - n + N)*volume, and on intrinsic material that sum is exactly zero while n*volume and p*volume are a dual cell each. A scale built from the doping alone is zero there and every row divides by it. An undoped bar returned a residual of nan and a message blaming the LU factorization for being singular |
| 2026-08-22 | Arora mobility is opt in, and the constant model stays the default | Every Phase 1 and 2 number was taken with the constant model, and Arora moves the current by five percent even at doping too low for it to be doing anything, because its N -> 0 limit is 1340 rather than the tabulated 1417. Making it the default would silently invalidate every measurement already in this file |
| 2026-08-22 | Mobility is averaged from nodes to edges arithmetically | Scharfetter-Gummel derives its edge flux assuming the coefficients are constant along the edge, so one value per edge is what the scheme asks for and the only question is which. The mean is the honest reading of a quantity that is genuinely varying, and on a mesh graded to a junction the two endpoints sit in nearly the same doping anyway |
| 2026-08-22 | Auger is opt in as well, and off by default | It changes nothing measurable below high injection, and the same argument about invalidating recorded numbers applies. Its exact tangent is also not sign definite, unlike the SRH one, so turning it on by default would quietly remove a property the Gummel path documents relying on |

## Known deviations from reference

Things that do not match DEVSIM or an analytic result, with the reason, so they
are not rediscovered as bugs later.

| Item | Deviation | Reason | Acceptable? |
|---|---|---|---|
| n_i vs Nc, Nv, Eg | `sqrt(Nc*Nv)*exp(-Eg/(2*V_T))` = 1.0757e10, we use 1.0e10. 7.6 percent in n_i, 16 percent in n_i^2 | n_i is anchored to the textbook value by decision. Nc and Nv are the measured 300 K values. They are not mutually consistent and cannot both be primary | Yes for now. It will matter the first time something computes a band edge position or a Fermi level from Nc. Revisit at Phase 2 |
| Intrinsic Debye length | docs/06-constants.md says "roughly 24 um". The doc's own formula with the doc's own n_i gives **40.885 um** | 24 um reproduces exactly as `sqrt(eps*V_T/(2*q*1.45e10))` = 24.01 um, so that line carries both an extra factor of 2 and the superseded 1.45e10 n_i. The doping table in the same section (128, 40, 13, 4.1, 0.4 nm) is consistent with the formula, so only the intrinsic line is wrong | Corrected in the doc on 2026-08-21. Code follows the formula, and a test pins 40.885 um so a revert cannot take the code with it |
| Built-in potential, 1e16/1e16 | docs/06-constants.md says "about 0.695 V". With n_i = 1.0e10 it is **0.7143 V** | 0.695 V is the n_i = 1.45e10 answer. Same superseded constant as the line above | Corrected in the doc on 2026-08-21, and in phases/PHASE-1.md. This one bites in Phase 1, where V_bi is an acceptance target and a 19 mV offset reads exactly like a boundary condition sign error, so a test asserts 0.7143 and separately asserts it is not 0.695 |
| Symbolic factorization reuse | docs/02-numerics.md calls it a "significant speedup for free". It is not available at all | scipy's splu exposes no symbolic and numeric split. See the Broke entry | Corrected in the doc on 2026-08-21, with the measurement that killed it. The interface is built so a UMFPACK or KLU backend can deliver it later |
| V_bi in the Phase 1 acceptance criteria | phases/PHASE-1.md asks for V_T*ln(Na*Nd/n_i^2) to under 0.5 percent and then says "expect about 0.695 V". Those disagree by 2.8 percent, five times the tolerance | The same superseded n_i = 1.45e10 as the constants doc. The formula gives 0.7143 V with our n_i | The formula wins. A test asserts 0.7143 V and asserts it is not 0.695, so the discrepancy stays visible |
| Depletion width on a heavily doped side | x_p/x_n = Nd/Na is only recovered when the depleted width exceeds the local Debye length | At 100 to 1 doping the heavy side depletes over 0.74 Debye lengths, so it is entirely smeared and has no abrupt edge to find. Measured ratio 29 against a predicted 100 | Yes. This is the depletion approximation failing, not the solver. Tested at 10 to 1 where it holds to 30 percent |
| Reverse bias without continuation | -10 V needs 46 Newton iterations against a budget of 50 | The step limiter caps psi at 5 V_T per step, so a 10 V shift needs many of them | Acceptable for now. Bias continuation in Phase 3 is the proper fix, not a bigger budget |
| Current continuity below 0.25 V forward | phases/PHASE-2.md asks for Jn + Jp constant to 1e-6 at any bias. Measured: 3.3e-10 at 0.5 V, 1.3e-8 at 0.4 V, 2.8e-7 at 0.3 V, 1.5e-5 at 0.2 V, 6.6e-4 at 0.1 V | Not a conservation error. Jn is the difference of two edge terms of size (Dn/h)*n, and near equilibrium they cancel to nothing, so the relative spread is machine epsilon times the ratio of a flux term to the current. Measured spread over eps times that ratio sits between 0.4 and 1.2 across seven decades of bias, which is what a subtraction out of digits looks like and is not what a broken scheme looks like | Yes, and tested as such. The gate is asserted at 0.3 V and above, and the cancellation model is asserted separately at every bias including reverse |
| Terminal current sum below 0.25 V forward | Asked for 1e-8 relative. Holds from 0.3 V up (5.6e-9, 2.6e-10, 2.0e-12). At -1 V it is 2.5e-5 relative | The leftover is the sum of the interior residuals, and those are edge flux differences with the same cancellation. In absolute terms it is 1e-13 A/cm^2 against an arithmetic floor of 6.3e-13 | Yes. The strict 1e-8 test runs at forward bias and a separate test bounds the low bias case against the floor rather than against a relative tolerance |
| Where Gummel gives up | phases/PHASE-2.md expects degradation above roughly 0.6 V and asks for the bias at which it fails. It does not fail. Continued in 0.05 V steps it takes 4 cycles at 0.3 V, 12 at 0.7 V, 25 at 0.9 V, 60 at 1.1 V, and the per cycle convergence rate climbs from 0.51 at 0.9 V to 0.95 at 1.8 V | The degradation is exactly as predicted, it is just graceful. Two things help: the Poisson block is solved nonlinearly at fixed quasi-Fermi levels rather than with frozen densities, and continuation hands each solve a good guess. Where a solve does exceed its budget the continuation driver halves the step and the retry succeeds | Yes, and better than the phase expected. The rate curve is the honest answer to the question and it is what motivates Phase 3 |
| Peak ideality factor | 1.79 rather than 2.0, at 0.16 V on the 1e18 device | Two reasons, both physical. The depletion region narrows under forward bias, so the recombination volume shrinks and the current rises slightly faster than exp(V/2V_T), which pulls the apparent ideality below 2. And diffusion current still contributes a few percent at the peak. A sum of two mechanisms has an apparent ideality strictly between theirs | Yes. 2.0 is the limit of a single idealized mechanism, and a solver that reported exactly 2 would be reporting the formula rather than the device |
| Boltzmann statistics at 1e18 | The ideality crossover device runs at 1e18, where n/Nc is 0.035 and Joyce-Dixon puts the Fermi level correction at about 0.3 mV | docs/01-physics.md defers Fermi-Dirac to Phase 5. The correction shifts I_s slightly and leaves a slope alone, and the ideality is a slope | Yes for this use. It would not be acceptable for a quantitative I_s claim at that doping, and no such claim is made |
| "Gummel failed at 1.0 V" | phases/PHASE-3.md makes converging at 1.0 V "where Gummel failed" the headline criterion of the phase. Gummel does not fail at 1.0 V, and it does not fail at 2.0 V either | Measured on the 1e16 diode, warm started along 0.05 V continuation steps with a 2000 cycle budget: Gummel takes 3 cycles at 0.1 V, 6 at 0.6 V, 46 at 1.0 V, 101 at 1.2 V, 224 at 1.5 V and 466 at 2.0 V, converging every time. Newton takes 4 to 6 across that whole range and does not grow. An earlier reading that Gummel stalled at 1.63 V was the 200 cycle default budget in solve_bias, not a divergence | The criterion is restated rather than dropped. What is tested is that Newton reaches 1.0 V in under a third of the cycles and reaches it cold. The real answer to the phase's question is that Gummel degrades without bound instead of failing, and by 2.0 V it costs 116 times more |
| Cold start above roughly 1.3 V | initial_state, which is the Phase 1 Poisson solve at frozen quasi-Fermi levels, does not converge on a 1e16 diode above about 1.3 V, or on a coarse 1e15 diode above about 1.1 V. It raises rather than returning | The limit is in the guess, not in the coupled Newton. Newton converges at 1.4 V on a 1e18 device and at 2.0 V on a 1e16 one whenever it is handed any state to start from | Yes, and it is what docs/05-pitfalls.md already prescribes: continue from equilibrium, always. Continuation reaches 2.0 V in 8 solves with no retries. Where there is no ramp to come up, the Gummel prelude covers it |
| Asymmetric junction at 1e20 / 1e14 | Was: the electron continuity family stalled at a scaled residual of 2.8e-9 against a threshold of 1.0e-10. **Fixed on 2026-08-22.** | The diagnosis in the row this replaces was wrong. It blamed a single term scale spanning six decades of doping and proposed per node row scaling. The actual cause was that the scale was measured once at the starting guess, and the electron term scale on this device grows by 660000 between equilibrium and 1 V because the minority density on the 1e20 side is injected up by exp(V/V_T). The solve was converged to 4.5e-15 against the terms it actually had, and the threshold was 660000 times too strict | Fixed, and the device converges at 1 V in 14 steps. Boltzmann statistics are still invalid at 1e20 and no quantitative claim is made there; what is claimed is that the solver reports convergence honestly. A test pins it |
| Arora against the tabulated undoped mobility | docs/06-constants.md lists mu_n = 1417 and mu_p = 470 for undoped silicon, and separately gives Arora parameters whose N -> 0 limit is 88 + 1252 = 1340 and 54.3 + 407 = 461.3. 5.4 percent apart for electrons, 1.8 for holes | Both are measured, from different fits to different data. Arora is fitted over the doped range where it gets used rather than at an intrinsic limit nobody measures | Yes, and the reason the constant model stays the default. Switching a device to Arora moves its current by five percent even at doping low enough that the model should be doing nothing, and that would otherwise read as a bug. A test pins both numbers |

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

### 2026-08-25, later, Phase 4 starts: the 1D code stops being 1D

**Landed:** Stages 1 and 2 of the Phase 4 plan, Stage 3 in substance, and one
Stage 5 prerequisite. The suite went 1097 to 1183 tests, coverage 99.89, ruff
and mypy clean throughout. Every new module is at 100 percent. A MOS capacitor
now solves and gets flatband right to machine precision.

**The edge list refactor.** `Mesh1D` has carried `edge_nodes` and `node_edges`
since Phase 0 and nothing ever read them. All 41 assembly sites worked out
their neighbours by slicing, which is a statement that edge e joins node e and
node e+1: true in 1D, false in 2D. `discretize/geometry.py` now carries the
truth instead, as an `EdgeGeometry` holding the edge list, the dual face area
the flux crosses, and the permittivity of the edge. It is one optional
argument on each assembly, defaulting to the 1D contiguous silicon chain, so
no existing call site changed.

The defaults are **exactly** 1.0, which is the whole trick. Multiplying by 1.0
is exact in IEEE754, so `eps_r * dual_face * d / h` collapses back to `d / h`
with the same bit pattern. That claim is measured, not argued: the residual,
all 370 Jacobian nonzeros and the three term scales, computed on a 1e18 / 1e15
device with Arora array diffusivity and Auger at a perturbed state, are
**bit for bit identical** to the pre-refactor code, checked by running the same
script in a worktree at 80ddbec.

Two things worth keeping. Permittivity belongs to the Poisson flux only and
never to a carrier flux, since an electron crossing an edge does not care what
the edge is made of. And the scatter is `np.add.at`, not `np.bincount`, which
is faster on paper but refuses complex weights, and `coupled.py` is
deliberately generic over complex128 so complex step and the AC solve both run
through it. Measured, `add.at` was the quicker of the two anyway, 0.51 ms
against 0.80 on 200k edges.

**Mesh2D.** The tensor product of two `Mesh1D` axes, so the grading solver and
the dual grid are inherited and grading a 2D mesh at a junction is just
grading the axis. Structured rather than Delaunay, per the phase doc.
`mesh/quality.py` is written and tested anyway, because it is the gate that
has to exist before the first unstructured mesh, not after the symptom.

**Complex linear algebra.** `solve/linear.py` forced float64 in three places,
so it could not have factorized the AC system at all. The one that would have
been hard to find is the pattern replay: the cached CSC matrix is float64 after
a DC solve, and writing complex values into it drops the imaginary part with
only a warning, which is exactly the DC-then-AC sequence a C-V sweep does on
one solver. The dtype is now part of what counts as an unchanged pattern.

**Stage 3, most of it.** Three pieces, none of them wired into a Device yet.

*Work functions.* There was no electron affinity and no metal work function
anywhere, and the gate condition is written in terms of Phi_MS, so nothing
could be built without them. The semiconductor side uses `asinh(N/(2 n_i))`,
the same choice docs/05-pitfalls.md already forces on the contact potential and
for the same reason. n+ poly on p-type 1e16 comes out at -0.9192 V, the
textbook number.

*Regions.* `device/regions.py` maps materials onto **cells**, not nodes, and
that distinction is the substance of it. Putting the interface on a node line
makes every edge lie wholly in one material, but the face a horizontal edge
crosses spans half a cell above and half below, and at the interface those are
different materials. So permittivity is an area weighted sum over the cells
either side, and an interface node's dual cell is only half semiconductor.
Classifying nodes instead hands the whole interface row one material, which
moves the effective oxide thickness by half a mesh cell, and t_ox is exactly
what the accumulation capacitance is measured against.

`semiconductor_volume` replaces the plain dual volume wherever the equations
integrate something that only exists in silicon. An oxide node gets zero, which
turns its Poisson row into a bare Laplacian, which is what an insulator is.

*The gate.* `apply_dirichlet_nodes` already took a node list, so most of the
gate contact is a data type. The gate potential is deliberately written without
reference to the substrate: docs/01-physics.md gives `psi_gate = V_gate -
Phi_MS`, which depends on the doping under the gate, and a contact has no
business knowing that. Since psi here is measured from the intrinsic level,
whose work function is exactly `chi + Eg/2`, the same statement is
`psi_gate = V_gate + (chi + Eg/2 - Phi_M)`. That the two agree is checked
across three substrate dopings and all three gate materials, because the doping
only cancels if the algebra is right.

*The first MOS stack.* All of the above composes and solves.
`tests/analytic/test_mos_electrostatics.py` puts a 10 nm oxide on 100 nm of
1e16 p-type silicon with an n+ poly gate and solves equilibrium Poisson on it.

| Check | Result |
|---|---|
| Flatband at V_gate = Phi_MS | psi spread 1.8e-15, Newton takes 0 steps |
| Oxide potential linear in y | 1e-15 relative, so Laplace with no charge |
| Slope ratio across the interface | 0.3310 against eps_ox/eps_Si = 0.3333 |
| Control, oxide given eps_Si | ratio moves to 0.857 |

The flatband result is the one that matters. The flat profile goes in as the
guess and is already the solution, so the gate work function, the substrate
contact potential and the intrinsic reference agree exactly rather than
approximately, and those are computed by three pieces of code that never
otherwise meet.

The 0.7 percent on the slope ratio is physical. The discrete statement at the
interface node is Gauss over its dual cell, half of which is silicon holding
depletion charge, so the displacement genuinely jumps by that charge. It
widens exactly where the charge grows, to 0.12 in accumulation where a sheet
of holes sits at the surface, which is why the check is made in depletion.

*One keystone.* Both meshes now answer `scaled(scale)` with the same three
things. Callers used to write `mesh.volume / scale.x_0` by hand, which is wrong
in 2D where the dual volume is an area and wants x_0 squared. That error is
silent: the device comes out the wrong size by a factor of the Debye length,
converges, and reports a capacitance off by orders of magnitude.

**Broke:** Two real mistakes, one of them expensive.

1. **I ran `git checkout --` on a file whose refactor was uncommitted** and
   destroyed the whole `coupled.py` half of the work. It was cleanup after a
   mutation experiment and the file looked like it only held the mutation. It
   held four hours of refactor. Recovered by replaying the refactor scripts I
   happened to have saved in the scratchpad. The rule now is commit before the
   mutation experiment, not after. Worth noting that another session offered
   `git reset --hard HEAD~1` as the fix, which would have made it strictly
   worse: nothing was committed to reset back to, so it would have thrown away
   the good poisson and continuity commit as well.

2. **My first attempt at the Arora Jacobian test had no teeth and passed
   anyway.** Recorded in the entry below. Same failure mode showed up again
   here and was caught the same way: the first version of the edge list tests
   would have passed a mutation that scattered with `arange` instead of the
   edge list, because in 1D those are the same array. Every new test in this
   session was checked by injecting the fault it claims to guard. The one that
   matters most is edge reversal: four things flip when an edge turns round,
   the sign of X, which Bernoulli factor is which, which node each multiplies,
   and the sign of the scatter, and all four have to cancel.

Two smaller corrections, both from measuring instead of assuming. The 2D dual
cells do **not** sum to the domain area exactly, only to about 1e-16, because
summing an outer product is a different association from multiplying the two
1D sums. And the 1D sum is not unconditionally exact either: it is exact on the
Phase 0 acceptance case, which is where the "exactly zero relative error" in
this file comes from, and off by a bit on other node counts. The existing 1D
tests always used a tolerance. Only my new test and my docstring overclaimed.

**Open:** The Stage 3 physics works, but only when assembled by hand in a
test. `build_device` and `assemble_poisson` are still typed to Mesh1D, and
there is no `Device` holding a Mesh2D, a RegionMap and a gate. That wiring is
Stage 4 and it is deliberately not started: the physics was proved first so
the device API can be designed against something that already runs, rather
than the other way round.

One design smell worth recording rather than churning on: `EdgeGeometry` and
`ScaledMesh` live in `discretize/`, and `mesh/` now imports them, which inverts
the layering docs/03-architecture.md describes. There is no import cycle and
the `solve/` import test still passes, so nothing is broken. If it starts to
grate, the fix is to move `geometry.py` into `mesh/`.

Stage 0 of the plan, the DEVSIM golden data, is deliberately not started. It needs an interpreter that is not this one, since the venv is Python
3.14 and DEVSIM will have no wheel for it, and it is the one piece of the plan
that reaches outside this machine. Tier 4 remains exactly as unverified as it
was. Stages 3 to 6, regions and the oxide, the MOS capacitor, and `extract/cv.py`,
are untouched.

One thing given up knowingly: the Jacobian index arithmetic used to be cached
across Newton steps, keyed on the node count, which an arbitrary edge list
cannot be. That was ten percent of the coupled solve when it was first cached.
The suite time did not move and docs/03-architecture.md says not to optimize
before Phase 5.

**Next:** Stage 3, regions and the Si/SiO2 interface. The interface condition
falls out of box integration for free once each edge carries its own
permittivity, which is the main reason for choosing that discretization.

### 2026-08-25, a debugging pass over the whole codebase

**Landed:** Nothing in `ddsim/` changed. I went looking for defects across the
whole solver and did not find one, which is worth recording as a measurement
rather than as a mood. What I checked, independently of the existing tests:
the three gates (965 passed, ruff and mypy clean); coverage, which reports zero
missed statements and three partial branch arcs; a full suite run under
`np.seterr(divide, invalid, over = "raise")`, which passes, so nothing is
quietly producing a NaN and recovering from it; the tier 3 invariants
re-implemented from docs/04-validation.md against my own code rather than the
project helpers; and the extremes, 77 to 500 K, 1e12 to 1e21 cm^-3, 11 to 1001
nodes, and forward bias to 1.40 V, all of which converge with no non-finite or
non-positive density anywhere.

Two of those deserve their numbers written down. Current continuity came out at
3.32e-10 at 0.5 V against the 3.3e-10 this file already records, so the low bias
spread really is the subtraction-out-of-digits model and not a broken scheme.
And the exactly round minimum densities that made me suspect a clamp, 1.000e-06
on the reference diode and 1.000e-10 at 1e20, are the Dirichlet pinned
equilibrium minority values `n_i/Na`. Physical, not a floor.

What did land is in `tests/unit/test_coupled.py`. The nine block Jacobian
acceptance criterion was only ever running with scalar `Dn`, because the
`models` fixture took the default `mobility="constant"`. Two separate holes
came out of that, and they needed two separate fixes.

**Broke:** The interesting failure was my own first fix. I parametrized the
`models` fixture over constant and Arora, ran the suite, saw it green, and
nearly stopped there. It has no teeth on that axis. The shared `device` fixture
is a 1e16 / 1e16 junction, so `abs(net doping)` is 1e16 on every node, and Arora
reads only the total doping: `Dn` comes back with **one unique value across all
nineteen edges**. A constant array broadcasts exactly like a scalar, so the
"array" case verified the array code path while verifying nothing about whether
the array is aligned to the edges it belongs to. I only found this because I
injected a reversed per-edge `Dn` to prove the new test could fail, and it did
not.

That mutation also went to the wrong function twice before it landed, which is
its own note: `discretize/coupled.py` assembles the coupled 3N Jacobian itself
and never calls `continuity.electron_continuity_jacobian`. Mutating the latter
does nothing to the block tests. The two Jacobians are genuinely separate code,
and I had assumed one fed the other.

The fix is a second fixture, `lopsided_bar`, on a 1e18 / 1e15 profile where
`Dn` takes three distinct values with a factor of 4.7 across the device and is
not symmetric under reversal. With the reversed `Dn` mutation in
`coupled.py`, that test fails on `dF_N/dN` and all four variants of the old one
still pass, which is the demonstration that it earns its place.

The Auger axis turned out to matter more than I expected. Scaling
`AugerRecombination.d_rate_dn` by 1.5 is caught by exactly four tests in the
suite, and all four are the `+auger` variants added here. Before this, a fifty
percent error in the Auger electron linearization passed all 965 tests. The
existing Auger test at `test_coupled_transport.py:448` exercises the model but
never differentiates it.

Also fixed: `README.md` was sitting modified in the working tree, reverted to
the Phase 1 text and missing the whole I-V results section. It was not a partial
edit. The working copy was 5 diff lines from `fdc16dd`, the oldest committed
README, and 156 from the committed Phase 3 one, so it had been overwritten
wholesale with a stale revision. Restored from HEAD.

**Open:** Tier 4 is still the real gap and none of this touched it. Everything
above is the code agreeing with itself and with closed form limits, which is
exactly what tier 4 exists to be independent of. Also unpinned, and left that
way deliberately: the Gummel path's own `electron_continuity_jacobian` has no
per-edge `Dn` alignment test. Nothing caught the reversal mutation there. It is
a linear solve for `n` at fixed `psi`, so a wrong coefficient gives a wrong
answer rather than slow convergence, and doping dependent mobility through the
Gummel path is currently untested.

**Next:** Phase 4 as before, unchanged by this pass.

### 2026-08-22, later, the fix and finish pass on Phase 3

**Landed:** The phase is complete. Scope items 6 and 7 arrived, three real
defects were fixed, and the suite got twice as fast.

- `physics/mobility.py`. Arora, checked against its four analytic limits and
  then against two literature numbers, because the limits would pass for any
  parameter set with the right shape: 1230 cm^2/Vs at 1e16 and 280 at 1e18.
  Doping dependent mobility costs the Jacobian nothing, since the doping does
  not move during a solve, and it slots in by making Dn and Dp arrays over
  edges, which every assembly already accepted. Caughey-Thomas in Phase 5 will
  not be free the same way: it depends on the potential across the edge and so
  puts a dmu/dpsi term into every flux derivative.
- Auger, and a `SumOfRecombination` so mechanisms compose. Checked by the thing
  that actually identifies Auger rather than by a threshold: SRH goes as n at
  high injection and Auger as n^3, so their ratio has to go as n^2, which is a
  hundredfold per decade of density. Measured across eight decades it is 99.1,
  99.9, then 100.0 to four figures, and the crossover where Auger overtakes SRH
  lands at about 6e17 cm^-3.
- The Gummel against Newton table in the Current state section, which is the
  honest answer to what the phase was asking.

**Broke:** Four, and none of them had a failing test to announce them. Three
were found by reading the code for edge cases and one by mutation.

1. **An intrinsic bar returned a residual of nan and blamed the LU
   factorization.** The Poisson term scale was built from the net doping alone,
   and on undoped material that is exactly zero, so every row divided by zero
   and the factorization then reported a singular matrix three call frames
   later. That is the worst kind of diagnostic: it sends you debugging the
   linear algebra when the problem is a divide by zero in the row weights.

   The residual carries -(p - n + N)*volume. On intrinsic material the sum is
   zero but the terms going into it are n*volume and p*volume, a whole dual
   cell each, and the scale has to count the terms rather than the sum. That
   is the same principle the rest of the scaling already rests on and I had
   applied it to the continuity rows and not to this one. An undoped bar now
   converges with a residual of exactly 0.0 and n = p = n_i everywhere.

2. **The row scale was frozen at the starting guess, and that was wrong on
   every device, not just the one where it showed.** docs/02-numerics.md asks
   for a scale that does not depend on the starting iterate. I read that as
   "compute it once at the guess". It means "must not depend on how converged
   the start is", which is a different statement. The terms a residual is built
   from are a property of the state, and on a forward biased junction the flux
   terms grow with the injected density.

   Measured: on the 1e16 diode at 1 V the electron term scale is 28 times
   larger at the answer than at the guess. On the 1e20 / 1e14 junction it is
   660000 times larger, because the minority electron density on the heavily
   doped side is injected up by exp(V/V_T). So the threshold there was 660000
   times too strict and the solve reported failure at 2.8e-9 while sitting at
   4.5e-15 against the terms it actually had.

   The diagnosis in the previous entry's deviations table was also wrong, and I
   have replaced rather than annotated that row. It blamed one global scale
   spanning six decades of doping and proposed per node row scaling. I built
   the per node version far enough to measure it before noticing that the per
   node and per family numbers agreed to within a factor of two at the
   converged state, which is not what a six decade problem looks like. The
   difference was entirely guess against answer.

   The threshold itself still never moves. It stays residual_rtol against a
   scale of one, and what gets re-measured is the size of the terms. Iteration
   counts on the 1e16 diode are unchanged at 4, 8, 9 and 11 for 0.6 to 1.2 V.

3. **The block verification had stopped covering the code that runs.** Sharing
   the Bernoulli pair created a second assembly path, the tests still
   differentiated the standalone functions, and a mutation of the exact SRH
   tangent inside the shared path left every block test green. Two code paths
   where only one is verified is how a verification stops being one.

   Found by re-running the seven mutations after the refactor and getting six.
   Not by reading, and I had read that code twice. A bit for bit equality test
   between the two paths now pins them together, and the mutation set is back
   to seven of seven.

4. **I asserted a threshold for the Auger crossover instead of measuring it.**
   The first version of that test said Auger must exceed SRH by more than a
   thousandfold at a scaled density of 1e9. It is 411 times. Nothing was wrong
   with the model; the number was a guess written as if it were a measurement,
   which is the same failure as the seven digit printout in the previous entry.
   The test now asserts the scaling law, which is both true and much more
   specific about what Auger is.

**Efficiency, all measured before and after.**

The coupled solve was evaluating B four times per Newton step. The residual,
the Jacobian and the row scales all want the same pair, and going through the
three public functions computed it three times over. Sharing it, and caching
the triplet index arrays that never change between steps, took the assembly
overhead from forty percent of a solve to about half that. A coupled solve is
fourteen percent faster and the factorization is now the largest single cost,
which is where it should be.

The test suite went from 17 s to 8 s. Almost all of that was one test that
spent 14 s reaching an assertion about an error message: it induced a Gummel
stall on a 201 node mesh, and inducing a stall costs one full failed solve per
continuation halving, of which the default is ten. `iv_sweep` gained a
`min_step` so a caller who already knows a sweep may stall does not have to pay
for ten refinements of where. That is a real gap in the API rather than a test
convenience, and the test now runs in 0.13 s.

**Open:**

- `initial_state` still gives up above roughly 1.3 V, in the deviations table.
  Continuation and the Gummel prelude both cover it.
- `apply_dirichlet_nodes` is now the largest remaining assembly overhead at
  about six percent of a solve. It masks over every triplet each step, and for
  a fixed contact set and a fixed sparsity pattern that work is constant. Worth
  a cached plan if 1D solve time ever matters, which it does not yet: at the
  10k to 100k unknowns Phase 4 brings, the factorization dominates completely
  and this is noise.
- The two unchanged Phase 2 deferrals, the junction half cell offset and n_i
  against Nc and Nv.
- **Tier 4 has never run.** `data/golden/` is empty and DEVSIM is not
  installed. The Phase 3 brief asks for diodes 1 and 2 of the benchmark set to
  pass, and that criterion is unmet rather than passed. Nothing in this repo
  has ever been compared against an independent solver: everything is checked
  against closed form limits, invariants, and the two internal solvers agreeing
  with each other, which is a real body of evidence and is not the same claim.
  Installing DEVSIM and generating `data/golden/*.csv` is the single largest
  remaining gap in the project's validation.

**Next:** Phase 4. Two dimensions, the MOS capacitor, and C-V by small signal
AC around the converged DC solution, which reuses the Jacobian this phase
built.

### 2026-08-22, Phase 3, the coupled Newton

**Landed:** Scope items 1 to 5 of phases/PHASE-3.md. The full 3N Jacobian, node
interleaved, verified block by block; damping; the coupled driver; the Gummel
prelude. Item 8 is dropped and 6 and 7 are untouched.

- `discretize/coupled.py`. Nine blocks, each written against the residual so a
  device engineer can check it by eye, plus the row scaling, the psi limiter
  and the contact application. The only new algebra in the whole phase is the
  potential derivative of the two fluxes, and the thing to see there is that
  the minus sign in front of the second Bernoulli factor cancels against the
  minus from d(-X)/dX, so the two terms add rather than subtract:

      dJn_e/dX_e = (Dn/h_e) * ( B'(X_e)*n_{e+1} + B'(-X_e)*n_e )

  That makes dF_n/dpsi a Laplacian shaped stencil with conductance G_e, and
  dF_p/dpsi the same stencil negated, which is the statement that raising the
  potential at one end of an edge pushes electrons one way and holes the other.

- `tests/reference/complexstep.py` and the nine block verification. Worst
  disagreement 3.5e-14 against a 1e-10 criterion, and exactly 0.0 on a flat
  bar. Three states: the diode at equilibrium, the diode perturbed in all
  three variables, and a uniform bar constructed to put X = 0 on every edge.

- `solve_bias_newton` and `solve_bias_hybrid` in `device/transport.py`, and two
  new injection points in `solve/newton.py`.

Cold from the Poisson guess with no continuation, the 1e16 diode on 201 nodes
takes 4 steps at 0.6 V, 8 at 0.8 V, 9 at 1.0 V and 11 at 1.2 V, with the
residual reaching 1e-16 and no density going negative anywhere. Continuation
from 0 to 1 V takes 6 solves against a budget of 40, and never has to retry a
step; to 2.0 V it takes 8.

Current continuity holds on the coupled solutions. With recombination off the
spread of Jn + Jp is 6.9e-8 at 0.4 V, 1.4e-9 at 0.5 V and 8.6e-13 at 1.0 V,
against a gate of 1e-6, and the terminal currents sum to zero to better than
1e-8 of the largest from 0.5 V up.

**Broke:** Seven things. The first four are the interesting ones and three of
them were my own reasoning rather than the code.

1. **The complex step reference reported B'(0) as 0.0 instead of -0.5, and the
   thing that made it a real problem is that x = 0 is the common case.** The
   Phase 0 scalar reference writes the real part of expm1(x + iy) as
   `expm1(x)*cos(y) + (cos(y) - 1)` and notes in its own docstring that cos(y)
   rounds to exactly 1.0 for a 1e-20 step. That is true. What follows from it
   is that the second term is exactly 0.0, which is harmless everywhere except
   at x = 0, where that term is the entire real part. B(ih) then comes back as
   exactly 1 with no imaginary part. Writing it as `-2*sin(y/2)^2` squares
   after the sine instead of subtracting against one and it survives at any
   step size. A uniformly doped region at equilibrium has X = 0 on every edge,
   so this is not a corner, and without the fix the two flux-versus-potential
   blocks would have been checked against nothing on exactly the states the
   solver starts from.

   This does not rescue the range Phase 0 documented as untrustworthy. The loss
   near the origin is inside B's own algebra, `h*expm1(x) - x*h*exp(x)` forming
   a result of order `h*x^2/2`, and that is still about `2*eps/abs(x)`. It
   rescues the single point x = 0 and nothing between there and about 1e-5.

2. **I asserted in a docstring that the diode at equilibrium has X = 0 across
   the quasi neutral regions, then measured it and found no edge at zero at
   all.** The smallest edge potential difference is 2.2e-3 and the largest is
   6.5. An abrupt junction on a 20 node mesh leaves structure in psi
   everywhere, so those regions are flat to a few parts in a thousand rather
   than flat exactly. The fix was to construct a uniform bar from the closed
   form instead, which does have X exactly zero, and to say so in the test. It
   is the same failure this file keeps recording: a claim that sounds right,
   written into a docstring, never measured.

3. **One residual threshold cannot serve two equation families.** The Poisson
   rows carry a charge of order N*volume and the continuity rows carry a
   current of order (D/h)*n, and on a 1e16 device in scaled units those are
   9.4e6 and 6.9e6 against a hole scale of 7.9e14. A single max abs(F) is set
   by the largest and declares the other two converged when they are millions
   of times above their own floors. Each family is now divided by the largest
   single term that goes into it, which is a diagonal left preconditioner and
   changes no answer.

4. **Every solve above 0.2 V reported failure while sitting on the exact
   answer, and the cause was the update measure, not the solver.** The residual
   was at 1.8e-16 and the reported update was 8.9e-10 against a threshold of
   1e-10, forever. `newton_solve` measures `max abs(dx)` over the whole vector,
   and n is 1e6 in scaled units, so its last representable bit is 1e-10 and the
   measure has a floor six decades above the potential's. docs/02-numerics.md
   already says to measure the carrier change as `max abs(dn)/(n + n_i)`; I had
   read that as a Gummel convenience and it is a requirement. The 1.0 V solve
   was converging in 9 steps the whole time and being reported as a 30 step
   failure.

5. **The complex step size 1e-20 injects a rounding of its own.** A linear
   function differentiated with it comes back one ulp low on any coefficient
   needing more mantissa bits than fl(1e-20) has to spare: a coefficient of -7
   returns -6.999999999999999. `2**-70` makes both the multiply inside the
   function and the divide outside it exact, so a linear function comes back
   bit for bit. One ulp is nothing against a 1e-10 criterion; a reference that
   is exact where it can be is worth more when a block does disagree.

6. **I nearly wrote up a wrong conclusion because a diagnostic script applied
   the wrong contacts.** Investigating a stall on a 1e20 / 1e14 junction I
   reassembled the residual at the converged state with `base.contacts`, which
   are at 0 V, rather than the biased device's, and got numbers that disagreed
   with the solver's own by eleven decades. The first instinct was that the
   solver was inconsistent. It was the script. Rerun properly it reproduced the
   solver's residual to the last digit and pointed cleanly at the electron
   family, which is the finding recorded in the deviations table.

7. **I claimed Newton and Gummel agree to better than 1e-9 and the number was
   read off a truncated printout.** Both solvers reported an anode current of
   `1.550886e+04` at 1.0 V, seven digits, identical, so I wrote the agreement
   down as better than 1e-9 and moved on. Written as a test at that tolerance
   it fails: the real agreement on the validation diode is 3.6e-9 at 1.0 V and
   1.4e-7 at 0.3 V.

   Chasing it turned out to be worth it, because the answer is not what the
   symptom suggests. Tightening Gummel's `update_tol` from its default 1e-8 to
   1e-10 moves the 1.0 V agreement to 1.9e-11, and tightening it further to
   1e-12 or 1e-14 does not move it again. Gummel stops when its density update
   falls below 1e-8 and the terminal current is very nearly proportional to
   that density, so the disagreement is Gummel's stopping rule and Newton is
   the more accurate of the two at its own defaults. The test now runs Gummel
   tight and asserts 1e-10, which measures agreement instead of measuring
   Gummel's tolerance.

   The lesson is the printout, not the tolerance. Seven digits of agreement is
   evidence of seven digits of agreement and I recorded it as nine.

8. **The convergence tests catch five of seven deliberate Jacobian errors, and
   the two they miss are the ones about recombination.** Mutating the exact SRH
   tangent back to the Gummel frozen slope, and deleting the dF_n/dp cross term
   entirely, both leave the residual history identical to three significant
   figures on a 1e16 diode. The block verification catches both. This is the
   measured argument for why phases/PHASE-3.md makes the block check
   non-negotiable and separate from the convergence check, and it is worth more
   than the argument by assertion I would otherwise have written: a Jacobian
   error that does not show up in the convergence rate is not hypothetical, two
   of the seven I tried are exactly that.

All seven deliberate mutations are caught by the block verification. The list
is in the commit for the assembly.

The coupled solve also sits consistently about eight times further into the
cancellation noise than Gummel on the Jn + Jp invariant, so its gate starts one
continuation step later, at 0.4 V rather than 0.3 V. Both fall off the same
cliff as the bias rises, which is the signature of the cancellation the Phase 2
deviations table describes rather than of a conservation error in either:

    V      0.2      0.3      0.4      0.5      0.6      1.0
    newton 1.4e-4   3.4e-6   6.9e-8   1.4e-9   4.8e-11  8.6e-13
    gummel 2.3e-5   3.4e-7   8.8e-9   4.2e-10  8.2e-12  5.3e-14

**Open:**

- Scope items 6 and 7, Auger recombination and doping dependent mobility. Both
  seams exist already. Doping dependent mobility is on the critical path to the
  project's success criterion, since velocity saturation and threshold roll-off
  cannot emerge from a constant mobility.
- The 1e20 / 1e14 electron family stall, in the deviations table. Per node row
  scaling is the obvious fix. Not urgent, because Boltzmann statistics are
  already documented as invalid at that doping.
- `initial_state` gives up above roughly 1.3 V, in the deviations table.
  Continuation and the Gummel prelude both cover it and docs/05-pitfalls.md
  already says to continue from equilibrium always, so this is a limit rather
  than a defect. It is worth knowing that it is the guess and not the coupled
  solve, because the symptom looks identical.
- The two unchanged Phase 2 deferrals, the junction half cell offset and n_i
  against Nc and Nv. Still deliberate, still not backlog.

**Next:** Auger recombination, then Arora or Masetti mobility.

### 2026-08-21, night, CI is green

**Landed:** The oldest open item in this file is closed. The remote is
`github.com/yaasshh09/DDSim`, everything is pushed, and the workflow that had
been written and never executed has now executed and passed.

Run 32516425400 on 21ecf00, four jobs, all green: lint, which is ruff and mypy
on 3.13, and the full suite on 3.11, 3.12 and 3.13. Sixty-one seconds end to
end. Every step inside every job succeeded, so the coverage gate at 95 passed
on all three interpreters too, since pytest is configured to fail under it.

The matrix is the part I could not check properly from here. Local is 3.14.6,
and the most I could do before the push was parse every source and test file
under a 3.11 feature version, which proves syntax and nothing else. Runtime
behaviour on the oldest supported interpreter is now measured rather than
assumed, which is the whole reason the matrix exists.

**Broke:** One, and it is a claim I repeated three times without checking.

**"CI has still never executed" was false from the Phase 2 entry onward.** The
Actions API lists two runs, not one. The second is 32426149110 on 98d342f, the
phase 1 writeup commit, dated 2026-08-20, and it passed. 98d342f is an ancestor
of main, so that was a genuine push of this branch, not a stray. Which means CI
had already run once, green, before I wrote that it never had, and I then
carried the line forward into two more entries because I was copying my own
previous "Open" list rather than checking the remote. There was no remote
configured in my working copy, and I let that stand in for there being no remote
at all. The two are not the same claim and only one of them was mine to make.

Nothing downstream depended on it, which is the only reason it was cheap. The
lesson is the same one this file keeps recording in different costumes: a fact
that gets copied between entries stops being checked, and the copying is what
makes it feel verified.

**Open:** Two, both unchanged and both genuinely deferred rather than pending.

- The right continuity convention still puts the metallurgical junction half a
  cell off the refined node. Changing it moves every validated number and is a
  decision to take deliberately.
- n_i against Nc and Nv is documented but unresolved, and stays that way until
  something needs an absolute band edge. Phase 5.

**Next:** Phase 3. Full Newton on the coupled 3N system, then bias continuation
on top of it.

### 2026-08-21, night, documentation debt and the stagnation guard

**Landed:** Every item on the open list that could be closed without making a
physics decision. No numbers moved. 761 tests became 767, coverage 99.84
percent, ruff and mypy clean.

The documentation backlog has been carried since Phase 0 and it was the whole of
what was actually fixable, so it went first.

- `docs/06-constants.md`. The intrinsic Debye length is 40.9 um, not 24 um. The
  built-in potential for 1e16 / 1e16 is 0.7143 V, not 0.695 V. Both old numbers
  are the same formula at n_i = 1.45e10, and the Debye one carried a stray
  factor of 2 as well. The extrinsic Debye table went to three significant
  figures from the code rather than the rounded values it had. And the file now
  says out loud that n_i, Nc, Nv and Eg are mutually inconsistent by 7.6 percent
  in n_i, which it had only ever implied, with the anchoring scheme written out
  and the note that it stops being bookkeeping the moment something computes a
  band edge.
- `docs/02-numerics.md`. Three corrections plus one I had not counted. The
  Bernoulli derivative recipe is replaced by what the code ships, with the
  cancellation analysis that forced it and the measured errors on both sides of
  the branch boundary. The "exp overflow otherwise" note on the negative branch
  is gone, because expm1 of a negative argument cannot overflow and the branch
  was a short circuit that buys nothing. The symbolic factorization claim is
  replaced by the measurement that killed it, 23 ms to 352 ms and 6.1 times the
  fill, along with what is genuinely reusable, which is the pattern half of the
  COO to CSC conversion. The fourth was the convergence criteria section, which
  still described an absolute residual threshold. It now carries the scale, the
  flux floor underneath the scale, and the carrier change measured as
  max abs(dn)/(n + n_i).
- `phases/PHASE-1.md`. The 0.695 V acceptance figure contradicted the formula
  printed on the line above it by 2.8 percent against a 0.5 percent tolerance.
  The formula wins and the phase says so.
- `docs/04-validation.md`. Tier 2 gains the ohmic resistor, which was the one
  case in the tier where the solver is not allowed to miss by a percent, with a
  note to run it first because it pins the Einstein relation, the drift sign,
  the contacts, the scaling and the current extraction with one number. Tier 3
  gains mirror symmetry, the right continuity caveat that goes with it, and the
  low bias caveat on both current continuity and the terminal current sum, which
  were stated as unconditional gates and are not.

Then the stagnation guard, which was the one open item that was code.
`newton_solve` takes `stagnation_window`, default 4, and gives up once the
residual has been identical to the last bit across that many consecutive history
entries while the update is already inside `update_tol`. Both halves are load
bearing: a frozen residual under a still moving iterate is a solver taking real
steps that happen not to help, which is a different failure and gets the whole
budget, and there is a test that holds it to that. The check runs after the
convergence test, so it cannot turn a success into a failure. `converged` stays
False either way, so this is a diagnosis speed change and not a semantics change
in the outcome, which is what made it safe to take unilaterally. The message
names the frozen residual, the threshold, the update and the two things that
cause it.

Proved inert three ways, in increasing order of how much they are worth. Psi, n,
p and the terminal currents at seven biases from -1 V to +0.7 V, snapshotted
with the guard on and with it off, byte identical. The whole suite with the
guard defaulted off, where the only failure is the guard's own test and the
other 766 are unmoved. And the one that actually settles it: instrumenting
newton_solve across the analytic, convergence, invariant and regression suites,
the guard fires zero times. It cannot move a device number because no device
solve in the project reaches the state it triggers on.

That last measurement is also the honest limit of the change. The threshold fix
in the previous entry removed the case that motivated the guard, so today it is
latent and only the synthetic unit tests exercise it. It is insurance against
the next threshold that sits under a floor, not a saving anyone will feel now.
One caveat I want on the record rather than buried: the guard reasons from four
bit-identical residuals under a settled update, and while that means the
residual is completely insensitive to the updates being taken, it is not a proof
that accumulated drift could never eventually shift it. `stagnation_window=None`
restores the old behaviour exactly and a test pins that, which is the intended
escape hatch if it ever cuts a solve short.

**Broke:** Two, both mine, both in claims rather than code.

1. **The resistor accuracy in the last entry is wrong.** I wrote that it "comes
   out at 1e-12 and does not move under refinement". The first half only holds
   at the smallest bias. Measured across net doping from -1e16 to 1e18 including
   the near-intrinsic cases: 4.3e-12 worst at 1e-4 V, 3.4e-10 at 1e-2 V, 3.1e-9
   at 0.1 V. The growth is the Gummel tolerance, not the discretization, and the
   refinement half of the claim is fine, 1.4e-10 on 11 nodes against 2.0e-10 on
   401. I had taken the number from the lowest bias case and generalised it. The
   validation doc now carries the table rather than the headline, and the test
   gate at 1e-7 has two orders of headroom over the worst case rather than the
   five I would have assumed.

2. **Correcting the docs made two test docstrings lie.** Both the V_bi test and
   the intrinsic Debye test open by quoting what the docs say, and both docs no
   longer say it. Fixed, and both tests kept, because the numbers they reject
   are the ones every other silicon reference prints and a doc edit that reverts
   should not be able to take the code with it quietly.

**Open:** Three, and all three are now blocked on something other than effort.

- CI has still never executed. Unchanged and unchangeable from here: there is no
  remote and I do not push. What I can say is that the three commands the
  workflow actually runs, `ruff check ddsim tests`, `mypy`, and
  `pytest --cov=ddsim`, were run verbatim and are clean, and that every source
  and test file parses under a 3.11 feature version, which is the oldest entry
  in the matrix and the one most likely to break first.
- The right continuity convention still puts the metallurgical junction half a
  cell off the node the mesh refines to. Deliberately untouched. Changing it
  moves every validated number in the project and that is a decision to take on
  purpose, not as part of a cleanup.
- n_i against Nc and Nv is documented properly now but still unresolved, and it
  stays unresolved until something needs an absolute band edge. That is Phase 5.

**Next:** unchanged. Push to a remote and confirm the workflow is green, then
start Phase 3, full Newton on the coupled 3N system.

### 2026-08-21, evening, performance and debugging pass

**Landed:** An optimisation pass that moved no numbers, and one real bug found
while probing for them.

The bias sweep was spending more time in Python than in SuperLU, so I went
after the overhead rather than the arithmetic. Everything in that commit is the
same computation arranged to run once instead of twice, checked by snapshotting
psi, n, p, Jn, Jp, the terminal currents and the Gummel iteration counts at four
biases before and after. Every value is bit for bit identical.

- `solve/linear.py` keeps the part of the COO to CSC conversion that depends
  only on the sparsity pattern. Across Newton steps only the values move, so
  the sort permutation and the duplicate grouping are computed once and
  replayed as a gather plus a segmented sum. The replay is tested against
  scipy's own conversion bit for bit, duplicate summation order included. This
  was the single biggest win, and it is the thing the module docstring already
  said a real backend would do with the pattern.
- `newton_solve` takes a solver to reuse, so a Gummel cycle stops rebuilding
  the Poisson pattern on every call. It was rebuilding it forty times a sweep.
- The Poisson residual and Jacobian share one pair of exponentials, and the two
  halves of each continuity system share one Bernoulli pair. Each was computing
  the same thing twice per assembly, and the Bernoulli pair alone was 65 percent
  of a continuity assembly.
- Dirichlet is applied to every contact in one pass over the triplets instead of
  rebuilding the whole triplet array once per contact.
- The graded mesh solves every candidate split's growth ratio at once and builds
  spacing arrays only for splits that can win. `np.add.at` on unique contiguous
  indices became slice addition. `ScaleFactors` and `Device` net doping cache
  their derived values, safe because both are frozen.

Sweep 119 ms to 68 ms, mesh build 10.7 ms to 3.0 ms, test suite 26 s to 15 s.

The scalar bisection the mesh used to call is now dead in production, so it
moved to `tests/reference/grading.py` and became the oracle, in the same role
`highprec.py` plays for Bernoulli. The vectorised solver has to match it to the
last bit rather than to a tolerance.

Two test files the suite was missing:

- `tests/analytic/test_ohmic_resistor.py`. A uniformly doped bar has to obey
  J = sigma V / L. Unlike every other Tier 2 target this is not a limit the
  solver is allowed to miss by a percent: with constant mobility and Boltzmann
  statistics the system reduces to Ohm's law identically. It comes out at 1e-12
  and does not move under refinement, because Scharfetter-Gummel is exact for a
  constant field. It pins the Einstein relation, the drift sign, the contacts,
  the scaling and the current extraction with one number.
- `tests/invariant/test_mirror_symmetry.py`. Build the diode back to front and
  it has to give the same current. It does, to 9e-16.

**Broke:** Three, in increasing order of how much they were worth finding.

1. **My first pruning bound for the mesh split search was wrong, and the
   fallback hid it.** I argued that every cell on one side grows by the same
   ratio, so a split cannot score below max(ratio_left, ratio_right), and pruned
   on that. Tests passed and the mesh was identical, because I had written a
   fallback that rescans every split if the bound is ever violated. What I
   missed is that `_side_spacings` rescales each side to land on its length
   exactly, and for a side of one cell the ratio solve returns 1.0 without
   solving anything, so that single cell becomes the whole side rather than
   h_min. The two cells at the pivot are then 500 to 1 apart against a bound of
   1.008. The fallback fired on every call, so the pruning saved nothing and the
   mesh build barely moved. Putting the pivot junction into the bound turns it
   into the exact score up to rounding, and only then did 10.7 ms become 3.0 ms.
   The lesson is that a correctness fallback with no counter on it is invisible.

2. **A mirrored diode disagreed with the original by 0.13 percent, at every
   bias, in both directions.** Too systematic to be arithmetic. The mesh turned
   out symmetric to 4e-20 and the doping arrays differed at exactly one node out
   of 201. `Step` is right continuous, so the node sitting exactly on the
   junction takes the n side value one way round and the p side value the other,
   which moves the metallurgical junction by one cell. These diodes are short
   base, half a micron against a diffusion length of 175 um, so the saturation
   current goes as 1/W and one cell of base width is worth a tenth of a percent.
   Confirmed by shifting the step by one h_min on a single device, which
   reproduced the mirror disagreement to six digits. Not a defect, and the right
   continuity convention is already in the decision log above, but the
   interaction was not obvious and both facts are now pinned by tests.

3. **The Poisson residual threshold sat below the arithmetic floor of the
   residual for lightly doped material, and a converged solve reported
   failure.** This is the real one. The threshold is built from the doping
   charge in the largest dual cell, which is right as far as it goes: it does not
   depend on where the iteration started, which is what lets a warm Gummel cycle
   report success. But the residual is also a difference of two face fluxes of
   size psi/h, and a difference cannot be resolved below eps times the size of
   the things being differenced. The two scale in opposite directions. The
   charge falls linearly with the doping while psi is only logarithmic in it, so
   the flux floor stays put, and below about 1e13 the threshold sinks underneath
   it.

   On a 1e12 uniform bar the residual reached 6.8e-12 at the third iteration and
   sat there, unchanged to the last bit, for the remaining forty-seven, with an
   update of 4.4e-16 throughout. Newton had solved it at step three and then
   raised. I found it by accident, building a near intrinsic resistor to
   exercise the hole term in the conductivity.

   The scale is now raised to clear the flux floor, and only when the floor
   would otherwise bind, so every device where the charge already clears it
   keeps the threshold it had. Verified by snapshot, unchanged bit for bit.
   1e12 and 1e11 now converge in 3 iterations instead of failing after 50. High
   resistivity substrates run at exactly these dopings, so this was a range the
   project needed rather than a curiosity.

   What made it slow to spot is that the failure message gave the final residual
   and the final update but not the threshold. Without the number being compared
   against, a residual stuck above a threshold reads exactly like a solve that is
   merely slow, which is a different problem with a different fix. The message
   carries it now.

**Open:**

- Newton spins its whole iteration budget when the residual has stopped moving
  and the update is already below tolerance. Reporting failure there is correct
  and documented, since both criteria have to pass, but spending forty-seven
  iterations to do it is waste. A stagnation guard changes generic solver
  semantics and I did not want to make that call unilaterally.
- The right continuity convention means the shipped `pn_diode` puts the
  metallurgical junction half a cell off the node the mesh refines to. A box
  integrated abrupt junction would arguably give that node zero net doping,
  since its dual cell is half p and half n. That is a physics decision that
  would move every validated number, so it stays as it is unless taken
  deliberately.
- docs/04-validation.md Tier 2 has no resistor case. The test exists now, the
  doc does not mention it.
- CI has still never run. Unchanged.

**Next:** unchanged. Push to a remote and confirm the workflow is green, then
start Phase 3.

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
analytic 1.330 to 1.353e-10 from the configured lifetimes and diffusion
lengths, so between 0.1 and 1.6 percent where the phase asks for 10. The range
on the analytic side is real and it is the looser of the two: the expression
needs a quasi-neutral width, and taking the depletion edge at 0 V rather than
at the middle of the fitting window moves it by 1.7 percent. The simulated
value is the tight one, converged to five figures under refinement from 101 to
801 nodes, which is what licenses quoting an agreement at all. The short base coth factor is
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

### 2026-08-26, later: tier 4 gets its MOS half

**Landed:** Benchmarks 4 and 5 of docs/04-validation.md, the two MOS capacitor
rows the diode session had to leave out. `data/golden/` now holds five curves
instead of three, and `tests/regression/test_devsim_mos.py` compares gate
charge and C-V against them. Suite 1243 to 1378, ruff and mypy clean.

ddsim agrees with DEVSIM to 0.44 percent on the 5 nm device and 0.053 percent
on the 20 nm one, against a 2 percent budget. Better than that: refining
ddsim's surface mesh drives the disagreement to 15 ppm, at which point it stops
falling because it has reached the reference's own mesh error. The two codes
agree to the joint discretization limit of both, which is as strong a statement
as this tier can make.

**Broke:** The interface mesh line in the generator had `ps` and `ns` the wrong
way round. devsim grades between mesh lines and each line names its spacing
separately per direction, `ps` walking in +x and `ns` walking in -x. The
interface is the only line on the device with material on both sides, so it is
the only place the two differ and the only place the mistake is silent: the
silicon surface got meshed at the oxide spacing and the oxide at the surface
spacing. The mesh looked refined. It refined the wrong region.

What gave it away was a refinement that did nothing. `devsim_h_surface` was
five times finer than ddsim's mesh by construction, and dividing it by five
again moved the converged answer by zero, which is not how a mesh behaves near
an inversion layer. It was refining the oxide, whose potential is a straight
line. Meanwhile `devsim_oxide_cells` was the knob that actually resolved the
inversion layer, entirely by accident, which is why the recorded convergence
number was a real measurement of the wrong thing.

Two lessons rather than one. The first is that I asserted the convention
instead of reading it. The second is that the docstring saying "the oxide is
uniform because a straight line needs no grading" was correct and untested for
exactly as long as it was wrong in code, because the parameter it described was
not connected to the oxide. Now measured: the gate charge is identical from 2
oxide cells to 32 to within 8e-15, which is round off.

The golden mesh is now picked off a ladder rather than by eye. Surface spacing
is the only knob the answer feels: 8e-8 cm sits 4.3e-4 from converged, 5e-9 is
6.8e-5, and 1.25e-9 is where it stops moving. Bulk spacing contributes 3.6e-6.
Mesh convergence went from 3.7e-3 and 1.3e-3 to 5.5e-5 and 4.4e-6, which moves
the reference from 19 percent of the tolerance budget to 0.27 percent. The
golden charge itself moved about 0.4 percent in deep inversion, so the old
files were wrong by a fifth of the tolerance they were about to be asserted
against.

**A tolerance is a weak test, so there is a second one.** ddsim clears the
doc's 2 percent by a factor of four and a half, which means a bare 2 percent
assertion would sit green through a 1 percent error in the permittivity or the
work function. `test_disagreement_is_ddsim_discretization` runs the comparison
down a refinement ladder and requires the disagreement to shrink by at least
2x per step. Nothing in the model refines away: a wrong constant, a missing
interface term or a sign error all produce a residual that sits exactly where
it is under refinement. Only discretization moves.

That test earns its place, and I checked rather than assumed it. Perturbing the
oxide permittivity by 1 percent in `device/regions.py`, which the constants
mirror does not see and which the 2 percent charge tolerance swallows, fails
the convergence test. Every other test here was mutated too: the work function,
the Varshni band gap, the recorded convergence number, and a benchmark geometry
edited without regenerating. All fail on the line they guard.

**Open:**

- Benchmarks 6 to 9 are the MOSFETs. No golden data and no device yet.
- The ddsim mesh in the benchmarks stays at 121 nodes and h_min 5e-8, which is
  where the 0.44 percent comes from. Deliberate: it is the default a user gets,
  so the benchmark should measure it rather than a mesh chosen to look good.
- Three numbers in docs/06-constants.md are still wrong, all tracing to a
  superseded n_i = 1.45e10. Unchanged from the earlier entry.

**Next:** Phase 5, the C-V extraction work, or benchmark 6, the 1 um NMOS.
