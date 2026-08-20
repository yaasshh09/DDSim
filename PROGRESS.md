# Progress

State of the world. Read this first, every session. Append before ending, every
session. Newest entry at the top.

## Current state

**Active phase:** Phase 0, code complete and green locally
**Blocked on:** CI has never run. The repo has no remote and I am not to push,
so the "CI green" acceptance criterion is unverified. Everything else in the
Phase 0 acceptance list passes.
**Next action:** push to a GitHub remote and confirm the workflow goes green on
the 3.11, 3.12 and 3.13 matrix, then start Phase 1

Local verification, Python 3.14.6, numpy 2.5.2, scipy 1.18.0:

    pytest   246 passed
    coverage 99.83 percent, gate is 95
    ruff     clean
    mypy     clean

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

## Known deviations from reference

Things that do not match DEVSIM or an analytic result, with the reason, so they
are not rediscovered as bugs later.

| Item | Deviation | Reason | Acceptable? |
|---|---|---|---|
| n_i vs Nc, Nv, Eg | `sqrt(Nc*Nv)*exp(-Eg/(2*V_T))` = 1.0757e10, we use 1.0e10. 7.6 percent in n_i, 16 percent in n_i^2 | n_i is anchored to the textbook value by decision. Nc and Nv are the measured 300 K values. They are not mutually consistent and cannot both be primary | Yes for now. It will matter the first time something computes a band edge position or a Fermi level from Nc. Revisit at Phase 2 |
| Intrinsic Debye length | docs/06-constants.md says "roughly 24 um". The doc's own formula with the doc's own n_i gives **40.885 um** | 24 um reproduces exactly as `sqrt(eps*V_T/(2*q*1.45e10))` = 24.01 um, so that line carries both an extra factor of 2 and the superseded 1.45e10 n_i. The doping table in the same section (128, 40, 13, 4.1, 0.4 nm) is consistent with the formula, so only the intrinsic line is wrong | No, the doc line should be corrected to 40.9 um. Code follows the formula |
| Built-in potential, 1e16/1e16 | docs/06-constants.md says "about 0.695 V". With n_i = 1.0e10 it is **0.7143 V** | 0.695 V is the n_i = 1.45e10 answer. Same superseded constant as the line above | No, the doc line should be corrected to 0.714 V. This one bites in Phase 1, where V_bi is an acceptance target and a 19 mV offset reads exactly like a boundary condition sign error |
| Symbolic factorization reuse | docs/02-numerics.md calls it a "significant speedup for free". It is not available at all | scipy's splu exposes no symbolic and numeric split. See the Broke entry | No, the doc claim should be softened. The interface is built so a UMFPACK or KLU backend can deliver it later |

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
