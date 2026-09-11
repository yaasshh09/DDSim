# Pitfalls

Read this when something diverges, oscillates, or converges to nonsense.

## Debugging order

Follow this sequence. Do not skip to step 4.

1. **Check signs.** Run the current continuity invariant. Run a forward-biased
   diode and check that current flows the correct direction. Most "convergence
   problems" are sign errors wearing a disguise.
2. **Check the Jacobian.** Compare every block against finite difference or
   complex step on a 10 node mesh. A wrong derivative turns quadratic convergence
   into stagnation, and stagnation looks exactly like ill-conditioning.
3. **Check scaling.** Print the magnitude of every term in the residual. If terms
   differ by more than 6 orders of magnitude, the scaling is wrong.
4. **Then, and only then**, consider damping and continuation step size.

Adding damping before step 1 hides the bug and costs days.

## Specific traps

**Refining `h_min_y` alone on a MOSFET.** `nmos` builds the silicon axis and
the oxide axis with separate calls to `graded_mesh_1d` and concatenates them,
so the `max_ratio` guard that refuses a neighbouring cell jump above 1.5 never
sees the Si/SiO2 seam. At the defaults the seam is already smooth, 5e-8 of
oxide against 5e-8 of silicon surface, ratio 1.000. Halving `h_min_y` on its
own leaves the oxide where it was and drives that ratio to 2, then 4, then 8,
with no complaint from anything. Measured on the 1 um device: the drain current
moves 5.0, 1.6 and 0.57 percent down the three halvings, which reads like a
converging refinement and is partly a widening discontinuity. Refine `n_oxide`
alongside it, and check the seam rather than trusting the guard.

**Working in unscaled units.** Silicon at 1e20 cm^-3 next to a depletion region
at 1e-10 cm^-3, with lengths in cm and permittivity around 1e-12 F/cm. The
condition number will be astronomical and you will blame SciPy. Scale first.

**`exp(x) - 1` instead of `expm1(x)`.** Catastrophic cancellation near zero.
Silently wrong results in exactly the low-field regions where you expect
central differencing behavior.

**The Bernoulli asymmetry.** Electrons: `B(X)*n_right - B(-X)*n_left`. Holes:
`B(X)*p_left - B(-X)*p_right`. The node the B(X) factor attaches to flips between
carriers. Reversing it gives a solver that converges cleanly to reversed current.
See `docs/02-numerics.md` for the derivation.

**`V_T * log(N / n_i)` for contact potential.** Breaks when N is near zero or
negative, which happens in compensated or lightly doped regions. Use
`V_T * asinh(N / (2*n_i))` everywhere. No exceptions.

**Fermi-Dirac skipped in source/drain.** At 1e20 cm^-3 Boltzmann statistics
overestimate carrier density substantially, and the Einstein relation no longer
holds. Your I-V will be wrong by a large factor and you will not know why.
Either implement it or document explicitly that Phase 5 uses Boltzmann and
carries that error.

**Full field magnitude in Caughey-Thomas.** The model needs the field component
parallel to current flow, which on a mesh means the field along the edge. Using
|E| makes vertical MOS gate field suppress channel mobility incorrectly, and your
Id comes out far too low.

**Surface mobility ignored in a MOSFET.** Bulk mobility in an inversion layer is
too high by 2 to 3x. Your Id will be too high by a similar factor. Lombardi or an
equivalent is not optional if you want quantitative MOSFET results.

**Obtuse triangles in 2D.** Negative Voronoi dual areas, negative conductances,
broken M-matrix, negative carrier densities in regions that make no physical
sense. Check mesh quality at construction time and refuse to proceed on a bad
mesh. Symptom is localized negative n or p that persists under refinement.

**Under-resolving the Debye length at a junction.** At 1e18 cm^-3, L_D is roughly
4 nm. A 10 nm uniform mesh cannot resolve that junction and will produce a
smeared, wrong depletion region. Grade the mesh.

**Clamping negative densities.** Tempting, always wrong. It masks a broken
discretization and produces a solution that satisfies no equation. If densities
go negative, find the cause.

**Damping n and p updates.** Damp psi. Accept full updates on n and p, or better,
solve in quasi-Fermi potentials where the dynamic range is small. Damping density
updates slows convergence without improving robustness.

**Starting from a poor initial guess at high bias.** There is no such thing as a
good initial guess at 1 V forward bias. Continue from equilibrium. Always.

**Continuation step growing too fast.** Growth factor of 2 overshoots into
non-convergence repeatedly and wastes more time than it saves. Use 1.5, and cap.

**Forgetting to reuse the symbolic factorization.** The sparsity pattern is
constant across Newton iterations. Recomputing the ordering every step is pure
waste.

**Building the frontend early.** A visualization of a wrong solution is a
liability. It creates false confidence and the plots look convincing. Solver
first, validated, then UI.

## Symptom to cause table

| Symptom | Most likely cause |
|---|---|
| Newton stagnates, residual flat | Wrong Jacobian entry |
| Newton diverges immediately | Sign error, or no damping on psi |
| Overflow / NaN on first iteration | Unscaled units, or unbounded exp |
| Negative n or p, 1D | Sign error in SG assembly |
| Negative n or p, 2D only | Obtuse triangles, negative dual areas |
| Oscillating carrier profile | Central differencing sneaking in, SG not applied |
| Current wrong sign | Bernoulli left/right asymmetry reversed |
| Jn + Jp not constant | Assembly or boundary condition error |
| SS below 59.5 mV/dec | Definite bug. Physically impossible at 300K |
| Id too high by ~3x in MOSFET | Missing surface mobility model |
| Id too low, saturates early | Field-dependent mobility using \|E\| not E_parallel |
| I-V off by orders of magnitude | n_i mismatch, or Boltzmann in degenerate region |
| Converges at low bias, fails above 0.6 V | Expected for pure Gummel. Switch to Newton |
| Depletion region smeared | Mesh under-resolves local Debye length |
| Results differ from DEVSIM by 2x | Model settings not matched between tools |

## When genuinely stuck

Reduce to the smallest failing case. A 20 node 1D uniform mesh, one junction,
constant mobility, recombination off, zero bias. That configuration has a
closed-form answer. If it fails there, the bug is structural and every hour spent
on the full MOSFET is wasted.
