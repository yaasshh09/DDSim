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

**devsim dropping a mesh line you asked for, and the contacts on it.** A line
added with `add_2d_mesh_line` is not guaranteed to appear. If the rows graded
up to it land a node close enough to where it goes, devsim keeps the node and
discards the line, at the node's coordinate. Asking the MOSFET generator for a
surface spacing of 1.5625e-9 cm under rows graded from the implant depth puts
the top silicon row at 9.99998817e-5 instead of 1e-4, a gap of 1.18e-10 cm, and
the silicon then has no node on the interface at all. Everything defined at
y = t_si matches nothing: both surface contacts and the si_ox interface are
created empty, and the only word about it comes later, from the first thing to
use one, as `Contact "source" on Device ... does not exist`. Nothing says the
interface went too, and a mesh built that way is not a coarser device, it is a
different one. `check_mesh_landed` in the generator refuses it at build time by
comparing the top silicon row against `t_si` and asking devsim for its contact
and interface lists. Do not infer mesh structure from whether a solve converged.

**Picking a mesh spacing for a doping profile instead of deriving it.** The
MOSFET source and drain are a Gaussian of width `implant_shape(...)[0]`, and
the generator's rows through it were a flat 1e-6 cm, which on this process is
1.21 sigma. More than a standard deviation per row puts the metallurgical
junction in the wrong place, and it shows up nowhere near the junction: the
subthreshold drain current depends exponentially on the surface potential, so a
sub millivolt error in the barrier is percent of current. The reference's own
halved mesh check moved 4.6 percent at zero gate. Splitting the halving by axis
showed the lateral columns carried 0.036 percent of it and these rows carried
2.9 of the 2.93. A quarter of a sigma fixes it and an eighth changes nothing
further. ddsim never had it, because it grades its rows rather than naming a
spacing, and `test_ddsim_resolves_the_implant` holds it there.

**Reading a mesh convergence number where the finer mesh is noisier.** A
convergence check assumes the refined answer is the better one. In the off
state of a MOSFET it is not necessarily: the drain current is a small
difference of much larger fluxes, and a finer mesh has more edges to lose
digits across. Measured on the 1 um device at zero gate and 50 mV, the halved
mesh settles to a largest potential move of 1.1e-16 V and holds its drain
current to ten figures over 120 further passes, and still leaves drain and
source 0.99 percent apart, while the shipping mesh balances to 1.65e-4. The two
meshes differ by 0.53 percent at that point, so the gap is a quarter of what
the finer of them knows about itself. That is not a mesh error, and counting it
as one asks for a refinement that cannot help.

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
| devsim: `Contact "x" does not exist` after a mesh built fine | A requested mesh line was merged into a neighbouring node, so everything defined on it is empty. Check the boundary row landed |
| Mesh refinement moves the off state and nothing else | Rows too coarse for the implant sigma, not the surface spacing |
| Halving the mesh moves less than the terminals fail to cancel | Not a mesh error. The off state current is a flux cancellation and the finer mesh cancels worse |

## When genuinely stuck

Reduce to the smallest failing case. A 20 node 1D uniform mesh, one junction,
constant mobility, recombination off, zero bias. That configuration has a
closed-form answer. If it fails there, the bug is structural and every hour spent
on the full MOSFET is wasted.
