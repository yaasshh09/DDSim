# Pitfalls

Read this when something diverges, oscillates, or converges to nonsense.

## Debugging order

Follow this order, and don't skip to step 4.

1. **Check signs.** Run the current continuity invariant. Run a forward biased
   diode and check the current flows the right way. Most "convergence
   problems" are sign errors in disguise.
2. **Check the Jacobian.** Compare every block against finite differences or
   complex step on a 10 node mesh. A wrong derivative turns quadratic
   convergence into stagnation, and stagnation looks exactly like ill
   conditioning.
3. **Check scaling.** Print the size of every term in the residual. If the
   terms differ by more than 6 orders of magnitude, the scaling is wrong.
4. **Then, and only then**, think about damping and the continuation step.

Adding damping before step 1 hides the bug and costs you days.

## Specific traps

**Refining `h_min_y` alone on a MOSFET.** `nmos` builds the silicon axis and
the oxide axis with separate calls to `graded_mesh_1d` and glues them
together, so the `max_ratio` guard that refuses a neighbouring cell jump above
1.5 never sees the Si/SiO2 seam. At the defaults the seam is already smooth,
5e-8 of oxide against 5e-8 of silicon surface, a ratio of 1.000. Halving
`h_min_y` on its own leaves the oxide where it was and pushes that ratio to 2,
then 4, then 8, and nothing complains. On the 1 um device the drain current
moves 5.0, 1.6 and 0.57 percent over the three halvings. That reads like a
converging refinement and is partly a widening discontinuity. Refine
`n_oxide` along with it, and check the seam yourself instead of trusting the
guard.

**devsim dropping a mesh line you asked for, and the contacts on it.** A line
added with `add_2d_mesh_line` isn't guaranteed to show up. If the rows graded
up to it land a node close enough to where it goes, devsim keeps the node and
throws the line away, at the node's coordinate. Ask the MOSFET generator for a
surface spacing of 1.5625e-9 cm under rows graded from the implant depth and
the top silicon row lands at 9.99998817e-5 instead of 1e-4, a gap of
1.18e-10 cm, so the silicon has no node on the interface at all. Everything
defined at y = t_si matches nothing: both surface contacts and the si_ox
interface get created empty, and the only sign of it comes later, from the
first thing that uses one, as `Contact "source" on Device ... does not exist`.
Nothing mentions that the interface went too, and a mesh built that way isn't
a coarser device, it's a different one. `check_mesh_landed` in the generator
refuses it at build time by comparing the top silicon row against `t_si` and
asking devsim for its contact and interface lists. Don't infer mesh structure
from whether a solve converged.

**Picking a mesh spacing for a doping profile instead of deriving it.** The
MOSFET source and drain are a Gaussian of width `implant_shape(...)[0]`, and
the generator's rows through it were a flat 1e-6 cm, which on this process is
1.21 sigma. More than a standard deviation per row puts the metallurgical
junction in the wrong place, and the damage shows up nowhere near the
junction: the subthreshold drain current depends exponentially on the surface
potential, so a sub millivolt error in the barrier is percent of current. The
reference's own halved mesh check moved 4.6 percent at zero gate. Splitting
the halving by axis showed the lateral columns carried 0.036 percent of that
and these rows carried 2.9 of the 2.93. A quarter of a sigma fixes it, and an
eighth changes nothing further. ddsim never had this problem, because it
grades its rows instead of naming a spacing, and
`test_ddsim_resolves_the_implant` keeps it that way.

**Reading a mesh convergence number where the finer mesh is noisier.** A
convergence check assumes the refined answer is the better one. In a MOSFET's
off state it isn't necessarily: the drain current is a small difference of
much bigger fluxes, and a finer mesh has more edges to lose digits across. On
the 1 um device at zero gate and 50 mV, the halved mesh settles to a largest
potential move of 1.1e-16 V and holds its drain current to ten figures over
120 more passes, and still leaves drain and source 0.99 percent apart, while
the shipping mesh balances to 1.65e-4. The two meshes differ by 0.53 percent
there, a quarter of what the finer one knows about itself. That isn't a mesh
error, and counting it as one asks for a refinement that can't help.

**Working in unscaled units.** Silicon at 1e20 cm^-3 next to a depletion
region at 1e-10 cm^-3, with lengths in cm and a permittivity around
1e-12 F/cm. The condition number will be astronomical and you'll blame SciPy.
Scale first.

**`exp(x) - 1` instead of `expm1(x)`.** Catastrophic cancellation near zero,
giving quietly wrong results in exactly the low field regions where you expect
central differencing behaviour.

**The Bernoulli asymmetry.** Electrons: `B(X)*n_right - B(-X)*n_left`. Holes:
`B(X)*p_left - B(-X)*p_right`. The node the B(X) factor attaches to flips
between carriers. Reverse it and the solver converges cleanly to a reversed
current. The derivation is in `docs/02-numerics.md`.

**`V_T * log(N / n_i)` for the contact potential.** Breaks when N is near zero
or negative, which happens in compensated or lightly doped regions. Use
`V_T * asinh(N / (2*n_i))` everywhere. No exceptions.

**Skipping Fermi-Dirac in the source and drain.** At 1e20 cm^-3 Boltzmann
statistics overestimate the carrier density a lot, and the Einstein relation
stops holding. The I-V comes out wrong by a large factor and you won't know
why. Either implement it or write down explicitly that Phase 5 uses Boltzmann
and carries that error.

**The full field magnitude in Caughey-Thomas.** The model needs the field
component along the current, which on a mesh means the field along the edge.
Using |E| lets the vertical MOS gate field wrongly suppress the channel
mobility, and Id comes out far too low.

**Ignoring surface mobility in a MOSFET.** Bulk mobility in an inversion layer
is two to three times too high, and Id will be too high by about the same
factor. Lombardi or something like it isn't optional if you want quantitative
MOSFET results.

**Obtuse triangles in 2D.** Negative Voronoi dual areas, negative
conductances, a broken M-matrix, and negative carrier densities in places that
make no physical sense. Check mesh quality when the mesh is built and refuse
to go on with a bad one. The symptom is localized negative n or p that
survives refinement.

**Under-resolving the Debye length at a junction.** At 1e18 cm^-3, L_D is
about 4 nm. A 10 nm uniform mesh can't resolve that junction and gives you a
smeared, wrong depletion region. Grade the mesh.

**Clamping negative densities.** Tempting, and always wrong. It hides a broken
discretization and produces a solution that satisfies no equation at all. If
densities go negative, find out why.

**Damping the n and p updates.** Damp psi. Take the full n and p updates, or
better, solve in quasi-Fermi potentials where the dynamic range is small.
Damping density updates slows convergence without making it any sturdier.

**Starting from a poor initial guess at high bias.** There's no such thing as
a good initial guess at 1 V forward bias. Continue from equilibrium, always.

**Letting the continuation step grow too fast.** A growth factor of 2
overshoots into non-convergence over and over and wastes more time than it
saves. Use 1.5, and cap it.

**Trying to reuse the symbolic factorization through SciPy.** The sparsity
pattern stays the same across Newton iterations, so reusing the ordering looks
free. With `splu` it isn't: the `permc_spec="NATURAL"` workaround costs about
six times more fill, as docs/02-numerics.md measures. Reuse the COO to CSC
pattern work instead, and factorize fresh with COLAMD.

**Building the frontend early.** A visualization of a wrong solution is a
liability. It creates false confidence, because the plots look convincing.
Solver first, validated, then the UI.

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
| SS below 59.5 mV/dec | Definitely a bug. Physically impossible at 300K |
| Id too high by ~3x in MOSFET | Missing surface mobility model |
| Id too low, saturates early | Field dependent mobility using \|E\| instead of E_parallel |
| I-V off by orders of magnitude | n_i mismatch, or Boltzmann in a degenerate region |
| Gummel cycles pile up above 0.6 V | Expected. Gummel slows without bound at high injection. Switch to Newton |
| Depletion region smeared | Mesh under-resolves the local Debye length |
| Results differ from DEVSIM by 2x | Model settings not matched between the tools |
| devsim: `Contact "x" does not exist` after a mesh built fine | A requested mesh line got merged into a neighbouring node, so everything defined on it is empty. Check the boundary row landed |
| Mesh refinement moves the off state and nothing else | Rows too coarse for the implant sigma, not the surface spacing |
| Halving the mesh moves less than the terminals fail to cancel | Not a mesh error. The off state current is a flux cancellation, and the finer mesh cancels worse |

## When genuinely stuck

Shrink it to the smallest failing case: a 20 node uniform 1D mesh, one
junction, constant mobility, recombination off, zero bias. That setup has a
closed form answer. If it fails there, the bug is structural, and every hour
spent on the full MOSFET is wasted.
