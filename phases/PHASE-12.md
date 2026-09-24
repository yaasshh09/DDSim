# Phase 12: Error estimates and adaptive refinement

**Target: 2 to 3 weeks. Docs: 02-numerics (Mesh, Error estimation), 04-validation (Convergence order studies).**

Neither tool attaches an error bar to its answers today. After this phase,
every current, capacitance and threshold DDSim reports comes with an
estimate of its discretization error, and the mesh refines itself until that
estimate is below a tolerance you set. That's the error control axis on the
scoreboard, and DEVSIM has nothing built in for it.

## Scope

1. **Goal-oriented estimates by the dual weighted residual method.** The goal
   Q is the quantity the user asked for: a terminal current, a charge, or
   Vth. The steps:
   - Solve the adjoint `J^T z = dQ/dx` at the converged state. It's one more
     solve with the factorization already in hand (`trans='T'`).
   - Prolong the solution and z onto the mesh refined once everywhere.
   - Assemble the residual there once, with no solve.
   - The estimate is eta = -z_fine . F_fine(u_prolonged).

   Its local contributions per cell are the refinement indicator.
2. **Refinement on the tensor mesh.** Grid lines are inserted where the
   indicator flags a cell. On a tensor mesh a new line runs across the whole
   device, so refinement costs more nodes than it would on a triangle mesh.
   That ceiling gets a `ponytail:` comment, and Phase 15 lifts it.
3. **An adaptive loop**: solve, estimate, refine, and repeat until
   |eta| < tol or the node budget runs out. It never stops silently at the
   budget: the result says it didn't reach tol.
4. Error bars carried through `extract/` into the API frames, and drawn on
   the plots.
5. The scoreboard's error control axis is recomputed.

## Analytic limits, write these tests first

- **Effectivity.** The ratio of estimated to true error must sit in
  [0.5, 2] on benchmarks 1 to 8. True error is measured against a Richardson
  extrapolation from three uniform refinements, which Phase 8 already builds.
  This is the test the whole phase rests on.
- **Exactness on a linear goal.** For a problem whose discrete solution is
  exact, like a uniform resistor at low bias, the estimate is zero to
  roundoff.
- **The estimate shrinks at the observed order.** Under uniform refinement,
  eta falls at the rate the order studies already measure: first order at
  junctions, second order in smooth regions.
- **Adaptive beats hand-graded.** Reaching each benchmark's tolerance takes
  fewer nodes than the hand-graded reference mesh on at least 6 of the 8
  benchmarks. If it doesn't, that's reported, not hidden.
- **Refinement that changes nothing is a bug.** If refining a flagged cell
  moves the goal by exactly zero, the indicator isn't connected to the goal.
  That exact case exists as a test.

## Acceptance criteria

- All of the above, shown failing first.
- Every number in the README results tables shows its error estimate next
  to it.
- The error control axis on the scoreboard reads above zero for DDSim, with
  the share of results carrying an estimate.

## Honest limits

A DWR estimate is only as good as the linearization. Near breakdown
(Phase 14) and in strong high injection, effectivity can drift. Those cases
get their own effectivity test when they arrive, not an assumption carried
over from here.
