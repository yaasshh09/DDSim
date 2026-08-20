# Phase 3: Full Newton and robust continuation

**Target: 2 weeks, will probably take 3. Docs: 02-numerics (full Newton), 05-pitfalls.**

The hardest phase and the one with the most actual value. Budget for overrun.
This is where the project stops being a homework exercise.

## Scope

1. Full 3N x 3N Jacobian for (psi, n, p), assembled simultaneously
2. Node-interleaved ordering (psi_0, n_0, p_0, psi_1, ...) for fill reduction
3. Jacobian verification harness against complex-step differentiation
4. Damping: Bank-Rose or step limiting on psi at 5*V_T per iteration
5. Hybrid driver: Gummel for 3 to 5 iterations, then switch to Newton, with
   fallback to Gummel on Newton stall
6. Auger recombination
7. Doping-dependent mobility (Arora or Masetti)
8. Symbolic factorization reuse across Newton steps

## Acceptance criteria

- **Every Jacobian block** matches complex-step differentiation to 1e-10 on a
  20 node mesh. All nine blocks, individually tested. Non-negotiable.
- Newton converges quadratically. Residual history shows the characteristic drop.
- **Converges at 1.0 V forward bias**, high injection, where Gummel failed.
  This is the headline result of the phase.
- Gummel and Newton produce identical solutions to solver tolerance at every
  bias where both converge
- Continuation from 0 to 1 V in under 40 total solves
- No negative carrier densities at any bias
- All Phase 1 and 2 tests still green
- Diode #1 and #2 in the DEVSIM regression set pass at stated tolerance

## Do not

- Add impact ionization. It is a positive feedback term and will destroy
  convergence you have not yet earned.
- Move to 2D.

## If it will not converge

Follow the debugging order in `docs/05-pitfalls.md`. Signs, then Jacobian, then
scaling, then damping. Do not add damping first. Reduce to a 20 node uniform mesh
with constant mobility and recombination off; that case has a closed-form answer.

## Definition of done

Converged solution at 1 V forward bias, Jacobian verification test in CI, and a
`PROGRESS.md` note on what actually broke and how you found it. That note is
worth writing carefully. It is the most interesting engineering content the
project will generate.
