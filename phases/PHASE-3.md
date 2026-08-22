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
6. Auger recombination **(done, with SumOfRecombination so mechanisms compose)**
7. Doping-dependent mobility **(done, Arora. Off by default: its N -> 0 limit
   is 1340 against the tabulated 1417, so switching a device to it moves the
   current by five percent even where the model should do nothing.)**
8. ~~Symbolic factorization reuse across Newton steps~~ **Dropped.** Phase 0
   measured this. scipy exposes no symbolic and numeric split, and the standard
   workaround gives 6.1x fill and a 46x slowdown in 2D. See PROGRESS.md.

## Acceptance criteria

- **Every Jacobian block** matches complex-step differentiation to 1e-10 on a
  20 node mesh. All nine blocks, individually tested. Non-negotiable.
- Newton converges quadratically. Residual history shows the characteristic drop.
- **Converges at 1.0 V forward bias**, high injection. This is the headline
  result of the phase. **The "where Gummel failed" clause is wrong and was
  removed.** Gummel does not fail at 1.0 V or at 2.0 V; it degrades without
  bound, taking 46 cycles at 1.0 V and 466 at 2.0 V against Newton's 4. The
  tested claim is that Newton gets there in under a third of the cycles and
  gets there cold, from the Poisson guess, with no continuation. See the
  deviations table in PROGRESS.md for the measurement.
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
