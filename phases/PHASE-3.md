# Phase 3: Full Newton and robust continuation

**Target: 2 weeks, will probably take 3. Docs: 02-numerics (full Newton), 05-pitfalls.**

The hardest phase, and the one with the most real value, so budget for it to
overrun. This is where the project stops being a homework exercise.

## Scope

1. The full 3N x 3N Jacobian for (psi, n, p), assembled all at once
2. Node interleaved ordering (psi_0, n_0, p_0, psi_1, ...) to cut fill
3. A Jacobian verification harness against complex step differentiation
4. Damping: Bank-Rose, or step limiting on psi at 5*V_T per iteration
5. A hybrid driver: Gummel for 3 to 5 iterations, then Newton, falling back to
   Gummel if Newton stalls
6. Auger recombination **(done, with SumOfRecombination so mechanisms
   compose)**
7. Doping dependent mobility **(done, Arora. Off by default: its N -> 0 limit
   is 1340 against the tabulated 1417, so switching a device to it moves the
   current by five percent even where the model should do nothing.)**
8. ~~Symbolic factorization reuse across Newton steps~~ **Dropped.** Phase 0
   measured this. SciPy exposes no symbolic and numeric split, and the
   standard workaround costs 6.1x the fill and a 46x slowdown in 2D. See
   docs/07-decisions.md.

## Acceptance criteria

- **Every Jacobian block** matches complex step differentiation to 1e-10 on a
  20 node mesh. All nine blocks, each tested on its own. Not optional.
- Newton converges quadratically, and the residual history shows the
  characteristic drop.
- **Converges at 1.0 V forward bias**, in high injection. That's the headline
  result of the phase. **The "where Gummel failed" clause was wrong and has
  been removed.** Gummel doesn't fail at 1.0 V or at 2.0 V; it just keeps
  getting slower, taking 46 cycles at 1.0 V and 466 at 2.0 V against Newton's
  4. What's tested is that Newton gets there in under a third of the cycles,
  and gets there cold, from the Poisson guess, with no continuation. The
  measurement is in the deviations table in docs/07-decisions.md.
- Gummel and Newton give identical solutions to solver tolerance at every bias
  where both converge
- Continuation from 0 to 1 V in under 40 solves total
- No negative carrier densities at any bias
- Every Phase 1 and 2 test still green
- Diodes #1 and #2 in the DEVSIM regression set pass at their stated
  tolerance

## Do not

- Add impact ionization. It's a positive feedback term and it'll wreck
  convergence you haven't earned yet.
- Move to 2D.

## If it will not converge

Follow the debugging order in `docs/05-pitfalls.md`: signs, then the
Jacobian, then scaling, then damping. Don't add damping first. Shrink it to a
20 node uniform mesh with constant mobility and recombination off, since that
case has a closed form answer.

## Definition of done

A converged solution at 1 V forward bias, the Jacobian verification test in
CI, and a written note on what actually broke and how it got found. That note
is worth writing carefully. It's the most interesting engineering content the
project produces.
