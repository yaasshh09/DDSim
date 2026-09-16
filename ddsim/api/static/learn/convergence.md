---
title: What converged means
summary: The tests a solve has to pass before its answer is accepted, and the two knobs that set how hard it tries.
docs: 02-numerics.md#Convergence criteria; 02-numerics.md#Current conservation; 07-decisions.md#Physics decisions log
---

## In plain words

An iterative solver never lands exactly on the answer. It gets closer with
each step, and at some point it has to decide that it is close enough. Doing
that honestly is harder than it sounds, because there are two different ways
to be fooled. A solver can stop changing because it has arrived, or because
it is stuck. And the equations can look satisfied overall while one region of
the device is still far off. So a Newton solve here is only called converged
when two things are true at the same step: the unknowns have stopped moving,
and every equation at every node is satisfied to a tight tolerance measured
against the size of its own terms. Passing one test and not the other is a
failure, and it is reported as one. A Gummel solve is judged on how much a
whole cycle changes things.

Two knobs control this. `max_iterations` is the budget: how many Newton
iterations, or Gummel cycles on the diode sweep, one bias point may use before
it counts as a failure. On the diode and transfer sweeps continuation then
retries with a smaller step, and on a C-V sweep the curve stops there. The
defaults are 200 Gummel cycles on the diode sweep, 30 Newton iterations on the
transfer curve and 50 on the C-V sweep. `update_tol` is offered on the diode
sweep only. It is how small the change over a whole Gummel cycle has to be,
one part in a hundred million by default, in scaled units. The tolerances
inside Newton are fixed rather than offered. A job can end as done with a
curve that stopped short, and then the message says where it stalled. Every
point the curve does hold passed these tests.

## In more depth

A coupled Newton solve, the transfer curve's, needs both of these in the same
iteration.

**The update.** Measured per equation family on the damped step actually
taken, in scaled units,

$$\max\left(\max|\Delta\psi|,\ \max\frac{|\Delta n|}{n + n_i},\ \max\frac{|\Delta p|}{p + n_i}\right) < 10^{-10}$$

The potential is measured absolutely, since it is already in thermal voltages
and a change in it is a relative change in everything it drives. The densities
are measured relative to themselves, floored at $n_i$, which is 1 in scaled
units. A pure relative change would be ruled by nodes holding $10^{-15}$
carriers that carry no charge worth converging, and an absolute one by the
majority carrier. The raw $\max|\Delta x|$ cannot work at all: with $n$ at
$10^6$ in scaled units its last bit is $10^{-10}$, and on a $10^{16}$ diode every
solve above 0.2 V reported failure while sitting on the exact answer.

**The residual.** Measured row by row against that row's own terms. For each
row the residual is divided by the largest single term it was assembled from,
a charge for Poisson and a flux for continuity, re-measured at every iterate,
and the largest result in each family has to be below
$10^{-12} + 10^{-10}$. A row whose terms have fallen below machine epsilon
times the largest in its family is skipped, since dividing roundoff by
something that small manufactures a number rather than measuring one. Per row
matters because terms span decades within one family. On a $10^{17}$ /
$10^{20}$ junction the electron flux terms span 11.5 decades, and a measure
using one number per family reported a cold solve at 0.4 V converged after
zero Newton steps, still on the equilibrium guess, giving a current of
$1.2 \times 10^{-10}$ against the $7.4 \times 10^{-4}$ Gummel finds. The per
row measure reads $0.97$ at that guess and $7.1 \times 10^{-15}$ at the answer.

The equilibrium Poisson solve behind a C-V point uses the same two criteria on
one unknown: $\max|\Delta\psi| < 10^{-10}$, and $\max|F|$ below
$10^{-12} + 10^{-10}\,Q$, with $Q$ the doping charge in the largest dual cell. That
scale is raised only when it would sink under the roundoff floor of the flux
differences, which happens below about $10^{13}$ cm$^{-3}$. A Gummel cycle
converges when its largest block update, the same three measures, is below
`update_tol`, and its Poisson block must pass both criteria inside every
cycle.

One guard ends a solve early: if the residual has been identical to the last
bit for four evaluations while the update is already inside tolerance, the
residual is on its arithmetic floor and no further step can move it. That is
still reported as a failure, just sooner. The docs also list current
continuity, $J_n + J_p$ constant across a 1D device with recombination off, as
a third check. No solve tests it while running. It is asserted by the test
suite, where it is the strongest single check the project has.
