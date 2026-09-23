---
title: What converged means
summary: The tests a solve has to pass before its answer counts, and the two knobs that set how hard it tries.
docs: 02-numerics.md#Convergence criteria; 02-numerics.md#Current conservation; 07-decisions.md#Physics decisions log
---

## In plain words

An iterative solver never lands exactly on the answer. It creeps closer with
every step, and at some point it has to decide it's close enough. Making that
call honestly is harder than it sounds, because there are two ways to get
fooled. A solver can stop changing because it's arrived, or because it's
stuck. And the equations can look fine overall while one corner of the
device is still way off.

So here a Newton solve only counts as converged when two things are true at
the same step. The unknowns have stopped moving. And every equation at every
node is satisfied to a tight tolerance, measured against the size of its own
terms. Pass one test and fail the other, and it's a failure, reported as one.
A Gummel solve gets judged on how much a whole cycle changes things.

Two knobs control this. `max_iterations` is the budget: how many Newton
iterations (or Gummel cycles, on the diode sweep) one bias point gets before
it counts as a failure. On the diode and transfer sweeps, continuation then
retries with a smaller step. On a C-V sweep the curve stops there. The
defaults are 200 Gummel cycles on the diode sweep, 30 Newton iterations on
the transfer curve and 50 on the C-V sweep. `update_tol` is only on the diode
sweep. It's how small the change over a whole Gummel cycle has to be, one
part in a hundred million by default, in scaled units. The tolerances inside
Newton are fixed, not offered as knobs. A job can finish as done with a curve
that stopped short, and then the message tells you where it stalled. Every
point the curve does have passed these tests.

## In more depth

A coupled Newton solve, like the transfer curve's, needs both of these in the
same iteration.

**The update.** Measured per equation family on the damped step actually
taken, in scaled units,

$$\max\left(\max|\Delta\psi|,\ \max\frac{|\Delta n|}{n + n_i},\ \max\frac{|\Delta p|}{p + n_i}\right) < 10^{-10}$$

The potential is measured absolutely, since it's already in thermal voltages
and a change in it is a relative change in everything it drives. The
densities are measured relative to themselves, floored at $n_i$, which is 1
in scaled units. A pure relative change would be ruled by nodes holding
$10^{-15}$ carriers that carry no charge worth converging, and an absolute
one by the majority carrier. The raw $\max|\Delta x|$ can't work at all. With
$n$ at $10^6$ in scaled units its last bit is $10^{-10}$, and on a $10^{16}$
diode every solve above 0.2 V reported failure while sitting right on the
exact answer.

**The residual.** Measured row by row against that row's own terms. Each
row's residual is divided by the largest single term it was built from, a
charge for Poisson and a flux for continuity, re-measured at every iterate.
The largest result in each family has to be below $10^{-12} + 10^{-10}$. A
row whose terms have dropped below machine epsilon times the largest in its
family gets skipped, since dividing roundoff by something that small makes up
a number instead of measuring one. Per row matters because terms span
decades within one family. On a $10^{17}$ / $10^{20}$ junction the electron
flux terms span 11.5 decades. A measure using one number per family called a
cold solve at 0.4 V converged after zero Newton steps, still sitting on the
equilibrium guess, with a current of $1.2 \times 10^{-10}$ against the
$7.4 \times 10^{-4}$ Gummel finds. The per row measure reads $0.97$ at that
guess and $7.1 \times 10^{-15}$ at the answer.

The equilibrium Poisson solve behind a C-V point uses the same two criteria
on one unknown: $\max|\Delta\psi| < 10^{-10}$, and $\max|F|$ below
$10^{-12} + 10^{-10}\,Q$, with $Q$ the doping charge in the largest dual
cell. That scale only gets raised when it would sink under the roundoff floor
of the flux differences, which happens below about $10^{13}$ cm$^{-3}$. A
Gummel cycle converges when its largest block update (the same three
measures) is below `update_tol`, and its Poisson block has to pass both
criteria inside every cycle.

One guard ends a solve early. If the residual has stayed identical to the
last bit for four evaluations while the update is already inside tolerance,
the residual is on its arithmetic floor and no further step can move it.
That's still reported as a failure, just sooner. The docs also list current
continuity ($J_n + J_p$ constant across a 1D device with recombination off)
as a third check. No solve tests it while running. The test suite asserts it,
and there it's the strongest single check the project has.
