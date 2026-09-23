---
title: A failed job
summary: What it means when a sweep ends as failed, how that's different from a refusal or a stall, and what to try next.
docs: 02-numerics.md#Convergence criteria
---

## In plain words

A job fails when the work stops with an error before the sweep can hand back
a curve. The status at the top reads `failed`, and the red message under the
solve button gives the reason, starting with the kind of error. For a Newton
solve it also names the equation family with the biggest residual and the one
with the biggest update in the last attempt, which tells you whether Poisson
or a carrier equation got stuck.

Anything drawn while it ran stays on screen, but a failed job hands back no
curve. The point slider stays off and there's no profile to look at.

Two other outcomes look similar but aren't. A refusal happens before any job
starts: a mistyped knob, a contact the device doesn't have, or a geometry the
builder won't accept. The status then reads `refused`. A stall is a job that
finished with part of a curve, and it has its own explanation.

In practice a failure nearly always means the very first point couldn't be
solved. Things to try: bring `start` back toward 0, raise `max_iterations`,
and check the mesh is fine enough for the doping.

## In more depth

Every later point in a sweep is reached by continuation, which halves a step
that fails and keeps the points it has, so trouble there ends as a stall, not
a failure. The first point has nothing to continue from.

- On `iv`, the point at `start` is a Gummel solve from the equilibrium guess.
  That guess is itself a Poisson solve, and on a 1e16 diode it can't be built
  above about 1.3 V, so a high `start` fails before any cycle runs. Below
  that, a Gummel solve that hasn't converged within `max_iterations` fails the
  job too.
- On `transfer`, the first point is ramped in as a fraction of every applied
  bias, and the final solve always happens at the full bias. If that one
  doesn't converge, the job fails instead of returning an answer at a
  fraction nobody asked for.

So the bias step that causes a failure is the jump to the first point, which
`step` doesn't control. A large `step` later in the sweep costs retries, not
the job.

A solve only counts as converged when both its update and its residual are
inside their thresholds: the Gummel update below `update_tol`, or for Newton
each equation family's residual below 1e-10 of that family's own term size and
its update below 1e-10. Running out of iterations first is a failure. So is a
residual frozen to the last bit while the update is already small. The solve
stops early there, but it still reports failure, because spending the rest of
the budget can't change the answer.

A mesh that under-resolves the doping is a quieter problem.
docs/02-numerics.md asks for spacing at a junction below half the local Debye
length, which is 40.9 nm at 1e16 cm^-3 and 4.09 nm at 1e18. A mesh coarser
than that might fail to converge, but it's just as likely to converge to a
smeared answer. A success doesn't prove the mesh was fine enough.
