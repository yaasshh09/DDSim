---
title: A failed job
summary: What it means when a sweep ends as failed, how that differs from a refusal or a stall, and what to try next.
docs: 02-numerics.md#Convergence criteria
---

## In plain words

A job fails when the work stops with an error before the sweep can hand back a
curve. The status at the top reads `failed`, and the red message under the
solve button gives the reason, starting with the kind of error. For a Newton
solve it also names the equation family with the largest residual and the one
with the largest update in the last attempt, which tells you whether Poisson or
a carrier equation was stuck.

Anything drawn while it ran stays on screen, but a failed job hands back no
curve, so the point slider stays off and there is no profile to look at.

Two other outcomes look similar and are not. A refusal happens before any job
starts: a mistyped knob, a contact the device does not have, or a geometry the
builder will not accept, and the status reads `refused`. A stall is a job that
finished with part of a curve, and it has its own explanation.

In practice a failure almost always means the very first point could not be
solved. Things to try: bring `start` back towards 0, raise `max_iterations`,
and check the mesh is fine enough for the doping.

## In more depth

Every later point in a sweep is reached by continuation, which halves a step
that fails and keeps the points it has, so trouble there ends as a stall rather
than a failure. The first point has nothing to continue from.

- On `iv`, the point at `start` is a Gummel solve from the equilibrium guess.
  That guess is itself a Poisson solve, and on a 1e16 diode it cannot be built
  above about 1.3 V, so a high `start` fails before any cycle runs. Below that,
  a Gummel solve that has not converged within `max_iterations` fails the job
  too.
- On `transfer`, the first point is ramped in as a fraction of every applied
  bias, and the final solve is always taken at the full bias. If that one does
  not converge, the job fails rather than returning an answer at a fraction
  nobody asked for.

So the bias step that causes a failure is the jump to the first point, which
`step` does not control. A large `step` later in the sweep costs retries, not
the job.

A solve counts as converged only when both its update and its residual are
inside their thresholds: the Gummel update below `update_tol`, or for Newton
each equation family's residual below 1e-10 of that family's own term size and
its update below 1e-10. Running
out of iterations first is a failure. So is a residual frozen to the last bit
while the update is already small: the solve stops early there, but it still
reports failure, because spending the rest of the budget cannot change the
answer.

A mesh that under-resolves the doping is a quieter problem. docs/02-numerics.md
asks for spacing at a junction below half the local Debye length, which is
40.9 nm at 1e16 cm^-3 and 4.09 nm at 1e18. A mesh coarser than that may fail to
converge, but it is just as likely to converge to a smeared answer, so a
success does not prove the mesh was fine enough.
