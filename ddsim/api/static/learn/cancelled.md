---
title: A cancelled job
summary: What the cancel button does, when the solver actually stops, and why a cancelled sweep leaves no result behind.
docs: 02-numerics.md#Bias continuation
---

## In plain words

The cancel button asks the running sweep to stop. The status first reads
`cancelling`, then `cancelled` once the solver notices. It notices the next
time it reports an iteration, so on a slow solve there can be a short wait
between the two.

Whatever was already drawn stays on screen: the residual lines and any curve
points that had landed. But a cancelled job hands back no result. Nothing
half finished gets passed off as the answer, so the curve isn't replaced by a
finished one, the point slider stays off, and there's no profile to look at.
If you want the points, let the sweep finish, or give it a shorter voltage
list.

If the sweep had already finished by the time your click arrived, nothing
gets stopped and the status reads `already finished`. The curve it found is
kept, since throwing away a result that exists would be worse than ignoring
a late click.

## In more depth

A solve is a tight numerical loop that never looks up, except to report. So
cancelling piggybacks on the report. Every frame the solver sends (a Newton
iteration, a Gummel cycle, a continuation attempt or a finished point) goes
through one function that first checks whether the job has been asked to
stop. If it has, that function raises inside the solver, and the exception
unwinds the whole continuation ladder at once. The job is marked cancelled
and its result is left empty on purpose: the sweep stopped somewhere nobody
chose, and its partial state is only what the page had already been sent.

The catch is the gaps between reports. A solve that isn't reporting can't be
cancelled until it reports again. The sparse factorization inside a Newton
iteration, the extra linear solve that gives a C-V point its capacitance, and
the equilibrium Poisson solve that builds the first guess for an `iv` or
`transfer` sweep all run without reporting. A click during one of them waits
for it to end.

Shutting the server down cancels every running job the same way, and waits
up to 30 seconds for them to reach their next report.
