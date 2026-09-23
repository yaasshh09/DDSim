---
title: Reading the residual plot
summary: What each line and dot on the residual plot means, what a healthy solve looks like, and what a stall looks like.
docs: 02-numerics.md#Convergence criteria; 02-numerics.md#Full Newton (Phase 3); 02-numerics.md#Gummel iteration (Phase 2); 02-numerics.md#Bias continuation
---

## In plain words

The residual plot is the solver thinking out loud. Every time a solve takes a
step it adds a point showing how far the equations still are from being
satisfied. The horizontal axis just counts those points in order across the
whole job, so a sweep over many biases draws one solve after another, side by
side. The vertical axis is logarithmic by default, because the interesting
part spans many powers of ten. All the numbers are scaled and have no units.

A healthy sweep looks like a row of steep slides. Each bias point starts well
above where it'll finish and drops a few powers of ten per step over a
handful of steps, with the drops getting bigger as it closes in. Then the
next point starts again higher up. On the diode sweep the purple line isn't a
residual. It's how much each Gummel cycle changed things, and it falls more
gently, in a straight line, because Gummel converges more slowly.

A stall looks different: a line that goes flat and stays flat, a note that
keeps saying "step limited", or dark dots piling up, each one a bias step
that failed and got retried smaller. When a Newton bias step gets rejected,
the note names which equation had the biggest residual and which had the
biggest update. That's where to start looking for the cause.

## In more depth

What each mark is:

- **Blue, green and red** are the residual of the coupled Newton solve split
  by equation family: Poisson (which fixes psi), electron continuity (which
  fixes n) and hole continuity (which fixes p). Each point is the largest,
  over every row of that family, of the residual divided by that row's own
  terms, so it's a relative size. The solve is judged on the largest of the
  three against about $10^{-10}$. There's one point per evaluation, including
  iteration 0 at the starting guess.
- **Blue alone**, on a C-V sweep, is the equilibrium Poisson residual
  $\max|F|$. It isn't split and isn't divided by row terms, so it's judged
  against $10^{-12} + 10^{-10}\,Q$, with $Q$ the doping charge in the largest
  dual cell, not against $10^{-10}$.
- **Purple** is the Gummel cycle update, not a residual: the largest of
  $\max|\Delta\psi|$ and the relative density changes over the cycle, judged
  against `update_tol`.
- **Dark dots** are rejected continuation attempts. Each one sits on the last
  point of the failed attempt, at the height of that point's plotted value:
  the largest family residual for Newton, the cycle update for Gummel.

The axis spans every family so the smaller two don't fall off a log plot. The
equilibrium solve that builds a cold starting guess reports nothing, since
its residual belongs to a different system. The low field prelude of a
velocity saturation solve and every pass of the surface mobility fixed point
do report, because their iterations count.

On a log axis the two convergence rates have different shapes. Gummel's is
linear, with the update shrinking by a roughly constant factor per cycle, so
it draws a straight line whose slope flattens as injection rises. Newton's is
quadratic, with the exponent roughly doubling each step, so it draws a curve
bending steeply down until it hits the roundoff floor.

The plot draws Newton's residuals, not its updates, and convergence needs
both, measured per family. That's why the note on a rejected attempt names
the family with the largest residual and the family with the largest update
separately. The page doesn't know the tolerances, and naming only the
residual can point at the wrong test. On one stalled MOSFET every residual
sat near $10^{-14}$, comfortably converged on the plot, while the electron
update was stuck at $1.8 \times 10^{-10}$, just above its $10^{-10}$
threshold. A line frozen flat while the update is already inside tolerance is
a residual on its arithmetic floor, and the solve stops early after four
identical evaluations instead of burning its budget.
