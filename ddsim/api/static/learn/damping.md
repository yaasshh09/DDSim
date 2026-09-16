---
title: Damping and step limiting
summary: Shortening a Newton step that would overshoot, what "step limited" means, and why it cannot fix a wrong answer.
docs: 02-numerics.md#Full Newton (Phase 3); 05-pitfalls.md#Debugging order; 05-pitfalls.md#Specific traps
---

## In plain words

A Newton step is a guess about where the answer is, made by treating curved
equations as straight lines. Carrier densities depend exponentially on the
potential: every extra 60 mV or so multiplies a density by ten. So when the
guess is far off, the step Newton asks for can move the potential by volts,
the densities by dozens of decades, and the next iterate lands somewhere
wildly worse than where it started, sometimes too large for the computer to
represent at all.

Damping means taking only part of that step. This simulator does it in the
simplest way: if a step would move the potential anywhere in the device by
more than a fixed amount, the potential change is shrunk until it does not.
The changes to the electron and hole densities are taken in full. When that
happens the note beside the residual plot says "step limited". A few limited
steps at the start of a solve are normal. A solve that is limited step after
step is not converging, it is crawling, and the cure is usually a smaller bias
step rather than more damping. Damping can make a solve slower or steadier. It
cannot turn a wrong equation into a right one, and a sign error that damping
tames still gives a wrong answer.

## In more depth

The rule is a cap on the potential update of $5 V_T$ per step, which is 5.0 in
scaled units and about 129 mV. There is no line search and no Bank-Rose
damping, only this limit.

In the coupled solve, if $\max|\Delta\psi| > 5$, the $\psi$ part of the update
is multiplied by the single factor $5 / \max|\Delta\psi|$, and the $n$ and $p$
parts are left untouched. Scaling the $\psi$ sub-vector by one factor, rather
than clipping each entry separately, keeps the potential update pointing in
the direction Newton chose. Scaling the whole vector by one factor would not
work here: the densities are around six decades larger than $\psi$ in scaled
units, so that factor would be set entirely by the density update and $\psi$
would barely move. The docs and the pitfalls list both say to damp $\psi$ and
take the densities in full, because damping the densities slows convergence
without making it any more robust. The equilibrium Poisson solve behind the
C-V sweep has only $\psi$, so there the whole update is scaled by one factor to
the same cap.

Each Newton frame carries a damping factor as well as the limited flag: the
smallest ratio of the step taken to the step requested over every component
that Newton asked to move. It is 1.0 for a full step. It is measured per
component because a ratio of the two maxima would be set by the density
update and read exactly 1.0 on every limited step of a MOSFET solve. The page shows only the
flag.

Why not simply damp harder? Measured on a $10^{15}$ cm$^{-3}$ diode at 1.2 V on
41 nodes, Newton from the Poisson guess diverges: thirty steps, twenty
eight of them limited. Capping at $2 V_T$ instead of 5 does converge, in forty
seven steps with forty limited. Capping at $1 V_T$ or $0.5 V_T$ does not
converge within sixty. The same device reaches 1.2 V in seven solves through
continuation, or in eight Newton steps after two Gummel cycles. When a solve
diverges the debugging order is fixed: check signs, then the Jacobian, then
the scaling, and only then the damping and the continuation step.
