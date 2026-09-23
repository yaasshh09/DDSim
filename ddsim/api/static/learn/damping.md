---
title: Damping and step limiting
summary: Shortening a Newton step that would overshoot, what "step limited" means, and why it can't fix a wrong answer.
docs: 02-numerics.md#Full Newton (Phase 3); 05-pitfalls.md#Debugging order; 05-pitfalls.md#Specific traps
---

## In plain words

A Newton step is a guess about where the answer is, made by treating curved
equations as straight lines. Carrier densities depend exponentially on the
potential: every extra 60 mV or so multiplies a density by ten. So when the
guess is way off, Newton can ask to move the potential by whole volts and the
densities by dozens of powers of ten. The next guess lands somewhere far
worse than where it started, sometimes at numbers too big for the computer to
store at all.

Damping means only taking part of that step. This simulator does it the
simplest way. If a step would move the potential anywhere in the device by
more than a fixed amount, the potential change gets shrunk until it doesn't.
The electron and hole changes are taken in full. When that happens, the note
beside the residual plot says "step limited". A few limited steps at the
start of a solve are normal. A solve that's limited step after step isn't
converging, it's crawling, and the fix is usually a smaller bias step, not
more damping. Damping can make a solve slower or steadier. It can't turn a
wrong equation into a right one. If damping tames a sign error, you still get
a wrong answer.

## In more depth

The rule caps the potential update at $5 V_T$ per step, which is 5.0 in
scaled units and about 129 mV. There's no line search and no Bank-Rose
damping, just this limit.

In the coupled solve, if $\max|\Delta\psi| > 5$, the $\psi$ part of the
update gets multiplied by the single factor $5 / \max|\Delta\psi|$, and the
$n$ and $p$ parts are left alone. Scaling the $\psi$ sub-vector by one
factor, instead of clipping each entry separately, keeps the potential update
pointing where Newton wanted it to. Scaling the whole vector by one factor
wouldn't work here. The densities are around six decades bigger than $\psi$
in scaled units, so the density update would set that factor on its own and
$\psi$ would barely move. The docs and the pitfalls list both say to damp
$\psi$ and take the densities in full, because damping the densities slows
convergence without making it any sturdier. The equilibrium Poisson solve
behind the C-V sweep only has $\psi$, so there the whole update is scaled by
one factor to the same cap.

Each Newton frame carries a damping factor alongside the limited flag: the
smallest ratio of step taken to step requested, over every component Newton
asked to move. A full step reads 1.0. It's measured per component because a
ratio of the two maxima would be set by the density update and read exactly
1.0 on every limited step of a MOSFET solve. The page only shows the flag.

Why not just damp harder? On a $10^{15}$ cm$^{-3}$ diode at 1.2 V on 41
nodes, Newton from the Poisson guess diverges: thirty steps, twenty eight of
them limited. Capping at $2 V_T$ instead of 5 does converge, in forty seven
steps with forty limited. Capping at $1 V_T$ or $0.5 V_T$ doesn't converge
within sixty. The same device reaches 1.2 V in seven solves through
continuation, or in eight Newton steps after two Gummel cycles. When a solve
diverges, the debugging order is fixed: check signs, then the Jacobian, then
the scaling, and only then the damping and the continuation step.
