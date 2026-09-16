---
title: Newton's method
summary: Solving the potential, the electrons and the holes all at once, and why its residual falls so steeply once it is close.
docs: 02-numerics.md#Full Newton (Phase 3); 02-numerics.md#Linear algebra; 02-numerics.md#Convergence criteria; 07-decisions.md#Known deviations from reference
---

## In plain words

Newton's method solves the three equations together instead of in turns. At
the current guess it asks, for every equation at every node, how far that
equation is from being satisfied, which is the residual. It also works out how
each residual would change if each unknown were nudged a little. Treating that
relationship as a straight line, it jumps to the point where every residual
would be zero at once. The equations are not straight lines, so the jump lands
near the answer rather than on it, and the method repeats from there.

Far from the answer that jump can be wild, which is why the step is limited
and why the bias is walked up gradually. Close to the answer it is spectacular:
each step roughly squares the error, so the number of correct digits about
doubles every iteration. The MOSFET transfer curve runs Newton at every point.
On the residual plot you see three lines per iteration, one for each equation:
blue for Poisson, which fixes psi, green for electron continuity, which fixes
n, and red for hole continuity, which fixes p. A C-V sweep also runs Newton,
but on Poisson alone, so it draws a single blue line.

## In more depth

With the unknowns gathered into one vector $x$ and the residual of every
equation into $F(x)$, each Newton step solves the linear system

$$J\,\Delta x = -F, \qquad J = \frac{\partial F}{\partial x}$$

and moves to $x + \Delta x$, after the damping rule has had its say. There are
three unknowns per node, $\psi$, $n$ and $p$, so a mesh of $N$ nodes gives a
$3N \times 3N$ system. The unknowns are interleaved by node,
$(\psi_0, n_0, p_0, \psi_1, \dots)$, rather than blocked by variable, which
keeps coupled entries close together and cuts fill in the factorization. At
each node the Jacobian has a $3 \times 3$ block,

$$\begin{pmatrix} \partial F_\psi/\partial\psi & \partial F_\psi/\partial n & \partial F_\psi/\partial p \\ \partial F_n/\partial\psi & \partial F_n/\partial n & \partial F_n/\partial p \\ \partial F_p/\partial\psi & \partial F_p/\partial n & \partial F_p/\partial p \end{pmatrix}$$

plus couplings to its neighbours through the Scharfetter-Gummel fluxes. Every
block is checked against a finite difference or complex step Jacobian in the
test suite, because one wrong derivative turns quadratic convergence into a
stall that looks exactly like bad conditioning. Before the solve each row is
divided by one number per equation family, the largest term anywhere in that
family. That is a diagonal preconditioner: it does not change the solution of
the linear system, only how much of it survives an LU factorization in double
precision. The factorization is SciPy's sparse LU with COLAMD ordering, done
fresh at every step.

Inside its basin Newton converges quadratically, $\|e_{k+1}\| \approx C\,\|e_k\|^2$.
On a log plot that is not a straight line but a curve that bends ever more
steeply downwards, for example a residual of $10^{-2}$ followed by roughly $10^{-4}$, then
$10^{-8}$, then $10^{-16}$ or the roundoff floor, whichever comes first.
Measured on the $10^{16}$ cm$^{-3}$ diode, Newton needs 4 to 6 steps anywhere
from 0.1 V to 2.0 V, where Gummel needs 46 cycles at 1.0 V and 466 at 2.0 V.
Outside the basin none of that holds, which is why the transfer curve gives
Newton a small budget of 30 iterations per point and relies on continuation
and on the potential step limit rather than on a larger budget. Two more solves
feed the same plot: a cold solve with velocity saturation first runs the low
field models and continues from that answer, and with surface scattering on
each sweep of the surface mobility fixed point is a Newton solve of its own.
The equilibrium solve that builds a cold guess reports nothing, because its
residual is a different quantity on a different system.
