---
title: Newton's method
summary: Solving for the potential, the electrons and the holes all at once, and why its residual drops so steeply once it's close.
docs: 02-numerics.md#Full Newton (Phase 3); 02-numerics.md#Linear algebra; 02-numerics.md#Convergence criteria; 07-decisions.md#Known deviations from reference
---

## In plain words

Newton's method solves the three equations together instead of taking turns.
At the current guess it checks every equation at every node and measures how
far off it is. That's the residual. It also works out how each residual would
shift if each unknown got nudged a little. Then it pretends everything is a
straight line and jumps to the spot where every residual would hit zero at
once. The equations aren't straight lines, so the jump lands near the answer,
not on it, and the method goes again from there.

Far from the answer that jump can go wild, which is why the step gets limited
and the bias gets walked up gradually. Close to the answer it's amazing. Each
step roughly squares the error, so the number of correct digits about
doubles every time. The MOSFET transfer curve runs Newton at every point. On
the residual plot you get three lines per iteration, one per equation: blue
for Poisson (which fixes psi), green for electron continuity (which fixes n)
and red for hole continuity (which fixes p). A C-V sweep also runs Newton,
but on Poisson alone, so it draws a single blue line.

## In more depth

Gather the unknowns into one vector $x$ and the residual of every equation
into $F(x)$. Each Newton step solves the linear system

$$J\,\Delta x = -F, \qquad J = \frac{\partial F}{\partial x}$$

and moves to $x + \Delta x$ once the damping rule has had its say. There are
three unknowns per node, $\psi$, $n$ and $p$, so a mesh of $N$ nodes gives a
$3N \times 3N$ system. The unknowns are interleaved by node,
$(\psi_0, n_0, p_0, \psi_1, \dots)$, not blocked by variable, which keeps
coupled entries close together and cuts fill in the factorization. At each
node the Jacobian has a $3 \times 3$ block,

$$\begin{pmatrix} \partial F_\psi/\partial\psi & \partial F_\psi/\partial n & \partial F_\psi/\partial p \\ \partial F_n/\partial\psi & \partial F_n/\partial n & \partial F_n/\partial p \\ \partial F_p/\partial\psi & \partial F_p/\partial n & \partial F_p/\partial p \end{pmatrix}$$

plus couplings to its neighbours through the Scharfetter-Gummel fluxes. The
test suite checks every block against a finite difference or complex step
Jacobian, because one wrong derivative turns quadratic convergence into a
stall that looks exactly like bad conditioning. Before the solve each row is
divided by one number per equation family, the largest term anywhere in that
family. That's a diagonal preconditioner. It doesn't change the solution of
the linear system, only how much of it survives an LU factorization in double
precision. The factorization is SciPy's sparse LU with COLAMD ordering, done
fresh every step.

Inside its basin Newton converges quadratically,
$\|e_{k+1}\| \approx C\,\|e_k\|^2$. On a log plot that isn't a straight line.
It's a curve that bends down more and more steeply: a residual of $10^{-2}$,
then roughly $10^{-4}$, then $10^{-8}$, then $10^{-16}$ or the roundoff floor,
whichever comes first. Measured on the $10^{16}$ cm$^{-3}$ diode, Newton needs
4 to 6 steps anywhere from 0.1 V to 2.0 V, where Gummel needs 46 cycles at
1.0 V and 466 at 2.0 V. Outside the basin none of that holds. That's why the
transfer curve gives Newton a small budget of 30 iterations per point and
leans on continuation and the potential step limit instead of a bigger
budget. Two more solves feed the same plot. A cold solve with velocity
saturation first runs the low field models and continues from that answer.
With surface scattering on, each pass of the surface mobility fixed point is
a Newton solve of its own. The equilibrium solve that builds a cold guess
reports nothing, because its residual is a different quantity on a different
system.
