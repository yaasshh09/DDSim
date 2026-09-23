---
title: The Scharfetter-Gummel scheme
summary: How the current between two neighbouring nodes gets computed so it stays right even where densities change by powers of ten.
docs: 02-numerics.md#Scharfetter-Gummel discretization; 02-numerics.md#The Bernoulli function; 02-numerics.md#Why this problem is hard; 05-pitfalls.md#Specific traps
---

## In plain words

To get the current between two neighbouring nodes, the obvious move is to
take the density at each one, average them for the drift part and subtract
them for the diffusion part. That works when the density changes gently. In a
depletion region it doesn't. The field there hits about a hundred thousand
volts per centimetre, and the electron density can drop by several powers of
ten between two neighbouring nodes. The simple average gives nonsense. The
densities start zigzagging from node to node or even go negative, which no
real device ever does.

The Scharfetter-Gummel scheme gets around this by not assuming the density
changes in a straight line between the nodes. It only assumes the current and
the field are constant along that short edge, then solves the drift diffusion
equation along it exactly. The density between the nodes comes out
exponential, which is how carriers really behave in a field. The current ends
up as a weighted difference of the two node densities, and the weights adjust
themselves to the voltage drop. With a small drop it acts like the simple
difference. With a big drop it takes its value from the upstream node, the
one the carriers are coming from. Every current in this simulator is
computed this way.

## In more depth

Across the edge from node $i$ to node $i+1$, psi is taken as linear, so the
field is constant. With $D_n = \mu_n V_T$ the electron current is

$$J_n = q D_n \left(\frac{dn}{dx} - \frac{n}{V_T}\frac{d\psi}{dx}\right)$$

Set $X = (\psi_{i+1} - \psi_i)/V_T$, hold $J_n$ constant, solve the resulting
linear equation for $n(x)$, and impose $n_i$ and $n_{i+1}$ at the ends. You
get

$$J_{n,\,i+1/2} = \frac{q D_n}{h}\left(B(X)\,n_{i+1} - B(-X)\,n_i\right), \qquad J_{p,\,i+1/2} = \frac{q D_p}{h}\left(B(X)\,p_i - B(-X)\,p_{i+1}\right)$$

with the Bernoulli function $B(x) = x / (e^x - 1)$. In scaled units $V_T$ and
$q$ drop out and $X$ is just the difference of the two scaled potentials.
The asymmetry matters: $B(X)$ multiplies the right node for electrons and the
left node for holes. Swap them and the solver converges cleanly to a current
of the wrong sign.

The same notation shows why the naive difference fails. Central differencing
writes $J_n \propto (n_{i+1} - n_i) - X\,(n_i + n_{i+1})/2$, so the
coefficients on the two nodes are $1 - X/2$ and $-(1 + X/2)$. One of them
flips sign once the drop across a single edge passes two thermal voltages in
either direction, and that breaks the property that keeps densities positive.
$B$ is positive everywhere, so the SG coefficients never flip. Since
$B(0) = 1$ and $B(-x) = B(x) + x$, small $|X|$ reduces to central
differencing. For large $|X|$ one term vanishes and the other tends to $|X|$,
which is pure upwinding. Each edge carries a single flux shared by its two
nodes, so the discrete current is conserved to machine precision.

Evaluating $B$ is its own little problem. The code never writes $e^x - 1$,
which loses every digit near zero. It has three branches: $x / \text{expm1}(x)$
below $-10^{-4}$, the series $1 - x/2 + x^2/12 - x^4/720$ for
$|x| \le 10^{-4}$, and $-x e^{-x} / \text{expm1}(-x)$ above $10^{-4}$. That
last form keeps every representable value out to the underflow near
$x = 745$. The derivative the Newton Jacobian needs has five branches of its
own, with a series to $x^7$ inside $|x| \le 0.1$. In 2D the flux gets
multiplied by the semiconductor share of the edge's dual face. Under
Fermi-Dirac statistics $X$ is taken across each carrier's degeneracy
corrected potential instead of $\psi$, and the form doesn't change.
