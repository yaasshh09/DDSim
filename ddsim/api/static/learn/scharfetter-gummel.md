---
title: The Scharfetter-Gummel scheme
summary: How the current between two neighbouring nodes is computed so that it stays right where densities change by decades.
docs: 02-numerics.md#Scharfetter-Gummel discretization; 02-numerics.md#The Bernoulli function; 02-numerics.md#Why this problem is hard; 05-pitfalls.md#Specific traps
---

## In plain words

To find the current between two neighbouring nodes, the obvious move is to
take the density at each, average them for the drift part and subtract them
for the diffusion part. That works when the density changes gently. Inside a
depletion region it does not change gently: the field there reaches about a
hundred thousand volts per centimetre, and between two neighbouring nodes the
electron density can fall by several decades. The simple average then gives
nonsense, and the solver produces densities that oscillate from node to node
or even go negative, which no real device does.

The Scharfetter-Gummel scheme avoids this by not assuming the density varies
in a straight line between the nodes. It assumes only that the current and the
field are constant along that short edge, and then solves the drift diffusion
equation along the edge exactly. The density between the nodes comes out
exponential, which is how carriers really behave in a field, and the current
is a weighted difference of the two node densities with weights that adjust
themselves to the potential drop. With little drop it behaves like the simple
difference. With a large drop it takes its value from the upstream node, the
one the carriers are flowing from. Every current in this simulator is computed
this way.

## In more depth

Across the edge from node $i$ to node $i+1$, psi is taken as linear, so the
field is constant. With $D_n = \mu_n V_T$ the electron current is

$$J_n = q D_n \left(\frac{dn}{dx} - \frac{n}{V_T}\frac{d\psi}{dx}\right)$$

Setting $X = (\psi_{i+1} - \psi_i)/V_T$, holding $J_n$ constant, solving the
resulting linear equation for $n(x)$ and imposing $n_i$ and $n_{i+1}$ at the
ends gives

$$J_{n,\,i+1/2} = \frac{q D_n}{h}\left(B(X)\,n_{i+1} - B(-X)\,n_i\right), \qquad J_{p,\,i+1/2} = \frac{q D_p}{h}\left(B(X)\,p_i - B(-X)\,p_{i+1}\right)$$

with the Bernoulli function $B(x) = x / (e^x - 1)$. In scaled units $V_T$ and
$q$ drop out and $X$ is just the difference of the two scaled potentials.
The asymmetry matters: $B(X)$ multiplies the right node for electrons and the
left node for holes. Reversed, the solver converges cleanly to a current of
the wrong sign.

Why the naive difference fails is visible in the same notation. Central
differencing writes $J_n \propto (n_{i+1} - n_i) - X\,(n_i + n_{i+1})/2$, so
the coefficients on the two nodes are $1 - X/2$ and $-(1 + X/2)$, and one of
them changes sign once the drop across a single edge exceeds two thermal
voltages in either direction. That sign change breaks the property that keeps
densities positive. $B$ is positive everywhere, so
the SG coefficients never change sign. Since $B(0) = 1$ and $B(-x) = B(x) + x$,
small $|X|$ reduces to central differencing, and for large $|X|$ one term
vanishes and the other tends to $|X|$, which is pure upwinding. Because each
edge carries a single flux shared by its two nodes, the discrete current is
conserved to machine precision.

Evaluating $B$ is its own small problem. The code never writes $e^x - 1$, which
loses every digit near zero. It has three branches: $x / \text{expm1}(x)$
below $-10^{-4}$, the series $1 - x/2 + x^2/12 - x^4/720$ for $|x| \le 10^{-4}$,
and $-x e^{-x} / \text{expm1}(-x)$ above $10^{-4}$. That last form keeps every
representable value out to the underflow near $x = 745$. The derivative the
Newton Jacobian needs has its own five branches, with a series to $x^7$ inside
$|x| \le 0.1$. In 2D the flux is multiplied by the semiconductor share of the
edge's dual face. Under Fermi-Dirac statistics $X$ is taken across each
carrier's degeneracy corrected potential rather than $\psi$, and the form is
unchanged.
