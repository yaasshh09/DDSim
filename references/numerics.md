# Numerics

This is the most important file in the repo. Read all of it before writing
solver code.

## Why this problem is hard

There are three specific reasons, and everything below responds to one of
them.

**1. Exponential stiffness.** n = n_i * exp(psi / V_T) with V_T = 25.85 mV. A
one volt swing is a factor of exp(38.7), roughly 6e16. A naive Newton step of
0.5 V overflows on the spot, and carrier densities span 25 orders of magnitude
across a single device.

**2. Convection dominance.** In a depletion region the field reaches 1e5 V/cm.
The continuity equation is strongly convection dominated, with a cell Peclet
number far above 1, and central differencing gives you oscillating, negative
carrier densities. It's a classic CFD failure showing up in device physics.

**3. No good initial guess.** The coupled system at an arbitrary bias has no
analytic starting point. You have to ramp up from equilibrium.

## Scaling (de Mari)

Do this first, in Phase 0, before anything else. Never work in raw SI or raw
CGS units, or the conditioning will destroy you.

Reference quantities:

    psi_0 = V_T                              [V]
    C_0   = n_i        (or max|net_doping|)  [cm^-3]
    x_0   = L_D = sqrt(eps * V_T / (q * C_0))  [cm]
    D_0   = max(Dn, Dp)                      [cm^2/s]
    mu_0  = D_0 / V_T                        [cm^2/(V s)]
    t_0   = x_0^2 / D_0                      [s]
    J_0   = q * D_0 * C_0 / x_0              [A/cm^2]
    R_0   = D_0 * C_0 / x_0^2                [cm^-3 s^-1]

Scaled Poisson comes out clean and dimensionless:

    lap(psi) = -(p - n + N)

Scaled continuity, steady state:

    div(Jn) = R
    div(Jp) = -R

The scaled SG current on an edge of scaled length h:

    Jn = (Dn / h) * (B(dpsi) * n_right - B(-dpsi) * n_left)

V_T has dropped out of the Bernoulli argument because psi is already scaled
by V_T. That's the whole point.

**Choice of C_0.** Using n_i keeps Poisson in its cleanest form. Using
max|net_doping| conditions better when the doping spans many decades. Start
with n_i. If Phase 5 conditioning turns out bad, switch and rerun every test.
Whichever you pick, set it in one place and never hardcode it anywhere else.

## The Bernoulli function

    B(x) = x / (exp(x) - 1)

The whole discretization hinges on evaluating this correctly. Implement it in
Phase 0 with full branch handling, and test it before anything else.

Properties:

    B(0) = 1
    B(-x) = B(x) + x                        <- strongest identity, use as a test
    B(x) -> -x         as x -> -inf
    B(x) -> x*exp(-x)  as x -> +inf         (underflows to 0 safely)
    B'(0) = -1/2

Branch structure. These are the thresholds `ddsim/physics/bernoulli.py`
ships, tuned against an 80 digit `decimal` reference rather than a back of
the envelope estimate:

| Range | Evaluation |
|---|---|
| x < -1e-4 | x / expm1(x) |
| \|x\| <= 1e-4 | series: 1 - x/2 + x^2/12 - x^4/720 |
| x > 1e-4 | -x * exp(-x) / expm1(-x) |

There's no large \|x\| branch, and none is needed. For x < -80, expm1(x)
saturates to exactly -1.0 and x / expm1(x) returns exactly -x by itself, so a
short circuit there buys nothing. Nothing overflows either, since expm1 of a
negative argument lives in [-1, 0). On the positive side, writing the branch
as -x*exp(-x)/expm1(-x) keeps every representable value from B(80) = 1.44e-33
down to the underflow at x = 745. A hard "return 0.0 above 80" would throw
away 290 decades of perfectly good numbers.

Use `numpy.expm1`, never `exp(x) - 1`. The latter loses all its precision
near zero.

The Newton Jacobian also needs the derivative dB/dx. The textbook form is

    B'(x) = (exp(x) * (1 - x) - 1) / (exp(x) - 1)^2

and it isn't accurate enough to use as written. `exp(x)*(1 - x) - 1`
subtracts two order one quantities to get an order x^2 result, so the
relative error goes as eps/x^2. Against the 80 digit reference it's 1.8e-8 at
x = 1e-4 and 7.7e-8 at 1e-5, five orders of magnitude short of the 1e-13
target. Rewritten with expm1 as

    B'(x) = (E * (1 - x) - x) / E^2,      E = expm1(x)

the cancellation drops to order x and you get three orders of magnitude back.
The derivative branches are:

| Range | Evaluation |
|---|---|
| x < -80 | -1.0 |
| -80 <= x < -0.1 | (E*(1 - x) - x) / E^2 |
| \|x\| <= 0.1 | series: -1/2 + x/6 - x^3/180 + x^5/5040 - x^7/151200 |
| 0.1 < x <= 80 | (E*(1 - x) - x) / E^2 |
| x > 80 | (1 - x) * exp(-x) |

The series needs terms to x^7 and a window out to 0.1 for both sides of the
boundary to sit near 1e-15. The obvious thing to write, a three term series
cut off at 1e-4, leaves a 1e-8 step at the branch boundary.

Complex step differentiation is worth having as a check, but it isn't exact
here, and near the origin it'll lie to you. It removes the cancellation from
the differencing, not the cancellation inside B's own algebra, and B has
plenty of the latter near zero. It loses roughly eps/\|x\|: I measured
9.3e-11 at x = 1e-6 against 3.3e-15 at x = 0.1. Use it as the criterion for
0.1 <= \|x\| <= 300, where it really is exact, and use a high precision
reference for everything closer in. Also, neither numpy nor math has a
complex expm1, and a reference built on the naive `exp(z) - 1` is three orders
worse again, which looks exactly like a bug in the code it's supposed to be
checking.

## Scharfetter-Gummel discretization

The central idea: assume Jn and E are constant on the edge between nodes i
and i+1, then integrate the current relation analytically over that edge.

### Derivation (keep this, it is how you get the signs right)

On the edge psi is linear, so E = -dpsi/dx is constant. With Dn = mu_n * V_T:

    Jn = q*Dn * (dn/dx - n * dpsi/dx / V_T)

Let X = (psi_{i+1} - psi_i) / V_T and a = X/h. Then

    dn/dx - a*n = c,  where c = Jn / (q*Dn)

Solving the linear ODE and applying n(x_i) = n_i, n(x_{i+1}) = n_{i+1}:

    Jn_{i+1/2} = (q*Dn/h) * ( B(X)*n_{i+1} - B(-X)*n_i )

The same route for holes gives:

    Jp_{i+1/2} = (q*Dp/h) * ( B(X)*p_i - B(-X)*p_{i+1} )

**Memorize the asymmetry.** For electrons the B(X) factor multiplies the right
hand node. For holes it multiplies the left hand node. Get this backwards and
the solver converges beautifully to a physically wrong answer with the current
reversed. Write a test that checks the sign of the current through a forward
biased diode.

### Why it works

B(x) is an exponential fitting function. At low field (|X| << 1) it reduces to
central differencing, and at high field it turns into pure upwinding, all by
itself. The scheme is unconditionally stable, strictly current conserving on
the mesh, and stays accurate on coarse grids where central differencing would
fall apart.

### Current conservation

Because SG comes from the edge flux, the discrete divergence of the discrete
current is exactly zero in a source free steady state, to machine precision.
In a 1D diode with recombination off, Jn + Jp has to be identical at every
node. That's the single strongest correctness invariant in the whole project.
Assert it.

## Mesh

**1D (Phases 1-3).** Uniform first, then graded and refined at junctions. The
spacing at a junction has to resolve the local Debye length:

    h_local < L_D(local doping) / 2

At 1e18 cm^-3 in silicon, L_D is about 4 nm. Don't try to resolve a 1e20
source or drain junction with a uniform mesh.

**2D (Phases 4-5).** Box integration finite volume on a Delaunay
triangulation. Finite volume because it's naturally conservative, and because
SG lives on edges, which maps straight onto the primal edge and dual face
structure.

The critical requirement: the triangulation has to be **boundary conforming
Delaunay**, so every dual face (Voronoi facet) sits on the correct side of its
primal edge. Obtuse triangles give negative dual face areas, which give
negative conductances, which wreck the M-matrix property and break the maximum
principle. The symptom is carrier densities going negative somewhere for no
physical reason. Check triangle quality when the mesh is built, and reject or
refine obtuse elements.

For a rectangular MOSFET, a structured tensor product mesh is legitimate and
sidesteps the whole issue. Consider going that way in Phase 4, and only move
to unstructured meshes if you need non-rectangular geometry.

## Solution strategy

### Nonlinear Poisson at equilibrium (Phase 1)

Start here. Substitute Boltzmann statistics into Poisson to eliminate n and p:

    lap(psi) = -(n_i*exp(-psi/V_T) - n_i*exp(psi/V_T) + N)

One unknown per node, and Newton on this scalar equation. The Jacobian's
diagonal contribution is strictly positive, so it's well conditioned and
converges reliably. The initial guess comes from charge neutrality:

    psi_init = V_T * asinh(N / (2*n_i))

Use asinh, not log. See `docs/01-physics.md`, ohmic contacts.

### Gummel iteration (Phase 2)

Decouple the three equations and cycle through them:

1. Solve nonlinear Poisson for psi, holding phi_n and phi_p fixed
2. Solve the electron continuity, now linear in n, for n
3. Solve hole continuity for p
4. Check the update norm and repeat

Convergence is linear. Very robust at low bias, and it degrades badly at high
injection where psi, n and p are strongly coupled. That degradation is
expected, and it's the reason Phase 3 exists. Don't fight it in Phase 2.

Where you can, solve continuity in quasi-Fermi potentials rather than
densities. That squeezes the dynamic range from 25 decades down to a few
volts.

### Full Newton (Phase 3)

Assemble the full 3N x 3N Jacobian for (psi, n, p) and solve everything at
once. Convergence is quadratic, but only from inside the basin of attraction.

The block structure per node:

    [ dF_psi/dpsi   dF_psi/dn    dF_psi/dp  ]
    [ dF_n/dpsi     dF_n/dn      dF_n/dp    ]
    [ dF_p/dpsi     dF_p/dn      dF_p/dp    ]

Ordering matters for fill in. Interleave by node (psi_0, n_0, p_0, psi_1, ...)
instead of blocking by variable. Better locality, less fill.

**Verify the Jacobian.** Before you trust Newton, compare every assembled
block against a finite difference or complex step Jacobian on a small mesh.
One wrong derivative turns quadratic convergence into stagnation, and you'll
spend days blaming the conditioning. The test is cheap, and it isn't
optional.

**Damping.** Bank-Rose damping, or plain step limiting on psi:

    dpsi_max = 5 * V_T per Newton step (scaled: 5.0)

Clamp the psi update and take the n and p updates in full. If n or p goes
negative after an update, the damping is too loose or a sign is wrong. Check
the signs first.

### Strategy in practice

Run Gummel for 3 to 5 iterations to get into the basin, then switch to
Newton. If Newton stalls or the residual climbs, fall back to Gummel for a
few more and try again. Log every switch.

## Bias continuation

Never jump to the target bias. Ramp.

    V = 0
    dV = 0.05
    while V < V_target:
        try solve at V+dV, using solution at V as initial guess
        if converged:
            V += dV
            dV = min(dV * 1.5, dV_max)     # grow cautiously
        else:
            dV = dV / 2                     # retry smaller
            if dV < dV_min: fail loudly

Structurally this is the same as source stepping in SPICE. Write it as a
generic continuation driver that takes a solve callback, not baked into the
device code. It'll get reused as is in the SPICE project.

Reuse the previous converged solution as the initial guess. That's the entire
reason continuation works.

### The first point is a jump too

A sweep ramps between its points, but it has to start at one of them, and on
a device with more than one terminal that starting point already carries the
biases the sweep doesn't sweep. A MOSFET transfer curve starts at zero gate
with the drain already at 1 V. So the solve that starts the ramp is exactly
the jump the rule above forbids, taken from a guess that knows nothing about
any of it.

Ramp it the same way, on one scalar fraction of every applied bias at once,
from the all zero device out to the one asked for. `solve_bias_ramped` does
that, with the same continuation driver. Two things it does that aren't
obvious:

- It only kicks in cold. A continuation step already arrives with the
  neighbouring solution, and that guess is worth more than anything a fresh
  ramp produces.
- It takes its last step at the device's own bias whether the ramp got there
  or not. A stalled ramp is holding a converged solve at some fraction, and
  handing that back would be a wrong answer wearing a converged flag.

## Linear algebra

Sparse, non-symmetric and ill conditioned. At these sizes a direct solve is
the right call.

- `scipy.sparse.linalg.splu` with COLAMD ordering, the SciPy default.
- Assemble in COO, convert to CSC once, then factorize.
- 1D: a few hundred to a few thousand unknowns. Trivial.
- 2D MOSFET: 10k to 100k unknowns. splu is still fine, seconds per solve.
- Don't reach for iterative solvers (GMRES, BiCGStab) unless a solve takes
  more than 30 seconds. With this conditioning, preconditioning would be a
  project of its own.

Reusing the symbolic factorization across Newton steps is the obvious thing to
want, since the sparsity pattern never changes. SciPy won't let you. `splu`
takes `permc_spec` as a string and hands `perm_c` back, with no way to pass a
symbolic factorization or precomputed permutation in, so there's no symbolic
and numeric split to exploit.

The usual workaround, keeping `perm_c` and refactorizing `A[:, perm_c]` with
`permc_spec="NATURAL"`, makes things worse here, and by a lot. On a five point
stencil, 23 ms turns into 352 ms at 100x100 and 134 ms into 6146 ms at
200x200. The column gather isn't the cost; it's 0.6 ms. The factorization
itself blows up, from 645,750 nonzeros in L and U under COLAMD to 3,933,424
under the pre-permuted NATURAL run, 6.1 times more fill. SuperLU's COLAMD path
does a column elimination tree postordering that the NATURAL path skips, so
`perm_c` alone doesn't reproduce the ordering SuperLU actually eliminated
with. Factorize fresh with COLAMD every time.

What is worth reusing is the part of the assembly that only depends on the
pattern. Going from COO triplets to CSC costs a sort and a duplicate
summation, and across Newton steps only the values change, so both can be
computed once and replayed as a gather plus a segmented sum. That's real, and
it's the single biggest win in the solve loop. Keep a fingerprint of the
pattern so a future UMFPACK or KLU backend, which does expose the symbolic and
numeric split, can deliver the rest.

## Convergence criteria

Check all three, not just one:

1. Update norm: max |dpsi| < 1e-10 (scaled), and carrier change under 1e-8
2. Residual norm: ||F|| below a threshold built from the size of the terms F
   is assembled from, on the scaled system
3. Current continuity: max deviation of (Jn + Jp) across nodes < 1e-6
   relative

Criterion 3 is physical rather than algebraic, and it catches failures the
other two miss. Report all three at every step in verbose mode, and report
the threshold next to the residual, not just the residual. A residual sitting
above its threshold looks like a slow solve when it's really an unreachable
threshold, and those need different fixes.

Three refinements only show up once real devices are running.

**The residual threshold can't be absolute.** The Poisson residual scales with
the doping, and so does its roundoff floor, so a fixed 1e-10 that's easy at
1e16 is out of reach at 1e18. Making it relative to the initial residual fixes
that for a cold start and breaks warm starts, which is what every Gummel cycle
after the first one is. A solve handed the answer already starts on its
roundoff floor, and a threshold a decade below that floor can never be met.
Build the scale from something that doesn't depend on the starting iterate.
For Poisson, the doping charge in the largest dual cell works.

**And that scale has a floor of its own.** The residual is also a difference
of two face fluxes of size eps*psi/h, and a difference can't resolve below
eps times the size of the things being differenced. The two scales move in
opposite directions: the charge falls linearly with doping while psi is only
logarithmic in it. So below about 1e13 cm^-3 the charge based threshold sinks
under the flux floor, and a perfectly converged solve reports failure. On a
1e12 bar the residual reached 6.8e-12 at iteration three and sat there,
unchanged to the last bit, for the remaining forty-seven, with an update of
4.4e-16 the whole time. Raise the scale to clear the flux floor, and only when
the floor would otherwise bind, so every device that already clears it keeps
the threshold it had. High resistivity substrates live at exactly these
dopings.

**Measure the carrier change as max |dn| / (n + n_i).** A pure relative change
gets dominated by nodes where the density is 1e-15 and physically irrelevant.
An absolute change gets dominated by the majority carrier. The floor at n_i is
1 in scaled units and says what you actually mean: a carrier below the
intrinsic density carries no charge worth converging.

Finally, stop early when the residual has frozen to the last bit while the
update is already inside tolerance. Both criteria still have to pass, so it's
still a failure and still reported as one. But the rest of the budget can't
change the answer, and spending it only delays the diagnosis.

## Small-signal AC (Phase 4, for C-V)

Don't time step. Linearize around the converged DC solution and solve the
complex valued system at frequency omega:

    (J_dc + i*omega*M) * x = b

where M is the mass matrix (the dQ/dt terms) and J_dc is the Jacobian already
assembled for the DC Newton solve. The terminal admittance is
Y = G + i*omega*C, and the capacitance is Im(Y) / omega.

This reuses the DC Jacobian entirely. It's about 60 lines once Phase 3 works.

### The mass matrix, planned for Phase 10

What's shipped so far is only the omega to zero limit, where M drops out (that limit is what
`extract/cv.py` computes). The frequency solve needs M written down.

With unknowns (psi, n, p) node interleaved as in `discretize/coupled.py`, the
continuity residuals with their time derivatives are

    F_n,i = R_i*volume_i + (dn_i/dt)*volume_i - (Jn_{i+1/2} - Jn_{i-1/2})
    F_p,i = (Jp_{i+1/2} - Jp_{i-1/2}) + R_i*volume_i + (dp_i/dt)*volume_i

so M is diagonal, with volume_i at the n row and the p row of node i and
nothing at the Poisson row. Both signs are positive, because the time
derivative sits on the same side as R in both equations. In scaled units,
time is measured in t_0 = x_0^2 / D_0. The test for M differentiates a time
stepped residual by finite differences and compares it entry by entry. It
exists because a sign error in M still gives a plausible-looking admittance.

The system at frequency omega is complex. SciPy's `splu` factorizes complex
matrices, so the same linear solver wrapper carries over.

## Time integration (Phase 11)

Planned. Backward Euler first, then variable step BDF2.

Backward Euler solves, per step,

    F(x_{k+1}) + M (x_{k+1} - x_k) / dt = 0

by Newton, with Jacobian J + M/dt. Adding M/dt to the diagonal of the
continuity blocks makes the matrix more diagonally dominant, so a transient
step is easier than the DC solve at the same bias. A step that fails is
halved, the same way a continuation step is.

BDF2 with variable steps uses the standard variable coefficient form. The
local truncation error is estimated from the gap between the explicit
predictor and the converged corrector, and that estimate controls step
acceptance and the next dt. Order is measured, not assumed: halving dt on a
smooth problem must cut the error by 2 for backward Euler and by 4 for BDF2.

The terminal current in a transient includes the displacement current, the
time derivative of the contact charge that `extract/cv.py` already knows how
to read. Without it, the terminal currents stop summing to zero.

## Error estimation (Phase 12)

Planned. The dual weighted residual estimate for a goal Q, such as a terminal
current:

1. Solve the adjoint J^T z = dQ/dx at the converged state, reusing the
   factorization with `trans='T'`.
2. Prolong the solution and z onto the mesh refined once everywhere.
3. Assemble the residual F_fine of the prolonged solution there. There's no
   solve on the fine mesh.
4. The estimate is eta = -z_fine . F_fine. The per cell terms are the
   refinement indicator.

The check that matters is the effectivity index, estimate over true error,
where true error comes from Richardson extrapolation. It has to sit in
[0.5, 2] on every benchmark that has a Richardson reference.

## Iterative linear solves (Phase 16)

Planned for 3D, where a direct LU fills in too much. GMRES preconditioned by
an incomplete LU, inside an inexact Newton whose linear tolerance follows the
Eisenstat-Walker forcing rule. `splu` stays the default in 1D and 2D. On
every 2D benchmark, the two must reach the same converged state within the
Newton tolerance.
