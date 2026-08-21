# Numerics

The most important file in this repo. Read it fully before writing solver code.

## Why this problem is hard

Three specific reasons. Everything below is a response to one of them.

**1. Exponential stiffness.** n = n_i * exp(psi / V_T) with V_T = 25.85 mV. A one
volt swing is a factor of exp(38.7), roughly 6e16. A naive Newton step of 0.5 V
overflows immediately. Carrier densities span 25 orders of magnitude across a
single device.

**2. Convection dominance.** In a depletion region the field reaches 1e5 V/cm.
The continuity equation is strongly convection-dominated, cell Peclet number far
above 1. Central differencing produces oscillating and negative carrier
densities. This is a classic CFD failure mode showing up in device physics.

**3. No good initial guess.** The coupled system at arbitrary bias has no
analytic starting point. You must ramp from equilibrium.

## Scaling (de Mari)

Do this first, in Phase 0, before anything else. Never work in raw SI or raw CGS
units. Conditioning will destroy you otherwise.

Reference quantities:

    psi_0 = V_T                              [V]
    C_0   = n_i        (or max|net_doping|)  [cm^-3]
    x_0   = L_D = sqrt(eps * V_T / (q * C_0))  [cm]
    D_0   = max(Dn, Dp)                      [cm^2/s]
    mu_0  = D_0 / V_T                        [cm^2/(V s)]
    t_0   = x_0^2 / D_0                      [s]
    J_0   = q * D_0 * C_0 / x_0              [A/cm^2]
    R_0   = D_0 * C_0 / x_0^2                [cm^-3 s^-1]

Scaled Poisson becomes clean and dimensionless:

    lap(psi) = -(p - n + N)

Scaled continuity, steady state:

    div(Jn) = R
    div(Jp) = -R

Scaled SG current on an edge of scaled length h:

    Jn = (Dn / h) * (B(dpsi) * n_right - B(-dpsi) * n_left)

Note V_T has vanished from the Bernoulli argument because psi is already scaled
by V_T. That is the whole point.

**Choice of C_0.** Using n_i keeps the Poisson equation in its cleanest form.
Using max|net_doping| gives better conditioning when doping spans many decades.
Start with n_i. If Phase 5 conditioning is bad, switch and re-run all tests.
Whichever you pick, put it in one place and never hardcode it elsewhere.

## The Bernoulli function

    B(x) = x / (exp(x) - 1)

Everything in the discretization hinges on evaluating this correctly. Implement
it in Phase 0 with full branch handling and test it before anything else.

Properties:

    B(0) = 1
    B(-x) = B(x) + x                        <- strongest identity, use as a test
    B(x) -> -x         as x -> -inf
    B(x) -> x*exp(-x)  as x -> +inf         (underflows to 0 safely)
    B'(0) = -1/2

Branch structure. These are the thresholds `ddsim/physics/bernoulli.py` ships,
after tuning against an 80 digit `decimal` reference rather than against a back
of the envelope estimate:

| Range | Evaluation |
|---|---|
| x < -1e-4 | x / expm1(x) |
| \|x\| <= 1e-4 | series: 1 - x/2 + x^2/12 - x^4/720 |
| x > 1e-4 | -x * exp(-x) / expm1(-x) |

There is no large-\|x\| branch and none is needed. For x < -80, expm1(x)
saturates to exactly -1.0 and x / expm1(x) returns exactly -x on its own, so a
short circuit there buys nothing. Nothing overflows either: expm1 of a negative
argument lives in [-1, 0). On the positive side, writing the positive branch as
-x*exp(-x)/expm1(-x) keeps every representable value from B(80) = 1.44e-33 down
to the underflow at x = 745, where a hard "return 0.0 above 80" would throw away
290 decades of perfectly good numbers.

Use `numpy.expm1`, never `exp(x) - 1`. The latter loses all precision near zero.

You also need the derivative dB/dx for the Newton Jacobian. The textbook form is

    B'(x) = (exp(x) * (1 - x) - 1) / (exp(x) - 1)^2

and it is not accurate enough to use as written. `exp(x)*(1 - x) - 1` subtracts
two order-one quantities to produce an order x^2 result, so the relative error
goes as eps/x^2. Measured against the 80 digit reference it is 1.8e-8 at
x = 1e-4 and 7.7e-8 at 1e-5, which misses the 1e-13 target by five orders of
magnitude. Rewrite it with expm1 as

    B'(x) = (E * (1 - x) - x) / E^2,      E = expm1(x)

which moves the cancellation down to order x and buys back three orders of
magnitude. The derivative branches are:

| Range | Evaluation |
|---|---|
| x < -80 | -1.0 |
| -80 <= x < -0.1 | (E*(1 - x) - x) / E^2 |
| \|x\| <= 0.1 | series: -1/2 + x/6 - x^3/180 + x^5/5040 - x^7/151200 |
| 0.1 < x <= 80 | (E*(1 - x) - x) / E^2 |
| x > 80 | (1 - x) * exp(-x) |

The series needs terms to x^7 and a window out to 0.1 for both sides of the
boundary to sit near 1e-15. A three term series cut off at 1e-4, which is the
obvious thing to write, leaves a 1e-8 step at the branch boundary.

Complex-step differentiation is worth having as a check, but it is not exact
here and it will lie to you near the origin. It removes the cancellation from
the differencing, not the cancellation inside B's own algebra, and B has plenty
of the latter near zero. It loses roughly eps/\|x\|: measured 9.3e-11 at
x = 1e-6 against 3.3e-15 at x = 0.1. Use it as the criterion for
0.1 <= \|x\| <= 300, where it genuinely is exact, and use a high precision
reference for everything closer in. Note also that neither numpy nor math has a
complex expm1, and a reference built on the naive `exp(z) - 1` is three orders
worse again, which reads exactly like a bug in the implementation it is meant to
be checking.

## Scharfetter-Gummel discretization

The central idea. Assume Jn and E are constant on the edge between nodes i and
i+1, then integrate the current relation analytically over that edge.

### Derivation (keep this, it is how you get the signs right)

On the edge, psi is linear, so E = -dpsi/dx is constant. With Dn = mu_n * V_T:

    Jn = q*Dn * (dn/dx - n * dpsi/dx / V_T)

Let X = (psi_{i+1} - psi_i) / V_T and a = X/h. Then

    dn/dx - a*n = c,  where c = Jn / (q*Dn)

Solving the linear ODE and applying n(x_i) = n_i, n(x_{i+1}) = n_{i+1}:

    Jn_{i+1/2} = (q*Dn/h) * ( B(X)*n_{i+1} - B(-X)*n_i )

By the same route for holes:

    Jp_{i+1/2} = (q*Dp/h) * ( B(X)*p_i - B(-X)*p_{i+1} )

**Memorize the asymmetry.** For electrons the B(X) factor multiplies the
right-hand node. For holes it multiplies the left-hand node. Getting this
backwards produces a solver that converges beautifully to a physically wrong
answer with reversed current. Write a test that checks the sign of current
through a forward-biased diode.

### Why it works

B(x) is an exponential fitting function. At low field (|X| << 1) it reduces to
central differencing. At high field it becomes pure upwinding, automatically.
The scheme is unconditionally stable, strictly current conserving on the mesh,
and stays accurate on coarse grids where central differencing would fail.

### Current conservation

Because SG is derived from the edge flux, the discrete divergence of the discrete
current is exactly zero in a source-free steady state, to machine precision. In a
1D diode with recombination off, Jn + Jp must be identical at every node. This is
the single strongest correctness invariant in the entire project. Assert it.

## Mesh

**1D (Phases 1-3).** Uniform first. Then graded, refined at junctions. The mesh
spacing at a junction must resolve the Debye length there:

    h_local < L_D(local doping) / 2

At 1e18 cm^-3 in silicon, L_D is roughly 4 nm. Do not try to resolve a 1e20
source/drain junction with a uniform mesh.

**2D (Phases 4-5).** Box-integration finite volume on a Delaunay triangulation.
Finite volume because it is naturally conservative and because SG lives on edges,
which maps directly onto the primal-edge / dual-face structure.

Critical requirement: the triangulation must be **boundary-conforming Delaunay**
so that every dual face (Voronoi facet) is on the correct side of its primal
edge. Obtuse triangles produce negative dual face areas, which produce negative
conductances, which destroy the M-matrix property and break the maximum
principle. Symptom: carrier densities go negative in a region with no physical
reason. Check triangle quality at mesh construction time and reject or refine
obtuse elements.

For a rectangular MOSFET, a structured tensor-product mesh is legitimate and
avoids the whole issue. Consider taking that path in Phase 4 and only moving to
unstructured if you need non-rectangular geometry.

## Solution strategy

### Nonlinear Poisson at equilibrium (Phase 1)

Start here. Substitute Boltzmann statistics into Poisson to eliminate n and p:

    lap(psi) = -(n_i*exp(-psi/V_T) - n_i*exp(psi/V_T) + N)

One unknown per node. Newton on this scalar equation. The Jacobian diagonal
contribution is strictly positive, which makes it well-conditioned and reliably
convergent. Initial guess from charge neutrality:

    psi_init = V_T * asinh(N / (2*n_i))

Use asinh, not log. See `docs/01-physics.md`, ohmic contacts.

### Gummel iteration (Phase 2)

Decouple the three equations and cycle:

1. Solve nonlinear Poisson for psi, holding phi_n and phi_p fixed
2. Solve the (now linear in n) electron continuity for n
3. Solve hole continuity for p
4. Check the update norm, repeat

Convergence is linear. Very robust at low bias. Degrades badly at high injection
where the psi-n-p coupling is strong. That degradation is expected and is the
reason Phase 3 exists. Do not fight it in Phase 2.

Solve continuity in terms of quasi-Fermi potentials rather than densities when
possible. It compresses the dynamic range from 25 decades to a few volts.

### Full Newton (Phase 3)

Assemble the full 3N x 3N Jacobian for (psi, n, p) and solve simultaneously.
Quadratic convergence, but only from inside the basin of attraction.

Block structure per node:

    [ dF_psi/dpsi   dF_psi/dn    dF_psi/dp  ]
    [ dF_n/dpsi     dF_n/dn      dF_n/dp    ]
    [ dF_p/dpsi     dF_p/dn      dF_p/dp    ]

Ordering matters for fill-in. Interleave by node (psi_0, n_0, p_0, psi_1, ...)
rather than blocking by variable. Better locality, less fill.

**Verify the Jacobian.** Before trusting Newton, compare every assembled block
against a finite-difference or complex-step Jacobian on a small mesh. A single
wrong derivative turns quadratic convergence into stagnation and you will spend
days blaming conditioning. This test is cheap and it is non-negotiable.

**Damping.** Bank-Rose damping, or simple step limiting on psi:

    dpsi_max = 5 * V_T per Newton step (scaled: 5.0)

Clamp the psi update, accept the full n and p updates. If n or p goes negative
after an update, your damping is too loose or a sign is wrong. Check signs first.

### Strategy in practice

Gummel for 3 to 5 iterations to get into the basin, then switch to Newton. If
Newton stalls or the residual increases, fall back to Gummel for a few more and
retry. Log every switch.

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

This is structurally identical to source stepping in SPICE. Write it as a generic
continuation driver taking a solve callback, not baked into the device code. You
will reuse it verbatim in the SPICE project.

Reuse the previous converged solution as the initial guess. That is the entire
reason continuation works.

## Linear algebra

Sparse, non-symmetric, ill-conditioned. Direct solve is correct at these sizes.

- `scipy.sparse.linalg.splu` with COLAMD ordering. Default in SciPy.
- Assemble in COO, convert to CSC once, then factorize.
- 1D: a few hundred to a few thousand unknowns. Trivial.
- 2D MOSFET: 10k to 100k unknowns. splu still fine, seconds per solve.
- Do not reach for iterative solvers (GMRES, BiCGStab) unless a solve exceeds
  30 seconds. The matrix conditioning makes preconditioning its own project.

Reusing the symbolic factorization across Newton steps is the obvious thing to
want, since the sparsity pattern never changes. SciPy will not let you do it.
`splu` takes `permc_spec` as a string and hands `perm_c` back, with no way to
pass a symbolic factorization or a precomputed permutation in, so there is no
symbolic and numeric split to exploit.

The usual workaround, keeping `perm_c` and refactorizing `A[:, perm_c]` with
`permc_spec="NATURAL"`, is a pessimization here and a bad one. Measured on a
five point stencil: 23 ms becomes 352 ms at 100x100, 134 ms becomes 6146 ms at
200x200. The column gather is not the cost, it is 0.6 ms. The factorization
itself blows up, from 645,750 nonzeros in L and U under COLAMD to 3,933,424
under the pre-permuted NATURAL run, a factor of 6.1 more fill. SuperLU's COLAMD
path does a column elimination tree postordering that the NATURAL path skips, so
`perm_c` on its own does not reproduce the ordering SuperLU actually eliminated
with. Factorize fresh with COLAMD every time.

What is worth reusing is the part of the assembly that depends only on the
pattern. Going from COO triplets to CSC costs a sort and a duplicate summation,
and across Newton steps only the values move, so both can be computed once and
replayed as a gather plus a segmented sum. That is real and it is the single
biggest win in the solve loop. Keep a fingerprint of the pattern so a future
UMFPACK or KLU backend, which does expose the symbolic and numeric split, can
deliver the rest.

## Convergence criteria

Check all three, not just one:

1. Update norm: max |dpsi| < 1e-10 (scaled), and carrier change under 1e-8
2. Residual norm: ||F|| below a threshold built from the size of the terms F is
   assembled from, on the scaled system
3. Current continuity: max deviation of (Jn + Jp) across nodes < 1e-6 relative

Criterion 3 is physical rather than algebraic and it catches failures the other
two miss. Report all three at every step in verbose mode, and report the
threshold alongside the residual, not just the residual. A residual sitting
above a threshold reads as a slow solve when it is really an unreachable
threshold, and those have different fixes.

Three refinements that only show up once real devices are running.

**The residual threshold cannot be absolute.** The Poisson residual scales with
the doping, and so does its own roundoff floor, so a fixed 1e-10 that is
comfortable at 1e16 is unreachable at 1e18. Making it relative to the initial
residual fixes that for a cold start and breaks warm starts, which is what every
Gummel cycle after the first is: a solve handed the answer already starts on its
roundoff floor, and a threshold a decade under that floor can never be met.
Build the scale from something that does not depend on the starting iterate.
For Poisson, the doping charge in the largest dual cell works.

**And that scale has a floor of its own.** The residual is also a difference of
two face fluxes of size eps*psi/h, and a difference cannot resolve below eps
times the size of the things being differenced. The two scales move in opposite
directions: the charge falls linearly with doping while psi is only logarithmic
in it, so below about 1e13 cm^-3 the charge-based threshold sinks underneath the
flux floor and a perfectly converged solve reports failure. On a 1e12 bar the
residual reached 6.8e-12 at iteration three and sat there, unchanged to the last
bit, for the remaining forty-seven, with an update of 4.4e-16 throughout. Raise
the scale to clear the flux floor, and only when the floor would otherwise bind,
so every device that already clears it keeps the threshold it had. High
resistivity substrates live at exactly these dopings.

**Measure the carrier change as max |dn| / (n + n_i).** A pure relative change
is dominated by nodes where the density is 1e-15 and physically irrelevant. An
absolute change is dominated by the majority carrier. The floor at n_i is 1 in
scaled units and says what you mean: a carrier below the intrinsic density
carries no charge worth converging.

Finally, stop early when the residual has frozen to the last bit while the
update is already inside tolerance. Both criteria still have to pass, so this is
still a failure and still reported as one, but the remaining budget cannot
change the answer and spending it only delays the diagnosis.

## Small-signal AC (Phase 4, for C-V)

Do not time-step. Linearize around the converged DC solution and solve the
complex-valued system at frequency omega:

    (J_dc + i*omega*M) * x = b

where M is the mass matrix (dQ/dt terms) and J_dc is the Jacobian already
assembled for the DC Newton solve. Terminal admittance Y = G + i*omega*C.
Capacitance is Im(Y) / omega.

This reuses the DC Jacobian entirely. It is roughly 60 lines once Phase 3 works.
