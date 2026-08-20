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

Branch structure. Use these thresholds as a starting point and tune against
double precision:

| Range | Evaluation |
|---|---|
| x < -80 | return -x (exp overflow otherwise) |
| -80 <= x < -1e-4 | x / expm1(x) |
| \|x\| <= 1e-4 | series: 1 - x/2 + x^2/12 - x^4/720 |
| 1e-4 < x <= 80 | x / expm1(x) |
| x > 80 | return 0.0 (underflow, mathematically x*exp(-x)) |

Use `numpy.expm1`, never `exp(x) - 1`. The latter loses all precision near zero.

You also need the derivative dB/dx for the Newton Jacobian:

    B'(x) = (exp(x) * (1 - x) - 1) / (exp(x) - 1)^2

Same branch care applies. Near zero use the series -1/2 + x/6 - x^3/180.
Consider verifying B' against complex-step differentiation in a test, which is
exact to machine precision and catches algebra errors that finite differences
would hide.

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

Reuse the symbolic factorization across Newton steps if the sparsity pattern is
unchanged. It is, so do it. Significant speedup for free.

## Convergence criteria

Check all three, not just one:

1. Update norm: max |dpsi| < 1e-10 (scaled), and relative change in n, p < 1e-8
2. Residual norm: ||F|| < tol, absolute, on the scaled system
3. Current continuity: max deviation of (Jn + Jp) across nodes < 1e-6 relative

Criterion 3 is physical rather than algebraic and it catches failures the other
two miss. Report all three at every step in verbose mode.

## Small-signal AC (Phase 4, for C-V)

Do not time-step. Linearize around the converged DC solution and solve the
complex-valued system at frequency omega:

    (J_dc + i*omega*M) * x = b

where M is the mass matrix (dQ/dt terms) and J_dc is the Jacobian already
assembled for the DC Newton solve. Terminal admittance Y = G + i*omega*C.
Capacitance is Im(Y) / omega.

This reuses the DC Jacobian entirely. It is roughly 60 lines once Phase 3 works.
