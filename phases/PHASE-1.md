# Phase 1: Equilibrium Poisson, 1D

**Target: 1 to 2 weeks. Docs: 01-physics, 02-numerics (nonlinear Poisson), 04-validation.**

One unknown per node. No transport. This is the well-conditioned warm-up and it
establishes the assembly and Newton patterns used everywhere after.

## Scope

Substitute Boltzmann statistics into Poisson to eliminate n and p:

    lap(psi) = -(n_i*exp(-psi/V_T) - n_i*exp(psi/V_T) + N)

1. `physics/statistics.py`, Boltzmann forms
2. `discretize/poisson.py`, residual and Jacobian, 1D
3. `discretize/boundary.py`, ohmic Dirichlet using the asinh form
4. `solve/newton.py`, damped Newton, step limiting on psi
5. `device/builder.py` and `device/pn_diode.py`
6. Doping profiles as callables: uniform, step, gaussian, erfc, composable

## Acceptance criteria

- Newton converges in under 10 iterations from the charge-neutral initial guess,
  quadratically in the last 3 iterations. Log the residual history and confirm
  the quadratic tail visually.
- **Built-in potential**, 1e16/1e16 abrupt junction, matches
  `V_T*ln(Na*Nd/n_i^2)` to under 0.5 percent (expect 0.7143 V). An earlier
  revision said 0.695 V here, which is the same formula at the superseded
  n_i = 1.45e10 and contradicts the formula printed beside it by 2.8 percent,
  five times the tolerance. The formula wins.
- **Depletion width** matches the depletion approximation to under 3 percent at
  0, -1, and -5 V reverse bias
- **Debye decay**: potential from a doping step decays with the local L_D, fit to
  under 1 percent
- **Invariants**: `n*p == n_i^2` everywhere to 1e-8; bulk charge neutrality to
  1e-6; n > 0 and p > 0 at every node
- Mesh refinement study: error against the analytic V_bi decreases with h
- Jacobian verified against finite difference on a 20 node mesh

## Do not

- Add continuity equations
- Add recombination
- Apply a bias other than through the contact BC (no continuation yet)

## Definition of done

A PN diode at equilibrium, band diagram plot committed to the README, all Tier 2
analytic tests for equilibrium green.
