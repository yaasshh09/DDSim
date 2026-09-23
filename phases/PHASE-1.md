# Phase 1: Equilibrium Poisson, 1D

**Target: 1 to 2 weeks. Docs: 01-physics, 02-numerics (nonlinear Poisson), 04-validation.**

One unknown per node and no transport. This is the well conditioned warm-up,
and it sets the assembly and Newton patterns used everywhere after it.

## Scope

Substitute Boltzmann statistics into Poisson to eliminate n and p:

    lap(psi) = -(n_i*exp(-psi/V_T) - n_i*exp(psi/V_T) + N)

1. `physics/statistics.py`, the Boltzmann forms
2. `discretize/poisson.py`, residual and Jacobian, 1D
3. `discretize/boundary.py`, the ohmic Dirichlet condition in asinh form
4. `solve/newton.py`, damped Newton with step limiting on psi
5. `device/builder.py` and `device/pn_diode.py`
6. Doping profiles as callables: uniform, step, gaussian and erfc, all
   composable

## Acceptance criteria

- Newton converges in under 10 iterations from the charge neutral initial
  guess, quadratically over the last 3. Log the residual history and check the
  quadratic tail by eye.
- **Built-in potential** of a 1e16/1e16 abrupt junction matches
  `V_T*ln(Na*Nd/n_i^2)` to under 0.5 percent (expect 0.7143 V). An earlier
  version said 0.695 V here. That's the same formula at the old
  n_i = 1.45e10, and it contradicted the formula printed right next to it by
  2.8 percent, five times the tolerance. The formula wins.
- **Depletion width** matches the depletion approximation to under 3 percent
  at 0, -1 and -5 V reverse bias
- **Debye decay**: the potential from a doping step decays with the local
  L_D, fitted to under 1 percent
- **Invariants**: `n*p == n_i^2` everywhere to 1e-8, bulk charge neutrality to
  1e-6, and n > 0 and p > 0 at every node
- A mesh refinement study: the error against the analytic V_bi falls with h
- The Jacobian verified against finite differences on a 20 node mesh

## Do not

- Add continuity equations
- Add recombination
- Apply a bias any way other than through the contact boundary condition (no
  continuation yet)

## Definition of done

A PN diode at equilibrium, its band diagram plot committed to the README, and
every Tier 2 analytic test for equilibrium green.
