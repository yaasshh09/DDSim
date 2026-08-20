# DDSim

[![CI](https://github.com/yaasshh09/DDSim/actions/workflows/ci.yml/badge.svg)](https://github.com/yaasshh09/DDSim/actions/workflows/ci.yml)

A drift-diffusion semiconductor device simulator written from scratch. It solves
the Van Roosbroeck system, Poisson plus the electron and hole continuity
equations, self consistently, and produces device I-V and C-V characteristics
from geometry and doping alone.

Layer 2 of a four layer solver stack: AtomSIM solves the isolated atom, a band
module supplies effective masses, DDSim solves carrier transport, and a SPICE
layer solves the circuit. Every material parameter DDSim consumes should
eventually be computed by a layer below it.

## Where it is

Phase 1 of 6. A PN diode at thermal equilibrium, solved by damped Newton on the
nonlinear Poisson equation, validated against closed form device physics.

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaling, the Field type, Bernoulli, meshes, linear solver | done |
| 1 | Equilibrium Poisson in 1D, PN diode | done |
| 2 | Scharfetter-Gummel continuity, Gummel iteration | next |
| 3 | Full Newton, bias continuation, I-V | |
| 4 | 2D, MOS capacitor, C-V | |
| 5 | MOSFET, gate length sweep | |
| 6 | Compact model extraction for SPICE | |

## PN diode at equilibrium

1e16 / 1e16 abrupt junction, 4 um, 801 nodes. Everything below emerges from
solving Poisson with Boltzmann statistics. Nothing is fitted.

![PN diode at equilibrium](docs/images/pn_diode_equilibrium.png)

The bands bend by exactly the built-in potential, the Fermi level is flat
because this is equilibrium, n and p cross at n_i precisely at the metallurgical
junction, and the field is the triangle the depletion approximation predicts.

Measured against closed form results:

| Quantity | Simulated | Analytic | Error |
|---|---|---|---|
| Built-in potential, 1e16 / 1e16 | 0.71432 V | 0.71432 V | pinned by the contacts |
| Depletion width, 0 V | 0.4248 um | 0.4298 um | 1.17 % |
| Depletion width, -1 V | 0.6574 um | 0.6659 um | 1.27 % |
| Depletion width, -5 V | 1.2102 um | 1.2157 um | 0.45 % |
| Debye decay length into the bulk | fitted | L_D | under 1 % |
| n p / n_i^2 | 1.0 | 1.0 | 3e-16 |

Newton converges in 8 iterations from the charge neutral guess, with a
quadratic tail, at every doping level from 1e14 to 1e20 cm^-3.

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/pytest
```

```python
from ddsim.device.pn_diode import pn_diode
from ddsim.device.equilibrium import solve_equilibrium

device = pn_diode(Na=1e16, Nd=1e16)
state = solve_equilibrium(device)

psi = state.psi.to_physical(device.scale).data   # [V]
print(f"V_bi = {psi[-1] - psi[0]:.4f} V")
```

## How it is kept honest

A drift-diffusion solver with a sign error does not crash. It converges cleanly
to a physically wrong answer that looks entirely plausible. Everything about the
way this repo is built is a response to that.

**Scaled and physical units cannot mix.** Every array carries its unit, its
scaling state and its mesh location, and arithmetic across any of them raises
rather than coercing. A scaled potential of 38.7 and a physical potential of
1.0 V are the same thing, and nothing else would notice them being added.

**Every quantity is tested against an analytic limit**, not against whatever the
code currently produces. The Bernoulli function is checked against an 80 digit
reference, its branch thresholds tuned by measurement rather than taken from the
docs. Jacobians are verified by complex step differentiation, which is exact.

**Tests are written first.** In a numerics project, a test written after the
code tends to assert whatever the code already does.

Validation runs in four tiers: unit tests against textbook formulas, analytic
device tests against closed form results, invariants that hold for every solve,
and regression against DEVSIM. See `docs/04-validation.md`.

## Layout

    ddsim/core/        constants, de Mari scaling, the Field type
    ddsim/mesh/        1D meshes, uniform and graded
    ddsim/physics/     pure functions: Bernoulli, carrier statistics
    ddsim/discretize/  residual and Jacobian assembly, boundary conditions
    ddsim/solve/       Newton and the linear solver, no semiconductor knowledge
    ddsim/device/      composition: geometry and doping in, a Device out
    docs/              physics, numerics, architecture, validation, constants
    phases/            scope and acceptance criteria per phase

`PROGRESS.md` is the state of the world, including a log of every physics
decision and every place the code knowingly disagrees with a reference.

## Success criterion

Sweep MOSFET gate length from 1 um down to 50 nm and watch threshold voltage
roll-off, DIBL and velocity saturation emerge from the physics with no empirical
fitting. If any short channel effect is hardcoded or fitted, the project has
failed regardless of how good the plots look.
