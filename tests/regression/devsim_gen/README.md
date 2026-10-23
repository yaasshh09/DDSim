# DEVSIM golden data generation

Tier 4 of `docs/04-validation.md` compares ddsim against DEVSIM. This directory
holds everything needed to reproduce the curves in `data/golden/`, because
golden data whose provenance is a memory is not evidence.

## Why this is not part of the test suite

DEVSIM is a compiled solver with an Intel MKL dependency. Nothing in the
project venv imports it and nothing in CI installs it. The generator runs by
hand, writes CSV files, and those files are what the suite reads. That keeps
tier 4 running everywhere while the tool that produced the reference runs in
one place.

## Environment

The project venv is Python 3.14. DEVSIM publishes `cp39-abi3` wheels, which are
stable ABI and therefore install on 3.14 without a version specific build, so a
separate interpreter is not needed after all. A separate venv is still used, to
keep MKL out of the project environment:

```
py -3.14 -m venv .venv-devsim
.venv-devsim/Scripts/python.exe -m pip install devsim "mkl==2024.2.2"
```

DEVSIM needs a pinned MKL. It loads `mkl_rt.2.dll` and says so when it cannot:
"Could not find Intel MKL. The maximum tested version is mkl_rt.2.dll". MKL
2026.1 ships `mkl_rt.3.dll` and DEVSIM will not start against it. 2024.2.2 is
the newest release that still ships version 2 of the runtime.

`.venv-devsim/` is gitignored.

## Running

From the repository root:

```
.venv-devsim/Scripts/python.exe tests/regression/devsim_gen/generate_diodes.py
```

That writes all three diode curves into `data/golden/`, solving each on the
reference mesh and again on a mesh with every spacing halved so the header can
carry a mesh convergence figure. DEVSIM prints its Newton history to stdout, so
redirect it somewhere if the progress lines are what you want to read.

Options: name one or more benchmarks to generate a subset, `--no-mesh-check` to
skip the second solve, `--out` to write somewhere other than `data/golden`.

`generate_mos_cv.py` and `generate_mosfet.py` take the same arguments.

## The small signal generator runs every solve in its own process

`generate_ac.py` writes benchmarks 11 to 13: the MOS capacitor and the long diode through `solve(type="ac")`, and the resistor bar through `solve(type="noise")` with the diffusion source integrated against the impedance field. It starts one subprocess per solve, for two reasons.

1. `delete_circuit()` in devsim 2.11 deletes the circuit and then raises an error with an empty message, so a second device can't be set up cleanly in the same process.
2. `MKL_NUM_THREADS` only takes effect before MKL loads, and the thread count is one of the things the generator varies on purpose.

Each benchmark lists its runs in `devsim_runs` as (threads, extended precision) pairs. The golden stores the first and a `spread` column with the worst relative change over the rest. On the diode and the bar the spread is 5e-5 and 0. On the capacitor in inversion it reaches 14.8 percent, because the inversion layer is tied to the body only through generation and devsim's unscaled matrix loses that link to roundoff. The test widens its tolerance to three times the spread where that's bigger, and the mesh convergence figure in the header skips points whose spread is above 1e-3, since there the mesh is not what moves the answer.

## The transient generator calls BDF2 as TR-BDF2's second stage

`generate_transient.py` writes benchmark 14, reverse recovery on benchmark 12's diode. A voltage source drives the anode through a series resistor, and the run takes fixed steps from the DC state at the forward bias. devsim's `transient_bdf2` is the second stage of TR-BDF2, so it takes a `gamma` that defaults to 1, and at 1 the step's dt coefficient is zero and the first solve comes back nan. Uniform BDF2 with step h is that second stage at `gamma=0.5` over `tdelta=2h`. I checked this on a bare RC circuit, where it matches BDF2 worked by hand to ten digits. The first step is `transient_bdf1`, the same as ddsim's.

Two more things. The steps use relative error 1e-8, because at 1e-10 devsim's update cycles around 5e-9 in its own roundoff once the junction starts to discharge, and the solve gives up at step 108. And the DC state is reached with a 1e-6 ohm cm^2 resistor in place, then the resistor is raised to its real value and the source corrected until the anode sits at the forward bias again. The golden header records that resistance, and ddsim reads it from there, so both tools drive the same circuit.

## The MOSFET generator decides convergence itself

`generate_mosfet.py` does not trust what a devsim `solve` reports, and neither
should anything else on a device like this. A solve returns after a single
damped step and reports its relative and absolute update as exactly zero, which
is a statement about the size of that step rather than about the residual.
devsim's own `python_packages/ramp.py` takes that at face value, one solve per
bias step, and run that way this device hands back a transfer curve reading
-2.209939 A/cm at 0.1 V of gate with every step reported as a success.

So `settle` calls solve until the potential stops moving to 1e-9 V and then
checks the state is physically possible at all, and `solve_once` underneath it
backs the relative tolerance off 1e-8, 1e-6, 1e-4 until devsim stops refusing.
That last part is not sloppiness: at 1e-10 devsim refuses a solve whose relative
update is cycling at 2.17e-09 in its own roundoff, and the refusal is what used
to stop the gate walk dead at 0.26 V. See the dated rows in
`docs/07-decisions.md`.

## The MOSFET benchmarks come in two model sets

The diodes and the capacitors match model for model. The MOSFETs are run twice.

Benchmarks 6 to 9 use the reduced set: Boltzmann statistics and constant
mobility, without Arora, Caughey-Thomas or Lombardi. That still compares the 2D
transport, the geometry and the electrostatics, which is what threshold voltage
and DIBL are made of, but it does not compare the mobility models, and the
drain current it compares is not the drain current the README roll-off table
reports.

Benchmark 10 is the same five devices on the same meshes at the full Phase 5
stack: Fermi-Dirac by Joyce-Dixon, Arora corrected by Lombardi at the node and
wrapped in Caughey-Thomas on the edge. It is the only benchmark in the tier
where either mobility model or the statistics meets an implementation that is
not ddsim's, and it is what the README roll-off figure overlays. Pick a set with
the `models` field of `MosfetBenchmark`; the generator branches on it in
`build_physics` and the header records which one ran. See the dated rows in
`docs/07-decisions.md`.

Three things about the full stack in devsim that cost a day between them:

1. devsim chains a derivative through an edge model it is given by name, and
   does not chain one through a node model evaluated at `@n0`. Asked for
   `diff(Potential_n@n0, Potential@n0)` it returns exactly zero, on every edge,
   with no complaint, and the Newton then diverges from the first step. The
   derivative edge models of `vdiff_n` and `vdiff_p` are written out by hand
   for that reason, the same way devsim's own `simple_dd` writes out
   `vdiff:Potential@n0`.

2. A field magnitude written as `pow(E^2, 0.5)` differentiates to
   `E/pow(E^2, 0.5)`, which is a nan wherever the field is exactly zero, and in
   a neutral bulk that is most edges. Caughey-Thomas is written in terms of the
   squared ratio with devsim's own `+ 1e-300` guard instead.

3. Lombardi needs the field normal to the interface, which no edge carries, so
   it is evaluated at nodes and frozen inside a solve, with `settle` refreshing
   it between solves until the potential stops moving. That is the same outer
   fixed point ddsim runs, for the same reason.

## What is matched, and what is not

`parameters.py` is the single definition of the benchmark devices and of every
physical constant, and it is imported by both the generator and the test. The
constants are literal copies of `ddsim.core.constants`, and
`tests/regression/test_devsim_diodes.py::test_generator_constants_mirror_ddsim`
fails if any of them drifts, so a change on the ddsim side surfaces as a stale
golden file rather than as a quiet two percent.

DEVSIM's own `simple_physics` helpers are used for the equations, with every
parameter overridden. Left alone they would run eps_r = 11.1, q = 1.6e-19,
mu_n = 400 and mu_p = 200, none of which are ddsim's values.

Matched: Boltzmann statistics, Scharfetter-Gummel with the Einstein relation,
constant mobility, SRH with midgap traps and doping dependent Scharfetter
lifetimes, ideal ohmic contacts, the abrupt junction on a mesh node, and every
constant in `parameters.py`.

The mesh is deliberately not matched. Both codes solve the same continuum
problem on their own grid, which is the point. The generator reports how much
the answer moves when the DEVSIM mesh is halved so the size of that difference
is on the record.

## Regenerating after a physics change

Any change to a mirrored constant, to a model, or to a benchmark definition
makes the stored curves stale. The suite says so in three different ways: the
mirror test, the header versus benchmark test, and the comparison itself.
Regenerate, then read the diff on the CSV before committing it. A golden file
that changes for a reason nobody can state is how a regression suite stops
being one.
