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

The MKL pin matters. DEVSIM loads `mkl_rt.2.dll` and says so when it cannot:
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

Not matched, deliberately: the mesh. Both codes solve the same continuum
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
