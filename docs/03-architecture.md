# Architecture

## Layout

    ddsim/
      core/
        constants.py        # silicon constants, single source of truth
        scaling.py          # de Mari scale factors, to_scaled / to_physical
        field.py            # Field type, provenance and unit tracking
      mesh/
        mesh1d.py           # uniform and graded 1D
        mesh2d.py           # box-integration finite volume, Delaunay
        quality.py          # obtuse triangle detection, dual area checks
      physics/
        statistics.py       # Boltzmann, Fermi-Dirac, Joyce-Dixon
        mobility.py         # constant, Arora, Masetti, Caughey-Thomas, Lombardi
        recombination.py    # SRH, Auger
        bernoulli.py        # B(x) and dB/dx, branch handled
      discretize/
        poisson.py          # assembly + Jacobian
        continuity.py       # SG flux assembly + Jacobian
        boundary.py         # ohmic, MOS gate, reflecting
      solve/
        newton.py           # damped Newton, residual tracking
        gummel.py           # decoupled iteration
        continuation.py     # generic bias ramp driver, REUSED BY SPICE
        linear.py           # splu wrapper, symbolic factorization cache
      device/
        builder.py          # geometry + doping profile -> Device
        pn_diode.py
        mos_cap.py
        mosfet.py
      extract/
        iv.py               # I-V sweeps
        cv.py               # small-signal C-V
        params.py           # Vth, SS, DIBL, ideality extraction
        compact.py          # Phase 6: fit EKV/BSIM params for SPICE
      api/                  # Phase 7, FastAPI. Do not build early.
      cli.py                # Phase 7, includes `ddsim serve`
    tests/
      unit/                 # bernoulli, statistics, scaling, mobility
      analytic/             # depletion, Shockley, ideal MOS C-V
      invariant/            # current continuity, charge neutrality, np=ni^2
      regression/           # DEVSIM golden data
      convergence/          # mesh refinement order studies
    data/
      golden/               # DEVSIM reference curves, committed
    docs/
    phases/

## The Field type

The single most important abstraction. Ported in spirit from AtomSIM.

Every array of physical numbers carries three pieces of metadata:

1. **Unit** as a symbolic string, e.g. `"cm^-3"`, `"V"`, `"A/cm^2"`
2. **Scaling state**, an enum: `PHYSICAL` or `SCALED`
3. **Mesh location**, an enum: `NODE`, `EDGE`, or `CELL`

Rules enforced at runtime, cheaply:

- Arithmetic between two Fields of different scaling state raises. No exceptions,
  no coercion.
- Arithmetic between fields at different mesh locations raises. Node quantities
  and edge quantities are not interchangeable.
- `to_scaled()` and `to_physical()` are the only ways to change state, and they
  consult `scaling.py`, never a hardcoded factor.
- Unit strings are checked on add and subtract, and combined on multiply and
  divide. Full dimensional analysis is unnecessary; string matching on add and
  subtract catches almost everything.

Why this matters more here than in AtomSIM: de Mari scaling means every single
quantity in the codebase exists in two versions that look numerically plausible
in either form. A scaled potential of 40 and a physical potential of 1.03 V are
the same thing, and mixing them produces no error, no NaN, and no crash. It
produces a wrong answer that converges. The type system is the only defense.

Performance note: keep the underlying storage a plain NumPy array accessible as
`.data` so hot loops can bypass the wrapper. Check scaling state once at function
entry, then work on raw arrays inside.

## Module boundaries

**`core/` knows nothing about devices.** Constants, scaling, Field only.

**`physics/` is pure functions.** Takes arrays, returns arrays. No mesh, no
solver state, no device knowledge. Every function here is directly unit
testable against a textbook formula. Keep it that way.

**`discretize/` owns the mesh coupling.** Assembly functions take a mesh and a
state vector and return a residual and a Jacobian. Nothing else. They do not
solve, they do not iterate.

**`solve/` knows nothing about semiconductors.** It takes a callable returning
`(residual, jacobian)` and drives it to convergence. This is what makes
`continuation.py` reusable in SPICE without modification. Do not let a carrier
density leak into this package.

**`device/` composes.** Geometry and doping in, a `Device` object out that knows
its mesh, its regions, its contacts, and its material parameters.

**`extract/` is post-processing.** Takes converged solutions, produces numbers
and curves. No solving.

## The Device object

    Device
      .mesh          Mesh1D | Mesh2D
      .regions       list of Region (material, doping profile, geometry)
      .contacts      dict name -> Contact (type, node set, applied bias)
      .materials     dict name -> MaterialParams
      .state         State (psi, n, p as Fields) or None if unsolved

Doping profiles are callables of position, not arrays. `gaussian(peak, sigma)`,
`uniform(N)`, `erfc(...)`, composed by addition. This keeps the profile
independent of the mesh, which matters because Phase 5 refines the mesh
adaptively and the profile must be re-evaluable.

## Configuration

Device specs live in TOML or YAML, not in Python. One file per device.
Simulation settings (bias sweep, models enabled, tolerances) in a separate
section. Rationale: Phase 5 sweeps gate length across many runs, and that is a
config sweep, not a code change.

Every run writes its resolved config next to its results. Reproducibility.

## Frontend, Phase 7

A real deliverable with its own phase and its own acceptance criteria, in
`phases/PHASE-7.md`. Deferred until Phase 5 passes, not because it is optional
but because a solver with a beautiful UI and a sign error is worse than a CLI
that is correct.

The shape of it:

- FastAPI backend, same binary point transmission architecture as AtomSIM. A 2D
  field of psi, n, p plus a vector field of current density is the same shape of
  payload as isosurface data. Reuse that code.
- A solve is a job, not a request. It is submitted over HTTP, streams telemetry
  over a WebSocket while it runs, and can be cancelled. A MOSFET sweep is minutes
  of Newton solves and request-response cannot express that.
- Live telemetry is the point. Residual per equation family per Newton iteration,
  each continuation step as it lands, each sweep point as it finishes, so the
  curve draws itself and the convergence is visible rather than hidden behind a
  spinner. `solve/newton.py` takes an optional per-iteration callback for this,
  defaulting to None and bit for bit inert when unused.
- Frontend idiom: lab instrument, matching AtomSIM. Real TCAD viewers look like
  this. Filled contour plots, current density streamlines, a draggable cutline
  producing a band diagram along it, log-scale toggles everywhere.
- No physics in the client. It draws what the solver sends and computes nothing.
- Reuse the AtomSIM CSS constraint: `text-transform: uppercase` only on section
  headings. It will corrupt scientific notation and unit strings everywhere else.

## Performance

Do not optimize before Phase 5. Correctness first.

When you do: profile, do not guess. Expected hot spots in order are Jacobian
assembly, then LU factorization, then mobility evaluation. Assembly vectorizes
well over edges. If assembly is still dominant after vectorizing, Numba on the
edge loop is the next step. Do not rewrite in C++.
