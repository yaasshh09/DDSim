# Phase 7: Browser frontend and live solver telemetry

**Target: 2 to 3 weeks. Docs: 03-architecture (frontend, module boundaries), 02-numerics (what the telemetry is reporting).**

The solver stops being a thing only I can run. Someone opens a browser, describes
a device by geometry and doping, presses solve, and watches the Newton iteration
converge in front of them. Nothing in the UI computes physics. It draws what the
solver sends.

Runs after Phase 5. It can run before Phase 6, which is blocked on the SPICE
project existing.

## Why this is not a plotting page

A MOSFET Id-Vg sweep is minutes of wall clock across dozens of bias points, each
one a continuation ladder of Newton solves. Request and response cannot express
that. The unit of work is a job: submitted, streaming while it runs, cancellable.
That single decision drives everything else in this phase.

## Scope

1. `solve/newton.py` gains an optional per-iteration callback. `newton_solve`
   already appends to `residual_history` and `update_history` inside the loop, so
   the hook goes exactly where the append is. Default `None`. With no callback the
   solver must be bit for bit what it is today, measured the way the edge list
   refactor was measured.
2. `ddsim/api/`, FastAPI. Job submit over HTTP, telemetry over WebSocket, cancel.
   The package may only call the public functions the CLI would call. The import
   graph test extends to cover it: `api/` imports from `device/`, `extract/` and
   `solve/`, and nothing imports `api/`.
3. `ddsim/cli.py` with a `ddsim serve` command. docs/03-architecture.md already
   names this file and it does not exist yet.
4. Binary point transmission, the same architecture as AtomSIM. Fields cross the
   socket as float32 typed arrays behind a small JSON header, not as JSON numbers.
   A 200 by 100 node device is 20000 points per field; JSON costs roughly ten
   times the bytes and parses slowly enough to be visible.
5. Three telemetry streams over one socket:
   - per Newton iteration: iteration index, residual per equation family, update
     norm per family, the damping factor, whether the step was limited
   - per continuation step: the bias, the step size, converged or retried
   - per sweep point: the finished I-V or C-V point, so the curve draws itself
6. Field frames on completion and on request while running: psi, n, p, and the
   current density vector field. The client asks for a cadence and the server
   drops frames rather than making the solver wait.
7. Frontend in the lab-instrument idiom docs/03-architecture.md prescribes. Filled
   contour plots, current density streamlines, a draggable cutline producing a
   band diagram along it, log-scale toggles, and a live residual plot that is an
   honest picture of the solve rather than a spinner.
8. Device definition in the browser, taking the same config the CLI takes:
   geometry, doping profiles, contacts, which models are on, the bias plan.
   Nothing hardcoded per device.
9. Failure display. A diverged solve shows the residual stalling and names the
   equation family that stalled. It does not spin forever and it never reports a
   success it did not get.

## Acceptance criteria

- Someone who cloned the repo runs one command and has a working page. No build
  step that is not in the README.
- All three device classes run from the browser and agree bit for bit with what
  pytest gets on the same inputs: diode I-V, MOS C-V, MOSFET Id-Vg.
- The residual plot updates while the solve runs, not after it finishes. Asserted
  by requiring the first telemetry frame to arrive before the job completes.
- Telemetry off is inert. With no callback, residual history, iteration counts and
  terminal currents are bit for bit unchanged on one device from each class.
- Cancel actually stops the solve and the process is idle afterwards.
- Nothing in the frontend computes a physical quantity. Grep the client for `exp`,
  `log` and any constant in docs/06-constants.md and find nothing.
- `api/` adds no branch that changes a solved number.

## Do not

- Do not put physics in JavaScript. Every number on the screen came from the
  solver.
- Do not add a database, accounts, or multi-user anything. One local process.
- Do not stream a field on every Newton iteration by default. It will dominate the
  runtime of a well-conditioned solve.
- Do not start before Phase 5's acceptance criteria pass. A beautiful UI on a
  solver with a sign error is worse than a CLI that is correct.

## Testing

TDD applies here the same as everywhere, and the four tiers map onto it:

- Contract tests on the API schema, request and response, with no browser
  involved.
- An invariant test that the callback is inert, which is the same bit for bit
  argument the edge list refactor used.
- The module boundary test, extended to `api/`.
- One headless browser smoke test: submit a diode solve, assert a telemetry frame
  arrives before completion. This is the only new dev dependency the phase needs.

## Honest limits, state these in the README

Single user, local, no authentication, bound to localhost. This is an instrument,
not a service. A 50 nm MOSFET sweep takes minutes and the page says so rather than
pretending it is interactive. The browser shows the solver's answer and cannot
check it. The DEVSIM regressions in CI are what check it.

## Definition of done

A short screen capture in the README of a MOSFET Id-Vg solving live, residual
falling per iteration and the curve drawing point by point, plus the honest limits
section above.
