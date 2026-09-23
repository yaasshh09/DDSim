# Phase 7: Browser frontend and live solver telemetry

**Target: 2 to 3 weeks. Docs: 03-architecture (frontend, module boundaries), 02-numerics (what the telemetry is reporting).**

This is where the solver stops being something only I can run. Someone opens
a browser, describes a device by its geometry and doping, presses solve, and
watches Newton converge right in front of them. Nothing in the UI computes
physics. It draws what the solver sends.

It runs after Phase 5. It can go before Phase 6, which is blocked until the
SPICE project exists.

## Why this is not a plotting page

A MOSFET Id-Vg sweep is minutes of wall clock across dozens of bias points,
each one a continuation ladder of Newton solves. Request and response can't
express that. The unit of work is a job: submitted, streaming while it runs,
and cancellable. That one decision drives everything else in this phase.

## Scope

1. `solve/newton.py` gets an optional per iteration callback. `newton_solve`
   already appends to `residual_history` and `update_history` inside the
   loop, so the hook goes exactly where the append is. The default is `None`,
   and with no callback the solver has to be bit for bit what it is today,
   measured the same way the edge list refactor was.
2. `ddsim/api/`, in FastAPI: job submit over HTTP, telemetry over a
   WebSocket, and cancel. The package may only call the public functions the
   CLI would call. The import graph test extends to cover it: `api/` imports
   from `device/`, `extract/` and `solve/`, and nothing imports `api/`.
3. `ddsim/cli.py`, with a `ddsim serve` command. docs/03-architecture.md named
   this file before it existed.
4. Binary point transmission, the same architecture as AtomSIM. Fields cross
   the socket as float32 typed arrays behind a small JSON header, not as JSON
   numbers. A 200 by 100 node device is 20000 points per field, and JSON costs
   roughly ten times the bytes and parses slowly enough to notice.
5. Three telemetry streams over one socket:
   - per Newton iteration: the iteration index, the residual per equation
     family, the update norm per family, the damping factor, and whether the
     step was limited
   - per continuation step: the bias, the step size, and whether it converged
     or got retried
   - per sweep point: the finished I-V or C-V point, so the curve draws itself
6. Field frames on completion, and on request while running: psi, n, p and
   the current density vector field. The client asks for a cadence, and the
   server drops frames rather than make the solver wait.
7. A frontend in the lab instrument style docs/03-architecture.md describes:
   filled contour plots, current streamlines, a draggable cutline that gives a
   band diagram along it, log scale toggles, and a live residual plot that
   honestly shows the solve instead of a spinner.
8. Device definition in the browser, taking the same config the CLI takes:
   geometry, doping profiles, contacts, which models are on, and the bias
   plan. Nothing hardcoded per device.
9. Showing failure. A diverged solve shows the residual stalling and names
   the equation family that stalled. It doesn't spin forever, and it never
   reports a success it didn't get.

## Acceptance criteria

- Someone who clones the repo runs one command and has a working page, with
  no build step that isn't in the README.
- All three device classes run from the browser and agree bit for bit with
  what pytest gets on the same inputs: diode I-V, MOS C-V, MOSFET Id-Vg.
- The residual plot updates while the solve runs, not after it finishes. The
  test for that requires the first telemetry frame to arrive before the job
  completes.
- Telemetry off is inert. With no callback, the residual history, iteration
  counts and terminal currents are bit for bit unchanged on one device from
  each class.
- Cancel really stops the solve, and the process is idle afterwards.
- Nothing in the frontend computes a physical quantity. Grep the client for
  `exp`, `log` and every constant in docs/06-constants.md and find nothing.
- `api/` adds no branch that changes a solved number.

## Do not

- Don't put physics in JavaScript. Every number on the screen came from the
  solver.
- Don't add a database, accounts, or anything multi-user. One local process.
- Don't stream a field on every Newton iteration by default. It would
  dominate the runtime of a well conditioned solve.
- Don't start before Phase 5's acceptance criteria pass. A beautiful UI on a
  solver with a sign error is worse than a CLI that's correct.

## Testing

TDD applies here like everywhere else, and the four tiers map onto it:

- Contract tests on the API schema, request and response, with no browser
  involved.
- An invariant test that the callback is inert, the same bit for bit argument
  the edge list refactor used.
- The module boundary test, extended to `api/`.
- One headless browser smoke test: submit a diode solve and assert a
  telemetry frame arrives before it completes. That's the only new dev
  dependency the phase needs.

## Honest limits, state these in the README

Single user, local, no authentication, bound to localhost. This is an
instrument, not a service. A 50 nm MOSFET sweep takes minutes, and the page
says so instead of pretending to be interactive. The browser shows the
solver's answer and can't check it. The DEVSIM regressions in CI are what
check it.

## Definition of done

A short screen capture in the README of a MOSFET Id-Vg solving live, with the
residual falling per iteration and the curve drawing point by point, plus the
honest limits section above.

---

# Phase 7, part two: a teaching instrument and a sandbox

**Added 2026-09-16.** The page is for university students as much as for me.
It has to explain everything on it, and it has to be something a student can
play with. The instrument above is the foundation, and none of it changes.
What gets added is a layer that tells the student what they're looking at and
why, plus the freedom to build devices and poke at them.

## Who it is for

It's layered, so one page works for a second year student and a PhD student.
Every explanation opens with a plain paragraph someone in a first
semiconductors course can follow, then an "in more depth" part with the
actual equations: Poisson, drift diffusion, Scharfetter-Gummel, the Newton
linearisation. The depth part links to the section of docs/01-physics.md or
docs/02-numerics.md it condenses.

## Where the words live

Explanations sit next to the thing they explain, and tests keep them there.

- **Knobs** are explained by the docstrings of the Python constructors and
  sweeps they come from. `/api/schema` already reads those signatures; it
  grows a plain explanation and the unit for each knob, read from the `Args:`
  block. A knob can't reach the page without one, because a test refuses it.
- **Plots, physics and numerics** are short markdown files under
  `ddsim/api/static/learn/`, one per topic, each in the two layers above.
- **Guided experiments** are files under `ddsim/api/lessons/`, one per lesson:
  the device and sweep to start from, the steps, what to look for, the
  explanation, and the name of the claim the lesson makes.
- **Every lesson's claim is asserted by pytest on the real solve.** "Reverse
  bias widens the depletion region" is a test that solves the lesson's own
  device at the lesson's own biases and measures the width. A lesson that
  teaches something the solver doesn't do fails CI.
- Equations render with KaTeX and markdown with a small renderer, both
  vendored into `static/` with their licences, so the page works offline and
  there's still no build step.

## Stages

Each stage ships something usable on its own, in this order.

### Stage 1: explain everything

1. Every knob, plot, legend entry and status message has an explanation one
   click away, in the two layers.
2. The physics: what psi, n and p are, band diagrams, quasi-Fermi levels,
   depletion, recombination, mobility models, velocity saturation.
3. The numerics, because a student should learn how a simulator works and
   not only what a diode does: the mesh, scaling, Scharfetter-Gummel, Gummel
   versus Newton, damping, continuation, and what the residual plot honestly
   shows, per equation family.
4. Current streamlines, the last item of part one, land here, because current
   flow is one of the things that most needs explaining. The server sends the
   node current density vectors it computes from the edge fluxes. The client
   traces streamlines through that field, which is geometry on a vector it was
   handed and not physics, the same argument as the log axis. That relaxation
   gets its own row in docs/07-decisions.md.
5. A cutline across a 2D field, giving the band diagram along it, with the
   band edges computed on the server.

### Stage 2: sandbox basics

1. **Compare runs.** Every finished run stays on the plot as a faded overlay
   until cleared, labelled by the knobs that differ from the run before. The
   difference compares the requests, not the physics.
2. **Live sliders**, on the 1D devices only. Moving a slider cancels the job
   in flight and submits a new one, so jobs never pile up. MOSFETs keep the
   solve button, and the page says why: a 2D solve takes seconds to minutes.
3. **Fast presets.** Coarse meshes that solve in seconds, marked as coarse,
   with a one click switch to the converged mesh and a note on what changes.

### Stage 3: guided experiments

Five to start with, each with a tested claim:

1. The pn junction in equilibrium: the built in potential, depletion, the
   band diagram.
2. Forward and reverse bias: the diode equation, the ideality factor, and
   depletion width against bias.
3. The MOS capacitor: accumulation, depletion, inversion, and the C-V curve.
4. The MOSFET: threshold, the transfer curve, subthreshold slope.
5. Short channel effects: roll-off and DIBL emerging as the gate shrinks,
   and what this solver leaves out at 50 nm (quantum confinement above all),
   said plainly.

A lesson sets up its device, tells the student what to change and what to
watch, then explains what they saw. The student can leave the lesson at any
point and keep the device as a sandbox.

### Stage 4: the 1D device builder

Stack doped regions left to right, as many as you like, each with a length
and a doping, and contacts at both ends: pn, pin, p+n, npn and whatever a
student invents. It's built on the existing 1D mesh and doping profiles,
graded at every junction. Refusals explain themselves: a region shorter than
the mesh can resolve, a doping outside the range docs/01-physics.md gives for
the mobility and recombination models, a stack with no junction to grade
toward.

A device can be saved to and loaded from a JSON file, so students can hand
each other devices.

### Stage 5: drawing 2D devices

Rectangles only, because the 2D mesh is a tensor product: draw rectangles of
silicon and oxide, paint rectangular doping regions (uniform or Gaussian), and
place contacts along boundary segments, ohmic on silicon and gates on oxide
with a work function. Mesh lines fall on every rectangle edge and grade
toward every doping edge. Curved and slanted shapes are out, and the page says
why.

Guard rails before a solve, each refusal naming its reason: a silicon island
no ohmic contact touches floats; a gate sitting on silicon is a Schottky
contact this solver doesn't model; a feature smaller than the mesh resolves;
a node count over the budget the page states. A solve that fails anyway shows
the stalled family as usual, plus a plain explanation of what that usually
means.

Save and load as JSON, same as stage 4.

## Acceptance criteria for part two

- Every knob `/api/schema` offers carries a non empty explanation, and every
  numeric knob a unit. A knob without one fails a test.
- Every plot and legend entry on the page has an explanation file, and every
  link from an explanation into docs/ resolves. Both are tests.
- Every lesson's claim has a pytest check that runs the lesson's own device,
  and every lesson file names a claim that has a check.
- Compare runs keeps earlier curves exactly as drawn: an overlay is the
  earlier run's numbers, never recomputed.
- A slider moved five times in a second leaves one job running, not five, and
  the process is idle once it finishes. Asserted in the browser smoke test.
- A 1D stack drawn as the Phase 2 diode reproduces `pn_diode`'s I-V. If it
  isn't bit for bit, the tolerance goes in docs/07-decisions.md with the
  reason.
- A 2D drawing of the benchmark MOS capacitor and of the benchmark nmos
  reproduces the `mos_cap` C-V and `nmos` Id-Vg, to a recorded tolerance.
- Each guard rail has a test drawing that trips it and asserts the reason.
- Results on a device a student built are labelled on screen as the
  validated solver on an unvalidated structure.
- Still true from part one: nothing in the client computes a physical
  quantity, and nothing added here changes a solved number.

## Do not, for part two

- Don't let an explanation say something the solver doesn't do. Where the
  solver approximates, the explanation names the approximation.
- Don't hide failure behind friendliness. A device that won't solve says so,
  and says why when it can.
- Don't load anything from a CDN. The page works with no network.
- Don't add accounts or server side storage for student devices. The files
  students download are the storage.
