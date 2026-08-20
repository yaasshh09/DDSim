# DDSim

A drift-diffusion semiconductor device simulator written from scratch. Solves the
Van Roosbroeck system (Poisson + electron/hole continuity) self-consistently to
produce device I-V and C-V characteristics from geometry and doping alone.

This is layer 2 of a 4-layer solver stack:

| Layer | Project | Solves | Status |
|---|---|---|---|
| 0 | AtomSIM | Schrodinger, isolated atom | done |
| 1 | band module | Bloch states, E(k), m* | optional, later |
| 2 | **DDSim (this repo)** | Van Roosbroeck, carrier transport | active |
| 3 | SPICE | MNA + Newton, circuit level | next |

The stack thesis: every material parameter DDSim consumes should ultimately be
computed by a layer below it, and every device model SPICE consumes should be
extracted from DDSim. Keep that seam clean.

## Success criterion

Sweep MOSFET gate length from 1 um down to 50 nm and observe threshold voltage
roll-off, DIBL, and velocity saturation **emerge from the physics** with zero
empirical fitting. If any short-channel effect is hardcoded or fitted, the
project has failed regardless of how good the plots look.

## Read before writing code

Load only what the current task needs. Do not load all of these at once.

- `docs/01-physics.md` - governing equations, closure models, what is an
  approximation and why
- `docs/02-numerics.md` - Scharfetter-Gummel, Bernoulli, scaling, Gummel/Newton,
  continuation. **The single most important file.**
- `docs/03-architecture.md` - module layout, the Field type, invariants
- `docs/04-validation.md` - analytic test cases and DEVSIM regression
- `docs/05-pitfalls.md` - known traps, read this when something diverges
- `docs/06-constants.md` - silicon constants, single source of truth
- `phases/PHASE-N.md` - scope and acceptance criteria for the current phase

## Working agreement

**TDD is mandatory.** This codebase is a numerics project. A wrong sign converges
to a plausible-looking wrong answer. Every physical quantity has an analytic limit
somewhere. Write the test against that limit first.

**Never skip a phase.** Phases are ordered by conditioning difficulty, not by
feature appeal. Phase 3 (full Newton + continuation) is where the real work is.
Do not start 2D until 1D is fully validated.

**Scaled vs physical units never mix.** Every array carries its scaling state in
the type system. See `docs/03-architecture.md`. This is the number one source of
silent bugs in drift-diffusion codes.

**Prefer explicit over clever.** The Jacobian assembly should be readable by a
device engineer who does not know Python. No vectorization tricks that obscure
which term is which.

**When a solve diverges, do not add damping until you have checked signs.**
Damping hides sign errors. Run the invariant tests in `docs/04-validation.md`
first. Current continuity in 1D steady state is the strongest single check.

## Session protocol

1. Read `PROGRESS.md` first. It is the state of the world.
2. Read the active `phases/PHASE-N.md`.
3. Load only the docs that phase references.
4. Work. Commit in small units with the invariant that tests pass.
5. Append to `PROGRESS.md` before ending: what landed, what broke, what is next,
   any physics decision made and why.

## Stack

- Python 3.11+, NumPy, SciPy sparse (`scipy.sparse.linalg.splu`)
- pytest for everything
- matplotlib for plot generation in tests, headless
- Frontend deferred to Phase 5. FastAPI + React, mirroring AtomSIM's binary
  point transmission architecture. Do not build UI before the solver is right.

## Style

- No em dashes in any prose, code comment, or docstring.
- Symbols in code match `docs/01-physics.md` exactly. `psi`, `n`, `p`, `phi_n`,
  `phi_p`, `Nd`, `Na`. Do not invent alternate names.
- Units in every docstring, in square brackets, e.g. `[cm^-3]`, `[V]`, `[A/cm^2]`.
- CGS-adjacent convention: lengths in cm, concentrations in cm^-3. This matches
  the semiconductor literature. Do not switch to SI lengths.


## .gstack

to use the /browse skill from gstack for all web browsing, never use mcp__claude-in-chrome__* tools, and lists the available skills: /office-hours, /plan-ceo-review, /plan-eng-review, /plan-design-review, /design-consultation, /design-shotgun, /design-html, /review, /ship, /land-and-deploy, /canary, /benchmark, /browse, /connect-chrome, /qa, /qa-only, /design-review, /setup-browser-cookies, /setup-deploy, /setup-gbrain, /retro, /investigate, /document-release, /document-generate, /codex, /cso, /autoplan, /plan-devex-review, /devex-review, /careful, /freeze, /guard, /unfreeze, /gstack-upgrade, /learn. Then ask the user if they also want to add gstack to the current project so teammates get it.

as fable 5 is not usable without usage credits, run opus vhigh or max (ALONG WITH THE "fable-brain.md" DO NOT FORGET) for heavylifting of the code and physics and heavy reasoning and heavy thinking. For light thinking and reasoning, use opus low/medium or sonnet high

Never use emdashes. Ensure to humanize all texts and make all texts in first person.

Refer to fable-brain.md when using opus and after finishing large tasks to review

ALWAYS ENSURE THAT ALL THE COMMIT MESSAGES FOR THE WHOLE PROJECT ARE HUMANIZED AND SHORT AND SIMPLE

Never push to github on your own and never add yourself as contributor to any project. I am the standalone sole project creator.