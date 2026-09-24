# Roadmap: past DEVSIM

Phases 0 to 5 and 7 are done, and Phase 6 waits on SPICE. On the ten Tier 4
benchmarks DDSim agrees with DEVSIM 2.11. That shows DDSim is correct. It
doesn't show DDSim is better, and being better is now the goal.

DEVSIM is a general toolkit, so "better" has to be pinned to things I can
measure. That's why Phase 8 comes first: it builds the scoreboard, and every
phase after it has to move a number on it.

## What surpass means

Five axes. Each is a number or a checklist cell backed by a test, never a
sentence in a README.

| Axis | Measured as | DDSim wins when |
|---|---|---|
| Accuracy | error against an analytic limit or a Richardson extrapolated reference, at matched node count | its error at N nodes is at or below DEVSIM's on every benchmark that has a reference |
| Robustness | share of a fixed, seeded set of cold start solves that converge | strictly more converged cases than DEVSIM driven by my best hand-written ramp, not only by its stock ramp |
| Error control | share of reported results that carry an error estimate | above zero is already a lead; the target is every result |
| Speed | median wall time per converged bias point, one thread each | within 2x of DEVSIM. I don't expect Python assembly to beat C++, and I won't claim it does until it's measured |
| Features | a capability matrix, one row per capability, one test id per cell | every row DEVSIM has, plus the rows it doesn't |

The rule from CLAUDE.md still sits above all of these: nothing fitted. A
feature that needs a fitted parameter to agree with DEVSIM doesn't count.

## Where each tool stands today

DEVSIM 2.11's own API has these and DDSim doesn't. I checked the solve modes
from its docstring (`dc`, `ac`, `noise`, `transient_bdf1`, `transient_bdf2`,
`transient_tr`) and the mesh and circuit functions from its module listing in
`.venv-devsim`:

- transient simulation, three integrators
- small-signal AC at any frequency, and noise
- Gmsh meshes, unstructured, in 2D and 3D
- circuit elements attached to contacts
- cylindrical coordinates
- region interfaces, and with them heterojunctions
- user-written equations with derivatives taken for you

DDSim has these and DEVSIM doesn't ship them:

- automatic bias continuation that cold starts a MOSFET without a hand-written
  ramp. DEVSIM ships `python_packages/ramp.py`, which steps one bias and halves
  the step on failure. The scoreboard measures whether that's enough.
- a browser client that streams the residual live, with guided lessons
- a device check that refuses a bad input before solving it

Physics models are close to parity. DEVSIM's `python_packages` ship SRH,
Klaassen and Philips mobility, Philips velocity saturation and a Fermi
statistics module, and anything else is written as equations. DDSim builds
in Arora, Lombardi, Caughey-Thomas and Joyce-Dixon. Those are different model
choices, not a lead, and the capability matrix records them as such.

## The order

The phases are ordered the same way as 0 to 7, by how hard the numerics are and
by what each one needs to be checked against, not by how exciting it is.

| Phase | What | Axis it moves | Checked against |
|---|---|---|---|
| 8 | The scoreboard | all of them, as a measurement | DEVSIM, same devices, same models |
| 9 | Temperature, fixed oxide charge, incomplete ionization | features, accuracy | textbook limits in T; the flatband shift -Q_f/C_ox |
| 10 | Small-signal AC at any frequency, then noise | features | the long diode admittance G0 sqrt(1 + i omega tau); DEVSIM `ac` |
| 11 | Transient | features | the AC result from Phase 10; diode reverse recovery; DEVSIM `transient_bdf2` |
| 12 | Error estimates and adaptive refinement | error control, accuracy | Richardson extrapolation on the existing benchmarks |
| 13 | Bipolar transistor | features | the Gummel plot in low injection; DEVSIM |
| 14 | Schottky contacts and avalanche breakdown | features | thermionic emission I-V; breakdown voltage against doping |
| 15 | Unstructured 2D meshes | features, accuracy | the tensor mesh results must reappear on a triangulated rectangle |
| 16 | 3D | features, speed | a 3D device extruded from 2D must reproduce the 2D answer per unit width |
| 17 | Quantum correction | features | the Airy levels of a triangular well; DEVSIM's density gradient |
| 18 | Other materials and heterojunctions | features | the band offset at equilibrium; the homojunction limit; DEVSIM interfaces |

New Tier 4 benchmarks get numbered as they're planned: 11 to 13 in Phase 10,
14 in 11, 15 in 13, 16 and 17 in 14, 18 in 15, 19 in 16, 20 in 17, 21 in 18.
`docs/04-validation.md` lists them.

Why AC comes before transient: a small sinusoid run through the transient
solver has to give back the AC admittance at that frequency. That's two
independent methods agreeing, and it only works if AC exists first.

Why error estimates come after transient and before the new devices: every
device after Phase 12 ships with an error bar from day one, instead of getting
one added later.

Why 3D is late: it needs unstructured meshes, and a direct solver on a 3D
Jacobian runs out of memory quickly, so it also needs an iterative linear
solver. Both are big, and nothing before it depends on them.

## Phase 6 in the meantime

Phase 6 is still blocked on SPICE. Two pieces of it no longer have to wait.
The sweeps behind its scope item 1 already exist in `extract/iv.py` and
`extract/rolloff.py`; what's missing is running them over the full
(Vg, Vd, Vb, L) grid. The C tables SPICE needs for charge come out of
Phase 10. When SPICE exists, Phase 6 becomes mostly the fitting and the
netlist card.

## Rules that carry over unchanged

- TDD. Each phase lists its analytic limits. The test against the limit is
  written first and has to be shown failing first.
- A phase doesn't start until the one before it passes its acceptance criteria.
- Anything that changes a result or disagrees with a reference gets a dated row
  in `docs/07-decisions.md`.
- No physics in the browser client. Each phase that adds a device or a sweep
  exposes it through `ddsim/api/` the same way the existing ones are.
