# Phase 8: The scoreboard

**Target: 1 to 2 weeks. Docs: 04-validation (Tier 4 and Tier 5), 07-decisions, phases/ROADMAP.md.**

Agreement with DEVSIM is already proven. This phase measures the difference.
Every phase after this one has to move a number built here, so the numbers
have to be fair before they're flattering.

No new physics, no change to any solved number. If a result in `data/golden/`
moves during this phase, something is wrong.

## Scope

1. **The case set.** 200 robustness cases drawn with a fixed seed (20260924)
   from the knob ranges the API already declares in docstrings. Draw log
   uniform where the slider is logarithmic and uniform otherwise, reading the
   ranges through `ddsim/api/devices.py`, not a second copy of them. Each case
   is one device plus one target bias, solved cold from equilibrium. The case
   list is written to `data/scoreboard/cases.csv` once and never regenerated
   silently.
   - Only physical knobs are drawn: doping, geometry, oxide, work function
     and bias. Mesh knobs stay at their defaults, and each tool meshes the
     device its own way at benchmark quality. Drawing DDSim's mesh knobs would
     test DDSim's mesher, and DEVSIM has no equivalent knob to set.
   - The split is 60 pn diodes, 40 MOS capacitors and 100 NMOS, with the
     MOSFET on the full Phase 5 model stack, because the MOSFET is where
     robustness is hard. The diode and MOS capacitor use the benchmark 1 to 5
     models.
   - A draw that DDSim's device check refuses is drawn again, and the number
     of redraws is recorded. The drawn and stack devices aren't included,
     since DEVSIM can't be scripted from a drawing without writing a second
     geometry engine.
2. **Three drivers per case.**
   - DDSim, through the same public solve the CLI and the API call.
   - DEVSIM stock: its own `python_packages/ramp.py`, as shipped.
   - DEVSIM expert: the settle and ramp logic in
     `tests/regression/devsim_gen/generate_mosfet.py`, which is the best
     driver I know how to write for it.

   DDSim only wins robustness if it beats the expert driver. Beating the stock
   one alone is worth reporting, but it isn't the claim.
3. **What counts as converged.** A converged flag alone doesn't count; DEVSIM
   reported success on a transfer curve reading -2.2 A/cm (see
   `tests/regression/devsim_gen/README.md`). A case passes when:
   - the tool says it converged,
   - its terminal currents sum to zero to 1e-6 of the largest one, and
   - its answer agrees with a fine-step reference within that device family's
     Tier 4 tolerance.

   The reference is each tool solved with a tiny ramp step. If the two fine
   references disagree beyond tolerance, the case is dropped from the count
   and listed, not scored for either side.
4. **Accuracy against node count.** For benchmarks 1 to 10, solve each tool at
   the reference mesh, at half the spacing and at a quarter of it. Take the
   Richardson extrapolation per tool as that tool's converged value, and
   record error against node count. The headline number per benchmark: nodes
   needed to reach 1 percent error on the quantity the benchmark asserts.
5. **Speed.** Median wall time per converged bias point over five runs of
   benchmark sweeps 1 to 8, with `MKL_NUM_THREADS`, `OMP_NUM_THREADS` and
   `OPENBLAS_NUM_THREADS` all set to 1 for both tools. The CSV header records
   the CPU, the OS and both versions.
6. **The capability matrix**, `data/scoreboard/capabilities.csv`: one row per
   capability, a yes or no per tool, and evidence for every yes. For DDSim the
   evidence is a test id. For DEVSIM it's the API name, checked against
   `dir(devsim)` by the generator.
7. **Error control.** The share of results each tool reports with an error
   estimate. Both are at zero today. Phase 12 moves it.
8. Generators live in `tests/regression/devsim_gen/scoreboard.py` for DEVSIM
   and `tools/scoreboard.py` for DDSim. Both run by hand, like the golden
   data, and write CSVs with a provenance header into `data/scoreboard/`.
9. `tests/regression/test_scoreboard.py` reads the CSVs and:
   - checks their schema and provenance headers
   - fails if a DDSim number on the committed board gets worse than the last
     one committed (robustness count down, nodes-to-1-percent up, or a
     capability going from yes to no)
   - fails if any test id cited in the capability matrix doesn't exist
   - fails if the scoreboard table in the README disagrees with the CSVs

## Acceptance criteria

- All 200 cases run on all three drivers, with results committed.
- Accuracy curves for benchmarks 1 to 10 in both tools.
- The speed table, measured, whatever it says.
- The capability matrix, with every yes backed by evidence the test checks.
- Every check in item 9 has been shown to fail: break the thing it guards
  and make sure it goes red.
- A dated row in `docs/07-decisions.md` fixing the fairness choices: thread
  count, the pass rule, the drivers, the seed and the tolerance per family.

## Deliverable

A scoreboard table in the README, one row per axis, DDSim against DEVSIM,
with a link to the CSVs. Where DEVSIM wins, the table says so.

## Things not to do

- Don't write down which tool wins before it's measured. Speed in particular
  may well be a loss, and the table has to be able to show that.
- Don't tune a DDSim default during this phase to win a case. Tuning is what
  the later phases are for, and doing it here makes the baseline meaningless.
