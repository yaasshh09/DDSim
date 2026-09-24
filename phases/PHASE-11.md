# Phase 11: Transient

**Target: 2 to 3 weeks. Docs: 02-numerics (Time integration), 01-physics (The Van Roosbroeck system), 07-decisions.**

This reverses the `init` decision "Steady state only" in `docs/07-decisions.md`.
That needs a new dated row, not an edit to the old one, and the sentence
"Transients are out of scope" in `docs/01-physics.md` changes with it.

## Scope

1. **Backward Euler first.** One Newton solve per time step on
   `F(x_{k+1}) + M (x_{k+1} - x_k) / dt = 0`, using the mass matrix from
   Phase 10. The Jacobian is `J + M/dt`, which is better conditioned than
   `J`, so the DC machinery carries over unchanged.
2. **Then BDF2 with variable steps.** The local truncation error is
   estimated from the gap between the predictor and the corrector, the step
   is accepted or rejected against a tolerance, and the next step is
   resized. TR-BDF2 is a stretch goal.
3. **Terminal current includes displacement current.** At a contact,
   I = conduction + d/dt of the contact charge. The charge comes from the
   same place `extract/cv.py` already reads it. Without the displacement
   term, the currents don't sum to zero during a transient, and that failure
   is the test.
4. **A series resistor and a voltage or current source on any contact.** This
   is the smallest piece of circuit coupling that reverse recovery needs, and
   it also gives Phases 13 and 14 their current-driven contacts. Full nodal
   analysis stays in SPICE (layer 3), so the seam holds.
5. `extract/transient.py`: time sweeps, waveforms, and extraction of storage
   time and switching time.
6. The API streams one frame per accepted time step. This fits the WebSocket
   telemetry already there, and the client can animate the carrier profile
   over time. The client still computes nothing.

## Analytic limits, write these tests first

- **Rest stays at rest.** Starting from equilibrium at fixed bias, the state
  doesn't move over 1000 steps, to roundoff.
- **Long time is DC.** After a bias step, the state for t much greater than
  every time constant equals the DC solution at the new bias, to solver
  tolerance.
- **Order.** On a smooth problem, halving dt cuts the error against a fine
  step reference by 2 for backward Euler and 4 for BDF2. It's measured on a
  log-log fit, like the mesh order studies.
- **Transient agrees with AC.** Drive a 1 mV sinusoid, much smaller than V_T,
  run to periodic steady state, and take the fundamental. Its amplitude and
  phase must match Y(omega) from Phase 10 within 1 percent at three
  frequencies.
- **Charge conservation.** The terminal currents, displacement current
  included, sum to zero at every step to 1e-10 of the largest current.
- **Diode reverse recovery.** A long p+n diode switched from forward current
  I_F to reverse current I_R through the series resistor. The storage time
  t_s should satisfy Kingston's result, erf(sqrt(t_s/tau_p)) = I_F/(I_F + I_R),
  within 5 percent. Likely (unverified): this form is from Kingston, Proc.
  IRE 1954. It gets checked against the paper or Sze before it's asserted.
  The charge-control estimate t_s = tau_p ln(1 + I_F/I_R) is a sanity bound,
  not the target.

## Acceptance criteria

- Everything above passes, shown failing first.
- DEVSIM `solve(type="transient_bdf2")` golden for the reverse recovery
  waveform: benchmark 14, within 5 percent on storage time and on the
  current at every sample.
- The scoreboard's feature rows for transient and circuit elements move to
  yes, citing the tests.
- Speed recorded: time steps per second on benchmark 14 in both tools.

## Honest limits

The contact circuit is a resistor and a source, not a netlist. MOSFET
switching into a real load belongs to SPICE, using the models Phase 6 will
extract.
