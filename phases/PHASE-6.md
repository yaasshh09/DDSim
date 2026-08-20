# Phase 6: Bridge to SPICE

**Target: 1 to 2 weeks. Depends on the SPICE project existing.**

The reason both projects were worth doing. This phase is what turns two repos
into one thesis.

## Scope

1. `extract/compact.py`: sweep the MOSFET across a (Vg, Vd, Vb, W, L) grid,
   generate I-V and C-V tables
2. Fit EKV or BSIM3 compact model parameters to those tables. EKV is far simpler
   and adequate. Use least squares with sensible parameter bounds.
3. Emit a `.model` card in SPICE netlist syntax
4. Load into your SPICE, simulate a circuit
5. Report the compact model fitting error against the TCAD source data

## Acceptance criteria

- Compact model reproduces DDSim's Id-Vg to under 5 percent across the fitted
  range, in both linear and saturation
- Your SPICE simulates a CMOS inverter using the extracted model and produces a
  correct DC transfer curve with the expected switching threshold
- A ring oscillator built from those models oscillates at a frequency consistent
  with the extracted gate delay
- Emitted `.model` card also loads in ngspice and gives comparable results.
  Independent confirmation that the card is valid rather than only valid in your
  own tool.

## Stretch: the full chain

Simulate your LM358 EMG front end in your own SPICE, using transistor models
extracted from your own device simulator.

If the band structure module exists, parameterize Nc and Nv from computed
effective masses so the chain is real code end to end:

    quantum solver -> m* -> Nc, Nv -> n_i -> DDSim -> compact model -> SPICE -> circuit

## The README paragraph

This is the thing worth writing carefully. One paragraph, no adjectives, stating
plainly: the atom solver produced effective masses, the device solver consumed
them and produced transistor characteristics, the extraction fitted a compact
model, and the circuit simulator used it to simulate a real analog front end.
Every layer validated against an independent reference tool.

Let the diagram and the validation plots carry the weight. Do not editorialize.
