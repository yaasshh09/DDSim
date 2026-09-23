# Phase 6: Bridge to SPICE

**Target: 1 to 2 weeks. Depends on the SPICE project existing.**

The reason both projects were worth doing. This phase turns two repos into
one thesis.

## Scope

1. `extract/compact.py`: sweep the MOSFET across a (Vg, Vd, Vb, W, L) grid
   and generate I-V and C-V tables
2. Fit EKV or BSIM3 compact model parameters to those tables. EKV is far
   simpler and good enough. Use least squares with sensible parameter bounds.
3. Write out a `.model` card in SPICE netlist syntax
4. Load it into my SPICE and simulate a circuit
5. Report the compact model's fitting error against the TCAD data it came
   from

## Acceptance criteria

- The compact model reproduces DDSim's Id-Vg to under 5 percent across the
  fitted range, in both linear and saturation
- My SPICE simulates a CMOS inverter with the extracted model and gets a
  correct DC transfer curve with the expected switching threshold
- A ring oscillator built from those models oscillates at a frequency that
  fits the extracted gate delay
- The `.model` card also loads in ngspice and gives comparable results. That's
  independent confirmation the card is actually valid, not just valid in my
  own tool.

## Stretch: the full chain

Simulate my LM358 EMG front end in my own SPICE, using transistor models
extracted from my own device simulator.

If the band structure module exists, parameterize Nc and Nv from computed
effective masses so the chain is real code from end to end:

    quantum solver -> m* -> Nc, Nv -> n_i -> DDSim -> compact model -> SPICE -> circuit

## The README paragraph

This is the thing worth writing carefully. One paragraph, no adjectives,
stating plainly that the atom solver produced effective masses, the device
solver used them to produce transistor characteristics, the extraction fitted
a compact model, and the circuit simulator used that to simulate a real
analog front end, with every layer validated against an independent reference
tool.

Let the diagram and the validation plots carry the weight. Don't
editorialize.
