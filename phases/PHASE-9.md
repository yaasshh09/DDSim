# Phase 9: Temperature, fixed oxide charge, incomplete ionization

**Target: 1 to 2 weeks. Docs: 01-physics (Carrier statistics, Incomplete ionization, Temperature), 06-constants, 07-decisions.**

Three small physics additions, each with an exact analytic target. They come
first after the scoreboard because none of them makes the Jacobian harder to
solve at room temperature. Incomplete ionization is the one exception at low
temperature, and it's off by default.

## Scope

1. **Temperature as a knob.** `T` [K] on every device constructor and on
   `Device`, carried into `Scaling`. The models already take `T`: n_i through
   Varshni, Nc and Nv as T^1.5, Arora and Lombardi. What's missing is the
   plumbing. There are 8 places outside `constants.py` and `mobility.py`
   that default to `C.T_ROOM`, and each one has to take the device's `T`
   instead of falling back silently.
   - Caughey-Thomas saturation velocity gets its temperature dependence. The
     formula and its source go into `docs/06-constants.md` before any code
     uses them.
   - SRH lifetimes stay constant in `T`. That's a decision, so it gets a
     row in `docs/07-decisions.md`.
   - The knob range is measured end to end the same way the 2D ranges were
     on 2026-09-23. My guess is 200 to 500 K, but a guess isn't a range.
2. **Fixed oxide charge Q_f** [cm^-2, as a sheet density of elementary
   charges] on the MOS capacitor and the MOSFET. It's a sheet charge at the
   Si/SiO2 interface, so it enters Poisson as a source in the dual cells of
   the interface nodes and nowhere else.
3. **Incomplete ionization**, as a model flag that's off by default, using the
   two formulas already in `docs/01-physics.md`. Donor and acceptor ionization
   energies go into `docs/06-constants.md` from a cited source, one per
   dopant species the builders use.
4. The API gains the three knobs, with ranges.
5. New scoreboard rows: temperature, Q_f, incomplete ionization. DEVSIM can
   do all three if you write the equations yourself, so they count as parity,
   not as a lead, unless they're built in and tested in a way DEVSIM's aren't.

## Analytic limits, write these tests first

- **Nothing moves at 300 K.** With T = 300, Q_f = 0 and incomplete
  ionization off, every golden number is unchanged bit for bit. Checked the
  same way the edge list refactor was.
- **T reaches every model.** Build a device at 400 K and assert that every
  model object it holds reports 400. This test exists to catch a default
  argument that silently stayed at 300.
- **Built-in potential**: V_bi(T) = V_T ln(Na Nd / n_i(T)^2) at equilibrium, to
  solver tolerance, at 250, 300, 350 and 400 K.
- **Long-base diode saturation current**: the ratio J_0(T1)/J_0(T2) against
  n_i(T)^2 D(T)/L(T), computed from the code's own n_i, mobility and lifetime.
  2 percent, low injection only.
- **Thermal limit on subthreshold swing**: at every T, the long-channel
  MOSFET's minimum swing sits at or above ln(10) kT/q, and within 5 percent
  of ln(10) (kT/q)(1 + C_d/C_ox) with C_d taken from the solved depletion
  width. The 59.5 mV/dec gate from Phase 5 is the 300 K case of this.
- **Flatband shift**: the C-V curve with Q_f is the curve without it shifted
  by exactly -q Q_f / C_ox, to 1e-4 relative at matched capacitance. The
  MOSFET threshold shifts by the same amount to 1 mV.
- **Freeze-out in a uniform bar**: with Boltzmann statistics and a donor
  level Delta_E_d below Ec, the neutral electron density solves
  n(1 + 2n/N1) = Nd with N1 = Nc exp(-Delta_E_d/kT). The closed form is
  n = (N1/4)(sqrt(1 + 8 Nd/N1) - 1). Match it to 1e-10 relative at 100, 200
  and 300 K.

## Acceptance criteria

- All of the above pass, and each was shown failing first.
- Benchmark 1 regenerated in DEVSIM at 400 K, with DDSim within the same
  2 percent on log I.
- A known-deviation row for incomplete ionization above about 1e18 cm^-3.
  The simple model keeps predicting freeze-out there, while real silicon goes
  metallic (the Mott transition). It's an approximation I'm stating, not
  fixing.

## Honest limits

No bandgap narrowing, and lattice temperature is uniform: there's no self
heating. Both are listed in `docs/01-physics.md` as absent.
