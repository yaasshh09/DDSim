# Phase 18: Other materials and heterojunctions

**Target: 3 weeks. Docs: 01-physics (Boundary conditions), 06-constants, 03-architecture (The Device object).**

Everything so far is silicon. DEVSIM has region interfaces, so it can join
two materials. This phase adds a material database and the interface
conditions that go with it. It's also where the stack thesis gets tested
hardest, because every new material is a list of numbers that ideally comes
from a layer below.

## Scope

1. **A material table** in `docs/06-constants.md` and `core/constants.py`,
   one source per number: germanium, GaAs, and Si(1-x)Ge(x) as a function of
   x. It holds bandgap, electron affinity, Nc, Nv, permittivity and the
   mobility model parameters. Silicon's numbers don't move, and the golden
   data proves it.
2. **Heterointerfaces.** psi and the normal D are continuous. The band edges
   step by the affinity difference (Anderson's rule), and each side has its
   own n_i. Carriers cross an abrupt offset by thermionic emission, reusing
   the Phase 14 boundary form. That choice gets a decision row, since
   continuous quasi-Fermi levels is the other common choice.
3. **Graded composition.** For SiGe with x varying in space, the band edges
   vary smoothly. Scharfetter-Gummel then needs the effective potential form
   for each carrier: its band edge gradient acts as an extra field. The
   discretization of that goes into `docs/02-numerics.md` before any code.
4. **A SiGe heterojunction bipolar transistor**, reusing the Phase 13 device
   with a SiGe base.
5. The drawing editor gets a material per region, chosen from the table.

## Analytic limits, write these tests first

- **Silicon is untouched.** Every golden number stays bit for bit. This is a
  new code path, and silicon mustn't go through anything new.
- **Band offset at equilibrium.** On an abrupt n-N heterojunction, the
  conduction band step at the interface equals chi_1 - chi_2 exactly, and
  the Fermi level is flat to solver tolerance.
- **Split of the built-in potential.** On an abrupt anisotype junction in the
  depletion approximation, the ratio of potential drops on the two sides is
  eps_2 N_2 / (eps_1 N_1). Within 3 percent.
- **Homojunction limit.** A heterojunction between a material and itself,
  with zero offsets, reproduces the silicon diode to 1e-10.
- **HBT gain rises with the base bandgap drop.** The ratio of beta with and
  without a uniform Ge fraction in the base follows exp(Delta_Eg / kT) times
  the ratios of Nc Nv and of D_n in the base, within 10 percent in low
  injection. The exponent emerges from the band edges, not from a gain
  parameter.

## Acceptance criteria

- All of the above, shown failing first.
- Benchmark 21: an abrupt n-GaAs / N-AlGaAs or Si / SiGe heterojunction I-V
  (whichever DEVSIM's interface examples support most directly) against
  DEVSIM with the same interface condition, within 5 percent.
- Scoreboard rows for materials and heterojunctions move to yes.

## The stack

Every material parameter added here is written down with its source. When
the band module (layer 1) exists, Nc and Nv for each material should come
from its computed effective masses, and the test that replaces the table
entry with the computed one is the one that matters.
