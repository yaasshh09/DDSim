# Constants

Single source of truth. `core/constants.py` mirrors this file exactly. If a
number appears anywhere else in the codebase, that is a bug.

Convention: lengths in cm, concentrations in cm^-3, matching the semiconductor
literature. Do not switch to SI lengths.

## Fundamental

| Name | Value | Units |
|---|---|---|
| q | 1.602176634e-19 | C |
| k_B | 1.380649e-23 | J/K |
| eps_0 | 8.8541878128e-14 | F/cm |
| h | 6.62607015e-34 | J s |
| m_0 | 9.1093837015e-31 | kg |

## Temperature dependent, at T = 300 K

| Name | Value | Units | Note |
|---|---|---|---|
| V_T | 0.0258520 | V | kT/q |
| kT | 0.0258520 | eV | |
| SS_min | 0.059526 | V/decade | V_T * ln(10), thermodynamic floor |

Make T a parameter, not a constant. Every temperature-dependent value should be
a function of T. Room temperature results are the default, not the only case.

## Silicon

| Name | Value | Units | Note |
|---|---|---|---|
| eps_Si | 11.7 * eps_0 | F/cm | |
| Eg(300K) | 1.1242 | eV | Varshni: Eg(T) = 1.1696 - 4.73e-4 T^2/(T+636) |
| n_i(300K) | **1.0e10** | cm^-3 | see note below |
| Nc(300K) | 2.86e19 | cm^-3 | |
| Nv(300K) | 3.10e19 | cm^-3 | |
| chi (affinity) | 4.05 | eV | |
| M_c | 6 | | conduction band valleys |

### The n_i problem, read this

Silicon n_i at 300 K is quoted as 9.65e9 (Sproul and Green, the modern accepted
measurement), 1.0e10 (rounded, most common in teaching), and 1.45e10 (older
literature, still embedded in many tools and textbooks).

A factor of 1.5 in n_i is a factor of 2.25 in n_i^2, which propagates directly
into saturation current and shifts built-in potential by about 10 mV.

**Decision: use 1.0e10.** Rationale: it matches most textbook worked examples,
which are the analytic targets in Tier 2 validation.

**Non-negotiable:** set n_i explicitly in DEVSIM when generating golden data. Do
not accept its default. A silent n_i mismatch will show up as a clean-looking 2x
discrepancy in diode current that costs a day to track down.

Record any change to this value in `docs/07-decisions.md`, and regenerate all
golden data.

### n_i is not consistent with Nc, Nv and Eg, and that is deliberate

Nothing above says so, so say it here. The three values in the table do not
satisfy the relation that connects them:

    sqrt(Nc * Nv) * exp(-Eg / (2 * V_T)) = 1.0757e10 cm^-3

against the 1.0e10 we use. That is 7.6 percent in n_i and 16 percent in n_i^2.
Nc and Nv are measured 300 K values and n_i is anchored by the decision above,
so they cannot both be primary and no amount of rearranging makes them agree.

How the code resolves it: n_i(300) is pinned to 1.0e10 and the physics supplies
only the temperature dependence,

    n_i(T) = n_i(300) * (T/300)^(3/2)
                      * exp(Eg(300)/(2 V_T(300)) - Eg(T)/(2 V_T(T)))

where the power law carries the Nc*Nv scaling and the exponential carries the
gap. The ratio n_i^2 / (Nc Nv exp(-Eg/V_T)) is then exactly temperature
independent, which is tested. Eg comes from the Varshni formula rather than the rounded 1.1242 in the
table; the two differ by 8.1e-5 eV and the formula is the definition.

This is fine as long as nothing computes an absolute band edge position or a
Fermi level from Nc, because that is where a 7.6 percent inconsistency stops
being bookkeeping and starts being a wrong answer. Phase 5 degenerate statistics
is where it has to be settled.

## Silicon dioxide

| Name | Value | Units |
|---|---|---|
| eps_ox | 3.9 * eps_0 | F/cm |
| Eg_ox | 9.0 | eV |
| barrier to Si CB | 3.1 | eV |

## Work functions, for the MOS gate

The gate is a Dirichlet condition on psi with the work function difference
folded in, `psi_gate = V_gate - Phi_MS`. A wrong Phi_MS slides the whole C-V
curve along the voltage axis without changing its shape, so all three regimes
still look right. That is why Phase 4 gates flatband at 20 mV rather than
trusting the curve.

| Name | Value | Units | Note |
|---|---|---|---|
| chi_Si | 4.05 | eV | electron affinity, Si conduction edge below vacuum |
| Phi_M, n+ poly | 4.05 | eV | Fermi level at the conduction edge |
| Phi_M, midgap | 4.6121 | eV | chi + Eg/2, the usual tungsten model |
| Phi_M, p+ poly | 5.1741 | eV | chi + Eg, Fermi level at the valence edge |

The semiconductor side is

    Phi_S = chi + Eg/2 - phi_F,     phi_F = V_T * asinh(N / (2*n_i))

**asinh, not `V_T * ln(N/n_i)`.** Same reason as the contact potential in
docs/05-pitfalls.md: the log form is -inf at zero doping and nan for the other
sign, and both occur in a real substrate. asinh is smooth through zero and
antisymmetric, so intrinsic silicon lands exactly at midgap and equal n and p
doping give exactly opposite offsets. The two agree to twelve digits wherever
the log form is valid.

Computed values, for checking against a textbook worked example:

| Gate | Substrate | Phi_MS |
|---|---|---|
| n+ poly | p-type 1e15 | -0.8597 V |
| n+ poly | p-type 1e16 | -0.9192 V |
| n+ poly | p-type 1e17 | -0.9787 V |
| p+ poly | n-type 1e16 | +0.9192 V |

The -0.92 V at 1e16 is the standard NMOS number. The sign is the easy thing to
get wrong and it moves flatband by nearly two volts.

The two polysilicon values are idealisations: real degenerate poly sits a few
tens of meV inside the gap rather than exactly on the band edge, and heavy
doping narrows the gap as well. Both are far below the 20 mV gate, so this is
recorded rather than modelled.

## Mobility, undoped silicon at 300 K

| Name | Value | Units |
|---|---|---|
| mu_n | 1417 | cm^2/(V s) |
| mu_p | 470 | cm^2/(V s) |
| v_sat,n | 1.07e7 | cm/s |
| v_sat,p | 8.3e6 | cm/s |

### Arora model parameters

    mu = mu_min + mu_d / (1 + (N / N_ref)^A)

| Param | Electrons | Holes |
|---|---|---|
| mu_min | 88 * (T/300)^-0.57 | 54.3 * (T/300)^-0.57 |
| mu_d | 1252 * (T/300)^-2.33 | 407 * (T/300)^-2.23 |
| N_ref | 1.432e17 * (T/300)^2.546 | 2.67e17 * (T/300)^2.546 |
| A | 0.88 * (T/300)^-0.146 | 0.88 * (T/300)^-0.146 |

Masetti is the TCAD standard and is preferable if matching DEVSIM tightly.
Its parameters are in the DEVSIM documentation.

### Caughey-Thomas

    mu(E) = mu_0 / (1 + (mu_0 * E_par / v_sat)^beta)^(1/beta)

beta = 2 for electrons, beta = 1 for holes.

## SRH lifetimes

Defaults, Scharfetter doping dependence:

| Param | Electrons | Holes |
|---|---|---|
| tau_max | 1e-5 s | 3e-6 s |
| tau_min | 0 | 0 |
| N_ref | 5e16 cm^-3 | 5e16 cm^-3 |
| gamma | 1 | 1 |

## Auger coefficients

| Name | Value | Units |
|---|---|---|
| C_n | 2.8e-31 | cm^6/s |
| C_p | 9.9e-32 | cm^6/s |

## Derived, useful for sanity checks

At 300 K in silicon:

Extrinsic Debye length, `sqrt(eps_Si * V_T / (q * N))`:

| Doping (cm^-3) | Debye length | Note |
|---|---|---|
| 1e15 | 129 nm | |
| 1e16 | 40.9 nm | |
| 1e17 | 12.9 nm | |
| 1e18 | 4.09 nm | mesh must resolve this |
| 1e20 | 0.409 nm | continuum model straining |

Intrinsic Debye length at n_i = 1e10 is **40.9 um**, from the same formula with
N = n_i. An earlier revision of this file said 24 um. That number is
`sqrt(eps_Si * V_T / (2 * q * 1.45e10))`, so it carried both a stray factor of
2 and the superseded n_i = 1.45e10.

Built-in potential, 1e16 / 1e16 abrupt junction: **0.7143 V**, from
`V_T * ln(Na*Nd/n_i^2)`. An earlier revision said 0.695 V, which is the same
formula evaluated at n_i = 1.45e10. This one matters: it is an acceptance target
in phases/PHASE-1.md, and a 19 mV offset in V_bi reads exactly like a boundary
condition sign error.

## Provenance

Nc and Nv are currently hardcoded. They derive from effective masses:

    Nc = 2 * (2*pi * m_e* * k * T / h^2)^(3/2) * M_c
    Nv = 2 * (2*pi * m_h* * k * T / h^2)^(3/2)

If the band structure layer is ever built, these become computed values.
Structure `constants.py` so that swap requires touching one function, not
grepping the codebase.
