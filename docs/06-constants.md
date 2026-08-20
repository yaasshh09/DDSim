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

Record any change to this value in `PROGRESS.md`, and regenerate all golden data.

## Silicon dioxide

| Name | Value | Units |
|---|---|---|
| eps_ox | 3.9 * eps_0 | F/cm |
| Eg_ox | 9.0 | eV |
| barrier to Si CB | 3.1 | eV |

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

| Doping (cm^-3) | Debye length | Note |
|---|---|---|
| 1e15 | 128 nm | |
| 1e16 | 40 nm | |
| 1e17 | 13 nm | |
| 1e18 | 4.1 nm | mesh must resolve this |
| 1e20 | 0.4 nm | continuum model straining |

Intrinsic Debye length at n_i = 1e10 is roughly 24 um.

Built-in potential, 1e16 / 1e16 abrupt junction: about 0.695 V.

## Provenance

Nc and Nv are currently hardcoded. They derive from effective masses:

    Nc = 2 * (2*pi * m_e* * k * T / h^2)^(3/2) * M_c
    Nv = 2 * (2*pi * m_h* * k * T / h^2)^(3/2)

If the band structure layer is ever built, these become computed values.
Structure `constants.py` so that swap requires touching one function, not
grepping the codebase.
