# Physics

## Notation

Fixed. Code must match.

| Symbol | Code | Meaning | Units |
|---|---|---|---|
| psi | `psi` | electrostatic potential | V |
| n, p | `n`, `p` | electron, hole density | cm^-3 |
| phi_n, phi_p | `phi_n`, `phi_p` | quasi-Fermi potentials | V |
| Nd+, Na- | `Nd`, `Na` | ionized donor, acceptor | cm^-3 |
| N | `net_doping` | Nd - Na | cm^-3 |
| Jn, Jp | `Jn`, `Jp` | current densities | A/cm^2 |
| mu_n, mu_p | `mu_n`, `mu_p` | mobilities | cm^2/(V s) |
| Dn, Dp | `Dn`, `Dp` | diffusivities | cm^2/s |
| R | `R_net` | net recombination rate | cm^-3 s^-1 |
| V_T | `VT` | thermal voltage kT/q | V |
| n_i | `n_i` | intrinsic concentration | cm^-3 |
| L_D | `L_D` | Debye length | cm |
| eps | `eps` | permittivity | F/cm |

Sign convention: psi increases toward n-type. E = -grad(psi).

## The Van Roosbroeck system

Three coupled nonlinear PDEs. This is the entire model.

Poisson:

    div(eps * grad(psi)) = -q * (p - n + Nd - Na)

Electron continuity:

    dn/dt = (1/q) * div(Jn) - R + G

Hole continuity:

    dp/dt = -(1/q) * div(Jp) - R + G

Constitutive relations:

    Jn = q * mu_n * n * E + q * Dn * grad(n)
       = -q * mu_n * n * grad(psi) + q * Dn * grad(n)

    Jp = q * mu_p * p * E - q * Dp * grad(p)
       = -q * mu_p * p * grad(psi) - q * Dp * grad(p)

First term is drift, second is diffusion. Hence the name.

Steady state means dn/dt = dp/dt = 0. Phases 1 through 5 are all steady state.
Transient is out of scope. AC small-signal for C-V is done by perturbation, not
by time stepping. See Phase 4.

## Einstein relation

    Dn = mu_n * V_T,   Dp = mu_p * V_T

This holds only under Boltzmann statistics. Under degenerate doping (source/drain
at 1e20 cm^-3) the generalized form applies:

    Dn / mu_n = (V_T) * F_{1/2}(eta) / F_{-1/2}(eta)

Phase 2 uses the simple form. Phase 5 should use the generalized form in the
source/drain regions or note the error explicitly in `docs/07-decisions.md`.
Do not
silently keep Boltzmann and claim degenerate accuracy.

## Carrier statistics

Boltzmann (default through Phase 4):

    n = n_i * exp((psi - phi_n) / V_T)
    p = n_i * exp((phi_p - psi) / V_T)

Equilibrium check: phi_n = phi_p = 0 gives n*p = n_i^2. This is a unit test.

Fermi-Dirac (needed for degenerate regions, Phase 5):

    n = Nc * F_{1/2}((E_F - E_c) / kT) * (2/sqrt(pi))
    p = Nv * F_{1/2}((E_v - E_F) / kT) * (2/sqrt(pi))

F_{1/2} has no closed form. Use Joyce-Dixon inversion for the forward direction:

    (E_F - E_c)/kT ~= ln(n/Nc) + (1/sqrt(8)) * (n/Nc)
                      - (3/16 - sqrt(3)/9) * (n/Nc)^2 + ...

Valid to about n/Nc = 4. Beyond that use a rational approximation (Bednarczyk or
Halen-Pulfrey). Test any implementation against tabulated F_{1/2} values.

## Incomplete ionization

Ignore through Phase 4. Assume Nd+ = Nd, Na- = Na. At 300K and moderate doping
this is accurate to under 1 percent. If you add it later:

    Nd+ = Nd / (1 + g_d * exp((E_F - E_d)/kT)),  g_d = 2
    Na- = Na / (1 + g_a * exp((E_a - E_F)/kT)),  g_a = 4

## Recombination

### Shockley-Read-Hall

Dominant. Include from Phase 2 onward.

    R_SRH = (n*p - n_i^2) / (tau_p * (n + n1) + tau_n * (p + p1))

with n1 = p1 = n_i for a midgap trap. Sign: R_SRH > 0 means net recombination.

Doping-dependent lifetime (Scharfetter relation):

    tau = tau_min + (tau_max - tau_min) / (1 + (N_total / N_ref)^gamma)

### Auger

Matters only at high injection. Add in Phase 3 if convergence at 1 V forward bias
looks wrong on the high-current end.

    R_Auger = (Cn * n + Cp * p) * (n*p - n_i^2)

### Impact ionization

Out of scope unless you want breakdown curves. Chynoweth model, field-dependent,
makes convergence much harder because it is a positive feedback term. Do not add
it before Phase 5 is stable.

### Generation

G = 0 everywhere. No optical generation. This is not a photodiode simulator.

## Mobility models

Add in this order. Each one is a separate module with its own test.

**Phase 1-2: constant.** mu_n = 1417, mu_p = 470 cm^2/(V s). Undoped silicon
at 300K. Good enough to get a diode I-V with the right ideality factor.

**Phase 3: doping dependent (Arora or Masetti).** Arora is simpler:

    mu = mu_min + mu_d / (1 + (N/N_ref)^A)

with separate parameter sets for electrons and holes, all temperature scaled.
Masetti is the TCAD standard and adds a term for the mobility upturn at very
high doping. Use Masetti if you want to match DEVSIM closely.

**Phase 5: field dependent (Caughey-Thomas).** This is what produces velocity
saturation, which is what produces the short-channel Id ~ V_ov behavior.

    mu(E) = mu_0 / (1 + (mu_0 * E_parallel / v_sat)^beta)^(1/beta)

beta = 2 for electrons, beta = 1 for holes. E_parallel is the field component
along the current direction, which in practice means along the mesh edge. Using
the full field magnitude here is a common and wrong shortcut.

**Phase 5: surface mobility (Lombardi).** MOS channel carriers scatter off the
oxide interface. Without this your inversion-layer mobility is too high by a
factor of 2 to 3 and your Id is correspondingly wrong. Lombardi combines bulk,
acoustic phonon, and surface roughness terms by Matthiessen's rule.

Mobility is where a device simulator earns or loses its quantitative accuracy.
The PDE solve can be perfect and the answer still wrong by 3x if mobility is
wrong. Treat these models as first-class code, not as fudge factors.

## Boundary conditions

### Ohmic contacts

Charge neutrality plus thermal equilibrium at the contact node:

    p - n + Nd - Na = 0
    n * p = n_i^2

Solve the pair for n and p, then

    psi_contact = V_applied + V_T * asinh(N / (2 * n_i))

The asinh form is numerically stable. The naive `V_T * ln(N/n_i)` form breaks in
lightly doped or compensated regions where N can be near zero or negative. Use
asinh. This is a frequent bug source.

For the quasi-Fermi levels at an ohmic contact: phi_n = phi_p = V_applied.

### Schottky contacts

Out of scope.

### MOS gate

Not a semiconductor node. Insulator Poisson only, no continuity equations in the
oxide. Dirichlet on psi at the gate metal with the work function difference
folded in:

    psi_gate = V_gate - Phi_MS

At the Si/SiO2 interface, enforce continuity of the normal component of D
(displacement), not of E. Fixed interface charge Q_f if you want to model it.

### Reflecting / symmetry boundaries

Homogeneous Neumann on all three unknowns. Zero normal current, zero normal field.
Every boundary that is not a contact is reflecting.

## What emerges, and must not be hardcoded

These are outputs, never inputs. If any of them appears as a fitted parameter
anywhere in the code, that is a bug in the project, not a shortcut.

- Built-in potential of a junction
- Depletion width and its bias dependence
- Diode ideality factor, and the crossover from n=2 to n=1
- Threshold voltage and body effect
- Subthreshold slope, floored at 60 mV/decade at 300K
- DIBL and Vth roll-off with gate length
- Velocity saturation and the resulting linear-in-V_ov drain current

## Where drift-diffusion breaks down

Know these. They are the honest limits of the model and being able to state them
is most of the value of having written it.

- **Below about 50 nm channel length**, carriers become quasi-ballistic. Local
  field no longer determines local velocity. You need energy-balance
  (hydrodynamic) or Monte Carlo Boltzmann. This is why the sweep stops at 50 nm.
- **Velocity overshoot** is invisible to drift-diffusion by construction, because
  the model assumes instantaneous relaxation to the local field.
- **Quantum confinement** in the inversion layer shifts the centroid of charge
  away from the interface and increases effective oxide thickness. Needs a
  Schrodinger-Poisson correction. This is exactly where AtomSIM's solver could be
  reused if you want to go further.
- **Tunneling** (gate leakage, band-to-band) is absent. No barrier penetration in
  a classical transport model.

## Derivation chain, for the writeup

Drift-diffusion is not fundamental. It is the first two moments of the Boltzmann
transport equation under the relaxation-time approximation.

    Schrodinger in periodic potential
      -> Bloch states, band structure E(k)
      -> effective mass approximation near band edge, m* from band curvature
      -> semiclassical carriers, Boltzmann transport equation for f(r,k,t)
      -> zeroth moment: continuity equation
      -> first moment + relaxation time: drift-diffusion current
      -> close with Poisson: Van Roosbroeck system

The effective mass approximation is the step that licenses discarding quantum
mechanics for 1e17 carriers. Nc and Nv, which set n_i, come directly from m*:

    Nc = 2 * (2*pi * m_e* * k * T / h^2)^(3/2) * M_c
    Nv = 2 * (2*pi * m_h* * k * T / h^2)^(3/2)
    n_i = sqrt(Nc * Nv) * exp(-Eg / (2*k*T))

M_c = 6 for silicon (six equivalent conduction band minima).

This is the literal numerical handoff from the band layer. Keep `constants.py`
structured so those values can be swapped for computed ones without touching
anything else.
