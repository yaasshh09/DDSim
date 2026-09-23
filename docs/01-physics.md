# Physics

## Notation

This is fixed, and the code has to match it.

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

Sign convention: psi increases toward n-type, and E = -grad(psi).

## The Van Roosbroeck system

Three coupled nonlinear PDEs. That's the whole model.

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

The first term is drift and the second is diffusion, which is where the name
comes from.

Steady state means dn/dt = dp/dt = 0, and Phases 1 through 5 are all steady
state. Transients are out of scope. The AC small signal needed for C-V comes
from perturbation, not time stepping. See Phase 4.

## Einstein relation

    Dn = mu_n * V_T,   Dp = mu_p * V_T

This only holds under Boltzmann statistics. Under degenerate doping (a source
or drain at 1e20 cm^-3) the generalized form applies:

    Dn / mu_n = (V_T) * F_{1/2}(eta) / F_{-1/2}(eta)

That's written in the Gamma-normalised convention, where F_s is divided by
Gamma(s+1) so that F_s -> exp(eta) in the nondegenerate limit. In the plain
convention the tables use, which is also what `fermi_dirac_half` returns, the
same relation reads `Dn / mu_n = 2 * V_T * F_{1/2} / F_{-1/2}`. Both give
exactly V_T as eta goes to minus infinity, and that's the check that tells
them apart.

Phase 2 uses the simple form. Phase 5 should use the generalized form in the
source and drain, or record the error explicitly in `docs/07-decisions.md`.
Don't quietly keep Boltzmann and claim degenerate accuracy.

## Carrier statistics

Boltzmann (the default through Phase 4):

    n = n_i * exp((psi - phi_n) / V_T)
    p = n_i * exp((phi_p - psi) / V_T)

Equilibrium check: phi_n = phi_p = 0 gives n*p = n_i^2. That's a unit test.

Fermi-Dirac (needed in degenerate regions, Phase 5):

    n = Nc * F_{1/2}((E_F - E_c) / kT) * (2/sqrt(pi))
    p = Nv * F_{1/2}((E_v - E_F) / kT) * (2/sqrt(pi))

F_{1/2} has no closed form. Use the Joyce-Dixon inversion for the forward
direction:

    (E_F - E_c)/kT ~= ln(u) + A1*u + A2*u^2 + A3*u^3 + A4*u^4,     u = n/Nc

    A1 = 1/sqrt(8)          = +3.53553e-1
    A2 = 3/16 - sqrt(3)/9   = -4.95009e-3
    A3                      = +1.48386e-4
    A4                      = -4.42563e-6

**A2 is negative and enters with a plus sign.** This line used to read
`- (3/16 - sqrt(3)/9) * (n/Nc)^2`, which flips it. I checked against a Brent
inversion of the integral: written as above, the series lands 1.0e-4 from the
true eta at u = 4. With the sign flipped it lands 1.6e-1 away. Corrected
2026-09-01, see docs/07-decisions.md.

It's good to about n/Nc = 4. Past that, use a rational approximation
(Bednarczyk or Halen-Pulfrey), and test any implementation against tabulated
F_{1/2} values. `ddsim/physics/statistics.py` refuses anything above
n/Nc = 8, which is where the measured error in n and in the Einstein ratio is
still inside 1 percent.

## Doping range

The 1D devices are solved with Boltzmann statistics, the Arora mobility and
the Scharfetter lifetime, all functions of |net doping|. I use them over
**1e14 to 1e19 cm^-3**, and the 1D stack builder refuses a region outside
that.

The top end is where Boltzmann stops being close. I measured it on 2026-09-18
against Fermi-Dirac with Nc = 2.86e19: at 1e19 the Fermi level is 3.2 mV off
and the Einstein ratio D/(mu V_T) reads 1.122 instead of 1. At 3e19 it's
9.5 mV and 1.361, and it only gets worse from there. A degenerate region needs
the Fermi-Dirac path the MOSFET uses, not a wider range here. Complete
ionization, assumed below, is also at its weakest at the top end.

The bottom end isn't a model breaking. Mobility and lifetime have both
flattened out there: at 1e14 the Arora electron mobility is 0.16 percent
below its zero doping value and the Scharfetter lifetime is 0.2 percent below
tau_max. At 1e14 the Debye length is already 0.41 um, so a region a few
tenths of a micron long has no quasi-neutral part at all, and the bottom end
matches the range the pn diode's own knobs declare. An intrinsic layer, the i
of a pin, gets drawn as a 1e14 region.

A 2D drawing takes the same range. With Fermi-Dirac statistics on, it goes up
to **1e20 cm^-3**, the peak of the nmos source and drain, where n/Nc is 3.5
and the Joyce-Dixon inversion above still holds.

## Incomplete ionization

Ignored through Phase 4: Nd+ = Nd and Na- = Na. At 300 K and moderate doping
that's accurate to under 1 percent. If it ever gets added:

    Nd+ = Nd / (1 + g_d * exp((E_F - E_d)/kT)),  g_d = 2
    Na- = Na / (1 + g_a * exp((E_a - E_F)/kT)),  g_a = 4

## Recombination

### Shockley-Read-Hall

The dominant one, included from Phase 2 on.

    R_SRH = (n*p - n_i^2) / (tau_p * (n + n1) + tau_n * (p + p1))

with n1 = p1 = n_i for a midgap trap. Sign: R_SRH > 0 means net
recombination.

Doping dependent lifetime (the Scharfetter relation):

    tau = tau_min + (tau_max - tau_min) / (1 + (N_total / N_ref)^gamma)

### Auger

Only matters at high injection. Add it in Phase 3 if convergence at 1 V
forward bias looks off at the high current end.

    R_Auger = (Cn * n + Cp * p) * (n*p - n_i^2)

### Impact ionization

Out of scope unless you want breakdown curves. The Chynoweth model is field
dependent and makes convergence much harder, because it's a positive feedback
term. Don't add it before Phase 5 is stable.

### Generation

G = 0 everywhere. No optical generation. This isn't a photodiode simulator.

## Mobility models

Add these in order, each as its own module with its own test.

**Phase 1-2: constant.** mu_n = 1417, mu_p = 470 cm^2/(V s), undoped silicon
at 300 K. Good enough to get a diode I-V with the right ideality factor.

**Phase 3: doping dependent (Arora or Masetti).** Arora is simpler:

    mu = mu_min + mu_d / (1 + (N/N_ref)^A)

with separate parameter sets for electrons and holes, all temperature scaled.
Masetti is the TCAD standard and adds a term for the mobility upturn at very
high doping. Use Masetti if you want to match DEVSIM closely.

**Phase 5: field dependent (Caughey-Thomas).** This is what produces velocity
saturation, and velocity saturation is what produces the short channel
Id ~ V_ov behaviour.

    mu(E) = mu_0 / (1 + (mu_0 * E_parallel / v_sat)^beta)^(1/beta)

beta = 2 for electrons and 1 for holes. E_parallel is the field component
along the current, which in practice means along the mesh edge. Using the
full field magnitude here is a common shortcut, and a wrong one.

**Phase 5: surface mobility (Lombardi).** Carriers in a MOS channel scatter
off the oxide interface. Without this, the inversion layer mobility is two to
three times too high and Id is off by the same factor. Lombardi combines bulk,
acoustic phonon and surface roughness terms by Matthiessen's rule.

Mobility is where a device simulator earns or loses its quantitative accuracy.
The PDE solve can be perfect and the answer still three times off if the
mobility is wrong. Treat these models as first class code, not fudge factors.

## Boundary conditions

### Ohmic contacts

Charge neutrality plus thermal equilibrium at the contact node:

    p - n + Nd - Na = 0
    n * p = n_i^2

Solve the pair for n and p, then

    psi_contact = V_applied + V_T * asinh(N / (2 * n_i))

The asinh form is numerically stable. The naive `V_T * ln(N/n_i)` breaks in
lightly doped or compensated regions, where N can be near zero or negative.
Use asinh. This one bites often.

The quasi-Fermi levels at an ohmic contact are phi_n = phi_p = V_applied.

### Schottky contacts

Out of scope.

### MOS gate

Not a semiconductor node. The oxide gets Poisson only, no continuity
equations. The gate metal is a Dirichlet condition on psi with the work
function difference folded in:

    psi_gate = V_gate - Phi_MS

At the Si/SiO2 interface, enforce continuity of the normal component of D
(the displacement), not of E. Add a fixed interface charge Q_f if you want to
model one.

### Reflecting / symmetry boundaries

Homogeneous Neumann on all three unknowns: zero normal current, zero normal
field. Every boundary that isn't a contact is reflecting.

## What emerges, and must not be hardcoded

These are outputs, never inputs. If any of them shows up as a fitted
parameter anywhere in the code, that's a bug in the project, not a shortcut.

- Built-in potential of a junction
- Depletion width and its bias dependence
- Diode ideality factor, and the crossover from n=2 to n=1
- Threshold voltage and body effect
- Subthreshold slope, floored at 60 mV/decade at 300K
- DIBL and Vth roll-off with gate length
- Velocity saturation and the resulting linear-in-V_ov drain current

## Where drift-diffusion breaks down

Know these. They're the honest limits of the model, and being able to state
them is most of the value of having written it.

- **Below about 50 nm of channel**, carriers go quasi-ballistic and the local
  field no longer sets the local velocity. That needs energy balance
  (hydrodynamic) or Monte Carlo Boltzmann. It's why the sweep stops at 50 nm.
- **Velocity overshoot** can't show up in drift-diffusion by construction,
  since the model assumes carriers relax to the local field instantly.
- **Quantum confinement** in the inversion layer pushes the charge centroid
  away from the interface and thickens the effective oxide. It needs a
  Schrodinger-Poisson correction, and that's exactly where AtomSIM's solver
  could be reused if I take this further.
- **Tunnelling** (gate leakage, band to band) is absent. A classical transport
  model has no barrier penetration.

## Derivation chain, for the writeup

Drift-diffusion isn't fundamental. It's the first two moments of the
Boltzmann transport equation under the relaxation time approximation.

    Schrodinger in periodic potential
      -> Bloch states, band structure E(k)
      -> effective mass approximation near band edge, m* from band curvature
      -> semiclassical carriers, Boltzmann transport equation for f(r,k,t)
      -> zeroth moment: continuity equation
      -> first moment + relaxation time: drift-diffusion current
      -> close with Poisson: Van Roosbroeck system

The effective mass approximation is the step that lets you throw away quantum
mechanics for 1e17 carriers. Nc and Nv, which set n_i, come straight from m*:

    Nc = 2 * (2*pi * m_e* * k * T / h^2)^(3/2) * M_c
    Nv = 2 * (2*pi * m_h* * k * T / h^2)^(3/2)
    n_i = sqrt(Nc * Nv) * exp(-Eg / (2*k*T))

M_c = 6 for silicon, for its six equivalent conduction band minima.

This is the literal numerical handoff from the band layer. Keep
`constants.py` structured so those values can be swapped for computed ones
without touching anything else.
