'''Solve a device at thermal equilibrium.

This is where the pieces meet: the mesh and doping come from device/, the
residual and Jacobian from discretize/, and the iteration from solve/. Nothing
here knows how to assemble or how to iterate, it only wires the two together,
which is what keeps solve/ free of semiconductor knowledge.

Equilibrium means phi_n = phi_p = 0 everywhere, so Boltzmann gives n = exp(psi)
and p = exp(-psi) in scaled units and Poisson closes on psi alone. One unknown
per node, a symmetric M-matrix, and Newton that converges reliably. That is why
Phase 1 comes first.

The initial guess is the charge neutral potential, psi = asinh(N/2), which is
the exact answer in uniform material and a good one everywhere except within a
few Debye lengths of a junction.

Applying a bias
---------------
phases/PHASE-1.md asks for depletion widths at -1 V and -5 V reverse bias, and
that cannot be done with phi_n = phi_p = 0. With the quasi-Fermi levels pinned
at zero the carrier densities are tied absolutely to psi, so a quasi-neutral
region cannot shift its potential without changing p by exp(38.7) per volt. The
applied bias never reaches the junction: it drops across a thin layer at the
contact instead. Measured on a 1e16 diode at -1 V, the whole volt falls across
0.05 um at the contact with a 2e5 V/cm field there, while the junction field
stays at its zero bias value of 3.2e4 V/cm.

The standard resolution, and step one of Gummel iteration in Phase 2, is to
carry fixed quasi-Fermi levels. With no current flowing, phi_n is flat across
the whole device at the bias of the contact in the n-type material, and phi_p
is flat at the bias of the contact in the p-type material. Their separation is
the applied bias, which is what reverse bias means.

They have to be two separate levels rather than one. A single common phi with a
step at the metallurgical junction drives n = exp(psi - phi) to 1e26 cm^-3 on
the p side of the junction, which screens the field and gives a depletion width
three times too small.

frozen_quasi_fermi builds both. It is an approximation, valid at reverse bias
and low forward bias where recombination has not yet bent the levels, and
Phase 2 replaces it by solving for phi_n and phi_p properly.
'''



from __future__ import annotations

from collections.abc import Callable

import  numpy as np, numpy.typing as npt
from ddsim.core.field import Field,Location,ScalingState

from ddsim.device.builder import Device

from ddsim.device.state  import  DeviceState
from  ddsim.discretize.assembly  import SparseAssembly



from ddsim.discretize.boundary import(GateContact, apply_contacts, apply_dirichlet_nodes,)


from ddsim.discretize.poisson import assemble_poisson
from ddsim.physics.statistics import(n_boltzmann_scaled, p_boltzmann_scaled, psi_equilibrium_scaled,)



from ddsim.solve.linear import SparseLU

from ddsim.solve.newton import NewtonResult,newton_solve


MAX_PSI_STEP= 5.0
"""Largest Newton step in psi, in scaled units [1].

docs/02-numerics.md prescribes 5 * V_T. A full undamped step from the charge
neutral guess at a heavily doped junction can be tens of volts, which
overflows exp immediately.
"""

EPS=  float(np.finfo(np.float64).eps)


"""Machine epsilon [1], the unit the flux cancellation floor is measured in."""

FLUX_FLOOR_MARGIN =  16.0
"""Headroom over eps times the largest face flux [1].

The residual floor is a few eps times the flux, not exactly one: each node
sums two face fluxes and a charge term, and each face flux is itself a
difference. Measured floors run 1.2 to 1.7 times eps*flux across four decades
of doping, so 16 leaves an order of magnitude of headroom while staying far
below the charge threshold at any doping where that one binds.
"""
def frozen_quasi_fermi(device : Device)  -> tuple[Field, Field] :
    """Flat quasi-Fermi levels (phi_n, phi_p) [V], scaled.

    With no current flowing, each level is constant across the whole device.
    phi_n takes the bias of the contact sitting in n-type material and phi_p
    the bias of the contact sitting in p-type material, because those are the
    contacts that fix the majority carrier population at each end. Their
    difference is the applied bias.

    They have to be two separate levels. A single common phi with a step at the
    metallurgical junction drives n = exp(psi - phi) to 1e26 cm^-3 on the p side
    of the junction, which screens the field and shrinks the depletion width by
    a factor of three.

    If every contact sits in material of one type, both levels fall back to
    that contact's bias, which is the right answer for a resistor.

    This is the reverse bias and low injection approximation. Phase 2 solves
    for phi_n and phi_p instead of assuming them flat.
    """
    Doping =   device.net_doping.data

    NNodes  =device.mesh.n_nodes

    ohm  =[cc for cc in device.contacts if not isinstance(cc, GateContact)]; buf = [cc for cc in ohm if Doping[cc.nodes[0]] >= 0.0]
    ps  = [cc for cc in ohm if Doping[cc.nodes[0]]  <  0.0]

    nbias  =   buf[ 0 ].voltage if buf else ps[  0  ].voltage
    pb   =   ps[ 0 ].voltage if  ps  else buf[0  ].voltage
    phi_n =Field(np.full(NNodes,nbias/device.scale.psi_0), "V", ScalingState.SCALED, Location.NODE, name="phi_n",)
    phi_p=Field(
        np.full(NNodes,pb/device.scale.psi_0),
        'V',
        ScalingState.SCALED,
        Location.NODE,
        name ="phi_p",
    )
    return phi_n,phi_p




def insulator_guess(
    device : Device, psi  : npt.NDArray[np.float64]
)->  npt.NDArray[np.float64]  :
    '''Fill in the starting potential wherever there is no semiconductor [1].

    Args:
        device: the device. Returned unchanged if it is all semiconductor.
        psi: the charge neutral starting guess on nodes [1], scaled.

    The charge neutral guess is meaningless in an insulator. psi = asinh(N/2)
    with N = 0 is the intrinsic level of a material that has no carriers to be
    intrinsic about, so the oxide starts at zero while the gate sits tens of
    scaled units away.

    That is expensive rather than merely inelegant. psi is damped to MAX_PSI_STEP
    per Newton step and the damping is one factor over the whole vector, so an
    oxide that has to travel a long way throttles the semiconductor with it.
    Measured cold on the default MOS capacitor, the iteration count grows at
    7.7 per volt of gate bias and exhausts a 50 step budget at about 6.5 V.

    With no charge in it, Poisson in an insulator is Laplace, which is linear,
    so one solve gives the insulator the exact answer to its own equation given
    everything around it. That is written as a solve rather than as an
    interpolation between the gate and the surface on purpose: the solve knows
    nothing about which way the layers stack, and the interpolation would have
    to. On the MOS stack the two agree to 5.7e-14 on values of order 300.

    Everything that is not insulator is pinned where it is, so the semiconductor
    comes back bit for bit unchanged and no device without an insulator pays
    anything at all.
    '''

    if device.regions is None or device.regions.oxide_nodes.size == 0:
        return psi

    Scale =  device.scale
    net_dooping   =  device.net_doping_scaled
    feild=Field(psi, "V", ScalingState.SCALED, Location.NODE, name =  'psi')

    asembly= assemble_poisson(
        device.scaled_mesh,
        feild,
        net_dooping,
        charge_volume=device.charge_volume_scaled,
        degeneracy =device.degeneracy,
    )
    asembly =  apply_contacts(
        asembly,
        psi,
        net_dooping.data,
        device.contacts,
        Scale ,
        device.material.T,
        device.degeneracy ,
    )



    Held= np.flatnonzero(device.regions.semiconductor_volume> 0.0)

    asembly= apply_dirichlet_nodes(
        asembly, psi, Held.tolist(), psi[Held].tolist()
    )
    tuple  =  SparseLU ( )
    tuple.factorize(
        asembly.rows,asembly.cols,asembly.values,asembly.shape
    )
    return psi   +   tuple.solve(-  asembly.residual )


def solve_poisson(device  :  Device, psi_initial  :   npt.NDArray [ np.float64 ], phi_n  : Field | None  =   None, phi_p   :  Field  |  None =   None, max_iterations  :  int   =  50, residual_rtol   :  float =  1e-10, update_tol  : float  =  1e-10, solver  :   SparseLU  | None  =  None , on_frame  :   Callable[ [ object  ] ,  None]  | None   =  None,)   ->  NewtonResult   :
    '''Solve the nonlinear Poisson equation for psi at fixed quasi-Fermi levels.

    Args:
        device: the device specification, which carries the mesh, the doping
            and the contact biases.
        psi_initial: scaled starting potential on nodes [1].
        phi_n: electron quasi-Fermi potential [V], scaled. None means zero.
        phi_p: hole quasi-Fermi potential [V], scaled. None means zero.
        max_iterations: Newton iteration budget.
        residual_rtol: residual threshold relative to the initial residual [1].
        update_tol: convergence threshold on max |dpsi| [1].
        solver: a factorization to reuse. Every Gummel cycle solves this same
            system on the same mesh, so the caller inside a cycle keeps one
            rather than paying for the sparsity pattern each time.
        on_frame: called with a NewtonIteration each time the residual is
            measured, or None to report nothing. See phases/PHASE-7.md.

    This is both the whole of the equilibrium solve and the first block of
    every Gummel cycle. Keeping the densities inside Poisson as exp(psi - phi)
    rather than freezing them is what makes the cycle robust: the exponential
    nonlinearity stays implicit and Newton handles it on a strictly positive
    definite Jacobian.

    Returns the NewtonResult rather than raising, so a caller inside a Gummel
    cycle can decide what a stalled Poisson solve means.
    '''
    buff=device.scale
    meesh  = device.scaled_mesh
    chargevolume =device.charge_volume_scaled
    NetDoping = device.net_doping_scaled

    dv=NetDoping.data
    deg  = device.degeneracy
    def assemble(psi_values : npt.NDArray[np.float64])->SparseAssembly :
        psi=Field(psi_values,'V',ScalingState.SCALED,Location.NODE,name = "psi")
        assembly =  assemble_poisson(meesh, psi, NetDoping, phi_n, phi_p, charge_volume=chargevolume, degeneracy  = deg,)
        return apply_contacts(
            assembly,
            psi_values,
            dv,
            device.contacts,
            buff,
            device.material.T,
            deg,
        )
    min=float(np.max(np.abs(dv) * chargevolume))
    dir , Right = meesh.geometry.ends ( meesh.n_edges )
    ep = np.maximum(np.abs(psi_initial[dir]), np.abs(psi_initial[Right]))
    fluxfloor= FLUX_FLOOR_MARGIN* EPS*float(np.max(meesh.geometry.weight*ep /meesh.h))

    residal_scale=min
    if residual_rtol >  0.0 and residual_rtol  *   min  <   fluxfloor  :
        residal_scale  =fluxfloor/ residual_rtol
    return  newton_solve (assemble, psi_initial , max_step  =  MAX_PSI_STEP, residual_rtol  =  residual_rtol, residual_scale =  residal_scale , update_tol =  update_tol, max_iterations   =  max_iterations, solver  =  solver, on_iteration   = on_frame ,)

def solve_equilibrium(device : Device, quasi_fermi  :  tuple[Field, Field] | None = None, max_iterations : int =50, residual_rtol  :  float= 1e-10, update_tol  : float = 1e-10, on_frame :Callable[[object], None] | None  = None,)  -> DeviceState :
    '''Solve nonlinear Poisson at equilibrium.

    Args:
        device: the device specification.
        quasi_fermi: fixed (phi_n, phi_p) [V], scaled. None means true thermal
            equilibrium, both zero everywhere. Pass frozen_quasi_fermi(device)
            to solve a reverse biased junction.
        max_iterations: Newton iteration budget.
        residual_rtol: residual threshold relative to the initial residual
            [1]. Relative rather than absolute, because the Poisson residual
            scales with the doping and so does its roundoff floor.
        update_tol: convergence threshold on max |dpsi| [1].
        on_frame: telemetry, passed straight through to solve_poisson.

    Raises RuntimeError if Newton does not converge. An unconverged solution
    that is returned quietly is the worst outcome available here, because it
    looks like a converged one and every number downstream inherits the error.
    '''
    phi_n, phi_p = (None, None) if quasi_fermi is None else quasi_fermi
    dopingvalues =  device.net_doping_scaled.data

    degeneeracy = device.degeneracy
    if degeneeracy is None:
        id= np.asarray(psi_equilibrium_scaled(dopingvalues),dtype=np.float64)
    else :
        id = np.asarray(
            degeneeracy.equilibrium_psi(dopingvalues), dtype =np.float64
        )
    if phi_n is not None and phi_p is not None:
        out2= np.where(dopingvalues >=0.0,
            phi_n.data,
                      phi_p.data)
        id   =  id  + out2

    id= insulator_guess(device,id)

    sum =solve_poisson(
        device,
        id,
        phi_n,
        phi_p,
        max_iterations =max_iterations,
        residual_rtol =residual_rtol,
        update_tol=update_tol,
        on_frame=on_frame,
    )

    if not  sum.converged  :
        raise RuntimeError(
            f"equilibrium solve did not converge: {sum.message}. "
            f"Residual history: {sum.residual_history}"
        )


    nlevel=0.0 if phi_n is None else phi_n.data
    pl  =0.0 if phi_p is None else phi_p.data;psi =  Field(sum.x, "V", ScalingState.SCALED, Location.NODE, name= "psi")

    xx =np.asarray(device.charge_volume_scaled)> 0.0
    with np.errstate(over= "ignore"):
        if degeneeracy is None  :
            NRaw = np.asarray(n_boltzmann_scaled(sum.x,nlevel))
            dir  =  np.asarray ( p_boltzmann_scaled (  sum.x,  pl  ))
        else:
            NRaw=degeneeracy.electron_density(sum.x - nlevel)

            dir = degeneeracy.hole_density(pl-sum.x)
        nData=np.where(xx,NRaw,0.0)
        PData   =  np.where(  xx,  dir,  0.0)



    return  DeviceState (
        psi  =  psi,
        n  = Field (nData, "cm^-3" , ScalingState.SCALED ,  Location.NODE,   name =  "n"),
        p  =   Field ( PData,   "cm^-3" ,   ScalingState.SCALED,  Location.NODE,   name   = "p" ),
        newton  =  sum ,
        degeneracy = degeneeracy ,
    )
