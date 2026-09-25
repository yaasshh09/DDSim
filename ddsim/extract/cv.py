"""Terminal charge and small signal capacitance. Post processing, one extra solve.

Where the charge comes from
---------------------------
The same place the terminal current comes from in extract/iv.py: the residual
at the contact node, before the Dirichlet row replaces it. That residual is the
flux balance over the contact's dual cell, and what it is short by is exactly
the charge that has to be supplied there. Written that way the charges on the
terminals sum to zero identically rather than approximately, because the
interior residuals are zero at a converged solution and the sum of all of them
telescopes to the flux through the outer boundary, which is zero on a device
whose non contact boundaries are reflecting.

Reading it off the edge fluxes instead would be a second, independent
calculation of the same thing, and the two would agree to a few digits rather
than to the last bit.

Where the capacitance comes from
--------------------------------
docs/02-numerics.md asks for

    (J_dc + i omega M) x = b,   Y = G + i omega C,   C = Im(Y)/omega

and for an equilibrium device M is empty. Boltzmann statistics are substituted
into Poisson, so n and p are not unknowns and there are no dQ/dt terms to put
in a mass matrix: the carriers follow the potential instantaneously by
construction. What is left is the omega to zero limit, which is exact rather
than approximate, and it is the quasi static or low frequency C-V.

That limit is still a small signal solve and not a difference of two DC
solutions. Differentiating F(psi; V) = 0 with respect to the terminal bias
gives J dpsi/dV = -dF/dV, whose right hand side is nonzero only at the
Dirichlet rows of the terminal being swept. One linear solve on the DC Jacobian
gives dpsi/dV exactly, and dQ/dV follows by pushing it back through the
unpinned rows at the terminal. No step size, no truncation error.

Frequency dependence proper needs the coupled 3N system and its mass matrix,
which needs transport in two dimensions. That is Phase 6.

The high frequency end
----------------------
Reachable without any of that, by the approximation every textbook makes: a
signal fast compared to minority carrier generation leaves the inversion charge
where it is, so only the majority carrier responds. That is one term removed
from the Jacobian used for the derivative solve, and nothing at all changed in
the DC solution. It is what makes the depletion minimum a fixed number, because
the depletion region stops growing once the surface inverts but the frozen
inversion layer does not screen the small signal.

Which carrier is the minority one is read from the doping, not from the local
densities. At an inverted surface the electrons outnumber the holes and are
still the minority carrier: they are the ones that had to be generated.
"""

from __future__ import annotations

from  collections.abc import Callable

from dataclasses import dataclass

from enum  import  Enum
import numpy as np

import numpy.typing  as  npt

from ddsim.core  import  constants as C


from ddsim.core.field import Field, Location, ScalingState

from ddsim.device.builder import Device

from ddsim.device.equilibrium import frozen_quasi_fermi,solve_equilibrium

from ddsim.device.state import DeviceState


from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import apply_dirichlet_nodes
from ddsim.discretize.poisson import assemble_poisson
from ddsim.mesh.mesh1d import Mesh1D


from ddsim.solve.linear import SparseLU

class Response(Enum) :
    '''Which carriers follow the small signal.'''



    LOW_FREQUENCY='low_frequency'
    """Both of them, which is the omega to zero limit and is exact here."""

    HIGH_FREQUENCY ="high_frequency"
    """The majority carrier only.

    The standard high frequency approximation: the signal is fast compared to
    minority carrier generation, so the inversion charge cannot follow it. Not
    a limit of this formulation but a model of one, and it is stated as such.
    """




def _charge_unit(  device   :   Device, width  : float |  None )  ->  float   :
    """Physical charge per scaled charge, per unit terminal area [C/cm^2].

    The scaled charge is a density in units of C_0 integrated over a dual cell
    in units of x_0, so the power of x_0 is the dimension. A 1D device is a
    slab and its charge is already per unit area, so width is 1 there; a 2D
    device gives charge per unit depth and the terminal's own extent turns it
    into a density.
    """
    sca= device.scale

    if  isinstance (  device.mesh ,   Mesh1D )  :
        etent=1.0 if width is None else width
        return C.q* sca.C_0* sca.x_0/etent
    etent = device.mesh.x_axis.length if width is None else width
    return C.q*sca.C_0 *sca.x_0**2/etent


def _contact_nodes(device:Device,contact : str) ->tuple[int,...]:
    """The nodes a named contact covers, or a readable refusal."""
    for  chr  in device.contacts  :


        if chr.name==contact:
            return chr.nodes
    raise KeyError(
        f"no contact named {contact!r} on this device, which has "
        f"{sorted(chr.name for chr in device.contacts)}"
    )
QuasiFermi = tuple[Field, Field] | None


"""Fixed (phi_n, phi_p) [V], scaled, or None for true thermal equilibrium.

Whatever the state was solved with has to be passed back in, because the
residual and the Jacobian are rebuilt here and both depend on it. See
cv_sweep, which builds them itself and keeps the two consistent.
"""


def _bare_poisson(device :Device,state:DeviceState,quasi_fermi:QuasiFermi=None)->SparseAssembly:

    """The Poisson system at the solved state, with no contacts applied.

    Unpinned on purpose. A Dirichlet row throws away the flux balance at the
    contact, and that flux balance is the terminal charge.
    """
    psi=Field(
        state.psi.data,"V",ScalingState.SCALED,Location.NODE,name = 'psi'
    )
    phi_n, phi_p = (None, None)  if quasi_fermi is None else quasi_fermi
    return  assemble_poisson(device.scaled_mesh, psi, device.net_doping_scaled, phi_n , phi_p , charge_volume  =   device.charge_volume_scaled ,)


def terminal_charge(
    device  : Device,
    state :DeviceState,
    contact : str,
    quasi_fermi :  QuasiFermi= None,
    width  : float | None = None,
)-> float:
    """Charge on one terminal [C/cm^2].

    Args:
        device: the device the state was solved on.
        state: a converged solution.
        contact: name of the terminal.
        quasi_fermi: the fixed (phi_n, phi_p) the state was solved with. None
            means true equilibrium, both zero, which is what a capacitor with
            a grounded body has.
        width: extent of the terminal transverse to the field [cm], used to
            report per unit area. None takes the device width in 2D and 1 in
            1D, where the charge is already a density.

    Positive means positive charge on that terminal, so a MOS gate biased
    above flatband reports a positive charge.

    What balances it is not the other terminal. This is the flux of D over one
    contact's own cells, and on a MOS capacitor the body is an ohmic contact
    sitting in neutral bulk where the field has already died, so it holds
    almost nothing: measured at 1e-18 of the gate charge. The gate's partner is
    the depletion and inversion charge spread through the silicon, and it is
    the gate, the body and that volume charge which sum to zero on a converged
    solution. See test_the_body_contact_is_not_the_other_plate in
    tests/analytic/test_mos_cv.py, which is the guard on exactly this.
    """

    Assembly = _bare_poisson(device, state, quasi_fermi)
    junk =list(_contact_nodes(device,contact))

    return float(np.sum(Assembly.residual[junk]))*  _charge_unit(device, width)

def _frozen_diagonal(
    device:Device,state:DeviceState
)->npt.NDArray[np.float64]:

    """The minority carrier's contribution to the Jacobian diagonal [1].

    Each carrier contributes its own density times the cell volume, because
    dn/dpsi = n under Boltzmann statistics. Removing this term is what stops
    the minority carrier following the small signal.

    Minority is decided by the doping. In p-type material the electron is the
    minority carrier everywhere, including at an inverted surface where it
    outnumbers the hole, because it is still the one that had to be generated.
    """
    Doping   =   device.net_doping_scaled.data;min   = np.where( Doping  <  0.0,  state.n.data,   state.p.data )
    return np.asarray(min *device.charge_volume_scaled)
def small_signal_capacitance(
    device:Device,
    state:DeviceState,
    contact: str,
    response:Response=Response.LOW_FREQUENCY,
    quasi_fermi:QuasiFermi =None,
    width :float|None=None,
)->float :
    """dQ/dV at one terminal [F/cm^2], from one linear solve.

    Args:
        device: the device the state was solved on.
        state: a converged solution.
        contact: name of the terminal to sweep and to measure.
        response: which carriers follow the signal. See Response.
        quasi_fermi: the fixed (phi_n, phi_p) the state was solved with.
        width: extent of the terminal transverse to the field [cm].

    Exact, in the sense that it differentiates the solved system rather than
    differencing two solutions of it. There is no step size to choose and no
    truncation error to carry.
    """
    assmbly =  _bare_poisson(device, state, quasi_fermi)
    row, Cols,   vales  =   assmbly.rows, assmbly.cols,   assmbly.values

    if response is Response.HIGH_FREQUENCY:
        bin  =  np.arange(device.mesh.n_nodes, dtype  =  row.dtype)
        row  = np.concatenate([row, bin])
        Cols  = np.concatenate( [Cols,   bin ]  )
        vales = np.concatenate([vales, -_frozen_diagonal(device, state)])

    Nodes= list(_contact_nodes(device,contact))
    Dpsi = _potential_derivative(device,contact,row,Cols,vales)


    scattred=  np.zeros(device.mesh.n_nodes, dtype  =np.float64)
    np.add.at(scattred,row,vales*Dpsi[Cols])
    return float(np.sum(scattred[Nodes])) * _charge_unit(device, width)


def _potential_derivative(
    device : Device,
    contact:  str,
    rows:npt.NDArray[np.int64],
    cols: npt.NDArray[np.int64],
    values: npt.NDArray[np.float64],
)  ->npt.NDArray[np.float64]  :
    """dpsi/dV at every node [1 per V], for a unit bias on one terminal.

    Differentiating F(psi; V) = 0 leaves a linear system with the same
    Jacobian the DC solve ended on, a homogeneous right hand side, and the
    derivative of each Dirichlet target as its boundary data. Every terminal
    but the swept one is held, because its bias is not moving.

    Assembled through apply_dirichlet_nodes rather than by hand so that the
    column elimination is the same one the nonlinear solve uses. That helper
    solves J*delta = -F for a delta whose pinned entries are known, which is
    exactly this problem once F is zero and the known entries are derivatives
    instead of corrections.
    """
    NNodes   = device.mesh.n_nodes
    nod : list[int]  =  []

    lst  : list[  float  ]  =  [ ]
    for sorted in device.contacts   :
        derivvative  =  1.0 / device.scale.psi_0 if sorted.name ==  contact  else  0.0
        nod.extend(sorted.nodes)
        lst.extend([derivvative] *len(sorted.nodes))

    System= apply_dirichlet_nodes(SparseAssembly(residual =np.zeros(NNodes), rows=rows, cols =cols, values=values, shape= (NNodes, NNodes),), np.zeros(NNodes), nod, lst,)
    slver  =  SparseLU ( )
    slver.factorize(System.rows, System.cols, System.values, System.shape);  return slver.solve(-System.residual)



@dataclass(frozen=True)




class CVPoint:
    """One bias point of a C-V sweep."""
    gate_voltage: float
    """Applied bias at the swept terminal [V]."""

    capacitance :   float
    """Small signal capacitance there [F/cm^2]."""

    charge : float
    """Charge on that terminal [C/cm^2]."""


    state :DeviceState
    '''The converged solution, kept for band diagrams and profile plots.'''

@dataclass(frozen =True)

class CVCurve :
    """A capacitance sweep, complete or as far as it got."""
    contact:str
    """Name of the swept terminal."""



    response:Response
    """Which carriers were allowed to follow the signal."""

    points :  tuple[CVPoint, ...]

    """The converged bias points, in the order they were requested."""
    complete: bool
    """Whether every requested voltage was reached."""

    message  :  str  = ""
    """Why the sweep stopped, when it did not finish."""

    @property
    def gate_voltage(self)  -> npt.NDArray[np.float64] :
        """Applied bias at each point [V]."""
        return np.array([piont.gate_voltage for piont in self.points])


    @property
    def capacitance(self)->npt.NDArray[np.float64]:
        """Small signal capacitance at each point [F/cm^2]."""
        return np.array([Point.capacitance for Point in self.points])


    @property
    def charge(self) ->npt.NDArray[np.float64] :
        '''Terminal charge at each point [C/cm^2].'''
        return np.array([item2.charge for item2 in self.points])



    def __repr__(self)->str:
        State = "complete" if self.complete else "stopped early"
        if not self.points  :
            return  f"CVCurve {self.contact} empty, {State}"
        return(
            f"CVCurve {self.contact} {len(self.points)} points "
            f"{self.gate_voltage[0]:+.3g} to {self.gate_voltage[-1]:+.3g} V, "
            f"{self.response.value}, {State}"
        )

@dataclass(frozen=True)


class CVFrame:
    '''One finished capacitance point, reported while the sweep is running.

    Scalars only, for the reason IVFrame carries scalars only. Nothing here
    continues from the state, but a reader able to reach into it could still
    change the charge and capacitance read off it at the point after.
    '''

    index:int
    """Position in the requested voltage list, from zero."""

    gate_voltage  : float
    '''Applied bias at the swept terminal [V].'''

    capacitance  :  float
    """Small signal capacitance there [F/cm^2]."""
    charge :  float

    """Charge on that terminal [C/cm^2]."""


def cv_sweep(
    device :Device,
    contact : str,
    voltages :  list[float],
    response :  Response= Response.LOW_FREQUENCY,
    width: float  |  None  = None,
    max_iterations : int =50,
    on_frame:Callable[[object], None] |  None = None,
)  ->CVCurve :
    '''Sweep one terminal and measure the capacitance at each bias.

    Args:
        device: the device. Its own bias at the swept contact is overridden.
        contact: name of the terminal to sweep and to measure.
        voltages: the biases wanted [V].
        response: which carriers keep up with the small signal. With
            low_frequency both do. With high_frequency the minority carrier
            is held still, the way it is in a fast real measurement.
        width: extent of the terminal transverse to the field [cm].
        max_iterations: how many Newton steps each point gets before it
            counts as failed [1].
        on_frame: telemetry, or None to report nothing. Carries a
            NewtonIteration per Poisson iteration and a CVFrame per point.
            There is no ContinuationEvent on this path: the points are
            independent solves rather than a ramp. See phases/PHASE-7.md.

    Each point is solved from the charge neutral guess rather than continued
    from the one before it. Equilibrium Poisson with the insulator filled in
    converges in under twenty steps anywhere in the useful bias range, so
    continuation buys little, and independent points mean one bias that fails
    to converge does not spoil the ones after it.

    Every point carries flat quasi-Fermi levels at the bias of the ohmic
    contact rather than the zero of true equilibrium. For a grounded body
    those are the same thing and nothing moves. For a biased one they are the
    difference between a curve that shifts with the body and a curve that does
    not: with the levels pinned at zero, a body bias cannot reach the surface
    at all. It is screened inside a few Debye lengths of its own contact,
    because a quasi-neutral region cannot move its potential without moving
    its majority carrier density by exp(38.7) per volt. Measured on the
    default capacitor at 0.5 V of body bias, the stack that should have been
    flat was bent by the whole 0.5 V.

    The approximation is exact here rather than merely good, which is not
    true of the diode it was written for. A capacitor passes no current, so
    both levels really are flat and really do sit at the body potential.

    Returns everything it reached, with complete=False and a message if a
    point did not converge.
    '''
    _contact_nodes(device, contact)
    Points:  list[CVPoint]= []
    for yy in voltages  :
        chr  =  device.with_bias (  ** {  contact  :  yy  }  )
        lev= frozen_quasi_fermi(chr)
        try :
            sta =solve_equilibrium(
                chr,
                quasi_fermi =lev,
                max_iterations=max_iterations,
                on_frame= on_frame,
            )
        except RuntimeError  as Error  :


            return CVCurve(
                contact  = contact,
                response = response,
                points = tuple(Points),
                complete  =False,
                message  =  f"did not converge at {yy:+g} V: {Error}",
            )
        Capacitance =small_signal_capacitance(
            chr,
            sta,
            contact,
            response  =response,
            quasi_fermi  = lev,
            width =width,
        )
        s2  =terminal_charge(chr, sta, contact, quasi_fermi=  lev, width = width)

        Points.append (CVPoint (gate_voltage   =  yy , capacitance   =  Capacitance , charge  =  s2, state  =  sta ,))

        if on_frame is not None  :
            on_frame(CVFrame (index   =   len(  Points  )  -  1 , gate_voltage   =  yy, capacitance   = Capacitance, charge   =   s2 ,))
    return CVCurve(
        contact = contact,
        response = response,
        points  = tuple(Points),
        complete  =True,
    )
