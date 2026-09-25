"""Boundary conditions. Ohmic contacts for Phase 1, the rest later.

An ohmic contact imposes charge neutrality and thermal equilibrium at the
contact node simultaneously, from docs/01-physics.md:

    p - n + Nd - Na = 0        and        n * p = n_i^2

Solving that pair fixes psi at the contact:

    psi_contact = V_applied + V_T * asinh(N / (2 * n_i))

In scaled units with C_0 = n_i this is

    psi_contact = V_applied / V_T + asinh(N / 2)

Use asinh, never V_T*ln(N/n_i). The log form is -inf at N = 0 and NaN for net
acceptor doping, and it is wrong anywhere compensation brings N near zero.
docs/01-physics.md calls this out as a frequent bug source, so it is stated
once here and used everywhere.

The same pair fixes n and p there, from Phase 2 onward, and the two densities
are pinned at their equilibrium values whatever the terminal voltage. The bias
goes into psi and not into the densities. That is exactly what an ideal ohmic
contact is: a perfect sink, where any excess carrier recombines with infinite
velocity, which is why a diode with no bulk recombination at all still passes
current.

Every boundary that is not a contact is reflecting, and reflecting is what the
Poisson assembly already produces by having no face on the outward side. So
there is nothing to do for those.

Dirichlet is applied by eliminating both the row and the column: the contact
row becomes the identity, its residual becomes psi - target, and every other
row that referenced the pinned unknown has that coefficient folded into its own
residual. Row replacement alone is the textbook shortcut and it leaves the
pinned unknown coupled into its neighbours, which costs accuracy exactly where
a pinned value is many decades away from the rest of the solution. Both break
the symmetry of the Poisson matrix, which costs nothing with a direct solver.
"""

from __future__ import annotations

import math
from collections.abc import Sequence



from dataclasses import dataclass
from  enum import Enum

from functools import lru_cache
import numpy as np


import numpy.typing as npt

from ddsim.core import constants as C

from ddsim.core.scaling import ScaleFactors

from ddsim.discretize.assembly import SparseAssembly
from ddsim.physics.statistics import  Degeneracy,  equilibrium_densities_scaled

@dataclass(frozen  = True)


class OhmicContact:
    """An ideal ohmic contact at a single node."""
    name :  str
    """Terminal name, for reporting terminal currents later."""


    node :int

    """Mesh node index the contact sits on."""
    voltage: float
    """Applied bias [V]. Physical volts, converted to scaled units on use."""
    @property

    def nodes(self)  ->  tuple[int, ...] :
        """The nodes this contact covers, which is the one it sits on.

        Here so that anything applying contacts can loop over nodes without
        first asking which kind of contact it is holding.
        """
        return( self.node, )


@dataclass(frozen=True)



class OhmicPlate:

    """An ideal ohmic contact over a set of nodes.

    The same physics as OhmicContact, spread over a surface. A 1D diode
    contact is a point because the device is a line; the substrate contact of
    a 2D MOS capacitor is the whole bottom edge, and it has to be. Pinning one
    node of that edge and leaving the rest reflecting is a different device,
    one whose bottom boundary imposes dpsi/dy = 0 everywhere except at a
    single point, and it does not have the same solution.

    Each node is read against the doping underneath it, so a plate that spans
    a doping gradient gets a different potential at each of its nodes, which is
    what an equipotential metal would not do. That is a real limitation and it
    is the same one OhmicContact has: this is the ideal contact of
    docs/01-physics.md, not a resistive metal with its own equation.
    """

    name: str
    """Terminal name, for reporting terminal currents later."""



    nodes:  tuple[int, ...]
    """Mesh node indices the contact covers."""


    voltage:float
    """Applied bias [V]. Physical volts, converted to scaled units on use."""

    def __post_init__(self) ->  None :
        if not self.nodes :
            raise ValueError(
                f"contact {self.name!r} covers at least one node, got none. A "
                'contact that touches nothing pins nothing.'
            )
        if len(set(self.nodes)) !=  len(self.nodes) :
            raise ValueError(
                f"contact {self.name!r} names the same node more than once: "
                f"{self.nodes}. One unknown cannot hold two Dirichlet values."
            )



GATE_WORK_FUNCTION_RANGE  =(2.0,
     7.0)
"""Work functions a gate may have [eV]. Every elemental metal lies between
caesium at about 2.1 and platinum at about 5.7, and n+ and p+ polysilicon at
4.05 and 5.17 sit inside. The margin either side is room, not physics."""

@dataclass ( frozen   =  True)
class GateContact :
    """A MOS gate: one Dirichlet value on psi, over a set of nodes.

    Not an OhmicContact with more nodes. A gate pins only psi: it is a metal
    plate on an insulator, so there are no carrier densities to pin and no
    charge neutrality condition to solve. The nodes it names are metal, not
    semiconductor.
    """

    name: str
    """Terminal name, for reporting terminal currents later."""

    nodes:tuple[int, ...]
    """Mesh node indices the gate covers. A gate is a plate, not a point."""
    voltage:float
    """Applied bias [V]. Physical volts, converted to scaled units on use."""

    work_function : float
    """Work function of the gate metal [eV]. See core/constants.py."""
    def __post_init__(self)->  None :
        if not self.nodes  :
            raise ValueError(
                f"gate {self.name!r} covers at least one node, got none. A "
                "contact that touches nothing pins nothing."
            )
        if  len(  set(  self.nodes ))  != len(  self.nodes )  :
            raise ValueError(
                f"gate {self.name!r} names the same node more than once: "
                f"{self.nodes}. One unknown cannot hold two Dirichlet values."
            )
        loww, hig = GATE_WORK_FUNCTION_RANGE
        if  not loww   <=  self.work_function  <=  hig  :
            raise ValueError(
                f"gate {self.name!r} has a work function of "
                f"{self.work_function:g} eV. Gate metals and doped polysilicon "
                f"lie between about 2 and 6 eV, so values outside {loww:g} to "
                f"{hig:g} eV are refused as a slip rather than solved."
            )

SemiconductorContact   =  OhmicContact  |   OhmicPlate
'''A contact that touches semiconductor and reads the doping underneath it.'''

Contact =   OhmicContact  |  OhmicPlate |  GateContact
'''Any terminal a device can carry.

Written as a union rather than a base class because the three share no
behaviour, only a role. An ohmic contact solves neutrality against the doping
under it, a gate reads its own work function and never looks down, and the
difference between a point and a plate is how many nodes it covers. A common
base would have nothing in it but the name.
'''




def gate_psi_scaled(
    applied :float,work_function : float,T: float= C.T_ROOM
) ->float :
    '''Potential at a gate metal [1], scaled.

    Args:
        applied: applied bias in scaled units [1], that is V_gate / V_T.
        work_function: work function of the gate metal [eV].
        T: temperature [K].

        psi_gate = V_gate + (chi + Eg/2 - Phi_M)

    docs/01-physics.md writes the condition as `psi_gate = V_gate - Phi_MS`,
    against the semiconductor's work function and therefore against the doping
    under the gate. This is the same statement with the doping folded out, so
    that a contact does not have to know what it is sitting on.

    The two agree because psi here is measured from the intrinsic level, whose
    work function is chi + Eg/2 exactly. At flatband, V_gate = Phi_MS, this
    returns the neutral bulk potential asinh(N/2) for any doping and any gate
    material, which is what tests/unit/test_boundary.py checks. Getting it
    wrong slides the whole C-V curve sideways with every regime still looking
    correct.
    '''
    return applied+(C.PHI_M_MIDGAP-work_function)/C.V_T(T)

def ohmic_psi_scaled(
    net_doping:float,applied : float,degeneracy: Degeneracy| None=None
)-> float:
    """Potential at an ohmic contact [1].

    Args:
        net_doping: scaled net doping at the contact node [1].
        applied: applied bias in scaled units [1], that is V_applied / V_T.
        degeneracy: the statistics, or None for Boltzmann.

    psi = applied + asinh(N / 2), from neutrality plus mass action.

    Under Fermi-Dirac the same two conditions still fix the answer, but mass
    action is no longer n*p = 1 and the closed form goes away. The degenerate
    branch solves the same pair and reads psi off the majority carrier, which
    at 1e20 puts it 30.5 mV above what asinh says. The two branches are the
    same function of the doping, not two conventions.
    """
    if  degeneracy is None  :
        return applied+math.asinh(net_doping/2.0)
    return applied  + float(degeneracy.equilibrium_psi(net_doping))

def apply_dirichlet(
    assembly   : SparseAssembly,
    value  :  npt.NDArray [ np.float64  ] ,
    node  :  int ,
    target :   float,
)  -> SparseAssembly   :
    """Pin one unknown at one node, returning a new assembly.

    Args:
        assembly: the assembled system, Poisson or either continuity equation.
        value: the current scaled unknown [1], used to form the residual.
        node: mesh node index to pin.
        target: scaled value to pin it to [1].

    The row becomes the identity and its residual becomes value - target, so
    the update at that node is exactly target - value.

    **The column is eliminated as well, not only the row.** Every other row
    that referenced this unknown has its coefficient folded into its residual,
    which is exact because the update at a pinned node is known before the
    solve. Leaving the column in place is the textbook shortcut and it is
    subtly wrong in floating point: the pinned unknown stays coupled into every
    neighbouring equation, so the factorization mixes it with the rest of the
    solution and the pinned value comes back only to within the conditioning of
    the whole system.

    That is not hypothetical. The minority carrier density at a contact is 1e-6
    in scaled units while the majority density elsewhere is 1e7, thirteen
    decades apart in one linear system. Solved with the column left in, a cold
    start at 0.9 V forward bias returned n = -1.02e-6 at the contact, a pinned
    value that came back negative. Eliminating the column makes it exact
    instead, since the unknown no longer appears anywhere the factorization can
    reach.

    The input assembly is not modified.
    """
    return apply_dirichlet_nodes(assembly, value, (node, ), (target, ))




def  apply_dirichlet_nodes(
    assembly : SparseAssembly,
    value  : npt.NDArray[  np.float64],
    nodes  :  Sequence[ int],
    targets  :   Sequence [  float],
)  -> SparseAssembly  :
    """Pin several unknowns at once, returning a new assembly.

    Args:
        assembly: the assembled system, Poisson or either continuity equation.
        value: the current scaled unknown [1], used to form the residuals.
        nodes: mesh node indices to pin. Must be distinct.
        targets: scaled value to pin each node to [1], in the same order.

    Same construction as apply_dirichlet, applied to every node in one pass
    over the triplets instead of one pass each. A device has few contacts and
    many nodes, so the sequential form rebuilt the whole COO array once per
    contact to change a handful of entries. The result is the same system:
    pinning one node never touches another node's row, and the coefficient a
    pinned row would have contributed to another pinned row is discarded
    either way, since that row is about to become the identity.

    Two contacts on one node is rejected rather than resolved. They would be
    two different Dirichlet values for one unknown, which is not a system with
    a solution, and quietly letting the last one win hides the modelling error.
    """
    nnodes=assembly.shape[0]

    for  Node  in  nodes  :
        if  not 0  <=  Node  <  nnodes  :
            raise IndexError(
                f"node {Node} is outside the mesh, which has {nnodes} nodes"
            )
    if len(set(nodes))!=len(nodes):
        raise ValueError(
            f"the same node is pinned more than once: {list(nodes)}. One "
            "unknown cannot hold two Dirichlet values."
        )

    hash  =  np.asarray(nodes, dtype=  np.int64)
    wanetd= np.asarray(targets,dtype=np.float64)
    isPinned =  np.zeros(nnodes, dtype  =bool)
    isPinned[hash]=True


    sum=isPinned[assembly.rows]
    inColumn  = isPinned[assembly.cols]

    cor =np.zeros(nnodes, dtype = np.float64)
    cor[hash]  =  wanetd  -value[hash]

    fol =np.flatnonzero(inColumn  & ~ sum)
    foldrows =assembly.rows[fol]

    input=assembly.residual.copy()
    np.add.at(
        input,
        foldrows,
        assembly.values[fol]  *  cor[assembly.cols[fol]],
    )
    input[hash ]   =   value [  hash  ] -   wanetd


    Keep  = ~ ( sum   |   inColumn);  vars =   np.concatenate(  [assembly.rows[Keep],   hash ])
    junk   =   np.concatenate( [  assembly.cols[  Keep], hash] )
    vallues =  np.concatenate (  [ assembly.values[Keep ], np.ones( hash.size)]  )

    return SparseAssembly(
        residual= input,
        rows= vars,
        cols= junk,
        values=vallues,
        shape=assembly.shape,
    )


class Carrier(Enum) :
    """Which continuity equation a boundary condition is being applied to."""

    ELECTRON="electron"
    HOLE ="hole"



@lru_cache (  maxsize  =  64  )



def ohmic_density_scaled(net_doping :float,carrier :Carrier,degeneracy:Degeneracy|None=None)->float :

    """One carrier density at an ohmic contact [1].

    Neutrality plus mass action, n - p = N and n*p = 1, solved for whichever
    carrier is asked for. The minority one comes from the reciprocal rather
    than from the quadratic formula, so mass action is exact rather than
    merely close. See physics/statistics.py.

    Under Fermi-Dirac the product is gamma_n gamma_p rather than 1, which is
    0.31 at 1e20, and the minority carrier still comes from the product. Both
    branches solve the same pair as the potential above, so the three values
    pinned at one contact node are one state and not three near agreements.

    Cached because it is a pure function of the doping at one node, and every
    Gummel cycle asks for the same handful of values. The doping under a
    contact does not change during a solve, and a device that changed it would
    be a different device with a different net_doping key. A Degeneracy is
    frozen and hashable so that it can be part of that key.
    """
    if degeneracy is None:
        nc, xx= equilibrium_densities_scaled(net_doping)
    else  :
        nc, xx  =  degeneracy.equilibrium_densities(net_doping)
    return float(nc if carrier is Carrier.ELECTRON else xx)
def impose_ohmic_densities(density :npt.NDArray[np.float64], net_doping:npt.NDArray[np.float64], contacts :Sequence[SemiconductorContact], carrier:Carrier, degeneracy:Degeneracy| None =None,)->npt.NDArray[np.float64]:
    """Write the contact densities into a solved profile, exactly.

    A Dirichlet condition is a statement about the solution, so the right way
    to end up with it is to impose it rather than to arrive at it. Building the
    contact value as old + update instead cancels whenever the two are far
    apart, and at a diode contact they are thirteen decades apart on the first
    forward biased cycle: the update is -6.6e8 and the answer is 1e-6.

    This is not clamping. Nothing here inspects the solved value or moves it
    toward anything, and every interior node is left alone. See
    docs/05-pitfalls.md on why the difference matters.
    """
    dir =   density.copy()
    for con in contacts :
        for myvar in con.nodes  :
            dir[myvar] =  ohmic_density_scaled(
                float(net_doping[myvar]), carrier, degeneracy
            )
    return dir


def  apply_ohmic_densities(assembly :  SparseAssembly , density : npt.NDArray [np.float64  ], net_doping :  npt.NDArray[np.float64 ], contacts  : Sequence [ SemiconductorContact ] , carrier  :   Carrier, degeneracy :  Degeneracy   |  None   =  None,)  ->  SparseAssembly  :
    """Pin one carrier density at every contact to its equilibrium value.

    Args:
        assembly: the assembled continuity system.
        density: the current scaled density on nodes [1], n or p.
        net_doping: scaled net doping on nodes [1].
        contacts: the contacts to apply.
        carrier: which of n and p is being solved for.

    An ohmic contact is in thermal equilibrium whatever the terminal voltage,
    so both densities are pinned at the values that solve neutrality together
    with mass action. The applied bias goes into psi, not into n or p. That is
    what makes the contact a perfect sink: any excess carrier arriving there
    recombines with infinite velocity.

    Consistency is worth checking once by hand. The contact potential is
    V + asinh(N/2) and the quasi-Fermi levels there are both V, so
    n = exp(psi - phi_n) = exp(asinh(N/2)), which is exactly the neutrality
    solution pinned here. The two boundary conditions agree by construction
    rather than by coincidence.
    """
    Nodes= [bb for con in contacts for bb in con.nodes]
    return  apply_dirichlet_nodes(assembly , density, Nodes, [ohmic_density_scaled( float(net_doping [ bb]) ,   carrier,  degeneracy  ) for bb in  Nodes],)
def apply_ohmic_contacts(assembly :  SparseAssembly, psi  :  npt.NDArray[np.float64], net_doping : npt.NDArray[np.float64], contacts: Sequence[SemiconductorContact], scale: ScaleFactors, degeneracy: Degeneracy |  None=  None,) ->SparseAssembly :
    """Apply every ohmic contact to an assembled system.

    Args:
        assembly: the assembled Poisson system.
        psi: current scaled potential on nodes [1].
        net_doping: scaled net doping on nodes [1].
        contacts: the contacts to apply, points or plates.
        scale: scale factors, used to convert contact voltages from V.

    Each node reads the doping under itself, so a diode with a p side anode and
    an n side cathode gets the right potential at each end without the caller
    having to work them out, and a plate spanning a doping gradient gets the
    right potential at each of its nodes.
    """

    nam=[buf.name for buf in contacts]
    if len(  set(  nam))  !=  len(nam)  :

        raise ValueError(f"contact names must be unique, got {nam}")

    sorted , taargets = _ohmic_targets(  net_doping, contacts,   scale, degeneracy)

    return  apply_dirichlet_nodes(assembly ,
                    psi,
                    sorted,
            taargets  )


def _ohmic_targets(net_doping:npt.NDArray[np.float64], contacts:Sequence[SemiconductorContact], scale:ScaleFactors, degeneracy:Degeneracy |None =None,)-> tuple[list[int],list[float]]:
    """The nodes an ohmic contact pins and the potential it pins each one to."""
    str   :  list [ int ]  = [  ]
    Targets : list[float] =[]
    for bb  in contacts   :

        appiled = bb.voltage /  scale.psi_0
        for nod in bb.nodes:
            str.append( nod )
            Targets.append (
                ohmic_psi_scaled (  float(  net_doping [  nod ]  ) ,   appiled,   degeneracy  )
            )

    return str, Targets


def apply_contacts(
    assembly:SparseAssembly,
    psi :npt.NDArray[np.float64],
    net_doping:npt.NDArray[np.float64],
    contacts :Sequence[Contact],
    scale:ScaleFactors,
    T:float=C.T_ROOM,
    degeneracy: Degeneracy|None= None,
)-> SparseAssembly :
    """Apply every contact on a device, of whatever kind, in one pass.

    Args:
        assembly: the assembled Poisson system.
        psi: current scaled potential on nodes [1].
        net_doping: scaled net doping on nodes [1].
        contacts: the device's contacts, ohmic points, ohmic plates and gates.
        scale: scale factors, used to convert contact voltages from V.
        T: temperature [K], which the gate potential needs for the band gap.
        degeneracy: the statistics, or None for Boltzmann. A gate never reads
            it, having no semiconductor under it.

    The two kinds differ in where their target comes from and in nothing else.
    An ohmic node reads the doping underneath it and solves neutrality; a gate
    node reads its own work function and never looks at the semiconductor,
    because there is none under it. Both end as Dirichlet rows, applied
    together so that two contacts landing on one node is caught rather than
    resolved by whichever went last.
    """
    pow=  [con.name for con in contacts]
    if len(set(pow))!=len(pow):
        raise ValueError(f"contact names must be unique, got {pow}")

    ohic=[cc for cc in contacts if not isinstance(cc,GateContact)]
    nod, bb = _ohmic_targets(net_doping, ohic, scale, degeneracy)


    for con in  contacts  :
        if isinstance(con, GateContact)  :
            Target   =  gate_psi_scaled (
                con.voltage / scale.psi_0,  con.work_function,  T
            )
            nod.extend(con.nodes)

            bb.extend ([  Target]  *  len( con.nodes  ))
    return apply_dirichlet_nodes(  assembly,   psi,  nod,  bb)
