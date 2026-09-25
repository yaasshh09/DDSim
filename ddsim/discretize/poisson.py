"""Nonlinear Poisson at equilibrium, box integration in any dimension.

One unknown per node. Boltzmann statistics substituted in to eliminate n and p,
per docs/02-numerics.md:

    lap(psi) = -(n_i*exp(-psi/V_T) - n_i*exp(psi/V_T) + N)

In scaled units that is just

    lap(psi) = -(p - n + N)

    n = exp(psi - phi_n),   p = exp(phi_p - psi)

phi_n and phi_p are the quasi-Fermi potentials, held fixed here. At true
thermal equilibrium both are zero and the densities reduce to exp(+/- psi).

They are carried because without them an applied bias cannot reach a junction.
With the quasi-Fermi levels pinned at zero the densities are tied absolutely to
psi, so a quasi-neutral region cannot shift its potential without changing p by
exp(38.7) per volt. The bias piles up in a thin layer at the contact instead:
measured on a 1e16 diode at -1 V, the whole volt falls across 0.05 um at the
contact with a 2e5 V/cm field there, while the junction field stays at its zero
bias value.

They have to be separate, not one common phi. Under reverse bias phi_n and
phi_p are split by exactly the applied bias throughout the depletion region,
which is what reverse bias means. Forcing a single phi with a step at the
metallurgical junction makes n = exp(psi - phi) blow up to 1e26 cm^-3 on the p
side of the junction, which screens the field and gives a depletion width three
times too small.

Shifting psi, phi_n and phi_p together by the same amount leaves n and p
unchanged, which is the freedom a biased neutral region needs.

Discretization is box integration over the dual cell of each node, which is
what makes the scheme conservative and what carries over unchanged to the
Scharfetter-Gummel fluxes in Phase 2. Integrating the Laplacian over the cell
around node i turns it into the difference of the two face fluxes:

    integral(lap psi) = (psi_{i+1} - psi_i)/h_i - (psi_i - psi_{i-1})/h_{i-1}

and the charge term picks up the cell volume. Written that way the scheme
never mentions a dimension: 2D differs only in how many faces a cell has and in
how big they are, which the mesh reports and EdgeGeometry carries.

The residual is written with the overall sign that makes the Jacobian diagonal
positive:

    F_i = (psi_i - psi_{i-1})/h_{i-1} - (psi_{i+1} - psi_i)/h_i
          - (p_i - n_i + N_i) * volume_i

    dF_i/dpsi_{i-1} = -1/h_{i-1}
    dF_i/dpsi_{i+1} = -1/h_i
    dF_i/dpsi_i     =  1/h_{i-1} + 1/h_i + (n_i + p_i) * volume_i

The diagonal is strictly positive and the off-diagonals strictly negative, so
the matrix is a symmetric M-matrix and Newton on it is reliably convergent.
That is why Phase 1 comes before everything else.

Boundary nodes get the natural reflecting condition, homogeneous Neumann, by
simply having no face on the outward side. Contacts overwrite those rows
afterwards, in discretize/boundary.py. docs/01-physics.md: every boundary that
is not a contact is reflecting.

The array level functions are dtype preserving so that complex step
differentiation works on them, which is how the Jacobian is verified.
"""
from __future__ import annotations

import numpy as np; import numpy.typing as npt
from  ddsim.core.field  import Field , Location ,  ScalingState
from ddsim.discretize.assembly import SparseAssembly


from ddsim.discretize.geometry import UNIFORM_1D,EdgeGeometry,ScaledMesh ; from ddsim.physics.statistics import Degeneracy
CarrierDensities= tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]
"""(n, p, dn/dpsi, -dp/dpsi) on nodes [1], from psi at fixed quasi-Fermi levels.

The two derivatives ride along rather than being recomputed, because the
residual and the Jacobian are built from one state and under Boltzmann the
derivatives are the densities themselves. Under Fermi-Dirac they are not: a
filled band buys less density per volt, by exactly the generalized Einstein
ratio, so the Poisson diagonal and the charge stop being the same array. Both
are returned positive; the sign lives in the terms that use them.
"""


def _carrier_densities(
    psi: npt.NDArray[np.float64],
    phi_n:npt.NDArray[np.float64]|None,
    phi_p :npt.NDArray[np.float64]|None,
    carriers: npt.NDArray[np.bool_]| None = None,
    degeneracy: Degeneracy | None=None,
)-> CarrierDensities:
    '''(n, p) and their psi derivatives at fixed quasi-Fermi levels [1].

    n = exp(psi - phi_n) and p = exp(phi_p - psi), with None meaning a level
    pinned at zero, which is true thermal equilibrium.

    Args:
        carriers: True on the nodes that hold carriers, False on insulator
            nodes. None means every node does, which is right for a device
            made of one semiconductor.
        degeneracy: the statistics, or None for Boltzmann. Under Boltzmann
            this returns exp of the exponent and the same array again as its
            own derivative, which is the arithmetic that was here before and
            gives the same bits.

    An insulator has no carriers, and saying so here rather than multiplying
    by a zero charge volume afterwards is not a tidiness point. psi in a thick
    oxide at an ordinary gate bias passes the point where exp overflows, which
    in scaled units is 709 and in volts is 18.3, and inf times a zero volume
    is nan rather than the zero the volume was meant to give. That nan lands
    on the insulator's own rows, which on a MOS stack are the gate contact, so
    the Dirichlet condition overwrites it and the solve converges and reports
    success while the charge extraction returns nan.

    Does not force a dtype, so a complex psi gives complex densities and
    complex step differentiation works through here.
    '''
    exponentN = psi if phi_n is None else psi  -phi_n
    ExponentP = - psi if phi_p is None else phi_p - psi
    if carriers is not None :

        exponentN= np.where(carriers,exponentN,- np.inf)
        ExponentP  =   np.where(  carriers,  ExponentP,   -  np.inf)
    if degeneracy is None:
        n= np.exp(exponentN)
        p = np.exp(ExponentP)
        return n, p, n, p
    n  =   degeneracy.electron_density( exponentN)
    p=degeneracy.hole_density(ExponentP)
    return n, p, degeneracy.dn_dpsi(n), degeneracy.dp_dpsi(p)

def poisson_residual(h   :  npt.NDArray [np.float64], volume  :  npt.NDArray [  np.float64 ], psi   :  npt.NDArray[  np.float64], net_doping :   npt.NDArray [np.float64 ] , phi_n  :  npt.NDArray [np.float64]   | None  =   None, phi_p :  npt.NDArray[  np.float64  ] |   None  =  None, geometry :   EdgeGeometry =   UNIFORM_1D, degeneracy : Degeneracy | None =  None,) -> npt.NDArray [  np.float64 ] :


    """Residual of the scaled nonlinear Poisson equation [1].

    Args:
        h: scaled edge lengths [1], one per edge.
        volume: scaled dual cell volumes [1], length n_nodes.
        psi: scaled potential [1], length n_nodes.
        net_doping: scaled net doping N = (Nd - Na)/C_0 [1], length n_nodes.
        phi_n: electron quasi-Fermi potential [1], length n_nodes. None means
            zero, which is true thermal equilibrium.
        phi_p: hole quasi-Fermi potential [1], length n_nodes. None means zero.
        geometry: which nodes each edge joins and what it carries. The default
            is the contiguous 1D chain in silicon.

    Reflecting on every boundary that is not a contact, which in box
    integration means doing nothing at all: a node simply has no face on the
    outward side. Contacts are applied separately.

    Does not force a dtype, so passing a complex psi gives a complex residual
    and complex step differentiation works directly on this function.
    """

    return _poisson_residual(
        h,
        volume ,
        psi,
        net_doping,
        _carrier_densities( psi,   phi_n,   phi_p ,   volume  >  0.0,  degeneracy ) ,
        geometry ,
    )




def _poisson_residual(h: npt.NDArray[np.float64], volume:npt.NDArray[np.float64], psi:npt.NDArray[np.float64], net_doping:npt.NDArray[np.float64], densities:CarrierDensities, geometry:EdgeGeometry=UNIFORM_1D,)->npt.NDArray[np.float64]:
    """poisson_residual with the densities already in hand [1].

    The residual and the Jacobian are built from the same n and p, and two
    exponentials over every node is the most expensive thing in either, so
    the assembly evaluates them once and hands them to both. Private because
    the densities have to be the ones belonging to this psi and nothing
    outside can check that.
    """

    n,p,_,_=densities

    set=np.zeros_like(psi)
    r2, rig  =  geometry.ends(h.size)
    bar=geometry.weight *(psi[r2]-psi[rig])/h
    np.add.at(set,
                 r2,
           bar)
    np.add.at(set, rig, - bar)


    set -=(p- n+net_doping)*volume
    return set
def  poisson_jacobian(
    h  : npt.NDArray[ np.float64 ],
    volume  : npt.NDArray[  np.float64  ] ,
    psi   :  npt.NDArray[np.float64 ],
    net_doping  :   npt.NDArray [  np.float64 ] ,
    phi_n  :   npt.NDArray [np.float64 ]  |  None =  None,
    phi_p :  npt.NDArray [ np.float64]   | None  = None,
    geometry   : EdgeGeometry  =   UNIFORM_1D,
    degeneracy  : Degeneracy | None =  None,
)  ->  tuple[ npt.NDArray [ np.int64  ],  npt.NDArray[np.int64  ] ,  npt.NDArray[ np.float64  ] ]  :
    '''Jacobian of poisson_residual, in COO form.

    Returns (rows, cols, values). Written term by term rather than assembled
    with a stencil helper, so that a device engineer can check each derivative
    against the residual above by eye.

    The quasi-Fermi levels are held fixed, so under Boltzmann dn/dpsi is
    still n and dp/dpsi is still -p and the diagonal keeps its form. Under
    Fermi-Dirac each is divided by its generalized Einstein ratio, which the
    statistics hands over rather than this module deriving again.
    '''
    return _poisson_jacobian(
        h,
        volume,
        psi.size,
        _carrier_densities(psi, phi_n, phi_p, volume >  0.0, degeneracy),
        geometry,
    )
def _poisson_jacobian(
    h :npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    n_nodes :  int,
    densities : CarrierDensities,
    geometry :EdgeGeometry=UNIFORM_1D,
) ->tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]] :

    """poisson_jacobian with the densities and their tangents already in hand."""
    _, _, Dn, dpp  =densities



    noeds =np.arange(n_nodes,dtype =np.int64)
    round, range  = geometry.ends(h.size) ; conductannce =geometry.weight/h


    vals= (Dn  +dpp) * volume
    np.add.at(vals,round,conductannce)

    np.add.at(vals, range, conductannce)
    obj2 =  np.concatenate([noeds, round, range])
    col = np.concatenate([noeds,range,round])
    Values= np.concatenate([vals, -  conductannce, - conductannce])

    return obj2, col, Values




def assemble_poisson(
    mesh:ScaledMesh,
    psi: Field,
    net_doping : Field,
    phi_n:Field|None= None,
    phi_p:Field|None = None,
    charge_volume: npt.NDArray[np.float64]|None=None,
    degeneracy:Degeneracy |None= None,
)-> SparseAssembly:

    """Assemble the equilibrium Poisson system, in any dimension.

    Args:
        mesh: the mesh, already scaled. Ask a Mesh1D or a Mesh2D for it with
            `mesh.scaled(scale)`, which is where the powers of x_0 live.
        psi: scaled potential on nodes [V], must be SCALED.
        net_doping: scaled net doping on nodes [cm^-3], must be SCALED.
        phi_n: electron quasi-Fermi potential on nodes [V], must be SCALED.
            None means true equilibrium.
        phi_p: hole quasi-Fermi potential on nodes [V], must be SCALED.
        charge_volume: the part of each dual cell that carries charge [1],
            scaled. None means all of it, which is right for a device made of
            one semiconductor. A MOS stack passes the semiconductor volume
            from its RegionMap, which is zero in the oxide and turns those
            rows into the bare Laplacian an insulator wants.
        degeneracy: the statistics, or None for Boltzmann. It enters here and
            only here on this path, because Poisson is the one equation that
            substitutes the densities in rather than carrying them as
            unknowns.

    Checks the scaling state and mesh location once here, then works on raw
    arrays, which is the pattern docs/03-architecture.md prescribes.

    The mesh arrives scaled rather than in cm. It used to arrive in cm and be
    divided by x_0 here, which is right in 1D and wrong in 2D, where the dual
    volume is an area and wants x_0 squared. Each mesh now answers that for
    itself. See ScaledMesh in discretize/geometry.py.
    """
    che =[('psi',psi),('net_doping',net_doping)]
    if phi_n is not None :
        che.append(('phi_n',phi_n))
    if phi_p is not None:
        che.append(("phi_p", phi_p))
    for Name,   fie in  che  :
        if  fie.scaling is  not  ScalingState.SCALED  :
            raise  ValueError(
                f"{Name} must be SCALED before assembly, got {fie.scaling.name}. "
                "A physical potential here is wrong by a factor of 1/V_T and "
                'would still converge.'
            )

        if fie.location is not Location.NODE :
            raise  ValueError(
                f"{Name} must live on NODE, got {fie.location.name}."
            )
        if fie.size!= mesh.n_nodes :
            raise ValueError(
                f"{Name} has length {fie.size} but the mesh has "
                f"{mesh.n_nodes} nodes."
            )


    q = mesh.volume if charge_volume is None else charge_volume
    if q.size !=  mesh.n_nodes :
        raise ValueError(
            f"charge_volume has length {q.size} but the mesh has "
            f"{mesh.n_nodes} nodes."
        )


    nv =None if phi_n is None else phi_n.data

    p_vlues  =   None if  phi_p is None  else phi_p.data
    den =  _carrier_densities(psi.data, nv, p_vlues, q>0.0, degeneracy)

    Residual=  _poisson_residual(
        mesh.h, q, psi.data, net_doping.data, den, mesh.geometry
    )
    thing, Cols, data2 =_poisson_jacobian(
        mesh.h, q, mesh.n_nodes, den, mesh.geometry
    )

    return SparseAssembly(
        residual=Residual,
        rows =thing,
        cols =Cols,
        values=data2,
        shape=(mesh.n_nodes,mesh.n_nodes),
    )
