"""Structured tensor product 2D mesh, box integration finite volume.

phases/PHASE-4.md is explicit about starting here rather than with Delaunay: a
rectangular MOS capacitor does not need unstructured meshing, and a rectangle
cannot produce an obtuse triangle, so the negative dual area failure mode that
docs/05-pitfalls.md describes cannot arise. mesh/quality.py exists for the day
unstructured meshing does arrive, and refuses a mesh that would hit it.

Built as the tensor product of two Mesh1D axes. That is deliberate reuse and
not a shortcut. The grading solver and the dual grid construction are
inherited from the 1D mesh rather than written a second time, and grading a 2D
mesh towards a junction is then just grading the axis.

The dual cells sum to the domain area to about 1e-16 relative, not exactly.
Summing the outer product reassociates the terms compared with multiplying the
two 1D sums. The 1D sum is not unconditionally exact either: it is exact on
the Phase 0 acceptance case and off by one bit on other node counts.

Numbering
---------
Row major, x fastest:

    node(i, j) = j*nx + i

Edges come in two families, horizontal first then vertical. Within a family
they run in the same order as the nodes. Nothing downstream depends on that
ordering, and tests/unit/test_edge_list_assembly.py proves it: shuffling the
edge list does not change the answer. The ordering is chosen to be readable,
not to be relied on.

What box integration needs from an edge
---------------------------------------
Two lengths, and it is worth being careful about which is which.

    h          the distance between the two nodes, along the edge
    dual_face  the extent of the face the flux crosses, across the edge

For a horizontal edge, the face it crosses is vertical, so `dual_face` comes
from the **y** dual grid. For a vertical edge it comes from the x dual grid.
Swapping those two is the classic slip, and on a square mesh it makes no
difference at all, which is what makes it dangerous. Test on a rectangle.

Units and depth
---------------
Lengths in cm, following docs/06-constants.md. A 2D simulation is per unit
depth, so `dual_face` is a length [cm] and `volume` is an area [cm^2]. The
scaling that the assemblies want is therefore

    h          / x_0
    dual_face  / x_0
    volume     / x_0^2

which is the d-dimensional rule `volume / x_0^d` and `face / x_0^(d-1)`
evaluated at d = 2. In 1D the same rule gives a dual_face of x_0^0 = 1, which
is exactly the default that discretize/geometry.py carries, so the two cases
are the same statement rather than two conventions.

A device uniform in y solves to the same psi, n and p as the 1D device with
the same x profile. The residual is a constant multiple of the 1D one, the
constant being the device height in scaled units, so the solution is identical
while the residual is not. That is the acceptance criterion in PHASE-4.md and
it is checked in tests/analytic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np, numpy.typing as  npt

from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.geometry import  EdgeGeometry ,   ScaledMesh
from ddsim.mesh.mesh1d import Mesh1D,  uniform_mesh_1d


@dataclass(frozen =  True)
class Mesh2D:
    """A structured 2D mesh with its dual grid and edge list."""

    x_axis : Mesh1D
    '''The x axis, as a 1D mesh. Carries its own dual grid.'''

    y_axis : Mesh1D

    """The y axis, as a 1D mesh."""
    node_x:npt.NDArray[np.float64]
    """x position of every node [cm], length n_nodes."""
    node_y: npt.NDArray[np.float64]
    """y position of every node [cm], length n_nodes."""
    h :npt.NDArray[np.float64]
    """Length of every edge [cm], the distance between its two nodes."""

    dual_face  : npt.NDArray[np.float64]
    """Extent of the face every edge crosses [cm], per unit depth."""
    volume:  npt.NDArray[np.float64]
    """Dual cell area of every node [cm^2], per unit depth."""


    edge_nodes:npt.NDArray[np.int64]
    """Shape (n_edges, 2). The two nodes of each edge."""
    @property
    def nx(self)->int:
        """Nodes along x."""
        return self.x_axis.n_nodes


    @property
    def ny(self)-> int  :
        """Nodes along y."""
        return  self.y_axis.n_nodes
    @property
    def n_nodes(self) ->int :
        """Total nodes."""
        return int ( self.node_x.size)

    @property
    def n_edges(  self  )  ->   int  :
        """Total edges, both families."""
        return int (  self.h.size  )
    @property
    def n_horizontal(self)->int:
        """How many edges are in the horizontal family, which come first."""
        return (  self.nx   -  1) *  self.ny
    def node_at(self, i: int, j:  int)->  int  :
        """The node index at column i, row j."""
        return j *self.nx + i

    def edge_geometry(
        self,
        eps_r:npt.NDArray[np.float64] |float=1.0,
        semiconductor_face:npt.NDArray[np.float64] |None =None,
    )-> EdgeGeometry :
        """What the assemblies need to work on this mesh.

        Args:
            eps_r: permittivity of each edge relative to the scaling
                permittivity [1]. 1.0 everywhere in a single material device.
            semiconductor_face: the part of each edge's face a carrier flux
                can cross [cm]. None means all of it, which is right for a
                device made of one semiconductor.

        The dual faces are handed over unscaled, in cm. The device layer
        divides them by x_0, exactly as it already does for h and volume. See
        the module docstring on why the power of x_0 differs between them.
        """
        return EdgeGeometry(
            edge_nodes = self.edge_nodes,
            dual_face= self.dual_face,
            eps_r =eps_r,
            semiconductor_face = semiconductor_face,
        )


    def scaled(self , scale :   ScaleFactors, eps_r  :  npt.NDArray[ np.float64] | float  =  1.0, semiconductor_face  : npt.NDArray[ np.float64 ] |  None  =   None,) ->  ScaledMesh  :
        """This mesh in the units the assemblies work in.

        Args:
            scale: the de Mari scale factors.
            eps_r: permittivity of each edge relative to the scaling
                permittivity [1]. A single material device leaves it at 1.0;
                a MOS stack passes what device/regions.py worked out.
            semiconductor_face: the part of each edge's face a carrier flux
                can cross [cm]. None means all of it. A MOS stack passes what
                device/regions.py worked out, and it is scaled here by the
                same power of x_0 as the dual face, because it is one.

        The two powers of x_0 differ, and that is the whole point of asking
        the mesh instead of doing it at the call site. See ScaledMesh.
        """

        return ScaledMesh(h =self.h/scale.x_0, volume =self.volume/ scale.x_0**2, geometry =EdgeGeometry(edge_nodes=self.edge_nodes, dual_face= self.dual_face/scale.x_0, eps_r= eps_r, semiconductor_face=(None if semiconductor_face is None else semiconductor_face/ scale.x_0),),)

    def __repr__(self) ->str:

        return(
            f"Mesh2D {self.nx}x{self.ny} = {self.n_nodes} nodes, "
            f"{self.n_edges} edges, "
            f"{self.x_axis.length:.4e} by {self.y_axis.length:.4e} cm"
        )


def normal_field(
    mesh  :  Mesh2D, psi  :npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] :
    """Magnitude of the field normal to a horizontal interface, per node.

    Args:
        mesh: the tensor product mesh. Its y direction is the interface
            normal, which is what makes this a function of a Mesh2D and not
            of an arbitrary one.
        psi: potential at every node, in whatever units the caller works in.

    Returns:
        abs(dpsi/dy) at every node, in psi's units per unit length.

    What Lombardi surface mobility reads. It is deliberately nodal: a
    horizontal channel edge, which is the edge the whole model is about, has
    its normal field on the vertical edges above and below its two endpoints
    and not on itself, so there is no edge quantity to reach for. Working it
    out at nodes and averaging onto edges afterwards is the route the mobility
    already takes from Arora, and it gives one answer per node rather than one
    per node per asking edge.

    An interior node averages the two vertical edges either side of it. A node
    on the top or bottom row has one, and takes it. Dividing by a hardcoded
    two at the boundary would halve the field on exactly the row the gate sits
    on.

    This is a magnitude, since the model scatters a carrier off the interface
    the same way whichever side the gate is on, and physics/mobility.py
    refuses a signed argument rather than taking the absolute value itself.
    **The absolute value is taken after the averaging, not before**, and the
    two are not the same thing. Where the field reverses across a node, which
    is what a potential extremum in y is, the field at that node really is
    near zero, and averaging magnitudes would report the average of the two
    large ones instead. In a channel the vertical field does not reverse and
    the two agree, so the difference only shows up somewhere the model was
    going to be asked anyway.

    Only the y direction is normal here because the Si/SiO2 interface in this
    project's devices is a horizontal line on a tensor product mesh. An
    unstructured mesh, or a non planar interface, would need the interface
    normal carried per node instead, which is the shape DEVSIM uses. Nothing
    about that is cheaper and this geometry does not need it.
    """
    if psi.size   != mesh.n_nodes  :
        raise ValueError(
            f"psi has {psi.size} values but the mesh has {mesh.n_nodes} nodes"
        )

    verttical  =  slice( mesh.n_horizontal, None )
    bel, abo  =   mesh.edge_nodes[verttical,   0 ] , mesh.edge_nodes[verttical,  1 ]
    edgeField= (psi[abo]  - psi[bel]) / mesh.h[verttical]


    vals=np.zeros(mesh.n_nodes,dtype=np.float64)
    coount=  np.zeros(mesh.n_nodes, dtype  = np.float64)
    for Node in(bel ,   abo)  :
        np.add.at(vals, Node, edgeField)
        np.add.at(coount,Node,1.0)

    return np.abs(vals /  coount)


def tensor_mesh_2d(x_axis :  Mesh1D, y_axis :  Mesh1D) -> Mesh2D :
    """The tensor product of two 1D meshes.

    Args:
        x_axis: the x direction, uniform or graded.
        y_axis: the y direction.

    Either axis may be graded, which is how a 2D mesh resolves a junction.
    """

    Nx,Ny=x_axis.n_nodes,y_axis.n_nodes
    NodeX= np.tile(x_axis.x,Ny);  nodeY =  np.repeat(y_axis.x, Nx)
    Columns  = np.arange (Nx,  dtype  = np.int64 ); Rows =  np.arange ( Ny,  dtype   =  np.int64)


    HI ,   hJ   =   np.meshgrid(Columns [:-  1  ],   Rows ,   indexing =   "xy" );  h_fom= (hJ *Nx +HI).ravel()
    hrizontal  =  np.column_stack([h_fom, h_fom + 1])
    h_lenggth = np.tile (  x_axis.h, Ny)
    hf =np.repeat(y_axis.volume, Nx -1)



    VI,foo=np.meshgrid(Columns,Rows[:-1],indexing= 'xy')
    v = (foo  *  Nx  + VI).ravel()
    ver  =   np.column_stack(  [v ,   v  + Nx])
    VLength  = np.repeat(y_axis.h, Nx)
    range= np.tile(x_axis.volume, Ny - 1)
    dat= np.outer(y_axis.volume,x_axis.volume).ravel()
    return Mesh2D(x_axis= x_axis, y_axis=y_axis, node_x =NodeX, node_y = nodeY, h= np.concatenate([h_lenggth,VLength]), dual_face = np.concatenate([hf,range]), volume=dat, edge_nodes =np.concatenate([hrizontal,ver]).astype(np.int64),)


def uniform_mesh_2d(width :float,height: float,nx : int,ny :int)->Mesh2D:
    """A uniformly spaced rectangle, [0, width] by [0, height] [cm]."""
    return tensor_mesh_2d(uniform_mesh_1d(length = width, n_nodes = nx), uniform_mesh_1d(length =  height, n_nodes= ny),)
