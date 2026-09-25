from __future__ import annotations

from dataclasses import dataclass

import numpy as np, numpy.typing as  npt

from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.geometry import  EdgeGeometry ,   ScaledMesh
from ddsim.mesh.mesh1d import Mesh1D,  uniform_mesh_1d


@dataclass(frozen =  True)
class Mesh2D:

    x_axis : Mesh1D

    y_axis : Mesh1D

    node_x:npt.NDArray[np.float64]
    node_y: npt.NDArray[np.float64]
    h :npt.NDArray[np.float64]

    dual_face  : npt.NDArray[np.float64]
    volume:  npt.NDArray[np.float64]


    edge_nodes:npt.NDArray[np.int64]
    @property
    def nx(self)->int:
        return self.x_axis.n_nodes


    @property
    def ny(self)-> int  :
        return  self.y_axis.n_nodes
    @property
    def n_nodes(self) ->int :
        return int ( self.node_x.size)

    @property
    def n_edges(  self  )  ->   int  :
        return int (  self.h.size  )
    @property
    def n_horizontal(self)->int:
        return (  self.nx   -  1) *  self.ny
    def node_at(self, i: int, j:  int)->  int  :
        return j *self.nx + i

    def edge_geometry(
        self,
        eps_r:npt.NDArray[np.float64] |float=1.0,
        semiconductor_face:npt.NDArray[np.float64] |None =None,
    )-> EdgeGeometry :
        return EdgeGeometry(
            edge_nodes = self.edge_nodes,
            dual_face= self.dual_face,
            eps_r =eps_r,
            semiconductor_face = semiconductor_face,
        )


    def scaled(self , scale :   ScaleFactors, eps_r  :  npt.NDArray[ np.float64] | float  =  1.0, semiconductor_face  : npt.NDArray[ np.float64 ] |  None  =   None,) ->  ScaledMesh  :

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
    return tensor_mesh_2d(uniform_mesh_1d(length = width, n_nodes = nx), uniform_mesh_1d(length =  height, n_nodes= ny),)
