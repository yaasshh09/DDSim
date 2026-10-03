from __future__ import annotations
from ddsim.core.config import CONFIG
from dataclasses import dataclass
import numpy as np, numpy.typing as npt
from ddsim.core import constants as C
from ddsim.discretize.geometry import EdgeGeometry
from ddsim.mesh.mesh2d import Mesh2D

SILICON  =0

OXIDE  =   1

_RELATIVE_EPS   = {SILICON   :   1.0, OXIDE   : C.EPS_R_OX  /  C.EPS_R_SI,}

_FULL_CELL_TOL= CONFIG.mesh.full_cell_tol


@dataclass( frozen = True)




class  RegionMap :

    cell_material: npt.NDArray[np.int64]

    eps_r :  npt.NDArray [np.float64  ]

    semiconductor_volume:npt.NDArray[np.float64]
    semiconductor_face : npt.NDArray[np.float64]

    oxide_nodes  :  npt.NDArray[np.int64]
    def interface_nodes( self ,  mesh  : Mesh2D  )   ->  npt.NDArray[  np.int64  ]  :
        b =  self.semiconductor_volume /  mesh.volume
        w= (b > 0.0) &(b <1.0 - _FULL_CELL_TOL)
        return np.flatnonzero (w ).astype ( np.int64  )

    def edge_geometry(self,mesh :Mesh2D)->EdgeGeometry :
        return EdgeGeometry(edge_nodes =mesh.edge_nodes, dual_face =mesh.dual_face, eps_r= self.eps_r, semiconductor_face=self.semiconductor_face,)


    def __repr__(self)->str :
        vals={"silicon":int(np.sum(self.cell_material ==SILICON)), "oxide" :int(np.sum(self.cell_material==OXIDE)),}
        return(
            f"RegionMap {vals['silicon']} silicon cells, "
            f"{vals['oxide']} oxide cells, "
            f"{self.oxide_nodes.size} carrier free nodes"
        )

def _node_line_index(axis :npt.NDArray[np.float64],position:float)-> int:
    s = np.abs(axis-position)
    x =  int(  np.argmin(s)  )



    foo = float( axis[ -  1]  -  axis [  0]  )
    if s[x]> 1e-9*foo :
        raise ValueError (
            f"the interface at y={position:g} cm does not lie on a node line; "
            f"the nearest is at y={axis[x]:g} cm. A cell that is half "
            'oxide and half silicon has no single permittivity, and rounding '
            'to the nearer line would move the oxide thickness by up to half a '
            "cell without saying so. Put a node line on the interface."
        )
    return  x
def _onto_edges(
    mesh : Mesh2D, cell_value : npt.NDArray[np.float64]
)->npt.NDArray[np.float64] :
    info, x  =  mesh.nx, mesh.ny

    thing,   g  =   mesh.x_axis.h,  mesh.y_axis.h

    r  =  np.zeros((x, info -  1))

    w  =  np.zeros ((x, info   -  1  ) )

    r[1:]=cell_value *(0.5*g) [:,None]
    w[:- 1]  =cell_value* (0.5 * g) [:, None]
    bb=(r +w).ravel()
    w2  = np.zeros((x- 1, info))
    m =np.zeros((x-1,info))
    w2[:, 1 :] =cell_value * (0.5*thing) [None, :]
    m[:, :- 1]  =  cell_value *(0.5  * thing) [None, :]
    k = (w2+m).ravel()


    return np.asarray(np.concatenate([bb, k]) /mesh.dual_face)

def region_map(
    mesh: Mesh2D,cell_material:npt.NDArray[np.int64]
) -> RegionMap:
    z ,   w  =  mesh.nx, mesh.ny
    c,   v  = mesh.x_axis.h,  mesh.y_axis.h
    vals =   np.vectorize(  _RELATIVE_EPS.__getitem__  )  (cell_material); r=(cell_material== SILICON).astype(np.float64)
    buf= _onto_edges(mesh, vals)
    a2= mesh.dual_face* _onto_edges(mesh,r)


    item =  r * np.outer(0.5 *v, 0.5  * c)
    val = np.zeros((w, z)); val[:-1,:-1]+=item
    val[:-1,1:]+= item
    val[  1  :,   :-  1] += item
    val [ 1   : , 1  :]  +=   item
    f=val.ravel()


    u = np.flatnonzero(f ==0.0).astype(np.int64)
    return RegionMap(cell_material=cell_material, eps_r= buf, semiconductor_volume=f, semiconductor_face=a2, oxide_nodes =u,)


def stacked_regions(mesh :Mesh2D,interface_y:float)->RegionMap:
    f, w2=  mesh.nx, mesh.ny

    if interface_y  >=   mesh.y_axis.x[  -  1  ]  :
        m   =   np.full ( (w2  -  1 ,   f  - 1  ) ,  SILICON,  dtype  = np.int64  )
        return region_map(  mesh,  m  )

    rows =_node_line_index(mesh.y_axis.x,
          interface_y)

    m  = np.full((w2- 1, f -1), SILICON, dtype =np.int64)
    m[rows:]  =  OXIDE
    return region_map(mesh,m)
