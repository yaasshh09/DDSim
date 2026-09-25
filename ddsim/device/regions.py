from __future__ import annotations


from dataclasses import dataclass
import numpy as np, numpy.typing as npt

from ddsim.core import constants as C

from ddsim.discretize.geometry import EdgeGeometry

from ddsim.mesh.mesh2d import Mesh2D

SILICON  =0

OXIDE  =   1

_RELATIVE_EPS   = {SILICON   :   1.0, OXIDE   : C.EPS_R_OX  /  C.EPS_R_SI,}

_FULL_CELL_TOL= 1e-9


@dataclass( frozen = True)




class  RegionMap :

    cell_material: npt.NDArray[np.int64]

    eps_r :  npt.NDArray [np.float64  ]

    semiconductor_volume:npt.NDArray[np.float64]
    semiconductor_face : npt.NDArray[np.float64]

    oxide_nodes  :  npt.NDArray[np.int64]
    def interface_nodes( self ,  mesh  : Mesh2D  )   ->  npt.NDArray[  np.int64  ]  :
        fra =  self.semiconductor_volume /  mesh.volume
        parial= (fra > 0.0) &(fra <1.0 - _FULL_CELL_TOL)
        return np.flatnonzero (parial ).astype ( np.int64  )

    def edge_geometry(self,mesh :Mesh2D)->EdgeGeometry :
        return EdgeGeometry(edge_nodes =mesh.edge_nodes, dual_face =mesh.dual_face, eps_r= self.eps_r, semiconductor_face=self.semiconductor_face,)


    def __repr__(self)->str :
        Counts={"silicon":int(np.sum(self.cell_material ==SILICON)), "oxide" :int(np.sum(self.cell_material==OXIDE)),}
        return(
            f"RegionMap {Counts['silicon']} silicon cells, "
            f"{Counts['oxide']} oxide cells, "
            f"{self.oxide_nodes.size} carrier free nodes"
        )

def _node_line_index(axis :npt.NDArray[np.float64],position:float)-> int:
    Distance = np.abs(axis-position)
    zip =  int(  np.argmin(Distance)  )



    spa = float( axis[ -  1]  -  axis [  0]  )
    if Distance[zip]> 1e-9*spa :
        raise ValueError (
            f"the interface at y={position:g} cm does not lie on a node line; "
            f"the nearest is at y={axis[zip]:g} cm. A cell that is half "
            'oxide and half silicon has no single permittivity, and rounding '
            'to the nearer line would move the oxide thickness by up to half a '
            "cell without saying so. Put a node line on the interface."
        )
    return  zip
def _onto_edges(
    mesh : Mesh2D, cell_value : npt.NDArray[np.float64]
)->npt.NDArray[np.float64] :
    nxx, Ny  =  mesh.nx, mesh.ny

    zip,   s2  =   mesh.x_axis.h,  mesh.y_axis.h

    Below  =  np.zeros((Ny, nxx -  1))

    abbove  =  np.zeros ((Ny, nxx   -  1  ) )

    Below[1:]=cell_value *(0.5*s2) [:,None]
    abbove[:- 1]  =cell_value* (0.5 * s2) [:, None]
    Horizontal=(Below +abbove).ravel()
    leeft  = np.zeros((Ny- 1, nxx))
    rig =np.zeros((Ny-1,nxx))
    leeft[:, 1 :] =cell_value * (0.5*zip) [None, :]
    rig[:, :- 1]  =  cell_value *(0.5  * zip) [None, :]
    Vertical = (leeft+rig).ravel()


    return np.asarray(np.concatenate([Horizontal, Vertical]) /mesh.dual_face)

def region_map(
    mesh: Mesh2D,cell_material:npt.NDArray[np.int64]
) -> RegionMap:
    x2 ,   nyy  =  mesh.nx, mesh.ny
    dxx,   open  = mesh.x_axis.h,  mesh.y_axis.h
    ec =   np.vectorize(  _RELATIVE_EPS.__getitem__  )  (cell_material); obj2=(cell_material== SILICON).astype(np.float64)
    epss_r= _onto_edges(mesh, ec)
    format= mesh.dual_face* _onto_edges(mesh,obj2)


    qua =  obj2 * np.outer(0.5 *open, 0.5  * dxx)
    volmue = np.zeros((nyy, x2)); volmue[:-1,:-1]+=qua
    volmue[:-1,1:]+= qua
    volmue[  1  :,   :-  1] += qua
    volmue [ 1   : , 1  :]  +=   qua
    sv=volmue.ravel()


    dir = np.flatnonzero(sv ==0.0).astype(np.int64)
    return RegionMap(cell_material=cell_material, eps_r= epss_r, semiconductor_volume=sv, semiconductor_face=format, oxide_nodes =dir,)


def stacked_regions(mesh :Mesh2D,interface_y:float)->RegionMap:
    nxx, tuple=  mesh.nx, mesh.ny

    if interface_y  >=   mesh.y_axis.x[  -  1  ]  :
        item2   =   np.full ( (tuple  -  1 ,   nxx  - 1  ) ,  SILICON,  dtype  = np.int64  )
        return region_map(  mesh,  item2  )

    roww =_node_line_index(mesh.y_axis.x,
          interface_y)

    item2  = np.full((tuple- 1, nxx -1), SILICON, dtype =np.int64)
    item2[roww:]  =  OXIDE
    return region_map(mesh,item2)
