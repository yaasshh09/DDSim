from  __future__ import  annotations


from  dataclasses import dataclass

import numpy  as  np, numpy.typing as npt

EdgeQuantity   =  float  | npt.NDArray [ np.float64  ]


@dataclass(frozen= True)


class  EdgeGeometry  :
    edge_nodes  :  npt.NDArray[ np.int64 ] |  None   =  None
    dual_face:EdgeQuantity= 1.0
    eps_r: EdgeQuantity=1.0
    semiconductor_face:EdgeQuantity |None=None
    def __post_init__(self)-> None :
        if self.edge_nodes is None  :
            return


        if self.edge_nodes.ndim!= 2 or self.edge_nodes.shape[1]!=2:
            raise ValueError(
                "edge_nodes must have shape (n_edges, 2), got "
                f"{self.edge_nodes.shape}. Transposing it is the usual slip."
            )

        if np.any(self.edge_nodes[:, 0]==  self.edge_nodes[:, 1])  :
            raise ValueError(
                "an edge joins a node to itself, which has zero length and "
                'would divide by zero in every flux that crosses it'
            )

    def  ends ( self, n_edges :  int  )   ->  tuple[
        npt.NDArray[ np.int64] , npt.NDArray[np.int64]
    ]  :
        if self.edge_nodes is  None   :
            leeft =  np.arange( n_edges, dtype  =   np.int64);  return leeft,leeft+1
        if self.edge_nodes.shape[0]!= n_edges :
            raise ValueError(
                f"the edge list has {self.edge_nodes.shape[0]} edges but the "
                f"mesh has {n_edges}"
            )

        return self.edge_nodes[:, 0], self.edge_nodes[:, 1]
    def edge_count(self,n_nodes :int)->int:

        if self.edge_nodes is None :

            return n_nodes  - 1
        return  int (self.edge_nodes.shape[  0  ] )

    def ends_of(self, n_nodes :  int) ->  tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:

        return self.ends(self.edge_count(n_nodes))
    @property
    def  weight(  self )   -> EdgeQuantity   :
        return self.eps_r  * self.dual_face

    @property
    def  carrier_face(  self  )  -> EdgeQuantity  :

        if self.semiconductor_face is None  :
            return self.dual_face
        return self.semiconductor_face

UNIFORM_1D = EdgeGeometry()

@dataclass(frozen=True)

class ScaledMesh :

    h: npt.NDArray[np.float64]
    volume: npt.NDArray[np.float64]


    geometry:EdgeGeometry


    @property
    def n_nodes(self)-> int:
        return int(self.volume.size)
    @property
    def n_edges(self) ->int:

        return int(self.h.size)
