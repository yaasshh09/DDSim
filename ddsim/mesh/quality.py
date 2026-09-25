from __future__ import annotations

import numpy as np, numpy.typing as npt

class MeshQualityError(ValueError):
    ...

def _check_shape(triangles: npt.NDArray[np.int64]) ->None :
    if triangles.ndim !=2 or triangles.shape[1]!=3 :
        raise ValueError(
            f"triangles must have shape (n_triangles, 3), got {triangles.shape}"
        )


def triangle_angles (points : npt.NDArray[ np.float64 ],   triangles   : npt.NDArray [  np.int64  ])  ->  npt.NDArray [ np.float64  ]  :
    _check_shape(triangles)

    temp  = points[triangles]
    tuple=np.empty(triangles.shape,dtype =np.float64)
    for vretex in range(3) :
        her   = temp[ :,   vretex ]
        fir  = temp[:, (vretex  + 1) % 3] - her
        out2=temp[:,(vretex +2)% 3]-her

        crss= fir[:, 0] *  out2[:, 1]  -  fir[:, 1] *out2[:, 0]
        dott =  fir[ :,  0  ]  *  out2[:,  0 ]   + fir[:, 1  ]   * out2 [  :, 1  ];tuple[:, vretex] = np.arctan2(np.abs(crss), dott)
    return tuple
def triangle_areas(points:npt.NDArray[np.float64],triangles :npt.NDArray[np.int64])-> npt.NDArray[np.float64] :
    _check_shape(triangles)
    cor   =  points[triangles]
    First =   cor [ :,   1  ]   -   cor[: ,   0 ]
    sec = cor[:,2] - cor[:,0]
    cro = First[:,0]*sec[:,1]- First[:,1]* sec[:,0]
    return np.asarray(0.5*cro)


def obtuse_triangles(points :npt.NDArray[np.float64], triangles: npt.NDArray[np.int64], tolerance_deg :float =0.0,)->npt.NDArray[np.int64] :

    Limit =np.pi/2+ np.radians(tolerance_deg) ; return np.flatnonzero(triangle_angles(points, triangles).max(axis = 1)>  Limit)


def cotangent_edge_weights(points : npt.NDArray[np.float64], triangles : npt.NDArray[np.int64])  ->tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]] :

    zz=triangle_angles(points, triangles)

    faccing =  [ (  0,  1 ,  2 ) , ( 1 , 2 ,   0 ) , ( 2 ,   0 , 1  ) ]

    Pairs =  [];con=[]
    for Vertex, tmp ,  seecond in faccing   :
        Pairs.append(np.sort(triangles[:, [tmp, seecond]], axis  =1))

        with np.errstate(divide="ignore",invalid= "ignore"):
            con.append(0.5/ np.tan(zz[:,Vertex]))

    all_pars =   np.concatenate(Pairs)
    AllWeights =   np.concatenate(  con  )
    Edges, Inverse  = np.unique(all_pars,   axis   =   0,  return_inverse   =  True  )

    Weights = np.zeros(Edges.shape[0], dtype=  np.float64)
    np.add.at(Weights ,  Inverse.ravel(), AllWeights)

    return Edges.astype(np.int64 ) ,  Weights

def check_triangulation(
    points:  npt.NDArray[np.float64],
    triangles  :npt.NDArray[np.int64],
    tolerance_deg:  float=0.0,
    min_area  :float= 0.0,
)->  None :
    _check_shape(triangles)


    araes =  np.abs(triangle_areas(points, triangles))
    buf  =  np.flatnonzero (araes  <=  min_area  )
    if buf.size  :

        raise MeshQualityError(
            f"{buf.size} degenerate triangle(s), the first being "
            f"triangle {buf[0]} with area {araes[buf[0]]:.3e} "
            "cm^2. Three collinear points have no dual cell to integrate over."
        )


    ang  =  np.degrees( triangle_angles(points, triangles) )
    worstpertriangle  =   ang.max ( axis  =  1 )
    Offenders =obtuse_triangles(points, triangles, tolerance_deg)

    if Offenders.size :
        Worst  = Offenders[np.argmax(worstpertriangle[Offenders])]

        raise MeshQualityError(
            f"{Offenders.size} obtuse triangle(s), the worst being triangle "
            f"{Worst} at {worstpertriangle[Worst]:.3f} deg. An angle past 90 "
            'gives the edge facing it a negative finite volume conductance, '
            "which breaks the M-matrix property and lets carrier densities go "
            'negative for no physical reason. Refine or re-triangulate; this '
            "is not something the solver can be tuned around."
        )
