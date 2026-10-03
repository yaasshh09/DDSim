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

    rr  = points[triangles]
    g=np.empty(triangles.shape,dtype =np.float64)
    for row in range(3) :
        dd   = rr[ :,   row ]
        cnt  = rr[:, (row  + 1) % 3] - dd
        m=rr[:,(row +2)% 3]-dd

        k= cnt[:, 0] *  m[:, 1]  -  cnt[:, 1] *m[:, 0]
        h =  cnt[ :,  0  ]  *  m[:,  0 ]   + cnt[:, 1  ]   * m [  :, 1  ];g[:, row] = np.arctan2(np.abs(k), h)
    return g
def triangle_areas(points:npt.NDArray[np.float64],triangles :npt.NDArray[np.int64])-> npt.NDArray[np.float64] :
    _check_shape(triangles)
    val2   =  points[triangles]
    s2 =   val2 [ :,   1  ]   -   val2[: ,   0 ]
    u = val2[:,2] - val2[:,0]
    w = s2[:,0]*u[:,1]- s2[:,1]* u[:,0]
    return np.asarray(0.5*w)


def obtuse_triangles(points :npt.NDArray[np.float64], triangles: npt.NDArray[np.int64], tolerance_deg :float =0.0,)->npt.NDArray[np.int64] :

    ss =np.pi/2+ np.radians(tolerance_deg) ; return np.flatnonzero(triangle_angles(points, triangles).max(axis = 1)>  ss)


def cotangent_edge_weights(points : npt.NDArray[np.float64], triangles : npt.NDArray[np.int64])  ->tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]] :

    b2=triangle_angles(points, triangles)

    hh =  [ (  0,  1 ,  2 ) , ( 1 , 2 ,   0 ) , ( 2 ,   0 , 1  ) ]

    buf =  [];tmp2=[]
    for i, z ,  m in hh   :
        buf.append(np.sort(triangles[:, [z, m]], axis  =1))

        with np.errstate(divide="ignore",invalid= "ignore"):
            tmp2.append(0.5/ np.tan(b2[:,i]))

    h =   np.concatenate(buf)
    e =   np.concatenate(  tmp2  )
    d, x  = np.unique(h,   axis   =   0,  return_inverse   =  True  )

    w = np.zeros(d.shape[0], dtype=  np.float64)
    np.add.at(w ,  x.ravel(), e)

    return d.astype(np.int64 ) ,  w

def check_triangulation(
    points:  npt.NDArray[np.float64],
    triangles  :npt.NDArray[np.int64],
    tolerance_deg:  float=0.0,
    min_area  :float= 0.0,
)->  None :
    _check_shape(triangles)


    u =  np.abs(triangle_areas(points, triangles))
    ret  =  np.flatnonzero (u  <=  min_area  )
    if ret.size  :

        raise MeshQualityError(
            f"{ret.size} degenerate triangle(s), the first being "
            f"triangle {ret[0]} with area {u[ret[0]]:.3e} "
            "cm^2. Three collinear points have no dual cell to integrate over."
        )


    y  =  np.degrees( triangle_angles(points, triangles) )
    h  =   y.max ( axis  =  1 )
    w2 =obtuse_triangles(points, triangles, tolerance_deg)

    if w2.size :
        bar  = w2[np.argmax(h[w2])]

        raise MeshQualityError(
            f"{w2.size} obtuse triangle(s), the worst being triangle "
            f"{bar} at {h[bar]:.3f} deg. An angle past 90 "
            'gives the edge facing it a negative finite volume conductance, '
            "which breaks the M-matrix property and lets carrier densities go "
            'negative for no physical reason. Refine or re-triangulate; this '
            "is not something the solver can be tuned around."
        )
