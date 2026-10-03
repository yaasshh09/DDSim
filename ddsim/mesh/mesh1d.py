from __future__ import annotations
from dataclasses import dataclass
import numpy as np, numpy.typing as npt
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.geometry import UNIFORM_1D,ScaledMesh

_RATIO_TOLERANCE=  1e-14


_DEGENERATE_TOLERANCE= 1e-12

_SCORE_SLACK= 1e-9


@dataclass( frozen   =  True  )



class Mesh1D  :


    x: npt.NDArray[np.float64]


    h :  npt.NDArray[  np.float64  ]


    volume  : npt.NDArray[  np.float64 ]
    edge_nodes : npt.NDArray[np.int64]
    node_edges: tuple[tuple[int,...],...]


    def  scaled (  self,
      scale   :  ScaleFactors)  -> ScaledMesh   :
        return ScaledMesh(h=self.h / scale.x_0, volume=self.volume / scale.x_0, geometry=UNIFORM_1D,)

    @property

    def n_nodes(self) ->int :
        return int(self.x.size)

    @property
    def n_edges(self)-> int:
        return int(self.h.size)


    @property
    def length( self) -> float  :
        return float(self.x[-1] -self.x[0])

    def __repr__(self) ->  str :
        return(
            f"Mesh1D n_nodes={self.n_nodes} length={self.length:.4e} cm "
            f"h_min={self.h.min():.4e} h_max={self.h.max():.4e}"
        )


def _assemble(x :npt.NDArray[np.float64])->Mesh1D :

    c =np.diff(x)

    cnt = np.empty_like(x)
    cnt[1:-  1]= 0.5 * (c[:- 1] + c[1 :])
    cnt[0] =0.5 *c[0]
    cnt[  -  1  ]   =  0.5  *   c [-  1 ]

    y =  c.size ; stuff =  np.empty((y, 2), dtype=  np.int64)
    stuff[:, 0]  =np.arange(y)
    stuff[:, 1]=np.arange(1, y  + 1)

    aa:list[tuple[int,...]] = []
    for k  in  range( x.size)  :
        s2 =[]
        if k  >0:

            s2.append(k- 1)
        if k < y:
            s2.append(k)
        aa.append(tuple(s2))

    return  Mesh1D (  x   =   x ,  h =  c, volume   =  cnt , edge_nodes  =   stuff,
                  node_edges  = tuple(aa ) )


def uniform_mesh_1d(length :float, n_nodes  :int)  ->  Mesh1D :
    if length<= 0.0:
        raise ValueError(  f"length must be positive, got {length}"  )
    if n_nodes  < 2 :
        raise  ValueError (  f"a mesh needs at least 2 nodes, got {n_nodes}")

    return _assemble(np.linspace(0.0, length, n_nodes))




def _geometric_sums(
    h_min  : float,
    ratio   :  npt.NDArray[ np.float64],
    n_intervals : npt.NDArray[ np.float64 ],
)  ->  npt.NDArray[np.float64 ]  :


    with np.errstate(over= 'ignore', invalid = "ignore", divide = "ignore")  :
        f = h_min * (ratio **n_intervals - 1.0) / (ratio -  1.0)
    return np.where( ratio ==   1.0,  h_min   *  n_intervals ,  f  )


def _solve_ratios(side_length : float, h_min :  float, n_intervals:npt.NDArray[np.int64]) -> npt.NDArray[np.float64]  :
    ys= np.asarray(n_intervals, dtype= np.float64)
    s = np.full(ys.shape, np.nan)


    a =  h_min  * ys
    j =((ys >0.0) & (h_min <=side_length*(1.0 +_DEGENERATE_TOLERANCE)) & (a <= side_length*(1.0 + _DEGENERATE_TOLERANCE)))


    c = j   &   ((  ys  == 1.0  ) |  (  np.abs( a -  side_length )  <=  _DEGENERATE_TOLERANCE *   side_length  ))
    s[c]= 1.0

    v  =j & ~ c
    if not v.any (  )  :
        return s

    ys =  ys [v]
    out2  = np.ones(ys.shape)
    m   =  np.full(ys.shape , 2.0  )

    d = _geometric_sums(h_min,
                    m,
                  ys)  <  side_length
    while d.any()  :
        m[d] *=2.0

        if np.any( m  >  1e6  )  :
            return s
        d = _geometric_sums(h_min, m, ys)  <  side_length
    x  =np.ones(ys.shape, dtype  = bool)
    for _ in range(200) :
        nxt =  0.5* (out2 + m)
        d=  _geometric_sums(h_min, nxt, ys)  < side_length

        out2 = np.where(x &d, nxt, out2)
        m =np.where(x & ~d,nxt,m)
        x   &=  m   - out2  >  _RATIO_TOLERANCE  *   out2

        if not x.any():
            break


    s[v] = 0.5*(out2+ m)
    return  s
def _side_spacings(side_length :float,h_min:float,n_intervals: int,ratio : float) ->npt.NDArray[np.float64]:
    y2= h_min* ratio **np.arange(n_intervals,dtype = np.float64)
    return y2  *(side_length/  y2.sum())


def graded_mesh_1d(
    length : float,
    n_nodes  :int,
    refine_at  :  float,
    h_min: float,
    max_ratio :float = 1.5,
)  -> Mesh1D :
    if length <=  0.0 :
        raise ValueError(f"length must be positive, got {length}")

    if n_nodes< 2:
        raise ValueError(f"a mesh needs at least 2 nodes, got {n_nodes}")
    if h_min<=0.0 :
        raise ValueError(f"h_min must be positive, got {h_min}")
    if not 0.0 <= refine_at<=length:
        raise ValueError(
            f"refine_at must lie in [0, {length}], got {refine_at}"
        )

    aa   =   n_nodes -   1 ; row= refine_at
    ss =  length  -  refine_at
    if h_min  *aa  >length :
        raise ValueError(
            f"infeasible request: {aa} cells of at least h_min={h_min:g} cm "
            f"need {h_min * aa:g} cm but the domain is only {length:g} cm. "
            'Reduce h_min or reduce n_nodes.'
        )
    if row  ==  0.0  :
        d  =  np.array ( [0  ],   dtype   =  np.int64)
    elif ss==0.0 :
        d=np.array([aa], dtype = np.int64)
    else :
        d=np.arange(1,aa,dtype=np.int64)


    foo= d
    arr =aa  - d

    h=_solve_ratios(row,h_min,foo)
    c2  =  _solve_ratios(ss, h_min, arr)



    x=((foo==0)| np.isfinite(h)) &(
        (arr==0)|np.isfinite(c2)
    )
    if not x.any() :
        raise  ValueError (
            f"infeasible request: no split of {aa} cells reaches "
            f"h_min={h_min:g} cm at refine_at={refine_at:g} cm within a domain "
            f"of {length:g} cm."
        )
    b  = np.ones (  d.size)
    r  = foo   >=   2
    b[r]= h[r]
    prev=arr>= 2
    b[prev]= np.maximum(b[prev],c2[prev])


    c=np.ones(d.size)
    m = x& (foo  >0) & (arr> 0)
    val=(ss /_geometric_sums(h_min, c2[m], arr[m].astype(np.float64),))/(row / _geometric_sums(h_min,h[m],foo[m].astype(np.float64)))

    c[m]=np.maximum(val,1.0/val)
    jj  =np.maximum(b,
                c)
    jj[~ x]=np.inf



    def score_splits(indices :npt.NDArray[np.intp],)  ->tuple[npt.NDArray[np.float64]  |  None, float]:

        v: npt.NDArray[np.float64] |None=None
        flag = np.inf
        for num in indices :
            buf = []
            if foo[num]  > 0 :
                buf.append(_side_spacings(row, h_min, int(foo[num]), float(h[num]),)[::-  1])
            if arr [ num  ]  >  0  :
                buf.append(_side_spacings(ss, h_min, int(arr[num]), float(c2[num]),))

            e= np.concatenate(buf)
            b2   =  e[1 :  ]  / e[ :-  1]
            tt =  float(
                max (b2.max (),   (1.0  /  b2  ).max() )
            )
            if tt < flag:

                flag =tt

                v  =  e
        return v,flag

    k  = float (jj.min() )
    xs, cur =score_splits(
        np.flatnonzero(jj  <= k*(1.0  +_SCORE_SLACK))
    )

    if cur>k *(1.0 + _SCORE_SLACK) :
        xs,cur= score_splits(np.flatnonzero(x))
    assert xs is not None
    if cur>max_ratio:
        raise  ValueError(
            f"the gentlest mesh meeting these constraints jumps by "
            f"{cur:.3f} between neighbouring cells, above max_ratio="
            f"{max_ratio}. Add nodes, relax h_min, or raise max_ratio "
            'deliberately.'
        )

    s  =  np.empty( n_nodes ,  dtype =   np.float64  )
    s[0]  =  0.0
    s[1:] = np.cumsum(xs)
    u= int(np.argmin(np.abs(s - refine_at)))

    s[u]=refine_at
    s[-1]= length
    return _assemble(s)



def graded_mesh_1d_at(length :   float , n_nodes  : int , points  :  tuple[ float,  ...], h_min  :   float, max_ratio :  float   = 1.5,)  ->   Mesh1D   :
    if len(points)==1:

        if  not 0.0   < points[  0 ]  <   length  :
            raise ValueError(
                f"the point must lie inside (0, {length:g}), got {points[0]:g}"
            )

        return  graded_mesh_1d( length ,   n_nodes,  points[0 ],  h_min,  max_ratio)
    if not points :
        raise  ValueError ( "a graded mesh needs at least one point to grade towards")
    if any(np.diff(points)  <= 0.0) :
        raise ValueError(f"the points must be increasing, got {points}")
    if not(0.0 <points[0] and points[-  1] <  length):
        raise ValueError(
            f"every point must lie inside (0, {length:g}), got {points}"
        )

    cur:list[tuple[float,bool]] = [(points[0],True)]
    for u, num in zip(points[:-1], points[1  :], strict =True) :
        d  = 0.5*  (num -  u)
        cur+=[(d,False),(d,True)]
    cur.append((length-points[-1],False))
    j  = np.array([m2 for m2, _ in cur])
    xs  =np.floor(j /  h_min* (1.0 + _DEGENERATE_TOLERANCE)).astype(np.int64)

    z=n_nodes-1
    if z> xs.sum() :
        raise ValueError(
            f"n_nodes={n_nodes} is more than this mesh holds: at h_min={h_min:g} "
            f"cm everywhere it has room for {xs.sum() + 1} nodes. Use fewer "
            "nodes or a smaller h_min."
        )

    def wanted(g : float)->npt.NDArray[np.float64]:
        return np.asarray(np.log1p(g *j /h_min)/g)

    obj,i=1e-12,1e6
    for _  in range(200 )   :

        out2=np.sqrt(obj* i)
        if wanted(  out2  ).sum(  )  > z  :
            obj =out2

        else:
            i =out2
    r   =  np.clip( np.rint (wanted (i  )  ), 1 ,  xs ).astype (  np.int64  )

    rows  =z- int(r.sum())
    for h in sorted((0,len(cur) -1),key =lambda side: -j[side]):
        k= int(np.clip(r[h] +rows, 1, xs[h])) - int(r[h])
        r[h] +=k
        rows-= k

    if rows:
        raise ValueError (
            f"could not share {n_nodes} nodes between the sides of this mesh. "
            "Change n_nodes by one or two."
        )

    val   =  []
    for(m2, y), t in zip(cur, r, strict=True) :
        vv= float(_solve_ratios(m2,h_min,np.array([t]))[0])
        m=  _side_spacings(m2, h_min, int(t), vv)
        val.append(m[::-  1] if y else m)
    v2= np.concatenate(val)



    v=v2[1:] /v2[:-1]; hh= float(max(v.max(),(1.0/v).max()))
    if hh > max_ratio :
        raise ValueError(
            f"the gentlest mesh meeting these constraints jumps by {hh:.3f} "
            f"between neighbouring cells, above max_ratio={max_ratio}. Add "
            "nodes, relax h_min, or raise max_ratio deliberately."
        )



    nxt  = np.empty( n_nodes,   dtype =  np.float64)
    nxt[ 0] = 0.0

    nxt[1:]= np.cumsum(v2)
    x = np.cumsum(r) [0  ::  2][: len(points)]
    nxt[  x]  =  points
    nxt[- 1  ] =  length
    return _assemble( nxt)
def graded_mesh_1d_through(length :  float , n_nodes  :   int, lines :  tuple [ float,   ...], points :  tuple[float, ... ], h_min   :   float, max_ratio   : float  =  1.5,) -> Mesh1D   :
    if h_min  <=0.0 :
        raise ValueError(f"h_min must be positive, got {h_min}")
    for nxt, m in(("line", lines), ('point', points))  :
        for c2 in m :

            if not 0.0<= c2<=length  :
                raise ValueError(
                    f"every {nxt} must lie inside [0, {length:g}], got {c2:g}"
                )
    h =np.unique(np.concatenate([[0.0,length],lines,points]))
    r2= len(h)-1
    if n_nodes -  1  <  r2 :
        raise ValueError(
            f"n_nodes={n_nodes} cannot put a node on every line: the lines cut "
            f"the axis into {r2} spans, so it needs at least {r2 + 1} nodes"
        )

    tt= np.unique(np.asarray(points,dtype=np.float64))
    stuff  =int(np.floor(length / h_min  *  (1.0 +_DEGENERATE_TOLERANCE)))
    if tt.size and n_nodes-1  >stuff  :
        raise ValueError(
            f"n_nodes={n_nodes} is more than this mesh holds: at h_min={h_min:g} "
            f"cm everywhere it has room for {stuff + 1} nodes. Use fewer nodes "
            'or a smaller h_min.'
        )
    d=np.unique(
        np.concatenate([[0.0, length], tt, 0.5* (tt[1  :] +tt[:-  1])])
    )

    def  distance(at : npt.NDArray[np.float64]) ->  npt.NDArray[  np.float64  ]   :
        return np.asarray(np.min(np.abs(at[:, None]  - tt[None, :]), axis= 1))

    def piece(
        d_a  :  npt.NDArray[np.float64], d_b  : npt.NDArray[np.float64], g:  float
    )  ->  npt.NDArray[np.float64]:
        return  np.asarray(np.abs(np.log1p(g  *  d_b   /   h_min  )  -  np.log1p(  g   * d_a  /  h_min ) )   /   g)

    def integral(x:  npt.NDArray[np.float64], g :  float)-> npt.NDArray[np.float64] :
        if tt.size==0:

            return np.asarray(x  /h_min)
        yy  =  piece(distance(d[ :-   1 ]  ) ,   distance (  d [ 1  :]),  g)
        item   =   np.concatenate ( [[0.0  ] , np.cumsum(yy)]  )
        y2= np.clip(np.searchsorted(d,x,side='right')-1,0,yy.size -1)
        return np.asarray(item[y2] +  piece(distance(d[y2]), distance(x), g))

    idx= n_nodes  -1;  g  =  1.0
    if tt.size:

        k, w= 1e-12, 1e6
        for _ in range(200):
            z  =float(np.sqrt(k *  w))
            if integral(np.array([length]), z)  [0] >idx :
                k =  z
            else:
                w= z
        g  =  w



    x2   =   integral ( h ,  g);  rr=np.diff(x2)*idx /x2[-1]


    c = np.maximum(np.floor(rr), 1.0).astype(np.int64)
    while c.sum()>idx :
        i= np.flatnonzero(c >1)
        c[i[np.argmin((rr - c)[i])]] -=  1
    while  c.sum( ) <  idx  :
        c[np.argmax(rr - c)] +=  1
    t=[np.array([0.0])]

    for res2,s2,ii,u,w2 in zip(
        h[:-1],h[1:],x2[:-1],x2[1:],c,strict=True
    ):

        bb =np.linspace(ii,u,int(w2)+1)[1:- 1]
        r,   ss =  np.full_like(bb ,   res2  ) ,  np.full_like(bb , s2 )
        for _ in range(100) :
            s=0.5 *(r +ss);v=integral(s,g) < bb
            r =np.where(v, s, r)
            ss= np.where(v,ss,s)
        t += [0.5  *  (r+  ss), np.array([s2])]
    flag  =   np.concatenate (  t  )

    y=np.diff(flag)
    b = y[1:]/ y[:- 1]
    j =  float(max(b.max(), (1.0/ b).max()))
    if j   >   max_ratio  :
        raise ValueError(
            f"the gentlest mesh through these lines jumps by {j:.3f} "
            f"between neighbouring cells, above max_ratio={max_ratio}. Add "
            "nodes, relax h_min, or raise max_ratio deliberately."
        )
    return _assemble (  flag  )


def stacked_mesh_1d(* layers  : Mesh1D) ->  Mesh1D :
    if not  layers  :
        raise ValueError("a stack needs at least one layer, got none")
    for  h,  e  in  enumerate(layers  ) :
        if e.x[0  ]  !=   0.0   :
            raise ValueError(
                f"layer {h} starts at x={e.x[0]:g} cm rather than 0. "
                'Every constructor here returns a mesh on [0, length], so a '
                'layer that does not is one somebody has already translated, '
                "and stacking would translate it twice."
            )
    d = layers[0].x
    for e in  layers[1 :  ] :
        d=np.concatenate([d,d[- 1]+ e.x[1:]])

    return _assemble(d)
