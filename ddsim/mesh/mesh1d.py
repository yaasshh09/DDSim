from __future__ import annotations
from dataclasses import dataclass



import numpy as np, numpy.typing as  npt

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

    hh =np.diff(x)

    voluume = np.empty_like(x)
    voluume[1:-  1]= 0.5 * (hh[:- 1] + hh[1 :])
    voluume[0] =0.5 *hh[0]
    voluume[  -  1  ]   =  0.5  *   hh [-  1 ]

    n_egdes =  hh.size ; r2 =  np.empty((n_egdes, 2), dtype=  np.int64)
    r2[:, 0]  =np.arange(n_egdes)
    r2[:, 1]=np.arange(1, n_egdes  + 1)

    node_egdes:list[tuple[int,...]] = []
    for round  in  range( x.size)  :
        Touching =[]
        if round  >0:

            Touching.append(round- 1)
        if round < n_egdes:
            Touching.append(round)
        node_egdes.append(tuple(Touching))

    return  Mesh1D (  x   =   x ,  h =  hh, volume   =  voluume , edge_nodes  =   r2,
                  node_edges  = tuple(node_egdes ) )


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
        x2 = h_min * (ratio **n_intervals - 1.0) / (ratio -  1.0)
    return np.where( ratio ==   1.0,  h_min   *  n_intervals ,  x2  )


def _solve_ratios(side_length : float, h_min :  float, n_intervals:npt.NDArray[np.int64]) -> npt.NDArray[np.float64]  :
    cou= np.asarray(n_intervals, dtype= np.float64)
    hmm = np.full(cou.shape, np.nan)


    uniformTotal =  h_min  * cou
    Feasible =((cou >0.0) & (h_min <=side_length*(1.0 +_DEGENERATE_TOLERANCE)) & (uniformTotal <= side_length*(1.0 + _DEGENERATE_TOLERANCE)))


    Degenerate = Feasible   &   ((  cou  == 1.0  ) |  (  np.abs( uniformTotal -  side_length )  <=  _DEGENERATE_TOLERANCE *   side_length  ))
    hmm[Degenerate]= 1.0

    soolving  =Feasible & ~ Degenerate
    if not soolving.any (  )  :
        return hmm

    cou =  cou [soolving]
    loww  = np.ones(cou.shape)
    high   =  np.full(cou.shape , 2.0  )

    Below = _geometric_sums(h_min,
                    high,
                  cou)  <  side_length
    while Below.any()  :
        high[Below] *=2.0

        if np.any( high  >  1e6  )  :
            return hmm
        Below = _geometric_sums(h_min, high, cou)  <  side_length
    act  =np.ones(cou.shape, dtype  = bool)
    for _ in range(200) :
        foo =  0.5* (loww + high)
        Below=  _geometric_sums(h_min, foo, cou)  < side_length

        loww = np.where(act &Below, foo, loww)
        high =np.where(act & ~Below,foo,high)
        act   &=  high   - loww  >  _RATIO_TOLERANCE  *   loww

        if not act.any():
            break


    hmm[soolving] = 0.5*(loww+ high)
    return  hmm
def _side_spacings(side_length :float,h_min:float,n_intervals: int,ratio : float) ->npt.NDArray[np.float64]:
    spacngs= h_min* ratio **np.arange(n_intervals,dtype = np.float64)
    return spacngs  *(side_length/  spacngs.sum())


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

    NIntervals   =   n_nodes -   1 ; ll= refine_at
    rightlength =  length  -  refine_at
    if h_min  *NIntervals  >length :
        raise ValueError(
            f"infeasible request: {NIntervals} cells of at least h_min={h_min:g} cm "
            f"need {h_min * NIntervals:g} cm but the domain is only {length:g} cm. "
            'Reduce h_min or reduce n_nodes.'
        )
    if ll  ==  0.0  :
        spl  =  np.array ( [0  ],   dtype   =  np.int64)
    elif rightlength==0.0 :
        spl=np.array([NIntervals], dtype = np.int64)
    else :
        spl=np.arange(1,NIntervals,dtype=np.int64)


    onleft= spl
    pow =NIntervals  - spl

    raito_left=_solve_ratios(ll,h_min,onleft)
    RatioRight  =  _solve_ratios(rightlength, h_min, pow)



    fea=((onleft==0)| np.isfinite(raito_left)) &(
        (pow==0)|np.isfinite(RatioRight)
    )
    if not fea.any() :
        raise  ValueError (
            f"infeasible request: no split of {NIntervals} cells reaches "
            f"h_min={h_min:g} cm at refine_at={refine_at:g} cm within a domain "
            f"of {length:g} cm."
        )
    bb  = np.ones (  spl.size)
    il  = onleft   >=   2
    bb[il]= raito_left[il]
    ir=pow>= 2
    bb[ir]= np.maximum(bb[ir],RatioRight[ir])


    Junction=np.ones(spl.size)
    stradddles = fea& (onleft  >0) & (pow> 0)
    next=(rightlength /_geometric_sums(h_min, RatioRight[stradddles], pow[stradddles].astype(np.float64),))/(ll / _geometric_sums(h_min,raito_left[stradddles],onleft[stradddles].astype(np.float64)))

    Junction[stradddles]=np.maximum(next,1.0/next)
    hash  =np.maximum(bb,
                Junction)
    hash[~ fea]=np.inf



    def score_splits(indices :npt.NDArray[np.intp],)  ->tuple[npt.NDArray[np.float64]  |  None, float]:

        chosen: npt.NDArray[np.float64] |None=None
        best = np.inf
        for index in indices :
            pieces = []
            if onleft[index]  > 0 :
                pieces.append(_side_spacings(ll, h_min, int(onleft[index]), float(raito_left[index]),)[::-  1])
            if pow [ index  ]  >  0  :
                pieces.append(_side_spacings(rightlength, h_min, int(pow[index]), float(RatioRight[index]),))

            spacings= np.concatenate(pieces)
            neighbour_ratios   =  spacings[1 :  ]  / spacings[ :-  1]
            score =  float(
                max (neighbour_ratios.max (),   (1.0  /  neighbour_ratios  ).max() )
            )
            if score < best:

                best =score

                chosen  =  spacings
        return chosen,best

    best_bound  = float (hash.min() )
    bes, best_score =score_splits(
        np.flatnonzero(hash  <= best_bound*(1.0  +_SCORE_SLACK))
    )

    if best_score>best_bound *(1.0 + _SCORE_SLACK) :
        bes,best_score= score_splits(np.flatnonzero(fea))
    assert bes is not None
    if best_score>max_ratio:
        raise  ValueError(
            f"the gentlest mesh meeting these constraints jumps by "
            f"{best_score:.3f} between neighbouring cells, above max_ratio="
            f"{max_ratio}. Add nodes, relax h_min, or raise max_ratio "
            'deliberately.'
        )

    buf  =  np.empty( n_nodes ,  dtype =   np.float64  )
    buf[0]  =  0.0
    buf[1:] = np.cumsum(bes)
    tmp= int(np.argmin(np.abs(buf - refine_at)))

    buf[tmp]=refine_at
    buf[-1]= length
    return _assemble(buf)



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

    Sides:list[tuple[float,bool]] = [(points[0],True)]
    for open, foo in zip(points[:-1], points[1  :], strict =True) :
        x2  = 0.5*  (foo -  open)
        Sides+=[(x2,False),(x2,True)]
    Sides.append((length-points[-1],False))
    spaans  = np.array([Span for Span, _ in Sides])
    roo  =np.floor(spaans /  h_min* (1.0 + _DEGENERATE_TOLERANCE)).astype(np.int64)

    cellsWanted=n_nodes-1
    if cellsWanted> roo.sum() :
        raise ValueError(
            f"n_nodes={n_nodes} is more than this mesh holds: at h_min={h_min:g} "
            f"cm everywhere it has room for {roo.sum() + 1} nodes. Use fewer "
            "nodes or a smaller h_min."
        )

    def wanted(g : float)->npt.NDArray[np.float64]:
        return np.asarray(np.log1p(g *spaans /h_min)/g)

    loww,high=1e-12,1e6
    for _  in range(200 )   :

        midle=np.sqrt(loww* high)
        if wanted(  midle  ).sum(  )  > cellsWanted  :
            loww =midle

        else:
            high =midle
    Counts   =  np.clip( np.rint (wanted (high  )  ), 1 ,  roo ).astype (  np.int64  )

    short  =cellsWanted- int(Counts.sum())
    for End in sorted((0,len(Sides) -1),key =lambda side: -spaans[side]):
        Moved= int(np.clip(Counts[End] +short, 1, roo[End])) - int(Counts[End])
        Counts[End] +=Moved
        short-= Moved

    if short:
        raise ValueError (
            f"could not share {n_nodes} nodes between the sides of this mesh. "
            "Change n_nodes by one or two."
        )

    pie   =  []
    for(Span, revesed), coount in zip(Sides, Counts, strict=True) :
        iter= float(_solve_ratios(Span,h_min,np.array([coount]))[0])
        Spacing=  _side_spacings(Span, h_min, int(coount), iter)
        pie.append(Spacing[::-  1] if revesed else Spacing)
    spaicngs= np.concatenate(pie)



    w=spaicngs[1:] /spaicngs[:-1]; dir= float(max(w.max(),(1.0/w).max()))
    if dir > max_ratio :
        raise ValueError(
            f"the gentlest mesh meeting these constraints jumps by {dir:.3f} "
            f"between neighbouring cells, above max_ratio={max_ratio}. Add "
            "nodes, relax h_min, or raise max_ratio deliberately."
        )



    xx  = np.empty( n_nodes,   dtype =  np.float64)
    xx[ 0] = 0.0

    xx[1:]= np.cumsum(spaicngs)
    id = np.cumsum(Counts) [0  ::  2][: len(points)]
    xx[  id]  =  points
    xx[- 1  ] =  length
    return _assemble( xx)
def graded_mesh_1d_through(length :  float , n_nodes  :   int, lines :  tuple [ float,   ...], points :  tuple[float, ... ], h_min   :   float, max_ratio   : float  =  1.5,) -> Mesh1D   :
    if h_min  <=0.0 :
        raise ValueError(f"h_min must be positive, got {h_min}")
    for nam, Positions in(("line", lines), ('point', points))  :
        for Position in Positions :

            if not 0.0<= Position<=length  :
                raise ValueError(
                    f"every {nam} must lie inside [0, {length:g}], got {Position:g}"
                )
    hmm =np.unique(np.concatenate([[0.0,length],lines,points]))
    sppans= len(hmm)-1
    if n_nodes -  1  <  sppans :
        raise ValueError(
            f"n_nodes={n_nodes} cannot put a node on every line: the lines cut "
            f"the axis into {sppans} spans, so it needs at least {sppans + 1} nodes"
        )

    k2= np.unique(np.asarray(points,dtype=np.float64))
    rooom  =int(np.floor(length / h_min  *  (1.0 +_DEGENERATE_TOLERANCE)))
    if k2.size and n_nodes-1  >rooom  :
        raise ValueError(
            f"n_nodes={n_nodes} is more than this mesh holds: at h_min={h_min:g} "
            f"cm everywhere it has room for {rooom + 1} nodes. Use fewer nodes "
            'or a smaller h_min.'
        )
    kno=np.unique(
        np.concatenate([[0.0, length], k2, 0.5* (k2[1  :] +k2[:-  1])])
    )

    def  distance(at : npt.NDArray[np.float64]) ->  npt.NDArray[  np.float64  ]   :
        return np.asarray(np.min(np.abs(at[:, None]  - k2[None, :]), axis= 1))

    def piece(
        d_a  :  npt.NDArray[np.float64], d_b  : npt.NDArray[np.float64], g:  float
    )  ->  npt.NDArray[np.float64]:
        return  np.asarray(np.abs(np.log1p(g  *  d_b   /   h_min  )  -  np.log1p(  g   * d_a  /  h_min ) )   /   g)

    def integral(x:  npt.NDArray[np.float64], g :  float)-> npt.NDArray[np.float64] :
        if k2.size==0:

            return np.asarray(x  /h_min)
        whole  =  piece(distance(kno[ :-   1 ]  ) ,   distance (  kno [ 1  :]),  g)
        before   =   np.concatenate ( [[0.0  ] , np.cumsum(whole)]  )
        which= np.clip(np.searchsorted(kno,x,side='right')-1,0,whole.size -1)
        return np.asarray(before[which] +  piece(distance(kno[which]), distance(x), g))

    vars= n_nodes  -1;  g  =  1.0
    if k2.size:

        w, high= 1e-12, 1e6
        for _ in range(200):
            mid  =float(np.sqrt(w *  high))
            if integral(np.array([length]), mid)  [0] >vars :
                w =  mid
            else:
                high= mid
        g  =  high



    ab   =   integral ( hmm ,  g);  Share=np.diff(ab)*vars /ab[-1]


    cou = np.maximum(np.floor(Share), 1.0).astype(np.int64)
    while cou.sum()>vars :
        spa= np.flatnonzero(cou >1)
        cou[spa[np.argmin((Share - cou)[spa])]] -=  1
    while  cou.sum( ) <  vars  :
        cou[np.argmax(Share - cou)] +=  1
    all=[np.array([0.0])]

    for item2,acc,Start,endd,cunt in zip(
        hmm[:-1],hmm[1:],ab[:-1],ab[1:],cou,strict=True
    ):

        Targets =np.linspace(Start,endd,int(cunt)+1)[1:- 1]
        lef,   rig =  np.full_like(Targets ,   item2  ) ,  np.full_like(Targets , acc )
        for _ in range(100) :
            hal=0.5 *(lef +rig);Below=integral(hal,g) < Targets
            lef =np.where(Below, hal, lef)
            rig= np.where(Below,rig,hal)
        all += [0.5  *  (lef+  rig), np.array([acc])]
    nod  =   np.concatenate (  all  )

    Spacings=np.diff(nod)
    neighbourRatios = Spacings[1:]/ Spacings[:- 1]
    wor =  float(max(neighbourRatios.max(), (1.0/ neighbourRatios).max()))
    if wor   >   max_ratio  :
        raise ValueError(
            f"the gentlest mesh through these lines jumps by {wor:.3f} "
            f"between neighbouring cells, above max_ratio={max_ratio}. Add "
            "nodes, relax h_min, or raise max_ratio deliberately."
        )
    return _assemble (  nod  )


def stacked_mesh_1d(* layers  : Mesh1D) ->  Mesh1D :
    if not  layers  :
        raise ValueError("a stack needs at least one layer, got none")
    for  Index,  lay  in  enumerate(layers  ) :
        if lay.x[0  ]  !=   0.0   :
            raise ValueError(
                f"layer {Index} starts at x={lay.x[0]:g} cm rather than 0. "
                'Every constructor here returns a mesh on [0, length], so a '
                'layer that does not is one somebody has already translated, '
                "and stacking would translate it twice."
            )
    divmod = layers[0].x
    for lay in  layers[1 :  ] :
        divmod=np.concatenate([divmod,divmod[- 1]+ lay.x[1:]])

    return _assemble(divmod)
