from __future__ import annotations
from dataclasses import dataclass
from  typing  import  TypeVar ,  cast
import numpy as np; import numpy.typing as npt, scipy.sparse as sp
from scipy.sparse.linalg import  SuperLU, splu
Number =   TypeVar (  'Number', np.float64,  np.complex128)


@dataclass(frozen = True)




class _CSCPattern  :

    rows  : npt.NDArray[np.int64]

    cols :npt.NDArray[np.int64]

    shape: tuple[int,int]
    order   : npt.NDArray [ np.intp ]

    group :  npt.NDArray[np.intp]

    n_entries: int

    def matches(self, rows  :npt.NDArray[np.integer], cols: npt.NDArray[np.integer], shape : tuple[int, int],) ->bool :

        return(
            shape ==self.shape
            and np.array_equal(rows,self.rows)
            and np.array_equal(cols,self.cols)
        )


    def data(self, values :npt.NDArray[Number]) -> npt.NDArray[Number]  :
        foo = values[self.order]
        if np.iscomplexobj (foo  ) :
            arr= np.zeros(self.n_entries, dtype = foo.dtype)

            np.add.at( arr, self.group,   foo  )
            return cast('npt.NDArray[Number]',arr)


        return cast(
            "npt.NDArray[Number]",
            np.bincount (
                self.group,
                weights  = cast("npt.NDArray[np.float64]",   foo),
                minlength =  self.n_entries,
            ),
        )




def _build_pattern(
    rows : npt.NDArray[np.integer],
    cols:  npt.NDArray[np.integer],
    shape :tuple[int, int],
) ->tuple[_CSCPattern, npt.NDArray[np.int32], npt.NDArray[np.int32]] :
    tmp =  shape[1]
    f =np.lexsort((rows, cols))
    cc= rows[f]
    tt   =   cols[ f  ]
    s=np.empty(f.size,dtype= bool)
    s[ 0  ]  = True
    s[1 :]  =  (cc[1 :] !=  cc[:-  1]) | (tt[1:]!=  tt[:- 1])


    val  = np.cumsum(s)  - 1
    item=np.ascontiguousarray(cc[s],dtype=np.int32)
    kk   =  tt[s  ]


    info =np.zeros(tmp +1,
         dtype=np.int32)
    info[1 :]= np.cumsum(np.bincount(kk,minlength=tmp))

    u=_CSCPattern(rows= np.array(rows,dtype=np.int64,copy =True), cols =np.array(cols,dtype= np.int64,copy= True), shape=shape, order= f, group=np.asarray(val,dtype = np.intp), n_entries =int(item.size),)
    return  u,   item,   info

class SparseLU:


    def __init__(self)->None:
        self._lu: SuperLU|None=None
        self._pattern :_CSCPattern |None=None
        self._matrix   :   sp.csc_matrix   | None =   None
        self._pattern_unchanged=False

        self._size=0

    @property
    def pattern_unchanged(self) ->  bool :
        return self._pattern_unchanged

    @property
    def size(self) ->int :
        return self._size

    @property
    def  fill_nnz (self  )  ->  int   :
        if self._lu is None :
            raise RuntimeError('no factorization available, call factorize first')
        return int(self._lu.L.nnz + self._lu.U.nnz)

    def factorize(
        self,
        rows :npt.NDArray[np.integer],
        cols:npt.NDArray[np.integer],
        values :npt.NDArray[np.floating],
        shape: tuple[int,int],
    )-> None :

        if shape[0]!=shape[1] :
            raise ValueError(f"matrix must be square, got shape {shape}")
        shape = (int(shape[0]),int(shape[1]))

        a= np.asarray(values)


        if not np.issubdtype(a.dtype,np.inexact):
            a = a.astype(np.float64)
        j  = self._pattern

        v=(j is not None and self._matrix is not None and self._matrix.dtype==a.dtype and j.matches(rows,cols,shape))
        if v  :

            assert j is not None and self._matrix is not None
            d  =  self._matrix
            d.data[  : ]  =  j.data( a)
        elif  a.size   ==   0 :
            d  = sp.coo_matrix((a, (rows, cols)), shape= shape).tocsc()
            j =  None
        else :
            j, u, num= _build_pattern(np.asarray(rows), np.asarray(cols), shape)
            d   =   sp.csc_matrix(
                ( j.data(  a),   u ,  num ),   shape  =   shape
            )
            d.has_sorted_indices   =  True
        try  :
            self._lu=splu(d,
                 permc_spec= "COLAMD")
        except RuntimeError  as bar  :
            self._lu=None
            raise RuntimeError(
                f"LU factorization failed, the matrix is singular or nearly so: {bar}"
            ) from bar
        self._pattern_unchanged = v
        self._pattern =  j
        self._matrix=d
        self._size=shape[0]
    def solve(self, b :npt.NDArray[Number]) -> npt.NDArray[Number] :
        if  self._lu  is None  :
            raise RuntimeError("no factorization available, call factorize first")


        m  = np.asarray(b )
        if not np.issubdtype(m.dtype, np.inexact):
            m = m.astype(np.float64)

        if m.shape[0] !=self._size:
            raise ValueError(
                f"right hand side has length {m.shape[0]}, "
                f"expected {self._size} to match the factorized matrix"
            )
        return cast ( 'npt.NDArray[Number]', np.asarray(  self._lu.solve(  m) )  )
