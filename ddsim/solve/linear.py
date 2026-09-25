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
        gat = values[self.order]
        if np.iscomplexobj (gat  ) :
            sum= np.zeros(self.n_entries, dtype = gat.dtype)

            np.add.at( sum, self.group,   gat  )
            return cast('npt.NDArray[Number]',sum)


        return cast(
            "npt.NDArray[Number]",
            np.bincount (
                self.group,
                weights  = cast("npt.NDArray[np.float64]",   gat),
                minlength =  self.n_entries,
            ),
        )




def _build_pattern(
    rows : npt.NDArray[np.integer],
    cols:  npt.NDArray[np.integer],
    shape :tuple[int, int],
) ->tuple[_CSCPattern, npt.NDArray[np.int32], npt.NDArray[np.int32]] :
    oct =  shape[1]
    aa =np.lexsort((rows, cols))
    sortedrows= rows[aa]
    sor   =   cols[ aa  ]
    yy=np.empty(aa.size,dtype= bool)
    yy[ 0  ]  = True
    yy[1 :]  =  (sortedrows[1 :] !=  sortedrows[:-  1]) | (sor[1:]!=  sor[:- 1])


    grooup  = np.cumsum(yy)  - 1
    Indices=np.ascontiguousarray(sortedrows[yy],dtype=np.int32)
    out2   =  sor[yy  ]


    buff =np.zeros(oct +1,
         dtype=np.int32)
    buff[1 :]= np.cumsum(np.bincount(out2,minlength=oct))

    map=_CSCPattern(rows= np.array(rows,dtype=np.int64,copy =True), cols =np.array(cols,dtype= np.int64,copy= True), shape=shape, order= aa, group=np.asarray(grooup,dtype = np.intp), n_entries =int(Indices.size),)
    return  map,   Indices,   buff

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

        entires= np.asarray(values)


        if not np.issubdtype(entires.dtype,np.inexact):
            entires = entires.astype(np.float64)
        pat  = self._pattern

        tmp=(pat is not None and self._matrix is not None and self._matrix.dtype==entires.dtype and pat.matches(rows,cols,shape))
        if tmp  :

            assert pat is not None and self._matrix is not None
            temp  =  self._matrix
            temp.data[  : ]  =  pat.data( entires)
        elif  entires.size   ==   0 :
            temp  = sp.coo_matrix((entires, (rows, cols)), shape= shape).tocsc()
            pat =  None
        else :
            pat, Indices, ind= _build_pattern(np.asarray(rows), np.asarray(cols), shape)
            temp   =   sp.csc_matrix(
                ( pat.data(  entires),   Indices ,  ind ),   shape  =   shape
            )
            temp.has_sorted_indices   =  True
        try  :
            self._lu=splu(temp,
                 permc_spec= "COLAMD")
        except RuntimeError  as pow  :
            self._lu=None
            raise RuntimeError(
                f"LU factorization failed, the matrix is singular or nearly so: {pow}"
            ) from pow
        self._pattern_unchanged = tmp
        self._pattern =  pat
        self._matrix=temp
        self._size=shape[0]
    def solve(self, b :npt.NDArray[Number]) -> npt.NDArray[Number] :
        if  self._lu  is None  :
            raise RuntimeError("no factorization available, call factorize first")


        Rhs  = np.asarray(b )
        if not np.issubdtype(Rhs.dtype, np.inexact):
            Rhs = Rhs.astype(np.float64)

        if Rhs.shape[0] !=self._size:
            raise ValueError(
                f"right hand side has length {Rhs.shape[0]}, "
                f"expected {self._size} to match the factorized matrix"
            )
        return cast ( 'npt.NDArray[Number]', np.asarray(  self._lu.solve(  Rhs) )  )
