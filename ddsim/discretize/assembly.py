from __future__ import annotations
from dataclasses  import dataclass
import  numpy  as  np, numpy.typing as npt
@dataclass(frozen  = True)

class SparseAssembly :

    residual : npt.NDArray[np.float64]



    rows :  npt.NDArray[ np.int64]
    cols:npt.NDArray[np.int64]


    values: npt.NDArray[np.float64]

    shape:tuple[int,int]
