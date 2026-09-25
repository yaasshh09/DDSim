from __future__ import  annotations

import ast

import pathlib

import numpy as np
import pytest, scipy.sparse as sp


from ddsim.solve.linear import SparseLU



def tridiagonal(n :int,off: float= -1.0,diag:float=2.0)->sp.coo_matrix:
    row, cools, Values = [], [], []
    for temp in range(n) :

        row.append(  temp  )
        cools.append(temp)
        Values.append(diag)
        if temp > 0:

            row.append(temp) ; cools.append(temp  - 1)
            Values.append(off)
        if temp  < n -  1  :
            row.append(temp)
            cools.append(temp  + 1)
            Values.append(off)
    return sp.coo_matrix((np.array(Values), (np.array(row), np.array(cools))), shape = (n, n))

Triplets =  tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]

def as_arrays(matrix:sp.coo_matrix)->Triplets:

    return matrix.row, matrix.col, matrix.data,  matrix.shape



def test_solves_a_small_system(  )  -> None :
    yy  = sp.coo_matrix(np.array( [ [4.0, 1.0] ,  [1.0, 3.0  ]])  )
    next= SparseLU()
    next.factorize (*  as_arrays (  yy )  )
    B  =  np.array([1.0 ,   2.0 ] )
    np.testing.assert_allclose(next.solve(B), np.linalg.solve(yy.toarray(), B))

def test_solves_a_tridiagonal_system_against_a_dense_reference ( )   ->   None   :
    matix  = tridiagonal(50 ) ; slover=  SparseLU()
    slover.factorize(  *   as_arrays( matix  )  )

    bb =np.linspace(1.0,
              2.0,
            50)
    Expected = np.linalg.solve(matix.toarray(), bb)
    np.testing.assert_allclose(slover.solve(bb), Expected, rtol= 1e-12)

def test_solves_a_nonsymmetric_system()->None:
    mat =tridiagonal(30,off=-1.0)
    den =mat.toarray()
    den[ 0, -   1]   =   5.0;den[- 1,
              0]= -3.0
    Solver  = SparseLU()

    Solver.factorize(*as_arrays(sp.coo_matrix(den)))
    bar=np.ones(30)
    np.testing.assert_allclose(Solver.solve(bar),np.linalg.solve(den,bar),rtol= 1e-12)
def test_solves_multiple_right_hand_sides_with_one_factorization()->None:
    mat  =   tridiagonal(20  )
    sol= SparseLU(); sol.factorize(* as_arrays(mat))
    res =mat.toarray()
    for myvar in(np.ones(20), np.arange(20.0), np.linspace(- 1.0, 1.0, 20)):

        expcted =np.linalg.solve(res, myvar)


        np.testing.assert_allclose(  sol.solve( myvar  ) ,   expcted ,   rtol  =  1e-12  )


def  test_duplicate_coo_entries_are_summed(  )  -> None  :


    row = np.array([0, 0, 1])
    temp2   = np.array( [0 ,  0 , 1  ])
    out2=np.array([1.0,
                     3.0,
                     2.0])
    sol=SparseLU(); sol.factorize(row,temp2,out2,(2,2))

    np.testing.assert_allclose(sol.solve(np.array([8.0, 2.0])), [2.0, 1.0])


def test_first_factorization_reports_a_new_pattern()->  None :

    data2 = SparseLU()
    data2.factorize(* as_arrays(tridiagonal(10)))
    assert  data2.pattern_unchanged is False



def test_same_pattern_with_new_values_is_recognised()->None :
    sovler =SparseLU()
    sovler.factorize(*  as_arrays(tridiagonal(10 ) )  )

    sovler.factorize(* as_arrays(tridiagonal(10, diag  =  3.0)))
    assert sovler.pattern_unchanged is True



def test_a_changed_pattern_is_recognised() ->None:
    t2=  SparseLU()

    t2.factorize(  * as_arrays(  tridiagonal (  10 )  ) ); t2.factorize(*  as_arrays(tridiagonal(12)))
    assert t2.pattern_unchanged  is False



def test_a_changed_pattern_at_the_same_size_is_recognised()  ->None:

    Solver=  SparseLU()
    Solver.factorize(*  as_arrays(tridiagonal(10))); aa  =   tridiagonal(  10  ).toarray(  )
    aa[0,9]=1.0

    Solver.factorize(*as_arrays(sp.coo_matrix(aa)))
    assert Solver.pattern_unchanged is False
def test_refactorizing_with_new_values_gives_the_new_solution ( )  ->  None  :


    sorted=SparseLU()
    sorted.factorize(*as_arrays(tridiagonal(15, diag= 2.0)))
    frist= sorted.solve(np.ones(15))



    vars = tridiagonal(15, diag = 10.0)
    sorted.factorize(*as_arrays(vars))
    Second =sorted.solve(np.ones(15))

    object = np.linalg.solve(vars.toarray(), np.ones(15))

    np.testing.assert_allclose(Second,   object,   rtol  =  1e-12)

    assert not np.allclose(  frist,   Second )


def test_pattern_tracking_never_changes_the_answer()-> None:

    Matrix = tridiagonal(40, diag =5.0)


    next =  SparseLU( )
    next.factorize(*as_arrays(tridiagonal(40)))
    next.factorize(*as_arrays(Matrix))


    arr=SparseLU()
    arr.factorize( *  as_arrays(Matrix) )
    bb= np.linspace(0.0, 1.0, 40)
    np.testing.assert_array_equal(next.solve(bb),arr.solve(bb))


def test_ordering_is_colamd_not_natural()  ->  None:

    Side  = 30
    Laplacian=sp.kron(
        sp.eye(Side),
        sp.diags([[-1.0] *(Side -1),[4.0]*Side,[-1.0] *(Side- 1)],[-1,0,1]),
    ) +sp.kron(
        sp.diags([[-1.0]*(Side-1),[0.0] *Side,[-1.0]*(Side - 1)],[- 1,0,1]),
        sp.eye(Side),
    )
    bar= SparseLU()
    bar.factorize(*as_arrays(sp.coo_matrix(Laplacian)))
    fillFirst  =  bar.fill_nnz
    bar.factorize(*  as_arrays( sp.coo_matrix (  Laplacian  ))  )
    assert bar.pattern_unchanged is True
    assert bar.fill_nnz   ==  fillFirst

def test_singular_matrix_raises_an_informative_error()->  None :

    k2= np.array([0,1]);clos = np.array ([ 0,  1  ] )
    valuues = np.array([1.0,0.0])
    t2 = SparseLU()
    with pytest.raises(RuntimeError,match='singular'):
        t2.factorize(k2, clos, valuues, (2, 2))




def test_solving_before_factorizing_raises() ->  None  :
    solevr = SparseLU()
    with pytest.raises(RuntimeError, match = 'factorize') :
        solevr.solve(np.ones(3))




def test_non_square_matrix_raises() ->None  :
    Solver=SparseLU()
    with pytest.raises(ValueError,match ='square'):
        Solver.factorize(np.array([0]), np.array([0]), np.array([1.0]), (2, 3))


def test_right_hand_side_of_wrong_length_raises()  -> None :
    sol =   SparseLU ( )
    sol.factorize(* as_arrays(tridiagonal(5)))
    with pytest.raises(ValueError, match  = "length") :
        sol.solve(np.ones(4))

def test_solve_package_imports_nothing_semiconductor_specific() ->None:
    out2=  {"ddsim.core.constants", "ddsim.physics", 'ddsim.device', "ddsim.discretize",}
    pac=pathlib.Path(__file__).parents[2]/"ddsim"/'solve'

    for Source in pac.glob("*.py") :
        Tree=ast.parse(Source.read_text(encoding = "utf-8"))
        for r2 in ast.walk(Tree) :
            nam:list[str] = []
            if isinstance(r2,ast.Import):
                nam = [aias.name for aias in r2.names]
            elif isinstance(r2, ast.ImportFrom) and r2.module is not None :
                nam =  [r2.module]
            for tmp2 in nam:
                for Banned in out2 :
                    assert not tmp2.startswith(Banned), (
                        f"{Source.name} imports {tmp2}, which breaks the "
                        'solve/ module boundary'
                    )


def imported_modules(source: pathlib.Path)  -> list[str] :
    tere =  ast.parse(source.read_text(encoding ='utf-8'))
    Names :  list[str] =[]
    for Node in ast.walk(tere):
        if isinstance(Node,ast.Import) :
            Names  +=   [Alias.name  for  Alias in Node.names ]
        elif isinstance(Node,ast.ImportFrom)and Node.module is not None:
            Names.append(Node.module)
    return Names




def test_nothing_imports_the_api_package()->None :
    myvar = pathlib.Path(__file__).parents[2]/"ddsim"
    entry_poiints =  {"cli.py"}


    for soruce in myvar.rglob('*.py') :
        if soruce.parent.name == 'api' or soruce.name in entry_poiints:
            continue
        for Name in  imported_modules (soruce  ) :

            assert not Name.startswith('ddsim.api'),(
                f"{soruce.relative_to(myvar)} imports {Name}. api/ is a leaf: "
                'nothing in the solver may depend on the browser layer'
            )

def test_the_api_package_goes_through_the_public_layers()  ->   None   :

    xx= {
        'ddsim.core.constants',
        "ddsim.physics",
        "ddsim.discretize",
        "ddsim.mesh",
    }

    dat  = pathlib.Path(__file__).parents[2]/ 'ddsim' /'api'

    for  sou in dat.rglob ('*.py')  :

        for nmae in imported_modules(sou):

            for  vals  in  xx   :
                assert not nmae.startswith(vals),(
                    f"api/{sou.name} imports {nmae}, which reaches past "
                    'the public layers it is allowed to call'
                )



def  test_fill_nnz_before_factorizing_raises(  ) ->  None  :
    sollver =SparseLU()
    with pytest.raises(RuntimeError,match= "factorize"):

        _ = sollver.fill_nnz

def test_size_reports_the_factorized_dimension()-> None:

    buff= SparseLU()
    buff.factorize(* as_arrays(tridiagonal(7)))
    assert buff.size  ==7


def scattered_with_duplicates(n : int) ->tuple:
    Rows, Cols, foo =[], [], []
    for ii in reversed(range(n))  :
        if ii< n -  1  :
            Rows  +=  [  ii ,   ii  + 1  ] ; Cols   += [  ii   +   1,  ii ]

            foo +=  [- 1.0 , -   1.0]
    for ii in range(n):
        Rows.append(ii)

        Cols.append(  ii)
        foo.append(1.5)
    for ii in reversed(range(n)) :
        Rows.append(ii)
        Cols.append( ii  )
        foo.append(  2.5)
    return(
        np.array(  Rows ,   dtype  =  np.int64),
        np.array(  Cols,   dtype  = np.int64  ),
        np.array(  foo),
        ( n,  n) ,
    )


def test_the_conversion_matches_scipy_on_the_first_call_and_on_a_replay() ->None :
    Rows,col,valuues,shaape= scattered_with_duplicates(9)
    sol =SparseLU()
    sol.factorize(Rows,col,valuues,shaape)

    x2 =  sp.coo_matrix((valuues, (Rows, col)), shape =  shaape).tocsc()
    np.testing.assert_array_equal(sol._matrix.indptr, x2.indptr)


    np.testing.assert_array_equal(sol._matrix.indices, x2.indices)
    np.testing.assert_array_equal(sol._matrix.data,x2.data)
    Moved = valuues  * 3.0 + 0.5
    sol.factorize(Rows, col, Moved, shaape)
    assert sol.pattern_unchanged is True



    x2= sp.coo_matrix((Moved,(Rows,col)),shape=shaape).tocsc()
    np.testing.assert_array_equal(sol._matrix.indptr,x2.indptr)
    np.testing.assert_array_equal(  sol._matrix.indices,   x2.indices  )
    np.testing.assert_array_equal(sol._matrix.data, x2.data)




def test_a_system_with_no_triplets_is_reported_as_singular() ->None:
    soler=  SparseLU()
    emp=np.array([],dtype=np.int64)



    with pytest.raises(RuntimeError, match = "singular") :
        soler.factorize(emp, emp, np.array([]), (3, 3))
def _complex_system()   ->  tuple [np.ndarray,  np.ndarray,   np.ndarray,   tuple [int,  int  ]]  :

    min  = np.array([0, 0, 1, 1, 2, 2, 0], dtype =  np.int64);foo  = np.array([0, 1, 0, 2, 1, 2, 0], dtype= np.int64)

    vlaues =np.array(
        [1.0 + 1.0j,2.0,3.0,4.0 -2.0j,0.5j,5.0,0.25- 0.75j]
    )
    return min,  foo, vlaues,   (3 ,  3 )


def test_a_complex_system_solves() -> None  :
    map,  cools,   val,   sha  =  _complex_system(  )
    slice= sp.coo_matrix((val, (map, cools)), shape = sha).toarray()
    next =  np.array([  1.0  + 0.0j, 0.0  -   2.0j,   3.0  ])
    t2  =SparseLU()
    t2.factorize(map, cools, val, sha)


    xx  =  t2.solve( next  )

    assert np.iscomplexobj(xx)
    np.testing.assert_allclose(xx,np.linalg.solve(slice,next),rtol=1e-12)

def test_duplicate_complex_triplets_are_summed() -> None :

    next,  filter,   vars ,   sape =  _complex_system(  ); data2   =   SparseLU ( )
    data2.factorize(next,filter,vars,sape)

    hash=sp.coo_matrix((vars,(next,filter)),shape =sape).tocsc()
    np.testing.assert_allclose(data2._matrix.data, hash.data, rtol = 1e-14)
def test_the_same_pattern_replayed_with_complex_values_still_works() -> None:
    Rows, buf,  stuff ,   pow =  _complex_system ()
    sol=SparseLU()
    sol.factorize(Rows, buf, stuff, pow)
    Moved  =  stuff  *  ( 2.0   +  0.5j  )  +  0.25
    sol.factorize(Rows,buf,Moved,pow)
    assert sol.pattern_unchanged is True
    Dense  =  sp.coo_matrix(  ( Moved,  ( Rows, buf)  ),  shape  =   pow).toarray( );  q= np.array([1.0, 1.0j, - 1.0])
    np.testing.assert_allclose(
        sol.solve( q  ) ,   np.linalg.solve(  Dense, q ),   rtol =   1e-12
    )

def test_switching_from_real_to_complex_on_one_solver_is_safe()  -> None :
    Rows, col, stuff, foo =_complex_system()
    arr  = stuff.real.copy(  )



    Solver = SparseLU()
    Solver.factorize(Rows, col, arr, foo)
    assert Solver.solve(np.array([1.0,2.0,3.0])).dtype==np.float64

    Solver.factorize(  Rows ,   col,  stuff ,  foo  )
    desne = sp.coo_matrix((stuff,(Rows,col)),shape=foo).toarray()
    B=np.array([1.0+0.0j,0.0 -2.0j,3.0])
    np.testing.assert_allclose(Solver.solve (  B ),  np.linalg.solve(  desne,  B  ),  rtol  =  1e-12)


def test_a_real_system_still_comes_back_real()-> None :
    vars=np.array([0, 1, 2], dtype =  np.int64)

    tmp=np.array([0, 1, 2], dtype = np.int64);  Solver=SparseLU()

    Solver.factorize(vars, tmp, np.array([2.0, 4.0, 8.0]), (3, 3))

    X =Solver.solve(np.array([2.0, 4.0, 8.0]))

    assert X.dtype ==  np.float64
    np.testing.assert_allclose(X, [1.0, 1.0, 1.0], rtol  =1e-14)




def test_integer_triplets_and_an_integer_rhs_are_promoted()->None:
    roows  =  np.array(  [ 0,   1, 2 ],  dtype  =  np.int64)
    Cols = np.array([0,1,2],dtype=np.int64)


    Solver  =   SparseLU()
    Solver.factorize(roows, Cols, np.array([2, 4, 8], dtype  = np.int64), (3, 3))
    junk  = Solver.solve(np.array([2, 4, 8], dtype =np.int64))
    assert junk.dtype==np.float64
    np.testing.assert_allclose(junk,[1.0,1.0,1.0],rtol =1e-14)
