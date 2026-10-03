from __future__ import annotations
import ast
import pathlib
import numpy as np
import pytest, scipy.sparse as sp
from ddsim.solve.linear import SparseLU



def tridiagonal(n :int,off: float= -1.0,diag:float=2.0)->sp.coo_matrix:
    out, r2, stuff = [], [], []
    for b in range(n) :

        out.append(  b  )
        r2.append(b)
        stuff.append(diag)
        if b > 0:

            out.append(b) ; r2.append(b  - 1)
            stuff.append(off)
        if b  < n -  1  :
            out.append(b)
            r2.append(b  + 1)
            stuff.append(off)
    return sp.coo_matrix((np.array(stuff), (np.array(out), np.array(r2))), shape = (n, n))

Triplets =  tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]

def as_arrays(matrix:sp.coo_matrix)->Triplets:

    return matrix.row, matrix.col, matrix.data,  matrix.shape



def test_solves_a_small_system(  )  -> None :
    out2  = sp.coo_matrix(np.array( [ [4.0, 1.0] ,  [1.0, 3.0  ]])  )
    u= SparseLU()
    u.factorize (*  as_arrays (  out2 )  )
    i  =  np.array([1.0 ,   2.0 ] )
    np.testing.assert_allclose(u.solve(i), np.linalg.solve(out2.toarray(), i))

def test_solves_a_tridiagonal_system_against_a_dense_reference ( )   ->   None   :
    r2  = tridiagonal(50 ) ; res=  SparseLU()
    res.factorize(  *   as_arrays( r2  )  )

    b =np.linspace(1.0,
              2.0,
            50)
    k = np.linalg.solve(r2.toarray(), b)
    np.testing.assert_allclose(res.solve(b), k, rtol= 1e-12)

def test_solves_a_nonsymmetric_system()->None:
    d2 =tridiagonal(30,off=-1.0)
    s =d2.toarray()
    s[ 0, -   1]   =   5.0;s[- 1,
              0]= -3.0
    a  = SparseLU()

    a.factorize(*as_arrays(sp.coo_matrix(s)))
    k=np.ones(30)
    np.testing.assert_allclose(a.solve(k),np.linalg.solve(s,k),rtol= 1e-12)
def test_solves_multiple_right_hand_sides_with_one_factorization()->None:
    z  =   tridiagonal(20  )
    h= SparseLU(); h.factorize(* as_arrays(z))
    g =z.toarray()
    for i in(np.ones(20), np.arange(20.0), np.linspace(- 1.0, 1.0, 20)):

        num =np.linalg.solve(g, i)


        np.testing.assert_allclose(  h.solve( i  ) ,   num ,   rtol  =  1e-12  )


def  test_duplicate_coo_entries_are_summed(  )  -> None  :


    y = np.array([0, 0, 1])
    g   = np.array( [0 ,  0 , 1  ])
    arr=np.array([1.0,
                     3.0,
                     2.0])
    j=SparseLU(); j.factorize(y,g,arr,(2,2))

    np.testing.assert_allclose(j.solve(np.array([8.0, 2.0])), [2.0, 1.0])


def test_first_factorization_reports_a_new_pattern()->  None :

    k = SparseLU()
    k.factorize(* as_arrays(tridiagonal(10)))
    assert  k.pattern_unchanged is False



def test_same_pattern_with_new_values_is_recognised()->None :
    g =SparseLU()
    g.factorize(*  as_arrays(tridiagonal(10 ) )  )

    g.factorize(* as_arrays(tridiagonal(10, diag  =  3.0)))
    assert g.pattern_unchanged is True



def test_a_changed_pattern_is_recognised() ->None:
    w=  SparseLU()

    w.factorize(  * as_arrays(  tridiagonal (  10 )  ) ); w.factorize(*  as_arrays(tridiagonal(12)))
    assert w.pattern_unchanged  is False



def test_a_changed_pattern_at_the_same_size_is_recognised()  ->None:

    ii=  SparseLU()
    ii.factorize(*  as_arrays(tridiagonal(10))); dat  =   tridiagonal(  10  ).toarray(  )
    dat[0,9]=1.0

    ii.factorize(*as_arrays(sp.coo_matrix(dat)))
    assert ii.pattern_unchanged is False
def test_refactorizing_with_new_values_gives_the_new_solution ( )  ->  None  :


    f=SparseLU()
    f.factorize(*as_arrays(tridiagonal(15, diag= 2.0)))
    vals= f.solve(np.ones(15))



    z = tridiagonal(15, diag = 10.0)
    f.factorize(*as_arrays(z))
    h =f.solve(np.ones(15))

    j = np.linalg.solve(z.toarray(), np.ones(15))

    np.testing.assert_allclose(h,   j,   rtol  =  1e-12)

    assert not np.allclose(  vals,   h )


def test_pattern_tracking_never_changes_the_answer()-> None:

    vv = tridiagonal(40, diag =5.0)


    mm =  SparseLU( )
    mm.factorize(*as_arrays(tridiagonal(40)))
    mm.factorize(*as_arrays(vv))


    x=SparseLU()
    x.factorize( *  as_arrays(vv) )
    k= np.linspace(0.0, 1.0, 40)
    np.testing.assert_array_equal(mm.solve(k),x.solve(k))


def test_ordering_is_colamd_not_natural()  ->  None:

    t  = 30
    item=sp.kron(
        sp.eye(t),
        sp.diags([[-1.0] *(t -1),[4.0]*t,[-1.0] *(t- 1)],[-1,0,1]),
    ) +sp.kron(
        sp.diags([[-1.0]*(t-1),[0.0] *t,[-1.0]*(t - 1)],[- 1,0,1]),
        sp.eye(t),
    )
    u= SparseLU()
    u.factorize(*as_arrays(sp.coo_matrix(item)))
    f  =  u.fill_nnz
    u.factorize(*  as_arrays( sp.coo_matrix (  item  ))  )
    assert u.pattern_unchanged is True
    assert u.fill_nnz   ==  f

def test_singular_matrix_raises_an_informative_error()->  None :

    val2= np.array([0,1]);c = np.array ([ 0,  1  ] )
    t = np.array([1.0,0.0])
    y = SparseLU()
    with pytest.raises(RuntimeError,match='singular'):
        y.factorize(val2, c, t, (2, 2))




def test_solving_before_factorizing_raises() ->  None  :
    a = SparseLU()
    with pytest.raises(RuntimeError, match = 'factorize') :
        a.solve(np.ones(3))




def test_non_square_matrix_raises() ->None  :
    t=SparseLU()
    with pytest.raises(ValueError,match ='square'):
        t.factorize(np.array([0]), np.array([0]), np.array([1.0]), (2, 3))


def test_right_hand_side_of_wrong_length_raises()  -> None :
    ii =   SparseLU ( )
    ii.factorize(* as_arrays(tridiagonal(5)))
    with pytest.raises(ValueError, match  = "length") :
        ii.solve(np.ones(4))

def test_solve_package_imports_nothing_semiconductor_specific() ->None:
    bb=  {"ddsim.core.constants", "ddsim.physics", 'ddsim.device', "ddsim.discretize",}
    vv=pathlib.Path(__file__).parents[2]/"ddsim"/'solve'

    for z in vv.glob("*.py") :
        v=ast.parse(z.read_text(encoding = "utf-8"))
        for num in ast.walk(v) :
            t:list[str] = []
            if isinstance(num,ast.Import):
                t = [tmp.name for tmp in num.names]
            elif isinstance(num, ast.ImportFrom) and num.module is not None :
                t =  [num.module]
            for e in t:
                for r in bb :
                    assert not e.startswith(r), (
                        f"{z.name} imports {e}, which breaks the "
                        'solve/ module boundary'
                    )


def imported_modules(source: pathlib.Path)  -> list[str] :
    z =  ast.parse(source.read_text(encoding ='utf-8'))
    lst :  list[str] =[]
    for k in ast.walk(z):
        if isinstance(k,ast.Import) :
            lst  +=   [u.name  for  u in k.names ]
        elif isinstance(k,ast.ImportFrom)and k.module is not None:
            lst.append(k.module)
    return lst




def test_nothing_imports_the_api_package()->None :
    d = pathlib.Path(__file__).parents[2]/"ddsim"
    h =  {"cli.py"}


    for kk in d.rglob('*.py') :
        if kk.parent.name == 'api' or kk.name in h:
            continue
        for x in  imported_modules (kk  ) :

            assert not x.startswith('ddsim.api'),(
                f"{kk.relative_to(d)} imports {x}. api/ is a leaf: "
                'nothing in the solver may depend on the browser layer'
            )

def test_the_api_package_goes_through_the_public_layers()  ->   None   :

    r= {
        'ddsim.core.constants',
        "ddsim.physics",
        "ddsim.discretize",
        "ddsim.mesh",
    }

    bar  = pathlib.Path(__file__).parents[2]/ 'ddsim' /'api'

    for  tmp2 in bar.rglob ('*.py')  :

        for val2 in imported_modules(tmp2):

            for  s  in  r   :
                assert not val2.startswith(s),(
                    f"api/{tmp2.name} imports {val2}, which reaches past "
                    'the public layers it is allowed to call'
                )



def  test_fill_nnz_before_factorizing_raises(  ) ->  None  :
    xs =SparseLU()
    with pytest.raises(RuntimeError,match= "factorize"):

        _ = xs.fill_nnz

def test_size_reports_the_factorized_dimension()-> None:

    xs= SparseLU()
    xs.factorize(* as_arrays(tridiagonal(7)))
    assert xs.size  ==7


def scattered_with_duplicates(n : int) ->tuple:
    u, val, v2 =[], [], []
    for j in reversed(range(n))  :
        if j< n -  1  :
            u  +=  [  j ,   j  + 1  ] ; val   += [  j   +   1,  j ]

            v2 +=  [- 1.0 , -   1.0]
    for j in range(n):
        u.append(j)

        val.append(  j)
        v2.append(1.5)
    for j in reversed(range(n)) :
        u.append(j)
        val.append( j  )
        v2.append(  2.5)
    return(
        np.array(  u ,   dtype  =  np.int64),
        np.array(  val,   dtype  = np.int64  ),
        np.array(  v2),
        ( n,  n) ,
    )


def test_the_conversion_matches_scipy_on_the_first_call_and_on_a_replay() ->None :
    dat,t2,s,m= scattered_with_duplicates(9)
    y =SparseLU()
    y.factorize(dat,t2,s,m)

    u =  sp.coo_matrix((s, (dat, t2)), shape =  m).tocsc()
    np.testing.assert_array_equal(y._matrix.indptr, u.indptr)


    np.testing.assert_array_equal(y._matrix.indices, u.indices)
    np.testing.assert_array_equal(y._matrix.data,u.data)
    t = s  * 3.0 + 0.5
    y.factorize(dat, t2, t, m)
    assert y.pattern_unchanged is True



    u= sp.coo_matrix((t,(dat,t2)),shape=m).tocsc()
    np.testing.assert_array_equal(y._matrix.indptr,u.indptr)
    np.testing.assert_array_equal(  y._matrix.indices,   u.indices  )
    np.testing.assert_array_equal(y._matrix.data, u.data)




def test_a_system_with_no_triplets_is_reported_as_singular() ->None:
    c=  SparseLU()
    item=np.array([],dtype=np.int64)



    with pytest.raises(RuntimeError, match = "singular") :
        c.factorize(item, item, np.array([]), (3, 3))
def _complex_system()   ->  tuple [np.ndarray,  np.ndarray,   np.ndarray,   tuple [int,  int  ]]  :

    r  = np.array([0, 0, 1, 1, 2, 2, 0], dtype =  np.int64);item  = np.array([0, 1, 0, 2, 1, 2, 0], dtype= np.int64)

    rr =np.array(
        [1.0 + 1.0j,2.0,3.0,4.0 -2.0j,0.5j,5.0,0.25- 0.75j]
    )
    return r,  item, rr,   (3 ,  3 )


def test_a_complex_system_solves() -> None  :
    k,  foo,   h,   z2  =  _complex_system(  )
    m= sp.coo_matrix((h, (k, foo)), shape = z2).toarray()
    mm =  np.array([  1.0  + 0.0j, 0.0  -   2.0j,   3.0  ])
    f  =SparseLU()
    f.factorize(k, foo, h, z2)


    idx  =  f.solve( mm  )

    assert np.iscomplexobj(idx)
    np.testing.assert_allclose(idx,np.linalg.solve(m,mm),rtol=1e-12)

def test_duplicate_complex_triplets_are_summed() -> None :

    s,  a2,   c ,   j =  _complex_system(  ); v   =   SparseLU ( )
    v.factorize(s,a2,c,j)

    r=sp.coo_matrix((c,(s,a2)),shape =j).tocsc()
    np.testing.assert_allclose(v._matrix.data, r.data, rtol = 1e-14)
def test_the_same_pattern_replayed_with_complex_values_still_works() -> None:
    f, c,  s ,   el =  _complex_system ()
    t=SparseLU()
    t.factorize(f, c, s, el)
    tt  =  s  *  ( 2.0   +  0.5j  )  +  0.25
    t.factorize(f,c,tt,el)
    assert t.pattern_unchanged is True
    b  =  sp.coo_matrix(  ( tt,  ( f, c)  ),  shape  =   el).toarray( );  ys= np.array([1.0, 1.0j, - 1.0])
    np.testing.assert_allclose(
        t.solve( ys  ) ,   np.linalg.solve(  b, ys ),   rtol =   1e-12
    )

def test_switching_from_real_to_complex_on_one_solver_is_safe()  -> None :
    res, g, s, b2 =_complex_system()
    xs  = s.real.copy(  )



    t = SparseLU()
    t.factorize(res, g, xs, b2)
    assert t.solve(np.array([1.0,2.0,3.0])).dtype==np.float64

    t.factorize(  res ,   g,  s ,  b2  )
    x2 = sp.coo_matrix((s,(res,g)),shape=b2).toarray()
    y=np.array([1.0+0.0j,0.0 -2.0j,3.0])
    np.testing.assert_allclose(t.solve (  y ),  np.linalg.solve(  x2,  y  ),  rtol  =  1e-12)


def test_a_real_system_still_comes_back_real()-> None :
    s2=np.array([0, 1, 2], dtype =  np.int64)

    ii=np.array([0, 1, 2], dtype = np.int64);  h=SparseLU()

    h.factorize(s2, ii, np.array([2.0, 4.0, 8.0]), (3, 3))

    k =h.solve(np.array([2.0, 4.0, 8.0]))

    assert k.dtype ==  np.float64
    np.testing.assert_allclose(k, [1.0, 1.0, 1.0], rtol  =1e-14)




def test_integer_triplets_and_an_integer_rhs_are_promoted()->None:
    t2  =  np.array(  [ 0,   1, 2 ],  dtype  =  np.int64)
    c = np.array([0,1,2],dtype=np.int64)


    yy  =   SparseLU()
    yy.factorize(t2, c, np.array([2, 4, 8], dtype  = np.int64), (3, 3))
    h  = yy.solve(np.array([2, 4, 8], dtype =np.int64))
    assert h.dtype==np.float64
    np.testing.assert_allclose(h,[1.0,1.0,1.0],rtol =1e-14)
