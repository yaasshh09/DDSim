from __future__ import annotations
import numpy as np, pytest
from ddsim.core.field import Field,Location,ScalingState
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import apply_dirichlet
from ddsim.discretize.continuity import(assemble_electron_continuity, assemble_hole_continuity, electron_continuity_jacobian, electron_continuity_residual, electron_current, hole_continuity_jacobian, hole_continuity_residual, hole_current,)
from ddsim.mesh.mesh1d import Mesh1D, graded_mesh_1d, uniform_mesh_1d
from ddsim.physics.bernoulli import B
from ddsim.physics.recombination import NoRecombination,SRHRecombination
from ddsim.solve.linear import SparseLU
MICRON  = 1e-4

D_N =1.0

D_P = 0.3317

V_BI=27.6



def scaled_mesh(n_nodes: int=41, length:float =  MICRON, graded :  bool=  False)  ->  tuple[Mesh1D, np.ndarray, np.ndarray]:
    y   =   ScaleFactors.for_silicon ( )
    if graded:
        val=  graded_mesh_1d(length, n_nodes, refine_at = 0.5 *  length, h_min=2e-7)
    else :
        val  =uniform_mesh_1d(length, n_nodes)

    return val,  val.h /   y.x_0, val.volume  /  y.x_0




def junction_potential(mesh :  Mesh1D)   ->  np.ndarray  :


    b=0.5*mesh.length
    rr=  0.05* mesh.length
    return 0.5* V_BI * np.tanh((mesh.x- b)/ rr)



def solve_block(residual:np.ndarray, triplets:tuple[np.ndarray,np.ndarray,np.ndarray], density:np.ndarray, targets:tuple[float,float],)-> np.ndarray:
    res2,cc,d=triplets
    f = density.size

    x =  SparseAssembly(residual, res2, cc, d, (f, f))

    x =  apply_dirichlet(x, density, 0, targets[0])
    x= apply_dirichlet(x, density, f  - 1, targets[1])



    c =SparseLU()
    c.factorize(x.rows,   x.cols,  x.values, x.shape)
    return density  + c.solve(-  x.residual)




def as_field(values : np.ndarray, unit :  str, name : str) -> Field :
    return Field(values,unit,ScalingState.SCALED,Location.NODE,name=name)

def test_zero_field_reduces_to_plain_diffusion()->  None :
    vv =np.array([0.5,0.25])


    psi = np.zeros ( 3  )
    n =np.array([1.0,3.0,4.0])

    dat=  D_N * np.array([(3.0 -1.0)/  0.5, (4.0 -  3.0)  /  0.25])
    np.testing.assert_allclose(electron_current(vv,D_N,psi,n),dat,rtol =1e-15)

def test_zero_field_hole_flux_is_minus_the_gradient()->None:

    z= np.array([0.5, 0.25])
    psi=np.zeros(3)
    p = np.array([1.0, 3.0, 4.0])
    v = - D_P *  np.array([(3.0 - 1.0) /  0.5, (4.0 - 3.0) /  0.25])
    np.testing.assert_allclose(hole_current(z, D_P, psi, p), v, rtol =  1e-14)

def test_uniform_density_gives_pure_drift(  )   ->  None   :
    info=  np.array([0.4, 0.4]) ; psi =  np.array([0.0, 1.3, 2.9])
    n   = np.full (3 ,  7.0)

    d=np.diff(psi) / info
    np.testing.assert_allclose(electron_current ( info ,  D_N ,  psi , n ) ,   -  D_N   *   7.0   * d,   rtol   =  1e-14)
def  test_uniform_hole_density_gives_pure_drift_with_the_same_sign ( )  ->  None  :
    rr = np.array([0.4, 0.4])
    psi=np.array([0.0,1.3,2.9])
    p = np.full(3, 7.0)
    tmp3=np.diff(psi)/rr
    np.testing.assert_allclose(
        hole_current(rr, D_P, psi, p), - D_P *7.0*tmp3, rtol =1e-14
    )



def test_electron_current_flows_the_right_way_down_a_potential_drop()->  None :


    k = np.array([1.0,
              1.0]);  psi =np.array([2.0,1.0,0.0])
    n=np.full(3, 1e6)


    assert np.all(electron_current(k, D_N, psi, n) >0.0)


def test_hole_current_flows_the_same_way()  -> None:
    obj =np.array([1.0,1.0])

    psi = np.array([2.0, 1.0, 0.0])

    p  =  np.full(3, 1e6)
    assert np.all(hole_current(obj, D_P, psi, p) > 0.0)



def test_high_field_upwinds_the_electron_flux_to_the_left_node() ->  None :

    hh =  np.array([1.0])
    psi= np.array([0.0, 10.0])


    y   =   electron_current (hh,  D_N ,   psi ,   np.array ([ 1.0 , 0.0  ] ))

    r=electron_current(hh,D_N,psi,np.array([0.0,1.0]))
    np.testing.assert_allclose(abs(y /r),np.exp(10.0),rtol= 1e-12)
    assert abs(y)> abs(r)



def test_high_field_upwinds_the_hole_flux_to_the_right_node()-> None :


    m  = np.array([1.0])
    psi =np.array([0.0, 10.0])
    c= hole_current(m, D_P, psi, np.array([1.0, 0.0]))
    ret  = hole_current(m,  D_P,  psi,   np.array(  [  0.0,   1.0  ] ))
    np.testing.assert_allclose(abs(ret/ c), np.exp(10.0), rtol=  1e-12)
    assert  abs( ret)  >  abs (c  )

def test_hole_flux_is_the_electron_flux_with_the_potential_reversed() -> None :
    j,a2,_= scaled_mesh()
    psi =   junction_potential (j  );  nxt = np.exp(np.linspace(- 8.0, 8.0, j.n_nodes))



    np.testing.assert_allclose(hole_current(a2, D_P, psi, nxt), -electron_current(a2, D_P, - psi, nxt), rtol =  1e-13,)

def test_flux_survives_a_field_large_enough_to_overflow_exp()-> None:
    j = np.array([1.0])
    psi = np.array([0.0,800.0])

    n  =  np.array( [1e6 ,   1e6])

    item=electron_current(j,D_N,psi,n)

    assert np.all(np.isfinite(item));  np.testing.assert_allclose (item , -  D_N  * 1e6  * 800.0, rtol   =   1e-12)
def test_diffusivity_may_vary_per_edge()   ->  None   :
    k =  np.ones( 2 )
    psi =np.zeros(3)
    n =  np.array([0.0, 1.0, 3.0])
    lst=np.array([2.0,5.0])

    np.testing.assert_allclose (electron_current(k,  lst,   psi,  n ),   np.array ( [ 2.0 * 1.0 , 5.0 *   2.0  ]),  rtol  =   1e-15)

def test_electron_residual_is_zero_for_a_constant_current_solution() -> None :
    val, info, g = scaled_mesh(n_nodes = 11)
    psi= np.linspace(0.0,2.0,val.n_nodes)

    b  = np.diff( psi )
    k= 3.0


    n =np.empty(val.n_nodes)
    n[0]=5.0
    for bar in range(val.n_edges)  :
        n[bar  +  1] =  (
            k *  info[bar]  / D_N+float(np.asarray(B(- b[bar]))) *n[bar]
        ) /  float(np.asarray(B(b[bar])))
    d =  electron_continuity_residual(
        info, g ,   D_N ,  psi,   n,  np.zeros (  val.n_nodes  )
    )
    np.testing.assert_allclose(d[1 :- 1], 0.0, atol  = 1e-9 * k)

def test_electron_residual_picks_up_recombination()-> None :
    tmp,bar,vals = scaled_mesh(n_nodes=11)
    psi=np.zeros(tmp.n_nodes)

    n =np.ones(tmp.n_nodes)
    r = np.full(tmp.n_nodes, 0.25)

    np.testing.assert_allclose(
        electron_continuity_residual(bar,vals,D_N,psi,n,r),
        r*vals,
        rtol = 1e-14,
    )



def test_hole_residual_picks_up_recombination_with_the_same_sign() ->None  :
    a,u,j = scaled_mesh(n_nodes= 11)
    psi = np.zeros(a.n_nodes)
    p=np.ones(a.n_nodes)
    c=np.full(a.n_nodes,0.25)
    np.testing.assert_allclose(
        hole_continuity_residual(u, j, D_P, psi, p, c), c* j, rtol  =1e-14
    )


def dense(
    triplets :tuple[np.ndarray,np.ndarray,np.ndarray],n_nodes :int
)->np.ndarray :

    j,m,yy=triplets
    s  = np.zeros((n_nodes,
                 n_nodes))
    np.add.at(s, (j, m), yy)
    return s


def test_electron_jacobian_is_the_exact_derivative_of_the_residual()  -> None   :
    prev, k, z = scaled_mesh(graded = True)
    psi =junction_potential(prev)
    n   =  np.exp (  np.linspace ( -  14.0 , 14.0,  prev.n_nodes  ))
    y  =  np.zeros(  prev.n_nodes )

    dat =  electron_continuity_residual(k, z, D_N, psi, n, y)
    m =dense(electron_continuity_jacobian(k,z,D_N,psi,y),prev.n_nodes)
    np.testing.assert_allclose(m @ n, dat, rtol = 1e-12)

def test_hole_jacobian_is_the_exact_derivative_of_the_residual() ->None:
    h, t, i  =  scaled_mesh(graded=True)
    psi= junction_potential(h)
    p= np.exp(np.linspace(14.0,-14.0,h.n_nodes));  cnt  = np.zeros(h.n_nodes)
    x  =hole_continuity_residual(t, i, D_P, psi, p, cnt)
    v =dense(
        hole_continuity_jacobian(t,i,D_P,psi,cnt),h.n_nodes
    )
    np.testing.assert_allclose(v @ p,x,rtol=1e-12)




def test_recombination_slope_lands_on_the_diagonal_scaled_by_volume() ->  None :
    x, rows, y2 = scaled_mesh(n_nodes  =11)
    psi =junction_potential(x)
    d  =   np.zeros(x.n_nodes  )
    jj  =  np.linspace(1.0 , 2.0,   x.n_nodes)

    s = dense(
        electron_continuity_jacobian(rows, y2, D_N, psi, d), x.n_nodes
    )

    arr = dense(
        electron_continuity_jacobian(rows, y2, D_N, psi, jj), x.n_nodes
    )
    np.testing.assert_allclose(np.diag(arr -  s), jj * y2, rtol = 1e-10)
    info = arr-s
    np.fill_diagonal(info,0.0)
    np.testing.assert_array_equal(info,  np.zeros_like (info )  )

@pytest.mark.parametrize('graded', [False, True])


def test_electron_jacobian_is_an_m_matrix( graded  :  bool )   ->  None  :
    i,t,buf=scaled_mesh(graded = graded)
    psi=  junction_potential(i) ; out2=np.full(i.n_nodes,0.5)
    y = dense(electron_continuity_jacobian(t,buf,D_N,psi,out2),i.n_nodes)

    assert np.all(np.diag(y)>0.0)
    vv=y-np.diag(np.diag(y))
    assert np.all (vv  <=  0.0 )




@pytest.mark.parametrize('graded', [False, True])



def test_hole_jacobian_is_an_m_matrix(graded:bool)->None :

    k, j, r =scaled_mesh(graded=graded)
    psi = junction_potential(k)

    jj  = np.full(  k.n_nodes ,  0.5  )
    lst  = dense(hole_continuity_jacobian(  j,  r,  D_P ,  psi, jj), k.n_nodes)



    assert np.all(np.diag(lst) >0.0)
    a2  =   lst  -  np.diag( np.diag(lst  )  )
    assert np.all(a2 <=0.0)


def ramp_potential(mesh: Mesh1D)->np.ndarray:
    return 10.0 *  (1.0- mesh.x/  mesh.length)



@pytest.mark.parametrize('graded',[False,True])

@pytest.mark.parametrize(
    (  'potential' ,  'targets') ,
    [
        (ramp_potential , (  1e-6 ,   1e6 )),
        ( junction_potential,   ( 1e2 ,  1e6 )  ) ,
    ],
    ids =  ['uniform field', 'junction under injection'],
)
def test_solved_electron_current_is_constant_across_every_edge(graded:bool, potential, targets  : tuple[float, float])  ->None :
    w, h, stuff=  scaled_mesh(graded=graded)

    psi = potential(w)
    n  =  np.full(w.n_nodes, 1.0); t  =  np.zeros(w.n_nodes)


    row =  solve_block(electron_continuity_residual(h, stuff, D_N, psi, n, t), electron_continuity_jacobian(h, stuff, D_N, psi, t), n, targets= targets,)



    x=electron_current(h,D_N,psi,row)
    u =   np.max( np.abs(  x -  x.mean( )))  /   abs(  x.mean(  )  )
    assert u<1e-9,f"Jn varies by {u:.2e} across the mesh"

@pytest.mark.parametrize( "graded",  [ False,   True ])


@pytest.mark.parametrize(( "potential", "targets"  ), [(ramp_potential ,   (1e6,   1e-6  ) ), (  junction_potential,   ( 1e6,  1e2  )) ,], ids =  ['uniform field',   "junction under injection"] ,)

def  test_solved_hole_current_is_constant_across_every_edge (graded  :  bool,  potential,   targets   : tuple[  float ,   float  ])  ->  None  :
    y, j ,   bb  =  scaled_mesh(graded =  graded )
    psi   =  potential(y )
    p = np.full(y.n_nodes,
      1.0)


    g = np.zeros(y.n_nodes)

    d2  = solve_block(
        hole_continuity_residual( j,   bb ,   D_P,   psi ,  p,  g),
        hole_continuity_jacobian( j, bb ,  D_P ,   psi,  g),
        p,
        targets =   targets,
    )

    s2=hole_current(j,D_P,psi,d2)
    c =np.max(np.abs(s2 - s2.mean())) / abs(s2.mean()) ; assert c <1e-9,f"Jp varies by {c:.2e} across the mesh"

def  test_the_equilibrium_profile_is_an_exact_solution_carrying_no_current(  )  ->   None  :

    cur, j, _ =scaled_mesh(graded  = True)
    psi  =  junction_potential (  cur )

    n =np.exp(psi)


    a2 = electron_current(j, D_N, psi, n)
    z = np.max( np.abs (  ( D_N /   j)   *   n[  1   :] ))
    assert np.max(np.abs(a2)) < 1e-15 *z

def  test_the_equilibrium_profile_survives_a_block_solve(  )  -> None   :
    lst,v,jj=scaled_mesh(graded=True)
    psi  =   junction_potential(  lst)
    y  =np.exp(psi)
    d2 = np.zeros(lst.n_nodes)


    f= solve_block(
        electron_continuity_residual(v, jj, D_N, psi, y, d2),
        electron_continuity_jacobian(v, jj, D_N, psi, d2),
        y,
        targets  = (y[0], y[-1]),
    )

    np.testing.assert_allclose ( f ,  y ,  rtol  =  1e-12  )


def test_solved_electron_density_stays_positive_everywhere()->  None :
    v ,   zz,  rr =   scaled_mesh(graded   =  True) ; psi =junction_potential(v)
    n= np.full(v.n_nodes,
              1.0) ; p  =   np.full (  v.n_nodes,   1.0  )

    b  = SRHRecombination(tau_n = 1e-3, tau_p =1e-3)
    out2  = np.asarray( b.rate(  n, p ) )
    h,   _   =   b.electron_linearization(n , p )


    s=solve_block(
        electron_continuity_residual(zz,rr,D_N,psi,n,out2),
        electron_continuity_jacobian(zz,rr,D_N,psi,np.asarray(h)),
        n,
        targets =(1e-6,1e6),
    )



    assert np.all(s   >  0.0)



def test_recombination_bends_the_current_the_way_it_should()-> None:
    b , a ,  vv  =  scaled_mesh ()
    psi =  np.zeros (b.n_nodes  )
    n  = np.full(b.n_nodes, 1.0)
    f  =  np.full(  b.n_nodes ,   1.0)
    k2 = np.full(b.n_nodes, 1.0)
    m =  solve_block(
        electron_continuity_residual(a, vv, D_N, psi, n, f),
        electron_continuity_jacobian(a, vv, D_N, psi, k2),
        n,
        targets =  (10.0, 10.0),
    )

    y =  electron_current(a, D_N, psi, m)

    assert  np.max ( np.abs(y   -   y.mean(  ))  )  / abs(y  ).max()   >  1e-3; assert y[0]  <0.0 < y[- 1]
    assert np.all(np.diff(y)>0.0)



def  continuity_inputs( n_nodes   : int  =  21 )  ->  tuple :
    a=  uniform_mesh_1d(MICRON, n_nodes)
    ret  =  ScaleFactors.for_silicon()
    psi  =  as_field(junction_potential( a ),   "V",  "psi"  );  n   =  as_field(  np.full(n_nodes, 1.0  ) ,  "cm^-3" ,  "n" )


    p  =  as_field(np.full(n_nodes, 1.0), 'cm^-3', "p")
    return a,psi,n,p,ret


def  test_assemble_electron_matches_the_array_level_functions() -> None  :
    el,psi,n,p,b =continuity_inputs()
    d=  NoRecombination()
    r2 =assemble_electron_continuity(el,psi,n,p,d,b,D_N)
    mm=electron_continuity_residual(
        el.h/ b.x_0,
        el.volume/b.x_0,
        D_N,
        psi.data,
        n.data,
        np.zeros(el.n_nodes),
    )
    np.testing.assert_allclose(r2.residual,mm,rtol= 1e-14)
    assert r2.shape==(el.n_nodes,el.n_nodes)


def test_assemble_hole_matches_the_array_level_functions() ->None :

    d2, psi, n, p, r  = continuity_inputs()
    val2  = NoRecombination( )


    v =assemble_hole_continuity(d2,psi,n,p,val2,r,D_P)
    z=hole_continuity_residual(
        d2.h / r.x_0,
        d2.volume /r.x_0,
        D_P,
        psi.data,
        p.data,
        np.zeros(d2.n_nodes),
    )
    np.testing.assert_allclose(v.residual,z,rtol=1e-14)



def test_assemble_uses_the_recombination_model()->None:

    x2,   psi ,   n,  p,   g  =  continuity_inputs(  )
    v  = as_field(np.full(x2.n_nodes,
           1e3),
       "cm^-3",
                  "n")
    j  = assemble_electron_continuity(
        x2, psi, v, p, NoRecombination(), g, D_N
    )
    y2 =  assemble_electron_continuity(x2, psi, v, p, SRHRecombination(tau_n  = 1.0, tau_p = 1.0), g, D_N)



    assert np.max(np.abs(y2.residual - j.residual))  >  0.0

def test_assemble_rejects_physical_fields()->None :

    tmp3,psi,n,p,yy=continuity_inputs()
    b2 =Field(psi.data,'V',ScalingState.PHYSICAL,Location.NODE,name='psi')
    with  pytest.raises (  ValueError , match =  'SCALED')  :
        assemble_electron_continuity(
            tmp3, b2, n, p, NoRecombination(), yy, D_N
        )



def test_assemble_rejects_edge_fields()-> None:
    x,psi,n,p,out= continuity_inputs()

    i=Field(np.zeros(x.n_edges), "cm^-3", ScalingState.SCALED, Location.EDGE, name  = 'n')


    with  pytest.raises(  ValueError,  match =   'NODE'  )  :

        assemble_electron_continuity(
            x,psi,i,p,NoRecombination(),out,D_N
        )




def test_assemble_rejects_a_field_of_the_wrong_length() -> None  :
    h, psi, n, p, out = continuity_inputs()
    a  =  as_field(np.ones(h.n_nodes -  1), "cm^-3", "p")

    with  pytest.raises(  ValueError,
       match  =  "length" )  :
        assemble_hole_continuity(h, psi, n, a, NoRecombination(), out, D_P)
