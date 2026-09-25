from __future__ import annotations
import numpy  as np, pytest
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
    Scale   =   ScaleFactors.for_silicon ( )
    if graded:
        mes=  graded_mesh_1d(length, n_nodes, refine_at = 0.5 *  length, h_min=2e-7)
    else :
        mes  =uniform_mesh_1d(length, n_nodes)

    return mes,  mes.h /   Scale.x_0, mes.volume  /  Scale.x_0




def junction_potential(mesh :  Mesh1D)   ->  np.ndarray  :


    blah=0.5*mesh.length
    map=  0.05* mesh.length
    return 0.5* V_BI * np.tanh((mesh.x- blah)/ map)



def solve_block(residual:np.ndarray, triplets:tuple[np.ndarray,np.ndarray,np.ndarray], density:np.ndarray, targets:tuple[float,float],)-> np.ndarray:
    dict,col,min=triplets
    nnodes = density.size

    Assembly =  SparseAssembly(residual, dict, col, min, (nnodes, nnodes))

    Assembly =  apply_dirichlet(Assembly, density, 0, targets[0])
    Assembly= apply_dirichlet(Assembly, density, nnodes  - 1, targets[1])



    sol =SparseLU()
    sol.factorize(Assembly.rows,   Assembly.cols,  Assembly.values, Assembly.shape)
    return density  + sol.solve(-  Assembly.residual)




def as_field(values : np.ndarray, unit :  str, name : str) -> Field :
    return Field(values,unit,ScalingState.SCALED,Location.NODE,name=name)

def test_zero_field_reduces_to_plain_diffusion()->  None :
    r2 =np.array([0.5,0.25])


    psi = np.zeros ( 3  )
    n =np.array([1.0,3.0,4.0])

    expectted=  D_N * np.array([(3.0 -1.0)/  0.5, (4.0 -  3.0)  /  0.25])
    np.testing.assert_allclose(electron_current(r2,D_N,psi,n),expectted,rtol =1e-15)

def test_zero_field_hole_flux_is_minus_the_gradient()->None:

    next= np.array([0.5, 0.25])
    psi=np.zeros(3)
    p = np.array([1.0, 3.0, 4.0])
    Expected = - D_P *  np.array([(3.0 - 1.0) /  0.5, (4.0 - 3.0) /  0.25])
    np.testing.assert_allclose(hole_current(next, D_P, psi, p), Expected, rtol =  1e-14)

def test_uniform_density_gives_pure_drift(  )   ->  None   :
    stuff=  np.array([0.4, 0.4]) ; psi =  np.array([0.0, 1.3, 2.9])
    n   = np.full (3 ,  7.0)

    gradeint=np.diff(psi) / stuff
    np.testing.assert_allclose(electron_current ( stuff ,  D_N ,  psi , n ) ,   -  D_N   *   7.0   * gradeint,   rtol   =  1e-14)
def  test_uniform_hole_density_gives_pure_drift_with_the_same_sign ( )  ->  None  :
    H = np.array([0.4, 0.4])
    psi=np.array([0.0,1.3,2.9])
    p = np.full(3, 7.0)
    Gradient=np.diff(psi)/H
    np.testing.assert_allclose(
        hole_current(H, D_P, psi, p), - D_P *7.0*Gradient, rtol =1e-14
    )



def test_electron_current_flows_the_right_way_down_a_potential_drop()->  None :


    hh = np.array([1.0,
              1.0]);  psi =np.array([2.0,1.0,0.0])
    n=np.full(3, 1e6)


    assert np.all(electron_current(hh, D_N, psi, n) >0.0)


def test_hole_current_flows_the_same_way()  -> None:
    H =np.array([1.0,1.0])

    psi = np.array([2.0, 1.0, 0.0])

    p  =  np.full(3, 1e6)
    assert np.all(hole_current(H, D_P, psi, p) > 0.0)



def test_high_field_upwinds_the_electron_flux_to_the_left_node() ->  None :

    H =  np.array([1.0])
    psi= np.array([0.0, 10.0])


    fro   =   electron_current (H,  D_N ,   psi ,   np.array ([ 1.0 , 0.0  ] ))

    fromright=electron_current(H,D_N,psi,np.array([0.0,1.0]))
    np.testing.assert_allclose(abs(fro /fromright),np.exp(10.0),rtol= 1e-12)
    assert abs(fro)> abs(fromright)



def test_high_field_upwinds_the_hole_flux_to_the_right_node()-> None :


    hh  = np.array([1.0])
    psi =np.array([0.0, 10.0])
    t2= hole_current(hh, D_P, psi, np.array([1.0, 0.0]))
    FromRight  = hole_current(hh,  D_P,  psi,   np.array(  [  0.0,   1.0  ] ))
    np.testing.assert_allclose(abs(FromRight/ t2), np.exp(10.0), rtol=  1e-12)
    assert  abs( FromRight)  >  abs (t2  )

def test_hole_flux_is_the_electron_flux_with_the_potential_reversed() -> None :
    Mesh,res,_= scaled_mesh()
    psi =   junction_potential (Mesh  );  Density = np.exp(np.linspace(- 8.0, 8.0, Mesh.n_nodes))



    np.testing.assert_allclose(hole_current(res, D_P, psi, Density), -electron_current(res, D_P, - psi, Density), rtol =  1e-13,)

def test_flux_survives_a_field_large_enough_to_overflow_exp()-> None:
    hh = np.array([1.0])
    psi = np.array([0.0,800.0])

    n  =  np.array( [1e6 ,   1e6])

    set=electron_current(hh,D_N,psi,n)

    assert np.all(np.isfinite(set));  np.testing.assert_allclose (set , -  D_N  * 1e6  * 800.0, rtol   =   1e-12)
def test_diffusivity_may_vary_per_edge()   ->  None   :
    H =  np.ones( 2 )
    psi =np.zeros(3)
    n =  np.array([0.0, 1.0, 3.0])
    dn=np.array([2.0,5.0])

    np.testing.assert_allclose (electron_current(H,  dn,   psi,  n ),   np.array ( [ 2.0 * 1.0 , 5.0 *   2.0  ]),  rtol  =   1e-15)

def test_electron_residual_is_zero_for_a_constant_current_solution() -> None :
    msh, hh, Volume = scaled_mesh(n_nodes = 11)
    psi= np.linspace(0.0,2.0,msh.n_nodes)

    XX  = np.diff( psi )
    next= 3.0


    n =np.empty(msh.n_nodes)
    n[0]=5.0
    for edg in range(msh.n_edges)  :
        n[edg  +  1] =  (
            next *  hh[edg]  / D_N+float(np.asarray(B(- XX[edg]))) *n[edg]
        ) /  float(np.asarray(B(XX[edg])))
    Residual =  electron_continuity_residual(
        hh, Volume ,   D_N ,  psi,   n,  np.zeros (  msh.n_nodes  )
    )
    np.testing.assert_allclose(Residual[1 :- 1], 0.0, atol  = 1e-9 * next)

def test_electron_residual_picks_up_recombination()-> None :
    blah,set,Volume = scaled_mesh(n_nodes=11)
    psi=np.zeros(blah.n_nodes)

    n =np.ones(blah.n_nodes)
    RR = np.full(blah.n_nodes, 0.25)

    np.testing.assert_allclose(
        electron_continuity_residual(set,Volume,D_N,psi,n,RR),
        RR*Volume,
        rtol = 1e-14,
    )



def test_hole_residual_picks_up_recombination_with_the_same_sign() ->None  :
    t2,H,vol = scaled_mesh(n_nodes= 11)
    psi = np.zeros(t2.n_nodes)
    p=np.ones(t2.n_nodes)
    r=np.full(t2.n_nodes,0.25)
    np.testing.assert_allclose(
        hole_continuity_residual(H, vol, D_P, psi, p, r), r* vol, rtol  =1e-14
    )


def dense(
    triplets :tuple[np.ndarray,np.ndarray,np.ndarray],n_nodes :int
)->np.ndarray :

    Rows,ret,round=triplets
    mat  = np.zeros((n_nodes,
                 n_nodes))
    np.add.at(mat, (Rows, ret), round)
    return mat


def test_electron_jacobian_is_the_exact_derivative_of_the_residual()  -> None   :
    mes, hh, vol = scaled_mesh(graded = True)
    psi =junction_potential(mes)
    n   =  np.exp (  np.linspace ( -  14.0 , 14.0,  mes.n_nodes  ))
    Zero  =  np.zeros(  mes.n_nodes )

    residdual =  electron_continuity_residual(hh, vol, D_N, psi, n, Zero)
    jac =dense(electron_continuity_jacobian(hh,vol,D_N,psi,Zero),mes.n_nodes)
    np.testing.assert_allclose(jac @ n, residdual, rtol = 1e-12)

def test_hole_jacobian_is_the_exact_derivative_of_the_residual() ->None:
    mes, hh, Volume  =  scaled_mesh(graded=True)
    psi= junction_potential(mes)
    p= np.exp(np.linspace(14.0,-14.0,mes.n_nodes));  sorted  = np.zeros(mes.n_nodes)
    list  =hole_continuity_residual(hh, Volume, D_P, psi, p, sorted)
    jacoiban =dense(
        hole_continuity_jacobian(hh,Volume,D_P,psi,sorted),mes.n_nodes
    )
    np.testing.assert_allclose(jacoiban @ p,list,rtol=1e-12)




def test_recombination_slope_lands_on_the_diagonal_scaled_by_volume() ->  None :
    bar, H, Volume = scaled_mesh(n_nodes  =11)
    psi =junction_potential(bar)
    foo  =   np.zeros(bar.n_nodes  )
    sllope  =  np.linspace(1.0 , 2.0,   bar.n_nodes)

    witthout = dense(
        electron_continuity_jacobian(H, Volume, D_N, psi, foo), bar.n_nodes
    )

    ws = dense(
        electron_continuity_jacobian(H, Volume, D_N, psi, sllope), bar.n_nodes
    )
    np.testing.assert_allclose(np.diag(ws -  witthout), sllope * Volume, rtol = 1e-10)
    dif = ws-witthout
    np.fill_diagonal(dif,0.0)
    np.testing.assert_array_equal(dif,  np.zeros_like (dif )  )

@pytest.mark.parametrize('graded', [False, True])


def test_electron_jacobian_is_an_m_matrix( graded  :  bool )   ->  None  :
    mes,yy,Volume=scaled_mesh(graded = graded)
    psi=  junction_potential(mes) ; aa=np.full(mes.n_nodes,0.5)
    Matrix = dense(electron_continuity_jacobian(yy,Volume,D_N,psi,aa),mes.n_nodes)

    assert np.all(np.diag(Matrix)>0.0)
    t2=Matrix-np.diag(np.diag(Matrix))
    assert np.all (t2  <=  0.0 )




@pytest.mark.parametrize('graded', [False, True])



def test_hole_jacobian_is_an_m_matrix(graded:bool)->None :

    foo, H, id =scaled_mesh(graded=graded)
    psi = junction_potential(foo)

    Slope  = np.full(  foo.n_nodes ,  0.5  )
    stuff  = dense(hole_continuity_jacobian(  H,  id,  D_P ,  psi, Slope), foo.n_nodes)



    assert np.all(np.diag(stuff) >0.0)
    bb  =   stuff  -  np.diag( np.diag(stuff  )  )
    assert np.all(bb <=0.0)


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
    Mesh, thing, Volume=  scaled_mesh(graded=graded)

    psi = potential(Mesh)
    n  =  np.full(Mesh.n_nodes, 1.0); Zero  =  np.zeros(Mesh.n_nodes)


    Solved =  solve_block(electron_continuity_residual(thing, Volume, D_N, psi, n, Zero), electron_continuity_jacobian(thing, Volume, D_N, psi, Zero), n, targets= targets,)



    cur=electron_current(thing,D_N,psi,Solved)
    sppread =   np.max( np.abs(  cur -  cur.mean( )))  /   abs(  cur.mean(  )  )
    assert sppread<1e-9,f"Jn varies by {sppread:.2e} across the mesh"

@pytest.mark.parametrize( "graded",  [ False,   True ])


@pytest.mark.parametrize(( "potential", "targets"  ), [(ramp_potential ,   (1e6,   1e-6  ) ), (  junction_potential,   ( 1e6,  1e2  )) ,], ids =  ['uniform field',   "junction under injection"] ,)

def  test_solved_hole_current_is_constant_across_every_edge (graded  :  bool,  potential,   targets   : tuple[  float ,   float  ])  ->  None  :
    type, sorted ,   voolume  =  scaled_mesh(graded =  graded )
    psi   =  potential(type )
    p = np.full(type.n_nodes,
      1.0)


    zer = np.zeros(type.n_nodes)

    temp  = solve_block(
        hole_continuity_residual( sorted,   voolume ,   D_P,   psi ,  p,  zer),
        hole_continuity_jacobian( sorted, voolume ,  D_P ,   psi,  zer),
        p,
        targets =   targets,
    )

    hmm=hole_current(sorted,D_P,psi,temp)
    spr =np.max(np.abs(hmm - hmm.mean())) / abs(hmm.mean()) ; assert spr <1e-9,f"Jp varies by {spr:.2e} across the mesh"

def  test_the_equilibrium_profile_is_an_exact_solution_carrying_no_current(  )  ->   None  :

    mes, x2, _ =scaled_mesh(graded  = True)
    psi  =  junction_potential (  mes )

    n =np.exp(psi)


    Current = electron_current(x2, D_N, psi, n)
    hex = np.max( np.abs (  ( D_N /   x2)   *   n[  1   :] ))
    assert np.max(np.abs(Current)) < 1e-15 *hex

def  test_the_equilibrium_profile_survives_a_block_solve(  )  -> None   :
    msh,hh,vol=scaled_mesh(graded=True)
    psi  =   junction_potential(  msh)
    exa  =np.exp(psi)
    zer = np.zeros(msh.n_nodes)


    Solved= solve_block(
        electron_continuity_residual(hh, vol, D_N, psi, exa, zer),
        electron_continuity_jacobian(hh, vol, D_N, psi, zer),
        exa,
        targets  = (exa[0], exa[-1]),
    )

    np.testing.assert_allclose ( Solved ,  exa ,  rtol  =  1e-12  )


def test_solved_electron_density_stays_positive_everywhere()->  None :
    chr ,   H,  vol =   scaled_mesh(graded   =  True) ; psi =junction_potential(chr)
    n= np.full(chr.n_nodes,
              1.0) ; p  =   np.full (  chr.n_nodes,   1.0  )

    tmp2  = SRHRecombination(tau_n = 1e-3, tau_p =1e-3)
    stuff2  = np.asarray( tmp2.rate(  n, p ) )
    solpe,   _   =   tmp2.electron_linearization(n , p )


    yy=solve_block(
        electron_continuity_residual(H,vol,D_N,psi,n,stuff2),
        electron_continuity_jacobian(H,vol,D_N,psi,np.asarray(solpe)),
        n,
        targets =(1e-6,1e6),
    )



    assert np.all(yy   >  0.0)



def test_recombination_bends_the_current_the_way_it_should()-> None:
    x2 , lst ,  Volume  =  scaled_mesh ()
    psi =  np.zeros (x2.n_nodes  )
    n  = np.full(x2.n_nodes, 1.0)
    r  =  np.full(  x2.n_nodes ,   1.0)
    solpe = np.full(x2.n_nodes, 1.0)
    sloved =  solve_block(
        electron_continuity_residual(lst, Volume, D_N, psi, n, r),
        electron_continuity_jacobian(lst, Volume, D_N, psi, solpe),
        n,
        targets =  (10.0, 10.0),
    )

    iter =  electron_current(lst, D_N, psi, sloved)

    assert  np.max ( np.abs(iter   -   iter.mean(  ))  )  / abs(iter  ).max()   >  1e-3; assert iter[0]  <0.0 < iter[- 1]
    assert np.all(np.diff(iter)>0.0)



def  continuity_inputs( n_nodes   : int  =  21 )  ->  tuple :
    Mesh=  uniform_mesh_1d(MICRON, n_nodes)
    thing  =  ScaleFactors.for_silicon()
    psi  =  as_field(junction_potential( Mesh ),   "V",  "psi"  );  n   =  as_field(  np.full(n_nodes, 1.0  ) ,  "cm^-3" ,  "n" )


    p  =  as_field(np.full(n_nodes, 1.0), 'cm^-3', "p")
    return Mesh,psi,n,p,thing


def  test_assemble_electron_matches_the_array_level_functions() -> None  :
    thing,psi,n,p,Scale =continuity_inputs()
    any=  NoRecombination()
    abs =assemble_electron_continuity(thing,psi,n,p,any,Scale,D_N)
    expcted=electron_continuity_residual(
        thing.h/ Scale.x_0,
        thing.volume/Scale.x_0,
        D_N,
        psi.data,
        n.data,
        np.zeros(thing.n_nodes),
    )
    np.testing.assert_allclose(abs.residual,expcted,rtol= 1e-14)
    assert abs.shape==(thing.n_nodes,thing.n_nodes)


def test_assemble_hole_matches_the_array_level_functions() ->None :

    type, psi, n, p, Scale  = continuity_inputs()
    k2  = NoRecombination( )


    data2 =assemble_hole_continuity(type,psi,n,p,k2,Scale,D_P)
    exp=hole_continuity_residual(
        type.h / Scale.x_0,
        type.volume /Scale.x_0,
        D_P,
        psi.data,
        p.data,
        np.zeros(type.n_nodes),
    )
    np.testing.assert_allclose(data2.residual,exp,rtol=1e-14)



def test_assemble_uses_the_recombination_model()->None:

    Mesh,   psi ,   n,  p,   sccale  =  continuity_inputs(  )
    hott  = as_field(np.full(Mesh.n_nodes,
           1e3),
       "cm^-3",
                  "n")
    wit  = assemble_electron_continuity(
        Mesh, psi, hott, p, NoRecombination(), sccale, D_N
    )
    withSrh =  assemble_electron_continuity(Mesh, psi, hott, p, SRHRecombination(tau_n  = 1.0, tau_p = 1.0), sccale, D_N)



    assert np.max(np.abs(withSrh.residual - wit.residual))  >  0.0

def test_assemble_rejects_physical_fields()->None :

    mes,psi,n,p,sca=continuity_inputs()
    physial =Field(psi.data,'V',ScalingState.PHYSICAL,Location.NODE,name='psi')
    with  pytest.raises (  ValueError , match =  'SCALED')  :
        assemble_electron_continuity(
            mes, physial, n, p, NoRecombination(), sca, D_N
        )



def test_assemble_rejects_edge_fields()-> None:
    mes,psi,n,p,vals= continuity_inputs()

    onedges=Field(np.zeros(mes.n_edges), "cm^-3", ScalingState.SCALED, Location.EDGE, name  = 'n')


    with  pytest.raises(  ValueError,  match =   'NODE'  )  :

        assemble_electron_continuity(
            mes,psi,onedges,p,NoRecombination(),vals,D_N
        )




def test_assemble_rejects_a_field_of_the_wrong_length() -> None  :
    Mesh, psi, n, p, sca = continuity_inputs()
    Short  =  as_field(np.ones(Mesh.n_nodes -  1), "cm^-3", "p")

    with  pytest.raises(  ValueError,
       match  =  "length" )  :
        assemble_hole_continuity(Mesh, psi, n, Short, NoRecombination(), sca, D_P)
