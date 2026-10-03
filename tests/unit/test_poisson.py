from __future__ import annotations
import numpy as np
import pytest
import scipy.sparse as sp
from ddsim.core.field import Field, Location, ScalingState
from ddsim.core.scaling import ScaleFactors
from  ddsim.discretize.poisson import(
    assemble_poisson ,
    poisson_jacobian ,
    poisson_residual ,
)
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.physics.statistics  import psi_equilibrium_scaled
MICRON =  1e-4

def  scaled_mesh ( n_nodes  : int  =  21,   length  :  float  =  MICRON  )  ->  tuple :
    obj  = ScaleFactors.for_silicon (  )
    b  =  uniform_mesh_1d(length, n_nodes)

    return b.h/obj.x_0,b.volume /obj.x_0


@pytest.mark.parametrize("doping_scaled",[-1e6,-1.0,0.0,1.0,1e6,1e8])



def test_residual_is_zero_for_uniform_material_at_equilibrium(doping_scaled: float,) -> None :
    yy,vv =scaled_mesh(); i = np.full(yy.size  + 1, doping_scaled)
    psi  = np.full( yy.size  +   1, float(psi_equilibrium_scaled (doping_scaled ))  )
    y  =poisson_residual(yy, vv, psi, i)
    np.testing.assert_allclose(y, 0.0, atol = 1e-9 * max(1.0, abs(doping_scaled)))

def test_residual_is_nonzero_when_psi_is_off_equilibrium() ->None:
    ys, thing = scaled_mesh()


    z  =  np.full(  ys.size +   1, 1e6 )
    psi  = np.full(ys.size  + 1, float(psi_equilibrium_scaled(1e6))  + 0.1)


    assert np.max(  np.abs(  poisson_residual(  ys, thing,  psi,   z  )  ))  >  1.0



def test_laplacian_of_a_linear_potential_vanishes_in_the_interior()   ->   None   :
    nxt, dd  =scaled_mesh(n_nodes =21)
    psi =np.concatenate(([0.0],np.cumsum(nxt)))*3.0
    yy  =np.zeros(psi.size)

    z2 = poisson_residual(nxt,dd,psi,yy)
    r= -(np.exp(- psi)-np.exp(psi)+yy)* dd
    np.testing.assert_allclose(z2[1:-1],r[1:-1],atol =1e-14)


def test_boundary_rows_use_a_reflecting_condition()->None:
    z,   ii  = scaled_mesh(  n_nodes   =   5  );  psi = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    el  =np.zeros(5)
    a= poisson_residual(z,ii,psi,el)
    t  = np.exp(  1.0)
    num=np.exp(-1.0)
    k = - (0.0- 1.0)/ z[0]-(num-t)*ii[0]
    assert a[0]  == pytest.approx(k, rel =  1e-14)




def test_charge_term_has_the_sign_that_pulls_psi_toward_neutrality()->None  :
    t, kk  =   scaled_mesh (n_nodes = 11 )

    psi = np.zeros(11)

    u=poisson_residual(t,kk,psi,np.full(11,1e6))
    ii=poisson_residual(t,kk,psi,np.full(11,-1e6))

    assert  np.all(u [ 1 :-  1  ]   <   0.0)
    assert np.all(ii[1 :-1] >0.0)



def test_jacobian_matches_complex_step_differentiation()-> None:


    t, ys=scaled_mesh(n_nodes  = 20)

    k =np.random.default_rng(0);  psi  =  k.uniform(-  8.0, 8.0, 20)
    r =  k.uniform(- 1e5, 1e5, 20)

    a,c2,y =poisson_jacobian(t,ys,psi,r); d = sp.coo_matrix((y, (a, c2)), shape =(20, 20)).toarray()
    hh=1e-30
    for z2 in range(20):
        g= psi.astype(np.complex128)
        g[ z2  ]   += 1j   *   hh
        w  =  poisson_residual ( t,  ys, g ,   r ).imag /  hh
        np.testing.assert_allclose(d[:, z2], w, rtol =  1e-12, atol  =  1e-12)
def test_jacobian_matches_finite_difference_on_a_20_node_mesh()->None:

    d2, c  = scaled_mesh(n_nodes  =  20  )
    idx  =  np.random.default_rng(1)
    psi =idx.uniform(-5.0,5.0,20)
    g  = idx.uniform(- 1e4, 1e4, 20)
    tmp3,a,v=poisson_jacobian(d2,c,psi,g)

    foo   =  sp.coo_matrix( ( v ,  (tmp3 ,  a )  ) ,   shape  =  (  20 ,   20 )  ).toarray( )

    mm =  1e-6
    for ii in range(20):
        u = psi.copy()
        d = psi.copy()


        u[ii] +=  mm


        d[ii]-=mm
        rr =  (
            poisson_residual(d2, c, u, g)
            - poisson_residual(d2, c, d, g)
        )/ (2.0 *  mm)
        np.testing.assert_allclose(
            foo[:,ii],rr,rtol =1e-5,atol=1e-5
        )



def test_jacobian_is_tridiagonal()  -> None:
    f, x= scaled_mesh(n_nodes =15)
    mm, b, _  =  poisson_jacobian(f, x, np.zeros(15), np.zeros(15))
    assert np.all(np.abs(mm-b)<=1)

def test_jacobian_diagonal_is_strictly_positive()->None :

    m, c = scaled_mesh(n_nodes =15)
    psi =  np.linspace(- 10.0, 10.0, 15)
    h, cnt, flag = poisson_jacobian(m, c, psi, np.zeros(15))
    rr=flag[h ==cnt]
    assert np.all(rr> 0.0)

def test_jacobian_off_diagonals_are_negative()->None :
    d, i = scaled_mesh(n_nodes=  15)
    res2, r2, row = poisson_jacobian(d, i, np.zeros(15), np.zeros(15))
    assert  np.all ( row[res2   !=  r2  ]   < 0.0  )



def  test_jacobian_is_symmetric()   -> None  :

    j,s =scaled_mesh(n_nodes =15)

    psi   =  np.linspace(  -   3.0 ,
              3.0 ,
           15)
    z,  i ,  a  =  poisson_jacobian(j,   s,   psi , np.zeros( 15 )  );  v=sp.coo_matrix((a,(z,i)),shape=(15,15)).toarray()
    np.testing.assert_allclose(v, v.T, rtol =1e-14)

def test_jacobian_is_diagonally_dominant() ->None:
    h, bar = scaled_mesh(n_nodes  =15)
    r, a, b =poisson_jacobian(h, bar, np.zeros(15), np.zeros(15))
    stuff =sp.coo_matrix((b, (r, a)), shape = (15, 15)).toarray()
    cc =  np.abs(  np.diag( stuff) )
    s2 = np.abs(stuff).sum(axis= 1)-cc
    assert  np.all (  cc >=  s2 )
def test_jacobian_laplacian_block_matches_the_uniform_mesh_stencil()->None :
    a,   dat  = scaled_mesh( n_nodes =  11 )
    mm   =  a[0  ]
    d, v, i  =  poisson_jacobian(a, dat, np.zeros(11), np.zeros(11))
    b  =   sp.coo_matrix( (i,   (d,  v )  ) ,   shape   =  (11 , 11 ) ).toarray( )

    m2=2.0*dat
    for t in range(1, 10):
        assert b[t, t -1]== pytest.approx(- 1.0/ mm, rel  =  1e-14)
        assert b[  t, t + 1]  ==  pytest.approx( -   1.0  /   mm ,   rel  =   1e-14 )
        assert b[t, t] ==pytest.approx(
            2.0  / mm  +  m2[t], rel=  1e-14
        )
def test_assemble_checks_scaling_state_at_entry()  -> None :
    h  = uniform_mesh_1d(MICRON, 11)
    ys  =ScaleFactors.for_silicon()
    psi = Field(np.zeros(11), 'V', ScalingState.PHYSICAL, Location.NODE)
    u=Field(np.zeros(11),'cm^-3',ScalingState.SCALED,Location.NODE)

    with pytest.raises(ValueError, match= "SCALED") :
        assemble_poisson(h.scaled(ys), psi, u)
def  test_assemble_rejects_edge_located_fields(  )  ->   None   :
    f  =  uniform_mesh_1d(  MICRON,  11 )
    m=ScaleFactors.for_silicon()
    psi =   Field(np.zeros(  10 ),   'V',  ScalingState.SCALED, Location.EDGE )
    j =  Field ( np.zeros( 11 ), 'cm^-3' ,   ScalingState.SCALED,  Location.NODE )
    with pytest.raises(ValueError, match=  "NODE"):
        assemble_poisson(f.scaled(  m),   psi,  j )


def test_assemble_rejects_a_field_of_the_wrong_length() -> None:
    z = uniform_mesh_1d(MICRON,11)
    a=ScaleFactors.for_silicon()

    psi=Field(np.zeros(9),'V',ScalingState.SCALED,Location.NODE)
    ys= Field(np.zeros(11), "cm^-3", ScalingState.SCALED, Location.NODE)

    with pytest.raises(ValueError,match ="length|nodes"):
        assemble_poisson(z.scaled(a),psi,ys)

def  test_assemble_rejects_a_charge_volume_of_the_wrong_length()  ->  None   :
    m2 = uniform_mesh_1d(MICRON,
                  11)
    t   =  ScaleFactors.for_silicon ()
    psi = Field(np.zeros(11),"V",ScalingState.SCALED,Location.NODE)
    flag =  Field(np.zeros(11), 'cm^-3', ScalingState.SCALED, Location.NODE)

    with pytest.raises(ValueError, match =  "charge_volume")  :
        assemble_poisson(m2.scaled(t), psi, flag, charge_volume =  np.ones(9))




def test_assemble_scales_the_mesh_by_the_debye_length()->None:
    z2=  uniform_mesh_1d(MICRON, 11)
    vals= ScaleFactors.for_silicon()

    psi=  Field(np.zeros(11), "V", ScalingState.SCALED, Location.NODE)
    c2 = Field(np.zeros(11),'cm^-3',ScalingState.SCALED,Location.NODE)
    g=assemble_poisson(z2.scaled(vals),psi,c2)
    d2 = poisson_residual(z2.h  / vals.x_0, z2.volume /  vals.x_0, np.zeros(11), np.zeros(11))

    np.testing.assert_allclose(g.residual, d2, rtol =1e-14)


def test_assemble_returns_a_square_system_of_the_right_size() ->None  :
    d = uniform_mesh_1d(MICRON, 11)
    g  = ScaleFactors.for_silicon()
    psi =Field(np.zeros(11), "V", ScalingState.SCALED, Location.NODE)
    z =Field(np.zeros(11),'cm^-3',ScalingState.SCALED,Location.NODE)
    y=assemble_poisson(d.scaled(g),psi,z)
    assert y.shape  == (11, 11)
    assert y.residual.shape  == (11, )


def test_residual_uses_the_quasi_fermi_potentials_in_the_densities( ) ->  None  :

    t,v=scaled_mesh(n_nodes =11)
    psi=np.full(11,2.0); c  = np.zeros(  11)
    d= np.full(11,2.0)

    b = poisson_residual(t, v, psi, c, d, d)
    np.testing.assert_allclose(b, 0.0, atol =  1e-15)


def test_quasi_fermi_defaults_to_zero_which_is_true_equilibrium() ->  None:
    a, k =  scaled_mesh(n_nodes=  11)
    psi =  np.linspace(  -  2.0 , 2.0,  11  )
    stuff=np.zeros(11)

    aa= poisson_residual(a, k, psi, stuff)
    info =poisson_residual(
        a,k,psi,stuff,np.zeros(11),np.zeros(11)
    )
    np.testing.assert_array_equal(aa,info)


def test_a_uniform_shift_of_psi_and_both_levels_leaves_the_residual_alone() -> None :
    y,  u  = scaled_mesh(n_nodes  =  11 )
    psi  =  np.linspace(-  1.0, 1.0, 11)
    z  =  np.full (11, 1e6 )

    f=poisson_residual(
        y,u,psi,z,np.zeros(11),np.zeros(11)
    )


    ret = poisson_residual(
        y,u,psi+38.7,z,np.full(11,38.7),np.full(11,38.7)
    )
    np.testing.assert_allclose(ret, f, rtol = 1e-12, atol = 1e-12)

def  test_jacobian_with_quasi_fermi_matches_complex_step( )   -> None :
    aa,  f =  scaled_mesh( n_nodes   =  20)
    v =   np.random.default_rng( 7)
    psi   =  v.uniform( - 6.0 , 6.0 ,   20)
    y=v.uniform(- 1e5, 1e5, 20)
    phi_n  =   v.uniform( -   3.0,
                   3.0,
                 20)
    phi_p = v.uniform(-  3.0, 3.0, 20)

    c,a,tt=poisson_jacobian(
        aa,f,psi,y,phi_n,phi_p
    )
    t   =  sp.coo_matrix( (  tt,  ( c,  a)  ) ,  shape   = (  20,   20 )  ).toarray(  )


    s  =  1e-30
    for val in range(20):

        buf  =  psi.astype ( np.complex128 )

        buf[ val]  +=   1j  *   s
        z = (
            poisson_residual(
                aa,f,buf,y,phi_n,phi_p
            ).imag
            / s
        )


        np.testing.assert_allclose(t[:, val], z, rtol =  1e-12, atol= 1e-12)


def test_an_insulator_node_contributes_no_charge_however_large_psi_gets( )  :


    bb= 5
    ii  =  np.full (bb   - 1 ,  0.5  )
    h=np.full(bb,0.5)
    h[  -  2  :  ]  =   0.0


    psi =np.array([0.0,10.0,100.0,500.0,800.0])
    r= np.zeros(bb)


    i =poisson_residual(ii,h,psi,r)


    assert np.all(np.isfinite(i)), (
        'an insulator node produced a non finite residual, so its carrier '
        f"term was inf * 0: {i}"
    )

    rr,obj,w=poisson_jacobian(ii,h,psi,r)

    assert  np.all( np.isfinite (  w  ) ), (
        f"an insulator node produced a non finite Jacobian entry: {w}"
    )


def test_the_insulator_rows_are_exactly_the_laplacian() :

    aa  =  5
    t =np.full(aa-1, 0.5)
    b=np.full(aa,0.5)
    b[- 2:]= 0.0


    psi = np.array(  [0.0,   10.0, 100.0,   500.0 ,  800.0]  );cc  =   np.zeros( aa  )
    g=poisson_residual(t,b,psi,cc)


    for res2 in(3,4):
        h   = 0.0

        if res2  >  0:


            h+= (psi[res2]-psi[res2- 1])/t[res2 -1]
        if  res2  <   aa - 1   :

            h -= (psi[res2 +1]- psi[res2]) / t[res2]
        assert g[res2]==pytest.approx(h,rel= 1e-14),(
            f"insulator node {res2} carries a charge term it should not"
        )
def test_complex_step_survives_the_insulator_mask() :
    m   =  5
    zz = np.full(m  - 1, 0.5)
    b = np.full(m, 0.5)
    b[-2:]= 0.0
    tmp3 =np.zeros(m)
    psi   =  np.array ([ 0.0 , 1.0,  2.0 , 3.0,   4.0 ]  )

    d,f,x =poisson_jacobian(zz,b,psi,tmp3)
    w   =  sp.coo_matrix((  x,  (  d, f )  ),   shape  =  ( m,  m  )).toarray()


    a = 1e-30
    for h in range(m):
        flag = psi.astype(np.complex128)
        flag[h] +=  1j * a

        cur  = (
            poisson_residual(zz, b, flag, tmp3).imag  /a
        )
        np.testing.assert_allclose(
            w[:, h], cur, rtol= 1e-12, atol=  1e-12
        )
