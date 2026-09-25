from __future__ import annotations
import numpy as np

import  pytest

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
    sca  = ScaleFactors.for_silicon (  )
    msh  =  uniform_mesh_1d(length, n_nodes)

    return msh.h/sca.x_0,msh.volume /sca.x_0


@pytest.mark.parametrize("doping_scaled",[-1e6,-1.0,0.0,1.0,1e6,1e8])



def test_residual_is_zero_for_uniform_material_at_equilibrium(doping_scaled: float,) -> None :
    aa,vol =scaled_mesh(); temp2 = np.full(aa.size  + 1, doping_scaled)
    psi  = np.full( aa.size  +   1, float(psi_equilibrium_scaled (doping_scaled ))  )
    blah  =poisson_residual(aa, vol, psi, temp2)
    np.testing.assert_allclose(blah, 0.0, atol = 1e-9 * max(1.0, abs(doping_scaled)))

def test_residual_is_nonzero_when_psi_is_off_equilibrium() ->None:
    H, myvar = scaled_mesh()


    netDoping  =  np.full(  H.size +   1, 1e6 )
    psi  = np.full(H.size  + 1, float(psi_equilibrium_scaled(1e6))  + 0.1)


    assert np.max(  np.abs(  poisson_residual(  H, myvar,  psi,   netDoping  )  ))  >  1.0



def test_laplacian_of_a_linear_potential_vanishes_in_the_interior()   ->   None   :
    H, Volume  =scaled_mesh(n_nodes =21)
    psi =np.concatenate(([0.0],np.cumsum(H)))*3.0
    thing  =np.zeros(psi.size)

    temp2 = poisson_residual(H,Volume,psi,thing)
    cha= -(np.exp(- psi)-np.exp(psi)+thing)* Volume
    np.testing.assert_allclose(temp2[1:-1],cha[1:-1],atol =1e-14)


def test_boundary_rows_use_a_reflecting_condition()->None:
    H,   dict  = scaled_mesh(  n_nodes   =   5  );  psi = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    NetDoping  =np.zeros(5)
    obj2= poisson_residual(H,dict,psi,NetDoping)
    tmp  = np.exp(  1.0)
    P0=np.exp(-1.0)
    Expected = - (0.0- 1.0)/ H[0]-(P0-tmp)*dict[0]
    assert obj2[0]  == pytest.approx(Expected, rel =  1e-14)




def test_charge_term_has_the_sign_that_pulls_psi_toward_neutrality()->None  :
    H, Volume  =   scaled_mesh (n_nodes = 11 )

    psi = np.zeros(11)

    don=poisson_residual(H,Volume,psi,np.full(11,1e6))
    Acceptors=poisson_residual(H,Volume,psi,np.full(11,-1e6))

    assert  np.all(don [ 1 :-  1  ]   <   0.0)
    assert np.all(Acceptors[1 :-1] >0.0)



def test_jacobian_matches_complex_step_differentiation()-> None:


    H, temp2=scaled_mesh(n_nodes  = 20)

    vars =np.random.default_rng(0);  psi  =  vars.uniform(-  8.0, 8.0, 20)
    net_dopng =  vars.uniform(- 1e5, 1e5, 20)

    acc,tmp,chr =poisson_jacobian(H,temp2,psi,net_dopng); Assembled = sp.coo_matrix((chr, (acc, tmp)), shape =(20, 20)).toarray()
    ste=1e-30
    for col in range(20):
        val= psi.astype(np.complex128)
        val[ col  ]   += 1j   *   ste
        tmp2  =  poisson_residual ( H,  temp2, val ,   net_dopng ).imag /  ste
        np.testing.assert_allclose(Assembled[:, col], tmp2, rtol =  1e-12, atol  =  1e-12)
def test_jacobian_matches_finite_difference_on_a_20_node_mesh()->None:

    H, vol  = scaled_mesh(n_nodes  =  20  )
    cnt  =  np.random.default_rng(1)
    psi =cnt.uniform(-5.0,5.0,20)
    netdoping  = cnt.uniform(- 1e4, 1e4, 20)
    arr,input,val=poisson_jacobian(H,vol,psi,netdoping)

    bin   =  sp.coo_matrix( ( val ,  (arr ,  input )  ) ,   shape  =  (  20 ,   20 )  ).toarray( )

    Delta =  1e-6
    for col in range(20):
        Forward = psi.copy()
        min = psi.copy()


        Forward[col] +=  Delta


        min[col]-=Delta
        deriavtive =  (
            poisson_residual(H, vol, Forward, netdoping)
            - poisson_residual(H, vol, min, netdoping)
        )/ (2.0 *  Delta)
        np.testing.assert_allclose(
            bin[:,col],deriavtive,rtol =1e-5,atol=1e-5
        )



def test_jacobian_is_tridiagonal()  -> None:
    H, vol= scaled_mesh(n_nodes =15)
    tmp2, cools, _  =  poisson_jacobian(H, vol, np.zeros(15), np.zeros(15))
    assert np.all(np.abs(tmp2-cools)<=1)

def test_jacobian_diagonal_is_strictly_positive()->None :

    H, vol = scaled_mesh(n_nodes =15)
    psi =  np.linspace(- 10.0, 10.0, 15)
    dir, Cols, Values = poisson_jacobian(H, vol, psi, np.zeros(15))
    Diagonal=Values[dir ==Cols]
    assert np.all(Diagonal> 0.0)

def test_jacobian_off_diagonals_are_negative()->None :
    H, Volume = scaled_mesh(n_nodes=  15)
    out2, list, valuues = poisson_jacobian(H, Volume, np.zeros(15), np.zeros(15))
    assert  np.all ( valuues[out2   !=  list  ]   < 0.0  )



def  test_jacobian_is_symmetric()   -> None  :

    object,round =scaled_mesh(n_nodes =15)

    psi   =  np.linspace(  -   3.0 ,
              3.0 ,
           15)
    rwos,  col ,  max  =  poisson_jacobian(object,   round,   psi , np.zeros( 15 )  );  chr=sp.coo_matrix((max,(rwos,col)),shape=(15,15)).toarray()
    np.testing.assert_allclose(chr, chr.T, rtol =1e-14)

def test_jacobian_is_diagonally_dominant() ->None:
    dict, vol = scaled_mesh(n_nodes  =15)
    rws, col, next =poisson_jacobian(dict, vol, np.zeros(15), np.zeros(15))
    mat =sp.coo_matrix((next, (rws, col)), shape = (15, 15)).toarray()
    diagnal =  np.abs(  np.diag( mat) )
    yy = np.abs(mat).sum(axis= 1)-diagnal
    assert  np.all (  diagnal >=  yy )
def test_jacobian_laplacian_block_matches_the_uniform_mesh_stencil()->None :
    hh,   vol  = scaled_mesh( n_nodes =  11 )
    Spacing   =  hh[0  ]
    row, Cols, round  =  poisson_jacobian(hh, vol, np.zeros(11), np.zeros(11))
    Matrix  =   sp.coo_matrix( (round,   (row,  Cols )  ) ,   shape   =  (11 , 11 ) ).toarray( )

    charge_digaonal=2.0*vol
    for ii in range(1, 10):
        assert Matrix[ii, ii -1]== pytest.approx(- 1.0/ Spacing, rel  =  1e-14)
        assert Matrix[  ii, ii + 1]  ==  pytest.approx( -   1.0  /   Spacing ,   rel  =   1e-14 )
        assert Matrix[ii, ii] ==pytest.approx(
            2.0  / Spacing  +  charge_digaonal[ii], rel=  1e-14
        )
def test_assemble_checks_scaling_state_at_entry()  -> None :
    mes  = uniform_mesh_1d(MICRON, 11)
    dat  =ScaleFactors.for_silicon()
    psi = Field(np.zeros(11), 'V', ScalingState.PHYSICAL, Location.NODE)
    dop=Field(np.zeros(11),'cm^-3',ScalingState.SCALED,Location.NODE)

    with pytest.raises(ValueError, match= "SCALED") :
        assemble_poisson(mes.scaled(dat), psi, dop)
def  test_assemble_rejects_edge_located_fields(  )  ->   None   :
    msh  =  uniform_mesh_1d(  MICRON,  11 )
    sca=ScaleFactors.for_silicon()
    psi =   Field(np.zeros(  10 ),   'V',  ScalingState.SCALED, Location.EDGE )
    Doping =  Field ( np.zeros( 11 ), 'cm^-3' ,   ScalingState.SCALED,  Location.NODE )
    with pytest.raises(ValueError, match=  "NODE"):
        assemble_poisson(msh.scaled(  sca),   psi,  Doping )


def test_assemble_rejects_a_field_of_the_wrong_length() -> None:
    mes = uniform_mesh_1d(MICRON,11)
    Scale=ScaleFactors.for_silicon()

    psi=Field(np.zeros(9),'V',ScalingState.SCALED,Location.NODE)
    any= Field(np.zeros(11), "cm^-3", ScalingState.SCALED, Location.NODE)

    with pytest.raises(ValueError,match ="length|nodes"):
        assemble_poisson(mes.scaled(Scale),psi,any)

def  test_assemble_rejects_a_charge_volume_of_the_wrong_length()  ->  None   :
    mseh = uniform_mesh_1d(MICRON,
                  11)
    scaale   =  ScaleFactors.for_silicon ()
    psi = Field(np.zeros(11),"V",ScalingState.SCALED,Location.NODE)
    dping =  Field(np.zeros(11), 'cm^-3', ScalingState.SCALED, Location.NODE)

    with pytest.raises(ValueError, match =  "charge_volume")  :
        assemble_poisson(mseh.scaled(scaale), psi, dping, charge_volume =  np.ones(9))




def test_assemble_scales_the_mesh_by_the_debye_length()->None:
    yy=  uniform_mesh_1d(MICRON, 11)
    sca= ScaleFactors.for_silicon()

    psi=  Field(np.zeros(11), "V", ScalingState.SCALED, Location.NODE)
    Doping = Field(np.zeros(11),'cm^-3',ScalingState.SCALED,Location.NODE)
    Assembly=assemble_poisson(yy.scaled(sca),psi,Doping)
    temp = poisson_residual(yy.h  / sca.x_0, yy.volume /  sca.x_0, np.zeros(11), np.zeros(11))

    np.testing.assert_allclose(Assembly.residual, temp, rtol =1e-14)


def test_assemble_returns_a_square_system_of_the_right_size() ->None  :
    Mesh = uniform_mesh_1d(MICRON, 11)
    scaale  = ScaleFactors.for_silicon()
    psi =Field(np.zeros(11), "V", ScalingState.SCALED, Location.NODE)
    Doping =Field(np.zeros(11),'cm^-3',ScalingState.SCALED,Location.NODE)
    tmp=assemble_poisson(Mesh.scaled(scaale),psi,Doping)
    assert tmp.shape  == (11, 11)
    assert tmp.residual.shape  == (11, )


def test_residual_uses_the_quasi_fermi_potentials_in_the_densities( ) ->  None  :

    lst,vloume=scaled_mesh(n_nodes =11)
    psi=np.full(11,2.0); nd  = np.zeros(  11)
    phii= np.full(11,2.0)

    tmp = poisson_residual(lst, vloume, psi, nd, phii, phii)
    np.testing.assert_allclose(tmp, 0.0, atol =  1e-15)


def test_quasi_fermi_defaults_to_zero_which_is_true_equilibrium() ->  None:
    H, vol =  scaled_mesh(n_nodes=  11)
    psi =  np.linspace(  -  2.0 , 2.0,  11  )
    netDoping=np.zeros(11)

    next= poisson_residual(H, vol, psi, netDoping)
    wtih_zero =poisson_residual(
        H,vol,psi,netDoping,np.zeros(11),np.zeros(11)
    )
    np.testing.assert_array_equal(next,wtih_zero)


def test_a_uniform_shift_of_psi_and_both_levels_leaves_the_residual_alone() -> None :
    H,  Volume  = scaled_mesh(n_nodes  =  11 )
    psi  =  np.linspace(-  1.0, 1.0, 11)
    net  =  np.full (11, 1e6 )

    Base=poisson_residual(
        H,Volume,psi,net,np.zeros(11),np.zeros(11)
    )


    sihfted = poisson_residual(
        H,Volume,psi+38.7,net,np.full(11,38.7),np.full(11,38.7)
    )
    np.testing.assert_allclose(sihfted, Base, rtol = 1e-12, atol = 1e-12)

def  test_jacobian_with_quasi_fermi_matches_complex_step( )   -> None :
    H,  Volume =  scaled_mesh( n_nodes   =  20)
    rngg =   np.random.default_rng( 7)
    psi   =  rngg.uniform( - 6.0 , 6.0 ,   20)
    tmp=rngg.uniform(- 1e5, 1e5, 20)
    phi_n  =   rngg.uniform( -   3.0,
                   3.0,
                 20)
    phi_p = rngg.uniform(-  3.0, 3.0, 20)

    row,Cols,Values=poisson_jacobian(
        H,Volume,psi,tmp,phi_n,phi_p
    )
    ass   =  sp.coo_matrix( (  Values,  ( row,  Cols)  ) ,  shape   = (  20,   20 )  ).toarray(  )


    Step  =  1e-30
    for Column in range(20):

        Perturbed  =  psi.astype ( np.complex128 )

        Perturbed[ Column]  +=   1j  *   Step
        bar = (
            poisson_residual(
                H,Volume,Perturbed,tmp,phi_n,phi_p
            ).imag
            / Step
        )


        np.testing.assert_allclose(ass[:, Column], bar, rtol =  1e-12, atol= 1e-12)


def test_an_insulator_node_contributes_no_charge_however_large_psi_gets( )  :


    nnodes= 5
    hh  =  np.full (nnodes   - 1 ,  0.5  )
    voolume=np.full(nnodes,0.5)
    voolume[  -  2  :  ]  =   0.0


    psi =np.array([0.0,10.0,100.0,500.0,800.0])
    sum= np.zeros(nnodes)


    residuual =poisson_residual(hh,voolume,psi,sum)


    assert np.all(np.isfinite(residuual)), (
        'an insulator node produced a non finite residual, so its carrier '
        f"term was inf * 0: {residuual}"
    )

    obj2,col,val=poisson_jacobian(hh,voolume,psi,sum)

    assert  np.all( np.isfinite (  val  ) ), (
        f"an insulator node produced a non finite Jacobian entry: {val}"
    )


def test_the_insulator_rows_are_exactly_the_laplacian() :

    n_noddes  =  5
    hh =np.full(n_noddes-1, 0.5)
    voluume=np.full(n_noddes,0.5)
    voluume[- 2:]= 0.0


    psi = np.array(  [0.0,   10.0, 100.0,   500.0 ,  800.0]  );nd  =   np.zeros( n_noddes  )
    Residual=poisson_residual(hh,voluume,psi,nd)


    for tmp in(3,4):
        format   = 0.0

        if tmp  >  0:


            format+= (psi[tmp]-psi[tmp- 1])/hh[tmp -1]
        if  tmp  <   n_noddes - 1   :

            format -= (psi[tmp +1]- psi[tmp]) / hh[tmp]
        assert Residual[tmp]==pytest.approx(format,rel= 1e-14),(
            f"insulator node {tmp} carries a charge term it should not"
        )
def test_complex_step_survives_the_insulator_mask() :
    n_noddes   =  5
    hh = np.full(n_noddes  - 1, 0.5)
    Volume = np.full(n_noddes, 0.5)
    Volume[-2:]= 0.0
    str =np.zeros(n_noddes)
    psi   =  np.array ([ 0.0 , 1.0,  2.0 , 3.0,   4.0 ]  )

    vals,col,max =poisson_jacobian(hh,Volume,psi,str)
    ass   =  sp.coo_matrix((  max,  (  vals, col )  ),   shape  =  ( n_noddes,  n_noddes  )).toarray()


    iter = 1e-30
    for tmp2 in range(n_noddes):
        per = psi.astype(np.complex128)
        per[tmp2] +=  1j * iter

        item2  = (
            poisson_residual(hh, Volume, per, str).imag  /iter
        )
        np.testing.assert_allclose(
            ass[:, tmp2], item2, rtol= 1e-12, atol=  1e-12
        )
