"""Tests for discretize/poisson.py.

Scaled units throughout: psi in units of V_T, densities in units of n_i,
lengths in units of the Debye length x_0. In those units Poisson is

    lap(psi) = -(p - n + N)

with no coefficient at all, which is the entire reason for the scaling.

The residual is defined with the sign that makes the Jacobian diagonal
positive and the off-diagonals negative, so the matrix is an M-matrix:

    F_i = (psi_i - psi_{i-1})/h_{i-1} - (psi_{i+1} - psi_i)/h_i
          - (p_i - n_i + N_i) * volume_i

The strongest test here is that uniform material at psi = asinh(N/2) gives an
exactly zero residual. The Laplacian vanishes because psi is flat, and the
charge term vanishes because that psi is the neutrality solution. Any sign
error in either half breaks it.
"""


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
'''One micron [cm].'''

def  scaled_mesh ( n_nodes  : int  =  21,   length  :  float  =  MICRON  )  ->  tuple :
    """A uniform mesh with its edge lengths and volumes in scaled units."""
    sca  = ScaleFactors.for_silicon (  )
    msh  =  uniform_mesh_1d(length, n_nodes)

    return msh.h/sca.x_0,msh.volume /sca.x_0


@pytest.mark.parametrize("doping_scaled",[-1e6,-1.0,0.0,1.0,1e6,1e8])



def test_residual_is_zero_for_uniform_material_at_equilibrium(doping_scaled: float,) -> None :
    """The single strongest check on the signs in this module.

    Flat psi kills the Laplacian, and psi = asinh(N/2) kills the charge term.
    If either sign is wrong the two no longer cancel.
    """
    aa,vol =scaled_mesh(); temp2 = np.full(aa.size  + 1, doping_scaled)
    psi  = np.full( aa.size  +   1, float(psi_equilibrium_scaled (doping_scaled ))  )
    blah  =poisson_residual(aa, vol, psi, temp2)
    np.testing.assert_allclose(blah, 0.0, atol = 1e-9 * max(1.0, abs(doping_scaled)))

def test_residual_is_nonzero_when_psi_is_off_equilibrium() ->None:
    """Guards the test above against passing for a trivial reason."""
    H, myvar = scaled_mesh()


    netDoping  =  np.full(  H.size +   1, 1e6 )
    psi  = np.full(H.size  + 1, float(psi_equilibrium_scaled(1e6))  + 0.1)


    assert np.max(  np.abs(  poisson_residual(  H, myvar,  psi,   netDoping  )  ))  >  1.0



def test_laplacian_of_a_linear_potential_vanishes_in_the_interior()   ->   None   :
    """A linear psi has zero second derivative, so only the charge term is left.

    The charge term has to be evaluated at the same psi, not at zero. n and p
    are exponentials of psi, so a linear psi still carries charge.
    """
    H, Volume  =scaled_mesh(n_nodes =21)
    psi =np.concatenate(([0.0],np.cumsum(H)))*3.0
    thing  =np.zeros(psi.size)

    temp2 = poisson_residual(H,Volume,psi,thing)
    cha= -(np.exp(- psi)-np.exp(psi)+thing)* Volume
    np.testing.assert_allclose(temp2[1:-1],cha[1:-1],atol =1e-14)


def test_boundary_rows_use_a_reflecting_condition()->None:
    """Nodes that are not contacts get homogeneous Neumann, per 01-physics.

    Node 0 has no left face, so its row carries only the right flux.
    """
    H,   dict  = scaled_mesh(  n_nodes   =   5  );  psi = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    NetDoping  =np.zeros(5)
    obj2= poisson_residual(H,dict,psi,NetDoping)
    tmp  = np.exp(  1.0)
    P0=np.exp(-1.0)
    Expected = - (0.0- 1.0)/ H[0]-(P0-tmp)*dict[0]
    assert obj2[0]  == pytest.approx(Expected, rel =  1e-14)




def test_charge_term_has_the_sign_that_pulls_psi_toward_neutrality()->None  :
    """Net donors must push psi positive, per the sign convention."""
    H, Volume  =   scaled_mesh (n_nodes = 11 )

    psi = np.zeros(11)

    don=poisson_residual(H,Volume,psi,np.full(11,1e6))
    Acceptors=poisson_residual(H,Volume,psi,np.full(11,-1e6))

    assert  np.all(don [ 1 :-  1  ]   <   0.0)
    assert np.all(Acceptors[1 :-1] >0.0)



def test_jacobian_matches_complex_step_differentiation()-> None:
    """Exact to machine precision, unlike finite differences.

    The residual is built from sums, quotients and exp, all analytic, so the
    complex step trick applies cleanly here. It does not suffer the
    cancellation that limits it for the Bernoulli function.
    """


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
    """The check named in phases/PHASE-1.md, kept alongside the exact one."""

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

    """docs/02-numerics.md: this is what makes equilibrium Poisson easy."""
    H, vol = scaled_mesh(n_nodes =15)
    psi =  np.linspace(- 10.0, 10.0, 15)
    dir, Cols, Values = poisson_jacobian(H, vol, psi, np.zeros(15))
    Diagonal=Values[dir ==Cols]
    assert np.all(Diagonal> 0.0)

def test_jacobian_off_diagonals_are_negative()->None :
    """Together with a positive diagonal this makes it an M-matrix."""
    H, Volume = scaled_mesh(n_nodes=  15)
    out2, list, valuues = poisson_jacobian(H, Volume, np.zeros(15), np.zeros(15))
    assert  np.all ( valuues[out2   !=  list  ]   < 0.0  )



def  test_jacobian_is_symmetric()   -> None  :

    '''Box integration of the Laplacian is symmetric, and the charge term is
    diagonal, so the whole matrix is.'''
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
    """On a uniform mesh the interior stencil is exactly (-1, 2, -1)/h."""
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
    """A physical psi here would be wrong by a factor of 38.7 and still solve."""
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
    '''The one array here that is not a Field, so nothing else checks it.

    Phase 5 hands a charge volume to every 2D solve, where it is what keeps
    carriers out of the oxide. A short one is a mesh mismatch, not a mask.
    '''
    mseh = uniform_mesh_1d(MICRON,
                  11)
    scaale   =  ScaleFactors.for_silicon ()
    psi = Field(np.zeros(11),"V",ScalingState.SCALED,Location.NODE)
    dping =  Field(np.zeros(11), 'cm^-3', ScalingState.SCALED, Location.NODE)

    with pytest.raises(ValueError, match =  "charge_volume")  :
        assemble_poisson(mseh.scaled(scaale), psi, dping, charge_volume =  np.ones(9))




def test_assemble_scales_the_mesh_by_the_debye_length()->None:
    """The mesh is in cm but the equation is in units of x_0.

    Forgetting this is a silent error of many orders of magnitude, since
    x_0 is 40.9 um at intrinsic doping and a device is 1 um across.
    """
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

    """n = exp(psi - phi_n), p = exp(phi_p - psi), not exp(+/- psi).

    Without this, a bias applied at a contact cannot reach the junction. The
    carrier densities are tied absolutely to psi, so the neutral bulk cannot
    shift its potential without changing p by exp(38.7) per volt. The applied
    bias piles up in a thin layer at the contact instead. Measured on a 1e16
    diode at -1 V: the whole volt drops across 0.05 um at the contact, with a
    2e5 V/cm field there, while the junction field stays at its zero bias
    value of 3.2e4 V/cm.
    """
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
    """The gauge freedom that lets a bias propagate into the bulk.

    Shifting psi and phi by the same amount leaves n and p unchanged, so a
    quasi-neutral region can sit at any potential its contact demands.
    """
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


    """A zero charge volume has to mean no carriers, not carriers times zero.

    `assemble_poisson` documents `charge_volume` as "zero in the oxide", which
    "turns those rows into the bare Laplacian an insulator wants". That holds
    right up until the potential in the insulator is large enough to overflow
    the exponential, and then `inf * 0` is `nan` rather than the zero the
    docstring promises. The nan is not loud: it lands on the insulator's own
    rows, which on a real device are the gate contact and are overwritten by
    the Dirichlet condition before the solve ever sees them.

    Scaled psi of 800 is exp overflow. That is 20.7 V, which is an ordinary
    gate bias on a thick oxide.
    """


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

    """Not merely finite. The charge term has to be gone, not small.

    Checked against the box integrated Laplacian written out by hand on the
    insulator rows, which is what those rows are supposed to reduce to once
    the carriers are absent.
    """
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
    """The mask must not break the dtype contract this module documents.

    `_carrier_densities` promises it "does not force a dtype, so a complex
    psi gives complex densities and complex step differentiation works through
    here", and the mask puts a real -inf into a complex array to get a density
    of exactly zero. The complex step check above this one runs on a device
    made of one material, where the mask is all true and never exercises that
    path, so it is checked here on a mesh that has an insulator in it.
    """
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
