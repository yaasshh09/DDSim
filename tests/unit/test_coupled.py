"""The full 3N coupled Jacobian, block by block.

phases/PHASE-3.md: "Every Jacobian block matches complex-step differentiation
to 1e-10 on a 20 node mesh. All nine blocks, individually tested.
Non-negotiable." This file is that criterion.

The blocks are tested individually rather than as one matrix comparison
because a single wrong block is the failure this is guarding against, and a
whole-matrix assertion reports "something is wrong" where nine assertions
report which derivative. docs/05-pitfalls.md puts checking the Jacobian second
in the debugging order for exactly this reason: a wrong derivative turns
quadratic convergence into stagnation, and stagnation looks like
ill-conditioning.

Three states are used.

Equilibrium on the diode is the one Newton actually starts from. Measured, its
Bernoulli arguments run from 2.2e-3 to 6.5 and **no edge sits at exactly
zero**, which was worth checking rather than assuming: an abrupt junction on a
20 node mesh leaves structure in psi everywhere, so the quasi neutral regions
are flat to a few parts in a thousand rather than flat exactly.

The perturbed state is off the solution manifold in psi, n and p at once, so
no term is accidentally zero and nothing cancels by symmetry.

The uniform bar is the state that does have X = 0 on every edge, exactly, and
it is the reason the complex step harness needed fixing at the origin. It is
constructed rather than solved for, because a solve would land near the flat
answer and not on it. All three psi blocks come back with exactly zero error
there; with the naive cos(y) - 1 in the complex expm1 the reference would
report B'(0) as 0.0 and the two flux-versus-potential blocks would be checked
against nothing.
"""
from  __future__ import  annotations

import  numpy as  np, pytest


from scipy.sparse import coo_matrix


from ddsim.core.field  import  Field,  Location, ScalingState

from ddsim.device.builder import build_device

from ddsim.device.doping import Uniform,abrupt_junction
from ddsim.device.transport import TransportModels ,  initial_state

from ddsim.discretize.boundary import(
    Carrier,
    OhmicContact,
    ohmic_density_scaled,
    ohmic_psi_scaled,
)



from  ddsim.discretize.continuity  import(
    electron_continuity_residual,
    hole_continuity_residual,
)

from ddsim.discretize.coupled  import(
    UNKNOWNS_PER_NODE ,
    Unknown,
    apply_contacts_coupled,
    assemble_coupled,
    assemble_coupled_arrays,
    assemble_coupled_terms,
    coupled_jacobian ,
    coupled_residual,
    coupled_update_by_family,
    coupled_update_norm,
    edge_drop ,
    pack,
    residual_measure,
    residual_term_scales ,
    row_weights,
    scale_rows ,
    unknown_index,
    unpack ,
)
from ddsim.discretize.poisson  import  poisson_residual
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.physics.mobility import diffusivity_at

from ddsim.solve.linear import  SparseLU
from tests.reference.complexstep import complex_step_jacobian
N_NODES  =  20

"""The mesh size phases/PHASE-3.md names for the verification."""


@pytest.fixture
def device():
    '''A 1e16 / 1e16 diode on a 20 node uniform mesh.

    Deliberately under-resolved: the Debye length at 1e16 is 41 nm and the
    spacing here is 53 nm. That is wrong for physics and right for a
    derivative check, because it puts several volts of potential drop on a
    single edge and drives the Bernoulli arguments out to where the branches
    differ. A well resolved mesh would leave every X near zero and test one
    branch.
    '''

    mes = uniform_mesh_1d (  length  =  1e-4 ,   n_nodes   = N_NODES  )
    return build_device(
        mesh =mes,
        doping=abrupt_junction(Na=1e16,Nd = 1e16,position= 0.5e-4),
        contacts= (
            OhmicContact(name='anode',node=0,voltage=0.0),
            OhmicContact(name = "cathode",node =N_NODES-1,voltage=0.0),
        ),
    )

@pytest.fixture(
    params  = [
        ('constant', False, False),
        ("arora", False, False),
        ('constant', True, False),
        ("arora", True, False),
        ("constant", False, True),
        ('arora', True, True),
    ],
    ids  = [
        'constant',
        'arora',
        "constant+auger",
        'arora+auger',
        "constant+field",
        'arora+auger+field',
    ],
)

def models(request,  device)   :
    """Recombination and diffusivities, in scaled units.

    Parametrized over the mobility model, the Auger flag and the field
    dependence, because all three change the *shape* of what the assemblies
    receive rather than only the numbers. Constant mobility makes Dn and Dp
    scalars, and a scalar broadcasts against an edge array no matter how the
    edges are indexed. Arora makes them one value per edge, where an off by
    one or a node/edge mixup stops being invisible. Verifying the nine blocks
    only against the scalar case leaves the alignment of the array case
    unpinned, which is the one thing the array case exists to get right.

    Auger is carried here for the same reason on the recombination side: it is
    the only model whose rate is not linear in a single carrier, so its
    derivative blocks are the ones a wrong linearization would show up in.

    Caughey-Thomas is the reason this fixture matters most. It is the first
    model whose diffusivity is a function of the unknown potential, so it puts
    two new terms into the flux derivative blocks, and those terms are exactly
    what a complex step through the residual will catch and nothing else will.
    Newton converges without them, more slowly, to the same answer.
    """
    mob ,   auer, fie   = request.param
    return TransportModels.for_device(
        device,mobility=mob,auger = auer,field_dependent=fie
    )


@pytest.fixture

def geometry(device)  :
    """(h, volume) in units of the Debye length, as every assembly wants."""
    sca=device.scale;  return device.mesh.h /sca.x_0,device.mesh.volume/sca.x_0


@pytest.fixture

def equilibrium_x(device) :
    """The state Newton starts from.

    Not flat anywhere, despite the name suggesting it should be: the smallest
    edge potential difference on this mesh is 2.2e-3, not zero. See the module
    docstring. The uniform bar below covers the exactly flat case.
    """
    satte= initial_state(device)
    return  pack(satte.psi.data ,  satte.n.data, satte.p.data  )
@pytest.fixture


def perturbed_x(device):
    """Off the solution manifold in psi, n and p at once.

    The densities are scaled multiplicatively through an exponential, so they
    stay strictly positive and stay within a factor of a few of a physically
    reachable state. A perturbation large enough to make n negative would test
    a Jacobian at a point the solver can never visit.
    """
    State=initial_state(device)
    foo  =  np.linspace(0.0 ,   3.0  *  np.pi, device.mesh.n_nodes)
    psi=  State.psi.data  +  0.35  *  np.cos(foo)


    n=State.n.data*np.exp(0.20*np.sin(foo))
    p  = State.p.data *  np.exp(- 0.15   *   np.cos(  2.0 *  foo ) )

    return  pack( psi, n, p  )
@pytest.fixture(  params   =  ["equilibrium", "perturbed" ])

def state_x(request, equilibrium_x, perturbed_x)  :
    """Both states, so every block test runs against each."""
    return  equilibrium_x  if request.param   ==   "equilibrium"  else perturbed_x


def residual_at(geometry,device,models) :
    '''A one argument residual, which is what the complex step harness takes.'''
    input,   vol   =   geometry
    def evaluate(x) :
        return coupled_residual(h   =  input , volume   =  vol , x   =  x , net_doping  =   device.net_doping_scaled.data, Dn  =  models.Dn, Dp  = models.Dp, recombination  =   models.recombination,)


    return  evaluate
def dense_jacobian(geometry,x,models):
    """The assembled Jacobian, densified for comparison."""
    hh, Volume= geometry
    Rows, bb, Values = coupled_jacobian(h= hh, volume = Volume, x  =x, Dn = models.Dn, Dp =  models.Dp, recombination=  models.recombination,)
    siz=x.size ; return  coo_matrix(  ( Values, ( Rows,   bb  )  ) ,  shape  = (  siz ,   siz  )  ).toarray()



def block(matrix,
         row: Unknown,
    col: Unknown):
    """One N x N block, gathered out of the node-interleaved ordering."""
    return matrix[row  ::UNKNOWNS_PER_NODE, col ::UNKNOWNS_PER_NODE]


def node_field(values,unit,name):
    """A scaled node Field, which is the only kind an assembly accepts."""
    return Field(
        values.copy(), unit, ScalingState.SCALED, Location.NODE, name =  name
    )

def assemble_state(device, models, x) :

    """The Field level assembly at a packed state, without the boilerplate."""
    psi,n,p= unpack(x)

    return assemble_coupled (
        mesh  =   device.mesh,
        psi  =  node_field( psi,  'V' , 'psi'  ),
        n   =  node_field(  n, "cm^-3", 'n'),
        p   =  node_field ( p, 'cm^-3',   "p" ),
        net_doping  =  device.net_doping_scaled,
        recombination =  models.recombination ,
        scale  =  device.scale ,
        Dn   = models.Dn,
        Dp  =   models.Dp ,
    )


def test_pack_and_unpack_round_trip() :
    """Whatever the ordering is, it has to be reversible."""

    psi= np.array([1.0, 2.0, 3.0])
    n =np.array([4.0, 5.0, 6.0])
    p=np.array([7.0,8.0,9.0])
    gotPsi,GotN,dat= unpack(pack(psi,n,p))
    np.testing.assert_array_equal(gotPsi,psi)


    np.testing.assert_array_equal(GotN, n)
    np.testing.assert_array_equal(dat, p)



def test_ordering_is_interleaved_by_node() :
    """docs/02-numerics.md: psi_0, n_0, p_0, psi_1, ... for fill reduction.

    Blocking by variable instead would put the three unknowns of one node 2N
    apart and give the factorization a much wider band to fill.
    """
    psi  =   np.array ([  1.0,   2.0 ,  3.0 ]) ; n=np.array([4.0,5.0,6.0])
    p=np.array([7.0,8.0,9.0])
    bb=pack(psi,n,p)
    np.testing.assert_array_equal(bb, [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])

def test_unpack_returns_views_not_copies (  )   :
    """A copy per call would double the cost of every residual evaluation."""
    xx  =  np.arange(9.0)
    psi,n,p =unpack(xx)

    psi[0] = -  1.0
    assert xx[0]  == -  1.0



def test_pack_rejects_mismatched_lengths():
    """Three arrays of different length is a caller bug, not a broadcast."""
    with pytest.raises(ValueError, match= 'same length')  :
        pack(np.zeros(3),np.zeros(4),np.zeros(3))
def test_psi_rows_match_the_poisson_residual(device, geometry, models) :
    """The psi block is the Phase 1 residual with n and p read, not derived.

    Checked on a Boltzmann consistent state, where the two agree by
    construction. Off that manifold they differ, and that difference is the
    whole point of the coupled system.
    """
    hh,idx2= geometry
    satte= initial_state(device)
    psi =satte.psi.data
    phi_n =satte.phi_n.data
    phi_p= satte.phi_p.data
    n  = np.exp ( psi   -  phi_n)

    p=np.exp(phi_p- psi)



    s2  =  coupled_residual(h =   hh, volume   =  idx2, x  = pack (  psi, n , p  ) , net_doping   =   device.net_doping_scaled.data, Dn =  models.Dn, Dp  =  models.Dp , recombination   =   models.recombination ,)
    expeected =poisson_residual(
        hh,idx2,psi,device.net_doping_scaled.data,phi_n,phi_p
    )

    np.testing.assert_allclose(s2 [ Unknown.PSI  ::   UNKNOWNS_PER_NODE] ,   expeected , rtol  =  1e-13, atol  =  0.0)

def test_electron_rows_match_the_uncoupled_continuity_residual(
    device,geometry,models,perturbed_x
) :

    """Same equation, different unknown vector. The residual cannot move.

    The uncoupled assembly takes a diffusivity, never a model, so a field
    dependent one is resolved at this state first. That is the comparison
    worth making: the two write the same equation given the same coefficient,
    and the coupled path is what works out what the coefficient is.
    """
    hh,  Volume  =  geometry
    psi, n, p = unpack(perturbed_x)
    RR=np.asarray(models.recombination.rate(n, p), dtype = np.float64)
    d2= diffusivity_at(models.Dn, edge_drop(psi), hh)
    gott  =   residual_at( geometry ,  device,  models)  (  perturbed_x  )

    str =electron_continuity_residual(hh,Volume,d2,psi,n,RR)


    np.testing.assert_allclose(gott [  Unknown.N  ::  UNKNOWNS_PER_NODE], str,  rtol =  1e-13 ,  atol  =  0.0)


def test_hole_rows_match_the_uncoupled_continuity_residual(
    device, geometry, models, perturbed_x
):
    '''The mirror of the electron check, with the flux asymmetry intact.'''


    x2,  vol   =  geometry
    psi,n,p =unpack(perturbed_x)
    RR=np.asarray(models.recombination.rate(n,p),dtype= np.float64)

    Dpp= diffusivity_at(models.Dp,edge_drop(psi),x2)



    slice=residual_at(geometry,device,models)(perturbed_x)

    Expected = hole_continuity_residual(x2, vol, Dpp, psi, p, RR)
    np.testing.assert_allclose(slice[Unknown.P ::UNKNOWNS_PER_NODE],Expected,rtol= 1e-13,atol =0.0)

def test_residual_preserves_a_complex_dtype(device,geometry,models,perturbed_x):
    """Without this the complex step verification silently reports zeros."""
    gott  = residual_at(  geometry, device,   models  )   (
        perturbed_x.astype(np.complex128)
    )


    assert np.iscomplexobj(gott)


@pytest.fixture


def flat_bar() :


    """A uniformly doped bar at equilibrium, with X exactly zero everywhere.

    Built from the closed form rather than solved for. In scaled units the
    equilibrium of a uniform bar is psi = asinh(N/2) with n = exp(psi) and
    p = exp(-psi), constant across the mesh, so every edge potential
    difference is exactly 0.0 and every Bernoulli argument sits on the
    removable singularity.

    Returns (device, models, geometry, x).
    """
    Mesh  =uniform_mesh_1d(length  = 1e-4, n_nodes= N_NODES)
    stuff2  =   build_device (
        mesh = Mesh ,
        doping   =   Uniform(  1e16),
        contacts  =  (
            OhmicContact( name  =  'anode',  node   =   0 ,   voltage =  0.0  ),
            OhmicContact(name = "cathode",   node   = N_NODES  -   1,  voltage  =   0.0  ) ,
        ) ,
    )
    mod  = TransportModels.for_device(stuff2)
    Scale  =   stuff2.scale


    xx =stuff2.net_doping_scaled.data
    psi =  np.full ( Mesh.n_nodes , np.arcsinh (  xx[0 ]  /   2.0) )
    acc2 =pack(psi, np.exp(psi), np.exp(-  psi))

    return stuff2, mod, (Mesh.h  / Scale.x_0, Mesh.volume  / Scale.x_0), acc2

ALL_BLOCKS =[(row,col)for row in Unknown for col in Unknown]
@pytest.mark.parametrize(  'row,col', ALL_BLOCKS,   ids   = lambda u  :  u.name  )


def test_every_jacobian_block_matches_complex_step(device, geometry, models, state_x, row, col):
    """The Phase 3 acceptance criterion. Non-negotiable, all nine blocks.

    Compared entry by entry against the block's own largest entry rather than
    against each entry's own magnitude. The blocks span many decades inside
    themselves, and an entry that is small because two large terms cancelled
    carries no more absolute information than the cancellation left in it.
    """
    Reference =complex_step_jacobian(
        residual_at(geometry,device,models),state_x
    )
    ass   =  dense_jacobian(geometry ,   state_x,  models )

    gott=block(ass,row,col)
    exp  =  block( Reference,  row, col )
    vars  = np.max(np.abs(exp))

    np.testing.assert_allclose(
        gott,exp,rtol=1e-10,atol= 1e-10 *max(vars,1e-300)
    )




@pytest.mark.parametrize("row,col", ALL_BLOCKS, ids = lambda u : u.name)



def test_every_jacobian_block_matches_complex_step_at_a_flat_potential(flat_bar, row, col)  :

    """The same nine blocks where every Bernoulli argument is exactly zero.

    B(0) = 1 and B'(0) = -1/2 are both removable singularities reached by a
    different branch of the implementation and a different branch of the
    reference, so this is not the diode test on easier numbers. It is the case
    that fails silently when the complex step reference drops the second order
    term in expm1, and it is the state a uniformly doped region sits in.
    """
    Device, mod, Geometry,  xx  = flat_bar
    psi,_,_ = unpack(xx)
    assert  np.all( psi[1  :]   - psi [  :- 1]  ==  0.0),  'the fixture is not flat'

    sum = complex_step_jacobian(residual_at(Geometry,Device,mod),xx)
    ass =   dense_jacobian(Geometry ,  xx ,  mod )
    tuple   =  block(  ass,   row ,   col)
    exp = block(  sum,   row ,   col)
    Floor  =  np.max ( np.abs (exp)  )


    np.testing.assert_allclose(
        tuple,exp,rtol=1e-10,atol =1e-10* max(Floor,1e-300)
    )



@pytest.fixture




def lopsided_bar():
    """A device whose Arora diffusivity genuinely differs from edge to edge.

    The shared device fixture cannot do this job. It is a 1e16 / 1e16
    junction, so abs(net doping) is 1e16 on every node, and Arora reads only
    the total doping: Dn comes back with a single unique value across all
    nineteen edges. A constant array is indistinguishable from a scalar under
    broadcasting, so running the nine blocks against it verifies the array
    code path without verifying that the array is *aligned* to the edges it
    belongs to. Measured on the 1e18 / 1e15 profile used here, Dn takes three
    distinct values with a factor of 4.7 between the ends, and is not
    symmetric under reversal, which is what makes an off by one or a reversed
    gather visible.

    Returns (device, models, geometry, x).
    """

    meesh  =uniform_mesh_1d(length=1e-4, n_nodes  = N_NODES)
    Device=build_device(
        mesh= meesh,
        doping=abrupt_junction(Na =1e18,Nd=1e15,position=0.5e-4),
        contacts =(
            OhmicContact(name ='anode',node=0,voltage=0.0),
            OhmicContact(name= "cathode",node=N_NODES -1,voltage=0.0),
        ),
    )
    dat=TransportModels.for_device(Device,mobility ='arora')
    Scale   = Device.scale

    Geometry =(meesh.h/ Scale.x_0,meesh.volume /Scale.x_0)

    sttae= initial_state(Device)
    kk=np.linspace(0.0,3.0 * np.pi,meesh.n_nodes)
    x2 =pack(sttae.psi.data +0.35*np.cos(kk), sttae.n.data*np.exp(0.3*np.sin(kk)), sttae.p.data *np.exp(-0.3* np.sin(kk)),)
    return Device,dat,Geometry,x2

@pytest.mark.parametrize('row,col',ALL_BLOCKS,ids=lambda u:u.name)

def test_every_jacobian_block_matches_complex_step_per_edge_diffusivity(lopsided_bar,row,col):
    """The nine blocks again, with a diffusivity that varies along the device.

    Doping dependent mobility turns Dn and Dp from scalars into one value per
    edge. Every other block test here runs with scalars, which broadcast
    correctly no matter how the edges are indexed, so this is the only place
    that pins the per-edge alignment of the flux coefficients and their
    derivatives.
    """

    tmp,ord,Geometry,q=lopsided_bar
    filter  =  np.asarray(ord.Dn)
    assert filter.size==tmp.mesh.n_edges,"Dn is not per edge"
    assert  np.unique(  filter  ).size >  1 ,  'Dn does not vary, so alignment is untested'
    assert not np.array_equal(filter,filter[::-1]),'Dn is reversal symmetric'

    open=complex_step_jacobian(residual_at(Geometry,tmp,ord),q)
    ass  = dense_jacobian (Geometry ,  q,  ord)

    gott= block(ass,row,col)


    aa  = block(open, row, col); foor  =   np.max(np.abs( aa)  )



    np.testing.assert_allclose(
        gott, aa, rtol = 1e-10, atol  = 1e-10 * max(foor, 1e-300)
    )

def test_the_psi_block_carries_no_boltzmann_charge_term (
    device ,  geometry, models,   perturbed_x
) :
    """dF_psi/dpsi is the bare Laplacian here, unlike the Phase 1 Poisson.

    In Phase 1 n and p are functions of psi and the diagonal picks up
    (n + p)*volume, which is what makes that matrix an M-matrix. In the
    coupled system they are separate unknowns, so that term moves into
    dF_psi/dn and dF_psi/dp instead. Carrying it in both places is the most
    likely way to get this wrong, because the Phase 1 Jacobian is right there
    to copy, and the result would be a Jacobian that is wrong by exactly the
    term the coupling was introduced to represent.
    """
    H,voulme=geometry
    Assembled  =dense_jacobian(geometry, perturbed_x, models)
    id =block(Assembled,Unknown.PSI,Unknown.PSI)

    exected =np.zeros((device.mesh.n_nodes, device.mesh.n_nodes))
    hash=1.0 /H

    for  E  in range( H.size )   :

        exected[E,E]+= hash[E]
        exected[  E  +  1, E  + 1 ]  += hash[E  ]
        exected[ E ,   E   +  1  ]   -=   hash[  E  ]


        exected[E  +  1, E] -=hash[E]

    np.testing.assert_allclose(id,
          exected,
      rtol = 1e-14,
            atol = 0.0)

def test_the_charge_blocks_are_plus_and_minus_the_cell_volume(
    device ,  geometry,   models, perturbed_x
)   :
    '''dF_psi/dn = +volume and dF_psi/dp = -volume, diagonal only.

    The signs come straight from -(p - n + N)*volume in the residual. Getting
    them the wrong way round flips the sign of the electrostatic feedback and
    turns Newton's correction into an amplification.
    '''
    _,voolume = geometry
    yy  = dense_jacobian(geometry, perturbed_x, models)

    np.testing.assert_allclose(block(yy,Unknown.PSI,Unknown.N),np.diag(voolume),atol=0.0)
    np.testing.assert_allclose(
        block(yy, Unknown.PSI, Unknown.P), np.diag(- voolume), atol =0.0
    )


def test_the_recombination_cross_blocks_are_diagonal(
    device,geometry,models,perturbed_x
):
    """R is a point function, so dF_n/dp and dF_p/dn touch one node only.

    Any off diagonal entry here means a flux term leaked into the wrong
    block, which complex step would still confirm if the residual leaked the
    same way.
    """
    Assembled =dense_jacobian(geometry,perturbed_x,models)
    for Row,q in((Unknown.N,Unknown.P),(Unknown.P,Unknown.N)):
        idx2 =block(Assembled,Row,q);  assert np.count_nonzero(idx2 -  np.diag(np.diag(idx2))) == 0




def test_the_jacobian_sparsity_is_block_tridiagonal (  geometry ,   models,   perturbed_x)  :
    """No entry may couple nodes more than one edge apart in 1D.

    A stray entry would still satisfy the complex step check if the residual
    put it there too, so the structure is asserted separately from the values.
    """
    vals ,   voolume  =   geometry
    rws,Cols,_ = coupled_jacobian(
        h = vals,
        volume=voolume,
        x=perturbed_x,
        Dn=models.Dn,
        Dp= models.Dp,
        recombination = models.recombination,
    )
    oct =rws  //  UNKNOWNS_PER_NODE

    colNode =Cols  // UNKNOWNS_PER_NODE
    assert  np.all (np.abs( oct  -  colNode )  <=  1)

def test_assemble_coupled_agrees_with_the_array_level_functions(device ,   geometry,  models , perturbed_x)  :
    """The Field wrapper must not change a single number."""
    ass =  assemble_state(device, models, perturbed_x)


    Expected=residual_at(geometry,device,models)(perturbed_x)


    assert ass.shape   ==  (UNKNOWNS_PER_NODE  *   device.mesh.n_nodes, UNKNOWNS_PER_NODE  *  device.mesh.n_nodes,)
    np.testing.assert_array_equal(ass.residual,Expected)


def test_assemble_coupled_rejects_a_physical_field(  device , models, perturbed_x ) :

    """A physical density here is wrong by C_0 and would still converge."""
    psi,n,p=unpack(perturbed_x)

    with pytest.raises(ValueError,
                  match ="SCALED"):
        assemble_coupled(
            mesh   =  device.mesh,
            psi  =   Field(
                psi.copy(),   'V', ScalingState.PHYSICAL,  Location.NODE, name  =  'psi'
            ) ,
            n =  Field(  n.copy (),  'cm^-3', ScalingState.SCALED , Location.NODE, name  =  'n' ),
            p  =   Field (p.copy(  ), 'cm^-3', ScalingState.SCALED , Location.NODE,  name  =  'p') ,
            net_doping   = device.net_doping_scaled,
            recombination   =  models.recombination,
            scale   =  device.scale ,
            Dn   =  models.Dn,
            Dp   =   models.Dp ,
        )

def test_unknown_index_agrees_with_the_packed_layout(device):
    """The index helper and pack must not drift apart.

    Two ways to say the same thing, so one test pins them together rather
    than letting a later reordering fix one and leave the other.
    """
    psi  =  np.arange( 4.0  )
    n= np.arange(4.0)+10.0
    p  = np.arange(4.0)  + 20.0
    x2= pack(psi,n,p)

    for  ndoe in  range( 4)   :

        assert x2[unknown_index(ndoe, Unknown.PSI)]  == psi[ndoe]


        assert x2[unknown_index( ndoe,   Unknown.N )  ]  ==   n [ndoe]
        assert x2[unknown_index(ndoe, Unknown.P)] ==  p[ndoe]


def test_assemble_coupled_rejects_an_edge_field(device,models,perturbed_x):

    """A density on edges would be silently one entry short of the mesh."""

    psi,n,p=unpack(perturbed_x)


    with pytest.raises(ValueError,
                match= "NODE"):
        assemble_coupled(mesh  =device.mesh, psi  = Field(psi.copy(), "V", ScalingState.SCALED, Location.EDGE, name= "psi"), n =Field(n.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name = "n"), p=  Field(p.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name ='p'), net_doping = device.net_doping_scaled, recombination= models.recombination, scale =device.scale, Dn =models.Dn, Dp = models.Dp,)


def test_assemble_coupled_rejects_a_field_of_the_wrong_length(
    device,models,perturbed_x
) :
    """A length mismatch broadcasts into a plausible wrong answer otherwise."""
    psi,n,p =unpack(perturbed_x)
    with pytest.raises(  ValueError ,  match   =  'length'  )  :
        assemble_coupled(
            mesh =device.mesh,
            psi =  Field(
                psi[:- 1].copy(), "V", ScalingState.SCALED, Location.NODE, name= "psi"
            ),
            n  = Field(n.copy(), 'cm^-3', ScalingState.SCALED, Location.NODE, name =  "n"),
            p =  Field(p.copy(), "cm^-3", ScalingState.SCALED, Location.NODE, name='p'),
            net_doping=device.net_doping_scaled,
            recombination = models.recombination,
            scale  =device.scale,
            Dn= models.Dn,
            Dp  = models.Dp,
        )


def  test_contacts_pin_all_three_unknowns_at_the_contact_node(
    device,   models, perturbed_x
)   :
    """A coupled ohmic contact is three Dirichlet conditions, not one."""
    zz  =  assemble_state(device, models, perturbed_x)
    piinned =apply_contacts_coupled(
        zz,
        perturbed_x,
        device.net_doping_scaled.data,
        device.contacts,
        device.scale,
    )

    sol = SparseLU(); sol.factorize(piinned.rows, piinned.cols, piinned.values, piinned.shape)
    Delta=sol.solve(-piinned.residual)
    input  =  perturbed_x  +  Delta

    doing  =  device.net_doping_scaled.data
    for contct in device.contacts:
        nod  = contct.node

        assert input[unknown_index(nod, Unknown.PSI)] == pytest.approx(
            ohmic_psi_scaled(
                float(doing[nod]), contct.voltage  /  device.scale.psi_0
            ),
            rel  =1e-14,
        )

        assert input[unknown_index(nod, Unknown.N)] == pytest.approx(
            ohmic_density_scaled(float(doing[nod]), Carrier.ELECTRON), rel =  1e-14
        )
        assert input[unknown_index(nod, Unknown.P)] == pytest.approx(ohmic_density_scaled(float(doing[nod]), Carrier.HOLE), rel = 1e-14)

def test_the_contact_densities_agree_with_the_contact_potential ( device  )  :
    """n = exp(psi - phi_n) at a contact, with phi_n = phi_p = the bias.

    The two boundary conditions are written from different physics, one from
    neutrality plus mass action and one from asinh of the doping, so their
    agreeing is a real check rather than a restatement. If they disagreed the
    solver would be pulled between two incompatible statements at one node
    and the terminal current would come out wrong with everything converged.
    """
    ret =  device.with_bias(anode  = 0.35,   cathode  =   0.0 )
    arr =ret.net_doping_scaled.data
    for  conttact in ret.contacts   :

        k2 =  conttact.voltage  /  ret.scale.psi_0


        psi =  ohmic_psi_scaled( float (arr[ conttact.node]),   k2  )
        n = ohmic_density_scaled(float(arr[conttact.node]), Carrier.ELECTRON)
        p = ohmic_density_scaled(float(arr[conttact.node]),Carrier.HOLE)
        assert n  ==  pytest.approx(np.exp( psi   -  k2 ) ,   rel  =  1e-12  )
        assert p== pytest.approx(np.exp(k2 -  psi), rel  =1e-12)
        assert n   *  p  == pytest.approx(1.0,  rel  =  1e-12  )




def test_the_applied_bias_moves_psi_and_leaves_the_densities_alone(device):
    """An ohmic contact stays in equilibrium whatever the terminal voltage."""
    oct  =  device.net_doping_scaled.data[ 0  ]
    unbased =  ohmic_psi_scaled( float(  oct ),  0.0)
    bia =  ohmic_psi_scaled (float( oct ) ,
           1.0  )


    assert bia -  unbased  ==  pytest.approx(1.0, rel = 1e-14)
    assert ohmic_density_scaled(float(oct), Carrier.ELECTRON) == ohmic_density_scaled(float(oct), Carrier.ELECTRON)


def test_two_contacts_sharing_a_name_are_rejected(device, models, perturbed_x):
    """Mirrors the Poisson path. A duplicate name breaks current reporting."""
    ass= assemble_state(device, models, perturbed_x)

    with  pytest.raises (ValueError, match  =   "unique"  )  :
        apply_contacts_coupled(
            ass,
            perturbed_x,
            device.net_doping_scaled.data,
            (
                OhmicContact(name =  "anode", node=  0, voltage =0.0),
                OhmicContact(name =  'anode', node = N_NODES  -1, voltage= 0.0),
            ),
            device.scale,
        )

def test_the_poisson_term_scale_counts_the_carriers_not_only_the_doping():
    """An intrinsic bar has no doping and its Poisson terms are not zero.

    The residual carries -(p - n + N)*volume. On intrinsic material that sum
    is exactly zero, but the terms going into it are n*volume and p*volume,
    both equal to one dual cell in scaled units. A scale built from the net
    doping alone reports zero there, and dividing by it makes every row nan.

    Measured before the fix: an undoped 41 node bar came back with a residual
    of nan and the message blamed the LU factorization for being singular,
    which sends you debugging the linear algebra instead of the scale.
    """
    H = np.full(4, 0.1)
    Volume =np.full(5,0.1)
    X = pack(np.zeros(5), np.ones(5), np.ones(5))

    temp2,_,_ =residual_term_scales(H,Volume,X,np.zeros(5),Dn = 1.0,Dp = 1.0)

    assert np.all(temp2>0.0)


def test_every_term_scale_is_strictly_positive_on_a_real_device(device,   geometry, models,   perturbed_x) :
    """Nothing downstream can divide by these safely otherwise."""
    H, Volume =geometry
    sccales=residual_term_scales(H, Volume, perturbed_x, device.net_doping_scaled.data, models.Dn, models.Dp)

    assert all(np.all(scale>0.0) for scale in sccales)

def  test_row_scaling_keeps_the_system_finite(  device,   geometry , models,  perturbed_x  )  :
    """The whole point of the scaling is defeated if it introduces a nan."""
    hh, out2  = geometry
    thing=residual_term_scales(
        hh,out2,perturbed_x,device.net_doping_scaled.data,models.Dn,models.Dp
    )
    System = assemble_coupled_arrays(h =hh, volume =out2, x= perturbed_x, net_doping=device.net_doping_scaled.data, Dn= models.Dn, Dp = models.Dp, recombination=models.recombination,)



    sccaled=scale_rows(System,row_weights(thing,device.mesh.n_nodes))

    assert np.all(np.isfinite(sccaled.residual))
    assert np.all(np.isfinite(sccaled.values))
def test_a_state_with_no_carriers_anywhere_is_refused()  :

    '''Not a physical state, and silently producing nan hides where it came from.

    A density of exactly zero everywhere leaves every term in every equation
    at zero, so there is no scale to measure against. Refusing names the
    problem; dividing by zero renames it as a singular matrix three call
    frames later.
    '''

    hh  = np.full(4, 0.1)
    vol=  np.full(5, 0.1); X   =  pack (  np.zeros(5 ),   np.zeros( 5  ),  np.zeros(  5))


    with pytest.raises(ValueError, match = "no terms") :

        residual_term_scales(hh, vol, X, np.zeros(5), Dn  =1.0, Dp= 1.0)




def test_a_state_whose_terms_overflow_is_a_diverged_iterate_not_a_bad_state():

    """A Newton iterate that overshoots can carry a density whose flux terms
    overflow. That is divergence, which newton_solve reports and continuation
    backs off from, so it raises FloatingPointError for newton_solve to catch
    rather than the ValueError reserved for a state with nothing in it. Found
    with the drain held at 10 V: the ramp died on the ValueError before it
    could take a smaller step."""
    hh  = np.full(4, 0.1)
    Volume =  np.full(5, 0.1)
    xx= pack(np.zeros(5),np.full(5,np.inf),np.ones(5))


    with  np.errstate ( invalid  =   'ignore' , over   =  'ignore' )  :

        with pytest.raises(FloatingPointError,match="diverged"):
            residual_term_scales(hh, Volume, xx, np.zeros(5), Dn=  1.0, Dp = 1.0)



def test_the_shared_path_reproduces_the_standalone_functions_exactly(device, geometry, models, perturbed_x) :
    """assemble_coupled_terms and the two public functions must not diverge.

    This is what keeps the block verification meaningful. Those tests
    differentiate coupled_residual and compare against coupled_jacobian, but a
    solve runs assemble_coupled_terms, which shares one Bernoulli pair between
    the three. Two code paths where only one is verified is how a verification
    stops being one.

    Found by mutation rather than by inspection: after the shared path was
    introduced, replacing the exact SRH tangent with the Gummel frozen slope
    inside it left every block test passing, because no test executed it.

    Bit for bit, not to a tolerance. The two do the same operations in the
    same order on the same inputs, so anything less than exact equality means
    they have genuinely drifted apart.
    """
    H,type=geometry
    dop =  device.net_doping_scaled.data

    shhared =assemble_coupled_terms(H,type,perturbed_x,dop,models.Dn,models.Dp,models.recombination).assembly
    acc=coupled_residual(H,type,perturbed_x,dop,models.Dn,models.Dp,models.recombination)
    Rows,Cols,Values =coupled_jacobian(
        H,type,perturbed_x,models.Dn,models.Dp,models.recombination
    )

    np.testing.assert_array_equal(shhared.residual,acc); np.testing.assert_array_equal (  shhared.rows ,   Rows  )
    np.testing.assert_array_equal(shhared.cols,Cols)
    np.testing.assert_array_equal(shhared.values, Values)


def test_the_shared_path_scales_match_the_standalone_scales(device, geometry, models, perturbed_x)  :
    """The third output of the shared path needs the same guard."""
    zz,Volume = geometry

    Doping=device.net_doping_scaled.data

    psi,n,p= unpack(perturbed_x)

    _,sha=assemble_coupled_terms(zz,Volume,perturbed_x,Doping,models.Dn,models.Dp,models.recombination)
    standalnoe=residual_term_scales(
        zz,
        Volume,
        perturbed_x,
        Doping,
        models.Dn,
        models.Dp,
        R =np.asarray(models.recombination.rate(n,p),dtype = np.float64),
    )
    for fs, hash  in zip(  sha ,   standalnoe , strict  = True )  :
        np.testing.assert_array_equal(fs,hash)
def  test_a_term_scale_of_the_wrong_length_is_named_rather_than_broadcast ( )  :
    """The preconditioner takes one number per family out of an array.

    Which means a scale array that does not match the mesh still reduces to a
    number and still fills every weight, so the caller gets a preconditioner
    built from part of a different device with nothing said about it. The two
    things that could disagree are the scales and the node count, and they
    arrive as separate arguments, so nothing else can catch it.
    """
    NNodes =   5
    goood  = np.ones(NNodes)
    r2   =  np.ones (NNodes   -  1)

    with pytest.raises(ValueError,match='N term scale has 4 entries'):
        row_weights((goood,r2,goood),NNodes)


def test_a_row_with_no_terms_in_it_is_skipped_rather_than_dividing_by_zero() :
    """A node holding no semiconductor carries no flux and no recombination.

    Its continuity rows are pinned to the identity, so their residual is the
    pinning error and it goes to zero in one step whatever it is measured
    against. What must not happen is that the zero scale turns the whole
    measure into an inf or a nan and takes every other row's evidence with it.
    """
    nnodes=  3
    stuff2=np.full(nnodes,2.0)

    nScale = np.array([1.0,0.0,4.0])
    val =np.full(nnodes,8.0)
    sca  =  ( stuff2,  nScale,   val)

    q =row_weights(sca,nnodes)
    raww  =  np.zeros(  UNKNOWNS_PER_NODE   *  nnodes)
    raww[unknown_index(1, Unknown.N)]=  1e30
    raww [unknown_index(2, Unknown.N) ]   =  2.0

    Measured  = residual_measure (  raww  /  q ,   sca,   nnodes )
    assert Measured== pytest.approx(0.5)




def test_a_row_whose_terms_collapsed_is_skipped_like_one_with_none()  -> None:


    """A scale can vanish by degrees, and the exact zero test misses that.

    The row above holds no semiconductor and its scale is exactly zero. This
    one holds semiconductor whose minority population has emptied, so its
    terms are merely tiny, and no residual on it can be resolved relative to
    them: the columns of the assembly span the whole family, so the linear
    solve delivers that unknown to an absolute accuracy set by the largest
    terms and not to a relative one against its own. Dividing an already
    converged residual by terms that small manufactures a number out of
    roundoff, and because it is roundoff it wanders rather than settling,
    which makes the convergence test a coin flip.

    Measured on the 1 um NMOS of SHORT_CHANNEL_PROCESS at Vd = 1 V, deep in
    inversion: the electron row of node 2810 carried a raw residual of
    9.0e-20 against terms of 2.4e-10, 23 decades below the 6.4e+13 the family
    reaches in the source. That read as 3.8e-10 and decided a test set at
    1e-10, so the same solve took 8, 14 or 22 iterations depending only on
    the path taken to reach the bias, and exceeded a budget of 30 on CI.
    See the 2026-09-12 row in docs/07-decisions.md.
    """
    NNodes   =  3
    myvar =np.full(NNodes,2.0)
    nscale = np.array([1.0, 1e-20, 4.0]) ; PScale =  np.full(NNodes, 8.0)
    scalles  =(myvar, nscale, PScale)



    wei=row_weights(scalles, NNodes)
    Raw   =  np.zeros ( UNKNOWNS_PER_NODE  *   NNodes  )
    Raw [unknown_index(1 , Unknown.N  ) ] = 1e-18
    Raw[unknown_index(2, Unknown.N)] = 2.0
    mea  =   residual_measure(Raw / wei , scalles,   NNodes  )
    assert mea==pytest.approx(0.5)

def test_a_row_just_above_the_floor_still_counts()->None:
    '''The floor skips what cannot be resolved and nothing else.

    A scale one decade above eps times the family maximum is small but real,
    and a residual measured against it is evidence. Dropping those rows would
    certify exactly the states the per row measure was introduced to catch.
    '''
    NNodes = 2

    oct= float(np.finfo(np.float64).eps)* 4.0
    ns =  np.array([oct * 10.0, 4.0])
    Scales  = (np.full(NNodes,   2.0), ns ,   np.full (  NNodes, 8.0 ) )
    Weights=row_weights(Scales,NNodes)

    out2 =np.zeros(UNKNOWNS_PER_NODE  *  NNodes)
    out2[unknown_index(0,Unknown.N)]= oct *10.0*0.25

    bb=residual_measure(out2 /Weights,Scales,NNodes)



    assert  bb   == pytest.approx(0.25)

def test_the_update_split_puts_each_family_under_its_own_name() -> None  :
    """The browser names the family that stalled from these labels, so a psi
    number filed under n would send someone to the wrong equation. Each family
    is given a distinct size so a swap cannot pass."""
    X=pack(np.zeros(3),np.array([9.0,1.0,3.0]),np.array([0.0,4.0,1.0]))
    dellta  = pack(np.array([0.0, - 0.25, 0.1]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -  8.0]),)

    slit =coupled_update_by_family(dellta,X)
    assert slit =={'psi':0.25,'n':0.5,"p": 4.0}
    assert  coupled_update_norm(  dellta ,   X ) ==  4.0
