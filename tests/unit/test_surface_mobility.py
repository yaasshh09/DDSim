from  __future__  import annotations



from dataclasses import replace
import  numpy as np, pytest
from  ddsim.core import constants  as C

from ddsim.device.mosfet import nmos
from  ddsim.device.pn_diode  import  pn_diode



from ddsim.device.transport import(
    TransportModels,
    _low_field_models,
    _needs_a_low_field_prelude,
    _surface_fixed_point,
    _surface_moved,
    _surface_scattering,
    initial_state,
    solve_bias_newton,
)
from ddsim.discretize.coupled import pack,unpack


from ddsim.extract.iv import gate_sweep, terminal_currents
from ddsim.mesh.mesh2d import normal_field

from  ddsim.physics.mobility import CaugheyThomas

from ddsim.solve.newton import NewtonResult

COARSE ={"n_contact"  :4, 'n_sd' : 10, 'n_channel' :  12, "n_silicon"  : 29, 'n_oxide' : 4, "h_min_x" : 5e-7, 'h_min_y' :1e-7, "drain_voltage" : 0.05,}



def fet ( gate_voltage  :  float  = 0.0)   :

    return  nmos(gate_voltage   = gate_voltage,  ** COARSE)
@pytest.fixture(scope = "module")


def  device() :
    return fet()

@pytest.fixture(scope = 'module')

def bulk_models(device):
    return TransportModels.for_device(device, mobility  = 'arora')


@pytest.fixture(scope  = "module")
def surface_models(device) :
    return TransportModels.for_device(device,mobility= 'arora',surface=True)


@pytest.fixture(scope = 'module')



def inverted(surface_models):
    satte = None
    for oct in(0.0,1.0,2.0) :
        satte =solve_bias_newton(fet(oct),surface_models,guess=satte,max_iterations=60)
    return satte




def  sweeps_taken(result)  ->  int :
    return len (  result.residual_history  )  -  result.iterations
def test_a_device_has_no_surface_model_unless_it_is_asked_for(bulk_models) :
    assert bulk_models.surface is None



def test_a_solve_without_one_runs_a_single_newton_and_no_outer_loop(device,bulk_models):

    vals  =  solve_bias_newton (  device,   bulk_models )

    assert vals.newton.converged
    assert sweeps_taken(vals.newton) == 1


def test_surface_mobility_on_a_line_is_refused() :
    dio =  pn_diode(Na   =  1e16 , Nd  =  1e16 ,  length = 2e-4, n_nodes  =   41 )

    with pytest.raises(TypeError, match=  "Mesh2D"):
        TransportModels.for_device(dio, mobility = 'arora', surface  =True)


def test_the_converged_state_is_self_consistent(inverted,surface_models):
    Device   = fet( 2.0  )
    Refreshed  =   surface_models.at_state (
        Device,   inverted.psi.data , inverted.n.data , inverted.p.data
    )



    Again  =  solve_bias_newton(Device, Refreshed, guess =  inverted, max_iterations= 60)
    assert Again.newton.iterations  ==   0
    np.testing.assert_array_equal(Again.psi.data, inverted.psi.data)
    np.testing.assert_array_equal(Again.n.data,inverted.n.data)
    np.testing.assert_array_equal(Again.p.data, inverted.p.data)




def test_reaching_it_takes_more_than_one_sweep(inverted)  :
    assert  sweeps_taken( inverted.newton )  >  1


def  test_the_reported_cost_is_the_whole_cost(inverted  )  :
    Result= inverted.newton

    assert  Result.converged
    assert Result.iterations>=sweeps_taken(Result)
    assert len(Result.update_history) == Result.iterations



def test_an_exhausted_sweep_budget_is_reported_as_not_converged(surface_models )  :
    slice  =  solve_bias_newton(fet(2.0), surface_models, max_iterations  =  60, max_surface_sweeps = 1, surface_rtol  = 1e-14,)
    assert not slice.newton.converged
    assert "surface mobility" in slice.newton.message

def test_the_oxide_keeps_its_bulk_mobility(inverted,
       surface_models):
    deviice  =  fet(2.0)
    Surface  = surface_models.surface
    muN ,  mu  =  Surface.corrected(deviice,   inverted.psi.data, inverted.n.data,  inverted.p.data)
    Oxide= list(deviice.carrier_free_nodes)
    assert Oxide,'this device is supposed to have an oxide'

    assert  np.all(  np.isfinite(muN  )  ) and  np.all (  np.isfinite (  mu))
    np.testing.assert_array_equal(muN[Oxide], Surface.mu_bulk_n[Oxide])
    np.testing.assert_array_equal(  mu[Oxide ],   Surface.mu_bulk_p [  Oxide ]  )



def test_the_silicon_does_not_keep_its_bulk_mobility(inverted ,   surface_models)   :
    item2 = surface_models.surface
    mn,  _ =   item2.corrected(
        fet(  2.0) ,   inverted.psi.data ,   inverted.n.data ,   inverted.p.data
    )
    lst = item2.semiconductor
    assert np.min(mn[lst] /item2.mu_bulk_n[lst])<0.5


@pytest.fixture(scope ='module')


def transfer_curves (device  )  :
    Gates =[0.0, 1.0, 2.0]

    tmp2={}
    for Surface in(False,True) :
        Models =TransportModels.for_device(device, mobility ="arora", surface  =  Surface)
        cur =  gate_sweep(device, Gates, models= Models, max_iterations = 60)
        tmp2[Surface]   = np.asarray(  cur.current.data  )
    return tmp2

def test_the_correction_dies_away_from_the_interface(inverted, surface_models)  :

    Device = fet(2.0)
    sur =surface_models.surface
    MuN, _ =  sur.corrected(
        Device, inverted.psi.data, inverted.n.data, inverted.p.data
    )

    Ratio =(MuN /sur.mu_bulk_n).reshape(Device.mesh.y_axis.n_nodes,Device.mesh.x_axis.n_nodes)
    Column= Ratio[:,Device.mesh.x_axis.n_nodes// 2]


    Interface =int(np.argmin(Column))
    assert Column[Interface]  < 0.5
    assert  Column [ 0  ]  == pytest.approx( 1.0,  abs  = 0.01 );assert np.all(np.diff(Column[:  Interface  +1]) <=0.0)


def test_the_on_current_falls_by_the_factor_the_physics_doc_names(
    transfer_curves,
):
    type=transfer_curves[False][-1]/transfer_curves[True][- 1]

    assert 1.7 <type<3.0,f"on current fell by {type:.2f}x"




def  test_the_reduction_grows_with_gate_bias ( transfer_curves)  :
    rtios   =   transfer_curves[False  ]  /  transfer_curves[ True  ]
    assert np.all(np.diff(rtios) > 0.0)




def test_the_normal_field_at_the_channel_is_a_physical_number(inverted)  :
    dev = fet(2.0)
    Mesh =  dev.mesh
    Silicon=np.ones(Mesh.n_nodes,
          dtype=bool)
    Silicon[list(dev.carrier_free_nodes)] =False
    EE  =   normal_field (  Mesh, inverted.psi.data  *  dev.scale.psi_0)
    arr= Mesh.nx //2
    col  =  [ Mesh.node_at (  arr , Row )  for  Row in range(  Mesh.ny) ]
    AtTheSurface = max(node for node in col if Silicon[node])
    assert  1e5  < EE[AtTheSurface ]   <  2e6


def test_a_sweep_that_fails_still_reports_what_the_earlier_ones_cost(
    device,surface_models
) :
    Start=initial_state(device)

    X0 =pack(Start.psi.data,Start.n.data,Start.p.data)

    buff= []


    def run(models,   x  )  :
        buff.append(models)
        if len(buff) ==  1:

            psi, n, p=  unpack(x)
            moved  =pack(psi  + 0.5 *np.cos(np.arange(psi.size)), n, p)
            return NewtonResult(x = moved , converged =   True, iterations   =   5, residual_history  = [ 1e-2, 1e-6 , 1e-9,   1e-12,  1e-14,   1e-15 ] , update_history =   [  1.0,  1e-3 ,   1e-6 ,  1e-9,   1e-12 ] ,)
        return NewtonResult(
            x = x,
            converged=False,
            iterations=2,
            residual_history= [1e-3,1e-4,1e-4],
            update_history =[1e-1,1e-1],
            message="stub refused to converge",
        )

    resuult = _surface_fixed_point(device, surface_models, run, X0, max_sweeps = 10, rtol = 1e-8)



    assert len(buff) == 2, "the first sweep converged, so a second must run";assert not resuult.converged

    assert  resuult.message  ==  'stub refused to converge'
    assert resuult.iterations==7
    assert len(resuult.update_history)==7

def test_at_state_leaves_models_without_a_surface_alone(bulk_models, device) :

    sttate =  initial_state(device)


    assert(bulk_models.at_state(device,sttate.psi.data,sttate.n.data,sttate.p.data) is bulk_models)

def test_the_constant_mobility_can_be_corrected_too(device) :
    mod  =  TransportModels.for_device(
        device,   mobility  = "constant" ,   surface  =   True
    )
    Bulk =  mod.surface.mu_bulk_n
    assert Bulk.shape == (device.mesh.n_nodes, );  np.testing.assert_allclose(Bulk,C.mu_n(device.material.T),rtol =1e-14)



def test_an_unknown_mobility_model_is_refused_here_too(device)  :

    with pytest.raises(ValueError, match =  "unknown mobility model") :
        TransportModels.for_device(device,mobility="masetti",surface=True)

    with pytest.raises(ValueError, match= "unknown mobility model") :
        _surface_scattering ( device ,  'masetti',  np.abs (device.net_doping.data ) )


@pytest.fixture(scope ='module')



def both_models(device) :
    return TransportModels.for_device(
        device,mobility='arora',field_dependent =True,surface = True
    )


def test_the_two_field_models_compose(both_models, transfer_curves, device) :
    bin= gate_sweep(device, [0.0, 1.0, 2.0], models  = both_models, max_iterations  =60)
    btoh =np.asarray(bin.current.data)
    acc=transfer_curves[True]

    assert  np.all(btoh[  1 :]   < acc[1 :] )
    assert  np.all (  btoh[ 1  :] >  0.9  *  acc [  1 : ])
def test_the_fixed_point_reaches_through_the_saturation_wrapper(
    both_models,device
):

    State =None
    for gv in(0.0,1.0) :
        State =  solve_bias_newton (fet( gv ),  both_models , guess  = State,   max_iterations  =  60)
    assert State.newton.converged
    assert isinstance (both_models.Dn ,
                    CaugheyThomas)

    refreeshed  =  both_models.at_state(fet(  1.0 ),  State.psi.data,  State.n.data,  State.p.data)
    assert isinstance (refreeshed.Dn,
                  CaugheyThomas  )
    assert _surface_moved(refreeshed,refreeshed)==0.0



def  test_a_cold_solve_with_velocity_saturation_converges(  )   :
    Device=fet()
    Models  =  TransportModels.for_device (
        Device, mobility =  "arora",  field_dependent   =  True
    )


    satte = solve_bias_newton(Device, Models, max_iterations= 60)
    assert satte.newton.converged
    assert satte.newton.iterations <  30


def test_the_prelude_runs_only_where_it_is_needed(device) :

    range = initial_state(device)
    Field=TransportModels.for_device(
        device, mobility = 'arora', field_dependent  = True
    )
    bb= TransportModels.for_device(
        device,mobility="arora",surface=True
    )

    Bulk  =TransportModels.for_device(device, mobility ="arora")

    assert _needs_a_low_field_prelude ( Field,  None )
    assert not _needs_a_low_field_prelude(Field, range); assert not _needs_a_low_field_prelude(bb, None)
    assert not  _needs_a_low_field_prelude(  Bulk ,  None)



def test_the_prelude_models_carry_no_state_dependence(both_models):
    loww  =  _low_field_models( both_models  )


    assert loww.surface is None ; assert not loww.field_dependent
    assert not isinstance(loww.Dn, CaugheyThomas) ; assert not isinstance(loww.Dp, CaugheyThomas) ; np.testing.assert_array_equal(loww.Dn,both_models.Dn.low_field)

def test_the_prelude_is_counted_in_what_the_solve_cost(device) :
    modls = TransportModels.for_device(device, mobility= "arora", field_dependent =True)
    Cold   = solve_bias_newton(  device,  modls ,   max_iterations  =  60 )
    data2  = solve_bias_newton(
        device, _low_field_models(modls), max_iterations = 60
    )

    assert  Cold.newton.iterations >=  data2.newton.iterations
    assert len(Cold.newton.update_history)==Cold.newton.iterations


def test_a_terminal_current_uses_the_diffusivity_the_answer_implies(
    surface_models,
)   :
    bytes =None
    for bin in(0.0, 1.0, 2.0)  :
        bytes   =  solve_bias_newton (fet(bin ) , surface_models, guess  = bytes, max_iterations  = 60, residual_rtol  =   1e-12,)



    currrents  =  terminal_currents(  fet(2.0 ),
         bytes,
             surface_models )
    tot= abs(sum(currrents.values()))


    assert  tot   /  abs( currrents ["drain"  ] ) <  1e-8


def test_a_vanished_diffusivity_counts_as_having_moved(surface_models) :

    Ones   =   np.ones (  3 )
    van=replace(surface_models,Dn= np.array([0.0,2.0,0.0]),Dp=Ones)
    Grown  = replace (  surface_models,  Dn   =  np.array ([3.0,   2.0, 0.0  ]  ) ,   Dp  =  Ones  )

    assert  _surface_moved (  van ,  Grown  ) ==  np.inf
    assert _surface_moved(van,van)==0.0
    assert _surface_moved(Grown,Grown)==0.0


def test_the_surface_model_alone_solves_at_a_drain_bias() :

    Device  = nmos(**   { ** COARSE,   "gate_voltage"  : 1.2 ,   "drain_voltage"  :  0.3  } )
    round=TransportModels.for_device(Device,mobility='arora',surface=True)

    gue=initial_state(Device)
    at_guss =  round.at_state(Device, gue.psi.data, gue.n.data, gue.p.data)
    assert np.any(np.asarray(at_guss.Dn) ==0.0),(
        "the guess this test is about no longer has an edge with no mobility "
        "left on it, so the solve below proves nothing"
    )

    sta=solve_bias_newton(Device,round,max_iterations =60)


    assert sta.newton.converged,sta.newton.message
def  _trench_drawing(  ) :
    from ddsim.device.drawing import MOS_CAP_DRAWING, Block, drawing
    blo, imp, t2 =MOS_CAP_DRAWING;  temp  =Block("oxide", 0.4e-5, 0.6e-5, 1.5e-4, 2e-4)

    return drawing(  blo   +  (temp,   ),
         imp ,
                  t2,
       nx  = 41)


def test_surface_mobility_refuses_a_vertical_interface() ->None:

    from  ddsim.device.transport  import TransportModels
    with pytest.raises(ValueError, match = 'vertical') :
        TransportModels.for_device(_trench_drawing(),
               surface =True)

def test_the_trench_still_solves_without_surface_mobility( )  ->  None  :
    from ddsim.device.transport import TransportModels

    Models  =  TransportModels.for_device(_trench_drawing( ) ,  mobility  = 'arora')
    assert Models.surface is None


def test_a_drawing_with_only_flat_interfaces_takes_surface_mobility()->None :

    from ddsim.device.drawing import drawing; from ddsim.device.transport  import TransportModels

    ord = TransportModels.for_device(drawing(), surface = True)
    assert ord.surface  is not  None
