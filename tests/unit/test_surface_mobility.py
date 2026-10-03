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
    i = None
    for y2 in(0.0,1.0,2.0) :
        i =solve_bias_newton(fet(y2),surface_models,guess=i,max_iterations=60)
    return i




def  sweeps_taken(result)  ->  int :
    return len (  result.residual_history  )  -  result.iterations
def test_a_device_has_no_surface_model_unless_it_is_asked_for(bulk_models) :
    assert bulk_models.surface is None



def test_a_solve_without_one_runs_a_single_newton_and_no_outer_loop(device,bulk_models):

    aa  =  solve_bias_newton (  device,   bulk_models )

    assert aa.newton.converged
    assert sweeps_taken(aa.newton) == 1


def test_surface_mobility_on_a_line_is_refused() :
    tmp3 =  pn_diode(Na   =  1e16 , Nd  =  1e16 ,  length = 2e-4, n_nodes  =   41 )

    with pytest.raises(TypeError, match=  "Mesh2D"):
        TransportModels.for_device(tmp3, mobility = 'arora', surface  =True)


def test_the_converged_state_is_self_consistent(inverted,surface_models):
    row   = fet( 2.0  )
    stuff  =   surface_models.at_state (
        row,   inverted.psi.data , inverted.n.data , inverted.p.data
    )



    d2  =  solve_bias_newton(row, stuff, guess =  inverted, max_iterations= 60)
    assert d2.newton.iterations  ==   0
    np.testing.assert_array_equal(d2.psi.data, inverted.psi.data)
    np.testing.assert_array_equal(d2.n.data,inverted.n.data)
    np.testing.assert_array_equal(d2.p.data, inverted.p.data)




def test_reaching_it_takes_more_than_one_sweep(inverted)  :
    assert  sweeps_taken( inverted.newton )  >  1


def  test_the_reported_cost_is_the_whole_cost(inverted  )  :
    s= inverted.newton

    assert  s.converged
    assert s.iterations>=sweeps_taken(s)
    assert len(s.update_history) == s.iterations



def test_an_exhausted_sweep_budget_is_reported_as_not_converged(surface_models )  :
    res  =  solve_bias_newton(fet(2.0), surface_models, max_iterations  =  60, max_surface_sweeps = 1, surface_rtol  = 1e-14,)
    assert not res.newton.converged
    assert "surface mobility" in res.newton.message

def test_the_oxide_keeps_its_bulk_mobility(inverted,
       surface_models):
    h  =  fet(2.0)
    b  = surface_models.surface
    tmp3 ,  a2  =  b.corrected(h,   inverted.psi.data, inverted.n.data,  inverted.p.data)
    g= list(h.carrier_free_nodes)
    assert g,'this device is supposed to have an oxide'

    assert  np.all(  np.isfinite(tmp3  )  ) and  np.all (  np.isfinite (  a2))
    np.testing.assert_array_equal(tmp3[g], b.mu_bulk_n[g])
    np.testing.assert_array_equal(  a2[g ],   b.mu_bulk_p [  g ]  )



def test_the_silicon_does_not_keep_its_bulk_mobility(inverted ,   surface_models)   :
    d = surface_models.surface
    cc,  _ =   d.corrected(
        fet(  2.0) ,   inverted.psi.data ,   inverted.n.data ,   inverted.p.data
    )
    t = d.semiconductor
    assert np.min(cc[t] /d.mu_bulk_n[t])<0.5


@pytest.fixture(scope ='module')


def transfer_curves (device  )  :
    z2 =[0.0, 1.0, 2.0]

    s2={}
    for b in(False,True) :
        zz =TransportModels.for_device(device, mobility ="arora", surface  =  b)
        a =  gate_sweep(device, z2, models= zz, max_iterations = 60)
        s2[b]   = np.asarray(  a.current.data  )
    return s2

def test_the_correction_dies_away_from_the_interface(inverted, surface_models)  :

    g = fet(2.0)
    m =surface_models.surface
    t, _ =  m.corrected(
        g, inverted.psi.data, inverted.n.data, inverted.p.data
    )

    z =(t /m.mu_bulk_n).reshape(g.mesh.y_axis.n_nodes,g.mesh.x_axis.n_nodes)
    obj= z[:,g.mesh.x_axis.n_nodes// 2]


    y =int(np.argmin(obj))
    assert obj[y]  < 0.5
    assert  obj [ 0  ]  == pytest.approx( 1.0,  abs  = 0.01 );assert np.all(np.diff(obj[:  y  +1]) <=0.0)


def test_the_on_current_falls_by_the_factor_the_physics_doc_names(
    transfer_curves,
):
    z2=transfer_curves[False][-1]/transfer_curves[True][- 1]

    assert 1.7 <z2<3.0,f"on current fell by {z2:.2f}x"




def  test_the_reduction_grows_with_gate_bias ( transfer_curves)  :
    dat   =   transfer_curves[False  ]  /  transfer_curves[ True  ]
    assert np.all(np.diff(dat) > 0.0)




def test_the_normal_field_at_the_channel_is_a_physical_number(inverted)  :
    thing = fet(2.0)
    tmp =  thing.mesh
    e=np.ones(tmp.n_nodes,
          dtype=bool)
    e[list(thing.carrier_free_nodes)] =False
    vals  =   normal_field (  tmp, inverted.psi.data  *  thing.scale.psi_0)
    z= tmp.nx //2
    a  =  [ tmp.node_at (  z , f )  for  f in range(  tmp.ny) ]
    m = max(g for g in a if e[g])
    assert  1e5  < vals[m ]   <  2e6


def test_a_sweep_that_fails_still_reports_what_the_earlier_ones_cost(
    device,surface_models
) :
    tt=initial_state(device)

    y =pack(tt.psi.data,tt.n.data,tt.p.data)

    g= []


    def run(models,   x  )  :
        g.append(models)
        if len(g) ==  1:

            psi, n, p=  unpack(x)
            out2  =pack(psi  + 0.5 *np.cos(np.arange(psi.size)), n, p)
            return NewtonResult(x = out2 , converged =   True, iterations   =   5, residual_history  = [ 1e-2, 1e-6 , 1e-9,   1e-12,  1e-14,   1e-15 ] , update_history =   [  1.0,  1e-3 ,   1e-6 ,  1e-9,   1e-12 ] ,)
        return NewtonResult(
            x = x,
            converged=False,
            iterations=2,
            residual_history= [1e-3,1e-4,1e-4],
            update_history =[1e-1,1e-1],
            message="stub refused to converge",
        )

    u = _surface_fixed_point(device, surface_models, run, y, max_sweeps = 10, rtol = 1e-8)



    assert len(g) == 2, "the first sweep converged, so a second must run";assert not u.converged

    assert  u.message  ==  'stub refused to converge'
    assert u.iterations==7
    assert len(u.update_history)==7

def test_at_state_leaves_models_without_a_surface_alone(bulk_models, device) :

    m =  initial_state(device)


    assert(bulk_models.at_state(device,m.psi.data,m.n.data,m.p.data) is bulk_models)

def test_the_constant_mobility_can_be_corrected_too(device) :
    item  =  TransportModels.for_device(
        device,   mobility  = "constant" ,   surface  =   True
    )
    b2 =  item.surface.mu_bulk_n
    assert b2.shape == (device.mesh.n_nodes, );  np.testing.assert_allclose(b2,C.mu_n(device.material.T),rtol =1e-14)



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
    g= gate_sweep(device, [0.0, 1.0, 2.0], models  = both_models, max_iterations  =60)
    d =np.asarray(g.current.data)
    it=transfer_curves[True]

    assert  np.all(d[  1 :]   < it[1 :] )
    assert  np.all (  d[ 1  :] >  0.9  *  it [  1 : ])
def test_the_fixed_point_reaches_through_the_saturation_wrapper(
    both_models,device
):

    d2 =None
    for x2 in(0.0,1.0) :
        d2 =  solve_bias_newton (fet( x2 ),  both_models , guess  = d2,   max_iterations  =  60)
    assert d2.newton.converged
    assert isinstance (both_models.Dn ,
                    CaugheyThomas)

    bb  =  both_models.at_state(fet(  1.0 ),  d2.psi.data,  d2.n.data,  d2.p.data)
    assert isinstance (bb.Dn,
                  CaugheyThomas  )
    assert _surface_moved(bb,bb)==0.0



def  test_a_cold_solve_with_velocity_saturation_converges(  )   :
    tmp=fet()
    m2  =  TransportModels.for_device (
        tmp, mobility =  "arora",  field_dependent   =  True
    )


    h = solve_bias_newton(tmp, m2, max_iterations= 60)
    assert h.newton.converged
    assert h.newton.iterations <  30


def test_the_prelude_runs_only_where_it_is_needed(device) :

    d = initial_state(device)
    b=TransportModels.for_device(
        device, mobility = 'arora', field_dependent  = True
    )
    bar= TransportModels.for_device(
        device,mobility="arora",surface=True
    )

    zz  =TransportModels.for_device(device, mobility ="arora")

    assert _needs_a_low_field_prelude ( b,  None )
    assert not _needs_a_low_field_prelude(b, d); assert not _needs_a_low_field_prelude(bar, None)
    assert not  _needs_a_low_field_prelude(  zz ,  None)



def test_the_prelude_models_carry_no_state_dependence(both_models):
    num  =  _low_field_models( both_models  )


    assert num.surface is None ; assert not num.field_dependent
    assert not isinstance(num.Dn, CaugheyThomas) ; assert not isinstance(num.Dp, CaugheyThomas) ; np.testing.assert_array_equal(num.Dn,both_models.Dn.low_field)

def test_the_prelude_is_counted_in_what_the_solve_cost(device) :
    t = TransportModels.for_device(device, mobility= "arora", field_dependent =True)
    k   = solve_bias_newton(  device,  t ,   max_iterations  =  60 )
    xx  = solve_bias_newton(
        device, _low_field_models(t), max_iterations = 60
    )

    assert  k.newton.iterations >=  xx.newton.iterations
    assert len(k.newton.update_history)==k.newton.iterations


def test_a_terminal_current_uses_the_diffusivity_the_answer_implies(
    surface_models,
)   :
    vv =None
    for r in(0.0, 1.0, 2.0)  :
        vv   =  solve_bias_newton (fet(r ) , surface_models, guess  = vv, max_iterations  = 60, residual_rtol  =   1e-12,)



    d  =  terminal_currents(  fet(2.0 ),
         vv,
             surface_models )
    a= abs(sum(d.values()))


    assert  a   /  abs( d ["drain"  ] ) <  1e-8


def test_a_vanished_diffusivity_counts_as_having_moved(surface_models) :

    b   =   np.ones (  3 )
    t=replace(surface_models,Dn= np.array([0.0,2.0,0.0]),Dp=b)
    m  = replace (  surface_models,  Dn   =  np.array ([3.0,   2.0, 0.0  ]  ) ,   Dp  =  b  )

    assert  _surface_moved (  t ,  m  ) ==  np.inf
    assert _surface_moved(t,t)==0.0
    assert _surface_moved(m,m)==0.0


def test_the_surface_model_alone_solves_at_a_drain_bias() :

    s  = nmos(**   { ** COARSE,   "gate_voltage"  : 1.2 ,   "drain_voltage"  :  0.3  } )
    k=TransportModels.for_device(s,mobility='arora',surface=True)

    res=initial_state(s)
    row =  k.at_state(s, res.psi.data, res.n.data, res.p.data)
    assert np.any(np.asarray(row.Dn) ==0.0),(
        "the guess this test is about no longer has an edge with no mobility "
        "left on it, so the solve below proves nothing"
    )

    m=solve_bias_newton(s,k,max_iterations =60)


    assert m.newton.converged,m.newton.message
def  _trench_drawing(  ) :
    from ddsim.device.drawing import MOS_CAP_DRAWING, Block, drawing
    s, g, k =MOS_CAP_DRAWING;  f  =Block("oxide", 0.4e-5, 0.6e-5, 1.5e-4, 2e-4)

    return drawing(  s   +  (f,   ),
         g ,
                  k,
       nx  = 41)


def test_surface_mobility_refuses_a_vertical_interface() ->None:

    from  ddsim.device.transport  import TransportModels
    with pytest.raises(ValueError, match = 'vertical') :
        TransportModels.for_device(_trench_drawing(),
               surface =True)

def test_the_trench_still_solves_without_surface_mobility( )  ->  None  :
    from ddsim.device.transport import TransportModels

    d  =  TransportModels.for_device(_trench_drawing( ) ,  mobility  = 'arora')
    assert d.surface is None


def test_a_drawing_with_only_flat_interfaces_takes_surface_mobility()->None :

    from ddsim.device.drawing import drawing; from ddsim.device.transport  import TransportModels

    j = TransportModels.for_device(drawing(), surface = True)
    assert j.surface  is not  None
