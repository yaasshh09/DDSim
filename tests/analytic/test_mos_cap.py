from  __future__ import annotations
import  numpy  as np

import pytest


from scipy.optimize import brentq

from ddsim.core import constants as C

from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap  import  mos_cap

NA  = 1e16


T_OX= 1e-6
T_SI=2e-4

MILLIVOLT=1e-3

GATE_TOLERANCE  =20* MILLIVOLT


def  phi_F(  net_doping :  float)  -> float   :
    return  float(- C.V_T ( )  *  np.arcsinh(net_doping  /  ( 2.0 *  C.n_i ()  ) ))



def flatband_voltage(net_doping :  float, work_function : float) ->float  :


    return float(C.work_function_difference(work_function,net_doping))
def  oxide_capacitance(t_ox   :   float ) ->  float  :
    return C.eps_ox() /  t_ox


def max_depletion_width( net_doping :   float)  ->   float  :
    Na= abs(net_doping)
    return float(
        np.sqrt(2.0  * C.eps_Si()*  2.0  *phi_F(net_doping)  /  (C.q * Na))
    )



def threshold_voltage(net_doping : float,t_ox:float,work_function :float) -> float:


    Na =  abs(net_doping)

    q_depp  = C.q  *  Na  *  max_depletion_width(  net_doping)
    return(flatband_voltage( net_doping,  work_function) +   2.0 *  phi_F(net_doping ) +  q_depp  / oxide_capacitance(  t_ox  ))



def solved(net_doping  =-NA, work_function= C.PHI_M_N_POLY, ** kwargs) :
    Device  = mos_cap(substrate_doping  =net_doping, t_ox =  T_OX, t_si =  T_SI, work_function =work_function, **kwargs,)
    return Device,solve_equilibrium(Device)
def surface_potential(device,state) ->float:
    mes=device.mesh
    assert device.regions is not None
    buff = device.regions.interface_nodes(mes)
    coulmn  =   mes.nx  // 2
    bul  =mes.node_at(coulmn, 0);  ndoe  =int(buff[coulmn])
    return float((state.psi.data[ndoe]- state.psi.data[bul]) *  device.scale.psi_0)


def gate_bias_for ( target_psi_s   : float ,  bracket  =   (  -   3.0,   3.0),   **  kwargs )  ->   float   :

    def residual(v_gate:float)-> float :


        device, state = solved(gate_voltage =  v_gate, ** kwargs)

        return surface_potential(device,
                          state) - target_psi_s
    return float( brentq(  residual , *   bracket,   xtol =  1e-9  ))

@pytest.mark.parametrize('work_function', [C.PHI_M_N_POLY,  C.PHI_M_P_POLY, C.PHI_M_MIDGAP  ], ids  =  ["n+poly", "p+poly" ,   'midgap' ],)



@pytest.mark.parametrize("net_doping", [-  1e15, -  1e16, - 1e17], ids =  str)
def test_biasing_the_gate_at_phi_ms_leaves_the_stack_flat(
    work_function,net_doping
):
    v_fbb  =  flatband_voltage(  net_doping ,   work_function  )
    dvice,sta= solved(
        net_doping= net_doping,work_function =work_function,gate_voltage=v_fbb
    )
    psi  = sta.psi.data
    assert np.ptp(psi) <1e-11,f"psi spread {np.ptp(psi):g} in scaled units"
def test_the_flat_profile_is_already_the_answer():
    _, State= solved(gate_voltage =flatband_voltage(-NA, C.PHI_M_N_POLY))
    assert State.newton.iterations == 0



@pytest.mark.parametrize(
    'work_function',
    [C.PHI_M_N_POLY,C.PHI_M_P_POLY,C.PHI_M_MIDGAP],
    ids = ["n+poly","p+poly","midgap"],
)


def test_the_bias_that_removes_the_band_bending_is_the_flatband_voltage(work_function,):
    stuff2= gate_bias_for(0.0,work_function =work_function)
    exp=flatband_voltage(- NA,work_function)
    assert stuff2  == pytest.approx(exp, abs  =GATE_TOLERANCE)
    assert stuff2 == pytest.approx(exp, abs = 1e-5)




@pytest.mark.parametrize("net_doping",[-1e15,- 1e16,- 1e17],ids=str)
def  test_the_threshold_bias_matches_the_textbook_expression(net_doping  )  :
    tagret=2.0 * phi_F(net_doping)
    fou =gate_bias_for(tagret,net_doping=net_doping)
    Expected= threshold_voltage(net_doping,T_OX,C.PHI_M_N_POLY)
    assert fou==pytest.approx(Expected, abs= GATE_TOLERANCE)

    assert fou ==pytest.approx(Expected,abs= MILLIVOLT)

def test_the_threshold_moves_with_the_oxide_thickness_as_one_over_c_ox() :
    map  = 2.0 * phi_F(-NA)

    def residual(v_gate):
        device   = mos_cap (
            substrate_doping   =- NA,   t_ox   =  2  *  T_OX , t_si =  T_SI, gate_voltage  =   v_gate
        )
        return surface_potential(device,solve_equilibrium(device))-map

    Found = float(brentq(residual, -  3.0, 3.0, xtol= 1e-9))

    exxpected  =threshold_voltage(-NA, 2  * T_OX, C.PHI_M_N_POLY)
    sorted = threshold_voltage(-NA,T_OX,C.PHI_M_N_POLY)
    assert exxpected  -   sorted  ==  pytest.approx (C.q   *  NA  *   max_depletion_width(  - NA )   / oxide_capacitance(2  *  T_OX )   / 2 , rel = 1e-12 ,)
    assert Found  ==  pytest.approx(exxpected, abs =GATE_TOLERANCE)
def surface_charge_exact(net_doping :float,
               psi_s: float)-> float:
    Na =abs(net_doping)
    uu  =  psi_s/C.V_T()
    R  =   (C.n_i(  )   /  Na  )  **  2;Bracket = np.exp(-uu)+ uu-1.0+R *(np.exp(uu)- uu -1.0)
    return float(  np.sqrt(  2.0  *  C.eps_Si()  *  C.V_T( )  *   C.q   * Na *  Bracket))




def surface_field ( device, state)   ->  float  :

    mseh= device.mesh
    assert device.regions is not None; Column =mseh.nx  // 2
    sruface = int(device.regions.interface_nodes (  mseh  )   [ Column ])
    hash   =  sruface  -  mseh.nx
    psi =state.psi.data*device.scale.psi_0
    return float(
        (psi[sruface]- psi[hash])
        /(mseh.node_y[sruface] -mseh.node_y[hash])
    )




@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75], ids  = str)



def  test_the_surface_charge_matches_poisson_boltzmann(fraction ) :

    format   =  fraction *  2.0   *   phi_F( - NA)
    vGate = gate_bias_for(format)
    Device,tmp2 = solved(gate_voltage=vGate)
    PsiS=surface_potential(Device,tmp2)
    assert PsiS == pytest.approx(format,abs=1e-6)


    measued=C.eps_Si()*surface_field(Device,tmp2)
    assert measued  ==pytest.approx(surface_charge_exact(- NA, PsiS), rel  = 2e-3)
    map=np.sqrt(2.0* C.eps_Si()*C.q * NA*PsiS)
    assert abs(measued /map  -1.0)  >  0.02




def test_the_depletion_approximation_is_exact_at_threshold_by_cancellation()  :

    for net in(-1e15,- 1e16,-1e17):
        psii_s  =  2.0  *  phi_F ( net);  depleion  =  np.sqrt(2.0 * C.eps_Si()  *C.q  * abs(net)  * psii_s)
        assert surface_charge_exact( net, psii_s )  ==  pytest.approx (depleion ,   rel  =  1e-6)



def test_the_bulk_is_neutral_far_from_the_surface()  :
    Device , junk  =   solved(gate_voltage =  1.0 )
    object =Device.mesh
    hash = object.nx //  2
    d2 =object.node_at(hash, 0)
    oneUp =object.node_at(hash,1)
    assert abs(junk.psi.data[d2] -junk.psi.data[oneUp])<1e-9

def  test_the_oxide_potential_is_a_straight_line ()  :

    deevice ,   State =   solved ( gate_voltage =   1.0 )
    object  =   deevice.mesh
    assert deevice.regions is not None
    surfaceRow = int(deevice.regions.interface_nodes(object)  [0]) //object.nx

    cnt=  object.nx  // 2

    obj2=range(surfaceRow,object.ny)
    yy  =  np.array(  [  object.node_y[object.node_at( cnt, jj  )  ]  for  jj  in  obj2 ]  )
    psi =  np.array([State.psi.data[object.node_at(cnt, jj)] for jj in obj2])
    stuff=  np.polyfit(yy, psi, 1)
    hex= psi-np.polyval(stuff,yy)
    assert np.max(np.abs(hex)) <1e-10 * np.ptp(psi)

@pytest.mark.parametrize("v_gate", [-  2.0, 0.0, 1.0], ids = ['acc', 'zero', "inv"])

def test_no_carrier_density_is_negative_anywhere(v_gate) :

    devvice,sttate=solved(gate_voltage= v_gate) ; car  = np.asarray(devvice.charge_volume_scaled)>  0.0


    assert np.all(sttate.n.data[car]> 0.0)
    assert np.all(sttate.p.data[car] > 0.0)

    val= ~car
    assert  np.any (val ), 'this device has no oxide, so it checks nothing'
    assert np.all(sttate.n.data[val]  ==0.0);assert np.all(sttate.p.data[val]== 0.0)
def  test_the_solution_does_not_vary_across_the_device(  )  :


    bin, sta = solved(gate_voltage=  1.0)
    thing   =  bin.mesh

    psi  =  sta.psi.data.reshape( thing.ny,   thing.nx)
    spead  =np.ptp(psi, axis  = 1)
    assert np.max(spead) < 1e-12


def test_adding_columns_changes_nothing()  :
    _,Coarse=solved(gate_voltage =1.0,nx=3)
    lst, id=solved(gate_voltage =  1.0, nx = 7)
    ret  = Coarse.psi.data.reshape(  - 1,  3) [:,   0]
    fineColumn  = id.psi.data.reshape(- 1, 7)[:, 0]
    np.testing.assert_allclose(fineColumn,ret,rtol=1e-10)




def test_the_quasi_fermi_levels_are_undefined_in_the_oxide():

    arr,State =solved(gate_voltage =1.0); Carriers=np.asarray(arr.charge_volume_scaled)> 0.0
    ins= ~ Carriers
    assert np.any(ins), 'this device has no oxide, so it checks nothing'
    phi_n = State.phi_n.data
    phi_p =State.phi_p.data

    assert np.all(np.isfinite(phi_n[Carriers]))
    assert np.all(np.isfinite(phi_p[Carriers]))
    assert np.all(np.isnan(phi_n[ins]))
    assert np.all(np.isnan(phi_p[ins]))
