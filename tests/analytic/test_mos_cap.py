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

    val2  = C.q  *  Na  *  max_depletion_width(  net_doping)
    return(flatband_voltage( net_doping,  work_function) +   2.0 *  phi_F(net_doping ) +  val2  / oxide_capacitance(  t_ox  ))



def solved(net_doping  =-NA, work_function= C.PHI_M_N_POLY, ** kwargs) :
    d  = mos_cap(substrate_doping  =net_doping, t_ox =  T_OX, t_si =  T_SI, work_function =work_function, **kwargs,)
    return d,solve_equilibrium(d)
def surface_potential(device,state) ->float:
    buf=device.mesh
    assert device.regions is not None
    e = device.regions.interface_nodes(buf)
    b2  =   buf.nx  // 2
    v  =buf.node_at(b2, 0);  val  =int(e[b2])
    return float((state.psi.data[val]- state.psi.data[v]) *  device.scale.psi_0)


def gate_bias_for ( target_psi_s   : float ,  bracket  =   (  -   3.0,   3.0),   **  kwargs )  ->   float   :

    def residual(v_gate:float)-> float :


        t, y = solved(gate_voltage =  v_gate, ** kwargs)

        return surface_potential(t,
                          y) - target_psi_s
    return float( brentq(  residual , *   bracket,   xtol =  1e-9  ))

@pytest.mark.parametrize('work_function', [C.PHI_M_N_POLY,  C.PHI_M_P_POLY, C.PHI_M_MIDGAP  ], ids  =  ["n+poly", "p+poly" ,   'midgap' ],)



@pytest.mark.parametrize("net_doping", [-  1e15, -  1e16, - 1e17], ids =  str)
def test_biasing_the_gate_at_phi_ms_leaves_the_stack_flat(
    work_function,net_doping
):
    s  =  flatband_voltage(  net_doping ,   work_function  )
    h,num= solved(
        net_doping= net_doping,work_function =work_function,gate_voltage=s
    )
    psi  = num.psi.data
    assert np.ptp(psi) <1e-11,f"psi spread {np.ptp(psi):g} in scaled units"
def test_the_flat_profile_is_already_the_answer():
    _, yy= solved(gate_voltage =flatband_voltage(-NA, C.PHI_M_N_POLY))
    assert yy.newton.iterations == 0



@pytest.mark.parametrize(
    'work_function',
    [C.PHI_M_N_POLY,C.PHI_M_P_POLY,C.PHI_M_MIDGAP],
    ids = ["n+poly","p+poly","midgap"],
)


def test_the_bias_that_removes_the_band_bending_is_the_flatband_voltage(work_function,):
    k= gate_bias_for(0.0,work_function =work_function)
    j=flatband_voltage(- NA,work_function)
    assert k  == pytest.approx(j, abs  =GATE_TOLERANCE)
    assert k == pytest.approx(j, abs = 1e-5)




@pytest.mark.parametrize("net_doping",[-1e15,- 1e16,- 1e17],ids=str)
def  test_the_threshold_bias_matches_the_textbook_expression(net_doping  )  :
    f=2.0 * phi_F(net_doping)
    j =gate_bias_for(f,net_doping=net_doping)
    yy= threshold_voltage(net_doping,T_OX,C.PHI_M_N_POLY)
    assert j==pytest.approx(yy, abs= GATE_TOLERANCE)

    assert j ==pytest.approx(yy,abs= MILLIVOLT)

def test_the_threshold_moves_with_the_oxide_thickness_as_one_over_c_ox() :
    x  = 2.0 * phi_F(-NA)

    def residual(v_gate):
        r   = mos_cap (
            substrate_doping   =- NA,   t_ox   =  2  *  T_OX , t_si =  T_SI, gate_voltage  =   v_gate
        )
        return surface_potential(r,solve_equilibrium(r))-x

    info = float(brentq(residual, -  3.0, 3.0, xtol= 1e-9))

    j  =threshold_voltage(-NA, 2  * T_OX, C.PHI_M_N_POLY)
    m = threshold_voltage(-NA,T_OX,C.PHI_M_N_POLY)
    assert j  -   m  ==  pytest.approx (C.q   *  NA  *   max_depletion_width(  - NA )   / oxide_capacitance(2  *  T_OX )   / 2 , rel = 1e-12 ,)
    assert info  ==  pytest.approx(j, abs =GATE_TOLERANCE)
def surface_charge_exact(net_doping :float,
               psi_s: float)-> float:
    Na =abs(net_doping)
    cc  =  psi_s/C.V_T()
    a  =   (C.n_i(  )   /  Na  )  **  2;prev = np.exp(-cc)+ cc-1.0+a *(np.exp(cc)- cc -1.0)
    return float(  np.sqrt(  2.0  *  C.eps_Si()  *  C.V_T( )  *   C.q   * Na *  prev))




def surface_field ( device, state)   ->  float  :

    tmp2= device.mesh
    assert device.regions is not None; ok =tmp2.nx  // 2
    res = int(device.regions.interface_nodes (  tmp2  )   [ ok ])
    res2   =  res  -  tmp2.nx
    psi =state.psi.data*device.scale.psi_0
    return float(
        (psi[res]- psi[res2])
        /(tmp2.node_y[res] -tmp2.node_y[res2])
    )




@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75], ids  = str)



def  test_the_surface_charge_matches_poisson_boltzmann(fraction ) :

    r   =  fraction *  2.0   *   phi_F( - NA)
    stuff = gate_bias_for(r)
    dd,h = solved(gate_voltage=stuff)
    g=surface_potential(dd,h)
    assert g == pytest.approx(r,abs=1e-6)


    a=C.eps_Si()*surface_field(dd,h)
    assert a  ==pytest.approx(surface_charge_exact(- NA, g), rel  = 2e-3)
    f=np.sqrt(2.0* C.eps_Si()*C.q * NA*g)
    assert abs(a /f  -1.0)  >  0.02




def test_the_depletion_approximation_is_exact_at_threshold_by_cancellation()  :

    for y in(-1e15,- 1e16,-1e17):
        j  =  2.0  *  phi_F ( y);  f  =  np.sqrt(2.0 * C.eps_Si()  *C.q  * abs(y)  * j)
        assert surface_charge_exact( y, j )  ==  pytest.approx (f ,   rel  =  1e-6)



def test_the_bulk_is_neutral_far_from_the_surface()  :
    u , r  =   solved(gate_voltage =  1.0 )
    v =u.mesh
    it = v.nx //  2
    m =v.node_at(it, 0)
    k =v.node_at(it,1)
    assert abs(r.psi.data[m] -r.psi.data[k])<1e-9

def  test_the_oxide_potential_is_a_straight_line ()  :

    dat ,   y =   solved ( gate_voltage =   1.0 )
    aa  =   dat.mesh
    assert dat.regions is not None
    val2 = int(dat.regions.interface_nodes(aa)  [0]) //aa.nx

    h=  aa.nx  // 2

    z=range(val2,aa.ny)
    x  =  np.array(  [  aa.node_y[aa.node_at( h, a  )  ]  for  a  in  z ]  )
    psi =  np.array([y.psi.data[aa.node_at(h, a)] for a in z])
    k=  np.polyfit(x, psi, 1)
    s= psi-np.polyval(k,x)
    assert np.max(np.abs(s)) <1e-10 * np.ptp(psi)

@pytest.mark.parametrize("v_gate", [-  2.0, 0.0, 1.0], ids = ['acc', 'zero', "inv"])

def test_no_carrier_density_is_negative_anywhere(v_gate) :

    m,k=solved(gate_voltage= v_gate) ; mm  = np.asarray(m.charge_volume_scaled)>  0.0


    assert np.all(k.n.data[mm]> 0.0)
    assert np.all(k.p.data[mm] > 0.0)

    ret= ~mm
    assert  np.any (ret ), 'this device has no oxide, so it checks nothing'
    assert np.all(k.n.data[ret]  ==0.0);assert np.all(k.p.data[ret]== 0.0)
def  test_the_solution_does_not_vary_across_the_device(  )  :


    c, h = solved(gate_voltage=  1.0)
    v   =  c.mesh

    psi  =  h.psi.data.reshape( v.ny,   v.nx)
    val  =np.ptp(psi, axis  = 1)
    assert np.max(val) < 1e-12


def test_adding_columns_changes_nothing()  :
    _,k=solved(gate_voltage =1.0,nx=3)
    r, d=solved(gate_voltage =  1.0, nx = 7)
    b2  = k.psi.data.reshape(  - 1,  3) [:,   0]
    j  = d.psi.data.reshape(- 1, 7)[:, 0]
    np.testing.assert_allclose(j,b2,rtol=1e-10)




def test_the_quasi_fermi_levels_are_undefined_in_the_oxide():

    x,z =solved(gate_voltage =1.0); obj=np.asarray(x.charge_volume_scaled)> 0.0
    u= ~ obj
    assert np.any(u), 'this device has no oxide, so it checks nothing'
    phi_n = z.phi_n.data
    phi_p =z.phi_p.data

    assert np.all(np.isfinite(phi_n[obj]))
    assert np.all(np.isfinite(phi_p[obj]))
    assert np.all(np.isnan(phi_n[u]))
    assert np.all(np.isnan(phi_p[u]))
