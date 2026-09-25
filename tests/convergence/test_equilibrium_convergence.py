from __future__ import annotations
import  math

import numpy as np

import pytest
from ddsim.core import constants as C



from ddsim.device.equilibrium import solve_equilibrium,solve_poisson

from ddsim.device.pn_diode import pn_diode


MICRON=1e-4
def diode_on(n_nodes  : int, Na : float= 1e16, Nd : float =1e16) :

    return pn_diode(
        Na=Na,
        Nd=Nd,
        length= 4.0*MICRON,
        junction= 2.0 *MICRON,
        n_nodes= n_nodes,
        h_min=4.0*MICRON /(n_nodes-1),
    )



def peak_field(device, state) ->float:
    psi  =   state.psi.to_physical(device.scale  ).data

    return  float(np.max(  np.abs( -   np.diff (psi )  / device.mesh.h )  ) )


def  test_peak_field_converges_under_mesh_refinement()   ->  None  :
    couunts = [101, 201, 401, 801, 1601]
    Fields=[]
    for NNodes in couunts :
        idx2=diode_on(NNodes)
        Fields.append(peak_field(idx2, solve_equilibrium(idx2)))
    ref  = Fields[- 1]
    Errors  =   [abs(buff  -   ref)  /  ref for buff in Fields[:-  1]]


    assert all(
        later<  earlier for earlier, later in zip(Errors[:-1], Errors[1 :], strict  =  True)
    ), f"errors must shrink monotonically, got {Errors}"

def test_peak_field_converges_at_second_order ()  ->   None  :

    filter= [201,401,801,1601]
    Fields  =   []
    for yy in filter:
        Device =  diode_on(yy)
        Fields.append(peak_field(Device, solve_equilibrium(Device)))

    refrence = Fields[-1]
    buf  =   [abs(  stuff -  refrence ) /   refrence for  stuff  in Fields[:-  1]  ]

    oders=[
        math.log2(ear /lat)
        for ear,lat in zip(buf[:-1],buf[1:],strict = True)
    ]
    assert all(order  >  1.5 for order  in  oders) ,  f"observed orders {oders}"


def test_built_in_potential_is_mesh_independent()->  None :
    buff=[]
    for nnodes in(51,201,801) :
        dev  =diode_on(nnodes)
        psi  = solve_equilibrium(dev).psi.to_physical(dev.scale).data
        buff.append(psi[-  1] - psi[0])
    Expected =C.V_T()* math.log(1e16 * 1e16/C.n_i()**2)
    for val in buff :
        assert val==pytest.approx(Expected,rel= 1e-9)


def test_refinement_does_not_change_the_invariants() ->None :
    for  hmm  in(  51, 201,   801 ) :
        Device =  diode_on(  hmm)

        min  = solve_equilibrium( Device  )
        np.testing.assert_allclose(min.n.data *min.p.data, 1.0, rtol  = 1e-8)
        assert np.all(min.n.data> 0.0)

def test_newton_converges_in_under_ten_iterations_across_doping() ->  None :
    for Doping in(1e14,
           1e15,
      1e16,
                      1e17,
                 1e18,
      1e19,
           1e20):
        vals=pn_diode(
            Na= Doping, Nd =Doping, length = 4.0 * MICRON, junction  = 2.0 *  MICRON
        )
        k2=solve_equilibrium(vals)
        assert k2.newton.iterations<10,(
            f"{Doping:.0e} took {k2.newton.iterations}: "
            f"{k2.newton.residual_history}"
        )

def test_newton_residual_tail_is_quadratic() ->None:


    dev =pn_diode(Na  =  1e16, Nd  =  1e16, length= 4.0  * MICRON, junction =2.0 *MICRON)

    temp2=solve_equilibrium(dev)

    his = np.array(temp2.newton.residual_history)
    reelative  =  his/ his[0]
    usa =  reelative[reelative  > 100.0 * reelative[- 1]]


    x2   =  usa[- 3  :]
    assert len(x2)== 3,f"no usable tail in {his}"

    for Previous, dat in zip(x2[:- 1], x2[1 :], strict  =  True):
        assert dat < Previous/  100.0, 'a quadratic tail step gains many digits'

    rat= [
        dat / Previous**2
        for Previous, dat in zip(x2[:- 1], x2[1:], strict = True)
    ]

    assert max(rat)/min(rat)<10.0,f"C is not constant: {rat}"



def test_newton_ends_with_unlimited_steps()->  None:
    dev = pn_diode(Na=1e16, Nd = 1e16, length =4.0  * MICRON, junction= 2.0 *  MICRON)
    sta = solve_equilibrium(dev )
    assert sta.newton.limited_steps  < sta.newton.iterations

def test_residual_falls_by_many_orders_of_magnitude (  )   ->  None  :
    devcie   = pn_diode (Na = 1e16,   Nd =  1e16 ,   length  =   4.0  *  MICRON , junction   =  2.0   * MICRON  )

    State =solve_equilibrium(devcie)
    fir =State.newton.residual_history[0]


    las  =State.newton.residual_history[-  1] ; assert las/ fir  < 1e-14


def test_the_charge_neutral_guess_is_a_good_starting_point() -> None :
    deviice=pn_diode(Na = 1e16,Nd=1e16,length=4.0 * MICRON,junction = 2.0*MICRON)
    sttae   =   solve_equilibrium(  deviice )

    psi  = sttae.psi.data

    from  ddsim.physics.statistics  import  psi_equilibrium_scaled


    xx = np.asarray(psi_equilibrium_scaled(deviice.net_doping_scaled.data))
    Difference = np.abs(psi-  xx)
    obj2 =  2.0*  MICRON
    temp2= deviice.scale.x_0
    Far =  np.abs ( deviice.mesh.x -  obj2) >   40.0   * math.sqrt (
        C.eps_Si()  *  C.V_T(  )   /  (C.q *   1e16  )
    )
    assert  Difference [  Far  ].max(  )   <  1e-6, f"worst {Difference[Far].max():.3e}"
    assert Difference.max() >1.0,'the junction must actually need solving'


    assert temp2> 0.0

def test_newton_converges_on_lightly_doped_material() ->None:
    for Doping in(1e13, 1e12, 1e11, 1e10) :
        open   = pn_diode (
            Na  =   Doping ,   Nd  =  Doping, length  =  4.0   *  MICRON, junction =  2.0   *  MICRON
        )
        sta =solve_equilibrium(open)


        assert sta.newton is not None

        assert sta.newton.converged,(
            f"{Doping:.0e} did not converge: {sta.newton.message}"
        )
        assert sta.newton.iterations <10, (
            f"{Doping:.0e} took {sta.newton.iterations} iterations, which "
            'means the threshold is sitting on the floor rather than above it'
        )

def test_the_threshold_floor_does_not_loosen_a_normally_doped_solve()->None :

    for d2 in(1e15, 1e16, 1e18) :
        Device  = pn_diode(Na=  d2, Nd  =d2)
        sttae  =  solve_equilibrium(Device)

        assert  sttae.newton  is  not  None
        Charge= float(
            np.max(
                np.abs(Device.net_doping_scaled.data)
                * Device.mesh.volume
                /  Device.scale.x_0
            )
        )

        assert sttae.newton.residual_history[-1]< 1e-12  +  1e-10*  Charge


def test_a_stalled_solve_says_what_it_was_aiming_for() ->None :
    dev =  pn_diode(Na =1e16, Nd = 1e16)



    open  = solve_poisson(
        dev, np.zeros(dev.mesh.n_nodes), max_iterations =1
    )
    assert not open.converged
    assert "threshold" in open.message


    assert f"{open.residual_history[-1]:.3e}" in open.message
