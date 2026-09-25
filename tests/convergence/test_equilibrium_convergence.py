"""Mesh refinement and Newton convergence studies.

Two of the Phase 1 acceptance criteria live here:

- error against the analytic V_bi decreases with h
- Newton converges in under 10 iterations from the charge neutral guess, and
  quadratically over the last few

The refinement study is the one that would catch a discretization that is
consistent but not convergent, which no single mesh can.
"""
from __future__ import annotations
import  math

import numpy as np

import pytest
from ddsim.core import constants as C



from ddsim.device.equilibrium import solve_equilibrium,solve_poisson

from ddsim.device.pn_diode import pn_diode


MICRON=1e-4
"""One micron [cm]."""
def diode_on(n_nodes  : int, Na : float= 1e16, Nd : float =1e16) :

    """A uniformly refined diode, so that h halves as n_nodes doubles."""
    return pn_diode(
        Na=Na,
        Nd=Nd,
        length= 4.0*MICRON,
        junction= 2.0 *MICRON,
        n_nodes= n_nodes,
        h_min=4.0*MICRON /(n_nodes-1),
    )



def peak_field(device, state) ->float:
    """Largest field magnitude in the device [V/cm]."""
    psi  =   state.psi.to_physical(device.scale  ).data

    return  float(np.max(  np.abs( -   np.diff (psi )  / device.mesh.h )  ) )


def  test_peak_field_converges_under_mesh_refinement()   ->  None  :
    """The field is where discretization error shows up first.

    V_bi itself is pinned by the contacts, so it cannot be used to measure
    convergence of the interior scheme. The peak field is genuinely computed
    and it is the quantity a coarse mesh gets worst, since it lives at the
    junction where the curvature is highest.
    """
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
    """Box integration of the Laplacian is second order on a uniform mesh.

    Halving h should quarter the error. Anything close to first order would
    mean the dual cell volumes or the face fluxes are subtly wrong.
    """

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
    """It is set by the contacts, so refinement must not move it at all."""
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
    """The acceptance criterion, over the full doping range Phase 5 will need."""
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


    """Each step squares the residual once inside the basin of attraction.

    A typical history for this device is

        1.13e6, 7.16e5, 3.06e5, 46.3, 4.18, 2.93e-2, 1.27e-6, 2.09e-11

    The first three steps are step limited and only linear. The tail is the
    last three above the roundoff floor, where the residual falls by five
    orders of magnitude per step.

    Quadratic is checked without having to guess the constant: if
    r_{k+1} = C * r_k^2 then C is the same at every step, so the test asserts
    that the measured ratio stays put rather than that it hits some value.
    Linear convergence would make the ratio grow by orders of magnitude.
    """

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
    """A converged solve must finish with real Newton steps, not clamped ones.

    If the step limiter were still active at the end, the tail would be linear
    and the quadratic claim would be false.
    """
    dev = pn_diode(Na=1e16, Nd = 1e16, length =4.0  * MICRON, junction= 2.0 *  MICRON)
    sta = solve_equilibrium(dev )
    assert sta.newton.limited_steps  < sta.newton.iterations

def test_residual_falls_by_many_orders_of_magnitude (  )   ->  None  :
    devcie   = pn_diode (Na = 1e16,   Nd =  1e16 ,   length  =   4.0  *  MICRON , junction   =  2.0   * MICRON  )

    State =solve_equilibrium(devcie)
    fir =State.newton.residual_history[0]


    las  =State.newton.residual_history[-  1] ; assert las/ fir  < 1e-14


def test_the_charge_neutral_guess_is_a_good_starting_point() -> None :
    """psi = asinh(N/2) should already be right everywhere except the junction.

    If the initial residual were large across the whole device rather than
    concentrated near the junction, the guess would not be doing its job.
    """
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
    """The doping range above stops at 1e14, and below it the solve used to fail.

    The residual threshold is built from the doping charge in the largest dual
    cell, because that is the size of the terms the residual is made of and it
    does not depend on where the iteration started. But the residual has a
    second half, the difference of the two face fluxes, and that difference
    cannot be resolved below machine epsilon times the size of the fluxes
    themselves. The two scale in opposite directions: the charge falls with
    the doping while psi is only logarithmic in it, so the flux floor stays
    put.

    Below about 1e13 the charge threshold sinks under the flux floor, and then
    nothing can meet it. Measured on the 1e12 bar before the fix: the residual
    reached 6.8e-12 at the third iteration and sat there, unchanged to the last
    bit, for the remaining forty-seven, with an update of 4.4e-16 the whole
    time. Newton had solved it at step three and then reported failure.

    High resistivity substrates run at exactly these dopings, so this is a
    range the project needs rather than a curiosity.
    """
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

    """The floor is a floor, not an addition.

    Raising the threshold to clear the flux floor must not touch any device
    where the doping charge already clears it, otherwise every diode in the
    project silently converges to a looser answer than it used to.
    """
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
    """The message has to carry the threshold, not just the residual.

    A residual that has stopped moving while the update is already tiny means
    the threshold is under the arithmetic floor. Without the number to compare
    against, that reads exactly like a solve that is merely slow, which is a
    different problem with a different fix.
    """
    dev =  pn_diode(Na =1e16, Nd = 1e16)



    open  = solve_poisson(
        dev, np.zeros(dev.mesh.n_nodes), max_iterations =1
    )
    assert not open.converged
    assert "threshold" in open.message


    assert f"{open.residual_history[-1]:.3e}" in open.message
