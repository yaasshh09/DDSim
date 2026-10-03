from __future__ import annotations
import math
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
    val = [101, 201, 401, 801, 1601]
    c2=[]
    for ii in val :
        s=diode_on(ii)
        c2.append(peak_field(s, solve_equilibrium(s)))
    m  = c2[- 1]
    j  =   [abs(cur  -   m)  /  m for cur in c2[:-  1]]


    assert all(
        res2<  i for i, res2 in zip(j[:-1], j[1 :], strict  =  True)
    ), f"errors must shrink monotonically, got {j}"

def test_peak_field_converges_at_second_order ()  ->   None  :

    j= [201,401,801,1601]
    mm  =   []
    for s2 in j:
        xx =  diode_on(s2)
        mm.append(peak_field(xx, solve_equilibrium(xx)))

    t = mm[-1]
    a  =   [abs(  it -  t ) /   t for  it  in mm[:-  1]  ]

    c=[
        math.log2(r /u)
        for r,u in zip(a[:-1],a[1:],strict = True)
    ]
    assert all(lst  >  1.5 for lst  in  c) ,  f"observed orders {c}"


def test_built_in_potential_is_mesh_independent()->  None :
    kk=[]
    for idx in(51,201,801) :
        d2  =diode_on(idx)
        psi  = solve_equilibrium(d2).psi.to_physical(d2.scale).data
        kk.append(psi[-  1] - psi[0])
    b =C.V_T()* math.log(1e16 * 1e16/C.n_i()**2)
    for s2 in kk :
        assert s2==pytest.approx(b,rel= 1e-9)


def test_refinement_does_not_change_the_invariants() ->None :
    for  z  in(  51, 201,   801 ) :
        v =  diode_on(  z)

        x2  = solve_equilibrium( v  )
        np.testing.assert_allclose(x2.n.data *x2.p.data, 1.0, rtol  = 1e-8)
        assert np.all(x2.n.data> 0.0)

def test_newton_converges_in_under_ten_iterations_across_doping() ->  None :
    for zz in(1e14,
           1e15,
      1e16,
                      1e17,
                 1e18,
      1e19,
           1e20):
        dd=pn_diode(
            Na= zz, Nd =zz, length = 4.0 * MICRON, junction  = 2.0 *  MICRON
        )
        h=solve_equilibrium(dd)
        assert h.newton.iterations<10,(
            f"{zz:.0e} took {h.newton.iterations}: "
            f"{h.newton.residual_history}"
        )

def test_newton_residual_tail_is_quadratic() ->None:


    b2 =pn_diode(Na  =  1e16, Nd  =  1e16, length= 4.0  * MICRON, junction =2.0 *MICRON)

    e=solve_equilibrium(b2)

    m = np.array(e.newton.residual_history)
    row  =  m/ m[0]
    res =  row[row  > 100.0 * row[- 1]]


    xs   =  res[- 3  :]
    assert len(xs)== 3,f"no usable tail in {m}"

    for r, m2 in zip(xs[:- 1], xs[1 :], strict  =  True):
        assert m2 < r/  100.0, 'a quadratic tail step gains many digits'

    ret= [
        m2 / r**2
        for r, m2 in zip(xs[:- 1], xs[1:], strict = True)
    ]

    assert max(ret)/min(ret)<10.0,f"C is not constant: {ret}"



def test_newton_ends_with_unlimited_steps()->  None:
    r = pn_diode(Na=1e16, Nd = 1e16, length =4.0  * MICRON, junction= 2.0 *  MICRON)
    a = solve_equilibrium(r )
    assert a.newton.limited_steps  < a.newton.iterations

def test_residual_falls_by_many_orders_of_magnitude (  )   ->  None  :
    j   = pn_diode (Na = 1e16,   Nd =  1e16 ,   length  =   4.0  *  MICRON , junction   =  2.0   * MICRON  )

    s =solve_equilibrium(j)
    xs =s.newton.residual_history[0]


    b  =s.newton.residual_history[-  1] ; assert b/ xs  < 1e-14


def test_the_charge_neutral_guess_is_a_good_starting_point() -> None :
    cnt=pn_diode(Na = 1e16,Nd=1e16,length=4.0 * MICRON,junction = 2.0*MICRON)
    v   =   solve_equilibrium(  cnt )

    psi  = v.psi.data

    from  ddsim.physics.statistics  import  psi_equilibrium_scaled


    b = np.asarray(psi_equilibrium_scaled(cnt.net_doping_scaled.data))
    mm = np.abs(psi-  b)
    w =  2.0*  MICRON
    g= cnt.scale.x_0
    flag =  np.abs ( cnt.mesh.x -  w) >   40.0   * math.sqrt (
        C.eps_Si()  *  C.V_T(  )   /  (C.q *   1e16  )
    )
    assert  mm [  flag  ].max(  )   <  1e-6, f"worst {mm[flag].max():.3e}"
    assert mm.max() >1.0,'the junction must actually need solving'


    assert g> 0.0

def test_newton_converges_on_lightly_doped_material() ->None:
    for el in(1e13, 1e12, 1e11, 1e10) :
        s   = pn_diode (
            Na  =   el ,   Nd  =  el, length  =  4.0   *  MICRON, junction =  2.0   *  MICRON
        )
        x =solve_equilibrium(s)


        assert x.newton is not None

        assert x.newton.converged,(
            f"{el:.0e} did not converge: {x.newton.message}"
        )
        assert x.newton.iterations <10, (
            f"{el:.0e} took {x.newton.iterations} iterations, which "
            'means the threshold is sitting on the floor rather than above it'
        )

def test_the_threshold_floor_does_not_loosen_a_normally_doped_solve()->None :

    for j in(1e15, 1e16, 1e18) :
        obj  = pn_diode(Na=  j, Nd  =j)
        t  =  solve_equilibrium(obj)

        assert  t.newton  is  not  None
        s= float(
            np.max(
                np.abs(obj.net_doping_scaled.data)
                * obj.mesh.volume
                /  obj.scale.x_0
            )
        )

        assert t.newton.residual_history[-1]< 1e-12  +  1e-10*  s


def test_a_stalled_solve_says_what_it_was_aiming_for() ->None :
    h =  pn_diode(Na =1e16, Nd = 1e16)



    num  = solve_poisson(
        h, np.zeros(h.mesh.n_nodes), max_iterations =1
    )
    assert not num.converged
    assert "threshold" in num.message


    assert f"{num.residual_history[-1]:.3e}" in num.message
