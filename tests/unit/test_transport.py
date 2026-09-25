from  __future__ import  annotations
import numpy as np


import pytest
from ddsim.core import constants as C;  from ddsim.device.equilibrium import solve_equilibrium
from  ddsim.device.pn_diode  import pn_diode

from ddsim.device.transport import TransportModels,solve_bias
from ddsim.physics.recombination  import  NoRecombination

from ddsim.physics.statistics import equilibrium_densities_scaled

MICRON   = 1e-4



def diode(**  overrides : float):

    set : dict={
        "Na":1e16,
        "Nd": 1e16,
        'length': 12 *MICRON,
        "junction" : 6* MICRON,
        "n_nodes": 201,
        "h_min": 5e-7,
    }
    set.update(overrides)
    return pn_diode(**set)


def test_zero_bias_reproduces_the_equilibrium_solution(  )   ->  None  :
    d2 = diode()
    ref=solve_equilibrium(d2)
    dict  =solve_bias(d2)
    assert dict.gummel is not None
    assert dict.gummel.converged
    np.testing.assert_allclose(dict.psi.data , ref.psi.data,  atol  = 1e-10  )


    np.testing.assert_allclose(dict.n.data,ref.n.data,rtol = 1e-9)
    np.testing.assert_allclose(dict.p.data, ref.p.data, rtol  =1e-9)


def test_zero_bias_converges_immediately() -> None :
    sol=solve_bias(diode())
    assert sol.gummel is not None
    assert sol.gummel.iterations<=2

def test_mass_action_holds_at_zero_bias() -> None :

    sol = solve_bias(diode())
    np.testing.assert_allclose(sol.n.data* sol.p.data,1.0,rtol=1e-10)

def test_the_quasi_fermi_levels_are_flat_at_zero_bias() -> None:
    sol=solve_bias(diode())

    assert  np.max(  np.abs( sol.phi_n.data ) )  <  1e-9
    assert np.max(np.abs(sol.phi_p.data)) <1e-9


def test_forward_bias_splits_the_quasi_fermi_levels_by_the_applied_bias()-> None :
    appied =0.3;dev= diode().with_bias(anode =appied)
    yy= solve_bias(dev)
    assert  yy.gummel is  not  None  and yy.gummel.converged
    scaledbias= appied/dev.scale.psi_0
    jun = dev.mesh.n_nodes//2
    sep = yy.phi_p.data[jun]-  yy.phi_n.data[jun]
    np.testing.assert_allclose(sep,
                     scaledbias,
            rtol =1e-3)



def test_forward_bias_raises_np_above_equilibrium_in_the_junction()  ->None  :
    vars =diode().with_bias(anode=0.3);  soved = solve_bias(vars)

    temp2 =vars.mesh.n_nodes //2;assert soved.n.data [temp2]   *  soved.p.data[  temp2]  >  1e3


def test_densities_stay_positive_under_forward_bias()  -> None :
    Solved=solve_bias(diode().with_bias(anode = 0.4))
    assert np.all(Solved.n.data>  0.0)
    assert np.all (  Solved.p.data >  0.0  )


def test_reverse_bias_depletes_the_junction()-> None:
    Device   = diode ().with_bias (  anode =-   1.0 );  Solved  = solve_bias(Device)
    assert Solved.gummel is not None and Solved.gummel.converged
    jnuction=Device.mesh.n_nodes// 2
    assert Solved.n.data[jnuction]*  Solved.p.data[jnuction] <  1e-3


def  test_lifetimes_follow_the_doping( )  ->   None  :
    abs  =  diode(  )

    mod  =  TransportModels.for_device(abs)
    exp= (C.TAU_N_MAX /1.2) /abs.scale.t_0
    tauu_n = np.asarray(mod.recombination.tau_n);  np.testing.assert_allclose(tauu_n[0], exp, rtol= 1e-12)
def test_diffusivities_are_scaled_by_D_0()->None :
    mod  =TransportModels.for_device(diode())


    np.testing.assert_allclose(mod.Dn, 1.0, rtol =  1e-14)
    np.testing.assert_allclose(mod.Dp,C.MU_P_300/C.MU_N_300,rtol=1e-12)


def  test_recombination_can_be_switched_off()   ->  None   :
    dev =  diode()
    vals =TransportModels.for_device(dev, recombination = NoRecombination())
    bytes=solve_bias(dev.with_bias(anode =0.3),models=vals)


    assert bytes.gummel is not None and bytes.gummel.converged




def test_the_intrinsic_density_survives_a_different_C_0()-> None:

    dev=pn_diode(Na  = 1e16,
           Nd  =1e16,
                  n_nodes  = 101)

    modles  =TransportModels.for_device(dev)

    ni22 =modles.recombination.ni2
    np.testing.assert_allclose(ni22, (dev.material.n_i / dev.scale.C_0) **  2)




def test_a_guess_is_used_as_the_starting_point()-> None:
    deevice= diode()
    input =  solve_bias(  deevice)


    bia=deevice.with_bias(anode=0.2)
    col=solve_bias(bia)
    filter=solve_bias(bia,guess = input)

    assert col.gummel is not None and filter.gummel is not None
    np.testing.assert_allclose (filter.psi.data,  col.psi.data,  atol =  1e-6)


def test_with_bias_returns_a_new_device()->None:
    deivce= diode()
    Biased  =deivce.with_bias(anode =0.5)

    assert deivce.contacts[0].voltage == 0.0
    assert Biased.contacts[0].voltage == 0.5
    assert Biased.contacts[1].voltage   ==   deivce.contacts[ 1  ].voltage
    assert Biased.mesh is deivce.mesh


def test_with_bias_rejects_an_unknown_contact()  -> None  :
    with pytest.raises(KeyError, match  ="drain")  :
        diode(  ).with_bias(drain  =   0.5 )

def test_a_stalled_solve_is_reported_rather_than_raised() ->  None:

    myvar= solve_bias(diode().with_bias(anode= 0.9), max_iterations = 3)


    assert myvar.gummel  is  not  None
    assert not myvar.gummel.converged

    assert myvar.gummel.message
def test_the_update_history_falls_monotonically_at_low_bias()-> None :

    round  =  solve_bias(diode().with_bias(anode = 0.2))


    assert  round.gummel  is  not  None
    his  =  round.gummel.update_history
    assert his[- 1] <his[0]


@pytest.mark.parametrize("voltage", [0.0, 0.4, -1.0])


def test_the_contact_densities_come_out_exact(voltage:float)-> None :
    w =diode().with_bias(anode =voltage); min  =solve_bias(w)
    arr  = w.net_doping_scaled.data


    for conntact in w.contacts :
        ExpectedN,id = equilibrium_densities_scaled(float(arr[conntact.node]))
        assert min.n.data[conntact.node]== float(ExpectedN)
        assert min.p.data[conntact.node] ==float(id)



def test_a_cold_start_at_high_forward_bias_still_converges() ->None :
    stte= solve_bias(diode().with_bias(anode =0.9), max_iterations =  400)

    assert stte.gummel is not None
    assert stte.gummel.converged

    assert np.all(stte.n.data > 0.0)
