from __future__ import annotations


import numpy as np
import pytest

from ddsim.device.mosfet  import nmos



from  ddsim.device.transport  import(
    TransportModels ,
    solve_bias_newton ,
    solve_bias_ramped ,
)
from  ddsim.extract.rolloff import  SHORT_CHANNEL_PROCESS
DRAIN_BIAS =1.0




@pytest.fixture(  scope   =   "module"  )


def hot_device():
    return nmos(
        L_gate= 1e-4,
        gate_voltage = 0.0,
        drain_voltage = DRAIN_BIAS,
        ** SHORT_CHANNEL_PROCESS,
    )


@pytest.fixture(scope= 'module')




def models(hot_device):

    return TransportModels.for_device(hot_device, mobility= 'arora', field_dependent =  True, surface = True)


@pytest.fixture(scope ="module")



def ramped(hot_device,  models  )   :
    return solve_bias_ramped(hot_device, models = models)


@pytest.fixture(scope="module")
def  direct(hot_device,   models)   :
    return solve_bias_newton(hot_device, models=  models, guess = None)
def test_the_ramp_converges(ramped):


    assert ramped.newton is not None
    assert ramped.newton.converged, ramped.newton.message


def test_the_ramp_lands_on_the_same_root(ramped, direct) :

    assert  direct.newton is  not  None and direct.newton.converged; assert ramped.psi.data==pytest.approx(direct.psi.data, abs = 1e-8)

    for stuff2,directSide in((ramped.n.data,direct.n.data), (ramped.p.data,direct.p.data),):
        rel=  np.abs(stuff2 -  directSide)  /  (np.abs(directSide) + 1.0)
        assert np.max(rel) <1e-8

def test_the_final_step_is_a_newton_solve_and_not_a_limited_walk(ramped,direct):
    assert direct.newton is not None and ramped.newton is not None
    assert  direct.newton.limited_steps  >  0;  assert ramped.newton.limited_steps==0




def test_a_device_at_zero_bias_is_not_disturbed_by_the_ramp(models):

    bar= nmos(
        L_gate= 1e-4, gate_voltage =0.0, drain_voltage =  0.0, **SHORT_CHANNEL_PROCESS
    )

    x2= solve_bias_ramped(bar,models=models)
    t2=solve_bias_newton(bar,models=models,guess=None)

    assert x2.newton is not None and x2.newton.converged
    assert x2.psi.data == pytest.approx(t2.psi.data,abs=1e-10)



def test_the_ramp_builds_its_own_models_when_given_none(hot_device) :
    Ramped  =solve_bias_ramped(hot_device)
    assert Ramped.newton is not None and Ramped.newton.converged
