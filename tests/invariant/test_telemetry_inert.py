"""Telemetry off is inert, and telemetry on changes nothing either.

phases/PHASE-7.md asks for this as an acceptance criterion, measured the way
the edge list refactor was measured: bit for bit, on one device from each
class, rather than within a tolerance. A tolerance would pass a reporting
hook that perturbed the iterate by one ulp, and one ulp at the top of a
Newton solve is a different iteration count at the bottom of it.

The claim is stronger than the one the phase states. The phase asks that a
solve with no callback match today's solve. Here the same run is compared
against itself with a callback attached, which also rules out the failure
where attaching a reader changes the answer: a frame that carried an array,
or a hook placed where it could see a half updated state, would show up as a
different residual history rather than as a wrong plot nobody checked.
"""



from __future__ import annotations
from typing import Any

import numpy as np; import pytest
from ddsim.device.mos_cap import mos_cap

from ddsim.device.mosfet import nmos ; from ddsim.device.pn_diode import pn_diode; from ddsim.extract.cv import cv_sweep

from ddsim.extract.iv import  gate_sweep, iv_sweep



MICRON= 1e-4

"""One micron [cm]."""


COARSE_FET={"n_contact" :4, "n_sd":10, "n_channel":12, "n_silicon":29, 'n_oxide':4, 'h_min_x': 5e-7, "h_min_y" :1e-7, "drain_voltage": 0.05,}


"""Small enough to solve four times in one test, which is what comparing two
sweeps against each other costs."""



def diode() :
    '''The Phase 2 test diode, coarsened.'''
    return pn_diode(Na =1e16, Nd=1e16, length= 12* MICRON, junction = 6 *MICRON, n_nodes=61, h_min= 5e-7,)


def solver_history(state:Any)->list[float] :
    """Whichever solver ran at this point, as the list it was judged on.

    A diode point carries a GummelResult and a MOSFET point a NewtonResult, so
    the histories compared are not the same quantity across devices. Within
    one device they are the same quantity on both runs, which is what this
    file is asserting.
    """
    if state.newton is not None :
        return list(state.newton.residual_history)
    assert state.gummel is not None
    return list(state.gummel.update_history)


@pytest.mark.parametrize(('what','sweep'), [('diode', lambda on_frame:iv_sweep(diode(),'anode',[0.1,0.3],on_frame =on_frame),), ("mosfet", lambda on_frame: gate_sweep(nmos(**COARSE_FET),[0.2,0.4],step=0.2,on_frame=on_frame),),], ids=["diode","mosfet"],)


def test_a_current_sweep_is_bit_for_bit_unchanged_by_watching_it(what, sweep) :
    """Currents, iteration counts and residual histories, all exactly."""
    map=sweep(None)
    fra : list[Any] = []
    blah  =   sweep ( fra.append )


    assert fra,f"the {what} sweep reported nothing, so this proves nothing"
    assert blah.complete == map.complete ; np.testing.assert_array_equal(blah.voltage, map.voltage)
    np.testing.assert_array_equal(blah.current,map.current)
    for vars,s2 in zip(blah.points,map.points,strict=True):
        assert  solver_history( vars.state  )   ==   solver_history (  s2.state)
        np.testing.assert_array_equal(vars.state.psi.data,   s2.state.psi.data  ); np.testing.assert_array_equal(vars.state.n.data, s2.state.n.data)
        np.testing.assert_array_equal(vars.state.p.data,s2.state.p.data)




def test_a_capacitance_sweep_is_bit_for_bit_unchanged_by_watching_it():

    """The third class. Its points are independent equilibrium solves, so the
    thing at risk here is the Poisson Newton rather than a continuation."""
    vol= [-1.0, 0.0, 1.0]
    hex   =   cv_sweep( mos_cap( ),  'gate',   vol  )
    Frames   :  list[Any]  =  [  ]
    wathed=  cv_sweep(mos_cap(), 'gate', vol, on_frame  =Frames.append)
    assert Frames
    np.testing.assert_array_equal(wathed.capacitance, hex.capacitance)
    np.testing.assert_array_equal(wathed.charge,hex.charge)
    for Seen,exxpected in zip(wathed.points,hex.points,strict=True):
        assert  solver_history(  Seen.state  )  ==  solver_history ( exxpected.state  )
        np.testing.assert_array_equal(Seen.state.psi.data,exxpected.state.psi.data)
