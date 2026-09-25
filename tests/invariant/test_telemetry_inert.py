from __future__ import annotations
from typing import Any

import numpy as np; import pytest
from ddsim.device.mos_cap import mos_cap

from ddsim.device.mosfet import nmos ; from ddsim.device.pn_diode import pn_diode; from ddsim.extract.cv import cv_sweep

from ddsim.extract.iv import  gate_sweep, iv_sweep



MICRON= 1e-4


COARSE_FET={"n_contact" :4, "n_sd":10, "n_channel":12, "n_silicon":29, 'n_oxide':4, 'h_min_x': 5e-7, "h_min_y" :1e-7, "drain_voltage": 0.05,}



def diode() :
    return pn_diode(Na =1e16, Nd=1e16, length= 12* MICRON, junction = 6 *MICRON, n_nodes=61, h_min= 5e-7,)


def solver_history(state:Any)->list[float] :
    if state.newton is not None :
        return list(state.newton.residual_history)
    assert state.gummel is not None
    return list(state.gummel.update_history)


@pytest.mark.parametrize(('what','sweep'), [('diode', lambda on_frame:iv_sweep(diode(),'anode',[0.1,0.3],on_frame =on_frame),), ("mosfet", lambda on_frame: gate_sweep(nmos(**COARSE_FET),[0.2,0.4],step=0.2,on_frame=on_frame),),], ids=["diode","mosfet"],)


def test_a_current_sweep_is_bit_for_bit_unchanged_by_watching_it(what, sweep) :
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
