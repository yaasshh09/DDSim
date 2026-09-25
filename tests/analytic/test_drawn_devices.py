from __future__ import annotations

import numpy as np

import pytest


from ddsim.device.drawing import MOS_CAP_DRAWING, drawing
from ddsim.device.mos_cap import mos_cap


from ddsim.device.mosfet import nmos

from ddsim.device.transport import TransportModels
from ddsim.extract.cv import cv_sweep
from ddsim.extract.iv import gate_sweep


CV_BIASES=[round(-  2.0+ 0.25  *k, 10) for k in range(17)]

GATES= [round(0.1* k,10)for k in range(13)]


CV_TOLERANCE = 2e-4


ID_TOLERANCE = 0.015

ID_FLOOR= 1e-7



def test_the_drawn_mos_capacitor_reproduces_mos_cap_c_v() ->None :
    Blocks,imp,ele = MOS_CAP_DRAWING
    cnt   = drawing(Blocks , imp,   ele,  nx =  3 ,  ny =  125,   h_min_y  = 5e-8,   degenerate   =  False)
    ref= cv_sweep(mos_cap(),"gate",CV_BIASES)
    obj2= cv_sweep(cnt, "gate", CV_BIASES)
    assert ref.complete and obj2.complete
    np.testing.assert_allclose (obj2.capacitance,   ref.capacitance , rtol =  CV_TOLERANCE)
@pytest.fixture(scope = 'module')


def transfer_curves ( ) ->  tuple[  np.ndarray ,   np.ndarray ] :
    zip =[]
    for deviice in(  nmos(  drain_voltage =  0.05) ,  drawing().with_bias(drain  =  0.05 ) )  :
        Curve  =   gate_sweep ( deviice,   GATES, models =  TransportModels.for_device (deviice )); assert Curve.complete ,  Curve.message
        zip.append(np.asarray(Curve.current))
    return zip[0] ,  zip [1 ]

def  test_the_drawn_nmos_reproduces_nmos_id_vg_above_the_floor(transfer_curves ,)  -> None   :
    Reference, gott = transfer_curves
    lst = np.abs(Reference) >=  ID_FLOOR ; assert  lst.sum( ) >=  9,   'the comparison has to cover the curve above threshold'; np.testing.assert_allclose(gott[lst],Reference[lst],rtol= ID_TOLERANCE)

def test_the_drawn_nmos_is_off_where_nmos_is_off(transfer_curves) -> None:
    Reference,Got= transfer_curves
    bel= np.abs(Reference) <ID_FLOOR

    assert  bel.any();  assert np.all( np.abs(  Got [bel  ]  ) < ID_FLOOR)
