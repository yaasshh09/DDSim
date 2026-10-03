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
    z2,y2,u = MOS_CAP_DRAWING
    y   = drawing(z2 , y2,   u,  nx =  3 ,  ny =  125,   h_min_y  = 5e-8,   degenerate   =  False)
    f= cv_sweep(mos_cap(),"gate",CV_BIASES)
    w= cv_sweep(y, "gate", CV_BIASES)
    assert f.complete and w.complete
    np.testing.assert_allclose (w.capacitance,   f.capacitance , rtol =  CV_TOLERANCE)
@pytest.fixture(scope = 'module')


def transfer_curves ( ) ->  tuple[  np.ndarray ,   np.ndarray ] :
    aa =[]
    for j in(  nmos(  drain_voltage =  0.05) ,  drawing().with_bias(drain  =  0.05 ) )  :
        m  =   gate_sweep ( j,   GATES, models =  TransportModels.for_device (j )); assert m.complete ,  m.message
        aa.append(np.asarray(m.current))
    return aa[0] ,  aa [1 ]

def  test_the_drawn_nmos_reproduces_nmos_id_vg_above_the_floor(transfer_curves ,)  -> None   :
    dat, i = transfer_curves
    s = np.abs(dat) >=  ID_FLOOR ; assert  s.sum( ) >=  9,   'the comparison has to cover the curve above threshold'; np.testing.assert_allclose(i[s],dat[s],rtol= ID_TOLERANCE)

def test_the_drawn_nmos_is_off_where_nmos_is_off(transfer_curves) -> None:
    buf,h= transfer_curves
    i= np.abs(buf) <ID_FLOOR

    assert  i.any();  assert np.all( np.abs(  h [i  ]  ) < ID_FLOOR)
