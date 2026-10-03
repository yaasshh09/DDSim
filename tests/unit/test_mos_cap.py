from __future__ import annotations
import numpy as np
import pytest
from ddsim.core import constants as C
from ddsim.device.mos_cap import BODY,GATE,mos_cap
from ddsim.device.regions import OXIDE, SILICON
from ddsim.discretize.boundary import GateContact, OhmicPlate
NA= 1e16
T_OX = 1e-6


T_SI  = 2e-4



@pytest.fixture(scope= 'module')



def cap():


    return mos_cap(substrate_doping =-NA,t_ox =T_OX,t_si =T_SI)
def  test_the_layers_have_the_thicknesses_asked_for( cap)  :
    m=  cap.mesh.y_axis.x
    assert m[- 1] ==pytest.approx(T_SI +T_OX,rel=1e-14)
    assert np.count_nonzero(m== T_SI) ==1



def test_the_interface_lands_exactly_on_a_node_line(cap  )   :
    assert cap.mesh.y_axis.x[120  ] ==   T_SI

def test_the_cells_above_the_interface_are_oxide_and_below_are_silicon(cap) :

    assert cap.regions is not None
    j=cap.regions.cell_material
    assert np.all(j[:120] ==SILICON)
    assert np.all ( j[120  :  ]  == OXIDE )
def test_the_oxide_is_uniformly_meshed(cap) :

    j= np.diff(cap.mesh.y_axis.x[120:])
    np.testing.assert_allclose(j, j[0], rtol= 1e-12)



def test_the_silicon_is_graded_to_the_surface(cap):
    k=  np.diff(cap.mesh.y_axis.x[: 121])
    assert k[- 1] == pytest.approx(5e-8, rel = 1e-9)
    assert k[0]>100 *k[- 1]
    assert np.all(np.diff(k)<0.0),'spacing must fall towards the top'




def test_the_default_substrate_is_thicker_than_the_depletion_region() :
    thing   =  mos_cap(  ).mesh.y_axis.x ; assert thing[-1]  - 1e-6  > 5* 3.04e-5



def test_the_body_is_a_plate_across_the_whole_bottom_edge(cap) :
    yy  =next(f for f in cap.contacts if f.name == BODY)
    assert isinstance(yy, OhmicPlate) ; assert yy.nodes==tuple(range(cap.mesh.nx))

def test_the_gate_is_a_plate_across_the_whole_top_edge(cap) :
    vals  =  next (x for x in cap.contacts  if x.name  ==  GATE)
    assert  isinstance (vals, GateContact) ; a  = cap.mesh.n_nodes-cap.mesh.nx
    assert vals.nodes == tuple(range(a,cap.mesh.n_nodes))


def test_the_gate_carries_its_work_function_and_the_body_does_not():


    aa =  mos_cap(work_function = C.PHI_M_P_POLY)
    num = next(val2 for val2 in aa.contacts if val2.name == GATE)
    assert  isinstance(num, GateContact  )
    assert num.work_function == C.PHI_M_P_POLY
    assert not hasattr(next(m for m in aa.contacts if m.name==BODY),
                       'work_function')



def test_the_biases_land_on_the_terminals_they_name() :
    y  =  mos_cap(gate_voltage =   1.5 , body_voltage =-   0.25)

    r2   =  { u.name :  u.voltage for u  in  y.contacts  };  assert r2 =={GATE:1.5,BODY : -0.25}

def test_the_substrate_is_uniformly_doped_and_the_oxide_is_not_doped(cap) :
    assert cap.regions is not None

    v=cap.net_doping.data
    a = cap.regions.semiconductor_volume > 0.0

    np.testing.assert_allclose(v[a],- NA,rtol =1e-14)
    np.testing.assert_array_equal(v[~ a], 0.0)


def test_an_n_type_substrate_is_the_sign_flip_and_nothing_else():
    aa =  mos_cap(substrate_doping=- NA)
    val2 =mos_cap(substrate_doping =+NA)

    np.testing.assert_allclose(val2.net_doping.data,- aa.net_doping.data,rtol= 1e-14)


@pytest.mark.parametrize('bad', [0.0, - 1e-6], ids =  ['zero', "negative"])




def test_a_layer_with_no_thickness_is_refused( bad )  :
    with pytest.raises(ValueError,match ="t_ox must be positive"):
        mos_cap(t_ox = bad)
    with pytest.raises(ValueError, match = 't_si must be positive') :
        mos_cap(t_si =bad)

@pytest.mark.parametrize(
    "kwargs",[{"n_oxide":1},{'n_silicon':1}],ids=['oxide','silicon']
)

def test_a_layer_with_one_node_is_refused(kwargs) :

    with  pytest.raises (ValueError, match  =  'at least 2 nodes') :
        mos_cap( **   kwargs)



def  test_a_surface_spacing_too_coarse_for_the_substrate_is_reported ()   :
    with pytest.raises(ValueError, match=  "max_ratio"):
        mos_cap(n_silicon  =  8, h_min  =1e-8)



def  test_a_capacitor_reports_itself_as_a_2d_device (  cap)  :
    k2   =  repr ( cap  )
    assert 'by' in k2
    assert GATE in k2 and BODY in k2
