"""The MOS capacitor builder: what it constructs, and what it refuses.

The physics is tested in tests/analytic/test_mos_cap.py. This is about the
stack being the stack that was asked for, because almost every way of getting
it wrong still solves. An oxide half a cell thicker than requested converges
perfectly and reports a capacitance that is a percent low, and there is nothing
in the solution to say so.
"""

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
    item2=  cap.mesh.y_axis.x
    assert item2[- 1] ==pytest.approx(T_SI +T_OX,rel=1e-14)
    assert np.count_nonzero(item2== T_SI) ==1



def test_the_interface_lands_exactly_on_a_node_line(cap  )   :
    """Exactly. device/regions.py refuses anything else, and the reason it
    refuses is that half a cell of oxide is a percent of t_ox."""
    assert cap.mesh.y_axis.x[120  ] ==   T_SI

def test_the_cells_above_the_interface_are_oxide_and_below_are_silicon(cap) :

    assert cap.regions is not None
    Material=cap.regions.cell_material
    assert np.all(Material[:120] ==SILICON)
    assert np.all ( Material[120  :  ]  == OXIDE )
def test_the_oxide_is_uniformly_meshed(cap) :

    """Its potential is a straight line, so uniform cells resolve it exactly
    and grading would spend nodes on nothing.

    Uniform to 1e-12 rather than exactly. Stacking translates the oxide up by
    t_si, so its 2.5 nm cells are recovered by differencing numbers eight
    hundred times larger, and that costs about three digits. Measured spread
    1.1e-13 relative, which is eight orders below anything physical.
    """
    h= np.diff(cap.mesh.y_axis.x[120:])
    np.testing.assert_allclose(h, h[0], rtol= 1e-12)



def test_the_silicon_is_graded_to_the_surface(cap):
    """The inversion layer is a couple of nanometres thick and the substrate
    is two microns, so one spacing cannot serve both."""
    hsilicon=  np.diff(cap.mesh.y_axis.x[: 121])
    assert hsilicon[- 1] == pytest.approx(5e-8, rel = 1e-9)
    assert hsilicon[0]>100 *hsilicon[- 1]
    assert np.all(np.diff(hsilicon)<0.0),'spacing must fall towards the top'




def test_the_default_substrate_is_thicker_than_the_depletion_region() :
    """W_max at 1e16 is 304 nm. A substrate that is not several times that has
    its body contact holding the depletion region open, which moves every
    capacitance in the C-V curve."""
    yy   =  mos_cap(  ).mesh.y_axis.x ; assert yy[-1]  - 1e-6  > 5* 3.04e-5



def test_the_body_is_a_plate_across_the_whole_bottom_edge(cap) :
    bod  =next(c for c in cap.contacts if c.name == BODY)
    assert isinstance(bod, OhmicPlate) ; assert bod.nodes==tuple(range(cap.mesh.nx))

def test_the_gate_is_a_plate_across_the_whole_top_edge(cap) :
    acc  =  next (c for c in cap.contacts  if c.name  ==  GATE)
    assert  isinstance (acc, GateContact) ; Top  = cap.mesh.n_nodes-cap.mesh.nx
    assert acc.nodes == tuple(range(Top,cap.mesh.n_nodes))


def test_the_gate_carries_its_work_function_and_the_body_does_not():
    """A gate reads its own metal and never looks down; an ohmic contact reads
    the doping under it and has no work function of its own."""


    capp =  mos_cap(work_function = C.PHI_M_P_POLY)
    gaate = next(c for c in capp.contacts if c.name == GATE)
    assert  isinstance(gaate, GateContact  )
    assert gaate.work_function == C.PHI_M_P_POLY
    assert not hasattr(next(c for c in capp.contacts if c.name==BODY),
                       'work_function')



def test_the_biases_land_on_the_terminals_they_name() :
    capp  =  mos_cap(gate_voltage =   1.5 , body_voltage =-   0.25)

    bia   =  { blah.name :  blah.voltage for blah  in  capp.contacts  };  assert bia =={GATE:1.5,BODY : -0.25}

def test_the_substrate_is_uniformly_doped_and_the_oxide_is_not_doped(cap) :
    """The zero charge volume already makes the oxide doping irrelevant to the
    equations. This is for everything else that reads the array."""
    assert cap.regions is not None

    dop=cap.net_doping.data
    Silicon = cap.regions.semiconductor_volume > 0.0

    np.testing.assert_allclose(dop[Silicon],- NA,rtol =1e-14)
    np.testing.assert_array_equal(dop[~ Silicon], 0.0)


def test_an_n_type_substrate_is_the_sign_flip_and_nothing_else():
    """Net doping is the convention everywhere in this codebase, so a PMOS
    body is one minus sign rather than a different code path."""
    pt =  mos_cap(substrate_doping=- NA)
    ntype =mos_cap(substrate_doping =+NA)

    np.testing.assert_allclose(ntype.net_doping.data,- pt.net_doping.data,rtol= 1e-14)


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

    """One node is a surface, not a layer. It has no thickness to carry a
    field across, and the oxide's whole job is to carry one."""
    with  pytest.raises (ValueError, match  =  'at least 2 nodes') :
        mos_cap( **   kwargs)



def  test_a_surface_spacing_too_coarse_for_the_substrate_is_reported ()   :
    """graded_mesh_1d refuses a mesh whose neighbouring cells jump too hard,
    rather than returning one whose truncation error looks like physics."""
    with pytest.raises(ValueError, match=  "max_ratio"):
        mos_cap(n_silicon  =  8, h_min  =1e-8)



def  test_a_capacitor_reports_itself_as_a_2d_device (  cap)  :
    txt   =  repr ( cap  )
    assert 'by' in txt
    assert GATE in txt and BODY in txt
