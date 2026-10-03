from __future__ import annotations
import numpy as np
import pytest
from scipy.optimize import brentq
from ddsim.core import constants as C
from ddsim.device.doping import Coordinates
from ddsim.device.mosfet import BODY, DRAIN, GATE, SOURCE, nmos
from ddsim.device.regions import OXIDE,SILICON
from ddsim.discretize.boundary import GateContact,OhmicPlate

NA= 1e17


SD_PEAK=1e20


L_GATE= 1e-4


SD_LENGTH  = 4e-5


CONTACT_LENGTH=  2e-5
X_J =1.5e-5

LATERAL =  1e-5
T_OX  =  2e-6

T_SI  = 1e-4

H_MIN_X  =  2e-7


WIDTH   =   2.0   *  SD_LENGTH  +  L_GATE



@pytest.fixture(scope= "module")


def fet():
    return nmos(
        L_gate=L_GATE,
        sd_length =SD_LENGTH,
        contact_length=CONTACT_LENGTH,
        substrate_doping=-NA,
        sd_peak=SD_PEAK,
        x_j=X_J,
        lateral_diffusion=LATERAL,
        t_ox =T_OX,
        t_si =T_SI,
    )

def terminal(device,name):
    return next(lst for lst in device.contacts if lst.name ==  name)
def net_doping_at(device,x:float,y :float)->float:
    b = Coordinates(np.array([x]), np.array([y]))
    return float(device.doping(b) [0])



def test_the_device_has_four_terminals(fet):
    assert{b.name for b in fet.contacts} =={SOURCE,DRAIN,GATE,BODY}


def  test_the_gate_is_a_gate_and_the_rest_are_plates(fet  )  :
    assert isinstance(terminal(fet, GATE), GateContact)
    for j in(SOURCE, DRAIN, BODY) :

        assert isinstance(terminal(fet,j),OhmicPlate)




def  test_the_gate_sits_on_the_top_of_the_oxide(  fet)  :
    tmp=fet.mesh.y_axis.x[-  1]
    for m in terminal(fet, GATE).nodes :
        assert fet.mesh.node_y[m] ==tmp



def test_the_gate_spans_the_channel_and_nothing_else(fet):

    cc=np.sort(fet.mesh.node_x[list(terminal(fet, GATE).nodes)])

    assert  cc[ 0]  == pytest.approx(SD_LENGTH, rel =  1e-12 )
    assert cc[-1]==pytest.approx(SD_LENGTH+ L_GATE,rel =1e-12)

    foo= np.sort(np.unique(fet.mesh.node_x))

    rows =  foo[(foo >=cc[0]) &  (foo <=cc[- 1])]

    np.testing.assert_allclose(cc,rows,rtol=1e-12)

def test_the_source_and_drain_plates_sit_on_the_silicon_surface(fet) :
    for c in(SOURCE,DRAIN):
        for  b  in  terminal(fet ,   c).nodes   :
            assert  fet.mesh.node_y [ b  ]   ==  pytest.approx (  T_SI,   rel  =  1e-12)


def test_the_source_and_drain_contacts_stop_short_of_the_junction(fet):
    it =fet.mesh.node_x[list(terminal(fet, SOURCE).nodes)]


    i   =   fet.mesh.node_x[list ( terminal (  fet,  DRAIN).nodes  ) ]
    assert it.min() ==0.0
    assert it.max()==pytest.approx(CONTACT_LENGTH,rel= 1e-12)
    assert i.min()==pytest.approx(WIDTH-CONTACT_LENGTH,rel=1e-12)
    assert i.max() ==pytest.approx(WIDTH,rel =1e-12)
def test_the_body_contact_covers_the_whole_bottom(fet)  :
    y2=terminal(fet,BODY).nodes

    assert len(y2) ==fet.mesh.nx
    assert np.all( fet.mesh.node_y [ list(y2  )  ]  ==  0.0)

def  test_the_layers_have_the_thicknesses_asked_for(  fet) :
    v2=fet.mesh.y_axis.x

    assert v2[-  1]== pytest.approx(T_SI +  T_OX, rel =1e-14)
    assert np.count_nonzero(v2==  T_SI)  == 1

def  test_the_cells_below_the_interface_are_silicon_and_above_are_oxide( fet )   :
    assert fet.regions is not None
    item= int(np.flatnonzero(fet.mesh.y_axis.x == T_SI)[0])
    z =fet.regions.cell_material

    assert np.all(z[:item] ==SILICON)
    assert np.all(z[item  :]  ==OXIDE)



def test_the_silicon_is_graded_to_the_surface(fet):
    b=int(np.flatnonzero(fet.mesh.y_axis.x==T_SI)[0])
    t= np.diff(fet.mesh.y_axis.x[: b  +  1])
    assert t[-1]<t[0]/100.0



def test_the_mesh_is_finest_at_the_junctions (fet  )  :
    lst= np.sort(np.unique(fet.mesh.node_x))
    j =np.diff(lst)
    z = lst[np.argmin(j)]


    assert  z  ==   pytest.approx(  SD_LENGTH ,  abs =  2.0  *  float(np.min(j )  ) )




def test_no_doping_survives_inside_the_oxide(fet)  :
    assert fet.regions is not None
    j=np.zeros(fet.mesh.n_nodes,dtype =bool)
    j[fet.regions.oxide_nodes] = True

    np.testing.assert_array_equal(fet.net_doping.data[j], 0.0)
def  test_the_source_surface_reaches_the_concentration_asked_for(  fet)  :

    assert net_doping_at(fet, 0.0, T_SI) ==  pytest.approx(SD_PEAK-NA, rel= 1e-9)
def  test_the_channel_is_the_substrate(  fet )  :
    foo =net_doping_at(fet,0.5*WIDTH,T_SI)
    assert foo== pytest.approx(- NA, rel =  1e-6)



def test_the_substrate_is_the_substrate_under_the_source(fet) :
    assert net_doping_at(fet, 0.0, 0.0)== pytest.approx(- NA, rel=  1e-12)



def test_the_junction_sits_at_the_depth_asked_for (  fet ) :
    v =brentq(lambda y:net_doping_at(fet,0.0,y),T_SI - 2.0 *X_J,T_SI,xtol = 1e-16)
    assert T_SI -v ==  pytest.approx(X_J, rel =1e-9)



def test_the_junctions_encroach_under_the_gate_by_the_lateral_diffusion(fet) :
    r  =  brentq(
        lambda  x  :  net_doping_at(fet , x , T_SI  ) ,
        SD_LENGTH,
        0.5 *  WIDTH,
        xtol  =   1e-16,
    )

    assert r-SD_LENGTH== pytest.approx(LATERAL,rel=1e-9)
def test_the_metallurgical_channel_is_shorter_than_the_gate(fet) :

    c  =  brentq(
        lambda x: net_doping_at(fet, x, T_SI), SD_LENGTH, 0.5* WIDTH, xtol =1e-16
    )


    tt=brentq(
        lambda x:net_doping_at(fet,x,T_SI),
        0.5*WIDTH,
        SD_LENGTH+L_GATE,
        xtol= 1e-16,
    )
    assert tt-c==pytest.approx(
        L_GATE-2.0* LATERAL,rel=1e-9
    )
def  test_the_drain_is_the_source_mirrored(fet  )  :

    dat = fet.net_doping.data

    j=fet.mesh.node_x
    cur=np.empty(fet.mesh.n_nodes,dtype = np.int64)
    t= np.sort(np.unique(j))

    for h in  range(  fet.mesh.nx)   :
        assert t[h] ==  pytest.approx(
            WIDTH - t[fet.mesh.nx - 1- h], abs = 1e-16
        )
        for aa in range(fet.mesh.ny)  :

            cur[fet.mesh.node_at(h,aa)] =fet.mesh.node_at(fet.mesh.nx-1 - h,aa)
    np.testing.assert_allclose(dat,dat[cur],rtol=1e-12)


def test_a_gate_shorter_than_the_lateral_diffusion_is_refused():

    with pytest.raises (  ValueError, match = "no channel"  )   :
        nmos(L_gate=1e-5,lateral_diffusion=1e-5)


def test_a_contact_reaching_the_gate_edge_is_refused():
    with pytest.raises(ValueError,match="contact_length"):
        nmos(sd_length= 4e-5, contact_length = 4e-5)


def test_a_junction_deeper_than_the_silicon_is_refused( )  :
    with pytest.raises(ValueError, match = "x_j") :
        nmos(x_j=2e-4, t_si =1e-4)


def test_a_source_lighter_than_the_substrate_is_refused() :
    with pytest.raises(  ValueError,  match  =  "sd_peak" )   :

        nmos(sd_peak = 1e16,substrate_doping=-1e17)

def test_an_n_type_substrate_is_refused ( )  :

    with pytest.raises(ValueError,match='p-type'):
        nmos(substrate_doping = 1e17)


def test_a_negative_gate_length_is_refused() :
    with pytest.raises(ValueError,match ="L_gate"):
        nmos(L_gate =- 1e-4)




def  test_a_gate_work_function_can_be_chosen( ) :


    k= nmos(work_function=C.PHI_M_MIDGAP)

    assert terminal(k, GATE).work_function==  C.PHI_M_MIDGAP


def test_a_mesh_segment_with_one_node_is_refused( )  :
    with pytest.raises(ValueError,match="at least 2 nodes"):
        nmos(n_channel =1)



def test_a_gate_too_short_to_grade_is_meshed_uniformly() :
    g=nmos(L_gate=5e-6,lateral_diffusion=1e-6,x_j=2.5e-6,h_min_x=H_MIN_X)
    jj= np.sort(np.unique(g.mesh.node_x)) ; vals =  (jj  >= SD_LENGTH)   &  (  jj <= SD_LENGTH  + 5e-6) ; out2=np.diff(jj[vals])

    assert out2.max() <=H_MIN_X
    np.testing.assert_allclose(out2,out2[0],rtol=1e-12)
