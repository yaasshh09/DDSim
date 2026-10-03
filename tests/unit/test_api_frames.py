from __future__ import annotations
import json; import math, struct
import numpy as np; import pytest
from ddsim.api.frames import FieldFrame, decode_fields, encode , field_frame
from ddsim.device.equilibrium import solve_equilibrium
from ddsim.device.mos_cap import mos_cap
from ddsim.device.mosfet import nmos
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import TransportModels, solve_bias_ramped
from ddsim.extract.bands import band_edges
from ddsim.extract.cv import CVFrame
from  ddsim.extract.iv import IVFrame
from ddsim.solve.continuation import ContinuationEvent
from ddsim.solve.gummel import GummelIteration
from ddsim.solve.newton import NewtonIteration
COARSE_FET = {'n_contact' : 4, "n_sd": 10, 'n_channel': 12, 'n_silicon'  : 29, "n_oxide" : 4, "h_min_x" :  5e-7, "h_min_y" :  1e-7, "drain_voltage" : 0.05,}

def diode() :
    return pn_diode(n_nodes=61,h_min=5e-7)


def  as_json(frame)  -> dict   :
    s= encode(frame)

    assert isinstance(s, str)
    return  json.loads(s)


def header_of(message : bytes) -> dict:
    (xs, )=  struct.unpack_from("<I", message, 0)
    return json.loads(message[4: 4+  xs])



def test_a_newton_iteration_crosses_as_the_numbers_it_carries() ->None :
    k= as_json(NewtonIteration(iteration= 3,residual=1.5e-7,update=2.0e-4,damping=0.5,limited=True))

    assert  k  ==  {
        'type'  :  "newton",
        'iteration'  : 3,
        "residual"   :  1.5e-7,
        "update"  :   2.0e-4,
        'damping'  :  0.5,
        'limited'  :  True,
        "residual_by_family"  :  None,
        'update_by_family' :  None,
    }




def test_a_newton_iteration_carries_its_split_by_equation_family()->None:

    y=as_json(NewtonIteration(iteration = 2, residual =math.inf, update=1e-3, damping= 1.0, limited= False, residual_by_family = {'psi':1e-9,"n":math.inf,"p" :3e-7}, update_by_family= {"psi":1e-3,"n":2e-4,'p' :5e-5},))
    assert y["residual_by_family" ]   ==  { "psi" :   1e-9,   'n'  :  None,   'p'  : 3e-7  }
    assert  y["update_by_family"  ]  ==   {"psi"  :  1e-3 ,   "n"  :  2e-4,   "p"  :   5e-5  }


def test_the_first_newton_iteration_says_there_was_no_step()-> None  :
    tmp  = as_json(NewtonIteration(iteration  = 0, residual  = 8.0, update = None, damping =None, limited=False))

    assert tmp["update"] is None
    assert tmp["damping"]  is None

@pytest.mark.parametrize ("value",   [  math.inf,   -  math.inf, math.nan  ] )


def test_a_number_that_is_not_finite_crosses_as_null(value) ->  None:

    b2 = encode(GummelIteration(iteration=  4, update= value))
    assert "Infinity" not in b2
    assert 'NaN' not in b2
    assert json.loads(b2) ['update'] is None




def test_a_gummel_cycle_crosses_as_a_cycle()->None:
    assert as_json(GummelIteration(iteration  =  2, update = 3.0e-3)) == {
        'type'  : 'gummel',
        'iteration' :  2,
        "update" : 3.0e-3,
    }



def  test_a_continuation_attempt_carries_whether_it_was_accepted(  )  ->   None :
    vv=as_json(ContinuationEvent(parameter=0.35,step= 0.05,converged=False,message ="halving"))

    assert vv=={'type':'continuation', 'parameter': 0.35, 'step':0.05, 'converged': False, 'message': 'halving',}


def test_a_finished_current_point_crosses_as_a_point()  ->None :
    assert as_json(IVFrame(index=  2, voltage  = 0.3, current  =  1.25e-4)) == {
        "type": "point",
        "index" :  2,
        "voltage"  : 0.3,
        'current' : 1.25e-4,
    }

def test_a_finished_capacitance_point_crosses_as_its_own_kind()->None :
    b = as_json(
        CVFrame(index = 0,gate_voltage =- 1.0,capacitance=3.4e-8,charge=1e-8)
    )

    assert  b [  "type"  ]  == "cv_point"
    assert b["capacitance"]==3.4e-8


def test_something_that_is_not_a_frame_is_refused()->None:

    with pytest.raises(TypeError, match   =  "cannot be sent") :
        encode({"iteration": 1})



def test_a_field_frame_is_a_json_header_followed_by_float32()  ->None :
    ok   =  diode(  )
    w2= solve_equilibrium(ok)
    z=encode(field_frame(ok,w2,index=0,voltage=0.0))
    assert isinstance(z, bytes)
    x=  header_of(z)
    (j, )  =  struct.unpack_from("<I", z, 0)
    h=z[4+j:]
    assert x['type']=="fields"
    row = sum(r["length"]for r in x["arrays"])

    assert len(h)==4 *row




@pytest.mark.parametrize('index',[0,10,100,1000])


def test_the_float32_payload_starts_on_a_four_byte_boundary(index) ->None:
    b=  FieldFrame(
        index= index,
        voltage =0.5,
        shape= (3, ),
        arrays =  (('psi', 'V', np.array([0.1, 0.2, 0.3])), ),
    )

    r2= encode(b);(y,)=struct.unpack_from('<I',r2,0)
    assert(4+y)%4==0;  assert header_of(r2) ["index"] == index
    np.testing.assert_allclose(decode_fields(r2)  ["psi"], [0.1, 0.2, 0.3], rtol= 1e-6)


def test_the_arrays_arrive_in_the_order_the_header_lists_them()->None:

    y2= diode()
    d=solve_equilibrium(y2)


    g =decode_fields(encode(field_frame(y2,d,index = 0,voltage =0.0)))



    assert list(g)== ['x',"psi","n","p","Ec",'Ev',"Efn","Efp"] ; np.testing.assert_allclose(g['x'], y2.mesh.x, rtol =1e-6)



def test_the_fields_cross_in_physical_units() ->None:
    s2=diode()


    m = solve_equilibrium(s2)

    r  =  decode_fields(encode (field_frame (s2,   m, index  =  0,   voltage  =  0.0  )  ) )

    np.testing.assert_allclose(
        r[ "psi"] , m.psi.to_physical(  s2.scale ).data ,  rtol  =  1e-6
    )
    assert np.max(np.abs(r["psi"]))<5.0
    assert np.max(r [ "n"  ] )   > 1e15

def test_a_one_dimensional_device_says_it_has_one_axis( )  -> None :
    jj  =   diode()
    val=solve_equilibrium(jj)

    k =header_of(encode(field_frame(jj,val,index=0,voltage =0.0)))
    assert k [ 'shape' ]   ==   [  jj.mesh.n_nodes]


def test_a_two_dimensional_device_carries_both_axes_and_its_shape()   ->  None  :
    i  = nmos(**  COARSE_FET )
    d =solve_bias_ramped(i)

    s  =  encode (  field_frame(i ,  d , index  =   0, voltage  = 0.0))
    cnt =  header_of(s)
    c  = decode_fields(s)

    assert cnt["shape"] ==  [i.mesh.ny, i.mesh.nx]


    assert list(c)==["x","y","psi",'n',"p",'Ec',"Ev","Efn",'Efp']
    assert c["psi"].size ==  i.mesh.ny *  i.mesh.nx
    np.testing.assert_allclose(c["x"],i.mesh.x_axis.x,rtol=1e-6)
    np.testing.assert_allclose(c["y"],i.mesh.y_axis.x,rtol=1e-6)


def test_a_field_frame_says_which_bias_it_is_of() ->None:
    e = mos_cap(gate_voltage =-1.0)
    w2=solve_equilibrium(e)
    v =  header_of(encode(field_frame(e, w2, index=3, voltage =- 1.0)))


    assert v["index"] == 3
    assert v[ 'voltage' ]  == -  1.0

def test_the_units_of_every_array_travel_with_it ( )  ->   None :
    i =diode();  tt   =  solve_equilibrium (i  )


    f=header_of(encode(field_frame(i,tt,index= 0,voltage=0.0)))

    b ={item['name']:item["unit"]for item in f['arrays']}

    assert  b   ==   {
        "x"  :   "cm" ,
        "psi"  :   "V",
        "n"   : "cm^-3",
        "p"  :  "cm^-3",
        'Ec'  :  'eV',
        "Ev"  :  "eV",
        "Efn" : 'eV',
        'Efp' :  "eV" ,
    }



def  test_a_transport_frame_carries_the_node_currents()   ->   None :
    j= nmos(**COARSE_FET)


    buf  =   solve_bias_ramped ( j)
    x  = TransportModels.for_device(j)

    r = decode_fields(
        encode(field_frame(j, buf, index = 0, voltage =  0.0, models  =x))
    )
    assert list(r) [- 2:] ==["Jx",
         'Jy']

    assert r["Jx"].size  == j.mesh.nx * j.mesh.ny


def test_a_device_with_every_contact_at_one_bias_carries_no_current()  -> None:
    item= nmos(**COARSE_FET).with_bias(gate= 0.5,drain = 0.0)
    xx=solve_bias_ramped(item)
    m=TransportModels.for_device(item)

    buf = decode_fields(
        encode(field_frame(item,xx,index =0,voltage=0.5,models=m))
    )

    assert not np.any(buf['Jx'])
    assert  not  np.any(  buf [ "Jy"])

def test_a_drain_bias_leaves_the_current_in() -> None:
    r  = nmos (  ** COARSE_FET  ).with_bias(gate  = 0.5,   drain  = 0.1)
    lst=solve_bias_ramped(r)
    m =TransportModels.for_device(r)

    f =decode_fields(
        encode(field_frame(r, lst, index = 0, voltage=0.5, models  = m))
    )


    assert  np.nanmax (  np.abs(  f["Jx" ] )  ) >  0.0


def  test_the_band_edges_cross_in_ev()  ->  None  :
    h  =diode()
    z   =   solve_equilibrium ( h )


    tmp3 =decode_fields(encode(field_frame(h,z,index= 0,voltage= 0.0)))

    np.testing.assert_allclose(tmp3["Ec"],band_edges(h,z).Ec,rtol=1e-6)
