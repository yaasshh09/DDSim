from __future__ import annotations
import inspect
import pytest
from  ddsim.device.mosfet import  nmos
from ddsim.device.transport import TransportModels
from  ddsim.extract.iv  import gate_sweep
from ddsim.extract.rolloff import SHORT_CHANNEL_PROCESS
L_GATE  = 1e-4



DRAIN=0.05

GATES  =  ( 0.0 ,   1.5 )

CONVERGED =  5e-3


def working_mesh ()  ->  dict[str ,   float | int]  :

    b =  inspect.signature(nmos).parameters
    return {'n_silicon'   :  b [  'n_silicon' ].default, 'n_oxide' :   b[ "n_oxide"].default, "h_min_y"  :   b[ "h_min_y" ].default,}


def  halve(  mesh  :   dict[ str , float |   int  ]  )   ->  dict[  str ,   float |  int ]   :
    return{
        'n_silicon' : 2 * int(mesh['n_silicon'])  - 1,
        'n_oxide'  : 2 * int(mesh['n_oxide'])-  1,
        "h_min_y" : float(mesh["h_min_y"]) / 2.0,
    }

def double(mesh :dict[str,float|int]) ->dict[str,float|int] :
    return{
        "n_silicon": (int(mesh['n_silicon'])+ 1)//2,
        "n_oxide": (int(mesh["n_oxide"]) +1)// 2,
        'h_min_y':float(mesh["h_min_y"])*2.0,
    }



def drain_current(mesh  : dict[str, float |  int])  ->list[float]  :
    ret =nmos(L_gate=L_GATE, drain_voltage=DRAIN, degenerate=False, **mesh, **SHORT_CHANNEL_PROCESS,)
    f  = TransportModels.for_device(ret, mobility= "constant")
    ys = gate_sweep(ret,list(GATES),models=f)
    assert ys.complete,ys.message
    return[float(thing) for thing in ys.current]




@pytest.fixture(scope="module")


def ladder()  ->  dict [  str, list [  float]]  :
    tmp3  =  working_mesh(  )
    return{"coarse" : drain_current(double(tmp3)), 'working' :drain_current(tmp3), "fine" : drain_current(halve(tmp3)),}



@pytest.mark.parametrize('index', range(len(GATES)), ids =  [f"Vg{v:g}" for v in GATES])


def test_the_working_mesh_is_already_converged(
    ladder :  dict[str,   list[  float ]  ], index   : int
)   ->   None  :

    y   =   ladder ["working"  ] [index ]
    mm =ladder["fine"][index]
    x  =abs(mm  - y) / abs(mm)
    assert x < CONVERGED, (
        f"at Vg = {GATES[index]} V the drain current still moves {x:.2%} "
        f"when the vertical mesh is halved, which is not comfortably inside "
        f"the 5 percent benchmark 6 asserts. Refine the nmos defaults."
    )


@pytest.mark.parametrize("index", range(  len ( GATES  )  ) , ids   =  [ f"Vg{v:g}" for  v  in GATES  ]  )




def test_the_refinement_converges_rather_than_wandering(
    ladder :dict[str,list[float]],index:int
)->None :
    yy =ladder["coarse"] [index]; e = ladder["working"][index]
    res = ladder['fine'][index]
    assert abs(res -e)  <  abs(e  - yy), (
        f"at Vg = {GATES[index]} V the last halving moved the drain current "
        f"{abs(res - e):.4g} against {abs(e - yy):.4g} for "
        'the one before it, so this is drift and not convergence'
    )



def test_the_oxide_and_silicon_spacings_stay_matched() ->None  :


    k  = working_mesh( )
    y   =  SHORT_CHANNEL_PROCESS [ 't_ox'  ]
    j=y/(int(k['n_oxide']) -1)
    tmp3= j/float(k['h_min_y'])
    assert tmp3 == pytest.approx(1.0,rel =1e-12),(
        f"the oxide cell is {j:.3e} cm against a silicon surface "
        f"spacing of {k['h_min_y']:.3e}, a seam ratio of {tmp3:.3g}. "
        "Refine n_oxide alongside h_min_y."
    )
