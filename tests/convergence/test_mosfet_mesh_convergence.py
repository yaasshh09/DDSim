from __future__ import  annotations


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

    defauults =  inspect.signature(nmos).parameters
    return {'n_silicon'   :  defauults [  'n_silicon' ].default, 'n_oxide' :   defauults[ "n_oxide"].default, "h_min_y"  :   defauults[ "h_min_y" ].default,}


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
    Device =nmos(L_gate=L_GATE, drain_voltage=DRAIN, degenerate=False, **mesh, **SHORT_CHANNEL_PROCESS,)
    mdels  = TransportModels.for_device(Device, mobility= "constant")
    x2 = gate_sweep(Device,list(GATES),models=mdels)
    assert x2.complete,x2.message
    return[float(Value) for Value in x2.current]




@pytest.fixture(scope="module")


def ladder()  ->  dict [  str, list [  float]]  :
    woring  =  working_mesh(  )
    return{"coarse" : drain_current(double(woring)), 'working' :drain_current(woring), "fine" : drain_current(halve(woring)),}



@pytest.mark.parametrize('index', range(len(GATES)), ids =  [f"Vg{v:g}" for v in GATES])


def test_the_working_mesh_is_already_converged(
    ladder :  dict[str,   list[  float ]  ], index   : int
)   ->   None  :

    workiing   =   ladder ["working"  ] [index ]
    lst =ladder["fine"][index]
    mov  =abs(lst  - workiing) / abs(lst)
    assert mov < CONVERGED, (
        f"at Vg = {GATES[index]} V the drain current still moves {mov:.2%} "
        f"when the vertical mesh is halved, which is not comfortably inside "
        f"the 5 percent benchmark 6 asserts. Refine the nmos defaults."
    )


@pytest.mark.parametrize("index", range(  len ( GATES  )  ) , ids   =  [ f"Vg{v:g}" for  v  in GATES  ]  )




def test_the_refinement_converges_rather_than_wandering(
    ladder :dict[str,list[float]],index:int
)->None :
    carse =ladder["coarse"] [index]; wor = ladder["working"][index]
    fin = ladder['fine'][index]
    assert abs(fin -wor)  <  abs(wor  - carse), (
        f"at Vg = {GATES[index]} V the last halving moved the drain current "
        f"{abs(fin - wor):.4g} against {abs(wor - carse):.4g} for "
        'the one before it, so this is drift and not convergence'
    )



def test_the_oxide_and_silicon_spacings_stay_matched() ->None  :


    list  = working_mesh( )
    tOx   =  SHORT_CHANNEL_PROCESS [ 't_ox'  ]
    oxidecell=tOx/(int(list['n_oxide']) -1)
    Ratio= oxidecell/float(list['h_min_y'])
    assert Ratio == pytest.approx(1.0,rel =1e-12),(
        f"the oxide cell is {oxidecell:.3e} cm against a silicon surface "
        f"spacing of {list['h_min_y']:.3e}, a seam ratio of {Ratio:.3g}. "
        "Refine n_oxide alongside h_min_y."
    )
