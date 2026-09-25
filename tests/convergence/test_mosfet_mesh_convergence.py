"""Vertical mesh refinement of the MOSFET transfer curve.

The inversion layer is a nanometre or so thick and it is the only structure on
this device a mesh can miss. Everything Phase 5 claims, threshold voltage,
subthreshold slope, roll-off, DIBL, is read off a drain current that is carried
by that layer, so a drain current still moving under vertical refinement makes
every one of those numbers a statement about the mesh.

Measured on the 1 um device of `SHORT_CHANNEL_PROCESS` at 50 mV of drain, with
the silicon surface spacing and the oxide spacing halved together so the
Si/SiO2 seam ratio stays at 1.000 (see docs/05-pitfalls.md for what happens
when only one of them moves):

     41 si /  5 ox   h=5.00e-8   2835 nodes   Vg=0: 1.497272e-6   Vg=1.5: 1.257808
     61 si /  9 ox   h=2.50e-8   4347 nodes   Vg=0: 1.425611e-6   Vg=1.5: 1.247874
     81 si / 17 ox   h=1.25e-8   6111 nodes   Vg=0: 1.404635e-6   Vg=1.5: 1.242417
    101 si / 33 ox   h=6.25e-9   8379 nodes   Vg=0: 1.398684e-6   Vg=1.5: 1.240255
    121 si / 65 ox   h=3.13e-9  11655 nodes   Vg=0: 1.396946e-6   Vg=1.5: 1.239580

The moves at Vg = 0 are 4.79, 1.47, 0.42 and 0.12 percent, shrinking by a
consistent factor of about three, which is convergence rather than drift.

Subthreshold is the sensitive end, because there the current is drain junction
leakage computed as a difference of much larger fluxes, so it is the bias the
tolerance below is really set by.

Why the line is a tenth of the benchmark tolerance
--------------------------------------------------
`tests/regression/test_devsim_mosfet.py::test_golden_reference_is_converged`
holds the DEVSIM reference to a tenth of the 5 percent it is used to assert,
on the grounds that a reference carrying more mesh error than that is what the
comparison ends up measuring. The same argument applies to ddsim's half of the
comparison, and the two meshes are matched at the same surface spacing by
construction, so they have to clear the same bar or the benchmark row is a
statement about discretisation rather than about physics.

At the first four rows above ddsim did not clear it. That is what moved the
`nmos` defaults, and it moved every MOSFET number in the README with them.
"""

from __future__ import  annotations


import inspect
import pytest

from  ddsim.device.mosfet import  nmos



from ddsim.device.transport import TransportModels


from  ddsim.extract.iv  import gate_sweep

from ddsim.extract.rolloff import SHORT_CHANNEL_PROCESS
L_GATE  = 1e-4



"""The long device [cm]. No short channel effect left in it at 1 um."""
DRAIN=0.05

"""Linear region for the whole sweep, so no part of this is pinch off [V]."""

GATES  =  ( 0.0 ,   1.5 )
"""Subthreshold and full on [V]. The two ends the refinement is checked at."""

CONVERGED =  5e-3

'''Relative move allowed between the working mesh and the next halving [1].

A tenth of the 5 percent benchmark 6 asserts, which is the line
test_devsim_mosfet.py already draws on the DEVSIM side of the same comparison.
'''


def working_mesh ()  ->  dict[str ,   float | int]  :
    """The vertical mesh `nmos` uses when nobody passes one.

    Read off the signature rather than copied, so that this ladder is a claim
    about the default a caller actually gets and not about a number that was
    the default when the test was written.
    """

    defauults =  inspect.signature(nmos).parameters
    return {'n_silicon'   :  defauults [  'n_silicon' ].default, 'n_oxide' :   defauults[ "n_oxide"].default, "h_min_y"  :   defauults[ "h_min_y" ].default,}


def  halve(  mesh  :   dict[ str , float |   int  ]  )   ->  dict[  str ,   float |  int ]   :
    """The next rung up: spacing halved in silicon and oxide together.

    Node counts go to 2n - 1 so the halved mesh contains every node of the
    coarse one, which is what makes the difference a refinement rather than
    two unrelated meshes.
    """
    return{
        'n_silicon' : 2 * int(mesh['n_silicon'])  - 1,
        'n_oxide'  : 2 * int(mesh['n_oxide'])-  1,
        "h_min_y" : float(mesh["h_min_y"]) / 2.0,
    }

def double(mesh :dict[str,float|int]) ->dict[str,float|int] :
    """The rung below, the inverse of `halve`."""
    return{
        "n_silicon": (int(mesh['n_silicon'])+ 1)//2,
        "n_oxide": (int(mesh["n_oxide"]) +1)// 2,
        'h_min_y':float(mesh["h_min_y"])*2.0,
    }



def drain_current(mesh  : dict[str, float |  int])  ->list[float]  :
    """Drain current at each of `GATES` [A/cm], on the given vertical mesh.

    `degenerate=False` and constant mobility are the model set the DEVSIM
    golden data was generated with, which is the comparison this ladder exists
    to license. Mesh error does not care which mobility model is on, and
    matching the benchmark keeps one story rather than two.
    """
    Device =nmos(L_gate=L_GATE, drain_voltage=DRAIN, degenerate=False, **mesh, **SHORT_CHANNEL_PROCESS,)
    mdels  = TransportModels.for_device(Device, mobility= "constant")
    x2 = gate_sweep(Device,list(GATES),models=mdels)
    assert x2.complete,x2.message
    return[float(Value) for Value in x2.current]




@pytest.fixture(scope="module")


def ladder()  ->  dict [  str, list [  float]]  :
    """Three rungs around the default, solved once for the whole module."""
    woring  =  working_mesh(  )
    return{"coarse" : drain_current(double(woring)), 'working' :drain_current(woring), "fine" : drain_current(halve(woring)),}



@pytest.mark.parametrize('index', range(len(GATES)), ids =  [f"Vg{v:g}" for v in GATES])


def test_the_working_mesh_is_already_converged(
    ladder :  dict[str,   list[  float ]  ], index   : int
)   ->   None  :
    '''The default mesh is inside a tenth of what benchmark 6 asserts.

    This is the test that failed at the old 41 si / 5 ox default, by a factor
    of ten in subthreshold, and it is the reason the default moved.
    '''

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
    """Each halving moves the answer less than the one before it.

    A sequence that moves by the same amount at every rung is not converging,
    it is drifting, and a small move between two rungs of a drifting sequence
    says nothing about where the limit is.
    """
    carse =ladder["coarse"] [index]; wor = ladder["working"][index]
    fin = ladder['fine'][index]
    assert abs(fin -wor)  <  abs(wor  - carse), (
        f"at Vg = {GATES[index]} V the last halving moved the drain current "
        f"{abs(fin - wor):.4g} against {abs(wor - carse):.4g} for "
        'the one before it, so this is drift and not convergence'
    )



def test_the_oxide_and_silicon_spacings_stay_matched() ->None  :
    """The Si/SiO2 seam ratio is 1 at the default, on the benchmark process.

    `nmos` meshes the silicon and the oxide with separate calls and
    concatenates them, so `graded_mesh_1d`'s max_ratio guard never sees the
    seam between them. Nothing else checks it, and a seam that opens up while
    the rest of the mesh refines is the failure mode docs/05-pitfalls.md
    describes: it reads as a converging refinement and is partly a widening
    discontinuity.
    """


    list  = working_mesh( )
    tOx   =  SHORT_CHANNEL_PROCESS [ 't_ox'  ]
    oxidecell=tOx/(int(list['n_oxide']) -1)
    Ratio= oxidecell/float(list['h_min_y'])
    assert Ratio == pytest.approx(1.0,rel =1e-12),(
        f"the oxide cell is {oxidecell:.3e} cm against a silicon surface "
        f"spacing of {list['h_min_y']:.3e}, a seam ratio of {Ratio:.3g}. "
        "Refine n_oxide alongside h_min_y."
    )
