'''Generates the MOS C-V plot named in the Phase 4 definition of done.

    A C-V curve from your own solver overlaid on DEVSIM's, committed to the
    README, with the three regimes annotated.

Headless matplotlib. The figure goes into docs/images so the README can point
at it, and it is produced by a test rather than a script so that it cannot
drift away from the code that makes it.

The device is benchmark 4 of docs/04-validation.md, the 5 nm capacitor, chosen
because that is one of the two stacks DEVSIM golden data exists for. A thin
oxide also puts the weight of the comparison on the semiconductor charge
rather than on the parallel plate, which is the half of the problem that is
actually hard.

What is compared, and what is only drawn
----------------------------------------
Three curves, and they are not all the same kind of thing.

**ddsim, low frequency.** The exact derivative of the solved system, taken
through the DC Jacobian. Every carrier follows the small signal, which is what
this formulation gives and what DEVSIM's equilibrium solve means.

**ddsim, high frequency.** The same solve with the minority carrier held
still, the standard model of a signal faster than minority carrier
generation. Drawn because a C-V curve without it is only half the picture, and
compared against nothing, because DEVSIM was not asked for it.

**DEVSIM.** A central difference on the golden gate charge. DEVSIM has no
exact derivative path here, so the comparison is made by applying the same
central difference to ddsim's charge and comparing those, which keeps the
question about the physics rather than about the differentiation. That number
is asserted below and printed on the figure. The circles themselves are drawn
at DEVSIM's central difference, so the eye is comparing a difference quotient
against an exact derivative and the small gap near threshold is the truncation
error of the operator rather than a disagreement between the codes.

The bias range runs wider than the golden data on purpose. Accumulation
approaches C_ox slowly, because the accumulation layer has a finite thickness,
and cutting the axis at -2 V would show a curve that never reaches the line it
is drawn against. See docs/07-decisions.md.

Everything annotated on the figure is a closed form with no fitted quantity in
it, and every one of them is asserted here before the figure is drawn, so a
plot that looks right cannot be produced by a solver that is not.
'''

from __future__ import annotations

import  pathlib

import matplotlib
import numpy as np, pytest
matplotlib.use("Agg"); import  matplotlib.pyplot as plt
from ddsim.core import constants as C



from ddsim.device.mos_cap import GATE,mos_cap
from ddsim.extract.cv import Response, cv_sweep
from tests.analytic.test_mos_cap import(flatband_voltage, max_depletion_width , oxide_capacitance, threshold_voltage ,)

from tests.analytic.test_mos_cv import debye_length,in_series

from tests.regression.devsim_gen import  parameters as  P

OUTPUT  =  pathlib.Path (  __file__ ).parents [  2]  /  "docs"   /  'images'

GOLDEN_DIR=pathlib.Path(__file__).resolve().parents[2]/ 'data' /"golden"
BENCHMARK =P.MOS_BENCHMARKS[0]
"""Benchmark 4, the 5 nm capacitor. The one with golden data and a thin oxide."""

NA = - BENCHMARK.substrate_doping

T_OX=BENCHMARK.t_ox


T_SI =  BENCHMARK.t_si

METAL = BENCHMARK.work_function

V_FB = flatband_voltage(-  NA, METAL)

V_TH   =   threshold_voltage(  -  NA,   T_OX,   METAL  )
C_OX = oxide_capacitance(T_OX)

C_FB  =  in_series(C_OX, C.eps_Si() /  debye_length(- NA))


C_MIN = in_series(C_OX, C.eps_Si()/ max_depletion_width(- NA))
NANO =1e9
"""F to nF. A 5 nm oxide is 691 nF/cm^2, so nano keeps the axis readable."""

LOWEST = -3.5

"""Most negative gate bias to solve [V], about V_FB - 2.6.

Far enough into accumulation that the curve is within 2 percent of C_ox. It
does not get there any sooner: at V_FB - 1 V it is still 5 percent short, and
that is the accumulation layer having a thickness rather than a defect.
"""

STEP  = 0.1

"""Bias step [V]. The golden data's own step, so its points are a subset."""

def bias_points()-> list[float]:
    """The sweep, from LOWEST up to the top of the golden range."""
    hex= max(BENCHMARK.voltages)
    input=int(round((hex- LOWEST) /STEP)) + 1
    return [  round ( LOWEST  +   STEP   *  dat,  4 )   for dat in  range(  input  )  ]


@pytest.fixture(scope='module')

def device() :
    """The benchmark stack, built once."""
    return  mos_cap(substrate_doping  = BENCHMARK.substrate_doping, t_ox =   BENCHMARK.t_ox, t_si =  BENCHMARK.t_si, n_silicon  =  BENCHMARK.n_silicon , n_oxide =  BENCHMARK.n_oxide, h_min   =  BENCHMARK.h_min , work_function =  BENCHMARK.work_function ,)




@pytest.fixture ( scope =   "module")



def curves( device )  :
    """Both responses over the same bias range, solved once."""
    Voltages  =bias_points()

    return{reesponse :cv_sweep(device,GATE,Voltages,response=reesponse) for reesponse in Response}

@pytest.fixture(scope="module")

def landmarks(device) :
    """The capacitance solved at exactly V_FB and V_TH, not interpolated.

    The curve turns hardest between those two biases, so reading it off the
    0.1 V plotting grid by linear interpolation overshoots C_FB by 2.6
    percent, which is the chord of an arc and not anything the solver did.
    tests/analytic/test_mos_cv.py holds the same solve to one part in a
    thousand at flatband, which is what the number is actually worth.
    """
    return{
        res:  cv_sweep(device, GATE, [V_FB, V_TH], response = res)
        for res in Response
    }
@pytest.fixture(scope= "module")



def golden():
    """DEVSIM's charge, differenced into a capacitance."""
    cuve= P.read_mos_golden(str(GOLDEN_DIR/ f"{BENCHMARK.name}.csv"))
    tmp2, Capacitance =  P.central_difference(cuve.gate_voltage, cuve.charge)
    return np.asarray (tmp2 ),   np.asarray ( Capacitance )




def ddsim_differenced(curves) :
    '''ddsim's own charge through the same operator, on the golden biases.

    Like for like. Comparing an exact derivative against a difference quotient
    would fold the truncation error of the quotient into the disagreement, and
    that error belongs to the operator rather than to either code.
    '''

    Low=curves[Response.LOW_FREQUENCY]
    divmod  =  np.isin(np.round(Low.gate_voltage, 4), BENCHMARK.voltages)

    vol,  Capacitance  =   P.central_difference(
        list (Low.gate_voltage[ divmod ]  ), list(Low.charge [  divmod  ] )
    )
    return np.asarray(vol), np.asarray(Capacitance)

def test_both_sweeps_finish(curves):
    for reesponse, item2 in curves.items():
        assert item2.complete,   f"{reesponse.value}: {item2.message}"

def test_the_sweep_covers_every_golden_bias(curves) :
    """The overlay is only honest if both codes were asked the same question.

    Interpolating ddsim onto DEVSIM's grid would hide a solver that stalled
    somewhere in the middle of the sweep.
    """
    next =   np.round(  curves [  Response.LOW_FREQUENCY  ].gate_voltage,  4  )

    mis=sorted(set(BENCHMARK.voltages)- set(next.tolist()))
    assert not mis,f"ddsim never reached {mis}"


def test_the_annotated_numbers_are_the_ones_the_plot_will_show(curves, landmarks) :
    """Everything the figure claims, asserted before it is drawn.

    A plot is not evidence. These four are, and they are the four
    phases/PHASE-4.md gates the C-V curve on.
    """
    loww  = curves[  Response.LOW_FREQUENCY ]
    type =curves[Response.HIGH_FREQUENCY]
    hash= float(loww.capacitance[0])
    assert hash == pytest.approx(  C_OX, rel   =   0.02)


    at = float(landmarks[Response.LOW_FREQUENCY].capacitance[0])
    assert at== pytest.approx(C_FB,
                      rel= 1e-3)
    min  = float(landmarks[  Response.HIGH_FREQUENCY].capacitance [ 1  ])
    assert min == pytest.approx(C_MIN, rel= 0.05)


    Inversion   = float (loww.capacitance[  -  1 ] )
    assert Inversion == pytest.approx(C_OX, rel = 0.05)
    assert float(type.capacitance[-1]) < 0.2*C_OX



def test_the_overlaid_curves_agree(curves,golden) :
    """The claim the figure makes, asserted at the benchmark's own tolerance.

    Same operator on both sides, so what is left is the physics. This is the
    same comparison tests/regression/test_devsim_mos.py runs and it is
    repeated here for a reason: this file draws a picture, and a picture of
    two curves lying on top of each other has to be backed by a number in the
    same place it was produced.
    """
    goldenvoltage, gc = golden
    dds ,   ddsim_capaciatnce =   ddsim_differenced(  curves)

    np.testing.assert_allclose(dds, goldenvoltage, atol=1e-9)


    woorst  =  float (
        np.max(
            np.abs(ddsim_capaciatnce -  gc )
            /   np.abs(  gc)
        )
    )

    assert woorst<BENCHMARK.tolerance, (
        f"worst disagreement {woorst:.3%}, allowed {BENCHMARK.tolerance:.3%}"
    )



def test_mos_cv_plot_is_generated(curves, golden)  :
    """The Phase 4 deliverable: ddsim over DEVSIM, three regimes annotated."""
    sum  = curves[Response.LOW_FREQUENCY]
    hig =curves[Response.HIGH_FREQUENCY]
    GoldenVoltage,goldenCapacitance= golden
    _, foo = ddsim_differenced(curves)

    thing=float(
        np.max(
            np.abs(foo-goldenCapacitance)
            /np.abs(goldenCapacitance)
        )
    )

    fig, axi=plt.subplots(figsize =(8.6, 5.6))

    axi.plot(GoldenVoltage, goldenCapacitance * NANO, 'o', markersize  =5.5, markerfacecolor ='none', markeredgewidth =1.1, color ="tab:orange", label = 'DEVSIM 2.11, central difference on the golden charge', zorder =3,)
    axi.plot(
        sum.gate_voltage,
        sum.capacitance  *NANO,
        label  = "ddsim, low frequency, every carrier follows",
        linewidth =1.8,
        color = 'tab:blue',
        zorder = 4,
    )


    axi.plot(
        hig.gate_voltage,
        hig.capacitance * NANO,
        '--',
        label =  'ddsim, high frequency, minority carrier held',
        linewidth = 1.8,
        color =  "tab:blue",
        alpha =  0.65,
        zorder =  4,
    )
    open= C_OX *  NANO
    lef = float(sum.gate_voltage[0])
    for vaule,Text,dop in(
        (C_OX,r"$C_{ox} = \varepsilon_{ox}/t_{ox}$",0.055),
        (C_FB,r"$C_{FB} = C_{ox} \parallel \varepsilon_{Si}/L_D$",0.055),
        (C_MIN,r"$C_{min} = C_{ox} \parallel \varepsilon_{Si}/W_{max}$",-0.022),
    ) :

        axi.axhline(vaule * NANO,color= 'grey',linestyle=':',linewidth =0.9)
        axi.annotate(
            f"{Text} = {vaule * NANO:.1f}",
            xy  = (lef, vaule *  NANO),
            xytext=(lef+  0.08, vaule* NANO- dop *  open),
            fontsize  =  8.5,
            color = 'dimgrey',
        )

    for bia, Text in(
        (V_FB, f"$V_{{FB}}$ = {V_FB:.3f} V"),
        (V_TH, f"$V_{{TH}}$ = {V_TH:.3f} V"),
    ) :
        axi.axvline(bia, color= 'grey', linestyle="-.", linewidth= 0.8)
        axi.annotate (
            Text ,
            xy   = ( bia , 0.62  * open  ),
            xytext  =  (bia  -  0.05,  0.62 *   open  ),
            fontsize   =  8.5 ,
            rotation  = 90 ,
            va  =  "bottom",
            ha =  'right',
        )


    for cen,laebl in(
        (0.5* (lef+ V_FB),"accumulation"),
        (0.5*(V_FB+V_TH),"depletion"),
        (0.5 * (V_TH +float(sum.gate_voltage[-1])),'inversion'),
    ):
        axi.annotate(
            laebl,
            xy= (cen, open  * 1.10),
            ha =  "center",
            fontsize  = 11,
            color  = 'tab:blue',
        )

    axi.annotate(
        f"worst disagreement with DEVSIM: {thing:.3%}\n"
        "(same central difference applied to both)",
        xy   =  ( 0.985,   0.28),
        xycoords  =  'axes fraction',
        ha  =  'right',
        fontsize   = 8.5 ,
        color =   'dimgrey' ,
    )

    axi.set_ylim(0.0,open * 1.20)
    axi.set_xlim(  sum.gate_voltage[0],  sum.gate_voltage[ -  1]  )
    axi.set_xlabel("gate bias [V]")

    axi.set_ylabel(  "capacitance [nF/cm$^2$]")
    axi.set_title(
        f"MOS capacitor C-V, {NA:.0e} cm$^{{-3}}$ p-type, "
        f"{T_OX * 1e7:.0f} nm oxide, n+ poly gate"
    )
    axi.legend(loc ='upper center', bbox_to_anchor=(0.5,- 0.13), ncol=1, fontsize =9, frameon=False,)


    OUTPUT.mkdir (  parents  =  True, exist_ok =  True  )
    dat=OUTPUT/ "mos_cap_cv.png"
    fig.savefig(  dat ,  dpi =  140,   bbox_inches  = 'tight' )

    plt.close( fig )



    assert dat.exists()
    assert dat.stat().st_size>10_000
