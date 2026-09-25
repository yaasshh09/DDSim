"""Generates the gate length sweep plot named in the Phase 5 definition of done.

    Vth versus Lg from 1 um to 50 nm. This single plot is the argument that the
    project worked. Nothing in it was fitted; it came out of Poisson plus two
    continuity equations plus a doping profile.

Headless matplotlib, into docs/images so the README can point at it, and
produced by a test rather than a script so it cannot drift away from the code
that makes it.

What is on it, and why each panel is there
------------------------------------------
Three of the four acceptance criteria of phases/PHASE-5.md are visual, and the
fourth is a number this file asserts before drawing anything.

**Left, threshold against gate length.** Two curves, the same devices at two
drain biases. The fall from left to right is roll-off. The gap between the two
is drain induced barrier lowering, and it opens as the gate shortens because
the drain reaches further into a shorter channel.

**Right, subthreshold slope and the saturation exponent.** The slope against
the 59.5 mV/decade thermal limit, which nothing thermally activated can beat at
300 K, and the exponent against the 2 of an ideal long channel square law.
The first rises off its limit as the gate loses control of the barrier; the
second falls toward 1 as carriers stop going faster when the field is raised.

The models are the full Phase 5 stack: Arora doping dependent mobility inside
Lombardi surface scattering inside Caughey-Thomas, with Fermi-Dirac statistics.
Two of those are named in the scope of phases/PHASE-5.md as not optional, and
the third is what the exponent is measuring, so a sweep taken on the Phase 2
constant mobility would be measuring something else and reporting it under this
title.

DEVSIM is on this figure
------------------------
phases/PHASE-5.md asks for the sweep overlaid on DEVSIM, and the open markers
on the left panel are it, read from data/golden/fullstack_1um.csv and its four
siblings.

Those files are benchmark 10 of docs/04-validation.md and they exist because
benchmark 9 could not be used here. Benchmark 9 runs both codes at Boltzmann
statistics and constant mobility on purpose, matched model for model, because
that is what makes a disagreement about the geometry mean something. This
figure runs the full Phase 5 stack, so its drain current is not the current
those files hold and its thresholds are not theirs either: at 50 nm the
benchmark 9 data gives +0.0065 and -0.1181 V where this figure reports around
+0.09 and -0.03, because a constant current criterion rides on the current
scale and mobility sets that. Overlaying those would have drawn a gap that was
the model set rather than an error.

So DEVSIM was run again at the same stack, Fermi-Dirac by Joyce-Dixon with
Arora inside Lombardi inside Caughey-Thomas, on its own mesh and out of its own
expressions, and that is what is plotted. The two codes share the parameter
values and nothing else. The agreement is gated in
tests/regression/test_devsim_mosfet.py rather than here, since a figure is a
bad place to keep a tolerance. See the 2026-09-12 and 2026-09-13 rows in
docs/07-decisions.md.

Nothing here is fitted
----------------------
Every device on this plot is the same process from ddsim/extract/rolloff.py.
The oxide, the channel doping, the implant depth and the lateral encroachment
are identical across all five, and `L_gate` is the only argument that changes.
No short channel term exists anywhere in the solver to be turned on.
"""


from __future__ import annotations
import pathlib

import matplotlib, numpy as np, pytest
matplotlib.use ( 'Agg'  ) ; import matplotlib.pyplot as plt

from ddsim.extract.params import threshold_constant_current

from ddsim.extract.rolloff import(
    REFERENCE_CURRENT,
    SHORT_CHANNEL_PROCESS,
    gate_length_sweep,
)


from tests.regression.devsim_gen import parameters as P
GOLDEN =  pathlib.Path(__file__).resolve().parents[2]  / "data" / "golden"
"""Where benchmark 10's DEVSIM curves live."""
OUTPUT=pathlib.Path(__file__).parents[2]/"docs"/"images"
GATE_LENGTHS  =   [ 1e-4,   2e-5,  1e-5, 7e-6 ,   5e-6]


"""1 um down to 50 nm [cm]. The bottom is where drift-diffusion stops meaning
anything, not where the solver stops converging."""

GATE_VOLTAGES =list(np.round(np.arange(- 0.5,0.351,0.05),4)) +list(np.round(np.arange(0.4,1.401,0.1),4))
"""Fine through subthreshold, coarse above it. The slope is read over two
decades of current, which at 70 mV/decade is 140 mV wide and needs 0.05 V
steps to hold more than one point. Above threshold the curve is a power law
and 0.1 V resolves it, and every point is a coupled 2D solve.

The fine stretch used to stop at 0.15 V, which was right while every 2D drain
current was 1/x_0 too large and the constant current threshold therefore sat
about 170 mV low. With the current fixed the thresholds moved up and the window
moved with them, leaving exactly two points inside it. Two points still measure
a slope, but a slope from two points is one difference with no averaging in it,
and the subthreshold slope is a headline number of this phase. Reaching 0.35 V
puts four or five points in the window at every gate length in the sweep."""
DRAIN_LOW= 0.05
DRAIN_HIGH=1.0


THERMAL_LIMIT=  59.5
"""kT/q ln 10 at 300 K [mV/decade]."""

SQUARE_LAW=2.0


"""The exponent an ideal long channel MOSFET saturates with."""

@pytest.fixture(scope='module')

def sweep():
    return gate_length_sweep (
        gate_lengths   =  GATE_LENGTHS ,
        gate_voltages  =   GATE_VOLTAGES,
        drain_low  =  DRAIN_LOW,
        drain_high =   DRAIN_HIGH ,
    )
def falling(values ) ->  bool :
    """True when every step of the sequence goes down."""
    return bool(np.all(np.diff(np.asarray(values)) < 0.0))


def devsim_thresholds()-> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Benchmark 10's gate lengths [nm] and its two threshold curves [V].

    The same constant current criterion the ddsim sweep uses, at the same
    reference current, applied to DEVSIM's own transfer curves. Comparing two
    codes through two different estimators would measure the estimators.

    The leading points that are not yet on the rising part of the curve are
    dropped, bounded so the drop cannot swallow a real current, by the same
    `first_resolved_point` that tests/regression/test_devsim_mosfet.py applies
    to ddsim's curves. Both codes need it at this model set: on the 1 um device
    at 1 V of drain DEVSIM reports -4.5e-10 A/cm at -0.4 V of gate against a
    target of 1e-3, and on the 100 nm at 50 mV it reports a positive +3.5e-11
    that falls to +9.6e-12 at the next step. Sharing the rule is what keeps
    this an overlay of two codes rather than of two preprocessings.
    """
    aa: list[float] =[]
    lin : list[float]= []
    bb:list[float]=[]
    for naame in P.FULL_STACK_TREND :
        s2 =  P.MOSFET_BY_NAME [naame ]
        Golden=P.read_mosfet_golden(str(GOLDEN/f"{naame}.csv"))
        gat=np.asarray(Golden.gate_voltage,dtype = np.float64)
        type=REFERENCE_CURRENT / s2.L_gate
        aa.append(s2.L_gate *1e7)
        for cur, map in((Golden.drain_low, lin), (Golden.drain_high, bb),):
            val =  np.asarray(cur, dtype = np.float64)
            buf=P.first_resolved_point(val,type)
            map.append(threshold_constant_current(gat[buf :],val[buf:],type))
    return np.array(aa  ),   np.array( lin  ),   np.array (bb  )

def label_lengths(axis, lengths)  -> None :
    """Tick at the gate lengths that were solved, not at powers of ten.

    A log axis holding five points between 1000 and 50 nm labels two of them by
    default and leaves the reader counting minor ticks to find the rest.
    """
    axis.set_xticks(lengths)
    axis.set_xticklabels( [  f"{d2:.0f}"  for d2 in lengths]  )

    axis.set_xticks([],
                      minor  = True)

def  test_every_device_in_the_sweep_solved( sweep)  :
    """A stalled sweep still returns a curve, and a threshold extracted off a
    short one is a threshold for a device that was never solved."""
    assert len(sweep)==len(GATE_LENGTHS)
    for bb  in sweep :

        assert bb.linear.complete
        assert  bb.saturated.complete


def test_the_subthreshold_slope_beats_no_thermal_limit( sweep )  :
    """The primary sanity gate of phases/PHASE-5.md. Below 59.5 mV/decade at
    300 K is a bug at any gate length."""
    for poi in sweep:
        assert poi.subthreshold_slope  >=THERMAL_LIMIT


def test_the_subthreshold_slope_degrades_as_the_gate_shortens(sweep):
    """It sits on its limit while the gate owns the barrier and lifts off it
    once the drain starts sharing control. Asserted end to end rather than
    step by step: between 1 um and 200 nm this process has no short channel
    effect left to lose, so those three lengths are equal to a millivolt and
    a strict ordering there would be asserting noise."""
    Slopes =  [xx.subthreshold_slope for xx  in sweep ]


    assert Slopes[-1]>Slopes[0]+ 10.0
    assert np.all(np.diff(Slopes) > - 0.5)

def  test_the_threshold_rolls_off(  sweep)   :
    """Both extraction methods, both drain biases. The source and drain
    depletion regions share channel charge the gate would otherwise have to
    deplete itself, so a shorter channel needs less gate. Nothing about the
    doping or the oxide changed between these five devices."""
    assert falling([pont.threshold_linear for pont in sweep])
    assert falling([pont.threshold_saturated for pont in sweep]) ; assert falling([pont.threshold_extrapolated for pont in sweep])

def test_drain_induced_barrier_lowering_widens_the_gap(sweep)  :
    """Raising the drain pulls the source barrier down, so the saturated
    threshold sits below the linear one at every length, and the gap between
    them opens as the drain gets closer to the source."""
    for poi in sweep :
        assert poi.threshold_saturated <poi.threshold_linear
        assert poi.dibl >  0.0
    assert falling([-poi.dibl for poi in sweep])
def test_velocity_saturation_pulls_the_exponent_off_the_square_law(sweep):
    """Id goes as overdrive squared while the inversion charge and the
    velocity that carries it both rise with the gate. Once the channel field
    passes the critical field the velocity stops rising, one factor drops out,
    and the exponent falls toward 1.

    It is already below 2 at 1 um, because 1 V across a 1 um channel is around
    the critical field for electrons in silicon on its own, so this sweep never
    contains a device that is purely square law.
    """
    Exponents  = [Point.saturation_exponent  for  Point  in sweep  ]
    assert Exponents[0  ]  <=  SQUARE_LAW
    assert Exponents[-1]<1.2
    assert Exponents[-  1] > 1.0
    assert np.all(np.diff(Exponents)< 0.01)


def test_the_devsim_overlay_is_the_same_five_devices(sweep)  :
    '''The open markers sit at the gate lengths the curve was solved at.

    A reference plotted at gate lengths the sweep does not contain is not an
    overlay, it is two plots sharing an axis, and at a glance the figure would
    not say which. The agreement between the two is gated in
    tests/regression/test_devsim_mosfet.py, where a tolerance belongs.
    '''
    lenggths , temp , Saturated =  devsim_thresholds( )

    assert list(lenggths)==pytest.approx([Point.L_gate*1e7 for Point in sweep])


    assert np.all(Saturated < temp),('DEVSIM puts a saturated threshold above its linear one, which is not ' 'drain induced barrier lowering and would draw the shaded band upside ' "down")

def test_the_process_did_not_change_across_the_sweep(sweep):
    """The claim the whole figure rests on. If the oxide or the doping moved
    between devices, the roll-off would be a plot of the process rather than
    of the channel length."""

    assert 'L_gate' not in SHORT_CHANNEL_PROCESS
    assert "drain_voltage" not in SHORT_CHANNEL_PROCESS ; assert "gate_voltage" not in SHORT_CHANNEL_PROCESS



def test_mosfet_rolloff_plot_is_generated(  sweep  )  :


    len=  np.array([Point.L_gate* 1e7 for Point in sweep])
    vals = np.array([Point.threshold_linear for Point in sweep])
    tmp=np.array([Point.threshold_saturated for Point in sweep])
    Slopes=np.array([Point.subthreshold_slope for Point in sweep])
    myvar  =np.array([Point.saturation_exponent for Point in sweep])


    fig, (leeft, slice)= plt.subplots(1, 2, figsize = (11.4, 5.0))
    leeft.fill_between (
        len,
        tmp,
        vals ,
        color  =   "tab:blue",
        alpha =   0.12,
        label  =   (
            f"drain induced barrier lowering, {sweep[0].dibl:.0f} to "
            f"{sweep[-1].dibl:.0f} mV/V"
        ),
    )
    leeft.plot(
        len,
        vals,
        "o-",
        color = 'tab:blue',
        linewidth = 1.8,
        markersize = 5.5,
        label =f"$V_d$ = {DRAIN_LOW} V",
    )
    leeft.plot(
        len,
        tmp,
        "s--",
        color = "tab:red",
        linewidth  = 1.8,
        markersize = 5.0,
        label =  f"$V_d$ = {DRAIN_HIGH} V",
    )
    vars, devsimLinear, s2 =  devsim_thresholds()
    leeft.plot(vars, devsimLinear, 'o', markerfacecolor= "none", markeredgecolor = "tab:blue", markersize  = 11, markeredgewidth= 1.4, linestyle = "none", label = "DEVSIM, same models",)
    leeft.plot(
        vars,
        s2,
        's',
        markerfacecolor='none',
        markeredgecolor ="tab:red",
        markersize =10,
        markeredgewidth =1.4,
        linestyle = 'none',
    )
    leeft.annotate(
        f"{sweep[-1].dibl:.0f} mV/V",
        xy =   (  len [-   1], 0.5  *   (vals [  -   1  ] +  tmp[-  1 ] ) ),
        xytext  =  ( 1.6 * len [  -  1 ] ,  0.5 * (  vals[  -  1  ]   +   tmp[-  1  ])) ,
        fontsize  =  9 ,
        color  =  "dimgrey" ,
        va  =   'center',
    )
    leeft.set_xscale('log')
    leeft.invert_xaxis()
    label_lengths(leeft,len)
    leeft.set_xlabel('gate length [nm]'  );leeft.set_ylabel("threshold voltage [V]")
    leeft.set_title('threshold roll-off')
    leeft.legend(fontsize=9,frameon= False,loc ="lower left")

    leeft.grid(alpha= 0.25,which ="both")

    slice.plot(
        len,
        Slopes,
        'o-',
        color   =  "tab:green" ,
        linewidth  =  1.8 ,
        markersize  =  5.5,
        label  =  "subthreshold slope",
    )
    slice.axhline(
        THERMAL_LIMIT,
        color  = 'dimgrey',
        linestyle =  ":",
        linewidth =  1.2,
    )
    slice.text(
        len[0],
        THERMAL_LIMIT +1.0,
        f"$kT/q \\cdot \\ln 10$ = {THERMAL_LIMIT} mV/decade",
        fontsize  = 8.5,
        color  ="dimgrey",
        va  = 'bottom',
    )
    slice.set_xscale("log")

    slice.invert_xaxis()
    label_lengths(slice, len)
    slice.set_xlabel('gate length [nm]')
    slice.set_ylabel("subthreshold slope [mV/decade]" ,   color  =   'tab:green' )
    slice.tick_params(axis=  "y", labelcolor=  'tab:green')
    slice.set_ylim(THERMAL_LIMIT- 4.0,max(Slopes.max()+6.0,95.0))
    slice.set_title("gate control and velocity saturation")
    slice.grid(alpha = 0.25, which  = 'both')

    zz =slice.twinx()


    zz.plot(
        len,
        myvar,
        "^--",
        color= "tab:purple",
        linewidth= 1.8,
        markersize =  5.5,
        label  ="saturation exponent",
    )
    zz.axhline(SQUARE_LAW,color="tab:purple",linestyle=":",linewidth= 1.0)
    zz.text(
        len[-1],
        SQUARE_LAW-0.03,
        'long channel square law',
        fontsize=8.5,
        color ='tab:purple',
        ha= 'right',
        va='top',
    )
    zz.set_ylabel(
        r"$I_d \propto (V_g - V_{th})^{\alpha}$",  color  = "tab:purple"
    )
    zz.tick_params(axis =  "y" ,  labelcolor   =  'tab:purple' )
    zz.set_ylim(1.0,2.15)


    fig.suptitle(
        "NMOS gate length sweep, one process: "
        f"{SHORT_CHANNEL_PROCESS['t_ox'] * 1e7:.0f} nm oxide, "
        f"{-SHORT_CHANNEL_PROCESS['substrate_doping']:.0e} cm$^{{-3}}$ "
        "channel, lines ddsim, open markers DEVSIM",
        fontsize =  11,
    )
    fig.tight_layout()

    OUTPUT.mkdir (parents   =  True, exist_ok =  True)
    dat= OUTPUT /"mosfet_rolloff.png"

    fig.savefig(dat, dpi =  140, bbox_inches  = "tight")
    plt.close(  fig  )

    assert dat.exists()

    assert  dat.stat (  ).st_size  >  10_000
