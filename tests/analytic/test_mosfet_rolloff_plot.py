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
OUTPUT=pathlib.Path(__file__).parents[2]/"docs"/"images"
GATE_LENGTHS  =   [ 1e-4,   2e-5,  1e-5, 7e-6 ,   5e-6]


GATE_VOLTAGES =list(np.round(np.arange(- 0.5,0.351,0.05),4)) +list(np.round(np.arange(0.4,1.401,0.1),4))
DRAIN_LOW= 0.05
DRAIN_HIGH=1.0


THERMAL_LIMIT=  59.5

SQUARE_LAW=2.0


@pytest.fixture(scope='module')

def sweep():
    return gate_length_sweep (
        gate_lengths   =  GATE_LENGTHS ,
        gate_voltages  =   GATE_VOLTAGES,
        drain_low  =  DRAIN_LOW,
        drain_high =   DRAIN_HIGH ,
    )
def falling(values ) ->  bool :
    return bool(np.all(np.diff(np.asarray(values)) < 0.0))


def devsim_thresholds()-> tuple[np.ndarray,np.ndarray,np.ndarray]:
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
    axis.set_xticks(lengths)
    axis.set_xticklabels( [  f"{d2:.0f}"  for d2 in lengths]  )

    axis.set_xticks([],
                      minor  = True)

def  test_every_device_in_the_sweep_solved( sweep)  :
    assert len(sweep)==len(GATE_LENGTHS)
    for bb  in sweep :

        assert bb.linear.complete
        assert  bb.saturated.complete


def test_the_subthreshold_slope_beats_no_thermal_limit( sweep )  :
    for poi in sweep:
        assert poi.subthreshold_slope  >=THERMAL_LIMIT


def test_the_subthreshold_slope_degrades_as_the_gate_shortens(sweep):
    Slopes =  [xx.subthreshold_slope for xx  in sweep ]


    assert Slopes[-1]>Slopes[0]+ 10.0
    assert np.all(np.diff(Slopes) > - 0.5)

def  test_the_threshold_rolls_off(  sweep)   :
    assert falling([pont.threshold_linear for pont in sweep])
    assert falling([pont.threshold_saturated for pont in sweep]) ; assert falling([pont.threshold_extrapolated for pont in sweep])

def test_drain_induced_barrier_lowering_widens_the_gap(sweep)  :
    for poi in sweep :
        assert poi.threshold_saturated <poi.threshold_linear
        assert poi.dibl >  0.0
    assert falling([-poi.dibl for poi in sweep])
def test_velocity_saturation_pulls_the_exponent_off_the_square_law(sweep):
    Exponents  = [Point.saturation_exponent  for  Point  in sweep  ]
    assert Exponents[0  ]  <=  SQUARE_LAW
    assert Exponents[-1]<1.2
    assert Exponents[-  1] > 1.0
    assert np.all(np.diff(Exponents)< 0.01)


def test_the_devsim_overlay_is_the_same_five_devices(sweep)  :
    lenggths , temp , Saturated =  devsim_thresholds( )

    assert list(lenggths)==pytest.approx([Point.L_gate*1e7 for Point in sweep])


    assert np.all(Saturated < temp),('DEVSIM puts a saturated threshold above its linear one, which is not ' 'drain induced barrier lowering and would draw the shaded band upside ' "down")

def test_the_process_did_not_change_across_the_sweep(sweep):

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
