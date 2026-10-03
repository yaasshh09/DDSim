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
OUTPUT=pathlib.Path(__file__).parents[2] / "images"
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
    r: list[float] =[]
    tmp : list[float]= []
    d:list[float]=[]
    for a in P.FULL_STACK_TREND :
        flag =  P.MOSFET_BY_NAME [a ]
        lst=P.read_mosfet_golden(str(GOLDEN/f"{a}.csv"))
        i=np.asarray(lst.gate_voltage,dtype = np.float64)
        e=REFERENCE_CURRENT / flag.L_gate
        r.append(flag.L_gate *1e7)
        for z, g in((lst.drain_low, tmp), (lst.drain_high, d),):
            u =  np.asarray(z, dtype = np.float64)
            hh=P.first_resolved_point(u,e)
            g.append(threshold_constant_current(i[hh :],u[hh:],e))
    return np.array(r  ),   np.array( tmp  ),   np.array (d  )

def label_lengths(axis, lengths)  -> None :
    axis.set_xticks(lengths)
    axis.set_xticklabels( [  f"{m:.0f}"  for m in lengths]  )

    axis.set_xticks([],
                      minor  = True)

def  test_every_device_in_the_sweep_solved( sweep)  :
    assert len(sweep)==len(GATE_LENGTHS)
    for v2  in sweep :

        assert v2.linear.complete
        assert  v2.saturated.complete


def test_the_subthreshold_slope_beats_no_thermal_limit( sweep )  :
    for f in sweep:
        assert f.subthreshold_slope  >=THERMAL_LIMIT


def test_the_subthreshold_slope_degrades_as_the_gate_shortens(sweep):
    lst =  [dat.subthreshold_slope for dat  in sweep ]


    assert lst[-1]>lst[0]+ 10.0
    assert np.all(np.diff(lst) > - 0.5)

def  test_the_threshold_rolls_off(  sweep)   :
    assert falling([y.threshold_linear for y in sweep])
    assert falling([y.threshold_saturated for y in sweep]) ; assert falling([y.threshold_extrapolated for y in sweep])

def test_drain_induced_barrier_lowering_widens_the_gap(sweep)  :
    for out in sweep :
        assert out.threshold_saturated <out.threshold_linear
        assert out.dibl >  0.0
    assert falling([-out.dibl for out in sweep])
def test_velocity_saturation_pulls_the_exponent_off_the_square_law(sweep):
    j  = [u.saturation_exponent  for  u  in sweep  ]
    assert j[0  ]  <=  SQUARE_LAW
    assert j[-1]<1.2
    assert j[-  1] > 1.0
    assert np.all(np.diff(j)< 0.01)


def test_the_devsim_overlay_is_the_same_five_devices(sweep)  :
    m2 , s , info =  devsim_thresholds( )

    assert list(m2)==pytest.approx([k.L_gate*1e7 for k in sweep])


    assert np.all(info < s),('DEVSIM puts a saturated threshold above its linear one, which is not ' 'drain induced barrier lowering and would draw the shaded band upside ' "down")

def test_the_process_did_not_change_across_the_sweep(sweep):

    assert 'L_gate' not in SHORT_CHANNEL_PROCESS
    assert "drain_voltage" not in SHORT_CHANNEL_PROCESS ; assert "gate_voltage" not in SHORT_CHANNEL_PROCESS



def test_mosfet_rolloff_plot_is_generated(  sweep  )  :


    foo=  np.array([h.L_gate* 1e7 for h in sweep])
    z = np.array([h.threshold_linear for h in sweep])
    tmp2=np.array([h.threshold_saturated for h in sweep])
    c=np.array([h.subthreshold_slope for h in sweep])
    val2  =np.array([h.saturation_exponent for h in sweep])


    bb, (cc, nxt)= plt.subplots(1, 2, figsize = (11.4, 5.0))
    cc.fill_between (
        foo,
        tmp2,
        z ,
        color  =   "tab:blue",
        alpha =   0.12,
        label  =   (
            f"drain induced barrier lowering, {sweep[0].dibl:.0f} to "
            f"{sweep[-1].dibl:.0f} mV/V"
        ),
    )
    cc.plot(
        foo,
        z,
        "o-",
        color = 'tab:blue',
        linewidth = 1.8,
        markersize = 5.5,
        label =f"$V_d$ = {DRAIN_LOW} V",
    )
    cc.plot(
        foo,
        tmp2,
        "s--",
        color = "tab:red",
        linewidth  = 1.8,
        markersize = 5.0,
        label =  f"$V_d$ = {DRAIN_HIGH} V",
    )
    w, g, t =  devsim_thresholds()
    cc.plot(w, g, 'o', markerfacecolor= "none", markeredgecolor = "tab:blue", markersize  = 11, markeredgewidth= 1.4, linestyle = "none", label = "DEVSIM, same models",)
    cc.plot(
        w,
        t,
        's',
        markerfacecolor='none',
        markeredgecolor ="tab:red",
        markersize =10,
        markeredgewidth =1.4,
        linestyle = 'none',
    )
    cc.annotate(
        f"{sweep[-1].dibl:.0f} mV/V",
        xy =   (  foo [-   1], 0.5  *   (z [  -   1  ] +  tmp2[-  1 ] ) ),
        xytext  =  ( 1.6 * foo [  -  1 ] ,  0.5 * (  z[  -  1  ]   +   tmp2[-  1  ])) ,
        fontsize  =  9 ,
        color  =  "dimgrey" ,
        va  =   'center',
    )
    cc.set_xscale('log')
    cc.invert_xaxis()
    label_lengths(cc,foo)
    cc.set_xlabel('gate length [nm]'  );cc.set_ylabel("threshold voltage [V]")
    cc.set_title('threshold roll-off')
    cc.legend(fontsize=9,frameon= False,loc ="lower left")

    cc.grid(alpha= 0.25,which ="both")

    nxt.plot(
        foo,
        c,
        'o-',
        color   =  "tab:green" ,
        linewidth  =  1.8 ,
        markersize  =  5.5,
        label  =  "subthreshold slope",
    )
    nxt.axhline(
        THERMAL_LIMIT,
        color  = 'dimgrey',
        linestyle =  ":",
        linewidth =  1.2,
    )
    nxt.text(
        foo[0],
        THERMAL_LIMIT +1.0,
        f"$kT/q \\cdot \\ln 10$ = {THERMAL_LIMIT} mV/decade",
        fontsize  = 8.5,
        color  ="dimgrey",
        va  = 'bottom',
    )
    nxt.set_xscale("log")

    nxt.invert_xaxis()
    label_lengths(nxt, foo)
    nxt.set_xlabel('gate length [nm]')
    nxt.set_ylabel("subthreshold slope [mV/decade]" ,   color  =   'tab:green' )
    nxt.tick_params(axis=  "y", labelcolor=  'tab:green')
    nxt.set_ylim(THERMAL_LIMIT- 4.0,max(c.max()+6.0,95.0))
    nxt.set_title("gate control and velocity saturation")
    nxt.grid(alpha = 0.25, which  = 'both')

    b =nxt.twinx()


    b.plot(
        foo,
        val2,
        "^--",
        color= "tab:purple",
        linewidth= 1.8,
        markersize =  5.5,
        label  ="saturation exponent",
    )
    b.axhline(SQUARE_LAW,color="tab:purple",linestyle=":",linewidth= 1.0)
    b.text(
        foo[-1],
        SQUARE_LAW-0.03,
        'long channel square law',
        fontsize=8.5,
        color ='tab:purple',
        ha= 'right',
        va='top',
    )
    b.set_ylabel(
        r"$I_d \propto (V_g - V_{th})^{\alpha}$",  color  = "tab:purple"
    )
    b.tick_params(axis =  "y" ,  labelcolor   =  'tab:purple' )
    b.set_ylim(1.0,2.15)


    bb.suptitle(
        "NMOS gate length sweep, one process: "
        f"{SHORT_CHANNEL_PROCESS['t_ox'] * 1e7:.0f} nm oxide, "
        f"{-SHORT_CHANNEL_PROCESS['substrate_doping']:.0e} cm$^{{-3}}$ "
        "channel, lines ddsim, open markers DEVSIM",
        fontsize =  11,
    )
    bb.tight_layout()

    OUTPUT.mkdir (parents   =  True, exist_ok =  True)
    a= OUTPUT /"mosfet_rolloff.png"

    bb.savefig(a, dpi =  140, bbox_inches  = "tight")
    plt.close(  bb  )

    assert a.exists()

    assert  a.stat (  ).st_size  >  10_000
