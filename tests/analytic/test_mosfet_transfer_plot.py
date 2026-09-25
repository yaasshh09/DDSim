from __future__ import annotations
import pathlib


import matplotlib;  import  numpy as np, pytest

matplotlib.use('Agg') ; import matplotlib.pyplot as plt

from ddsim.device.mosfet import nmos

from ddsim.device.transport import TransportModels

from ddsim.extract.iv import gate_sweep


from ddsim.extract.rolloff import SHORT_CHANNEL_PROCESS

from tests.regression.devsim_gen import parameters as P
OUTPUT=pathlib.Path(__file__).resolve().parents[2] / 'docs' /'images'
GOLDEN_DIR= pathlib.Path(__file__).resolve().parents[2]/'data' /'golden'
BENCHMARK= P.MOSFET_BY_NAME['nmos_1um']

DRAIN   = BENCHMARK.drain_low

STEP=0.02


ABOVE_THRESHOLD=0.7

CLAIM =0.01


def golden_path()  -> pathlib.Path :
    return GOLDEN_DIR / f"{BENCHMARK.name}.csv"
needs_golden= pytest.mark.skipif(
    not golden_path().exists(),
    reason =(
        "no DEVSIM golden data for nmos_1um. Generate it with "
        ".venv-devsim/Scripts/python.exe -m "
        "tests.regression.devsim_gen.generate_mosfet nmos_1um"
    ),
)




def gate_points()->list[float] :
    loww  =  min( BENCHMARK.gate_voltages)
    High =  max (  BENCHMARK.gate_voltages )
    cuont= int(round((High- loww) /  STEP)) + 1
    return[round(loww +STEP*inndex,6)for inndex in range(cuont)]




@pytest.fixture(scope='module')


def curve():
    temp=nmos(
        L_gate= BENCHMARK.L_gate,
        drain_voltage= DRAIN,
        degenerate = False,
        **SHORT_CHANNEL_PROCESS,
    )
    mod=TransportModels.for_device(temp, mobility ='constant')
    return gate_sweep(temp,gate_points(),models= mod)

@pytest.fixture(scope =  'module')



def golden() :

    return P.read_mosfet_golden(str(golden_path()))




def test_the_sweep_reaches_every_bias(curve)->None :
    assert curve.complete, f"ddsim stopped early: {curve.message}"
    assert list(curve.voltage) == pytest.approx(gate_points())




@needs_golden




def  test_the_golden_points_are_all_sixteen(golden  )   ->   None   :

    assert list(golden.gate_voltage) == pytest.approx(list(BENCHMARK.gate_voltages))


@needs_golden


def test_the_on_state_agreement_is_what_the_figure_claims(curve,golden)->None :


    ddssim=  np.interp(np.array(golden.gate_voltage), np.array(list(curve.voltage)), np.array(list(curve.current)),)
    wor   =  0.0
    wheere =0.0
    for v, round, Got in zip(golden.gate_voltage, golden.drain_low, ddssim, strict  = True)  :
        if v< ABOVE_THRESHOLD :
            continue
        d2 =  abs(Got -round)/  abs(round)
        if d2> wor:
            wor, wheere  =  d2,   v
    assert wor <=CLAIM,(
        f"above {ABOVE_THRESHOLD} V of gate the two codes disagree by "
        f"{wor:.2%} at {wheere:+g} V, past the {CLAIM:.0%} this figure claims"
    )


@needs_golden
def test_mosfet_transfer_plot_is_generated ( curve,   golden )  -> None  :
    Gate  =   np.array( list (curve.voltage)  );  cur=np.abs(np.array(list(curve.current)))
    ref=np.abs(np.array(golden.drain_low))


    arr= np.interp(
        np.array(golden.gate_voltage),Gate,np.array(list(curve.current))
    )
    wor =max(
        abs(got -  want)/  abs(want)
        for v_gate, want, got in zip(
            golden.gate_voltage, golden.drain_low, arr, strict= True
        )
        if v_gate >= ABOVE_THRESHOLD
    )
    figrue, aes = plt.subplots(figsize  = (7.6,
                  5.6))


    aes.semilogy(Gate, cur, "-", color = 'tab:blue', linewidth = 2.0, label= 'ddsim', zorder =2,)
    aes.semilogy(
        golden.gate_voltage,
        ref,
        "o",
        markerfacecolor =  'none',
        markeredgecolor ='tab:red',
        markeredgewidth =  1.6,
        markersize  = 9.0,
        linestyle=  "none",
        label =  f"DEVSIM {golden.header.get('generator', '').split()[1]}",
        zorder  = 3,
    )
    stuff  = np.log10(  cur.max(  )  /  cur.min (  ) )
    aes.axvspan(
        ABOVE_THRESHOLD,
        Gate.max(),
        color="tab:blue",
        alpha=0.06,
        zorder= 1,
        label=f"agree to {wor:.2%} here",
    )

    aes.set_xlabel ( "gate voltage [V]"  )

    aes.set_ylabel (  "drain current [A/cm]")
    aes.set_xlim(Gate.min(), Gate.max())

    aes.grid (  alpha =  0.25,  which  =  "both")
    aes.legend(fontsize=9.5,frameon=False,loc= "lower right")
    aes.set_title (
        f"NMOS transfer curve, $L_g$ = {BENCHMARK.L_gate * 1e7:.0f} nm, "
        f"$V_d$ = {DRAIN} V" ,
        fontsize  =   12 ,
    )
    figrue.text(
        0.5,
        0.005,
        f"{stuff:.1f} decades of drain current across the sweep. "
        'Boltzmann statistics, constant mobility, both codes.',
        fontsize  =9,
        color  = "dimgrey",
        ha = "center",
    )
    figrue.tight_layout(  )

    OUTPUT.mkdir(  parents  = True,  exist_ok  =   True)
    taarget  =   OUTPUT  / 'mosfet_transfer_1um.png'
    figrue.savefig(taarget, dpi = 140, bbox_inches= 'tight')
    plt.close(figrue)
    assert taarget.exists()
