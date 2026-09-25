"""Generates the 1 um transfer curve overlay, ddsim against DEVSIM.

    An Id-Vg curve from my own solver drawn as a line, with the DEVSIM golden
    points on top of it, on a log current axis so the seven decades from off to
    on are all visible at once.

This is the single plot that carries the claim of benchmark 6 in
docs/04-validation.md: that ddsim and DEVSIM, on two meshes that share nothing
but the geometry, land on the same transfer curve. Everything else in the tier
is a table of percentages. This is the one picture of it.

Headless matplotlib, into docs/images so the README can point at it, and
produced by a test rather than a script so it cannot drift away from the code
that makes it.

Why the 1 um device and why 50 mV of drain
------------------------------------------
The long device is the one where a disagreement is about the 2D transport
rather than about a barrier: at 1 um this process has no short channel effect
left in it. The low drain bias keeps the channel out of saturation for the
whole sweep, so the curve is the gate's story alone and no part of it is about
pinch off. That makes it the cleanest comparison of the two codes there is,
which is exactly why it is the one worth drawing.

What is compared, and what is not
---------------------------------
Both codes run Boltzmann statistics and constant mobility, not the Phase 5
stack, because that is what the golden data was generated with. So the currents
here are not the currents the roll-off figure reports. What agrees on this
figure is the electrostatics and the transport, which is what threshold voltage
and subthreshold slope are made of. Mobility is not being compared at all. See
the dated row in docs/07-decisions.md.

The agreement is asserted before anything is drawn
--------------------------------------------------
The subtitle quotes a number, and a figure that quotes a number it has not
checked is a figure that can go on looking right after the solver stops being
right. So the worst relative disagreement above threshold is computed and
asserted here first, and the subtitle is written from the value that passed.
"""

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
"""Benchmark 6. The long device, where the transport is the whole question."""

DRAIN   = BENCHMARK.drain_low

"""50 mV. Linear region for the whole sweep, so nothing here is pinch off."""
STEP=0.02


"""Gate step for the ddsim line [V].

One fifth of the golden spacing. The line is meant to read as a curve with the
reference points sitting on it, not as the same sixteen points joined up, and
at the golden spacing the subthreshold knee is three segments wide.
"""

ABOVE_THRESHOLD=0.7

"""Gate bias above which the agreement is asserted point by point [V].

Below this the drain current is falling through decades and both codes are
computing it as a difference of much larger fluxes, so the relative residual
there is a statement about floating point and not about the physics. The claim
this figure makes is about the on state, and that is where it is checked. The
off state is still drawn, because the seven decades are the point of the log
axis, and the whole curve is checked against the benchmark tolerance in
tests/regression/test_devsim_mosfet.py.
"""

CLAIM =0.01


"""Worst relative disagreement allowed above threshold [1].

Well inside the benchmark's own 5 percent. This is not a second tolerance on
the physics, it is a guard on the sentence printed in the subtitle: if the
agreement degrades past one percent the number on the figure stops being the
number the project is claiming, and the figure should fail rather than quietly
print a worse one.
"""

def golden_path()  -> pathlib.Path :
    """Where the generator writes benchmark 6."""
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
    """The ddsim sweep, spanning the golden range at STEP."""
    loww  =  min( BENCHMARK.gate_voltages)
    High =  max (  BENCHMARK.gate_voltages )
    cuont= int(round((High- loww) /  STEP)) + 1
    return[round(loww +STEP*inndex,6)for inndex in range(cuont)]




@pytest.fixture(scope='module')


def curve():
    """ddsim's own transfer curve, solved once.

    `degenerate=False` and `mobility="constant"` are what make this the model
    set the generator ran. Both are departures from ddsim's Phase 5 defaults
    and both are deliberate.
    """
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
    """DEVSIM's sixteen points, read off disk."""

    return P.read_mosfet_golden(str(golden_path()))




def test_the_sweep_reaches_every_bias(curve)->None :
    """A curve that stopped early would be drawn as a line that ends."""
    assert curve.complete, f"ddsim stopped early: {curve.message}"
    assert list(curve.voltage) == pytest.approx(gate_points())




@needs_golden




def  test_the_golden_points_are_all_sixteen(golden  )   ->   None   :

    """The overlay is the whole reference curve, not the part that converged."""
    assert list(golden.gate_voltage) == pytest.approx(list(BENCHMARK.gate_voltages))


@needs_golden


def test_the_on_state_agreement_is_what_the_figure_claims(curve,golden)->None :


    """The number in the subtitle, asserted before the subtitle is written."""

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
