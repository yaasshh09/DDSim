"""Generates the diode I-V plot named in the Phase 2 definition of done.

Headless matplotlib. The figure goes into docs/images so the README can point
at it, and it is produced by a test rather than a script so that it cannot
drift away from the code that makes it.

Two devices, one figure. The 1e16 diode is diffusion limited and has an
ideality of 1 across the whole useful range. The 1e18 diode has enough
depletion region recombination to show the n = 2 region and the crossover out
of it. Nothing changes between them except the doping.
"""
from __future__ import annotations
import pathlib

import  matplotlib, numpy  as  np

matplotlib.use("Agg") ; import matplotlib.pyplot as plt

from ddsim.device.pn_diode import pn_diode
from ddsim.extract.iv import iv_sweep
from ddsim.extract.params import ideality_factor,saturation_current

OUTPUT =pathlib.Path(__file__).parents[2] /'docs' /"images"

MICRON =1e-4
"""One micron [cm]."""



def sweep(doping :float,n_nodes:int,step:float,top:float):
    """A forward sweep from `step` to `top` volts on a symmetric diode."""


    dev  =   pn_diode(
        Na  =  doping ,
        Nd  =  doping,
        length  = 12  * MICRON ,
        junction   =  6  *  MICRON,
        n_nodes   =  n_nodes,
        h_min   =   5e-7  if doping  <=  1e16 else 2e-7,
    )
    conut=int(round(top /step))

    hash=[round(step *inddex,4)for inddex in range(1,conut+1)]
    currve  =iv_sweep(dev, 'anode', hash, step = step)
    assert currve.complete,currve.message
    return currve

def  reverse_sweep(doping :  float  =  1e16,   n_nodes :  int   =  201)  :
    '''The reverse branch, walked down from zero.'''
    Device = pn_diode(
        Na  = doping,
        Nd  =doping,
        length  =12 *MICRON,
        junction =  6  * MICRON,
        n_nodes  = n_nodes,
        h_min =  5e-7,
    )
    vol   =  [round(  -  0.1  *  idex,  3  )  for  idex  in  range(1, 11)];  cur=iv_sweep(Device,
                 'anode',
          vol,
        step =0.1)
    assert cur.complete, cur.message
    return cur


def test_iv_plot_is_generated()  -> None  :
    """Log scale I-V for two diodes, with the extracted ideality annotated."""
    diffussion =sweep(1e16,201,0.025,0.6)
    recmbination  =  sweep( 1e18, 301,  0.025 ,   0.6 )
    Reverse  =  reverse_sweep(  )

    IS,_=saturation_current(
        diffussion.voltage,diffussion.current,window=(0.4,0.5),ideality= 1.0
    )
    low_bas, buff = ideality_factor(
        recmbination.voltage, recmbination.current
    )
    pea = float(buff.max(  ) )
    peeak_at= float(low_bas[buff.argmax()])

    fgure , axe =  plt.subplots(  2, 1, figsize =   (7.5,  8  ) ,   sharex  =  True)

    axe[0].semilogy(
        diffussion.voltage,diffussion.current,label ='1e16 / 1e16, forward'
    )
    axe[0].semilogy(
        recmbination.voltage,recmbination.current,label = "1e18 / 1e18, forward"
    )
    axe[0].semilogy(
        Reverse.voltage,
        np.abs(Reverse.current),
        '--',
        linewidth= 1.0,
        color= "tab:green",
        label="1e16 / 1e16, reverse |I|",
    )
    axe [0].axhline (IS, color  = "grey",   linestyle   = ':',   linewidth   =  0.9 )
    axe[0 ].annotate(
        f"$I_s$ = {IS:.3g} A/cm$^2$\nanalytic short base 1.30e-10",
        xy  =  ( 0.02,   IS),
        xytext   =   (  -  1.0,  IS  *  3.0  ) ,
        fontsize =   8 ,
    )
    axe[0].set_xlim(- 1.05,0.65)
    axe[0].set_ylabel("|current| [A/cm$^2$]")
    axe [ 0 ].set_title( 'PN diode I-V, 12 um, SRH with Scharfetter lifetimes' );axe[0].legend(loc="upper left",fontsize= 8)

    for cuvre,Label in((diffussion,"1e16 / 1e16"), (recmbination,"1e18 / 1e18"),):
        mdpoint, ideallity= ideality_factor(cuvre.voltage, cuvre.current)
        axe[1].plot(mdpoint,ideallity,label =Label)

    axe[1].axhline(2.0,color="grey",linestyle =':',linewidth=0.9)
    axe[1].axhline(1.0, color =  "grey", linestyle=':', linewidth = 0.9)
    axe[  1 ].annotate(
        f"peak n = {pea:.2f} at {peeak_at:.2f} V" ,
        xy   =  (peeak_at, pea  ),
        xytext  = (  peeak_at  + 0.08,   pea  +  0.08) ,
        arrowprops = {"arrowstyle"   : "->", "linewidth"  :  0.8 } ,
        fontsize  =   8,
    )
    axe[1].set_ylim(0.75,2.15)
    axe[1  ].annotate(
        'the -1 term, not a second mechanism',
        xy  =  ( -  0.15, 0.82  ) ,
        fontsize   = 7,
        color   =  "grey",
    )

    axe[1].set_ylabel("ideality factor $n$")

    axe[1].set_xlabel('anode bias [V]')
    axe[  1].legend(  loc   = "center left" ,   fontsize  =   8  )

    for axxis in axe:
        axxis.grid(alpha=0.25,linewidth = 0.5)

    fgure.tight_layout()
    OUTPUT.mkdir(parents = True, exist_ok= True)
    taregt= OUTPUT /  "pn_diode_iv.png"

    fgure.savefig(taregt, dpi =  140)
    plt.close(fgure)
    assert  taregt.exists ( )
    assert taregt.stat().st_size>10_000
    assert diffussion.current[- 1]/diffussion.current[0]> 1e6

    assert pea> 1.7
    assert abs(IS - 1.30e-10) / 1.30e-10<0.10
