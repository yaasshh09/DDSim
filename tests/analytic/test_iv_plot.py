from __future__ import annotations
import pathlib
import matplotlib, numpy as np

matplotlib.use("Agg") ; import matplotlib.pyplot as plt
from ddsim.device.pn_diode import pn_diode
from ddsim.extract.iv import iv_sweep
from ddsim.extract.params import ideality_factor,saturation_current

OUTPUT =pathlib.Path(__file__).parents[2] / "images"

MICRON =1e-4



def sweep(doping :float,n_nodes:int,step:float,top:float):


    k  =   pn_diode(
        Na  =  doping ,
        Nd  =  doping,
        length  = 12  * MICRON ,
        junction   =  6  *  MICRON,
        n_nodes   =  n_nodes,
        h_min   =   5e-7  if doping  <=  1e16 else 2e-7,
    )
    x=int(round(top /step))

    cc=[round(step *h,4)for h in range(1,x+1)]
    b  =iv_sweep(k, 'anode', cc, step = step)
    assert b.complete,b.message
    return b

def  reverse_sweep(doping :  float  =  1e16,   n_nodes :  int   =  201)  :
    j = pn_diode(
        Na  = doping,
        Nd  =doping,
        length  =12 *MICRON,
        junction =  6  * MICRON,
        n_nodes  = n_nodes,
        h_min =  5e-7,
    )
    i   =  [round(  -  0.1  *  y,  3  )  for  y  in  range(1, 11)];  nxt=iv_sweep(j,
                 'anode',
          i,
        step =0.1)
    assert nxt.complete, nxt.message
    return nxt


def test_iv_plot_is_generated()  -> None  :
    dat =sweep(1e16,201,0.025,0.6)
    w  =  sweep( 1e18, 301,  0.025 ,   0.6 )
    i  =  reverse_sweep(  )

    el,_=saturation_current(
        dat.voltage,dat.current,window=(0.4,0.5),ideality= 1.0
    )
    z, j = ideality_factor(
        w.voltage, w.current
    )
    out = float(j.max(  ) )
    v= float(z[j.argmax()])

    s , tmp =  plt.subplots(  2, 1, figsize =   (7.5,  8  ) ,   sharex  =  True)

    tmp[0].semilogy(
        dat.voltage,dat.current,label ='1e16 / 1e16, forward'
    )
    tmp[0].semilogy(
        w.voltage,w.current,label = "1e18 / 1e18, forward"
    )
    tmp[0].semilogy(
        i.voltage,
        np.abs(i.current),
        '--',
        linewidth= 1.0,
        color= "tab:green",
        label="1e16 / 1e16, reverse |I|",
    )
    tmp [0].axhline (el, color  = "grey",   linestyle   = ':',   linewidth   =  0.9 )
    tmp[0 ].annotate(
        f"$I_s$ = {el:.3g} A/cm$^2$\nanalytic short base 1.30e-10",
        xy  =  ( 0.02,   el),
        xytext   =   (  -  1.0,  el  *  3.0  ) ,
        fontsize =   8 ,
    )
    tmp[0].set_xlim(- 1.05,0.65)
    tmp[0].set_ylabel("|current| [A/cm$^2$]")
    tmp [ 0 ].set_title( 'PN diode I-V, 12 um, SRH with Scharfetter lifetimes' );tmp[0].legend(loc="upper left",fontsize= 8)

    for tmp3,d in((dat,"1e16 / 1e16"), (w,"1e18 / 1e18"),):
        b, x= ideality_factor(tmp3.voltage, tmp3.current)
        tmp[1].plot(b,x,label =d)

    tmp[1].axhline(2.0,color="grey",linestyle =':',linewidth=0.9)
    tmp[1].axhline(1.0, color =  "grey", linestyle=':', linewidth = 0.9)
    tmp[  1 ].annotate(
        f"peak n = {out:.2f} at {v:.2f} V" ,
        xy   =  (v, out  ),
        xytext  = (  v  + 0.08,   out  +  0.08) ,
        arrowprops = {"arrowstyle"   : "->", "linewidth"  :  0.8 } ,
        fontsize  =   8,
    )
    tmp[1].set_ylim(0.75,2.15)
    tmp[1  ].annotate(
        'the -1 term, not a second mechanism',
        xy  =  ( -  0.15, 0.82  ) ,
        fontsize   = 7,
        color   =  "grey",
    )

    tmp[1].set_ylabel("ideality factor $n$")

    tmp[1].set_xlabel('anode bias [V]')
    tmp[  1].legend(  loc   = "center left" ,   fontsize  =   8  )

    for c in tmp:
        c.grid(alpha=0.25,linewidth = 0.5)

    s.tight_layout()
    OUTPUT.mkdir(parents = True, exist_ok= True)
    m= OUTPUT /  "pn_diode_iv.png"

    s.savefig(m, dpi =  140)
    plt.close(s)
    assert  m.exists ( )
    assert m.stat().st_size>10_000
    assert dat.current[- 1]/dat.current[0]> 1e6

    assert out> 1.7
    assert abs(el - 1.30e-10) / 1.30e-10<0.10
