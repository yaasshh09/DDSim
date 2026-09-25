"""Generates the PN diode band diagram named in the Phase 1 definition of done.

Headless matplotlib. The figure is written into docs/images so the README can
point at it. It is produced by a test rather than a script so that it cannot
drift away from the code that makes it.
"""



from  __future__  import  annotations
import math


import pathlib

import matplotlib, numpy  as  np, pytest
matplotlib.use('Agg')
import matplotlib.pyplot  as plt

from ddsim.core import constants as C
from ddsim.device.equilibrium import solve_equilibrium

from  ddsim.device.pn_diode import pn_diode

OUTPUT  = pathlib.Path(__file__).parents[2] / "docs"/  'images'


def test_band_diagram_is_generated() -> None:
    '''A 1e16 / 1e16 diode at equilibrium: bands, carriers and field.'''
    bb  =  pn_diode(Na = 1e16, Nd  = 1e16, length =  4e-4, junction=  2e-4, n_nodes= 801)
    filter  =  solve_equilibrium( bb  )

    X= bb.mesh.x*1e4
    psi= filter.psi.to_physical(bb.scale).data
    n=filter.n.to_physical(bb.scale).data

    p=filter.p.to_physical(bb.scale).data

    chr   =  0.5 *   C.Eg ()
    EI = - psi
    blah =  EI + chr ; EV  =EI - chr
    EF =np.zeros_like(X)



    fieeld = -  np.diff (psi)  /   bb.mesh.h
    cenrtes =0.5 * (X[:- 1] + X[1  :])
    fig,aes = plt.subplots(3,1,figsize =(7.5,9),sharex= True)
    aes[0].plot(X,blah,label= '$E_c$')
    aes[0].plot(X,EV,label= '$E_v$')
    aes[0  ].plot(X, EI,   "--",  linewidth  =   0.9 ,  label =   '$E_i$'  )
    aes[0].plot(X,EF,':',linewidth =1.2,label= "$E_F$")

    aes[0].set_ylabel("energy [eV]")
    aes[0].legend(loc =  'center right',
              fontsize =  8)
    res =  psi[- 1] -psi[0]

    aes[  0].set_title(
        f"PN diode at equilibrium, 1e16 / 1e16, $V_{{bi}}$ = {res:.4f} V"
    )
    aes[1].semilogy(X, n, label=  "$n$")
    aes[1].semilogy(X,
                   p,
                label = "$p$")
    aes[1 ].axhline(  C.n_i(  ) ,   color  =  "grey",   linestyle  =   ':' ,  linewidth  =  0.9)
    aes[1].set_ylabel('density [cm$^{-3}$]')
    aes[1].set_ylim(1e2, 1e18)
    aes [  1 ].legend(  loc   =  "center right" ,  fontsize   =  8)

    aes[2].plot(cenrtes,fieeld*1e-3)
    aes[2  ].set_ylabel( "field [kV/cm]" )
    aes[2].set_xlabel("position [um]")
    for axi in aes:
        axi.grid(  alpha  =  0.25, linewidth  =   0.5)


    fig.tight_layout()
    OUTPUT.mkdir(parents  = True, exist_ok =  True) ; ret  =   OUTPUT  /   "pn_diode_equilibrium.png"
    fig.savefig(ret,dpi = 140)
    plt.close(fig)

    assert ret.exists( )
    assert ret.stat().st_size>10_000


    assert  res  ==   pytest.approx (
        C.V_T(  )   * math.log(1e16  * 1e16   /  C.n_i()  **  2  ) ,   rel  =  5e-3
    )
    assert n.max() /  n.min()  >  1e10
