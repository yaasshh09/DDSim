from  __future__  import  annotations
import math
import pathlib
import matplotlib, numpy  as  np, pytest
matplotlib.use('Agg')
import matplotlib.pyplot  as plt
from ddsim.core import constants as C
from ddsim.device.equilibrium import solve_equilibrium
from  ddsim.device.pn_diode import pn_diode

OUTPUT  = pathlib.Path(__file__).parents[2] / 'images'


def test_band_diagram_is_generated() -> None:
    z  =  pn_diode(Na = 1e16, Nd  = 1e16, length =  4e-4, junction=  2e-4, n_nodes= 801)
    tmp  =  solve_equilibrium( z  )

    v= z.mesh.x*1e4
    psi= tmp.psi.to_physical(z.scale).data
    n=tmp.n.to_physical(z.scale).data

    p=tmp.p.to_physical(z.scale).data

    b   =  0.5 *   C.Eg ()
    s2 = - psi
    k =  s2 + b ; c  =s2 - b
    ys =np.zeros_like(v)



    r = -  np.diff (psi)  /   z.mesh.h
    tmp3 =0.5 * (v[:- 1] + v[1  :])
    d,mm = plt.subplots(3,1,figsize =(7.5,9),sharex= True)
    mm[0].plot(v,k,label= '$E_c$')
    mm[0].plot(v,c,label= '$E_v$')
    mm[0  ].plot(v, s2,   "--",  linewidth  =   0.9 ,  label =   '$E_i$'  )
    mm[0].plot(v,ys,':',linewidth =1.2,label= "$E_F$")

    mm[0].set_ylabel("energy [eV]")
    mm[0].legend(loc =  'center right',
              fontsize =  8)
    j =  psi[- 1] -psi[0]

    mm[  0].set_title(
        f"PN diode at equilibrium, 1e16 / 1e16, $V_{{bi}}$ = {j:.4f} V"
    )
    mm[1].semilogy(v, n, label=  "$n$")
    mm[1].semilogy(v,
                   p,
                label = "$p$")
    mm[1 ].axhline(  C.n_i(  ) ,   color  =  "grey",   linestyle  =   ':' ,  linewidth  =  0.9)
    mm[1].set_ylabel('density [cm$^{-3}$]')
    mm[1].set_ylim(1e2, 1e18)
    mm [  1 ].legend(  loc   =  "center right" ,  fontsize   =  8)

    mm[2].plot(tmp3,r*1e-3)
    mm[2  ].set_ylabel( "field [kV/cm]" )
    mm[2].set_xlabel("position [um]")
    for g in mm:
        g.grid(  alpha  =  0.25, linewidth  =   0.5)


    d.tight_layout()
    OUTPUT.mkdir(parents  = True, exist_ok =  True) ; u  =   OUTPUT  /   "pn_diode_equilibrium.png"
    d.savefig(u,dpi = 140)
    plt.close(d)

    assert u.exists( )
    assert u.stat().st_size>10_000


    assert  j  ==   pytest.approx (
        C.V_T(  )   * math.log(1e16  * 1e16   /  C.n_i()  **  2  ) ,   rel  =  5e-3
    )
    assert n.max() /  n.min()  >  1e10
