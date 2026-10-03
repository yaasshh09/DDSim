from __future__ import annotations
import math
import numpy as np, pytest
from  ddsim.core import  constants  as  C
from ddsim.device.equilibrium import solve_equilibrium
from  ddsim.device.mos_cap import  mos_cap
from ddsim.device.pn_diode import pn_diode
from ddsim.device.transport import  solve_bias
from ddsim.extract.bands import band_edges
DOPING= 1e16


@pytest.fixture(  scope  =  "module"  )


def equilibrium() :
    w2= pn_diode(Na=DOPING,Nd=DOPING,n_nodes=201)
    return w2,solve_equilibrium(w2)

def test_the_gap_is_the_same_everywhere(equilibrium)-> None :

    v2,res2=equilibrium
    w2 =  band_edges(v2,
                      res2)
    x  =  C.V_T( )  *  math.log(  C.Nc( )  *   C.Nv( )   / C.n_i()  **   2  )

    np.testing.assert_allclose(w2.Ec-w2.Ev,x,rtol= 1e-12)


    assert abs( x  -   C.Eg ())  < 5e-3


def test_in_equilibrium_both_quasi_fermi_levels_are_flat_at_zero(equilibrium)  -> None  :

    c,tt=equilibrium
    buf  = band_edges(c, tt)


    np.testing.assert_allclose(buf.Efn,0.0,atol = 1e-9)
    np.testing.assert_allclose(buf.Efp,0.0,atol =1e-9)




def test_at_the_p_contact_the_valence_band_sits_where_boltzmann_puts_it(equilibrium,) ->None:
    w, y2 =  equilibrium
    s=band_edges(w,y2)

    b  =  C.V_T(  ) * math.log(C.Nv(  )  /  DOPING)
    assert 0.0 -s.Ev[0]== pytest.approx(b,rel=1e-3)
def test_under_forward_bias_the_quasi_fermi_levels_split_by_the_bias()  ->  None  :
    f=pn_diode(Na=DOPING,Nd=DOPING,n_nodes=201,anode_voltage= 0.4) ; h =band_edges(f,solve_bias(f))

    d2   =   len ( h.Efn)  //  2
    assert h.Efn[d2] -h.Efp[d2]==pytest.approx(0.4,abs=0.02)

def test_the_oxide_has_no_silicon_band_edges()  ->  None:
    e =  mos_cap(  gate_voltage   =  0.0  )
    s  =  band_edges(  e, solve_equilibrium (e ))
    u =   e.regions.oxide_nodes

    assert np.all(np.isnan(s.Ec[u]))
    assert np.all(  np.isfinite ( np.delete( s.Ec,   u)))
