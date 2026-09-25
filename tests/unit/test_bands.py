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
    Device= pn_diode(Na=DOPING,Nd=DOPING,n_nodes=201)
    return Device,solve_equilibrium(Device)

def test_the_gap_is_the_same_everywhere(equilibrium)-> None :

    dev,sta=equilibrium
    bnads =  band_edges(dev,
                      sta)
    ipmlied  =  C.V_T( )  *  math.log(  C.Nc( )  *   C.Nv( )   / C.n_i()  **   2  )

    np.testing.assert_allclose(bnads.Ec-bnads.Ev,ipmlied,rtol= 1e-12)


    assert abs( ipmlied  -   C.Eg ())  < 5e-3


def test_in_equilibrium_both_quasi_fermi_levels_are_flat_at_zero(equilibrium)  -> None  :

    r2,staate=equilibrium
    Bands  = band_edges(r2, staate)


    np.testing.assert_allclose(Bands.Efn,0.0,atol = 1e-9)
    np.testing.assert_allclose(Bands.Efp,0.0,atol =1e-9)




def test_at_the_p_contact_the_valence_band_sits_where_boltzmann_puts_it(equilibrium,) ->None:
    dev, State =  equilibrium
    baands=band_edges(dev,State)

    exp  =  C.V_T(  ) * math.log(C.Nv(  )  /  DOPING)
    assert 0.0 -baands.Ev[0]== pytest.approx(exp,rel=1e-3)
def test_under_forward_bias_the_quasi_fermi_levels_split_by_the_bias()  ->  None  :
    Device=pn_diode(Na=DOPING,Nd=DOPING,n_nodes=201,anode_voltage= 0.4) ; badns =band_edges(Device,solve_bias(Device))

    Middle   =   len ( badns.Efn)  //  2
    assert badns.Efn[Middle] -badns.Efp[Middle]==pytest.approx(0.4,abs=0.02)

def test_the_oxide_has_no_silicon_band_edges()  ->  None:
    devvice =  mos_cap(  gate_voltage   =  0.0  )
    bb  =  band_edges(  devvice, solve_equilibrium (devvice ))
    vars =   devvice.regions.oxide_nodes

    assert np.all(np.isnan(bb.Ec[vars]))
    assert np.all(  np.isfinite ( np.delete( bb.Ec,   vars)))
