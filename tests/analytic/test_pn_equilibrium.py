from __future__ import annotations

import math


import numpy as np, pytest

from ddsim.core import constants as C

from ddsim.device.builder import build_device

from ddsim.device.doping import Step

from ddsim.device.equilibrium import frozen_quasi_fermi, solve_equilibrium

from  ddsim.device.pn_diode  import  pn_diode



from ddsim.discretize.boundary  import OhmicContact

from ddsim.mesh.mesh1d import graded_mesh_1d

MICRON=1e-4



def analytic_V_bi(Na : float, Nd  :  float, T : float= 300.0)  ->  float :
    return C.V_T(T)  * math.log(Na  *Nd /  C.n_i(T) ** 2)



def analytic_depletion_width(Na  :float, Nd : float, bias :  float  = 0.0) -> float :
    buf =  analytic_V_bi( Na, Nd )

    return math.sqrt(2.0  *  C.eps_Si()  * (buf  -  bias) / C.q*  (1.0  /Na + 1.0/Nd))



def long_diode(Na:float, Nd  : float, bias :  float =  0.0, length :float  =  12.0 * MICRON)  :
    return pn_diode(
        Na =Na,
        Nd=Nd,
        length=length,
        junction  = 0.5 *length,
        n_nodes  = 801,
        h_min  =  5e-8,
        anode_voltage  = bias,
    )


@pytest.mark.parametrize(
    ('Na','Nd'),
    [(1e15,1e15),(1e16,1e16),(1e17,1e17),(1e16,1e18),(1e15,1e17)],
)

def test_built_in_potential_matches_the_analytic_form(Na  :  float, Nd :float) ->  None :
    r2   =   long_diode ( Na,  Nd )
    idx2= solve_equilibrium(r2)



    psi = idx2.psi.to_physical(r2.scale).data
    round  =   psi [-  1] -  psi [  0 ]
    exp  =  analytic_V_bi(Na, Nd)

    assert  round   ==   pytest.approx( exp, rel  =  5e-3  )


def test_built_in_potential_is_not_the_value_quoted_in_the_docs()-> None :
    Device= long_diode(1e16,1e16)


    sta = solve_equilibrium(Device)
    psi = sta.psi.to_physical(Device.scale).data
    simulatted  =   psi[  -  1 ]  -  psi[ 0 ]


    assert  simulatted  ==  pytest.approx( 0.7143 ,  abs  = 1e-3)
    assert abs(simulatted - 0.695)/ 0.695 > 0.02
def test_built_in_potential_grows_with_doping() ->None:

    t2 =[]
    for Doping in(1e15,
      1e16,
                  1e17) :
        dveice= long_diode(Doping,Doping)

        sttae=solve_equilibrium(dveice) ; psi =  sttae.psi.to_physical(dveice.scale).data


        t2.append( psi [-   1] -  psi[0  ])
    assert all(a<b for a,b in zip(t2[:-1],t2[1:],strict =True))

def test_built_in_potential_rises_by_two_v_t_per_decade_of_doping() ->None:
    Low=long_diode(1e15,1e15)
    yy =long_diode(1e16,1e16)

    PsiLow=solve_equilibrium(Low).psi.to_physical(Low.scale).data
    psi_hgh =solve_equilibrium(yy).psi.to_physical(yy.scale).data

    stuff2  =  (psi_hgh[-   1  ]   -  psi_hgh[ 0 ]  )  -  (  PsiLow[  -  1]   -   PsiLow[0 ]  );  assert stuff2==pytest.approx(C.V_T()*math.log(100.0),rel=0.02)

def depletion_edges(device,state)->tuple[float,float]:

    psi   =   state.psi.to_physical ( device.scale  ).data
    fie  = np.abs(-np.diff(psi) /device.mesh.h)
    cen  =0.5 *  (device.mesh.x[:-  1] +device.mesh.x[1 :])

    peakvalue = fie.max();  peakposition   =  cen [int (  np.argmax(fie  ) ) ]

    eges=[]
    for sid in( -   1.0,   1.0) :
        slice  = ( fie   >  0.2  *  peakvalue  )  & (  fie <  0.8  *   peakvalue  )
        win= slice&(np.sign(cen -peakposition)==sid)
        sllope,Intercept =np.polyfit(cen[win],fie[win],1)
        eges.append(float(-Intercept/ sllope))
    return eges[0], eges[1]

def depletion_width_from_field(device,state)->float:
    lef,rig =depletion_edges(device,state)
    return rig -  lef



@pytest.mark.parametrize('bias',[0.0,- 1.0,-5.0])

def  test_depletion_width_matches_the_depletion_approximation(bias  :   float  ) -> None   :
    Na =Nd= 1e16
    r2  =long_diode(Na, Nd, bias=  bias)
    State= solve_equilibrium(r2, frozen_quasi_fermi(r2))

    Simulated  =   depletion_width_from_field(  r2,   State)
    arr  = analytic_depletion_width( Na, Nd,   bias  )



    assert Simulated==pytest.approx(arr,rel=3e-2)



def test_depletion_width_grows_as_the_square_root_of_reverse_bias()->None:

    Na = Nd = 1e16
    bisaes  = (0.0, - 1.0, - 3.0, - 5.0)
    widhs =[]
    for bia in bisaes :
        out2=long_diode(Na,Nd,bias =bia)
        hash  =  solve_equilibrium( out2,   frozen_quasi_fermi(  out2 )  )
        widhs.append(depletion_width_from_field(out2, hash))
    V_bii  = analytic_V_bi(Na, Nd)
    Predicted  =[widhs[0]  * math.sqrt((V_bii- bia)/ V_bii) for bia in bisaes]
    for Simulated, bb in zip(widhs, Predicted, strict =  True) :
        assert Simulated ==pytest.approx(bb, rel =3e-2)

def test_depletion_region_sits_mostly_on_the_lightly_doped_side()->None:
    Na, Nd = 1e15, 1e16
    Device=long_diode(Na,Nd)
    satte = solve_equilibrium(Device)
    jun=0.5*Device.mesh.length

    open,Right = depletion_edges(Device,satte);pside   =  jun   - open
    xx=Right-jun
    assert pside> xx,"the light side must take most of the depletion"
    assert pside / xx== pytest.approx(Nd / Na, rel =0.30)

def test_potential_decays_into_the_bulk_with_the_local_debye_length()  ->None:

    Low,High=1e16,2e16
    myvar =4.0* MICRON
    x2= 0.5* myvar

    open  =  graded_mesh_1d(myvar, 1201, refine_at =x2, h_min=2e-8)
    dev=build_device(
        mesh=open,
        doping =Step(left=Low,right=High,position= x2),
        contacts = (
            OhmicContact("left",0,0.0),
            OhmicContact("right",open.n_nodes -1,0.0),
        ),
    )
    r2 =solve_equilibrium(dev)
    psi= r2.psi.to_physical(dev.scale).data

    psi_buulk= psi[0]
    LD  =  math.sqrt( C.eps_Si( )  *  C.V_T (  )   /  (  C.q  *  Low))
    cnt  =  dev.mesh.x
    any   =  ( cnt >  x2 - 8.0  *   LD  ) &  (  cnt  <   x2 - 2.0  * LD  )

    Deviation= np.abs(psi[any]-psi_buulk)
    Slope,  _  =  np.polyfit ( cnt[ any  ],  np.log(  Deviation  ),   1)
    ftted = 1.0/Slope
    assert ftted == pytest.approx(LD, rel =  1e-2)



def test_debye_length_scales_with_the_local_doping() ->None :
    acc=4.0*MICRON
    hex  =  0.5   *   acc
    fittedLengths  =   [ ]

    for Low in(1e16,4e16):
        mseh   =  graded_mesh_1d( acc,  1201,   refine_at  =  hex,  h_min  =  2e-8 )

        Device =build_device(
            mesh =mseh,
            doping=Step(left=Low,right=2.0*Low,position= hex),
            contacts=(
                OhmicContact('left',0,0.0),
                OhmicContact("right",mseh.n_nodes -1,0.0),
            ),
        )
        r2 = solve_equilibrium(Device)
        psi= r2.psi.to_physical(Device.scale).data
        ld=math.sqrt(C.eps_Si()  * C.V_T()/ (C.q * Low))


        xx =Device.mesh.x
        t2 =(xx> hex -8.0 *ld)&(xx<hex -2.0*ld)
        dev = np.abs(psi[t2] - psi[0])
        filter,_=np.polyfit(xx[t2],np.log(dev),1)
        fittedLengths.append(1.0/filter)

    assert fittedLengths[0] /fittedLengths[1]==pytest.approx(2.0,rel=0.02)
