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
    j =  analytic_V_bi( Na, Nd )

    return math.sqrt(2.0  *  C.eps_Si()  * (j  -  bias) / C.q*  (1.0  /Na + 1.0/Nd))



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
    item   =   long_diode ( Na,  Nd )
    y2= solve_equilibrium(item)



    psi = y2.psi.to_physical(item.scale).data
    d2  =   psi [-  1] -  psi [  0 ]
    u  =  analytic_V_bi(Na, Nd)

    assert  d2   ==   pytest.approx( u, rel  =  5e-3  )


def test_built_in_potential_is_not_the_value_quoted_in_the_docs()-> None :
    m= long_diode(1e16,1e16)


    z2 = solve_equilibrium(m)
    psi = z2.psi.to_physical(m.scale).data
    f  =   psi[  -  1 ]  -  psi[ 0 ]


    assert  f  ==  pytest.approx( 0.7143 ,  abs  = 1e-3)
    assert abs(f - 0.695)/ 0.695 > 0.02
def test_built_in_potential_grows_with_doping() ->None:

    ss =[]
    for item in(1e15,
      1e16,
                  1e17) :
        d= long_diode(item,item)

        z=solve_equilibrium(d) ; psi =  z.psi.to_physical(d.scale).data


        ss.append( psi [-   1] -  psi[0  ])
    assert all(c<num for c,num in zip(ss[:-1],ss[1:],strict =True))

def test_built_in_potential_rises_by_two_v_t_per_decade_of_doping() ->None:
    b=long_diode(1e15,1e15)
    h =long_diode(1e16,1e16)

    i=solve_equilibrium(b).psi.to_physical(b.scale).data
    num =solve_equilibrium(h).psi.to_physical(h.scale).data

    val  =  (num[-   1  ]   -  num[ 0 ]  )  -  (  i[  -  1]   -   i[0 ]  );  assert val==pytest.approx(C.V_T()*math.log(100.0),rel=0.02)

def depletion_edges(device,state)->tuple[float,float]:

    psi   =   state.psi.to_physical ( device.scale  ).data
    s  = np.abs(-np.diff(psi) /device.mesh.h)
    d  =0.5 *  (device.mesh.x[:-  1] +device.mesh.x[1 :])

    f = s.max();  v   =  d [int (  np.argmax(s  ) ) ]

    y=[]
    for j in( -   1.0,   1.0) :
        r  = ( s   >  0.2  *  f  )  & (  s <  0.8  *   f  )
        e= r&(np.sign(d -v)==j)
        a,z =np.polyfit(d[e],s[e],1)
        y.append(float(-z/ a))
    return y[0], y[1]

def depletion_width_from_field(device,state)->float:
    e,z =depletion_edges(device,state)
    return z -  e



@pytest.mark.parametrize('bias',[0.0,- 1.0,-5.0])

def  test_depletion_width_matches_the_depletion_approximation(bias  :   float  ) -> None   :
    Na =Nd= 1e16
    it  =long_diode(Na, Nd, bias=  bias)
    bar= solve_equilibrium(it, frozen_quasi_fermi(it))

    stuff  =   depletion_width_from_field(  it,   bar)
    xs  = analytic_depletion_width( Na, Nd,   bias  )



    assert stuff==pytest.approx(xs,rel=3e-2)



def test_depletion_width_grows_as_the_square_root_of_reverse_bias()->None:

    Na = Nd = 1e16
    thing  = (0.0, - 1.0, - 3.0, - 5.0)
    c =[]
    for a in thing :
        d=long_diode(Na,Nd,bias =a)
        d2  =  solve_equilibrium( d,   frozen_quasi_fermi(  d )  )
        c.append(depletion_width_from_field(d, d2))
    it  = analytic_V_bi(Na, Nd)
    g  =[c[0]  * math.sqrt((it- a)/ it) for a in thing]
    for x, ret in zip(c, g, strict =  True) :
        assert x ==pytest.approx(ret, rel =3e-2)

def test_depletion_region_sits_mostly_on_the_lightly_doped_side()->None:
    Na, Nd = 1e15, 1e16
    k=long_diode(Na,Nd)
    cur = solve_equilibrium(k)
    rr=0.5*k.mesh.length

    yy,arr = depletion_edges(k,cur);b2   =  rr   - yy
    x=arr-rr
    assert b2> x,"the light side must take most of the depletion"
    assert b2 / x== pytest.approx(Nd / Na, rel =0.30)

def test_potential_decays_into_the_bulk_with_the_local_debye_length()  ->None:

    ss,prev=1e16,2e16
    m =4.0* MICRON
    ret= 0.5* m

    vals  =  graded_mesh_1d(m, 1201, refine_at =ret, h_min=2e-8)
    ii=build_device(
        mesh=vals,
        doping =Step(left=ss,right=prev,position= ret),
        contacts = (
            OhmicContact("left",0,0.0),
            OhmicContact("right",vals.n_nodes -1,0.0),
        ),
    )
    j =solve_equilibrium(ii)
    psi= j.psi.to_physical(ii.scale).data

    s= psi[0]
    f  =  math.sqrt( C.eps_Si( )  *  C.V_T (  )   /  (  C.q  *  ss))
    i  =  ii.mesh.x
    g   =  ( i >  ret - 8.0  *   f  ) &  (  i  <   ret - 2.0  * f  )

    d= np.abs(psi[g]-s)
    a,  _  =  np.polyfit ( i[ g  ],  np.log(  d  ),   1)
    out2 = 1.0/a
    assert out2 == pytest.approx(f, rel =  1e-2)



def test_debye_length_scales_with_the_local_doping() ->None :
    dat=4.0*MICRON
    f  =  0.5   *   dat
    i  =   [ ]

    for s in(1e16,4e16):
        prev   =  graded_mesh_1d( dat,  1201,   refine_at  =  f,  h_min  =  2e-8 )

        val =build_device(
            mesh =prev,
            doping=Step(left=s,right=2.0*s,position= f),
            contacts=(
                OhmicContact('left',0,0.0),
                OhmicContact("right",prev.n_nodes -1,0.0),
            ),
        )
        c2 = solve_equilibrium(val)
        psi= c2.psi.to_physical(val.scale).data
        k=math.sqrt(C.eps_Si()  * C.V_T()/ (C.q * s))


        z =val.mesh.x
        ii =(z> f -8.0 *k)&(z<f -2.0*k)
        row = np.abs(psi[ii] - psi[0])
        t,_=np.polyfit(z[ii],np.log(row),1)
        i.append(1.0/t)

    assert i[0] /i[1]==pytest.approx(2.0,rel=0.02)
