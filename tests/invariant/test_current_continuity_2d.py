from __future__ import annotations
import numpy as np; import pytest
from ddsim.device.builder import build_device
from ddsim.device.doping import abrupt_junction
from ddsim.device.regions import stacked_regions
from ddsim.device.transport import TransportModels, solve_bias_newton
from ddsim.discretize.boundary import OhmicPlate
from ddsim.extract.iv import(
    current_densities,
    edge_current_face,
    terminal_currents,
)
from ddsim.mesh.mesh1d import graded_mesh_1d, stacked_mesh_1d , uniform_mesh_1d
from  ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.physics.recombination  import NoRecombination


MICRON =1e-4
LENGTH =12 * MICRON

JUNCTION =  6 *  MICRON

NX =81

HEIGHT =2*MICRON
NY  =5

T_OX = 0.1 *MICRON


N_OX   = 4



DOPING =  1e16


BIAS =0.5

def x_axis():
    return  graded_mesh_1d (
        length  =  LENGTH ,   n_nodes  =  NX,   refine_at  =  JUNCTION, h_min   =   5e-7
    )
def diode_2d(voltage  :float = BIAS)  :
    z= tensor_mesh_2d(x_axis(), uniform_mesh_1d(length=  HEIGHT, n_nodes= NY))

    return build_device(
        mesh   =  z,
        doping  =  abrupt_junction(  Na  = DOPING , Nd  = DOPING,  position   =  JUNCTION ),
        contacts =  (
            OhmicPlate(
                name  =  'anode',
                nodes  =  tuple(  z.node_at( 0, res2)   for  res2  in  range( z.ny)  ),
                voltage =  voltage ,
            ) ,
            OhmicPlate (
                name  = 'cathode',
                nodes =   tuple (z.node_at(  z.nx -  1 , out2 )  for  out2  in range ( z.ny ) ),
                voltage = 0.0 ,
            ) ,
        ),
    )



def capped_diode_2d(voltage:float  =BIAS):

    z  =   uniform_mesh_1d(  length  =  HEIGHT , n_nodes  = NY)

    u = uniform_mesh_1d(length  = T_OX, n_nodes =N_OX)
    ok  =tensor_mesh_2d(x_axis(), stacked_mesh_1d(z, u))
    y  =  stacked_regions(  ok ,
      interface_y   =   HEIGHT)

    return build_device(mesh = ok, doping  = abrupt_junction(Na = DOPING, Nd =DOPING, position = JUNCTION), contacts= (OhmicPlate(name =  'anode', nodes =tuple(ok.node_at(0, b) for b in range(NY)), voltage  =voltage,), OhmicPlate(name = "cathode", nodes = tuple(ok.node_at(ok.nx - 1, obj)for obj in range(NY)), voltage = 0.0,),), regions=y,)


def solved(device,recombination = None):
    ok=TransportModels.for_device(device,recombination=recombination)
    k = solve_bias_newton(device, models = ok)
    assert  k.newton is  not  None and  k.newton.converged,   (
        f"the 2D solve did not converge: {k.newton}"
    )
    return k, ok

def cut_currents(device,state,models) ->np.ndarray :
    foo =  device.mesh
    k,s= current_densities(device,state,models)
    a= k.data+ s.data
    v =edge_current_face(device)
    num=[]
    for f in range(  foo.nx  - 1)   :
        z=[g*(foo.nx - 1) +f for g in range(foo.ny)]
        num.append(float(np.sum(a[z]*v[z])))
    return np.asarray(num )
def spread ( values  : np.ndarray )  ->  float  :
    return float(np.max(np.abs(values -  values.mean()))/abs(values.mean()))




@pytest.fixture(scope =  'module')
def plain():
    b  =diode_2d()
    item,   prev  =  solved(b, NoRecombination(  ))
    return b, item, prev



@pytest.fixture(scope = 'module')


def capped(  ) :
    w = capped_diode_2d()
    m, tmp2 = solved(w, NoRecombination())
    return w,m,tmp2


def test_the_total_current_through_every_cut_is_the_same(plain)  -> None :
    cnt, z2, el =  plain
    val= cut_currents(cnt,z2,el)

    w2 = spread(val)
    assert w2< 1e-6, f"the cut current varies by {w2:.2e}"
def test_the_terminal_currents_sum_to_zero(plain)->None:

    d,h,s= plain
    i=terminal_currents(d, h, s)


    v2 = max(abs(info)for info in i.values())
    assert abs(sum(i.values()))<1e-8*v2




def test_no_density_is_negative_anywhere(plain)-> None:
    _, ret, _=plain

    assert np.all(ret.n.data>0.0)
    assert np.all(ret.p.data>0.0)



def test_the_current_runs_from_the_anode(plain)  ->None:
    v,x,item= plain
    m =terminal_currents(v,x,item)
    assert m['anode']  > 0.0
    assert m['cathode']<0.0


def test_the_solution_is_the_same_on_every_row(plain) -> None :
    i, m, _  =plain
    zz,r=i.mesh.nx,i.mesh.ny
    for ss,z in(('psi',m.psi),("n",m.n),("p",m.p)):

        bar =z.data.reshape(r, zz)
        for x in range(1,r):
            np.testing.assert_allclose(
                bar[x], bar[0], rtol = 1e-12, err_msg =  f"{ss} row {x}"
            )
def test_the_2d_answer_is_the_1d_answer(plain)->None:
    m, f,  _  = plain

    b= build_device(mesh =x_axis(), doping= abrupt_junction(Na =DOPING,Nd=DOPING,position= JUNCTION), contacts=(OhmicPlate(name='anode',nodes=(0,),voltage=BIAS), OhmicPlate(name ='cathode',nodes =(NX-1,),voltage=0.0),),)
    j , _  =  solved(  b , NoRecombination( ))
    d  = f.psi.data.reshape(m.mesh.ny, m.mesh.nx)
    np.testing.assert_allclose(d[0], j.psi.data, rtol =1e-10)
    d=f.n.data.reshape(m.mesh.ny,m.mesh.nx)
    np.testing.assert_allclose(d[0], j.n.data, rtol =1e-10)

    d= f.p.data.reshape(m.mesh.ny,m.mesh.nx)
    np.testing.assert_allclose(d[0],j.p.data,rtol =1e-10)


def test_the_terminal_current_is_the_1d_one_times_the_height(plain) ->  None :
    m, s2, res =plain
    arr   = cut_currents(m,   s2 , res)

    t =build_device(mesh =x_axis(), doping =  abrupt_junction(Na = DOPING, Nd =DOPING, position  = JUNCTION), contacts =  (OhmicPlate(name  =  "anode", nodes  =  (0, ), voltage = BIAS), OhmicPlate(name = "cathode", nodes =(NX - 1, ), voltage = 0.0),),)


    b, g =solved(t, NoRecombination())
    i=  terminal_currents(t, b, g)  ["anode"]
    assert arr.mean() ==pytest.approx(i *HEIGHT,rel= 1e-8)

def test_no_carrier_crosses_into_the_dielectric(capped) ->None:
    xx, ys, dat =  capped
    u,x =current_densities(xx,ys,dat)
    s=  np.flatnonzero(xx.regions.semiconductor_face== 0.0)
    assert s.size>  0
    np.testing.assert_array_equal(u.data[s],0.0)
    np.testing.assert_array_equal(x.data[  s ] , 0.0 )
def test_the_capped_device_still_conserves_current(capped) -> None :
    d,obj,ss= capped;m =  cut_currents (d,  obj,  ss)

    yy =spread(m);assert yy <1e-6, f"the cut current varies by {yy:.2e}"




def  test_the_capped_terminal_currents_sum_to_zero(capped)  -> None  :
    m2,w,u =capped
    it=terminal_currents(m2,w,u)

    f  =  max(abs(a)for a in it.values())

    assert abs(sum(it.values())) < 1e-8 * f


def  test_the_coupled_solve_reproduces_equilibrium_on_a_capped_device(  )  -> None :
    from ddsim.device.equilibrium import  frozen_quasi_fermi, solve_equilibrium
    y =capped_diode_2d(voltage= 0.0)
    h= solve_equilibrium(y, frozen_quasi_fermi(y))


    tmp3, _ =  solved(y, NoRecombination())

    np.testing.assert_allclose(
        tmp3.psi.data,h.psi.data,rtol=1e-10,atol=1e-12
    )
    np.testing.assert_allclose(tmp3.n.data,h.n.data,rtol=1e-10)
    np.testing.assert_allclose(tmp3.p.data, h.p.data, rtol= 1e-10)




def test_a_doping_dependent_mobility_works_on_a_2d_mesh() -> None:
    bar= diode_2d()
    ok = TransportModels.for_device(bar, recombination = NoRecombination(), mobility  = 'arora')

    g=solve_bias_newton(bar,models =ok)

    assert g.newton is  not  None and  g.newton.converged

    x=build_device(
        mesh = x_axis(),
        doping= abrupt_junction(Na  =DOPING, Nd =  DOPING, position=JUNCTION),
        contacts = (
            OhmicPlate(name  = 'anode', nodes = (0, ), voltage = BIAS),
            OhmicPlate(name = "cathode", nodes  = (NX- 1, ), voltage =  0.0),
        ),
    )
    r =TransportModels.for_device(
        x,recombination= NoRecombination(),mobility= 'arora'
    )
    val = solve_bias_newton(x, models  = r)
    assert val.newton is not None and val.newton.converged

    m = g.psi.data.reshape(bar.mesh.ny, bar.mesh.nx)

    np.testing.assert_allclose(m[0], val.psi.data, rtol  = 1e-10)
    f = cut_currents(bar, g, ok)
    d =terminal_currents(x,val,r)['anode']
    assert f.mean()== pytest.approx(d*HEIGHT,
         rel =1e-8)


def test_the_oxide_holds_no_carriers(capped) ->  None:
    yy,k,_=capped
    flag  = list(yy.carrier_free_nodes)
    assert  flag
    np.testing.assert_array_equal(k.n.data[flag], 0.0);  np.testing.assert_array_equal(k.p.data[flag],0.0)
