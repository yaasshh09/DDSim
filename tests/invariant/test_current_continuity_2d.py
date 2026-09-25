from __future__ import annotations
import numpy as np;  import pytest
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

from ddsim.mesh.mesh1d  import  graded_mesh_1d,   stacked_mesh_1d , uniform_mesh_1d

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
    mes= tensor_mesh_2d(x_axis(), uniform_mesh_1d(length=  HEIGHT, n_nodes= NY))

    return build_device(
        mesh   =  mes,
        doping  =  abrupt_junction(  Na  = DOPING , Nd  = DOPING,  position   =  JUNCTION ),
        contacts =  (
            OhmicPlate(
                name  =  'anode',
                nodes  =  tuple(  mes.node_at( 0, j)   for  j  in  range( mes.ny)  ),
                voltage =  voltage ,
            ) ,
            OhmicPlate (
                name  = 'cathode',
                nodes =   tuple (mes.node_at(  mes.nx -  1 , j )  for  j  in range ( mes.ny ) ),
                voltage = 0.0 ,
            ) ,
        ),
    )



def capped_diode_2d(voltage:float  =BIAS):

    silcion  =   uniform_mesh_1d(  length  =  HEIGHT , n_nodes  = NY)

    zip = uniform_mesh_1d(length  = T_OX, n_nodes =N_OX)
    mseh  =tensor_mesh_2d(x_axis(), stacked_mesh_1d(silcion, zip))
    ret  =  stacked_regions(  mseh ,
      interface_y   =   HEIGHT)

    return build_device(mesh = mseh, doping  = abrupt_junction(Na = DOPING, Nd =DOPING, position = JUNCTION), contacts= (OhmicPlate(name =  'anode', nodes =tuple(mseh.node_at(0, j) for j in range(NY)), voltage  =voltage,), OhmicPlate(name = "cathode", nodes = tuple(mseh.node_at(mseh.nx - 1, j)for j in range(NY)), voltage = 0.0,),), regions=ret,)


def solved(device,recombination = None):
    k2=TransportModels.for_device(device,recombination=recombination)
    State = solve_bias_newton(device, models = k2)
    assert  State.newton is  not  None and  State.newton.converged,   (
        f"the 2D solve did not converge: {State.newton}"
    )
    return State, k2

def cut_currents(device,state,models) ->np.ndarray :
    stuff =  device.mesh
    Jnn,Jpp= current_densities(device,state,models)
    ttal= Jnn.data+ Jpp.data
    w =edge_current_face(device)
    perCut=[]
    for ii in range(  stuff.nx  - 1)   :
        edg=[jj*(stuff.nx - 1) +ii for jj in range(stuff.ny)]
        perCut.append(float(np.sum(ttal[edg]*w[edg])))
    return np.asarray(perCut )
def spread ( values  : np.ndarray )  ->  float  :
    return float(np.max(np.abs(values -  values.mean()))/abs(values.mean()))




@pytest.fixture(scope =  'module')
def plain():
    dev  =diode_2d()
    sta,   zz  =  solved(dev, NoRecombination(  ))
    return dev, sta, zz



@pytest.fixture(scope = 'module')


def capped(  ) :
    d2 = capped_diode_2d()
    staate, Models = solved(d2, NoRecombination())
    return d2,staate,Models


def test_the_total_current_through_every_cut_is_the_same(plain)  -> None :
    Device, sta, d2 =  plain
    idx2= cut_currents(Device,sta,d2)

    bytes = spread(idx2)
    assert bytes< 1e-6, f"the cut current varies by {bytes:.2e}"
def test_the_terminal_currents_sum_to_zero(plain)->None:

    dev,w,Models= plain
    vals=terminal_currents(dev, w, Models)


    Largest = max(abs(value)for value in vals.values())
    assert abs(sum(vals.values()))<1e-8*Largest




def test_no_density_is_negative_anywhere(plain)-> None:
    _, sta, _=plain

    assert np.all(sta.n.data>0.0)
    assert np.all(sta.p.data>0.0)



def test_the_current_runs_from_the_anode(plain)  ->None:
    devce,staate,Models= plain
    Currents =terminal_currents(devce,staate,Models)
    assert Currents['anode']  > 0.0
    assert Currents['cathode']<0.0


def test_the_solution_is_the_same_on_every_row(plain) -> None :
    deivce, sttate, _  =plain
    Nx,nyy=deivce.mesh.nx,deivce.mesh.ny
    for nam,temp2 in(('psi',sttate.psi),("n",sttate.n),("p",sttate.p)):

        roows =temp2.data.reshape(nyy, Nx)
        for jj in range(1,nyy):
            np.testing.assert_allclose(
                roows[jj], roows[0], rtol = 1e-12, err_msg =  f"{nam} row {jj}"
            )
def test_the_2d_answer_is_the_1d_answer(plain)->None:
    divmod, sate,  _  = plain

    temp2= build_device(mesh =x_axis(), doping= abrupt_junction(Na =DOPING,Nd=DOPING,position= JUNCTION), contacts=(OhmicPlate(name='anode',nodes=(0,),voltage=BIAS), OhmicPlate(name ='cathode',nodes =(NX-1,),voltage=0.0),),)
    refeerence , _  =  solved(  temp2 , NoRecombination( ))
    Rows  = sate.psi.data.reshape(divmod.mesh.ny, divmod.mesh.nx)
    np.testing.assert_allclose(Rows[0], refeerence.psi.data, rtol =1e-10)
    Rows=sate.n.data.reshape(divmod.mesh.ny,divmod.mesh.nx)
    np.testing.assert_allclose(Rows[0], refeerence.n.data, rtol =1e-10)

    Rows= sate.p.data.reshape(divmod.mesh.ny,divmod.mesh.nx)
    np.testing.assert_allclose(Rows[0],refeerence.p.data,rtol =1e-10)


def test_the_terminal_current_is_the_1d_one_times_the_height(plain) ->  None :
    bb, staate, blah =plain
    Cuts   = cut_currents(bb,   staate , blah)

    chr =build_device(mesh =x_axis(), doping =  abrupt_junction(Na = DOPING, Nd =DOPING, position  = JUNCTION), contacts =  (OhmicPlate(name  =  "anode", nodes  =  (0, ), voltage = BIAS), OhmicPlate(name = "cathode", nodes =(NX - 1, ), voltage = 0.0),),)


    refernece, stuff2 =solved(chr, NoRecombination())
    tmp2=  terminal_currents(chr, refernece, stuff2)  ["anode"]
    assert Cuts.mean() ==pytest.approx(tmp2 *HEIGHT,rel= 1e-8)

def test_no_carrier_crosses_into_the_dielectric(capped) ->None:
    Device, staate, mod =  capped
    str,Jpp =current_densities(Device,staate,mod)
    blo=  np.flatnonzero(Device.regions.semiconductor_face== 0.0)
    assert blo.size>  0
    np.testing.assert_array_equal(str.data[blo],0.0)
    np.testing.assert_array_equal(Jpp.data[  blo ] , 0.0 )
def test_the_capped_device_still_conserves_current(capped) -> None :
    all,stte,moodels= capped;zz =  cut_currents (all,  stte,  moodels)

    dev =spread(zz);assert dev <1e-6, f"the cut current varies by {dev:.2e}"




def  test_the_capped_terminal_currents_sum_to_zero(capped)  -> None  :
    dev,sate,res =capped
    temp2=terminal_currents(dev,sate,res)

    thing  =  max(abs(value)for value in temp2.values())

    assert abs(sum(temp2.values())) < 1e-8 * thing


def  test_the_coupled_solve_reproduces_equilibrium_on_a_capped_device(  )  -> None :
    from ddsim.device.equilibrium import  frozen_quasi_fermi, solve_equilibrium
    hash =capped_diode_2d(voltage= 0.0)
    range= solve_equilibrium(hash, frozen_quasi_fermi(hash))


    t2, _ =  solved(hash, NoRecombination())

    np.testing.assert_allclose(
        t2.psi.data,range.psi.data,rtol=1e-10,atol=1e-12
    )
    np.testing.assert_allclose(t2.n.data,range.n.data,rtol=1e-10)
    np.testing.assert_allclose(t2.p.data, range.p.data, rtol= 1e-10)




def test_a_doping_dependent_mobility_works_on_a_2d_mesh() -> None:
    deevice= diode_2d()
    tmp = TransportModels.for_device(deevice, recombination = NoRecombination(), mobility  = 'arora')

    foo=solve_bias_newton(deevice,models =tmp)

    assert foo.newton is  not  None and  foo.newton.converged

    cnt=build_device(
        mesh = x_axis(),
        doping= abrupt_junction(Na  =DOPING, Nd =  DOPING, position=JUNCTION),
        contacts = (
            OhmicPlate(name  = 'anode', nodes = (0, ), voltage = BIAS),
            OhmicPlate(name = "cathode", nodes  = (NX- 1, ), voltage =  0.0),
        ),
    )
    refereence_models =TransportModels.for_device(
        cnt,recombination= NoRecombination(),mobility= 'arora'
    )
    ref = solve_bias_newton(cnt, models  = refereence_models)
    assert ref.newton is not None and ref.newton.converged

    Rows = foo.psi.data.reshape(deevice.mesh.ny, deevice.mesh.nx)

    np.testing.assert_allclose(Rows[0], ref.psi.data, rtol  = 1e-10)
    cts = cut_currents(deevice, foo, tmp)
    expectted =terminal_currents(cnt,ref,refereence_models)['anode']
    assert cts.mean()== pytest.approx(expectted*HEIGHT,
         rel =1e-8)


def test_the_oxide_holds_no_carriers(capped) ->  None:
    dev,sttate,_=capped
    Oxide  = list(dev.carrier_free_nodes)
    assert  Oxide
    np.testing.assert_array_equal(sttate.n.data[Oxide], 0.0);  np.testing.assert_array_equal(sttate.p.data[Oxide],0.0)
