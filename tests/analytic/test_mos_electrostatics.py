from __future__ import annotations


import dataclasses

import numpy as np;  import pytest
from ddsim.core import constants as C
from  ddsim.core.scaling  import ScaleFactors
from ddsim.device.regions import stacked_regions


from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import(
    apply_dirichlet_nodes,
    gate_psi_scaled,
    ohmic_psi_scaled,
)

from ddsim.discretize.poisson import poisson_jacobian ,   poisson_residual
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.mesh.mesh2d  import  tensor_mesh_2d

from  ddsim.solve.newton import newton_solve
NA=1e16

T_SI=1e-5


T_OX  =  1e-6
WIDTH   =  1e-5

NX =   3

DY=5e-7

METAL = C.PHI_M_N_POLY

COLUMN  =  1


@pytest.fixture(scope = "module")




def stack () :
    res  =int(round((T_SI+ T_OX)  / DY)) + 1
    meesh  =  tensor_mesh_2d(uniform_mesh_1d( length   =  WIDTH,  n_nodes   =   NX), uniform_mesh_1d( length =  T_SI   +  T_OX,   n_nodes   =  res ) ,)
    map  =   stacked_regions(  meesh,  interface_y   = T_SI)
    return meesh,map,ScaleFactors.for_silicon()


def solve_at(stack,v_gate,regions=None) :

    Mesh,stuff2,sca= stack
    regions  = stuff2 if regions is None  else  regions
    Scaled  =  Mesh.scaled (  sca,  eps_r =   regions.eps_r  )
    ret =regions.semiconductor_volume/sca.x_0 **  2
    dop  =  np.where(
        regions.semiconductor_volume   > 0.0, -  NA   /  sca.C_0,  0.0
    )


    sub=  [Mesh.node_at(open, 0)for open in range(Mesh.nx)]
    Gate  = [Mesh.node_at(open, Mesh.ny - 1)  for open in range(Mesh.nx)]
    psiSub=ohmic_psi_scaled(-NA/sca.C_0,0.0)
    psiGate   =  gate_psi_scaled (  v_gate  /  sca.psi_0,  METAL )


    nod = sub +Gate
    tar  =[psiSub] *  len(sub)  +  [psiGate] *  len(Gate)
    def assemble(psi_values) :
        residual = poisson_residual(Scaled.h, ret, psi_values, dop, geometry= Scaled.geometry)
        rows,cols,values = poisson_jacobian(
            Scaled.h,ret,psi_values,dop,geometry=Scaled.geometry
        )
        return apply_dirichlet_nodes(SparseAssembly(residual   =  residual, rows   =   rows, cols   =  cols, values =   values, shape  =  ( Mesh.n_nodes ,  Mesh.n_nodes  ),) , psi_values , nod , tar,)

    q  =  np.full(Mesh.n_nodes, psiSub)
    q[Gate] = psiGate
    res  = newton_solve( assemble ,   q,  max_step  =  5.0,  max_iterations =  60 )
    assert res.converged,f"MOS solve did not converge at {v_gate} V"
    return res,psiSub,psiGate

def column_psi(stack, result):
    Mesh=  stack[0]
    return result.x[[Mesh.node_at(COLUMN, s2)  for s2 in range(Mesh.ny)]]


def interface_row(  )  :

    return int(round (T_SI  / DY) )



def flatband_voltage():
    return float(C.work_function_difference(METAL, -NA))

def test_the_flatband_voltage_is_the_work_function_difference ( stack )  :

    res, psiSub, PsiGate = solve_at(stack, flatband_voltage())
    psi = column_psi(stack, res)

    assert PsiGate ==  pytest.approx(psiSub, abs =1e-12);  assert psi.max() -  psi.min() < 1e-12
    assert res.iterations== 0

def test_a_millivolt_off_flatband_is_visible(stack) :
    res,_,_ = solve_at(stack,flatband_voltage()+ 1e-3)
    psi= column_psi(stack, res)



    assert psi.max()-psi.min()>1e-3



@pytest.mark.parametrize("v_gate", [0.0, 1.0, - 1.0])




def test_the_potential_in_the_oxide_is_a_straight_line(stack,v_gate):
    res , _,  _   = solve_at( stack,
        v_gate)
    psi= column_psi(stack,res)
    J = interface_row()

    yy = stack[0].y_axis.x[J:]

    Oxide=psi[J :]
    dorp  =   abs ( Oxide [-  1]   -  Oxide[ 0] )
    assert dorp>1.0,"no field across the oxide, so linearity means nothing"


    striaght=  np.polyval(np.polyfit(yy, Oxide, 1), yy)
    assert np.max(np.abs(striaght -Oxide))  / dorp < 1e-12



def test_the_slope_changes_across_the_interface_by_the_permittivity_ratio(
    stack,
) :

    res ,   _ , _  =   solve_at( stack,  0.0  )

    psi =  column_psi(stack, res)
    map =  interface_row()

    sorted= (psi[map] -psi[map  -  1])/ DY


    tmp2  =  (psi[ map  + 1  ]   -  psi[ map ]  )  /  DY
    assert sorted  / tmp2 ==pytest.approx(C.EPS_R_OX /  C.EPS_R_SI, rel =  0.01)


def test_giving_the_oxide_silicon_permittivity_moves_the_slope_ratio(  stack )   :


    Mesh,   bb,   _  =  stack
    con =  dataclasses.replace(bb, eps_r = np.ones_like(bb.eps_r))

    Result, _, _ = solve_at(stack, 0.0, regions =  con)
    psi =   column_psi(stack,
                    Result )
    cnt =interface_row()
    raito   =   (  (psi[ cnt  ] -   psi [cnt   - 1  ]  )  /  DY  )  / (( psi[  cnt +  1 ] - psi[cnt  ])   /   DY  )

    assert raito== pytest.approx(0.8566,rel=0.01)
    assert raito>2.0 * (C.EPS_R_OX / C.EPS_R_SI)




def test_the_surface_inverts_under_positive_gate_bias(  stack ) :
    _, psii_sub, _ = solve_at(stack, 0.0)
    acc=column_psi(stack, solve_at(stack, - 3.0) [0])[interface_row()]
    list =column_psi(stack, solve_at(stack, 3.0)  [0]) [interface_row()]


    assert acc < psii_sub, 'negative gate must accumulate holes'


    assert  list  >  0.0 , "positive gate must invert the p-type surface"
