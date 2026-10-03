from __future__ import annotations
import dataclasses
import numpy as np; import pytest
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
    k  =int(round((T_SI+ T_OX)  / DY)) + 1
    j  =  tensor_mesh_2d(uniform_mesh_1d( length   =  WIDTH,  n_nodes   =   NX), uniform_mesh_1d( length =  T_SI   +  T_OX,   n_nodes   =  k ) ,)
    v  =   stacked_regions(  j,  interface_y   = T_SI)
    return j,v,ScaleFactors.for_silicon()


def solve_at(stack,v_gate,regions=None) :

    v,b,e= stack
    regions  = b if regions is None  else  regions
    c  =  v.scaled (  e,  eps_r =   regions.eps_r  )
    a =regions.semiconductor_volume/e.x_0 **  2
    j  =  np.where(
        regions.semiconductor_volume   > 0.0, -  NA   /  e.C_0,  0.0
    )


    u=  [v.node_at(y, 0)for y in range(v.nx)]
    info  = [v.node_at(y, v.ny - 1)  for y in range(v.nx)]
    w=ohmic_psi_scaled(-NA/e.C_0,0.0)
    s   =  gate_psi_scaled (  v_gate  /  e.psi_0,  METAL )


    g = u +info
    x2  =[w] *  len(u)  +  [s] *  len(info)
    def assemble(psi_values) :
        w2 = poisson_residual(c.h, a, psi_values, j, geometry= c.geometry)
        x,yy,tmp = poisson_jacobian(
            c.h,a,psi_values,j,geometry=c.geometry
        )
        return apply_dirichlet_nodes(SparseAssembly(residual   =  w2, rows   =   x, cols   =  yy, values =   tmp, shape  =  ( v.n_nodes ,  v.n_nodes  ),) , psi_values , g , x2,)

    k  =  np.full(v.n_nodes, w)
    k[info] = s
    aa  = newton_solve( assemble ,   k,  max_step  =  5.0,  max_iterations =  60 )
    assert aa.converged,f"MOS solve did not converge at {v_gate} V"
    return aa,w,s

def column_psi(stack, result):
    s=  stack[0]
    return result.x[[s.node_at(COLUMN, w)  for w in range(s.ny)]]


def interface_row(  )  :

    return int(round (T_SI  / DY) )



def flatband_voltage():
    return float(C.work_function_difference(METAL, -NA))

def test_the_flatband_voltage_is_the_work_function_difference ( stack )  :

    vals, mm, h = solve_at(stack, flatband_voltage())
    psi = column_psi(stack, vals)

    assert h ==  pytest.approx(mm, abs =1e-12);  assert psi.max() -  psi.min() < 1e-12
    assert vals.iterations== 0

def test_a_millivolt_off_flatband_is_visible(stack) :
    num,_,_ = solve_at(stack,flatband_voltage()+ 1e-3)
    psi= column_psi(stack, num)



    assert psi.max()-psi.min()>1e-3



@pytest.mark.parametrize("v_gate", [0.0, 1.0, - 1.0])




def test_the_potential_in_the_oxide_is_a_straight_line(stack,v_gate):
    foo , _,  _   = solve_at( stack,
        v_gate)
    psi= column_psi(stack,foo)
    r = interface_row()

    tmp = stack[0].y_axis.x[r:]

    w=psi[r :]
    i  =   abs ( w [-  1]   -  w[ 0] )
    assert i>1.0,"no field across the oxide, so linearity means nothing"


    thing=  np.polyval(np.polyfit(tmp, w, 1), tmp)
    assert np.max(np.abs(thing -w))  / i < 1e-12



def test_the_slope_changes_across_the_interface_by_the_permittivity_ratio(
    stack,
) :

    t ,   _ , _  =   solve_at( stack,  0.0  )

    psi =  column_psi(stack, t)
    ii =  interface_row()

    ss= (psi[ii] -psi[ii  -  1])/ DY


    r  =  (psi[ ii  + 1  ]   -  psi[ ii ]  )  /  DY
    assert ss  / r ==pytest.approx(C.EPS_R_OX /  C.EPS_R_SI, rel =  0.01)


def test_giving_the_oxide_silicon_permittivity_moves_the_slope_ratio(  stack )   :


    u,   val2,   _  =  stack
    b =  dataclasses.replace(val2, eps_r = np.ones_like(val2.eps_r))

    r, _, _ = solve_at(stack, 0.0, regions =  b)
    psi =   column_psi(stack,
                    r )
    a2 =interface_row()
    j   =   (  (psi[ a2  ] -   psi [a2   - 1  ]  )  /  DY  )  / (( psi[  a2 +  1 ] - psi[a2  ])   /   DY  )

    assert j== pytest.approx(0.8566,rel=0.01)
    assert j>2.0 * (C.EPS_R_OX / C.EPS_R_SI)




def test_the_surface_inverts_under_positive_gate_bias(  stack ) :
    _, f, _ = solve_at(stack, 0.0)
    vals=column_psi(stack, solve_at(stack, - 3.0) [0])[interface_row()]
    prev =column_psi(stack, solve_at(stack, 3.0)  [0]) [interface_row()]


    assert vals < f, 'negative gate must accumulate holes'


    assert  prev  >  0.0 , "positive gate must invert the p-type surface"
