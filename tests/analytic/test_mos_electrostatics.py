"""MOS electrostatics on the 2D stack: the first two material solve.

This is the piece where Stage 3 of the Phase 4 plan either works or does not.
It composes the structured mesh, the cell based region map, the per edge
permittivity, the carrier free oxide and the gate work function, and nothing
below it has ever been exercised together.

The three checks, in order of how much they would hurt to get wrong:

**Flatband.** Bias the gate at Phi_MS and the whole stack has to sit at one
potential, because that is what flatband means. It is the sharpest test in the
file because it needs the gate work function, the substrate contact potential
and the intrinsic reference to agree exactly, and those are computed by three
different pieces of code that never otherwise meet. A disagreement of a
millivolt slides the entire C-V curve sideways while every regime still looks
correct, which is the failure phases/PHASE-4.md gates at 20 mV.

**The oxide is empty.** No charge means Laplace, and Laplace in one dimension
means a straight line. If any carrier density leaked into the oxide, or if the
semiconductor volume were not zeroed there, the potential would bend.

**Permittivity does the work.** The potential slope changes across the
interface by the ratio of the permittivities, which is displacement continuity
observed rather than imposed. There is no interface code in this project; the
flux balance at the interface node is that condition already.
"""
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

"""p-type substrate [cm^-3]."""
T_SI=1e-5
'''Silicon thickness [cm], 100 nm.'''


T_OX  =  1e-6
"""Oxide thickness [cm], 10 nm."""
WIDTH   =  1e-5

NX =   3

DY=5e-7

"""Uniform y spacing [cm], 5 nm. Divides both thicknesses, so the interface
lands on a node line, which stacked_regions requires."""

METAL = C.PHI_M_N_POLY

COLUMN  =  1

"""The interior column, away from either side wall."""


@pytest.fixture(scope = "module")




def stack () :
    res  =int(round((T_SI+ T_OX)  / DY)) + 1
    meesh  =  tensor_mesh_2d(uniform_mesh_1d( length   =  WIDTH,  n_nodes   =   NX), uniform_mesh_1d( length =  T_SI   +  T_OX,   n_nodes   =  res ) ,)
    map  =   stacked_regions(  meesh,  interface_y   = T_SI)
    return meesh,map,ScaleFactors.for_silicon()


def solve_at(stack,v_gate,regions=None) :
    """Equilibrium Poisson on the MOS stack at one gate bias.

    Assembled here rather than through assemble_poisson, which is still typed
    to a 1D mesh. That is the wiring Stage 4 has to do properly; this proves
    the physics underneath it first.
    """

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
    '''psi down the interior column, bottom to top.'''
    Mesh=  stack[0]
    return result.x[[Mesh.node_at(COLUMN, s2)  for s2 in range(Mesh.ny)]]


def interface_row(  )  :

    return int(round (T_SI  / DY) )



def flatband_voltage():
    return float(C.work_function_difference(METAL, -NA))

def test_the_flatband_voltage_is_the_work_function_difference ( stack )  :

    """At V_gate = Phi_MS the whole stack sits at one potential.

    Measured spread 1.8e-15 in scaled units, which is a few ulps of a
    potential of order 14, so this is exact rather than merely small.

    Newton reports zero iterations, and that is the real content: the flat
    profile is handed in as the guess and is already the solution, so the
    residual starts at the floor. If any of the three potentials disagreed the
    solve would have to move, and it does not move at all.
    """
    res, psiSub, PsiGate = solve_at(stack, flatband_voltage())
    psi = column_psi(stack, res)

    assert PsiGate ==  pytest.approx(psiSub, abs =1e-12);  assert psi.max() -  psi.min() < 1e-12
    assert res.iterations== 0

def test_a_millivolt_off_flatband_is_visible(stack) :
    '''Guards the test above from passing because nothing is connected.

    20 mV is the gate the phase doc puts on flatband, so a test that could not
    see 20 mV would be worthless. One millivolt already moves the surface by
    far more than the flatband spread.
    '''
    res,_,_ = solve_at(stack,flatband_voltage()+ 1e-3)
    psi= column_psi(stack, res)



    assert psi.max()-psi.min()>1e-3



@pytest.mark.parametrize("v_gate", [0.0, 1.0, - 1.0])




def test_the_potential_in_the_oxide_is_a_straight_line(stack,v_gate):
    """No charge means Laplace, and Laplace across an insulator is linear.

    Measured relative deviation from a straight line is 4e-15 to 6e-15, which
    is the solver tolerance rather than any physics. A carrier density leaking
    into the oxide, or a semiconductor volume that was not zeroed there, would
    bend this visibly.
    """
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

    """Displacement continuity, observed rather than imposed.

    E_ox / E_si = eps_Si / eps_ox, so the ratio of the two slopes is
    eps_ox / eps_Si = 0.3333. Measured 0.3310 at zero bias, 0.7 percent below.

    The deviation is physical and not an error. The discrete statement at the
    interface node is Gauss's law over its dual cell, and half that cell is
    silicon holding depletion charge, so the displacement genuinely does jump
    by the charge in that half cell. The gap widens exactly where that charge
    grows: in accumulation at -1.9 V the ratio falls to 0.12, because there is
    then a sheet of holes sitting at the surface. Zero bias is depletion, where
    the half cell charge is small, which is why the check is made there.
    """
    res ,   _ , _  =   solve_at( stack,  0.0  )

    psi =  column_psi(stack, res)
    map =  interface_row()

    sorted= (psi[map] -psi[map  -  1])/ DY


    tmp2  =  (psi[ map  + 1  ]   -  psi[ map ]  )  /  DY
    assert sorted  / tmp2 ==pytest.approx(C.EPS_R_OX /  C.EPS_R_SI, rel =  0.01)


def test_giving_the_oxide_silicon_permittivity_moves_the_slope_ratio(  stack )   :


    """The control that proves eps_r is what produced the number above.

    Same geometry, same doping, same gate, with only the oxide permittivity
    changed to silicon's. The slope ratio moves from 0.331 to 0.857, a factor
    of 2.6. Without this the previous test could be measuring the mesh.
    """
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
    """A p-type surface driven positive is what a MOS capacitor is for.

    Not a tolerance, a direction. The surface potential has to cross from the
    p-type bulk value through intrinsic and out the other side, which is what
    inversion is. If the gate sign were reversed this would fail immediately
    and no capacitance would ever have to be computed to notice.
    """
    _, psii_sub, _ = solve_at(stack, 0.0)
    acc=column_psi(stack, solve_at(stack, - 3.0) [0])[interface_row()]
    list =column_psi(stack, solve_at(stack, 3.0)  [0]) [interface_row()]


    assert acc < psii_sub, 'negative gate must accumulate holes'


    assert  list  >  0.0 , "positive gate must invert the p-type surface"
