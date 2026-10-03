from __future__ import annotations
import math
import numpy as np
import pytest, scipy.sparse as sp
from ddsim.core import constants as C
from ddsim.core.scaling import ScaleFactors
from ddsim.discretize.assembly import SparseAssembly
from ddsim.discretize.boundary import(Carrier, GateContact, OhmicContact, OhmicPlate, apply_contacts, apply_dirichlet, apply_dirichlet_nodes, apply_ohmic_contacts, apply_ohmic_densities, gate_psi_scaled, ohmic_psi_scaled,)
from ddsim.discretize.poisson import poisson_jacobian, poisson_residual
from ddsim.mesh.mesh1d import uniform_mesh_1d
from ddsim.physics.statistics import psi_equilibrium_scaled


MICRON= 1e-4


def sample_assembly(n_nodes:int =11,doping:float= 1e6)->tuple :
    z =  ScaleFactors.for_silicon(  )
    k2= uniform_mesh_1d(MICRON,n_nodes)
    x =k2.h/ z.x_0
    xs  = k2.volume  / z.x_0

    psi=  np.full(n_nodes,
      float(psi_equilibrium_scaled(doping)))
    d = np.full(n_nodes, doping)

    tmp2, el, y=  poisson_jacobian(x, xs, psi, d)
    cnt=SparseAssembly(
        residual =poisson_residual(x,xs,psi,d),
        rows=tmp2,
        cols = el,
        values = y,
        shape= (n_nodes,n_nodes),
    )
    return cnt,psi




def _poisson_assembly(mesh: object,scale:ScaleFactors,psi:np.ndarray,net_doping :np.ndarray)->SparseAssembly :
    k2 =mesh.scaled(scale)

    out, u, c =poisson_jacobian(k2.h, k2.volume, psi, net_doping)
    return SparseAssembly(
        residual=  poisson_residual(k2.h, k2.volume, psi, net_doping),
        rows = out,
        cols = u,
        values = c,
        shape=  (psi.size, psi.size),
    )
def dense(assembly :   SparseAssembly  )  -> np.ndarray  :

    return sp.coo_matrix(
        (assembly.values, (assembly.rows, assembly.cols)), shape = assembly.shape
    ).toarray()




def test_contact_potential_at_zero_bias_is_the_equilibrium_potential()  ->   None  :
    assert  ohmic_psi_scaled ( 1e6 ,  0.0 )  ==  pytest.approx (
        float (psi_equilibrium_scaled (  1e6 )),  rel =  1e-15
    )

def test_contact_potential_adds_the_applied_bias()-> None:
    t2 = ohmic_psi_scaled(1e6,0.0)
    u  = ohmic_psi_scaled(1e6, 5.0)
    assert u -t2 ==pytest.approx(5.0,rel= 1e-14)
def test_contact_potential_uses_asinh_not_log()-> None  :
    h =1e6
    assert ohmic_psi_scaled(h, 0.0)==  pytest.approx(math.asinh(h / 2.0), rel  =1e-15)

def test_contact_potential_is_finite_in_compensated_material() -> None:
    for c in(-1e-8, 0.0, 1e-8, - 1e6)  :
        assert  math.isfinite (  ohmic_psi_scaled(  c ,   0.0)  )




def test_contact_potential_is_negative_for_p_type() ->None :
    assert  ohmic_psi_scaled( -  1e6,  0.0  )   <   0.0
def test_dirichlet_residual_is_the_potential_error() ->None:
    v,psi=sample_assembly()


    yy  = apply_dirichlet(v, psi, node=0, target=2.5)
    assert yy.residual[0]== pytest.approx(psi[0]- 2.5, rel =1e-15)


def test_dirichlet_residual_is_zero_when_psi_already_matches()-> None:
    k,psi =sample_assembly()
    j  =  apply_dirichlet (  k,  psi, node  =  0,   target   =   float(  psi [0] ) )

    assert j.residual[0  ] ==   0.0
def  test_dirichlet_row_becomes_the_identity (  )   ->  None :
    y2,psi = sample_assembly()
    d =  dense( apply_dirichlet (  y2,
      psi,
          node  =  3 ,
           target  =  0.0)  )
    f=np.zeros(y2.shape[1])
    f[3]= 1.0
    np.testing.assert_allclose(d[  3, : ],   f , atol   =   0.0 )




def test_dirichlet_eliminates_the_column_as_well_as_the_row()->None :
    el,psi =sample_assembly()

    k2  =  dense(apply_dirichlet ( el,  psi , node   =  3, target =   0.0) )


    c=np.zeros(el.shape[0])
    c[3] = 1.0
    np.testing.assert_allclose(k2[:,3],c,atol = 0.0)




def test_dirichlet_folds_the_column_into_the_neighbouring_residuals()  -> None:
    d,psi= sample_assembly();xx =0.0
    it  = xx  -psi[3]
    val  = dense(d  ) [  :,  3]

    i= apply_dirichlet(d, psi, node = 3, target = xx)

    y = d.residual+val* it


    y[  3  ] =  psi[  3]   - xx


    np.testing.assert_allclose(i.residual, y, rtol =  1e-14)

def test_dirichlet_leaves_untouched_rows_untouched()->  None :
    aa,psi= sample_assembly()
    j =dense(aa)
    t=dense(apply_dirichlet(aa,psi,node =3,target=0.0))
    r = [a for a in range(aa.shape[0])if abs(a- 3) > 1]


    np.testing.assert_allclose(t[r, :], j[r, :], rtol= 1e-15)



def test_the_pinned_value_survives_an_ill_conditioned_system()->  None :

    a = 11
    bb=np.arange(a,dtype=np.int64)

    b  = np.full(a -   1 ,
                  -  1e7  )
    flag=SparseAssembly(
        residual =np.full(a,1e7),
        rows=np.concatenate([bb,bb[:- 1],bb[1 :]]),
        cols=np.concatenate([bb,bb[1:],bb[:-1]]),
        values=np.concatenate([np.full(a,2e7),b,b]),
        shape=(a,a),
    )
    j  =   np.full(  a ,  1e7  )

    v = apply_dirichlet(flag, j, node=  0, target=1e-6)
    it=sp.csc_matrix(
        (v.values,(v.rows,v.cols)),
        shape =v.shape,
    )
    t  = sp.linalg.spsolve(it, -  v.residual)

    assert t[0]==1e-6 - j[0]
def test_dirichlet_does_not_mutate_the_original_assembly() ->  None:
    y,  psi =   sample_assembly( )
    g= y.residual.copy()
    apply_dirichlet(y,psi,node = 0,target = 9.0)

    np.testing.assert_array_equal( y.residual, g)



def test_dirichlet_rejects_a_node_outside_the_mesh(  )  ->  None   :
    yy, psi =  sample_assembly(n_nodes  =11)

    with pytest.raises(IndexError,match= 'node'):
        apply_dirichlet(yy,psi,node=11,target= 0.0)
def test_solving_a_dirichlet_row_reproduces_the_target_exactly() ->  None :
    m, psi  =  sample_assembly()
    f  =   3.0

    val=apply_dirichlet(m,psi,node=0,target=f)
    b2 =  dense(val)
    dat =np.linalg.solve(b2,- val.residual)
    assert psi[0] +dat[0]== pytest.approx(f, rel  =1e-12)


def test_two_contacts_pin_both_ends()-> None:
    c= ScaleFactors.for_silicon()

    c2,psi =sample_assembly(n_nodes =11)

    hh =np.full(11,1e6)
    cc  = (
        OhmicContact(name  ="anode", node  =0, voltage = 0.0),
        OhmicContact(name  =  'cathode', node = 10, voltage = 0.0),
    )
    vv  =  apply_ohmic_contacts(  c2,   psi,  hh,   cc , c )
    ii=dense(vv)
    for v2 in(0, 10):
        res2 = np.zeros(11)
        res2[v2]  = 1.0
        np.testing.assert_allclose(ii[v2, :], res2, atol=  0.0)




def test_contact_voltage_is_converted_from_volts_to_scaled_units()->None :
    f  = ScaleFactors.for_silicon(  )
    m,psi =sample_assembly(n_nodes =11)
    u=np.full(11,1e6)

    idx=(OhmicContact(name= 'anode',node=0,voltage=1.0),)
    jj   =   apply_ohmic_contacts( m, psi, u,  idx , f  )

    a=1.0/ f.psi_0+float(psi_equilibrium_scaled(1e6))

    assert jj.residual[0]== pytest.approx(psi[0]-a,rel=1e-12)



def test_contact_reads_the_doping_at_its_own_node() ->None:
    rows= ScaleFactors.for_silicon()


    s,psi=sample_assembly(n_nodes= 11) ; a= np.concatenate((np.full(5,-1e6),np.full(6,1e6)))


    f =(OhmicContact(name="anode", node= 0, voltage  =  0.0), OhmicContact(name  =  "cathode", node =  10, voltage = 0.0),)
    w =apply_ohmic_contacts(s,psi,a,f,rows)
    xs=psi[0]- w.residual[0]
    z2=psi[10]-w.residual[10]
    assert xs<  0.0, 'p side contact must sit at negative psi'

    assert z2  >  0.0, 'n side contact must sit at positive psi'



def  test_contact_names_must_be_unique ( ) -> None :
    d =ScaleFactors.for_silicon()
    y,psi = sample_assembly(n_nodes=11)
    k2=(OhmicContact(name ="anode", node =  0, voltage  = 0.0), OhmicContact(name  =  "anode", node=  10, voltage=  0.0),)
    with pytest.raises (  ValueError, match   =   "unique|duplicate" ) :
        apply_ohmic_contacts(y, psi, np.full(11, 1e6), k2, d)



def test_built_in_potential_is_the_difference_between_the_two_contacts() -> None :
    r=ScaleFactors.for_silicon()
    v=1e16 /r.C_0


    m  = ohmic_psi_scaled(- v, 0.0)
    item   =  ohmic_psi_scaled( v , 0.0  )
    b = (item - m)  *  r.psi_0
    x= r.psi_0 * math.log(1e16*1e16 / r.C_0 ** 2)
    assert b   == pytest.approx( x,  rel =   1e-6 )

def continuity_assembly(  n_nodes : int  =  5) ->  SparseAssembly  :
    i=  np.arange(n_nodes, dtype = np.int64)
    return  SparseAssembly(residual  =   np.full(  n_nodes , 3.0  ) , rows  = i , cols =   i, values  =  np.full( n_nodes,  2.0), shape  = (n_nodes , n_nodes  ),)


def test_ohmic_densities_pin_the_majority_carrier_to_the_doping() -> None:
    out2  =np.array([1e6, 1e6, 0.0, - 1e6, - 1e6])
    dat = np.zeros(5)
    r = (OhmicContact('cathode', 0, 0.0), )

    res2  =  apply_ohmic_densities (
        continuity_assembly(),  dat,   out2, r ,  Carrier.ELECTRON
    )

    np.testing.assert_allclose(-res2.residual[0],1e6,rtol= 1e-6)


def test_ohmic_densities_pin_the_minority_carrier_by_mass_action() -> None:
    flag =  np.array([1e6, 1e6, 0.0, - 1e6, -  1e6])
    item= np.zeros(5) ; w   = ( OhmicContact ('cathode',   0,  0.0) , )
    x= apply_ohmic_densities(
        continuity_assembly(),item,flag,w,Carrier.HOLE
    )

    np.testing.assert_allclose( -  x.residual [  0 ],  1e-6,  rtol  = 1e-6 )


def test_the_two_pinned_densities_satisfy_mass_action_exactly() ->None:
    g =np.array([- 1e6,0.0,1e6]); m =np.zeros(3)
    row  =(OhmicContact('anode', 0, 0.0), )

    n = - apply_ohmic_densities(
        continuity_assembly(3),m,g,row,Carrier.ELECTRON
    ).residual[0]


    p = -  apply_ohmic_densities(
        continuity_assembly(3),   m , g,   row,   Carrier.HOLE
    ).residual[0 ]
    np.testing.assert_allclose(n*p,1.0,rtol= 1e-15)
    np.testing.assert_allclose(p -n, 1e6, rtol =  1e-12)



def test_the_pinned_density_agrees_with_the_pinned_potential()->  None:

    w = -  1e6
    mm =0.4/ScaleFactors.for_silicon().psi_0


    psi=ohmic_psi_scaled(w,mm)
    d =np.array([w,w])
    j   =   (OhmicContact(  'anode' ,  0,  0.0 ),   )
    n= - apply_ohmic_densities(continuity_assembly(2), np.zeros(2), d, j, Carrier.ELECTRON).residual[0]
    np.testing.assert_allclose(n, math.exp(psi - mm), rtol=1e-12)


def test_ohmic_densities_apply_at_every_contact() -> None :
    cnt= np.array([- 1e6, 0.0, 1e6]); h  = (OhmicContact( 'anode' ,   0,  0.0 ) ,  OhmicContact ( "cathode",   2 , 0.0 ) )

    k2=apply_ohmic_densities(continuity_assembly(3), np.zeros(3), cnt, h, Carrier.ELECTRON)

    np.testing.assert_allclose(-k2.residual[0], 1e-6, rtol=1e-6)
    np.testing.assert_allclose(-  k2.residual[2], 1e6, rtol=  1e-6)
    assert k2.residual[1]== 3.0




def test_pinning_many_nodes_at_once_matches_pinning_them_one_at_a_time() ->None:
    z,psi=sample_assembly(n_nodes = 11)
    idx,  y =  [ 0 ,   5, 10],   [-  1.5,  0.25 , 2.0]

    u = z

    for e,thing in zip(idx,y,strict =True):
        u =apply_dirichlet(u,psi,e,thing)

    c2 =apply_dirichlet_nodes(z,psi,idx,y)
    np.testing.assert_array_equal(c2.residual, u.residual)
    np.testing.assert_array_equal(dense(c2), dense(u))


def  test_pinning_the_same_node_twice_is_rejected ( ) ->  None  :
    c,psi=sample_assembly(n_nodes=5)

    with pytest.raises(ValueError,match ="pinned more than once") :
        apply_dirichlet_nodes(c,psi,[2,2],[0.0,1.0])

def test_pinning_many_nodes_rejects_one_outside_the_mesh() ->None:
    g , psi =  sample_assembly ( n_nodes = 5  )


    with pytest.raises(IndexError,match ='outside the mesh'):
        apply_dirichlet_nodes(g, psi, [0, 9], [0.0, 1.0])


class TestGateContact :

    def test_a_midgap_gate_at_zero_bias_sits_at_zero_psi(self)->  None :

        assert gate_psi_scaled(0.0,C.PHI_M_MIDGAP) ==pytest.approx(
            0.0,abs= 1e-12
        )
    def  test_an_n_poly_gate_sits_half_a_gap_above_intrinsic(self) ->  None :

        row =(C.Eg() /2.0) /C.V_T()

        assert gate_psi_scaled(0.0,C.PHI_M_N_POLY)==pytest.approx(
            row,rel=1e-12
        )
    def test_bias_moves_the_gate_potential_one_for_one(self) ->None:
        cur =  gate_psi_scaled(0.0, C.PHI_M_N_POLY)


        assert gate_psi_scaled(  3.0,   C.PHI_M_N_POLY )  == pytest.approx(
            cur +  3.0 ,   rel  = 1e-12
        )
    @pytest.mark.parametrize('Na', [1e15, 1e16, 1e17])
    @pytest.mark.parametrize(
        "metal", [C.PHI_M_N_POLY,   C.PHI_M_MIDGAP ,  C.PHI_M_P_POLY]
    )
    def test_at_flatband_the_gate_sits_at_the_bulk_potential(self, Na : float, metal  : float)  ->None  :
        m2   =   ScaleFactors.for_silicon(  )

        x = -  Na   /  m2.C_0

        d = float(C.work_function_difference(metal, - Na))
        k = gate_psi_scaled(d /m2.psi_0, metal)
        b = ohmic_psi_scaled(x, 0.0)
        assert k==pytest.approx(b,abs= 1e-9)

    def test_a_gate_contact_holds_a_set_of_nodes(self)-> None:
        thing = GateContact(name= "gate",nodes=(4,5,6),voltage= 1.0,work_function=C.PHI_M_N_POLY)

        assert thing.nodes== (4,5,6)
        assert thing.voltage == 1.0


    def test_a_gate_with_no_nodes_is_refused(self) -> None :

        with pytest.raises(ValueError,match='at least one node'):
            GateContact(name =  "gate", nodes = (), voltage = 0.0, work_function=  C.PHI_M_N_POLY)

    @pytest.mark.parametrize("work_function",[0.0,1.0,8.0,4.05e6])
    def test_a_work_function_no_metal_has_is_refused (  self ,   work_function)  ->  None  :
        with pytest.raises(ValueError,
                 match  = 'work function') :
            GateContact(name= 'gate', nodes  = (4, ), voltage  =0.0, work_function = work_function)
    def test_a_gate_that_names_a_node_twice_is_refused(self)-> None:
        with pytest.raises(ValueError, match = 'more than once') :
            GateContact(
                name = "gate",
                nodes  = (4, 5, 4),
                voltage  =  0.0,
                work_function =  C.PHI_M_N_POLY,
            )

class TestOhmicPlate :
    def test_a_plate_holds_a_set_of_nodes(self)-> None :
        cc = OhmicPlate(name  =  'body', nodes = (0, 1, 2), voltage= 0.5)


        assert cc.nodes==(0,1,2)
        assert  cc.voltage == 0.5
    def test_a_point_contact_reports_its_one_node_as_a_set(self)-> None:

        assert OhmicContact(name='anode',node= 7,voltage = 0.0).nodes==(7,)
    def test_a_plate_with_no_nodes_is_refused(self)-> None:
        with pytest.raises(ValueError,match ="at least one node"):

            OhmicPlate(name = "body", nodes  =(), voltage = 0.0)
    def test_a_plate_that_names_a_node_twice_is_refused(self) ->None:
        with pytest.raises(ValueError, match =  "more than once") :

            OhmicPlate(name = "body", nodes= (3, 4, 3), voltage = 0.0)
    def test_a_plate_pins_every_node_it_covers(self) ->None:


        res =uniform_mesh_1d(MICRON, 7)
        z  = ScaleFactors.for_silicon()
        h  = np.linspace(1e15, 1e17, 7)/ z.C_0

        psi=np.asarray(psi_equilibrium_scaled(h))
        y2 =_poisson_assembly(res,z,psi,h)


        jj =  apply_ohmic_contacts(y2, psi, h, (OhmicPlate(name= "body", nodes= (0, 1, 2), voltage= 0.0), ), z,)
        bar  =sp.coo_matrix((jj.values, (jj.rows, jj.cols)), shape= jj.shape).tocsr()



        for xx in(0, 1, 2):
            assert  bar[ xx ,
                          xx ]   == pytest.approx (1.0  )
            assert bar[ xx  ].nnz  ==  1
            assert jj.residual[xx]==pytest.approx(
                psi[xx]-ohmic_psi_scaled(float(h[xx]),0.0),abs= 1e-14
            )
        assert bar[3].nnz> 1,'an unpinned node must keep its equation'
    def test_a_plate_and_the_same_nodes_as_point_contacts_agree(self)-> None :
        b= uniform_mesh_1d(MICRON,7)

        ret = ScaleFactors.for_silicon()
        k2  =  np.full( 7 ,   1e16 /   ret.C_0)
        psi= np.asarray(psi_equilibrium_scaled(k2))
        a=apply_ohmic_contacts(
            _poisson_assembly(b,ret,psi,k2),
            psi,
            k2,
            (OhmicPlate(name= 'body',nodes =(0,1,2),voltage = 0.25),),
            ret,
        )

        stuff  =  apply_ohmic_contacts(
            _poisson_assembly ( b,   ret,   psi,   k2 ),
            psi,
            k2,
            tuple(
                OhmicContact( name  =  f"c{ys}", node = ys ,   voltage  =   0.25 )
                for ys  in(0 ,  1 ,   2  )
            ),
            ret,
        )

        np.testing.assert_array_equal(a.residual,stuff.residual)
        np.testing.assert_array_equal(a.values,stuff.values)
class TestApplyContacts  :
    def test_it_applies_an_ohmic_contact_and_a_gate_in_one_pass(self)->None :
        h  =  uniform_mesh_1d(MICRON ,  5)
        c =ScaleFactors.for_silicon()
        res2 =  np.full(  5 ,  -   1e16   /  c.C_0);  psi  = np.asarray(  psi_equilibrium_scaled( res2  ))
        thing= (OhmicContact(name="body",node =0,voltage=0.0), GateContact(name = "gate",nodes =(4,),voltage =1.0,work_function= C.PHI_M_N_POLY),)


        x2= apply_contacts(
            _poisson_assembly(h,c,psi,res2),
            psi,
            res2,
            thing,
            c,
        )
        assert x2.residual[0] == pytest.approx(psi[0]  - ohmic_psi_scaled(float(res2[0]), 0.0), abs  =1e-14)
        assert x2.residual[4]==  pytest.approx(
            psi[4] -gate_psi_scaled(1.0  / c.psi_0, C.PHI_M_N_POLY), abs = 1e-14
        )
    def test_a_gate_ignores_the_doping_underneath_it(self) ->None :
        k =uniform_mesh_1d(MICRON,5)
        c2 =  ScaleFactors.for_silicon()
        psi   =  np.zeros( 5  )
        u = (GateContact(name='gate',nodes= (4,),voltage=0.0,work_function = C.PHI_M_MIDGAP),)

        i = apply_contacts(_poisson_assembly(k,c2,psi,np.full(5,1e18 /c2.C_0)), psi, np.full(5,1e18/c2.C_0), u, c2,)
        t =apply_contacts(_poisson_assembly(k, c2, psi, np.zeros(5)), psi, np.zeros(5), u, c2,)

        assert i.residual[4] == t.residual[4]

    def test_two_contacts_on_one_node_are_refused(self) ->  None :
        g   = uniform_mesh_1d ( MICRON,  5 )
        r = ScaleFactors.for_silicon()
        psi  = np.zeros ( 5 )
        aa  =  np.zeros( 5  )

        with pytest.raises(ValueError, match  = "more than once"):
            apply_contacts(_poisson_assembly(g, r, psi, aa), psi, aa, (OhmicContact(name  ='body', node =  0, voltage = 0.0), OhmicPlate(name = "plate", nodes= (0, 1), voltage = 0.0),), r,)


    def test_two_contacts_with_one_name_are_refused(self) -> None:
        s = uniform_mesh_1d(MICRON,5)
        h =  ScaleFactors.for_silicon ( )
        psi  =   np.zeros(  5  )
        f= np.zeros(5)

        with pytest.raises(ValueError, match  = "unique")  :
            apply_contacts (
                _poisson_assembly (  s , h, psi ,  f  ) ,
                psi ,
                f,
                (
                    OhmicContact (name =  "body" ,  node  =  0, voltage  =  0.0 ),
                    OhmicContact (  name  =  "body",  node = 4, voltage  =   0.0 ) ,
                ),
                h,
            )
