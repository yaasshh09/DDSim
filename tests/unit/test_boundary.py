from __future__ import annotations
import  math
import numpy  as np

import  pytest, scipy.sparse as sp


from ddsim.core import constants as C
from ddsim.core.scaling import ScaleFactors

from ddsim.discretize.assembly import SparseAssembly



from ddsim.discretize.boundary import(Carrier, GateContact, OhmicContact, OhmicPlate, apply_contacts, apply_dirichlet, apply_dirichlet_nodes, apply_ohmic_contacts, apply_ohmic_densities, gate_psi_scaled, ohmic_psi_scaled,)


from ddsim.discretize.poisson import poisson_jacobian, poisson_residual


from ddsim.mesh.mesh1d import uniform_mesh_1d


from ddsim.physics.statistics import psi_equilibrium_scaled


MICRON= 1e-4


def sample_assembly(n_nodes:int =11,doping:float= 1e6)->tuple :
    w =  ScaleFactors.for_silicon(  )
    mes= uniform_mesh_1d(MICRON,n_nodes)
    H =mes.h/ w.x_0
    bb  = mes.volume  / w.x_0

    psi=  np.full(n_nodes,
      float(psi_equilibrium_scaled(doping)))
    net = np.full(n_nodes, doping)

    Rows, cools, Values=  poisson_jacobian(H, bb, psi, net)
    ass=SparseAssembly(
        residual =poisson_residual(H,bb,psi,net),
        rows=Rows,
        cols = cools,
        values = Values,
        shape= (n_nodes,n_nodes),
    )
    return ass,psi




def _poisson_assembly(mesh: object,scale:ScaleFactors,psi:np.ndarray,net_doping :np.ndarray)->SparseAssembly :
    Scaled =mesh.scaled(scale)

    Rows, col, Values =poisson_jacobian(Scaled.h, Scaled.volume, psi, net_doping)
    return SparseAssembly(
        residual=  poisson_residual(Scaled.h, Scaled.volume, psi, net_doping),
        rows = Rows,
        cols = col,
        values = Values,
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
    equiilbrium = ohmic_psi_scaled(1e6,0.0)
    cnt  = ohmic_psi_scaled(1e6, 5.0)
    assert cnt -equiilbrium ==pytest.approx(5.0,rel= 1e-14)
def test_contact_potential_uses_asinh_not_log()-> None  :
    Net =1e6
    assert ohmic_psi_scaled(Net, 0.0)==  pytest.approx(math.asinh(Net / 2.0), rel  =1e-15)

def test_contact_potential_is_finite_in_compensated_material() -> None:
    for res in(-1e-8, 0.0, 1e-8, - 1e6)  :
        assert  math.isfinite (  ohmic_psi_scaled(  res ,   0.0)  )




def test_contact_potential_is_negative_for_p_type() ->None :
    assert  ohmic_psi_scaled( -  1e6,  0.0  )   <   0.0
def test_dirichlet_residual_is_the_potential_error() ->None:
    cnt,psi=sample_assembly()


    Constrained  = apply_dirichlet(cnt, psi, node=0, target=2.5)
    assert Constrained.residual[0]== pytest.approx(psi[0]- 2.5, rel =1e-15)


def test_dirichlet_residual_is_zero_when_psi_already_matches()-> None:
    aa,psi =sample_assembly()
    contrained  =  apply_dirichlet (  aa,  psi, node  =  0,   target   =   float(  psi [0] ) )

    assert contrained.residual[0  ] ==   0.0
def  test_dirichlet_row_becomes_the_identity (  )   ->  None :
    arr,psi = sample_assembly()
    foo =  dense( apply_dirichlet (  arr,
      psi,
          node  =  3 ,
           target  =  0.0)  )
    yy=np.zeros(arr.shape[1])
    yy[3]= 1.0
    np.testing.assert_allclose(foo[  3, : ],   yy , atol   =   0.0 )




def test_dirichlet_eliminates_the_column_as_well_as_the_row()->None :
    bb,psi =sample_assembly()

    mtrix  =  dense(apply_dirichlet ( bb,  psi , node   =  3, target =   0.0) )


    expeccted=np.zeros(bb.shape[0])
    expeccted[3] = 1.0
    np.testing.assert_allclose(mtrix[:,3],expeccted,atol = 0.0)




def test_dirichlet_folds_the_column_into_the_neighbouring_residuals()  -> None:
    assemlby,psi= sample_assembly();Target =0.0
    cor  = Target  -psi[3]
    clumn  = dense(assemlby  ) [  :,  3]

    q= apply_dirichlet(assemlby, psi, node = 3, target = Target)

    epxected = assemlby.residual+clumn* cor


    epxected[  3  ] =  psi[  3]   - Target


    np.testing.assert_allclose(q.residual, epxected, rtol =  1e-14)

def test_dirichlet_leaves_untouched_rows_untouched()->  None :
    tmp2,psi= sample_assembly()
    vals =dense(tmp2)
    afetr=dense(apply_dirichlet(tmp2,psi,node =3,target=0.0))
    buf = [ii for ii in range(tmp2.shape[0])if abs(ii- 3) > 1]


    np.testing.assert_allclose(afetr[buf, :], vals[buf, :], rtol= 1e-15)



def test_the_pinned_value_survives_an_ill_conditioned_system()->  None :

    NNodes = 11
    Index=np.arange(NNodes,dtype=np.int64)

    divmod  = np.full(NNodes -   1 ,
                  -  1e7  )
    assemly=SparseAssembly(
        residual =np.full(NNodes,1e7),
        rows=np.concatenate([Index,Index[:- 1],Index[1 :]]),
        cols=np.concatenate([Index,Index[1:],Index[:-1]]),
        values=np.concatenate([np.full(NNodes,2e7),divmod,divmod]),
        shape=(NNodes,NNodes),
    )
    valuues  =   np.full(  NNodes ,  1e7  )

    stuff = apply_dirichlet(assemly, valuues, node=  0, target=1e-6)
    matrrix=sp.csc_matrix(
        (stuff.values,(stuff.rows,stuff.cols)),
        shape =stuff.shape,
    )
    dleta  = sp.linalg.spsolve(matrrix, -  stuff.residual)

    assert dleta[0]==1e-6 - valuues[0]
def test_dirichlet_does_not_mutate_the_original_assembly() ->  None:
    Assembly,  psi =   sample_assembly( )
    bef= Assembly.residual.copy()
    apply_dirichlet(Assembly,psi,node = 0,target = 9.0)

    np.testing.assert_array_equal( Assembly.residual, bef)



def test_dirichlet_rejects_a_node_outside_the_mesh(  )  ->  None   :
    Assembly, psi =  sample_assembly(n_nodes  =11)

    with pytest.raises(IndexError,match= 'node'):
        apply_dirichlet(Assembly,psi,node=11,target= 0.0)
def test_solving_a_dirichlet_row_reproduces_the_target_exactly() ->  None :
    ass, psi  =  sample_assembly()
    temp2  =   3.0

    slice=apply_dirichlet(ass,psi,node=0,target=temp2)
    Matrix =  dense(slice)
    buf =np.linalg.solve(Matrix,- slice.residual)
    assert psi[0] +buf[0]== pytest.approx(temp2, rel  =1e-12)


def test_two_contacts_pin_both_ends()-> None:
    sca= ScaleFactors.for_silicon()

    junk,psi =sample_assembly(n_nodes =11)

    net_dooping =np.full(11,1e6)
    bytes  = (
        OhmicContact(name  ="anode", node  =0, voltage = 0.0),
        OhmicContact(name  =  'cathode', node = 10, voltage = 0.0),
    )
    con  =  apply_ohmic_contacts(  junk,   psi,  net_dooping,   bytes , sca )
    Matrix=dense(con)
    for Node in(0, 10):
        max = np.zeros(11)
        max[Node]  = 1.0
        np.testing.assert_allclose(Matrix[Node, :], max, atol=  0.0)




def test_contact_voltage_is_converted_from_volts_to_scaled_units()->None :
    s2  = ScaleFactors.for_silicon(  )
    asembly,psi =sample_assembly(n_nodes =11)
    NetDoping=np.full(11,1e6)

    w=(OhmicContact(name= 'anode',node=0,voltage=1.0),)
    thing   =   apply_ohmic_contacts( asembly, psi, NetDoping,  w , s2  )

    targget=1.0/ s2.psi_0+float(psi_equilibrium_scaled(1e6))

    assert thing.residual[0]== pytest.approx(psi[0]-targget,rel=1e-12)



def test_contact_reads_the_doping_at_its_own_node() ->None:
    sclae= ScaleFactors.for_silicon()


    asssembly,psi=sample_assembly(n_nodes= 11) ; acc= np.concatenate((np.full(5,-1e6),np.full(6,1e6)))


    con =(OhmicContact(name="anode", node= 0, voltage  =  0.0), OhmicContact(name  =  "cathode", node =  10, voltage = 0.0),)
    object =apply_ohmic_contacts(asssembly,psi,acc,con,sclae)
    anodeTarget=psi[0]- object.residual[0]
    CathodeTarget=psi[10]-object.residual[10]
    assert anodeTarget<  0.0, 'p side contact must sit at negative psi'

    assert CathodeTarget  >  0.0, 'n side contact must sit at positive psi'



def  test_contact_names_must_be_unique ( ) -> None :
    d2 =ScaleFactors.for_silicon()
    Assembly,psi = sample_assembly(n_nodes=11)
    Contacts=(OhmicContact(name ="anode", node =  0, voltage  = 0.0), OhmicContact(name  =  "anode", node=  10, voltage=  0.0),)
    with pytest.raises (  ValueError, match   =   "unique|duplicate" ) :
        apply_ohmic_contacts(Assembly, psi, np.full(11, 1e6), Contacts, d2)



def test_built_in_potential_is_the_difference_between_the_two_contacts() -> None :
    list=ScaleFactors.for_silicon()
    vars=1e16 /list.C_0


    aode  = ohmic_psi_scaled(- vars, 0.0)
    cthode   =  ohmic_psi_scaled( vars , 0.0  )
    vBi = (cthode - aode)  *  list.psi_0
    expeced= list.psi_0 * math.log(1e16*1e16 / list.C_0 ** 2)
    assert vBi   == pytest.approx( expeced,  rel =   1e-6 )

def continuity_assembly(  n_nodes : int  =  5) ->  SparseAssembly  :
    ind=  np.arange(n_nodes, dtype = np.int64)
    return  SparseAssembly(residual  =   np.full(  n_nodes , 3.0  ) , rows  = ind , cols =   ind, values  =  np.full( n_nodes,  2.0), shape  = (n_nodes , n_nodes  ),)


def test_ohmic_densities_pin_the_majority_carrier_to_the_doping() -> None:
    next  =np.array([1e6, 1e6, 0.0, - 1e6, - 1e6])
    den = np.zeros(5)
    set = (OhmicContact('cathode', 0, 0.0), )

    pin  =  apply_ohmic_densities (
        continuity_assembly(),  den,   next, set ,  Carrier.ELECTRON
    )

    np.testing.assert_allclose(-pin.residual[0],1e6,rtol= 1e-6)


def test_ohmic_densities_pin_the_minority_carrier_by_mass_action() -> None:
    Doping =  np.array([1e6, 1e6, 0.0, - 1e6, -  1e6])
    dir= np.zeros(5) ; Contacts   = ( OhmicContact ('cathode',   0,  0.0) , )
    Pinned= apply_ohmic_densities(
        continuity_assembly(),dir,Doping,Contacts,Carrier.HOLE
    )

    np.testing.assert_allclose( -  Pinned.residual [  0 ],  1e-6,  rtol  = 1e-6 )


def test_the_two_pinned_densities_satisfy_mass_action_exactly() ->None:
    arr =np.array([- 1e6,0.0,1e6]); Density =np.zeros(3)
    Contacts  =(OhmicContact('anode', 0, 0.0), )

    n = - apply_ohmic_densities(
        continuity_assembly(3),Density,arr,Contacts,Carrier.ELECTRON
    ).residual[0]


    p = -  apply_ohmic_densities(
        continuity_assembly(3),   Density , arr,   Contacts,   Carrier.HOLE
    ).residual[0 ]
    np.testing.assert_allclose(n*p,1.0,rtol= 1e-15)
    np.testing.assert_allclose(p -n, 1e6, rtol =  1e-12)



def test_the_pinned_density_agrees_with_the_pinned_potential()->  None:

    doing_value = -  1e6
    d2 =0.4/ScaleFactors.for_silicon().psi_0


    psi=ohmic_psi_scaled(doing_value,d2)
    dopng =np.array([doing_value,doing_value])
    con   =   (OhmicContact(  'anode' ,  0,  0.0 ),   )
    n= - apply_ohmic_densities(continuity_assembly(2), np.zeros(2), dopng, con, Carrier.ELECTRON).residual[0]
    np.testing.assert_allclose(n, math.exp(psi - d2), rtol=1e-12)


def test_ohmic_densities_apply_at_every_contact() -> None :
    set= np.array([- 1e6, 0.0, 1e6]); cotacts  = (OhmicContact( 'anode' ,   0,  0.0 ) ,  OhmicContact ( "cathode",   2 , 0.0 ) )

    bin=apply_ohmic_densities(continuity_assembly(3), np.zeros(3), set, cotacts, Carrier.ELECTRON)

    np.testing.assert_allclose(-bin.residual[0], 1e-6, rtol=1e-6)
    np.testing.assert_allclose(-  bin.residual[2], 1e6, rtol=  1e-6)
    assert bin.residual[1]== 3.0




def test_pinning_many_nodes_at_once_matches_pinning_them_one_at_a_time() ->None:
    asseembly,psi=sample_assembly(n_nodes = 11)
    slice,  tar =  [ 0 ,   5, 10],   [-  1.5,  0.25 , 2.0]

    Sequential = asseembly

    for Node,taret in zip(slice,tar,strict =True):
        Sequential =apply_dirichlet(Sequential,psi,Node,taret)

    Batched =apply_dirichlet_nodes(asseembly,psi,slice,tar)
    np.testing.assert_array_equal(Batched.residual, Sequential.residual)
    np.testing.assert_array_equal(dense(Batched), dense(Sequential))


def  test_pinning_the_same_node_twice_is_rejected ( ) ->  None  :
    Assembly,psi=sample_assembly(n_nodes=5)

    with pytest.raises(ValueError,match ="pinned more than once") :
        apply_dirichlet_nodes(Assembly,psi,[2,2],[0.0,1.0])

def test_pinning_many_nodes_rejects_one_outside_the_mesh() ->None:
    id , psi =  sample_assembly ( n_nodes = 5  )


    with pytest.raises(IndexError,match ='outside the mesh'):
        apply_dirichlet_nodes(id, psi, [0, 9], [0.0, 1.0])


class TestGateContact :

    def test_a_midgap_gate_at_zero_bias_sits_at_zero_psi(self)->  None :

        assert gate_psi_scaled(0.0,C.PHI_M_MIDGAP) ==pytest.approx(
            0.0,abs= 1e-12
        )
    def  test_an_n_poly_gate_sits_half_a_gap_above_intrinsic(self) ->  None :

        exxpected =(C.Eg() /2.0) /C.V_T()

        assert gate_psi_scaled(0.0,C.PHI_M_N_POLY)==pytest.approx(
            exxpected,rel=1e-12
        )
    def test_bias_moves_the_gate_potential_one_for_one(self) ->None:
        at =  gate_psi_scaled(0.0, C.PHI_M_N_POLY)


        assert gate_psi_scaled(  3.0,   C.PHI_M_N_POLY )  == pytest.approx(
            at +  3.0 ,   rel  = 1e-12
        )
    @pytest.mark.parametrize('Na', [1e15, 1e16, 1e17])
    @pytest.mark.parametrize(
        "metal", [C.PHI_M_N_POLY,   C.PHI_M_MIDGAP ,  C.PHI_M_P_POLY]
    )
    def test_at_flatband_the_gate_sits_at_the_bulk_potential(self, Na : float, metal  : float)  ->None  :
        sca   =   ScaleFactors.for_silicon(  )

        net_doing = -  Na   /  sca.C_0

        fltband = float(C.work_function_difference(metal, - Na))
        gtae = gate_psi_scaled(fltband /sca.psi_0, metal)
        x2 = ohmic_psi_scaled(net_doing, 0.0)
        assert gtae==pytest.approx(x2,abs= 1e-9)

    def test_a_gate_contact_holds_a_set_of_nodes(self)-> None:
        Gate = GateContact(name= "gate",nodes=(4,5,6),voltage= 1.0,work_function=C.PHI_M_N_POLY)

        assert Gate.nodes== (4,5,6)
        assert Gate.voltage == 1.0


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
        open = OhmicPlate(name  =  'body', nodes = (0, 1, 2), voltage= 0.5)


        assert open.nodes==(0,1,2)
        assert  open.voltage == 0.5
    def test_a_point_contact_reports_its_one_node_as_a_set(self)-> None:

        assert OhmicContact(name='anode',node= 7,voltage = 0.0).nodes==(7,)
    def test_a_plate_with_no_nodes_is_refused(self)-> None:
        with pytest.raises(ValueError,match ="at least one node"):

            OhmicPlate(name = "body", nodes  =(), voltage = 0.0)
    def test_a_plate_that_names_a_node_twice_is_refused(self) ->None:
        with pytest.raises(ValueError, match =  "more than once") :

            OhmicPlate(name = "body", nodes= (3, 4, 3), voltage = 0.0)
    def test_a_plate_pins_every_node_it_covers(self) ->None:


        Mesh =uniform_mesh_1d(MICRON, 7)
        sccale  = ScaleFactors.for_silicon()
        dopiing  = np.linspace(1e15, 1e17, 7)/ sccale.C_0

        psi=np.asarray(psi_equilibrium_scaled(dopiing))
        Assembly =_poisson_assembly(Mesh,sccale,psi,dopiing)


        x2 =  apply_ohmic_contacts(Assembly, psi, dopiing, (OhmicPlate(name= "body", nodes= (0, 1, 2), voltage= 0.0), ), sccale,)
        dat  =sp.coo_matrix((x2.values, (x2.rows, x2.cols)), shape= x2.shape).tocsr()



        for noode in(0, 1, 2):
            assert  dat[ noode ,
                          noode ]   == pytest.approx (1.0  )
            assert dat[ noode  ].nnz  ==  1
            assert x2.residual[noode]==pytest.approx(
                psi[noode]-ohmic_psi_scaled(float(dopiing[noode]),0.0),abs= 1e-14
            )
        assert dat[3].nnz> 1,'an unpinned node must keep its equation'
    def test_a_plate_and_the_same_nodes_as_point_contacts_agree(self)-> None :
        thing= uniform_mesh_1d(MICRON,7)

        sca = ScaleFactors.for_silicon()
        Doping  =  np.full( 7 ,   1e16 /   sca.C_0)
        psi= np.asarray(psi_equilibrium_scaled(Doping))
        asplate=apply_ohmic_contacts(
            _poisson_assembly(thing,sca,psi,Doping),
            psi,
            Doping,
            (OhmicPlate(name= 'body',nodes =(0,1,2),voltage = 0.25),),
            sca,
        )

        obj2  =  apply_ohmic_contacts(
            _poisson_assembly ( thing,   sca,   psi,   Doping ),
            psi,
            Doping,
            tuple(
                OhmicContact( name  =  f"c{node}", node = node ,   voltage  =   0.25 )
                for node  in(0 ,  1 ,   2  )
            ),
            sca,
        )

        np.testing.assert_array_equal(asplate.residual,obj2.residual)
        np.testing.assert_array_equal(asplate.values,obj2.values)
class TestApplyContacts  :
    def test_it_applies_an_ohmic_contact_and_a_gate_in_one_pass(self)->None :
        mes  =  uniform_mesh_1d(MICRON ,  5)
        sca =ScaleFactors.for_silicon()
        Doping =  np.full(  5 ,  -   1e16   /  sca.C_0);  psi  = np.asarray(  psi_equilibrium_scaled( Doping  ))
        conatcts= (OhmicContact(name="body",node =0,voltage=0.0), GateContact(name = "gate",nodes =(4,),voltage =1.0,work_function= C.PHI_M_N_POLY),)


        Pinned= apply_contacts(
            _poisson_assembly(mes,sca,psi,Doping),
            psi,
            Doping,
            conatcts,
            sca,
        )
        assert Pinned.residual[0] == pytest.approx(psi[0]  - ohmic_psi_scaled(float(Doping[0]), 0.0), abs  =1e-14)
        assert Pinned.residual[4]==  pytest.approx(
            psi[4] -gate_psi_scaled(1.0  / sca.psi_0, C.PHI_M_N_POLY), abs = 1e-14
        )
    def test_a_gate_ignores_the_doping_underneath_it(self) ->None :
        Mesh =uniform_mesh_1d(MICRON,5)
        sclae =  ScaleFactors.for_silicon()
        psi   =  np.zeros( 5  )
        Gate = (GateContact(name='gate',nodes= (4,),voltage=0.0,work_function = C.PHI_M_MIDGAP),)

        heeavy = apply_contacts(_poisson_assembly(Mesh,sclae,psi,np.full(5,1e18 /sclae.C_0)), psi, np.full(5,1e18/sclae.C_0), Gate, sclae,)
        aa =apply_contacts(_poisson_assembly(Mesh, sclae, psi, np.zeros(5)), psi, np.zeros(5), Gate, sclae,)

        assert heeavy.residual[4] == aa.residual[4]

    def test_two_contacts_on_one_node_are_refused(self) ->  None :
        Mesh   = uniform_mesh_1d ( MICRON,  5 )
        sale = ScaleFactors.for_silicon()
        psi  = np.zeros ( 5 )
        Doping  =  np.zeros( 5  )

        with pytest.raises(ValueError, match  = "more than once"):
            apply_contacts(_poisson_assembly(Mesh, sale, psi, Doping), psi, Doping, (OhmicContact(name  ='body', node =  0, voltage = 0.0), OhmicPlate(name = "plate", nodes= (0, 1), voltage = 0.0),), sale,)


    def test_two_contacts_with_one_name_are_refused(self) -> None:
        blah = uniform_mesh_1d(MICRON,5)
        min =  ScaleFactors.for_silicon ( )
        psi  =   np.zeros(  5  )
        Doping= np.zeros(5)

        with pytest.raises(ValueError, match  = "unique")  :
            apply_contacts (
                _poisson_assembly (  blah , min, psi ,  Doping  ) ,
                psi ,
                Doping,
                (
                    OhmicContact (name =  "body" ,  node  =  0, voltage  =  0.0 ),
                    OhmicContact (  name  =  "body",  node = 4, voltage  =   0.0 ) ,
                ),
                min,
            )
