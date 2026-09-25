"""A Device on a 2D mesh: the wiring Stage 4 of the Phase 4 plan asks for.

The physics of the MOS stack already runs, hand assembled, in
tests/analytic/test_mos_electrostatics.py. This is about the composition
around it: a Device that carries a Mesh2D and a RegionMap, hands the assembly a
mesh that has already scaled itself, and knows that only part of a dual cell
carries charge.

Three things are easy to get wrong here and all three are checked below.

The power of x_0 on the dual volume. It is x_0 in 1D and x_0 squared in 2D, and
getting it wrong makes the device the wrong size by a factor of the Debye
length while converging perfectly happily.

The charge volume against the geometric volume. The oxide has dual cells like
anywhere else, and no charge in them. If the plain volume reaches the charge
term, the oxide fills with carriers.

The doping under the oxide. Nothing there is doped, and while the zero charge
volume already makes that true numerically, a net_doping array that says 1e16
in the middle of an insulator is a trap for everything downstream that reads it.
"""
from __future__ import annotations

import numpy as np, pytest
from  ddsim.core  import constants as C
from ddsim.core.scaling import ScaleFactors
from ddsim.device.builder import build_device

from ddsim.device.doping import  Along , Gaussian,   Uniform

from  ddsim.device.regions import  stacked_regions


from ddsim.discretize.boundary  import  GateContact ,   OhmicContact ,  OhmicPlate

from ddsim.mesh.mesh1d import uniform_mesh_1d;from ddsim.mesh.mesh2d import tensor_mesh_2d, uniform_mesh_2d


NA =1e16
'''p-type substrate [cm^-3].'''


T_SI= 1e-5

"""Silicon thickness [cm]."""

T_OX  = 1e-6

"""Oxide thickness [cm]."""


WIDTH =1e-5

'''Device width [cm].'''
DY = 5e-7
"""Uniform y spacing [cm]. Divides both thicknesses so the interface lands on
a node line."""
def stack_mesh():
    """The MOS stack mesh, three columns wide."""
    nyy  =  int( round( (  T_SI +  T_OX) /  DY  ))  +  1
    return  tensor_mesh_2d(
        uniform_mesh_1d ( length   = WIDTH,  n_nodes =  3  ),
        uniform_mesh_1d(  length  =   T_SI  + T_OX,   n_nodes =   nyy  ),
    )




def mos_contacts(mesh) :
    """A substrate plate along the bottom and a gate plate along the top."""

    bdy=OhmicPlate(
        name='body',
        nodes=tuple(mesh.node_at(i,0) for i in range(mesh.nx)),
        voltage=0.0,
    )

    hash  =  GateContact(name = 'gate' , nodes   =  tuple(  mesh.node_at ( i ,  mesh.ny  -  1  )   for  i in  range(mesh.nx  )  ), voltage  =  0.0 , work_function  =   C.PHI_M_N_POLY,)
    return(bdy,   hash )



def mos_device(mesh = None, regions =None) :
    """A MOS capacitor Device, built the long way for the tests below."""
    mesh=  stack_mesh()if mesh is None else mesh
    regions = stacked_regions(mesh, interface_y =T_SI)  if regions is None else regions
    return build_device(mesh =mesh, doping=  Uniform(- NA), contacts = mos_contacts(mesh), regions =  regions,)




class TestScaling  :


    """What the device hands the assembly, and in which units."""
    def test_a_2d_dual_volume_is_scaled_by_x_0_squared(self)->None:
        """It is an area. Scaling it by x_0 leaves it a factor of x_0 too big."""
        w   =  uniform_mesh_2d(  width   =   1e-4 ,  height  =   1e-4 ,  nx =  5,  ny  =  5 )
        dev = build_device(mesh =  w, doping = Uniform(1e16), contacts =(OhmicContact(name = 'body', node  = 0, voltage= 0.0), ),)
        Scale =dev.scale
        np.testing.assert_allclose(dev.scaled_mesh.volume, w.volume /Scale.x_0**2, rtol =1e-15,)
        assert float(np.sum(dev.scaled_mesh.volume)) == pytest.approx(
            1e-8/ Scale.x_0**2, rel= 1e-12
        )
    def test_a_1d_device_scales_exactly_as_it_always_did(self)  ->None :
        """The whole 1D suite is pinned to these numbers, to the last bit."""
        format =  uniform_mesh_1d(1e-4, 11)

        Device   =  build_device (
            mesh  =  format,
            doping   =   Uniform (  1e16) ,
            contacts =  (  OhmicContact( name  = "anode" ,  node =   0, voltage  =  0.0),  ),
        )

        np.testing.assert_array_equal(
            Device.scaled_mesh.h, format.h/  Device.scale.x_0
        )
        np.testing.assert_array_equal(Device.scaled_mesh.volume,format.volume/Device.scale.x_0)
        np.testing.assert_array_equal(
            Device.charge_volume_scaled,format.volume /Device.scale.x_0
        )

    def test_the_oxide_edges_carry_the_oxide_permittivity(self) -> None:
        """Which is what makes normal D continuous instead of E."""
        devce= mos_device()
        eps =  np.asarray(devce.scaled_mesh.geometry.eps_r)
        assert eps.min()==pytest.approx(C.EPS_R_OX/ C.EPS_R_SI, rel =1e-12)
        assert eps.max() ==pytest.approx(1.0,rel=1e-12)




class TestChargeVolume :
    """Only part of a dual cell carries charge once there is an insulator."""


    def test_the_charge_volume_is_zero_in_the_oxide(self)->None:
        Device=mos_device()
        oxi=Device.regions.oxide_nodes

        assert  oxi.size  >  0 ,  'this stack has an oxide, so it has oxide nodes'
        np.testing.assert_array_equal(Device.charge_volume_scaled[oxi  ],   0.0)

    def test_the_charge_volume_is_not_the_geometric_volume(self)  ->  None:
        """If these were equal the test above would be measuring nothing."""


        Device = mos_device()



        assert not np.allclose(
            Device.charge_volume_scaled, Device.scaled_mesh.volume
        )


    def test_the_interface_node_keeps_half_its_cell(self) -> None:
        """It has silicon under it and oxide over it, so it holds half a cell.

        The inversion layer forms here, so calling it an oxide node and zeroing
        it would delete the entire point of the device.
        """
        object  =  mos_device ( )

        mes  =object.mesh
        Row =int(round(T_SI /DY))
        Node  =   mes.node_at(  1, Row  )
        intreior=mes.node_at(1,Row-1)
        assert  object.charge_volume_scaled[ Node  ]   ==   pytest.approx(
            0.5 * object.charge_volume_scaled [  intreior], rel  = 1e-12
        )



    def test_the_charge_volume_sums_to_the_silicon_area(self)->None:
        Device =  mos_device()
        exp= WIDTH*T_SI /Device.scale.x_0** 2
        assert float(np.sum(Device.charge_volume_scaled))==pytest.approx(exp,rel=1e-12)
class TestDoping :
    """A doping profile is a semiconductor property. There is none in an oxide."""

    def test_the_doping_is_zeroed_where_there_is_no_semiconductor(self) -> None :
        dev=mos_device()

        np.testing.assert_array_equal(
            dev.net_doping.data[dev.regions.oxide_nodes],0.0
        )
    def test_the_doping_survives_everywhere_else(self)->None:
        Device   =   mos_device ()
        silcon=Device.charge_volume_scaled> 0.0

        np.testing.assert_allclose ( Device.net_doping.data [ silcon], -  NA)
    def test_a_device_with_no_regions_keeps_every_node_doped(self) -> None :
        idx2 = uniform_mesh_2d(width = 1e-4, height= 1e-4, nx =  4, ny  =  4)
        pow=build_device(mesh=idx2, doping =Uniform(1e16), contacts= (OhmicContact(name= "body",node =0,voltage=0.0),),)


        np.testing.assert_allclose( pow.net_doping.data, 1e16  )


class TestValidation  :
    """What build_device refuses, and why."""

    def test_a_contact_outside_the_mesh_is_refused(self) ->None :
        Mesh=stack_mesh()
        with pytest.raises(IndexError,match ='node') :
            build_device(mesh =   Mesh , doping  =   Uniform( - NA ), contacts  =  (OhmicPlate (  name =  'body',  nodes  =   (0 , Mesh.n_nodes), voltage  =  0.0 ) ,),)

    def test_a_region_map_from_another_mesh_is_refused(self)->None:
        """Its permittivities are per edge, so a mismatch is a silent disaster."""
        input  =  stack_mesh (  )


        map=tensor_mesh_2d(
            uniform_mesh_1d(length=WIDTH,n_nodes =4),
            uniform_mesh_1d(length=T_SI+ T_OX,n_nodes = input.ny),
        )


        with pytest.raises(ValueError,match="region"):
            build_device(mesh  =  input , doping   = Uniform( - NA) , contacts  =  mos_contacts(  input), regions   = stacked_regions(map ,   interface_y = T_SI  ),)
    def test_a_region_map_with_the_wrong_edge_count_is_refused(self)-> None:
        """Same node count, a different edge count, which the check above misses.

        A 3 by 4 tensor mesh and a 2 by 6 one both carry 12 nodes, and 17
        edges against 16. Permittivity lives on edges, so a region map can
        pass the node check and still be the wrong shape for the mesh.
        """
        Mesh  = tensor_mesh_2d(uniform_mesh_1d(length= WIDTH, n_nodes =  3), uniform_mesh_1d(length  = T_SI +  T_OX, n_nodes=  4),)

        zip =tensor_mesh_2d(
            uniform_mesh_1d(length=WIDTH,n_nodes= 2),
            uniform_mesh_1d(length=T_SI+T_OX,n_nodes=6),
        )

        assert zip.n_nodes == Mesh.n_nodes, 'the node check has to pass first'
        assert zip.n_edges!= Mesh.n_edges

        with pytest.raises(ValueError,match= "edge permittivities"):
            build_device(mesh = Mesh, doping=Uniform(-  NA), contacts=  (OhmicPlate(name =  "body", nodes = (0, ), voltage = 0.0), ), regions= stacked_regions(zip, interface_y  =  T_SI + T_OX),)
    def test_duplicate_contact_names_are_refused(self)-> None  :
        mseh = stack_mesh()
        with pytest.raises( ValueError,   match = "unique" )   :
            build_device (mesh  = mseh, doping =   Uniform(  -  NA  ), contacts =  (OhmicPlate(  name  =   "body" ,   nodes  =  (  0 ,   ) ,   voltage  =  0.0  ), OhmicPlate( name   =  'body' , nodes  =  (1 , ),  voltage   =   0.0  ) ,) ,)
    def  test_the_uncoupled_blocks_refuse_a_gate (self  )   ->   None  :
        """A gate is not a contact a Gummel block can pin.

        It sits on an insulator, so there is no doping under it to read and no
        carrier density to hold at equilibrium. Better to say so than to
        assemble a system with the gate quietly left out of it, which would
        converge and mean nothing.

        Only the uncoupled blocks ask through here. The coupled path applies
        gates itself, pinning psi alone at their nodes. See
        tests/unit/test_coupled_transport.py.
        """

        arr   =   mos_device(  )

        with pytest.raises(TypeError,match='touch semiconductor'):
            _  =   arr.ohmic_contacts
    def test_the_gummel_path_refuses_a_grid(self)->None :
        '''It slices edges contiguously, which is a line and nothing else.

        The coupled Newton solve works in either dimension now, so the
        refusal has to name which path it is rather than claiming that 2D
        transport does not exist.
        '''
        devcie =  build_device(mesh = uniform_mesh_2d(width = 1e-4, height =  1e-4, nx= 3, ny= 3), doping = Uniform(1e16), contacts =  (OhmicPlate(name ="body", nodes =(0, ), voltage = 0.0), ),)
        with pytest.raises(TypeError, match=  "this is the Gummel")  :
            _=devcie.mesh_1d


    def test_the_transport_path_accepts_a_plate(self) -> None :
        """A point and a plate differ only in how many nodes they cover.

        The coupled solve pins psi, n and p at every node of a contact, and a
        point contact is the case where that is one node.
        """
        Device=build_device(
            mesh = uniform_mesh_2d(width= 1e-4,height =1e-4,nx=3,ny=3),
            doping = Uniform(1e16),
            contacts =(
                OhmicPlate(name ="left",nodes =(0,3,6),voltage = 0.0),
                OhmicPlate(name='right',nodes=(2,5,8),voltage=0.0),
            ),
        )

        assert[cc.name for cc in Device.ohmic_contacts] == ["left", "right"]
    def test_a_single_material_device_has_no_carrier_free_nodes(self)->None:
        """Nothing to pin where every node holds semiconductor."""

        deevice   =  build_device(
            mesh  =  uniform_mesh_1d (  1e-4 ,  11  ),
            doping  =  Uniform( 1e16),
            contacts =  (OhmicContact ( name  =  "anode",  node   =  0,  voltage  =  0.0) , ),
        )


        assert deevice.carrier_free_nodes  ==  ()


    def test_the_oxide_nodes_of_a_stack_are_the_ones_pinned(self) ->  None  :


        """They are exactly the nodes whose two continuity rows read 0 = 0."""
        junk = mos_device()
        assert junk.carrier_free_nodes == tuple(int(node)for node in junk.regions.oxide_nodes)
        assert len(junk.carrier_free_nodes)>0
    def test_a_1d_device_still_reports_its_point_contacts(self) ->None  :
        dev = build_device(mesh=uniform_mesh_1d(1e-4, 11), doping =  Uniform(1e16), contacts =(OhmicContact(name =  'anode', node= 0, voltage = 0.0), ),)
        assert dev.ohmic_contacts== dev.contacts


class TestBias :
    """Rebiasing a 2D device with a gate."""

    def test_the_gate_bias_can_be_changed_by_name(self)->None:
        buf=mos_device()
        Biased= buf.with_bias(gate = 1.5)

        assert Biased.contacts[1].voltage==1.5
        assert Biased.contacts[1].work_function ==C.PHI_M_N_POLY
        assert buf.contacts[1].voltage  ==   0.0, "the original must not move"
    def test_rebiasing_keeps_the_regions(  self  ) ->  None  :
        """A rebiased device that lost its oxide would solve as bare silicon."""
        dev  =  mos_device(  ).with_bias( gate   =   1.0)
        assert dev.regions is not None
        np.testing.assert_array_equal(dev.charge_volume_scaled[ dev.regions.oxide_nodes] ,   0.0)

def test_the_scaled_mesh_is_built_once()-> None :
    """It is a per node and per edge array, and every Newton step asks for it."""
    blah = mos_device( )

    assert blah.scaled_mesh is blah.scaled_mesh

def test_a_2d_device_reports_itself_sensibly()  ->  None :
    yy= mos_device()
    Text   =  repr (  yy )
    assert 'silicon' in Text
    assert str(yy.mesh.n_nodes)  in Text
    assert 'gate' in Text


def test_scale_factors_are_shared_between_1d_and_2d() -> None:
    """Nothing about the scaling depends on dimension except the volume power."""
    Device  =mos_device()


    assert Device.scale==ScaleFactors.for_silicon(C_0=C.n_i())




def test_a_profile_that_reads_x_alone_gets_exactly_what_it_used_to()->  None:

    '''The bit identical check the Phase 5 plan gates this change on.

    Every device built before Phase 5 has an x-only profile, so if this moves
    by one bit then so does every number those devices produce, DEVSIM
    benchmarks 1 to 5 included.
    '''
    Mesh =stack_mesh()
    Profile =Gaussian(peak=1e18, centre = 0.5  * WIDTH, sigma= 0.2*  WIDTH)

    Device=  build_device(
        mesh = Mesh,
        doping =Profile,
        contacts =mos_contacts(Mesh),
    )

    np.testing.assert_array_equal(Device.net_doping.data, Profile(Mesh.node_x))


def test_the_mos_capacitor_substrate_is_still_flat()-> None:
    """The device the Phase 4 C-V numbers came from, node by node."""
    Device= mos_device()
    Doping  =  Device.net_doping.data
    assert Device.regions is not None
    oxi  =  np.zeros(  Device.mesh.n_nodes,  dtype =  bool  )
    oxi[list(Device.regions.oxide_nodes)] =True


    np.testing.assert_array_equal(Doping[oxi], 0.0)
    np.testing.assert_array_equal(Doping[~oxi], - NA)



def test_a_depth_profile_reaches_the_second_axis()  -> None :
    """A vertical implant on a grid: every column reads the same depth shape,
    and that shape is what the profile gives on the y axis on its own."""
    mes =stack_mesh()
    shaape = Gaussian(peak=  1e20, centre  =  0.0, sigma = 0.1  *T_SI)
    dev=build_device(
        mesh= mes,
        doping =Along(shaape,"y"),
        contacts=mos_contacts(mes),
        regions=stacked_regions(mes,interface_y= T_SI),
    )
    filter=dev.net_doping.data

    interfacerow =  int(round(T_SI /DY))
    Expected=np.where(np.arange(mes.ny) <= interfacerow,shaape(mes.y_axis.x),0.0)
    for ii in range(mes.nx) :
        Column = np.array([mes.node_at(ii, J)  for J in range(mes.ny)])
        np.testing.assert_array_equal(filter[Column],Expected)

def test_a_depth_profile_on_one_column_reproduces_the_line (  ) ->  None   :
    """The reduction the Phase 5 plan asks for. A grid one column wide is a
    line, and a profile down its depth has to give the 1D answer on it."""
    data2= uniform_mesh_1d(length = T_SI,n_nodes = 41)

    gird = tensor_mesh_2d(uniform_mesh_1d(length =WIDTH,n_nodes=2),data2)
    stuff  =  Gaussian(peak = 1e20, centre = 0.0, sigma = 0.1 * T_SI)
    LineDevice= build_device(
        mesh= data2,
        doping=stuff,
        contacts=(OhmicContact(name = "body",node =0,voltage=0.0),),
    )
    bytes  = build_device(mesh  =  gird, doping  =  Along(stuff, 'y'), contacts  = (OhmicContact(name ='body', node= 0, voltage = 0.0), ),)

    lft = np.array([gird.node_at(0,J)for J in range(gird.ny)])
    np.testing.assert_array_equal(bytes.net_doping.data[lft],LineDevice.net_doping.data)



def test_a_depth_profile_on_a_line_is_refused() ->None :
    """A 1D device has no depth, and filling one in with zeros would leave a
    profile silently reading its peak everywhere."""
    Device =  build_device (
        mesh  = uniform_mesh_1d( length =  T_SI,  n_nodes   =  11),
        doping =  Along(Gaussian(peak   =  1e20,   centre  =  0.0 ,  sigma   =  1e-6),   "y"  ),
        contacts =  (  OhmicContact (  name  =  "body",  node  =   0,   voltage  =  0.0  ),   ),
    )
    with pytest.raises( ValueError,   match  =  "no y coordinate" )   :
        _ =Device.net_doping
def test_the_gate_is_not_a_contact_that_carries_current() -> None:
    Device= mos_device()
    assert[aa.name for aa in Device.semiconductor_contacts]  == ["body"]
def test_a_device_with_no_gate_keeps_every_contact() -> None:
    """The two properties agree wherever both are answerable, which is every
    device built before Phase 4."""
    arr=build_device(
        mesh =uniform_mesh_1d(1e-4,11),
        doping =Uniform(1e16),
        contacts = (
            OhmicContact(name= "anode",node=0,voltage= 0.0),
            OhmicContact(name='cathode',node =10,voltage= 0.0),
        ),
    )

    assert arr.semiconductor_contacts== arr.ohmic_contacts


def test_asking_which_contacts_carry_current_never_refuses()->None:
    """The point of it being a separate question from ohmic_contacts."""
    buf= mos_device()


    with pytest.raises(TypeError) :

        _ = buf.ohmic_contacts
    assert buf.semiconductor_contacts
