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


T_SI= 1e-5

T_OX  = 1e-6


WIDTH =1e-5

DY = 5e-7
def stack_mesh():
    u  =  int( round( (  T_SI +  T_OX) /  DY  ))  +  1
    return  tensor_mesh_2d(
        uniform_mesh_1d ( length   = WIDTH,  n_nodes =  3  ),
        uniform_mesh_1d(  length  =   T_SI  + T_OX,   n_nodes =   u  ),
    )




def mos_contacts(mesh) :

    x=OhmicPlate(
        name='body',
        nodes=tuple(mesh.node_at(rows,0) for rows in range(mesh.nx)),
        voltage=0.0,
    )

    a2  =  GateContact(name = 'gate' , nodes   =  tuple(  mesh.node_at ( m2 ,  mesh.ny  -  1  )   for  m2 in  range(mesh.nx  )  ), voltage  =  0.0 , work_function  =   C.PHI_M_N_POLY,)
    return(x,   a2 )



def mos_device(mesh = None, regions =None) :
    mesh=  stack_mesh()if mesh is None else mesh
    regions = stacked_regions(mesh, interface_y =T_SI)  if regions is None else regions
    return build_device(mesh =mesh, doping=  Uniform(- NA), contacts = mos_contacts(mesh), regions =  regions,)




class TestScaling  :


    def test_a_2d_dual_volume_is_scaled_by_x_0_squared(self)->None:
        bb   =  uniform_mesh_2d(  width   =   1e-4 ,  height  =   1e-4 ,  nx =  5,  ny  =  5 )
        d = build_device(mesh =  bb, doping = Uniform(1e16), contacts =(OhmicContact(name = 'body', node  = 0, voltage= 0.0), ),)
        vv =d.scale
        np.testing.assert_allclose(d.scaled_mesh.volume, bb.volume /vv.x_0**2, rtol =1e-15,)
        assert float(np.sum(d.scaled_mesh.volume)) == pytest.approx(
            1e-8/ vv.x_0**2, rel= 1e-12
        )
    def test_a_1d_device_scales_exactly_as_it_always_did(self)  ->None :
        t =  uniform_mesh_1d(1e-4, 11)

        u   =  build_device (
            mesh  =  t,
            doping   =   Uniform (  1e16) ,
            contacts =  (  OhmicContact( name  = "anode" ,  node =   0, voltage  =  0.0),  ),
        )

        np.testing.assert_array_equal(
            u.scaled_mesh.h, t.h/  u.scale.x_0
        )
        np.testing.assert_array_equal(u.scaled_mesh.volume,t.volume/u.scale.x_0)
        np.testing.assert_array_equal(
            u.charge_volume_scaled,t.volume /u.scale.x_0
        )

    def test_the_oxide_edges_carry_the_oxide_permittivity(self) -> None:
        r= mos_device()
        m =  np.asarray(r.scaled_mesh.geometry.eps_r)
        assert m.min()==pytest.approx(C.EPS_R_OX/ C.EPS_R_SI, rel =1e-12)
        assert m.max() ==pytest.approx(1.0,rel=1e-12)




class TestChargeVolume :


    def test_the_charge_volume_is_zero_in_the_oxide(self)->None:
        b2=mos_device()
        out=b2.regions.oxide_nodes

        assert  out.size  >  0 ,  'this stack has an oxide, so it has oxide nodes'
        np.testing.assert_array_equal(b2.charge_volume_scaled[out  ],   0.0)

    def test_the_charge_volume_is_not_the_geometric_volume(self)  ->  None:


        u = mos_device()



        assert not np.allclose(
            u.charge_volume_scaled, u.scaled_mesh.volume
        )


    def test_the_interface_node_keeps_half_its_cell(self) -> None:
        ss  =  mos_device ( )

        j  =ss.mesh
        g =int(round(T_SI /DY))
        obj  =   j.node_at(  1, g  )
        u=j.node_at(1,g-1)
        assert  ss.charge_volume_scaled[ obj  ]   ==   pytest.approx(
            0.5 * ss.charge_volume_scaled [  u], rel  = 1e-12
        )



    def test_the_charge_volume_sums_to_the_silicon_area(self)->None:
        v =  mos_device()
        i= WIDTH*T_SI /v.scale.x_0** 2
        assert float(np.sum(v.charge_volume_scaled))==pytest.approx(i,rel=1e-12)
class TestDoping :

    def test_the_doping_is_zeroed_where_there_is_no_semiconductor(self) -> None :
        k=mos_device()

        np.testing.assert_array_equal(
            k.net_doping.data[k.regions.oxide_nodes],0.0
        )
    def test_the_doping_survives_everywhere_else(self)->None:
        k   =   mos_device ()
        c2=k.charge_volume_scaled> 0.0

        np.testing.assert_allclose ( k.net_doping.data [ c2], -  NA)
    def test_a_device_with_no_regions_keeps_every_node_doped(self) -> None :
        rr = uniform_mesh_2d(width = 1e-4, height= 1e-4, nx =  4, ny  =  4)
        y=build_device(mesh=rr, doping =Uniform(1e16), contacts= (OhmicContact(name= "body",node =0,voltage=0.0),),)


        np.testing.assert_allclose( y.net_doping.data, 1e16  )


class TestValidation  :

    def test_a_contact_outside_the_mesh_is_refused(self) ->None :
        u=stack_mesh()
        with pytest.raises(IndexError,match ='node') :
            build_device(mesh =   u , doping  =   Uniform( - NA ), contacts  =  (OhmicPlate (  name =  'body',  nodes  =   (0 , u.n_nodes), voltage  =  0.0 ) ,),)

    def test_a_region_map_from_another_mesh_is_refused(self)->None:
        s  =  stack_mesh (  )


        w=tensor_mesh_2d(
            uniform_mesh_1d(length=WIDTH,n_nodes =4),
            uniform_mesh_1d(length=T_SI+ T_OX,n_nodes = s.ny),
        )


        with pytest.raises(ValueError,match="region"):
            build_device(mesh  =  s , doping   = Uniform( - NA) , contacts  =  mos_contacts(  s), regions   = stacked_regions(w ,   interface_y = T_SI  ),)
    def test_a_region_map_with_the_wrong_edge_count_is_refused(self)-> None:
        b  = tensor_mesh_2d(uniform_mesh_1d(length= WIDTH, n_nodes =  3), uniform_mesh_1d(length  = T_SI +  T_OX, n_nodes=  4),)

        v =tensor_mesh_2d(
            uniform_mesh_1d(length=WIDTH,n_nodes= 2),
            uniform_mesh_1d(length=T_SI+T_OX,n_nodes=6),
        )

        assert v.n_nodes == b.n_nodes, 'the node check has to pass first'
        assert v.n_edges!= b.n_edges

        with pytest.raises(ValueError,match= "edge permittivities"):
            build_device(mesh = b, doping=Uniform(-  NA), contacts=  (OhmicPlate(name =  "body", nodes = (0, ), voltage = 0.0), ), regions= stacked_regions(v, interface_y  =  T_SI + T_OX),)
    def test_duplicate_contact_names_are_refused(self)-> None  :
        a2 = stack_mesh()
        with pytest.raises( ValueError,   match = "unique" )   :
            build_device (mesh  = a2, doping =   Uniform(  -  NA  ), contacts =  (OhmicPlate(  name  =   "body" ,   nodes  =  (  0 ,   ) ,   voltage  =  0.0  ), OhmicPlate( name   =  'body' , nodes  =  (1 , ),  voltage   =   0.0  ) ,) ,)
    def  test_the_uncoupled_blocks_refuse_a_gate (self  )   ->   None  :

        mm   =   mos_device(  )

        with pytest.raises(TypeError,match='touch semiconductor'):
            _  =   mm.ohmic_contacts
    def test_the_gummel_path_refuses_a_grid(self)->None :
        cnt =  build_device(mesh = uniform_mesh_2d(width = 1e-4, height =  1e-4, nx= 3, ny= 3), doping = Uniform(1e16), contacts =  (OhmicPlate(name ="body", nodes =(0, ), voltage = 0.0), ),)
        with pytest.raises(TypeError, match=  "this is the Gummel")  :
            _=cnt.mesh_1d


    def test_the_transport_path_accepts_a_plate(self) -> None :
        i=build_device(
            mesh = uniform_mesh_2d(width= 1e-4,height =1e-4,nx=3,ny=3),
            doping = Uniform(1e16),
            contacts =(
                OhmicPlate(name ="left",nodes =(0,3,6),voltage = 0.0),
                OhmicPlate(name='right',nodes=(2,5,8),voltage=0.0),
            ),
        )

        assert[v2.name for v2 in i.ohmic_contacts] == ["left", "right"]
    def test_a_single_material_device_has_no_carrier_free_nodes(self)->None:

        k   =  build_device(
            mesh  =  uniform_mesh_1d (  1e-4 ,  11  ),
            doping  =  Uniform( 1e16),
            contacts =  (OhmicContact ( name  =  "anode",  node   =  0,  voltage  =  0.0) , ),
        )


        assert k.carrier_free_nodes  ==  ()


    def test_the_oxide_nodes_of_a_stack_are_the_ones_pinned(self) ->  None  :


        dd = mos_device()
        assert dd.carrier_free_nodes == tuple(int(b)for b in dd.regions.oxide_nodes)
        assert len(dd.carrier_free_nodes)>0
    def test_a_1d_device_still_reports_its_point_contacts(self) ->None  :
        k = build_device(mesh=uniform_mesh_1d(1e-4, 11), doping =  Uniform(1e16), contacts =(OhmicContact(name =  'anode', node= 0, voltage = 0.0), ),)
        assert k.ohmic_contacts== k.contacts


class TestBias :

    def test_the_gate_bias_can_be_changed_by_name(self)->None:
        b=mos_device()
        f= b.with_bias(gate = 1.5)

        assert f.contacts[1].voltage==1.5
        assert f.contacts[1].work_function ==C.PHI_M_N_POLY
        assert b.contacts[1].voltage  ==   0.0, "the original must not move"
    def test_rebiasing_keeps_the_regions(  self  ) ->  None  :
        t2  =  mos_device(  ).with_bias( gate   =   1.0)
        assert t2.regions is not None
        np.testing.assert_array_equal(t2.charge_volume_scaled[ t2.regions.oxide_nodes] ,   0.0)

def test_the_scaled_mesh_is_built_once()-> None :
    j = mos_device( )

    assert j.scaled_mesh is j.scaled_mesh

def test_a_2d_device_reports_itself_sensibly()  ->  None :
    r= mos_device()
    obj   =  repr (  r )
    assert 'silicon' in obj
    assert str(r.mesh.n_nodes)  in obj
    assert 'gate' in obj


def test_scale_factors_are_shared_between_1d_and_2d() -> None:
    x2  =mos_device()


    assert x2.scale==ScaleFactors.for_silicon(C_0=C.n_i())




def test_a_profile_that_reads_x_alone_gets_exactly_what_it_used_to()->  None:

    z =stack_mesh()
    b2 =Gaussian(peak=1e18, centre = 0.5  * WIDTH, sigma= 0.2*  WIDTH)

    cc=  build_device(
        mesh = z,
        doping =b2,
        contacts =mos_contacts(z),
    )

    np.testing.assert_array_equal(cc.net_doping.data, b2(z.node_x))


def test_the_mos_capacitor_substrate_is_still_flat()-> None:
    r= mos_device()
    v  =  r.net_doping.data
    assert r.regions is not None
    tt  =  np.zeros(  r.mesh.n_nodes,  dtype =  bool  )
    tt[list(r.regions.oxide_nodes)] =True


    np.testing.assert_array_equal(v[tt], 0.0)
    np.testing.assert_array_equal(v[~tt], - NA)



def test_a_depth_profile_reaches_the_second_axis()  -> None :
    thing =stack_mesh()
    vv = Gaussian(peak=  1e20, centre  =  0.0, sigma = 0.1  *T_SI)
    s=build_device(
        mesh= thing,
        doping =Along(vv,"y"),
        contacts=mos_contacts(thing),
        regions=stacked_regions(thing,interface_y= T_SI),
    )
    w=s.net_doping.data

    m =  int(round(T_SI /DY))
    k2=np.where(np.arange(thing.ny) <= m,vv(thing.y_axis.x),0.0)
    for it in range(thing.nx) :
        ok = np.array([thing.node_at(it, xs)  for xs in range(thing.ny)])
        np.testing.assert_array_equal(w[ok],k2)

def test_a_depth_profile_on_one_column_reproduces_the_line (  ) ->  None   :
    vv= uniform_mesh_1d(length = T_SI,n_nodes = 41)

    x = tensor_mesh_2d(uniform_mesh_1d(length =WIDTH,n_nodes=2),vv)
    kk  =  Gaussian(peak = 1e20, centre = 0.0, sigma = 0.1 * T_SI)
    item= build_device(
        mesh= vv,
        doping=kk,
        contacts=(OhmicContact(name = "body",node =0,voltage=0.0),),
    )
    d  = build_device(mesh  =  x, doping  =  Along(kk, 'y'), contacts  = (OhmicContact(name ='body', node= 0, voltage = 0.0), ),)

    v = np.array([x.node_at(0,h)for h in range(x.ny)])
    np.testing.assert_array_equal(d.net_doping.data[v],item.net_doping.data)



def test_a_depth_profile_on_a_line_is_refused() ->None :
    it =  build_device (
        mesh  = uniform_mesh_1d( length =  T_SI,  n_nodes   =  11),
        doping =  Along(Gaussian(peak   =  1e20,   centre  =  0.0 ,  sigma   =  1e-6),   "y"  ),
        contacts =  (  OhmicContact (  name  =  "body",  node  =   0,   voltage  =  0.0  ),   ),
    )
    with pytest.raises( ValueError,   match  =  "no y coordinate" )   :
        _ =it.net_doping
def test_the_gate_is_not_a_contact_that_carries_current() -> None:
    g= mos_device()
    assert[d.name for d in g.semiconductor_contacts]  == ["body"]
def test_a_device_with_no_gate_keeps_every_contact() -> None:
    val=build_device(
        mesh =uniform_mesh_1d(1e-4,11),
        doping =Uniform(1e16),
        contacts = (
            OhmicContact(name= "anode",node=0,voltage= 0.0),
            OhmicContact(name='cathode',node =10,voltage= 0.0),
        ),
    )

    assert val.semiconductor_contacts== val.ohmic_contacts


def test_asking_which_contacts_carry_current_never_refuses()->None:
    z= mos_device()


    with pytest.raises(TypeError) :

        _ = z.ohmic_contacts
    assert z.semiconductor_contacts
