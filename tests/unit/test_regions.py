from __future__ import annotations
import numpy as np ; import pytest


from ddsim.core import constants as C

from ddsim.device.regions import(OXIDE, SILICON, RegionMap, stacked_regions,)



from ddsim.mesh.mesh2d import uniform_mesh_2d

WIDTH   =  3e-5


HEIGHT=2e-5
NX =4
NY= 5

INTERFACE = 1e-5



@pytest.fixture




def mesh() :
    return uniform_mesh_2d(width =  WIDTH, height = HEIGHT, nx= NX, ny =  NY)


@pytest.fixture
def regions(mesh)  :
    return stacked_regions(mesh,interface_y= INTERFACE)



def test_the_cells_are_split_at_the_interface(mesh, regions):

    assert regions.cell_material.shape==(NY- 1,NX- 1)


    np.testing.assert_array_equal(regions.cell_material[: 2],SILICON); np.testing.assert_array_equal( regions.cell_material[2  :  ], OXIDE )

def test_only_nodes_strictly_inside_the_oxide_have_no_carriers(mesh,regions):
    intterface_nodes= [mesh.node_at(val,2)for val in range(NX)]
    for nde in intterface_nodes :
        assert nde not in set(regions.oxide_nodes.tolist())
    for bb  in(  3,   4)   :
        for val  in range (  NX)  :
            assert mesh.node_at(val,bb) in set(regions.oxide_nodes.tolist())


def test_an_oxide_node_has_no_semiconductor_volume(mesh, regions):

    assert np.all(regions.semiconductor_volume[regions.oxide_nodes] == 0.0)


def test_the_interface_node_keeps_exactly_half_its_dual_cell(mesh, regions) :

    temp2  =mesh.node_at(1, 2)


    assert  regions.semiconductor_volume[temp2 ] == pytest.approx(0.5  * mesh.volume [ temp2  ] , rel  =  1e-14)



def test_a_bulk_silicon_node_keeps_all_of_its_dual_cell( mesh ,   regions)  :

    nod=mesh.node_at(1,1)


    assert  regions.semiconductor_volume [ nod]  ==   pytest.approx(mesh.volume [nod  ],  rel  =   1e-14)


def  test_the_semiconductor_volume_sums_to_the_silicon_area (mesh,  regions )  :
    assert regions.semiconductor_volume.sum() ==pytest.approx(
        WIDTH*INTERFACE, rel =1e-14
    )



def test_an_edge_wholly_in_one_material_carries_that_permittivity(mesh, regions) :
    epsR =regions.eps_r

    SiliconEdge  = 1   *   (  NX  -   1 )  +  0

    assert epsR[SiliconEdge] ==pytest.approx(1.0,rel= 1e-14)

    oe  =   4  * ( NX   -  1 ) +   0
    assert epsR[oe]== pytest.approx(
        C.EPS_R_OX  /C.EPS_R_SI, rel = 1e-14
    )

def  test_an_interface_edge_carries_the_average_of_the_two_sides( mesh, regions)   :
    interace_edge =  2  *(NX -1)  + 0
    exppected =   0.5   *  (  1.0  +  C.EPS_R_OX  /  C.EPS_R_SI  )

    assert regions.eps_r[interace_edge]  ==  pytest.approx( exppected ,   rel  = 1e-14)




def test_a_single_material_device_is_all_ones(mesh):

    reg= stacked_regions(mesh,interface_y=HEIGHT * 2.0)

    np.testing.assert_allclose(reg.eps_r, 1.0,  rtol   =  0.0  )
    assert reg.oxide_nodes.size ==0
    np.testing.assert_allclose(reg.semiconductor_volume,  mesh.volume , rtol  =   1e-14)



def test_the_geometry_it_produces_carries_the_permittivity(mesh,regions) :
    set =regions.edge_geometry(mesh)

    np.testing.assert_allclose(np.asarray(set.eps_r), regions.eps_r, rtol  = 0.0)
    np.testing.assert_array_equal(set.edge_nodes, mesh.edge_nodes)



def  test_an_interface_that_misses_every_node_line_is_refused(mesh ) :
    with  pytest.raises (  ValueError,   match   =   'node line')   :
        stacked_regions (mesh ,  interface_y  = 1.2e-5 )


def test_a_region_map_reports_what_it_is(mesh,
                 regions):
    assert "silicon" in repr(regions).lower()
    assert  isinstance( regions,  RegionMap  )



def  test_the_interface_nodes_are_the_row_the_two_materials_share(mesh,  regions )  :
    abs  =  [  mesh.node_at (  ii,
                2)   for  ii  in range (mesh.nx)  ]
    np.testing.assert_array_equal(regions.interface_nodes(mesh), abs)


def test_an_interface_node_is_a_semiconductor_node(mesh, regions)  :
    vals=regions.interface_nodes(mesh)
    assert not set(vals.tolist()) &set(regions.oxide_nodes.tolist())
    assert np.all(regions.semiconductor_volume[vals]>0.0)



def test_an_interface_node_holds_less_than_its_whole_dual_cell(mesh,regions) :
    hex= regions.interface_nodes(mesh)
    np.testing.assert_array_less(
        regions.semiconductor_volume[hex],mesh.volume[hex]
    )



def  test_a_single_material_device_has_no_interface ( mesh )   :
    rgions=stacked_regions(mesh,interface_y =HEIGHT)
    assert rgions.interface_nodes(mesh ).size  ==  0


def horizontal_edge(mesh,
   i : int,
        j: int) ->int :
    return j   *  (  mesh.nx - 1)  +  i

def vertical_edge(mesh,i:int,j:int)->int:
    return mesh.n_horizontal +j *mesh.nx  +  i



def test_an_edge_wholly_in_silicon_offers_its_whole_face(mesh,regions):
    edg = horizontal_edge(mesh,0,1)
    assert  regions.semiconductor_face[ edg ] ==  pytest.approx (
        mesh.dual_face[edg  ],  rel   =  1e-14
    )


def test_an_edge_inside_the_oxide_offers_no_face_at_all(mesh,regions):

    assert regions.semiconductor_face[horizontal_edge(mesh, 0, 3)] ==  0.0
    assert  regions.semiconductor_face[  horizontal_edge(mesh , 0 , 4 )]   ==   0.0




def test_a_vertical_edge_leaving_the_interface_offers_nothing(mesh,regions):
    assert regions.semiconductor_face[vertical_edge(mesh, 0, 2)]==0.0


def  test_a_vertical_edge_below_the_interface_keeps_its_whole_face(mesh , regions )   :
    buff =vertical_edge(mesh, 0, 1)
    assert regions.semiconductor_face[buff]==  pytest.approx(mesh.dual_face[buff], rel =1e-14)




def test_an_edge_along_the_interface_offers_half_its_face(mesh,regions):
    val  =  horizontal_edge (  mesh,
          0,
                    2)

    assert regions.semiconductor_face[val]==pytest.approx(
        0.5 *mesh.dual_face[val],rel= 1e-14
    )




def test_a_single_material_device_offers_every_face_whole(mesh)  :
    Regions  = stacked_regions(mesh , interface_y   =  HEIGHT   *  2.0)

    np.testing.assert_array_equal(Regions.semiconductor_face, mesh.dual_face)



def test_the_geometry_it_produces_carries_the_carrier_face(mesh, regions) :
    geo =  regions.edge_geometry(mesh)

    np.testing.assert_array_equal(
        np.asarray(geo.carrier_face),regions.semiconductor_face
    )
