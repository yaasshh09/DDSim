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
    s2= [mesh.node_at(dd,2)for dd in range(NX)]
    for f in s2 :
        assert f not in set(regions.oxide_nodes.tolist())
    for ret  in(  3,   4)   :
        for dd  in range (  NX)  :
            assert mesh.node_at(dd,ret) in set(regions.oxide_nodes.tolist())


def test_an_oxide_node_has_no_semiconductor_volume(mesh, regions):

    assert np.all(regions.semiconductor_volume[regions.oxide_nodes] == 0.0)


def test_the_interface_node_keeps_exactly_half_its_dual_cell(mesh, regions) :

    h  =mesh.node_at(1, 2)


    assert  regions.semiconductor_volume[h ] == pytest.approx(0.5  * mesh.volume [ h  ] , rel  =  1e-14)



def test_a_bulk_silicon_node_keeps_all_of_its_dual_cell( mesh ,   regions)  :

    z=mesh.node_at(1,1)


    assert  regions.semiconductor_volume [ z]  ==   pytest.approx(mesh.volume [z  ],  rel  =   1e-14)


def  test_the_semiconductor_volume_sums_to_the_silicon_area (mesh,  regions )  :
    assert regions.semiconductor_volume.sum() ==pytest.approx(
        WIDTH*INTERFACE, rel =1e-14
    )



def test_an_edge_wholly_in_one_material_carries_that_permittivity(mesh, regions) :
    tmp2 =regions.eps_r

    lst  = 1   *   (  NX  -   1 )  +  0

    assert tmp2[lst] ==pytest.approx(1.0,rel= 1e-14)

    buf  =   4  * ( NX   -  1 ) +   0
    assert tmp2[buf]== pytest.approx(
        C.EPS_R_OX  /C.EPS_R_SI, rel = 1e-14
    )

def  test_an_interface_edge_carries_the_average_of_the_two_sides( mesh, regions)   :
    el =  2  *(NX -1)  + 0
    m2 =   0.5   *  (  1.0  +  C.EPS_R_OX  /  C.EPS_R_SI  )

    assert regions.eps_r[el]  ==  pytest.approx( m2 ,   rel  = 1e-14)




def test_a_single_material_device_is_all_ones(mesh):

    u= stacked_regions(mesh,interface_y=HEIGHT * 2.0)

    np.testing.assert_allclose(u.eps_r, 1.0,  rtol   =  0.0  )
    assert u.oxide_nodes.size ==0
    np.testing.assert_allclose(u.semiconductor_volume,  mesh.volume , rtol  =   1e-14)



def test_the_geometry_it_produces_carries_the_permittivity(mesh,regions) :
    it =regions.edge_geometry(mesh)

    np.testing.assert_allclose(np.asarray(it.eps_r), regions.eps_r, rtol  = 0.0)
    np.testing.assert_array_equal(it.edge_nodes, mesh.edge_nodes)



def  test_an_interface_that_misses_every_node_line_is_refused(mesh ) :
    with  pytest.raises (  ValueError,   match   =   'node line')   :
        stacked_regions (mesh ,  interface_y  = 1.2e-5 )


def test_a_region_map_reports_what_it_is(mesh,
                 regions):
    assert "silicon" in repr(regions).lower()
    assert  isinstance( regions,  RegionMap  )



def  test_the_interface_nodes_are_the_row_the_two_materials_share(mesh,  regions )  :
    v2  =  [  mesh.node_at (  num,
                2)   for  num  in range (mesh.nx)  ]
    np.testing.assert_array_equal(regions.interface_nodes(mesh), v2)


def test_an_interface_node_is_a_semiconductor_node(mesh, regions)  :
    t2=regions.interface_nodes(mesh)
    assert not set(t2.tolist()) &set(regions.oxide_nodes.tolist())
    assert np.all(regions.semiconductor_volume[t2]>0.0)



def test_an_interface_node_holds_less_than_its_whole_dual_cell(mesh,regions) :
    b2= regions.interface_nodes(mesh)
    np.testing.assert_array_less(
        regions.semiconductor_volume[b2],mesh.volume[b2]
    )



def  test_a_single_material_device_has_no_interface ( mesh )   :
    prev=stacked_regions(mesh,interface_y =HEIGHT)
    assert prev.interface_nodes(mesh ).size  ==  0


def horizontal_edge(mesh,
   i : int,
        j: int) ->int :
    return j   *  (  mesh.nx - 1)  +  i

def vertical_edge(mesh,i:int,j:int)->int:
    return mesh.n_horizontal +j *mesh.nx  +  i



def test_an_edge_wholly_in_silicon_offers_its_whole_face(mesh,regions):
    c = horizontal_edge(mesh,0,1)
    assert  regions.semiconductor_face[ c ] ==  pytest.approx (
        mesh.dual_face[c  ],  rel   =  1e-14
    )


def test_an_edge_inside_the_oxide_offers_no_face_at_all(mesh,regions):

    assert regions.semiconductor_face[horizontal_edge(mesh, 0, 3)] ==  0.0
    assert  regions.semiconductor_face[  horizontal_edge(mesh , 0 , 4 )]   ==   0.0




def test_a_vertical_edge_leaving_the_interface_offers_nothing(mesh,regions):
    assert regions.semiconductor_face[vertical_edge(mesh, 0, 2)]==0.0


def  test_a_vertical_edge_below_the_interface_keeps_its_whole_face(mesh , regions )   :
    a2 =vertical_edge(mesh, 0, 1)
    assert regions.semiconductor_face[a2]==  pytest.approx(mesh.dual_face[a2], rel =1e-14)




def test_an_edge_along_the_interface_offers_half_its_face(mesh,regions):
    a  =  horizontal_edge (  mesh,
          0,
                    2)

    assert regions.semiconductor_face[a]==pytest.approx(
        0.5 *mesh.dual_face[a],rel= 1e-14
    )




def test_a_single_material_device_offers_every_face_whole(mesh)  :
    tmp2  = stacked_regions(mesh , interface_y   =  HEIGHT   *  2.0)

    np.testing.assert_array_equal(tmp2.semiconductor_face, mesh.dual_face)



def test_the_geometry_it_produces_carries_the_carrier_face(mesh, regions) :
    x2 =  regions.edge_geometry(mesh)

    np.testing.assert_array_equal(
        np.asarray(x2.carrier_face),regions.semiconductor_face
    )
