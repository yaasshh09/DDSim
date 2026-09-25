from __future__ import annotations


import  numpy as  np;  import pytest
from ddsim.mesh.mesh1d import graded_mesh_1d,uniform_mesh_1d

from ddsim.mesh.mesh2d import normal_field, tensor_mesh_2d, uniform_mesh_2d


@pytest.fixture



def mesh()  :
    return uniform_mesh_2d(width=2e-4,height= 1e-4,nx= 5,ny =4)

def test_the_node_count_is_the_product_of_the_axes(mesh) :

    assert  mesh.nx ==  5
    assert mesh.ny  ==  4

    assert mesh.n_nodes ==20


def test_the_edge_count_is_both_families_added(mesh ) :
    assert mesh.n_edges==(5- 1) * 4+5*(4-1)
    assert mesh.n_edges ==mesh.h.size ==mesh.dual_face.size



def test_the_dual_cells_sum_to_the_domain_area(mesh):
    assert mesh.volume.sum()==pytest.approx(2e-4 *1e-4,rel=1e-14)




def  test_every_edge_joins_two_geometrically_adjacent_nodes ( mesh )  :
    Left, rig =   mesh.edge_nodes[ :,  0 ],   mesh.edge_nodes[ :,  1 ]

    next= mesh.node_x[rig] -mesh.node_x[Left]
    idx2 =  mesh.node_y[rig] -mesh.node_y[Left]

    np.testing.assert_allclose(np.hypot(next,idx2),mesh.h,rtol=1e-14)

    assert  np.all( (  next  ==  0.0 )  ^   (  idx2  == 0.0  ) )
def test_the_dual_faces_are_all_strictly_positive(mesh) :
    assert np.all(mesh.dual_face>0.0)
    assert np.all(mesh.volume>0.0)
    assert  np.all ( mesh.h  > 0.0 )


def test_no_edge_joins_a_node_to_itself_and_all_are_in_range(mesh):

    assert np.all(mesh.edge_nodes  >=0); assert np.all(mesh.edge_nodes < mesh.n_nodes)
    assert np.all(mesh.edge_nodes[:, 0] != mesh.edge_nodes[:, 1])


def test_every_node_is_touched_by_at_least_two_edges(mesh):
    touhces=np.bincount(mesh.edge_nodes.ravel(),minlength= mesh.n_nodes)
    assert touhces.min() ==2
    assert touhces.max()  == 4


def test_a_horizontal_edge_carries_the_vertical_dual_extent(mesh):
    stuff2 =uniform_mesh_1d(length =2e-4,n_nodes= 5)
    ya=  uniform_mesh_1d(length =  1e-4, n_nodes  =  4)

    n_horziontal=(5 - 1)*4
    hor  =mesh.dual_face[:n_horziontal]

    veritcal_face=  mesh.dual_face[n_horziontal :]
    assert set(np.round(hor, 18))<=set(np.round(ya.volume, 18))
    assert set(np.round(veritcal_face,18)) <= set(np.round(stuff2.volume,18))

def test_the_x_structure_matches_the_1d_mesh_it_was_built_from (  ) :

    x_ais =  graded_mesh_1d(length= 1e-4, n_nodes = 41, refine_at  =  0.5e-4, h_min = 1e-7)
    yAxis= uniform_mesh_1d(length = 1e-5, n_nodes =  3)
    zz= tensor_mesh_2d(x_ais,yAxis)

    np.testing.assert_array_equal(zz.node_x[:  zz.nx], x_ais.x)
    np.testing.assert_array_equal(zz.node_x[zz.nx : 2*zz.nx],x_ais.x)




def test_a_graded_axis_survives_the_tensor_product() :
    XAxis =  graded_mesh_1d(length = 1e-4, n_nodes  =  41, refine_at=0.5e-4, h_min = 1e-7)
    abs= tensor_mesh_2d(XAxis, uniform_mesh_1d(length =1e-5, n_nodes =3))


    assert abs.volume.sum()== pytest.approx(1e-4 * 1e-5, rel  =1e-15)
    assert abs.h.min()  == pytest.approx(  XAxis.h.min( ) , rel =  1e-15)

def  test_a_mesh_needs_at_least_two_nodes_on_each_axis(  )   :
    with pytest.raises (  ValueError, match  = 'at least 2 nodes' )  :
        uniform_mesh_2d(width = 1e-4, height  =  1e-4, nx  =  1, ny =  4)
    with pytest.raises(ValueError, match ="at least 2 nodes")  :
        uniform_mesh_2d(width   =  1e-4, height =  1e-4 , nx  =  4,  ny =   1)
def test_node_at_agrees_with_the_row_major_numbering(mesh) :
    for  J in range (mesh.ny)   :
        for ii in range(mesh.nx):
            tmp2   =   mesh.node_at(  ii,
                      J)
            assert mesh.node_x[tmp2]== mesh.x_axis.x[ii]
            assert mesh.node_y [ tmp2 ]   ==  mesh.y_axis.x [ J]



    assert mesh.node_at(0,0)==0
    assert mesh.node_at(mesh.nx- 1, mesh.ny -1) == mesh.n_nodes -1


def test_the_repr_says_the_shape_and_the_size(mesh):
    data2=  repr(mesh)
    assert "5x4" in data2
    assert "20 nodes" in data2;assert f"{mesh.n_edges} edges" in data2
def  test_the_geometry_it_hands_the_assemblies_is_consistent( mesh) :
    geo  =  mesh.edge_geometry(  )


    tmp2,rig=geo.ends(mesh.n_edges)
    np.testing.assert_array_equal(tmp2, mesh.edge_nodes[:, 0])

    np.testing.assert_array_equal(rig,mesh.edge_nodes[:,1])
    np.testing.assert_array_equal (  np.asarray (geo.dual_face  ),  mesh.dual_face )


def test_a_uniform_vertical_gradient_is_recovered_exactly(mesh):
    grradient=3.7e4; psi= grradient*mesh.node_y

    EE =  normal_field(  mesh , psi )


    np.testing.assert_allclose(EE,
              grradient,
       rtol=1e-12)




def test_a_purely_horizontal_potential_has_no_normal_field(mesh) :
    psi =  5.0  *  mesh.node_x

    np.testing.assert_allclose(normal_field(mesh, psi), 0.0, atol = 1e-9)


def test_the_normal_field_is_a_magnitude(mesh) :

    psi = 2.5e4 * mesh.node_y

    any=normal_field(mesh,psi)
    Down =   normal_field(mesh,  - psi)
    assert np.all ( any  >  0.0)
    np.testing.assert_allclose(any, Down, rtol =  1e-14)

def test_an_interior_node_averages_the_edges_either_side ( )   :

    id= graded_mesh_1d(length  = 3e-5, n_nodes =  3, refine_at  = 0.0, h_min=1e-5, max_ratio  =  3.0)


    acc=tensor_mesh_2d(uniform_mesh_1d(length=1e-5,n_nodes = 2),id)
    psi  = np.array ( [  0.0,  0.0,   1.0,  1.0,   3.0,  3.0  ] )
    hh =  id.h

    out2 = normal_field(acc, psi)


    Lower, bb  = 1.0  /  hh[0], 2.0 /  hh[1]
    np.testing.assert_allclose(out2[ 0],  Lower,   rtol  =   1e-12 )
    np.testing.assert_allclose(out2[2],0.5* (Lower +bb),rtol=1e-12)


    np.testing.assert_allclose(out2[4], bb, rtol =1e-12)
def test_a_boundary_row_uses_the_single_edge_it_has(mesh) :
    psi=  1.1e4 * mesh.node_y
    EE=normal_field(mesh,psi) ; list   =   EE[ :  mesh.nx ]
    topp = EE[- mesh.nx  :]

    np.testing.assert_allclose(list, 1.1e4, rtol  =1e-12);np.testing.assert_allclose(topp, 1.1e4, rtol=  1e-12)
def test_the_normal_field_is_one_value_per_node(mesh) :
    psi =np.zeros(mesh.n_nodes)
    assert normal_field(mesh, psi).shape == (mesh.n_nodes, )



def test_a_potential_of_the_wrong_length_is_refused(mesh) :
    with pytest.raises(ValueError, match= 'node') :


        normal_field(mesh,np.zeros(mesh.n_nodes + 1))


def test_a_reversing_field_averages_to_near_zero_before_the_magnitude():
    yAxis = uniform_mesh_1d(length  =2e-5, n_nodes =3)
    mseh =tensor_mesh_2d(uniform_mesh_1d(length=1e-5,n_nodes =2),yAxis)
    psi=  np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])


    w =   normal_field( mseh,   psi  )
    np.testing.assert_allclose(w[2 :4],0.0,atol=1e-9)
    assert np.all(w[:2]>0.0),"the boundary rows still see their one edge"
