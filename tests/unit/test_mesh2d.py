from __future__ import annotations
import numpy as np; import pytest
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
    s, tmp2 =   mesh.edge_nodes[ :,  0 ],   mesh.edge_nodes[ :,  1 ]

    b= mesh.node_x[tmp2] -mesh.node_x[s]
    b2 =  mesh.node_y[tmp2] -mesh.node_y[s]

    np.testing.assert_allclose(np.hypot(b,b2),mesh.h,rtol=1e-14)

    assert  np.all( (  b  ==  0.0 )  ^   (  b2  == 0.0  ) )
def test_the_dual_faces_are_all_strictly_positive(mesh) :
    assert np.all(mesh.dual_face>0.0)
    assert np.all(mesh.volume>0.0)
    assert  np.all ( mesh.h  > 0.0 )


def test_no_edge_joins_a_node_to_itself_and_all_are_in_range(mesh):

    assert np.all(mesh.edge_nodes  >=0); assert np.all(mesh.edge_nodes < mesh.n_nodes)
    assert np.all(mesh.edge_nodes[:, 0] != mesh.edge_nodes[:, 1])


def test_every_node_is_touched_by_at_least_two_edges(mesh):
    bar=np.bincount(mesh.edge_nodes.ravel(),minlength= mesh.n_nodes)
    assert bar.min() ==2
    assert bar.max()  == 4


def test_a_horizontal_edge_carries_the_vertical_dual_extent(mesh):
    tmp3 =uniform_mesh_1d(length =2e-4,n_nodes= 5)
    c=  uniform_mesh_1d(length =  1e-4, n_nodes  =  4)

    b=(5 - 1)*4
    bb  =mesh.dual_face[:b]

    out=  mesh.dual_face[b :]
    assert set(np.round(bb, 18))<=set(np.round(c.volume, 18))
    assert set(np.round(out,18)) <= set(np.round(tmp3.volume,18))

def test_the_x_structure_matches_the_1d_mesh_it_was_built_from (  ) :

    t =  graded_mesh_1d(length= 1e-4, n_nodes = 41, refine_at  =  0.5e-4, h_min = 1e-7)
    r2= uniform_mesh_1d(length = 1e-5, n_nodes =  3)
    d= tensor_mesh_2d(t,r2)

    np.testing.assert_array_equal(d.node_x[:  d.nx], t.x)
    np.testing.assert_array_equal(d.node_x[d.nx : 2*d.nx],t.x)




def test_a_graded_axis_survives_the_tensor_product() :
    s =  graded_mesh_1d(length = 1e-4, n_nodes  =  41, refine_at=0.5e-4, h_min = 1e-7)
    y= tensor_mesh_2d(s, uniform_mesh_1d(length =1e-5, n_nodes =3))


    assert y.volume.sum()== pytest.approx(1e-4 * 1e-5, rel  =1e-15)
    assert y.h.min()  == pytest.approx(  s.h.min( ) , rel =  1e-15)

def  test_a_mesh_needs_at_least_two_nodes_on_each_axis(  )   :
    with pytest.raises (  ValueError, match  = 'at least 2 nodes' )  :
        uniform_mesh_2d(width = 1e-4, height  =  1e-4, nx  =  1, ny =  4)
    with pytest.raises(ValueError, match ="at least 2 nodes")  :
        uniform_mesh_2d(width   =  1e-4, height =  1e-4 , nx  =  4,  ny =   1)
def test_node_at_agrees_with_the_row_major_numbering(mesh) :
    for  cnt in range (mesh.ny)   :
        for cc in range(mesh.nx):
            k   =   mesh.node_at(  cc,
                      cnt)
            assert mesh.node_x[k]== mesh.x_axis.x[cc]
            assert mesh.node_y [ k ]   ==  mesh.y_axis.x [ cnt]



    assert mesh.node_at(0,0)==0
    assert mesh.node_at(mesh.nx- 1, mesh.ny -1) == mesh.n_nodes -1


def test_the_repr_says_the_shape_and_the_size(mesh):
    r2=  repr(mesh)
    assert "5x4" in r2
    assert "20 nodes" in r2;assert f"{mesh.n_edges} edges" in r2
def  test_the_geometry_it_hands_the_assemblies_is_consistent( mesh) :
    y2  =  mesh.edge_geometry(  )


    bar,m=y2.ends(mesh.n_edges)
    np.testing.assert_array_equal(bar, mesh.edge_nodes[:, 0])

    np.testing.assert_array_equal(m,mesh.edge_nodes[:,1])
    np.testing.assert_array_equal (  np.asarray (y2.dual_face  ),  mesh.dual_face )


def test_a_uniform_vertical_gradient_is_recovered_exactly(mesh):
    r2=3.7e4; psi= r2*mesh.node_y

    u =  normal_field(  mesh , psi )


    np.testing.assert_allclose(u,
              r2,
       rtol=1e-12)




def test_a_purely_horizontal_potential_has_no_normal_field(mesh) :
    psi =  5.0  *  mesh.node_x

    np.testing.assert_allclose(normal_field(mesh, psi), 0.0, atol = 1e-9)


def test_the_normal_field_is_a_magnitude(mesh) :

    psi = 2.5e4 * mesh.node_y

    s=normal_field(mesh,psi)
    res2 =   normal_field(mesh,  - psi)
    assert np.all ( s  >  0.0)
    np.testing.assert_allclose(s, res2, rtol =  1e-14)

def test_an_interior_node_averages_the_edges_either_side ( )   :

    ii= graded_mesh_1d(length  = 3e-5, n_nodes =  3, refine_at  = 0.0, h_min=1e-5, max_ratio  =  3.0)


    y2=tensor_mesh_2d(uniform_mesh_1d(length=1e-5,n_nodes = 2),ii)
    psi  = np.array ( [  0.0,  0.0,   1.0,  1.0,   3.0,  3.0  ] )
    m =  ii.h

    v = normal_field(y2, psi)


    t, f  = 1.0  /  m[0], 2.0 /  m[1]
    np.testing.assert_allclose(v[ 0],  t,   rtol  =   1e-12 )
    np.testing.assert_allclose(v[2],0.5* (t +f),rtol=1e-12)


    np.testing.assert_allclose(v[4], f, rtol =1e-12)
def test_a_boundary_row_uses_the_single_edge_it_has(mesh) :
    psi=  1.1e4 * mesh.node_y
    w=normal_field(mesh,psi) ; k   =   w[ :  mesh.nx ]
    g = w[- mesh.nx  :]

    np.testing.assert_allclose(k, 1.1e4, rtol  =1e-12);np.testing.assert_allclose(g, 1.1e4, rtol=  1e-12)
def test_the_normal_field_is_one_value_per_node(mesh) :
    psi =np.zeros(mesh.n_nodes)
    assert normal_field(mesh, psi).shape == (mesh.n_nodes, )



def test_a_potential_of_the_wrong_length_is_refused(mesh) :
    with pytest.raises(ValueError, match= 'node') :


        normal_field(mesh,np.zeros(mesh.n_nodes + 1))


def test_a_reversing_field_averages_to_near_zero_before_the_magnitude():
    s = uniform_mesh_1d(length  =2e-5, n_nodes =3)
    u =tensor_mesh_2d(uniform_mesh_1d(length=1e-5,n_nodes =2),s)
    psi=  np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])


    r2 =   normal_field( u,   psi  )
    np.testing.assert_allclose(r2[2 :4],0.0,atol=1e-9)
    assert np.all(r2[:2]>0.0),"the boundary rows still see their one edge"
