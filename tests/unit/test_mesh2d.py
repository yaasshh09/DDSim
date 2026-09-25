'''Structured tensor product 2D mesh, box integration.

phases/PHASE-4.md says to start structured rather than Delaunay, and
docs/02-numerics.md says why: a rectangle cannot produce an obtuse triangle, so
the negative dual area failure mode that docs/05-pitfalls.md warns about cannot
arise at all. Unstructured meshing waits until geometry demands it.

The mesh is built as the tensor product of two Mesh1D axes, which is not a
shortcut. It means the grading logic and the dual grid construction are
inherited rather than written a second time, and the dual cell sum invariant
comes with them.

That invariant is a tolerance rather than an equality, which I only found by
checking. See test_the_dual_cells_sum_to_the_domain_area below.
'''
from __future__ import annotations


import  numpy as  np;  import pytest
from ddsim.mesh.mesh1d import graded_mesh_1d,uniform_mesh_1d

from ddsim.mesh.mesh2d import normal_field, tensor_mesh_2d, uniform_mesh_2d


@pytest.fixture



def mesh()  :
    """A 5 by 4 node rectangle, deliberately not square."""
    return uniform_mesh_2d(width=2e-4,height= 1e-4,nx= 5,ny =4)

def test_the_node_count_is_the_product_of_the_axes(mesh) :

    assert  mesh.nx ==  5
    assert mesh.ny  ==  4

    assert mesh.n_nodes ==20


def test_the_edge_count_is_both_families_added(mesh ) :
    """(nx-1)*ny horizontal, nx*(ny-1) vertical. No diagonals."""
    assert mesh.n_edges==(5- 1) * 4+5*(4-1)
    assert mesh.n_edges ==mesh.h.size ==mesh.dual_face.size



def test_the_dual_cells_sum_to_the_domain_area(mesh):
    """The 2D version of the invariant Mesh1D holds itself to.

    Not asserted as exactly zero, and it was worth checking rather than
    assuming. Summing the outer product visits the terms in a different order
    from multiplying the two 1D sums, so the last bit can move: measured at
    3.3e-16 relative on this fixture. The 1D sum is not unconditionally exact
    either, which is easy to miss because it happens to be exact on the Phase 0
    acceptance case. It is exact for a 1 um domain on 200 nodes and off by
    1.4e-16 for the same domain on 5. The tolerance here matches the one the
    1D tests already use.
    """
    assert mesh.volume.sum()==pytest.approx(2e-4 *1e-4,rel=1e-14)




def  test_every_edge_joins_two_geometrically_adjacent_nodes ( mesh )  :
    """h is the actual distance between the two nodes of the edge.

    This is the test that catches a wrong node numbering convention. If the
    row major index arithmetic is off, the edge list still looks plausible and
    the counts still come out right, but some edge will join two nodes that
    are not neighbours and its length will not match the distance between them.
    """
    Left, rig =   mesh.edge_nodes[ :,  0 ],   mesh.edge_nodes[ :,  1 ]

    next= mesh.node_x[rig] -mesh.node_x[Left]
    idx2 =  mesh.node_y[rig] -mesh.node_y[Left]

    np.testing.assert_allclose(np.hypot(next,idx2),mesh.h,rtol=1e-14)

    assert  np.all( (  next  ==  0.0 )  ^   (  idx2  == 0.0  ) )
def test_the_dual_faces_are_all_strictly_positive(mesh) :
    """A negative dual face is the 2D failure mode. Structured cannot have one."""
    assert np.all(mesh.dual_face>0.0)
    assert np.all(mesh.volume>0.0)
    assert  np.all ( mesh.h  > 0.0 )


def test_no_edge_joins_a_node_to_itself_and_all_are_in_range(mesh):

    assert np.all(mesh.edge_nodes  >=0); assert np.all(mesh.edge_nodes < mesh.n_nodes)
    assert np.all(mesh.edge_nodes[:, 0] != mesh.edge_nodes[:, 1])


def test_every_node_is_touched_by_at_least_two_edges(mesh):
    """A corner has two, an edge node three, an interior node four."""
    touhces=np.bincount(mesh.edge_nodes.ravel(),minlength= mesh.n_nodes)
    assert touhces.min() ==2
    assert touhces.max()  == 4


def test_a_horizontal_edge_carries_the_vertical_dual_extent(mesh):
    """The face a horizontal flux crosses is vertical, and vice versa.

    Getting this pair swapped is the classic box integration slip. On a
    non-square mesh it changes the answer; on a square one it does not, which
    is exactly why the fixture is 5 by 4 and the domain is 2:1.
    """
    stuff2 =uniform_mesh_1d(length =2e-4,n_nodes= 5)
    ya=  uniform_mesh_1d(length =  1e-4, n_nodes  =  4)

    n_horziontal=(5 - 1)*4
    hor  =mesh.dual_face[:n_horziontal]

    veritcal_face=  mesh.dual_face[n_horziontal :]
    assert set(np.round(hor, 18))<=set(np.round(ya.volume, 18))
    assert set(np.round(veritcal_face,18)) <= set(np.round(stuff2.volume,18))

def test_the_x_structure_matches_the_1d_mesh_it_was_built_from (  ) :
    """A row of the 2D mesh has to be the 1D mesh, node for node."""

    x_ais =  graded_mesh_1d(length= 1e-4, n_nodes = 41, refine_at  =  0.5e-4, h_min = 1e-7)
    yAxis= uniform_mesh_1d(length = 1e-5, n_nodes =  3)
    zz= tensor_mesh_2d(x_ais,yAxis)

    np.testing.assert_array_equal(zz.node_x[:  zz.nx], x_ais.x)
    np.testing.assert_array_equal(zz.node_x[zz.nx : 2*zz.nx],x_ais.x)




def test_a_graded_axis_survives_the_tensor_product() :
    """Grading is inherited, not reimplemented."""
    XAxis =  graded_mesh_1d(length = 1e-4, n_nodes  =  41, refine_at=0.5e-4, h_min = 1e-7)
    abs= tensor_mesh_2d(XAxis, uniform_mesh_1d(length =1e-5, n_nodes =3))


    assert abs.volume.sum()== pytest.approx(1e-4 * 1e-5, rel  =1e-15)
    assert abs.h.min()  == pytest.approx(  XAxis.h.min( ) , rel =  1e-15)

def  test_a_mesh_needs_at_least_two_nodes_on_each_axis(  )   :
    """A one node axis has no edges and is not a 2D mesh."""
    with pytest.raises (  ValueError, match  = 'at least 2 nodes' )  :
        uniform_mesh_2d(width = 1e-4, height  =  1e-4, nx  =  1, ny =  4)
    with pytest.raises(ValueError, match ="at least 2 nodes")  :
        uniform_mesh_2d(width   =  1e-4, height =  1e-4 , nx  =  4,  ny =   1)
def test_node_at_agrees_with_the_row_major_numbering(mesh) :
    """The accessor and the actual node positions have to tell one story."""
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
    """edge_geometry packages the edge list and the dual faces together."""
    geo  =  mesh.edge_geometry(  )


    tmp2,rig=geo.ends(mesh.n_edges)
    np.testing.assert_array_equal(tmp2, mesh.edge_nodes[:, 0])

    np.testing.assert_array_equal(rig,mesh.edge_nodes[:,1])
    np.testing.assert_array_equal (  np.asarray (geo.dual_face  ),  mesh.dual_face )


def test_a_uniform_vertical_gradient_is_recovered_exactly(mesh):
    '''The straightest possible check. psi = g*y everywhere makes dpsi/dy the
    constant g at every node including the two boundary rows, so anything that
    misreads a spacing or drops an edge shows up as a number that is not g.'''
    grradient=3.7e4; psi= grradient*mesh.node_y

    EE =  normal_field(  mesh , psi )


    np.testing.assert_allclose(EE,
              grradient,
       rtol=1e-12)




def test_a_purely_horizontal_potential_has_no_normal_field(mesh) :
    """psi varying only along x has no y derivative anywhere. This is the test
    that fails if the two edge families get swapped, and swapping them is the
    single most likely error here: both are arrays over edges of the same mesh
    and neither carries a label saying which it is."""
    psi =  5.0  *  mesh.node_x

    np.testing.assert_allclose(normal_field(mesh, psi), 0.0, atol = 1e-9)


def test_the_normal_field_is_a_magnitude(mesh) :

    """Lombardi takes abs(E_perp) and refuses a signed one, so the sign has to
    be gone before it gets there. A gate above and a gate below the same
    channel produce the same scattering."""
    psi = 2.5e4 * mesh.node_y

    any=normal_field(mesh,psi)
    Down =   normal_field(mesh,  - psi)
    assert np.all ( any  >  0.0)
    np.testing.assert_allclose(any, Down, rtol =  1e-14)

def test_an_interior_node_averages_the_edges_either_side ( )   :
    """Where the two vertical edges at a node disagree, the node takes their
    mean. A graded mesh makes them disagree on purpose: the same potential
    difference across a shorter edge is a larger field.

    Worked by hand rather than against the function, so the test knows the
    answer independently.
    """

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
    """The top and bottom rows have one vertical neighbour, not two. Dividing
    by a hardcoded two there would halve the field at exactly the row the gate
    sits on, which is the row the whole model is about."""
    psi=  1.1e4 * mesh.node_y
    EE=normal_field(mesh,psi) ; list   =   EE[ :  mesh.nx ]
    topp = EE[- mesh.nx  :]

    np.testing.assert_allclose(list, 1.1e4, rtol  =1e-12);np.testing.assert_allclose(topp, 1.1e4, rtol=  1e-12)
def test_the_normal_field_is_one_value_per_node(mesh) :
    psi =np.zeros(mesh.n_nodes)
    assert normal_field(mesh, psi).shape == (mesh.n_nodes, )



def test_a_potential_of_the_wrong_length_is_refused(mesh) :
    """One value per node, and the failure is otherwise a broadcast that
    silently produces the wrong shape rather than an error."""
    with pytest.raises(ValueError, match= 'node') :


        normal_field(mesh,np.zeros(mesh.n_nodes + 1))


def test_a_reversing_field_averages_to_near_zero_before_the_magnitude():
    """The magnitude is taken after the averaging, not before, and this is the
    test that tells the two apart.

    A potential with a minimum on the middle row has a field pointing one way
    below it and the other way above. The field at the node itself is zero,
    and that is what a centred difference says. Taking abs() of each edge
    first and then averaging would report the full size of both instead, and
    would put a large normal field at exactly the places a device has a
    potential well.
    """
    yAxis = uniform_mesh_1d(length  =2e-5, n_nodes =3)
    mseh =tensor_mesh_2d(uniform_mesh_1d(length=1e-5,n_nodes =2),yAxis)
    psi=  np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])


    w =   normal_field( mseh,   psi  )
    np.testing.assert_allclose(w[2 :4],0.0,atol=1e-9)
    assert np.all(w[:2]>0.0),"the boundary rows still see their one edge"
