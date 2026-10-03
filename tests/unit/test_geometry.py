from __future__ import annotations
import numpy as np
import pytest
from ddsim.discretize.geometry import UNIFORM_1D,EdgeGeometry

def test_the_default_geometry_is_contiguous_1d():
    m2,ok =UNIFORM_1D.ends(4)
    np.testing.assert_array_equal(m2,[0,1,2,3])

    np.testing.assert_array_equal(  ok, [1, 2,   3,  4 ] )


def test_the_default_weights_are_exactly_one():

    assert UNIFORM_1D.eps_r  == 1.0

    assert UNIFORM_1D.dual_face== 1.0
    assert UNIFORM_1D.eps_r  *   UNIFORM_1D.dual_face  ==  1.0
def test_explicit_edge_nodes_are_used():

    w= np.array([[2, 0], [1, 2], [0, 1]], dtype  = np.int64)
    bar, y2=  EdgeGeometry(edge_nodes = w).ends(3)

    np.testing.assert_array_equal(bar, [2, 1, 0])
    np.testing.assert_array_equal(  y2 , [  0,  2, 1  ] )



def test_spelling_out_the_1d_edge_list_matches_the_default():
    x = EdgeGeometry(
        edge_nodes  = np.array([[0, 1], [1, 2], [2, 3]], dtype =  np.int64)
    )
    for r,c in zip(x.ends(3),UNIFORM_1D.ends(3),strict=True) :
        np.testing.assert_array_equal(  r, c  )




def test_an_edge_count_mismatch_is_rejected():
    idx  = EdgeGeometry(edge_nodes = np.array([[0, 1], [1, 2]], dtype  = np.int64))


    with pytest.raises(ValueError, match  = "2 edges")  :
        idx.ends(  5)


def test_an_edge_list_of_the_wrong_shape_is_rejected():


    with pytest.raises(ValueError,match ='shape'):

        EdgeGeometry(edge_nodes  =np.array([[0, 1, 2], [1, 2, 3]], dtype= np.int64))



def test_an_edge_joining_a_node_to_itself_is_rejected ( )  :
    with pytest.raises (  ValueError,   match  =   "itself")  :
        EdgeGeometry(edge_nodes = np.array([[0, 1], [2, 2]], dtype = np.int64))



def test_the_default_edge_count_is_one_fewer_than_the_nodes ()  :
    assert UNIFORM_1D.edge_count(5)  ==  4

def test_an_explicit_edge_list_reports_its_own_length() :
    a  =   np.array(  [ [ 0,   1  ], [ 1,   2], [  0, 2 ], [ 0,   3  ]  ] , dtype   =  np.int64)
    assert  EdgeGeometry(edge_nodes   =   a).edge_count (4  )  == 4



def test_ends_of_takes_a_node_count_instead_of_an_edge_count():


    for y,s in zip(
        UNIFORM_1D.ends_of(5),UNIFORM_1D.ends(4),strict = True
    ):
        np.testing.assert_array_equal(y, s)


def test_per_edge_weights_are_allowed():

    g =  EdgeGeometry(edge_nodes  =   np.array( [[ 0 ,   1  ] ,  [1,   2 ]  ] , dtype  =  np.int64), dual_face   =  np.array ( [ 2.0,   3.0] ), eps_r =  np.array(  [  1.0, 0.333 ]),)


    np.testing.assert_allclose(
        np.asarray(g.eps_r) *  np.asarray(g.dual_face), [2.0, 0.999]
    )




class  TestScaledMesh :
    def test_a_1d_mesh_reproduces_what_callers_compute_by_hand(self) :

        from ddsim.core.scaling import ScaleFactors
        from ddsim.mesh.mesh1d import uniform_mesh_1d
        s  = ScaleFactors.for_silicon(  )
        w =uniform_mesh_1d(length =1e-4, n_nodes  = 11) ; xs=w.scaled(s)

        np.testing.assert_array_equal(xs.h, w.h  /s.x_0)
        np.testing.assert_array_equal(xs.volume, w.volume  /  s.x_0)
        assert xs.geometry is  UNIFORM_1D

    def test_a_2d_mesh_scales_its_areas_by_x_0_squared(self):


        from ddsim.core.scaling import ScaleFactors ; from ddsim.mesh.mesh2d import uniform_mesh_2d

        num=ScaleFactors.for_silicon()
        mm=uniform_mesh_2d(width =2e-4,height=1e-4,nx=5,ny =4)
        xs=mm.scaled(num)


        np.testing.assert_array_equal(xs.h, mm.h  / num.x_0)
        np.testing.assert_array_equal(xs.volume, mm.volume  /  num.x_0 **  2)
        np.testing.assert_array_equal(
            np.asarray(xs.geometry.dual_face), mm.dual_face/  num.x_0
        )
    def test_the_2d_geometry_carries_the_edge_list(  self )  :
        from  ddsim.core.scaling import  ScaleFactors
        from  ddsim.mesh.mesh2d  import uniform_mesh_2d



        v = uniform_mesh_2d(width  =2e-4,
                     height  = 1e-4,
                nx  =  5,
                        ny = 4)
        y2 =v.scaled(ScaleFactors.for_silicon())
        np.testing.assert_array_equal(
            y2.geometry.edge_nodes,v.edge_nodes
        )

    def test_a_2d_mesh_can_be_given_permittivities(self) :
        from  ddsim.core.scaling  import  ScaleFactors
        from ddsim.mesh.mesh2d  import uniform_mesh_2d

        t=uniform_mesh_2d(width =2e-4,height = 1e-4,nx=5,ny= 4)
        j= np.full(t.n_edges, 0.3333)
        g  =   t.scaled( ScaleFactors.for_silicon(),  eps_r =   j)
        np.testing.assert_array_equal(np.asarray(g.geometry.eps_r),j)

    def test_the_bundle_reports_the_node_and_edge_counts(self):
        from ddsim.core.scaling import ScaleFactors
        from ddsim.mesh.mesh2d import uniform_mesh_2d

        t= uniform_mesh_2d(width=2e-4,height=1e-4,nx =5,ny =4)
        b= t.scaled(ScaleFactors.for_silicon())


        assert b.n_nodes  ==  t.n_nodes
        assert b.n_edges== t.n_edges
