from __future__ import annotations
import numpy as np
import pytest
from ddsim.mesh.quality import(MeshQualityError, check_triangulation , cotangent_edge_weights, obtuse_triangles, triangle_angles,)
EQUILATERAL =(np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/  2]]), np.array([[0, 1, 2]]),)


RIGHT  =(
    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    np.array([[0, 1, 2]]),
)

OBTUSE = (
    np.array([[0.0,0.0],[1.0,0.0],[3.0,0.5]]),
    np.array([[0,1,2]]),
)



def test_an_equilateral_triangle_has_three_sixty_degree_angles():
    d = triangle_angles(*  EQUILATERAL)

    np.testing.assert_allclose(np.degrees(d), 60.0, rtol  = 1e-12)
    assert d.shape  == (1, 3)


def test_the_angles_of_any_triangle_sum_to_pi():

    for it,u in(EQUILATERAL,RIGHT,OBTUSE):
        r  =  triangle_angles( it,   u  )
        np.testing.assert_allclose(r.sum(axis= 1), np.pi, rtol =  1e-12)



def  test_a_right_triangle_is_not_obtuse( )  :
    e   =  triangle_angles(  * RIGHT )

    assert np.isclose( np.degrees (  e ).max(  ),  90.0 )
    assert obtuse_triangles(* RIGHT).size==  0
    check_triangulation(*RIGHT)



def test_an_obtuse_triangle_is_detected():
    t  = triangle_angles(* OBTUSE)
    assert np.degrees(t).max()> 90.0
    np.testing.assert_array_equal(obtuse_triangles(*OBTUSE),
          [0])



def test_check_triangulation_refuses_a_deliberately_obtuse_mesh():
    with pytest.raises(MeshQualityError,match ="obtuse"):
        check_triangulation(* OBTUSE)


def test_the_refusal_names_the_offending_triangle_and_its_angle() :
    with pytest.raises(MeshQualityError )   as  t :
        check_triangulation(* OBTUSE)

    z= str(t.value)

    assert 'triangle 0' in z
    assert  "deg" in  z

def test_cotangent_weights_are_positive_on_an_acute_mesh():
    w,i =cotangent_edge_weights(*EQUILATERAL)
    assert w.shape  ==   (3 ,  2 )
    assert np.all(i  > 0.0)


def test_the_hypotenuse_of_a_right_triangle_gets_zero_weight() :
    m2, z2 = cotangent_edge_weights(*  RIGHT)
    k=np.flatnonzero(
        (m2[:,0] == 1) &(m2[:,1]==2)
    )
    assert k.size == 1;  assert z2[k[0]]== pytest.approx(0.0,abs= 1e-15)



def  test_an_obtuse_triangle_produces_a_negative_weight(  )   :
    _, k =cotangent_edge_weights(* OBTUSE)

    assert k.min( )  <   0.0

def test_two_triangles_sharing_an_edge_add_their_cotangents():

    kk = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    c  = np.array([[0, 1, 2], [0, 2, 3]])
    ret ,   w   =   cotangent_edge_weights(kk ,  c)
    assert ret.shape  ==  (5, 2)
    a  =  np.flatnonzero(  ( ret[:,   0] == 0)   &   (ret[  :,  1  ] ==   2  ))
    assert w[a[0]]  ==pytest.approx(0.0, abs  = 1e-15)
    check_triangulation(kk, c)




def test_a_degenerate_triangle_is_refused() :

    tmp = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    z2 =  np.array([[0, 1, 2]])



    with pytest.raises(MeshQualityError, match=  'degenerate')  :
        check_triangulation(tmp,z2)



def test_a_triangle_list_of_the_wrong_shape_is_refused():
    with pytest.raises( ValueError,   match  =  'shape'  )   :
        triangle_angles(EQUILATERAL[0],np.array([[0,1],[1,2]]))


def  test_the_tolerance_is_honoured ( )   :
    idx = (np.array([[0.0, 0.0], [1.0, 0.0], [-1e-9, 1.0]]), np.array([[0, 1, 2]]),)
    assert obtuse_triangles(*  idx).size  == 1
    assert obtuse_triangles(* idx, tolerance_deg = 1e-3).size == 0
    check_triangulation(*idx,tolerance_deg=1e-3)
