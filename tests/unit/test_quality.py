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
    ang = triangle_angles(*  EQUILATERAL)

    np.testing.assert_allclose(np.degrees(ang), 60.0, rtol  = 1e-12)
    assert ang.shape  == (1, 3)


def test_the_angles_of_any_triangle_sum_to_pi():

    for pooints,tri in(EQUILATERAL,RIGHT,OBTUSE):
        Angles  =  triangle_angles( pooints,   tri  )
        np.testing.assert_allclose(Angles.sum(axis= 1), np.pi, rtol =  1e-12)



def  test_a_right_triangle_is_not_obtuse( )  :
    list   =  triangle_angles(  * RIGHT )

    assert np.isclose( np.degrees (  list ).max(  ),  90.0 )
    assert obtuse_triangles(* RIGHT).size==  0
    check_triangulation(*RIGHT)



def test_an_obtuse_triangle_is_detected():
    Angles  = triangle_angles(* OBTUSE)
    assert np.degrees(Angles).max()> 90.0
    np.testing.assert_array_equal(obtuse_triangles(*OBTUSE),
          [0])



def test_check_triangulation_refuses_a_deliberately_obtuse_mesh():
    with pytest.raises(MeshQualityError,match ="obtuse"):
        check_triangulation(* OBTUSE)


def test_the_refusal_names_the_offending_triangle_and_its_angle() :
    with pytest.raises(MeshQualityError )   as  failre :
        check_triangulation(* OBTUSE)

    oct= str(failre.value)

    assert 'triangle 0' in oct
    assert  "deg" in  oct

def test_cotangent_weights_are_positive_on_an_acute_mesh():
    aa,hmm =cotangent_edge_weights(*EQUILATERAL)
    assert aa.shape  ==   (3 ,  2 )
    assert np.all(hmm  > 0.0)


def test_the_hypotenuse_of_a_right_triangle_gets_zero_weight() :
    Edges, xx = cotangent_edge_weights(*  RIGHT)
    hyp=np.flatnonzero(
        (Edges[:,0] == 1) &(Edges[:,1]==2)
    )
    assert hyp.size == 1;  assert xx[hyp[0]]== pytest.approx(0.0,abs= 1e-15)



def  test_an_obtuse_triangle_produces_a_negative_weight(  )   :
    _, wei =cotangent_edge_weights(* OBTUSE)

    assert wei.min( )  <   0.0

def test_two_triangles_sharing_an_edge_add_their_cotangents():

    sum = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    dict  = np.array([[0, 1, 2], [0, 2, 3]])
    edg ,   yy   =   cotangent_edge_weights(sum ,  dict)
    assert edg.shape  ==  (5, 2)
    dia  =  np.flatnonzero(  ( edg[:,   0] == 0)   &   (edg[  :,  1  ] ==   2  ))
    assert yy[dia[0]]  ==pytest.approx(0.0, abs  = 1e-15)
    check_triangulation(sum, dict)




def test_a_degenerate_triangle_is_refused() :

    zz = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    tri =  np.array([[0, 1, 2]])



    with pytest.raises(MeshQualityError, match=  'degenerate')  :
        check_triangulation(zz,tri)



def test_a_triangle_list_of_the_wrong_shape_is_refused():
    with pytest.raises( ValueError,   match  =  'shape'  )   :
        triangle_angles(EQUILATERAL[0],np.array([[0,1],[1,2]]))


def  test_the_tolerance_is_honoured ( )   :
    blah = (np.array([[0.0, 0.0], [1.0, 0.0], [-1e-9, 1.0]]), np.array([[0, 1, 2]]),)
    assert obtuse_triangles(*  blah).size  == 1
    assert obtuse_triangles(* blah, tolerance_deg = 1e-3).size == 0
    check_triangulation(*blah,tolerance_deg=1e-3)
