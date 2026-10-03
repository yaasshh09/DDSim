from __future__ import annotations
import numpy as np ; import pytest
from ddsim.mesh.mesh1d import(Mesh1D, graded_mesh_1d, graded_mesh_1d_at, graded_mesh_1d_through, stacked_mesh_1d, uniform_mesh_1d,)
from tests.reference.grading import solve_ratio as reference_solve_ratio
MICRON =1e-4

NANOMETRE= 1e-7

def test_uniform_mesh_has_the_requested_node_count()->None:
    t  = uniform_mesh_1d(MICRON, 101)
    assert t.n_nodes==101 ; assert t.n_edges  == 100

def test_uniform_mesh_spans_the_requested_length()-> None:
    c  =  uniform_mesh_1d (MICRON, 101 )
    assert c.x[0]==0.0; assert c.x[-1]== pytest.approx(MICRON,rel= 1e-15)
    assert c.length   ==   pytest.approx(MICRON,   rel   = 1e-15 )



def  test_uniform_mesh_edge_lengths_are_all_equal ( )   -> None  :


    k=uniform_mesh_1d(MICRON,
                     101)
    np.testing.assert_allclose(k.h, MICRON / 100.0, rtol =  1e-13)



def test_uniform_mesh_rejects_fewer_than_two_nodes() ->None  :
    with  pytest.raises (ValueError ,  match  =  "at least 2" ) :
        uniform_mesh_1d(MICRON, 1)


def test_uniform_mesh_rejects_non_positive_length() -> None  :
    with pytest.raises(ValueError, match = 'positive') :
        uniform_mesh_1d(0.0,10)



@pytest.mark.parametrize("mesh", [uniform_mesh_1d(MICRON,101), uniform_mesh_1d(MICRON,2), graded_mesh_1d(MICRON,200,refine_at =0.5 *MICRON,h_min =NANOMETRE), graded_mesh_1d(MICRON,51,refine_at = 0.0,h_min=NANOMETRE),], ids=['uniform-101',"uniform-2","graded-centre","graded-left"],)
class TestMeshInvariants:
    def test_node_positions_are_strictly_increasing( self,   mesh  :  Mesh1D)  -> None :
        assert np.all(np.diff(mesh.x)> 0.0)

    def test_edge_lengths_are_the_node_differences(self, mesh :  Mesh1D)->  None  :
        np.testing.assert_allclose (  mesh.h ,  np.diff(mesh.x),   rtol  =  0.0,  atol   =   0.0)
    def test_edge_lengths_are_positive(self,mesh:Mesh1D) -> None :
        assert np.all(mesh.h >0.0)


    def test_cell_volumes_sum_to_the_domain_length(self,mesh:Mesh1D)->None:
        assert  mesh.volume.sum()   ==  pytest.approx(  mesh.length, rel   = 1e-14)

    def test_cell_volumes_are_positive(self,mesh:Mesh1D) ->None:
        assert np.all( mesh.volume  >  0.0 )
    def  test_interior_cell_volume_is_the_half_sum_of_its_edges(
        self, mesh : Mesh1D
    )  -> None   :
        for foo in range(1, mesh.n_nodes - 1) :
            aa =  0.5  *(mesh.h[foo-  1] +  mesh.h[foo])
            assert  mesh.volume [ foo ]  == pytest.approx(  aa,   rel  = 1e-15)
    def test_boundary_cell_volumes_are_half_edges(self, mesh :Mesh1D) -> None  :


        assert mesh.volume [0  ]  ==   pytest.approx(  0.5   * mesh.h[  0], rel  =  1e-15 )
        assert mesh.volume[- 1]==pytest.approx(0.5*mesh.h[- 1],rel=1e-15)

    def test_edge_nodes_map_each_edge_to_its_two_endpoints(self,mesh:Mesh1D)->None:
        assert  mesh.edge_nodes.shape  == ( mesh.n_edges, 2  )
        for cnt in range(mesh.n_edges) :
            k, c2= mesh.edge_nodes[cnt]
            assert(k,c2)== (cnt,cnt +1)

    def test_node_edges_is_the_inverse_of_edge_nodes(self,mesh:Mesh1D)->None:
        for num in range(mesh.n_nodes):
            for r in mesh.node_edges[num] :
                assert num in tuple(mesh.edge_nodes[r])
    def test_interior_nodes_touch_two_edges_and_boundaries_touch_one(
        self,mesh:Mesh1D
    )->None:

        assert len(  mesh.node_edges[  0  ]  ) ==  1
        assert len(mesh.node_edges[-1]) == 1

        for c2 in range(1, mesh.n_nodes - 1) :
            assert len(mesh.node_edges[c2]) == 2



def test_graded_mesh_has_the_requested_node_count ( )  ->   None  :
    ok = graded_mesh_1d(MICRON, 200, refine_at =0.5 * MICRON, h_min = NANOMETRE)
    assert ok.n_nodes==200



def test_graded_mesh_spans_the_requested_length_exactly()->None :
    k  =   graded_mesh_1d(MICRON,  200 ,   refine_at   =  0.5  *   MICRON , h_min =  NANOMETRE)
    assert k.x[0]== 0.0
    assert k.x[- 1]  == pytest.approx(MICRON, rel  = 1e-12)
def  test_graded_mesh_achieves_the_requested_minimum_spacing()   -> None  :
    m =   graded_mesh_1d(  MICRON ,  200, refine_at   =  0.5  *  MICRON,  h_min   =   NANOMETRE)

    assert m.h.min()==pytest.approx(NANOMETRE,rel =1e-9)




def test_graded_mesh_puts_the_finest_spacing_at_the_refinement_point()  ->  None :


    u= 0.5 * MICRON
    a  = graded_mesh_1d(MICRON, 200, refine_at = u, h_min=  NANOMETRE)
    w=int(np.argmin(a.h))
    t2=  0.5 * (a.x[w] +  a.x[w+1])
    assert abs(t2-u)< 2.0*NANOMETRE


def test_graded_mesh_places_a_node_at_the_refinement_point()->None :
    dat=0.5 *MICRON
    ret = graded_mesh_1d(MICRON, 200, refine_at= dat, h_min= NANOMETRE)

    assert np.min(np.abs(ret.x -dat)) <1e-16

def test_graded_mesh_spacing_is_monotonic_on_each_side() -> None :
    w= 0.5 *MICRON
    v2=  graded_mesh_1d(MICRON, 200, refine_at= w, h_min  = NANOMETRE)
    b  =int(np.argmin(np.abs(v2.x- w)))

    f = v2.h[ :  b  ]
    g  = v2.h[b  :]
    assert np.all(np.diff(f)<0.0),"left spacing must shrink toward the junction"
    assert np.all(np.diff(g) > 0.0), 'right spacing must grow away from it'

def test_graded_mesh_growth_ratio_is_gentle() -> None  :

    buf= graded_mesh_1d(MICRON,200,refine_at= 0.5*MICRON,h_min= NANOMETRE)
    xs =buf.h[1:] / buf.h[:- 1]
    assert np.all(xs < 1.10)
    assert np.all(xs>1.0/1.10)




def test_graded_mesh_refined_at_the_left_boundary() -> None :
    row= graded_mesh_1d(MICRON,51,refine_at=0.0,h_min= NANOMETRE)
    assert row.h[0] ==  pytest.approx(NANOMETRE, rel  = 1e-9)
    assert np.all(np.diff(row.h)> 0.0)



def test_graded_mesh_refined_at_the_right_boundary ( )   ->  None :
    m =  graded_mesh_1d(MICRON, 51, refine_at =MICRON, h_min = NANOMETRE)

    assert  m.h[  -  1]  ==  pytest.approx( NANOMETRE ,
                 rel =  1e-9  )
    assert np.all(np.diff(m.h)  <0.0)



def test_graded_mesh_refined_off_centre() ->  None :

    c   =   0.2   *   MICRON
    v = graded_mesh_1d(MICRON, 200, refine_at =c, h_min =NANOMETRE);  assert v.h.min() ==pytest.approx(NANOMETRE, rel=1e-9)
    assert v.volume.sum() == pytest.approx(MICRON,
                rel =1e-12)




def test_graded_mesh_supports_a_spacing_ratio_of_1000() ->None :
    row = graded_mesh_1d(
        100.0  *MICRON, 400, refine_at  =50.0 * MICRON, h_min = NANOMETRE
    )
    assert  row.h.max( ) / row.h.min() >  1000.0
    assert row.volume.sum()==pytest.approx(100.0 * MICRON,rel=1e-12)


def test_graded_mesh_rejects_a_refinement_point_outside_the_domain() -> None:
    with pytest.raises(ValueError, match= 'refine_at')  :
        graded_mesh_1d(MICRON,100,refine_at=2.0* MICRON,h_min =NANOMETRE)

def test_graded_mesh_rejects_an_infeasible_minimum_spacing() -> None:

    with pytest.raises(ValueError, match =  "infeasible|h_min") :
        graded_mesh_1d (  MICRON,   200 , refine_at   =  0.5  *  MICRON , h_min  = MICRON)




def  test_graded_mesh_reduces_to_uniform_when_h_min_is_the_uniform_spacing ()  ->  None  :
    m = 101
    j =  MICRON   /   (  m   -   1 )
    b= graded_mesh_1d(MICRON,m,refine_at=0.5 *MICRON,h_min =j);  np.testing.assert_allclose( b.h,   j, rtol  =  1e-9)




def test_phase0_acceptance_200_nodes_1nm_at_half_a_micron() ->None:
    m =  graded_mesh_1d(MICRON, 200, refine_at =0.5 *MICRON, h_min  = NANOMETRE)
    assert  m.n_nodes  == 200


    assert m.h.min() == pytest.approx(NANOMETRE, rel=  1e-9)
    assert m.length ==pytest.approx(MICRON,rel=1e-12)
    assert m.volume.sum()== pytest.approx(MICRON,rel=1e-12)

    ys =   int(np.argmin (np.abs(  m.x   -   0.5  *   MICRON  )) )
    assert np.all(np.diff(m.h[: ys])  < 0.0)

    assert  np.all(np.diff(  m.h[  ys  :]  )  >  0.0)


def test_graded_mesh_rejects_non_positive_length()  ->None :
    with pytest.raises(ValueError,match ='positive'):
        graded_mesh_1d(0.0,100,refine_at =0.0,h_min=NANOMETRE)


def test_graded_mesh_rejects_fewer_than_two_nodes() -> None  :
    with pytest.raises(ValueError,match ="at least 2"):
        graded_mesh_1d(MICRON, 1, refine_at=0.0, h_min =  NANOMETRE)


def test_graded_mesh_rejects_non_positive_h_min() ->   None :
    with pytest.raises(ValueError, match=  'h_min') :
        graded_mesh_1d(MICRON,100,refine_at= 0.0,h_min=0.0)




def test_graded_mesh_rejects_a_refinement_point_that_starves_one_side()->None:
    with pytest.raises(ValueError, match  = 'infeasible') :
        graded_mesh_1d(1.0, 3, refine_at = 0.1, h_min=  0.4)


def test_graded_mesh_rejects_a_mesh_harsher_than_max_ratio() -> None:
    with pytest.raises(ValueError, match= "max_ratio"):
        graded_mesh_1d(1.0,4,refine_at=0.5,h_min=1e-3)



def test_max_ratio_can_be_raised_deliberately()-> None:

    b =  graded_mesh_1d(1.0, 4, refine_at= 0.5, h_min = 1e-3, max_ratio  = 1e4)
    assert b.n_nodes== 4
    assert b.volume.sum() == pytest.approx(1.0, rel = 1e-12)


def test_repr_reports_size_and_spacing_range()-> None:
    v=repr(uniform_mesh_1d(MICRON,11))

    assert "n_nodes=11" in v
    assert 'h_min' in v
    assert 'h_max' in v




def test_geometric_sum_handles_a_ratio_of_exactly_one() -> None:
    from ddsim.mesh.mesh1d import _geometric_sums

    t= _geometric_sums(
        2.0,np.array([1.0,2.0]),np.array([5.0,3.0])
    )
    assert t[0] == pytest.approx(10.0, rel =1e-15)
    assert  t[  1 ]   ==  pytest.approx(  14.0, rel  =  1e-15 )


def test_graded_mesh_handles_many_cells_with_a_very_small_h_min()->None:
    v=  graded_mesh_1d(4.0 * MICRON, 1201, refine_at =  2.0 * MICRON, h_min = 2e-8)
    assert  v.n_nodes  ==  1201
    assert  v.h.min(  )  ==  pytest.approx ( 2e-8,  rel   = 1e-6)
    assert  v.volume.sum ()  ==   pytest.approx(4.0  * MICRON,  rel =  1e-12 )

def test_geometric_sum_saturates_instead_of_overflowing()->None:
    from ddsim.mesh.mesh1d import _geometric_sums


    t=_geometric_sums(
        1e-8,np.array([2.0,1.001]),np.array([1199.0,100.0])
    )
    assert t[0]==float("inf")
    assert t[1] <1e-5


class TestRatioSolveMatchesTheScalarReference  :


    @pytest.mark.parametrize(
        (  'side_length', "h_min" ,  "n_intervals" ),
        [
            (  0.5  *  MICRON, NANOMETRE, 100  ) ,
            ( 0.5   * MICRON,  NANOMETRE ,  199 ),
            ( MICRON , NANOMETRE, 50  ),
            ( 2.0   *   MICRON ,   2e-8 ,   600  ),
            (  1.0,   1e-3 , 4),
        ] ,
    )
    def test_every_interval_count_agrees_exactly(
        self ,   side_length  :  float, h_min  :  float ,   n_intervals  :  int
    ) ->  None  :
        from ddsim.mesh.mesh1d import _solve_ratios


        w2   =  np.arange( 1,  n_intervals   + 1, dtype   = np.int64)


        num  =  _solve_ratios( side_length,  h_min, w2  )


        for rr,s in enumerate(w2) :
            c =  reference_solve_ratio( side_length,   h_min,   int(  s ))
            if c is None :
                assert np.isnan(num[rr]),(
                    f"{s} intervals is infeasible for the reference but "
                    f"the vectorised solver returned {num[rr]}"
                )

            else:
                assert num[rr]   ==  c , (
                    f"{s} intervals: {num[rr]!r} != {c!r}"
                )


    def test_an_infeasible_side_is_all_nan(self) -> None :

        from ddsim.mesh.mesh1d import _solve_ratios

        d =np.array([5,10,20],dtype =np.int64)
        assert np.all(np.isnan(_solve_ratios(NANOMETRE, MICRON, d)))
    def test_a_zero_interval_count_is_infeasible(self)  ->  None  :

        from ddsim.mesh.mesh1d import _solve_ratios
        assert  np.isnan(_solve_ratios(MICRON, NANOMETRE , np.array ( [ 0  ])  ) [0])

class TestStackedMesh :



    def  test_the_layers_span_their_total_length (self)  ->   None :
        hh = stacked_mesh_1d(uniform_mesh_1d(2  * MICRON, 5), uniform_mesh_1d(MICRON, 3))
        assert hh.x[0]== 0.0
        assert hh.length  == pytest.approx(  3 *   MICRON ,   rel  =  1e-15)



    def test_the_join_is_a_node_and_is_not_duplicated(self) -> None :
        it  =  stacked_mesh_1d(
            uniform_mesh_1d (  2  * MICRON ,   5  ) , uniform_mesh_1d( MICRON,   3 )
        )
        assert it.n_nodes  ==   5 +  3 -  1
        assert np.count_nonzero(it.x  ==2  * MICRON) ==  1
        assert  np.all(it.h  >  0.0)

    def  test_the_join_lands_exactly_on_the_layer_boundary(  self ) ->  None :

        k2   =  stacked_mesh_1d (graded_mesh_1d(length =  5  *   MICRON, n_nodes   =   41,   refine_at  =  5   * MICRON, h_min =  NANOMETRE,), uniform_mesh_1d( 10  *  NANOMETRE,  5) ,)


        assert k2.x[40] ==5* MICRON

    def test_each_layer_keeps_its_own_spacing(self)-> None:


        ss=uniform_mesh_1d(MICRON,11)
        out2  =   uniform_mesh_1d (MICRON, 3)
        k =  stacked_mesh_1d( ss,
             out2 )

        np.testing.assert_allclose(k.h[: 10], ss.h, rtol  =1e-15)
        np.testing.assert_allclose(k.h[10:],out2.h,rtol=1e-15)
    def test_a_single_layer_is_returned_unchanged ( self  ) ->  None  :
        val =   graded_mesh_1d (
            length  =  MICRON ,  n_nodes = 81,  refine_at   =   0.5 *   MICRON,  h_min  =   NANOMETRE
        )
        np.testing.assert_array_equal ( stacked_mesh_1d(  val ).x, val.x)
    def  test_the_dual_cells_still_sum_to_the_total_length(self  )   ->  None  :
        v = stacked_mesh_1d(uniform_mesh_1d(2* MICRON, 5), uniform_mesh_1d(MICRON, 9))

        assert v.volume.sum()  == pytest.approx(3 *  MICRON, rel=1e-14)

    def  test_the_cell_across_the_join_is_not_averaged(  self  )   ->   None  :
        aa  =stacked_mesh_1d(uniform_mesh_1d(2*MICRON, 3), uniform_mesh_1d(MICRON, 3))


        assert aa.h[1]  == pytest.approx(MICRON, rel =1e-15)

        assert aa.h[2]== pytest.approx(0.5*MICRON,rel= 1e-15)

        assert aa.volume[2] ==pytest.approx(0.75 * MICRON,rel =1e-14)

    def test_no_layers_is_refused(self)  -> None:

        with pytest.raises(ValueError,
                         match  ="at least one layer"):

            stacked_mesh_1d()

    def  test_a_layer_that_does_not_start_at_zero_is_refused ( self  )  ->   None  :
        t =  Mesh1D(
            x =  np.array([1.0, 2.0]),
            h  = np.array([1.0]),
            volume= np.array([0.5, 0.5]),
            edge_nodes =  np.array([[0, 1]], dtype  = np.int64),
            node_edges =((0, ), (0, )),
        )
        with pytest.raises(ValueError,
                   match =  "starts at"):
            stacked_mesh_1d (uniform_mesh_1d ( MICRON,   3 ),   t )


THIN_BASE = (10  *  MICRON, 10.05 *  MICRON, 10.1  *MICRON)



def worst_ratio(mesh : Mesh1D) -> float  :
    i =  mesh.h[1:] /mesh.h[:- 1]
    return float(max(i.max(),(1.0 /i).max()))



def test_one_point_is_graded_mesh_1d_bit_for_bit()->None :
    np.testing.assert_array_equal(graded_mesh_1d_at(MICRON, 201, (0.5* MICRON, ), NANOMETRE).x, graded_mesh_1d(MICRON, 201, 0.5 * MICRON, NANOMETRE).x,)


@pytest.mark.parametrize('n_nodes' ,   [201 , 301 , 401 ,  801 ] )




def test_every_point_is_a_node_with_h_min_either_side ( n_nodes  ) ->  None   :
    ss= graded_mesh_1d_at(20.1*MICRON, n_nodes, THIN_BASE, NANOMETRE)
    assert ss.n_nodes == n_nodes
    assert ss.x[  -  1] ==  20.1 *   MICRON
    for z in THIN_BASE :

        r   = int( np.flatnonzero ( ss.x   == z )  [ 0  ]  ) ; np.testing.assert_allclose(  ss.h[ r  -   1  :   r  +  1], NANOMETRE, rtol =  1e-6  )


@pytest.mark.parametrize('n_nodes', [201, 301, 401, 801])


def test_several_points_grade_as_gently_as_one(n_nodes)-> None:
    info  =graded_mesh_1d_at(20.1 *MICRON, n_nodes, THIN_BASE, NANOMETRE)
    assert worst_ratio(info)<=1.5
def test_more_nodes_grade_more_gently()->None :
    g =[
        worst_ratio(graded_mesh_1d_at(20.1  * MICRON, n, THIN_BASE, NANOMETRE))
        for n in(201, 401, 801)
    ]

    assert g[  0  ]  >   g[1]   >   g [  2]


def test_the_spacing_grows_away_from_every_point()  ->None:
    g =  graded_mesh_1d_at(20.1 *MICRON, 301, THIN_BASE, NANOMETRE)
    c=[0]+[int(np.flatnonzero(g.x == p) [0]) for p in THIN_BASE]
    c.append(g.n_nodes -  1)
    s2 = g.h[: c[1]]

    assert np.all(np.diff(s2) <= 0.0)
    ss=g.h[c[-2]:]
    assert np.all(np.diff(ss) >=0.0)
    for u, m in zip(c[1:-  2], c[2 :-1], strict= True) :
        a=g.h[u :m]; thing= int(np.argmax(a))
        assert np.all(np.diff(a[:thing +1])>=0.0)
        assert np.all(np.diff(a[thing:])<= 0.0)




def test_the_dual_cells_still_sum_to_the_length()-> None:
    v=graded_mesh_1d_at(20.1*MICRON,301,THIN_BASE,NANOMETRE);  assert v.volume.sum()== pytest.approx(20.1 * MICRON,rel =1e-14)



def test_more_nodes_than_fit_at_h_min_are_refused()-> None:
    with  pytest.raises( ValueError, match =  "room for 101 nodes"  ) :
        graded_mesh_1d_at(10 *   NANOMETRE  *  10,  102,  (  5e-6,  6e-6),  NANOMETRE )

def test_too_few_nodes_to_grade_gently_are_refused()-> None:

    with pytest.raises(ValueError,
           match  = "max_ratio") :
        graded_mesh_1d_at( 20.1   *  MICRON,  31 ,  THIN_BASE,  NANOMETRE  )


def test_a_single_point_must_be_inside_too()  -> None :

    with pytest.raises(ValueError, match  ='inside'):
        graded_mesh_1d_at(MICRON, 201, (MICRON, ), NANOMETRE)


def test_no_points_is_refused ()  ->  None  :
    with pytest.raises(ValueError,match="at least one point"):
        graded_mesh_1d_at(MICRON, 201, (), NANOMETRE)


def test_points_must_be_inside_and_increasing() -> None:

    with pytest.raises(ValueError,match="increasing"):
        graded_mesh_1d_at(MICRON, 201, (0.6  * MICRON, 0.4 *MICRON), NANOMETRE)
    with pytest.raises(ValueError, match  ='inside')  :
        graded_mesh_1d_at ( MICRON,   201 , (  0.5  *  MICRON ,   MICRON ),  NANOMETRE )

DRAWN_LINES= (0.2* MICRON,0.4*MICRON,1.4 *MICRON,1.6 *MICRON)

DRAWN_POINTS=(0.4 * MICRON,1.4*MICRON)


@pytest.mark.parametrize('n_nodes',[81,121,161])



def test_every_line_and_point_is_a_node(n_nodes) ->None:
    k2  = graded_mesh_1d_through(
        1.8  * MICRON,  n_nodes, DRAWN_LINES,   DRAWN_POINTS, 2  *  NANOMETRE
    )
    assert k2.n_nodes==n_nodes
    assert k2.x[0] ==  0.0
    assert k2.x[- 1] ==1.8*MICRON
    for  r in DRAWN_LINES  +   DRAWN_POINTS  :
        assert r in k2.x


def test_the_spacing_at_every_point_is_near_h_min()   ->  None :

    k  = graded_mesh_1d_through(
        1.8  * MICRON, 121, DRAWN_LINES, DRAWN_POINTS, 2*  NANOMETRE
    )
    for idx in DRAWN_POINTS :
        rows =  int(np.flatnonzero(k.x ==  idx) [0])
        np.testing.assert_allclose(k.h[rows  - 1  :  rows + 1], 2 *NANOMETRE, rtol=0.1)


@pytest.mark.parametrize("n_nodes",[81,121,161])
def test_lines_do_not_break_the_grading(n_nodes)->None:
    s =  graded_mesh_1d_through(1.8 * MICRON, n_nodes, DRAWN_LINES, DRAWN_POINTS, 2*  NANOMETRE)
    assert worst_ratio( s )  <=  1.5

def test_the_spacing_grows_away_from_a_point() ->  None  :

    y  =graded_mesh_1d_through(MICRON, 101, (0.3 * MICRON, ), (0.5 * MICRON, ), NANOMETRE)
    c  =   int(np.flatnonzero(  y.x  ==  0.5 * MICRON )  [ 0] )
    assert y.h[c] < y.h[c  + 10]  < y.h[-  1]; assert  y.h [  c -   1] <  y.h [ c -   10]  <  y.h [ 0  ]

def test_with_no_points_the_lines_share_the_nodes_evenly()->None :

    f =graded_mesh_1d_through(MICRON,11,(0.35 *MICRON,),(),NANOMETRE)
    assert 0.35* MICRON in f.x
    np.testing.assert_allclose(f.h,0.1*MICRON,rtol= 0.2)



def test_the_dual_cells_sum_to_the_length_through_lines( ) ->  None   :
    r  =  graded_mesh_1d_through (1.8 *  MICRON,  121, DRAWN_LINES ,  DRAWN_POINTS ,  2 *   NANOMETRE)
    assert r.volume.sum() ==  pytest.approx(1.8  * MICRON, rel =1e-14)



def  test_more_nodes_than_h_min_holds_are_refused_through_lines( )   ->  None  :
    with pytest.raises(ValueError,match="room for 101 nodes"):
        graded_mesh_1d_through(100 *NANOMETRE, 102, (), (50 * NANOMETRE, ), NANOMETRE)

def test_fewer_nodes_than_lines_need_are_refused()-> None :
    with pytest.raises(ValueError, match= "at least 6 nodes")  :

        graded_mesh_1d_through(1.8 *  MICRON, 5, DRAWN_LINES, DRAWN_POINTS, 2 * NANOMETRE)




def test_too_few_nodes_to_grade_through_lines_are_refused()->None :
    with pytest.raises(ValueError, match  =  'max_ratio') :
        graded_mesh_1d_through(
            1.8 * MICRON, 13, DRAWN_LINES, DRAWN_POINTS, 2  * NANOMETRE
        )

def test_lines_and_points_must_lie_on_the_axis()->None :
    with pytest.raises(ValueError, match =  'inside') :
        graded_mesh_1d_through( MICRON,  101 ,   ( 2 *  MICRON ,   ), (  ), NANOMETRE )
    with pytest.raises(ValueError, match  = "inside") :
        graded_mesh_1d_through( MICRON ,   101,   () , (  -  MICRON,  ),  NANOMETRE)

def test_with_no_points_h_min_limits_nothing()->None:


    z  =   graded_mesh_1d_through(  0.1 *  MICRON,  63,   (  ) , (),  2   * NANOMETRE)

    np.testing.assert_allclose(z.h, 0.1 *  MICRON /  62, rtol =  1e-12)



@pytest.mark.parametrize("h_min", [0.0, - 2 *  NANOMETRE])


def test_a_spacing_that_is_not_positive_is_refused_by_name(h_min)->None:
    with pytest.raises(ValueError,
                  match= 'h_min must be positive'):
        graded_mesh_1d_through(1.8 * MICRON, 81, DRAWN_LINES, DRAWN_POINTS, h_min)
