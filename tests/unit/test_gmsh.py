from __future__ import annotations
import numpy as np
import pytest
from ddsim.mesh.gmsh import read_msh, write_msh
from ddsim.mesh.mesh1d import graded_mesh_1d, uniform_mesh_1d
from ddsim.mesh.mesh2d import tensor_mesh_2d
from ddsim.mesh.trimesh import triangulated

SQUARE = """$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
3
1 1 "bottom"
1 2 "top"
2 3 "silicon"
$EndPhysicalNames
$Entities
4 4 1 0
1 0 0 0 0
2 1 0 0 0
3 1 1 0 0
4 0 1 0 0
1 0 0 0 1 0 0 1 1 2 1 -2
2 1 0 0 1 1 0 0 2 2 -3
3 0 1 0 1 1 0 1 2 2 3 -4
4 0 0 0 0 1 0 0 2 4 -1
1 0 0 0 1 1 0 1 3 4 1 2 3 4
$EndEntities
$Nodes
5 4 1 4
0 1 0 1
1
0 0 0
0 2 0 1
2
1 0 0
0 3 0 1
3
1 1 0
0 4 0 1
4
0 1 0
2 1 0 0
$EndNodes
$Elements
3 4 1 4
1 1 1 1
1 1 2
1 3 1 1
2 3 4
2 1 2 2
3 1 2 3
4 1 3 4
$EndElements
"""


def test_a_gmsh_4_1_file_reads_its_nodes_triangles_and_groups(tmp_path) -> None:
    p = tmp_path / "square.msh"
    p.write_text(SQUARE, encoding="utf-8")
    g = read_msh(p)
    np.testing.assert_array_equal(g.points, [[0, 0], [1, 0], [1, 1], [0, 1]])
    np.testing.assert_array_equal(g.triangles, [[0, 1, 2], [0, 2, 3]])
    assert g.region_of_triangles == ("silicon", "silicon")
    assert g.contact_nodes("bottom") == (0, 1)
    assert g.contact_nodes("top") == (2, 3)
    assert g.mesh().volume.sum() == pytest.approx(1.0, rel=1e-14)


def test_an_unknown_contact_is_refused_with_the_names_it_has(tmp_path) -> None:
    p = tmp_path / "square.msh"
    p.write_text(SQUARE, encoding="utf-8")
    with pytest.raises(KeyError, match="bottom"):
        read_msh(p).contact_nodes("gate")


def test_a_written_mesh_reads_back_exactly(tmp_path) -> None:
    m = triangulated(tensor_mesh_2d(graded_mesh_1d(length=2e-4, n_nodes=21, refine_at=0.7e-4, h_min=3e-6), uniform_mesh_1d(length=1e-4, n_nodes=4)))
    pts = np.column_stack([m.node_x, m.node_y])
    left = [(i, j) for i, j in m.edge_nodes.tolist() if pts[i, 0] == 0 and pts[j, 0] == 0]
    p = tmp_path / "out.msh"
    write_msh(p, pts, m.triangles, {"bulk": np.arange(m.triangles.shape[0])}, {"left": np.array(left)})
    g = read_msh(p)
    np.testing.assert_array_equal(g.points, pts)
    np.testing.assert_array_equal(g.triangles, m.triangles)
    assert g.contact_nodes("left") == (0, 21, 42, 63)
    assert set(g.region_of_triangles) == {"bulk"}


def test_a_file_that_is_not_4_1_is_refused(tmp_path) -> None:
    p = tmp_path / "old.msh"
    p.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", encoding="utf-8")
    with pytest.raises(ValueError, match="4.1"):
        read_msh(p)
