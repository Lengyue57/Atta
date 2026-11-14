"""Keep geometric facts stable under translations and orientation changes."""
import numpy as np
import pytest

from atta.mesh import Mesh
from atta.metrology import measure
from atta.primitives import box


@pytest.mark.parametrize("center", [(0, 0, 0), (3, -2, 1), (1e3, -1e3, 1e3),
                                    (1e6, -1e6, 1e6)])
@pytest.mark.parametrize("reverse", [False, True])
def test_solid_volume_and_centroid_are_translation_and_winding_invariant(center, reverse):
    mesh = box([.01, .02, .03], center=center)
    if reverse:
        mesh = Mesh(mesh.verts, mesh.faces[:, ::-1], mesh.meta)
    assert mesh.volume() == pytest.approx(6e-6, rel=5e-8)
    np.testing.assert_allclose(mesh.centroid(), center, atol=2e-10, rtol=0)


def test_metrology_facts_are_stable_at_large_world_coordinates():
    mesh = box([.01, .02, .03], center=(1e6, -1e6, 1e6))
    facts = measure(mesh, wall=False, dihedral=False)
    assert facts["volume_mm3"] == pytest.approx(6000.0, rel=5e-8)
    np.testing.assert_allclose(facts["centroid_mm"], [1e9, -1e9, 1e9], atol=2e-4, rtol=0)


@pytest.mark.parametrize("scale", [1e-9, 1e-6, .01, 10.0])
def test_tiny_solid_metrics_facts_and_gate_preserve_scale(scale):
    from atta.facts import facts
    from atta.verify import verify as verify_mesh
    mesh = box([scale, scale * 2, scale * 3])
    metrics = measure(mesh, curvature=True)
    assert metrics["volume_mm3"] / (6 * scale**3 * 1e9) == pytest.approx(1)
    assert np.asarray(metrics["bbox_mm"]["size"]) / (scale * 1000) == pytest.approx([1, 2, 3])
    assert metrics["dihedral"]["max_deg"] == 90
    assert facts(mesh)["volume_mm3"] > 0
    assert verify_mesh(mesh, level="standard")["verdict"] == "pass"


def test_tiny_and_far_away_hashes_do_not_alias():
    tiny = box([1e-12]*3)
    assert tiny.hash() != box([2e-12]*3).hash()
    assert box([1]*3, center=[1e13, 0, 0]).hash() != box([1]*3, center=[2e13, 0, 0]).hash()


def test_topology_detects_geometric_collapse_with_distinct_indices():
    mesh = Mesh([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]])
    assert mesh.topology()["degenerate_faces"] == 1


def test_zero_volume_has_finite_centroid():
    mesh = Mesh(np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0]]),
                np.array([[0, 1, 2]]))
    assert mesh.volume() == 0
    np.testing.assert_array_equal(mesh.centroid(), [0, 0, 0])
