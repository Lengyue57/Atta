"""Regression checks for bounded native simplification (kernel units: metres)."""
import numpy as np
import pytest

from atta.errors import AttaError
from atta.mesh import Mesh, concat
from atta.primitives import box, sphere, torus
from atta.repair import laplacian_smooth, quadric_decimate


def test_simplify_reaches_target_without_mutating_input():
    source = sphere(0.05, segments=128, rings=64)
    original_verts, original_faces = source.verts.copy(), source.faces.copy()
    source.meta["source"] = "sphere"
    result = quadric_decimate(source, 5000, max_error=0.0005)
    report = result.meta["decimation"]
    assert report["target_reached"]
    assert len(result.faces) <= 5000
    assert result.topology()["closed"]
    assert result.topology()["genus"] == 0
    assert report["surface_error_bound"] <= report["max_error"]
    assert report["evaluations"] <= 12
    assert abs(result.volume() / source.volume() - 1) < 0.01
    assert result.meta["source"] == "sphere"
    np.testing.assert_array_equal(source.verts, original_verts)
    np.testing.assert_array_equal(source.faces, original_faces)
    assert "decimation" not in source.meta


def test_unreachable_target_keeps_error_budget():
    source = sphere(0.05, segments=64, rings=32)
    result = quadric_decimate(source, 4, max_error=1e-6)
    report = result.meta["decimation"]
    assert len(result.faces) > 4
    assert not report["target_reached"]
    assert report["reason"] in ("error_budget_limited", "no_reduction_within_budget")
    assert report.get("surface_error_bound", 0) <= 1e-6
    assert result.topology()["closed"]


def test_disconnected_tangent_solids_stay_separate():
    first = sphere(0.01, segments=64, rings=32)
    second = sphere(0.01, center=(0.02, 0, 0), segments=64, rings=32)
    source = concat([first, second])
    result = quadric_decimate(source, 1800, max_error=0.00015)
    assert len(result.faces) < len(source.faces)
    assert result.topology()["n_components"] == 2
    assert result.topology()["genus"] == source.topology()["genus"]
    assert result.topology()["closed"]


def test_thin_torus_preserves_handle():
    source = torus(0.025, 0.001, seg_major=128, seg_minor=32)
    result = quadric_decimate(source, 1500, max_error=0.00002)
    assert result.topology()["closed"]
    assert result.topology()["genus"] == 1
    assert result.topology()["n_components"] == 1
    assert len(result.faces) < len(source.faces)
    assert result.meta["decimation"]["surface_error_bound"] <= 0.00002


def test_open_input_is_preserved_with_reason():
    source = box([0.01, 0.01, 0.01])
    source = Mesh(source.verts, source.faces[:-1], {})
    result = quadric_decimate(source, 4)
    np.testing.assert_array_equal(source.verts, result.verts)
    np.testing.assert_array_equal(source.faces, result.faces)
    assert result.meta["decimation"]["reason"] == "input_not_closed_manifold"


def test_zero_budget_and_already_small_mesh_are_explicit_noops():
    source = sphere(0.01, segments=32, rings=16)
    result = quadric_decimate(source, 4, max_error=0)
    assert result.meta["decimation"]["reason"] == "zero_error_budget"
    np.testing.assert_array_equal(source.faces, result.faces)
    result = quadric_decimate(source, len(source.faces))
    assert result.meta["decimation"]["reason"] == "already_within_target"
    assert result.meta["decimation"]["target_reached"]


def test_double_precision_at_large_world_coordinates():
    source = sphere(0.001, segments=64, rings=32).transform(
        translate=(1e6, -1e6, 1e6))
    result = quadric_decimate(source, 1200, max_error=1e-5)
    assert len(result.faces) < len(source.faces)
    assert result.topology()["closed"]
    np.testing.assert_allclose(result.bbox()["center"], source.bbox()["center"], atol=1e-5)
    assert result.meta["decimation"]["surface_error_bound"] <= 1e-5


def test_sparse_taubin_matches_neighbour_reference():
    source = sphere(0.01, segments=32, rings=16)
    neighbours = [set() for _ in source.verts]
    for face in source.faces:
        for a, b in zip(face, np.roll(face, -1)):
            neighbours[a].add(b)
            neighbours[b].add(a)
    positions = source.verts.copy()
    for _ in range(3):
        for weight in (0.5, -0.53):
            average = np.array([positions[sorted(group)].mean(axis=0)
                                for group in neighbours])
            positions += weight * (average - positions)
    result = laplacian_smooth(source, iterations=3)
    np.testing.assert_allclose(result.verts, positions, atol=1e-17, rtol=1e-13)
    np.testing.assert_array_equal(result.faces, source.faces)


def test_sparse_smoothing_preserves_all_boundary_vertices():
    source = Mesh(np.array([[0., 0, 0], [1., 0, 0], [1., 1., 0], [0., 1., 0],
                            [.3, .4, .2]]),
                  np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]))
    result = laplacian_smooth(source, iterations=3)
    np.testing.assert_array_equal(result.verts[:4], source.verts[:4])
    assert not np.allclose(result.verts[4], source.verts[4])


@pytest.mark.parametrize("kwargs", [
    {"max_error": -1}, {"max_error": float("nan")},
    {"max_error": float("inf")}, {"max_iterations": 0},
    {"max_iterations": 33},
])
def test_invalid_budgets_are_rejected(kwargs):
    with pytest.raises(AttaError):
        quadric_decimate(box([0.01] * 3), 4, **kwargs)
