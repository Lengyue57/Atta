"""Compositional geometry, units, bounded work and deterministic sampling."""
import math

import numpy as np
import pytest

from atta import fieldtree as F
from atta.errors import AttaError
from atta.features import evaluate
from atta.mesh import Mesh
from atta.surfaces import thicken


def ball(radius=10, center=None):
    return {"op": "sphere", "radius": radius, "center": center or [0, 0, 0]}


def graph(tree, **kwargs):
    return [{"id": "shape", "op": "field_tree", "args": {
        "tree": tree, "bounds": [[-15]*3, [15]*3], "spacing": 1, **kwargs}}]


def test_field_tree_sphere_units_and_analytic_volume():
    mesh = evaluate(graph(ball()))
    assert mesh.topology()["closed"] and mesh.topology()["genus"] == 0
    assert mesh.volume() == pytest.approx(4/3 * math.pi * .01**3, rel=.012)
    assert mesh.bbox()["size"] == pytest.approx([.02]*3, abs=.0001)
    assert mesh.meta["field"]["distance_exact"] is False
    assert not mesh.meta["sampling"]["coarsened"]


def test_csg_semantics_and_noncommutative_difference():
    a, b = ball(2), ball(1, [1, 0, 0])
    points = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]])
    x, y = F.compile_tree(a)(points), F.compile_tree(b)(points)
    assert F.compile_tree({"op": "difference", "children": [a, b]})(points) == pytest.approx(np.maximum(x, -y))
    assert F.compile_tree({"op": "intersection", "children": [a, b]})(points) == pytest.approx(np.maximum(x, y))
    assert F.compile_tree({"op": "union", "children": [a, b]})(points) == pytest.approx(np.minimum(x, y))


def test_local_transform_rotates_and_scales_distance():
    tree = {"op": "transform", "translate": [3, 4, 5], "rotate_deg": [0, 0, 90], "scale": 2,
            "child": {"op": "box", "size": [4, 2, 2]}}
    values = F.compile_tree(tree)(np.array([[3, 8, 5], [5, 4, 5], [3, 9, 5], [3, 4, 5]]))
    assert values == pytest.approx([0, 0, 1, -2], abs=1e-14)


def test_tree_order_preserves_noise_inside_twist_and_chunk_independence():
    noisy = {"op": "noise", "amplitude": .7, "wavelength": 4, "seed": 91, "octaves": 3,
             "child": {"op": "box", "size": [12, 8, 18], "round": 2}}
    tree = {"op": "twist", "angle_deg": 120, "length": 18, "child": noisy}
    a = evaluate(graph(tree, chunk_points=127))
    b = evaluate(graph(tree, chunk_points=32768))
    smooth = evaluate(graph({**tree, "child": noisy["child"]}))
    assert a.hash() == b.hash()
    assert a.hash() != smooth.hash()
    assert a.topology()["closed"]


def test_tpms_intersection_is_closed_and_holey():
    tree = {"op": "intersection", "children": [ball(), {"op": "gyroid", "period": 8, "band": .4}]}
    mesh = evaluate(graph(tree, spacing=.6))
    topo = mesh.topology()
    assert topo["closed"] and topo["genus"] > 0
    assert mesh.volume() < 4/3 * math.pi * .01**3


def test_shell_cavity_has_two_closed_components_and_correct_volume():
    mesh = evaluate(graph({"op": "shell", "thickness": 3, "child": ball(8)}, spacing=.5))
    assert mesh.topology()["closed"]
    exact = 4/3 * math.pi * ((.0095)**3 - (.0065)**3)
    assert mesh.volume() == pytest.approx(exact, rel=.02)


@pytest.mark.parametrize("scale", [1e-8, 1e-5, 1., 1e5])
def test_tilted_patch_thickening_has_scale_invariant_volume(scale):
    verts = scale * np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=float)
    shell = thicken(Mesh(verts, [[0, 1, 2], [0, 2, 3]]), scale / 10)
    assert shell.topology()["closed"]
    assert shell.volume() / scale**3 == pytest.approx(.1)
    assert np.asarray(shell.bbox()["size"]) / scale == pytest.approx([.1, 1, 1])


def test_sample_and_work_budget_fail_before_grid_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("allocation occurred before budget validation")
    monkeypatch.setattr(F.np, "empty", forbidden)
    with pytest.raises(AttaError, match="budget"):
        F.mesh_field(F.compile_tree(ball()), [[-20]*3, [20]*3], .01)
    costly = {"op": "noise", "child": ball(), "amplitude": 1, "wavelength": 1, "octaves": 8}
    with pytest.raises(AttaError, match="budget"):
        F.mesh_field(F.compile_tree(costly), [[-20]*3, [20]*3], 1, max_evaluations=1000)


def test_coarsening_explicit_and_min_feature_rejects_underresolution():
    mesh = evaluate(graph(ball(), spacing=.1, max_samples=20000, budget_policy="coarsen"))
    sampling = mesh.meta["sampling"]
    assert sampling["samples"] <= 20000 and sampling["coarsened"]
    with pytest.raises(AttaError) as e:
        evaluate(graph(ball(), spacing=.1, max_samples=20000, budget_policy="coarsen", min_feature=.5))
    assert e.value.code == "field_under_resolved"


def test_field_output_budget_and_open_boundary_are_explicit_errors():
    with pytest.raises(AttaError) as e:
        evaluate(graph(ball(), max_triangles=100))
    assert e.value.code == "sdf_too_dense"
    with pytest.raises(AttaError) as e:
        evaluate(graph(ball(100)))
    assert e.value.code == "sdf_open"


@pytest.mark.parametrize("tree", [
    {"op": "sphere", "radius": -1}, {"op": "sphere", "radius": True},
    {"op": "sphere", "radius": "2"}, {"op": "sphere", "radius": float("nan")},
    {"op": "sphere", "radius": 1, "unknown": 3},
    {"op": "noise", "child": ball(), "amplitude": 1, "wavelength": 0},
    {"op": "transform", "scale": [1, 2, 3], "child": ball()},
    {"op": "shell", "thickness": 1}, {"op": "union", "children": []},
])
def test_bad_tree_parameters_fail_cleanly(tree):
    with pytest.raises(AttaError) as e:
        F.compile_tree(tree)
    assert e.value.path and e.value.fix


def test_cyclic_or_deep_python_tree_is_bounded():
    tree = {"op": "shell", "thickness": 1}
    tree["child"] = tree
    with pytest.raises(AttaError) as e:
        F.compile_tree(tree)
    assert e.value.code == "field_budget"


def test_rotated_field_hash_is_independent_of_single_point_chunks():
    tree = {"op": "transform", "rotate_deg": [0, 0, 45], "child": ball(1)}
    field = F.compile_tree(tree)
    meshes = [F.mesh_field(field, [[-1.5]*3, [1.5]*3], .2, chunk_points=n)
              for n in (1, 127, 32768)]
    assert len({mesh.hash() for mesh in meshes}) == 1


def test_field_rejects_unrepresentable_interpolation_detail():
    field = F.compile_tree(ball(.01, [1e12]*3))
    bounds = np.array([[-.015]*3, [.015]*3]) + 1e12
    with pytest.raises(AttaError, match="floating-point resolution"):
        F.mesh_field(field, bounds, .002)


def test_thicken_rejects_thickness_lost_at_far_origin():
    verts = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]]) + [1e16, 0, 0]
    with pytest.raises(AttaError) as e:
        thicken(Mesh(verts, [[0, 1, 2], [0, 2, 3]]), .1)
    assert e.value.code == "thicken_precision"


@pytest.mark.parametrize("tree,scale", [
    ({"op": "sphere", "radius": 1e308}, 1e308),
    ({"op": "sphere", "radius": 1e-323}, 1e-3),
    ({"op": "smooth_union", "radius": 1e-323,
      "children": [ball(1), ball(1)]}, 1e-3),
    ({"op": "twist", "angle_deg": 90, "length": 1e-323,
      "child": ball(1)}, 1e-3),
])
def test_unit_scaling_overflow_or_underflow_fails_cleanly(tree, scale):
    with pytest.raises(AttaError) as e:
        F.compile_tree(tree, scale=scale)
    assert e.value.code == "bad_arg"


@pytest.mark.parametrize("bounds,spacing", [
    ([[-1e308] * 3, [1e308] * 3], 1e307),
    ([[-2] * 3, [2] * 3], 1e-320),
])
def test_unrepresentable_sampling_grid_fails_cleanly(bounds, spacing):
    with pytest.raises(AttaError) as e:
        F.mesh_field(F.compile_tree(ball()), bounds, spacing)
    assert e.value.code == "bad_arg"


def test_chunk_size_cannot_change_world_noise_coordinates():
    tree = {"op": "transform", "translate": [40, 0, 0], "child": {
        "op": "noise", "child": ball(), "amplitude": 1, "wavelength": 3, "seed": 12}}
    f = F.compile_tree(tree)
    g = F.compile_tree(tree["child"])
    points = np.random.default_rng(3).normal(size=(10, 3))
    assert f(points + [40, 0, 0]) == pytest.approx(g(points), abs=1e-13)
