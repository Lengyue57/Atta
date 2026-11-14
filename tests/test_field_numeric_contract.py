"""Independent scale and invalid-input oracles for composable fields."""
import numpy as np
import pytest

from atta import fieldtree as F, sdf
from atta.errors import AttaError
from atta.features import evaluate


@pytest.mark.parametrize("scale", [1e-200, 1e-170, 1., 1e160, 1e200])
@pytest.mark.parametrize("tree,point,expected", [
    ({"op": "sphere", "radius": 1}, [2, 0, 0], 1),
    ({"op": "box", "size": [2, 2, 2]}, [2, 1, 1], 1),
    ({"op": "cylinder", "radius": 1, "height": 2}, [2, 0, 0], 1),
    ({"op": "torus", "major_radius": 3, "minor_radius": 1}, [5, 0, 0], 1),
    ({"op": "capsule", "a": [0, 0, -1], "b": [0, 0, 1], "radius": 1}, [2, 0, 0], 1),
])
def test_primitive_distance_scales_without_square_overflow(tree, point, expected, scale):
    values = F.compile_tree(tree, scale=scale)(np.array([point]) * scale)
    np.testing.assert_allclose(values / scale, [expected], atol=0, rtol=1e-14)


def test_large_transform_of_tiny_sphere_preserves_distance_sign():
    field = F.compile_tree({"op": "transform", "scale": 1e200,
                            "child": {"op": "sphere", "radius": 1e-200}})
    np.testing.assert_allclose(field([[0, 0, 0], [.5, 0, 0], [2, 0, 0]]),
                               [-1, -.5, 1], atol=0, rtol=1e-14)


@pytest.mark.parametrize("tree,points", [
    ({"op": "gyroid", "period": 1e-309}, [[0, 0, 0]]),
    ({"op": "twist", "angle_deg": 180, "length": 1e-300,
      "child": {"op": "sphere", "radius": 1}}, [[0, 0, 1e300]]),
    ({"op": "transform", "scale": 1e-300,
      "child": {"op": "sphere", "radius": 1}}, [[1e300, 0, 0]]),
    ({"op": "transform", "scale": 1e300,
      "child": {"op": "sphere", "radius": 1}}, [[1e-300, 0, 0]]),
])
def test_nonrepresentable_arithmetic_cannot_be_masked_by_csg(tree, points):
    with pytest.raises(AttaError) as error:
        F.compile_tree({"op": "union", "children": [
            {"op": "sphere", "radius": 1}, tree]})(points)
    assert error.value.code in {"bad_arg", "field_invalid"}
    assert "children[1]" in error.value.path


@pytest.mark.parametrize("tree", [
    {"op": "sphere", "radius": 1, "center": np.array(1.)},
    {"op": "sphere", "radius": 1, 1: 2},
    {"op": "sphere", "radius": np.bool_(True)},
    {"op": "cylinder", "radius": 1, "height": 2, "axis": np.array(["x", "z"])},
])
def test_invalid_tree_types_raise_atta_error(tree):
    with pytest.raises(AttaError):
        F.compile_tree(tree)


@pytest.mark.parametrize("points", [[["bad", 0, 0]], [[10**400, 0, 0]], [0, 0, 0]])
def test_invalid_query_coordinates_raise_atta_error(points):
    with pytest.raises(AttaError):
        F.compile_tree({"op": "sphere", "radius": 1})(points)


@pytest.mark.parametrize("tree", [
    {"op": "offset", "distance": 1e-323, "child": {"op": "sphere", "radius": 1}},
    {"op": "noise", "amplitude": 1e-323, "wavelength": 1,
     "child": {"op": "sphere", "radius": 1}},
    {"op": "twist", "angle_deg": 1e-323, "length": 1,
     "child": {"op": "sphere", "radius": 1}},
])
def test_nonzero_adjustments_cannot_silently_underflow(tree):
    with pytest.raises(AttaError):
        F.compile_tree(tree, scale=1e-3)


def test_huge_integer_bounds_fail_at_both_public_boundaries():
    bounds = [[-10**400] * 3, [10**400] * 3]
    with pytest.raises(AttaError):
        F.mesh_field(F.compile_tree({"op": "sphere", "radius": 1}), bounds, 1)
    with pytest.raises(AttaError):
        evaluate([{"id": "field", "op": "field_tree", "args": {
            "tree": {"op": "sphere", "radius": 1}, "bounds": bounds, "spacing": 1}}])


@pytest.mark.parametrize("call", [
    lambda: sdf.grid_shape([-1e308] * 3, [1e308] * 3, 1e307),
    lambda: sdf.grid_shape([-10**400] * 3, [10**400] * 3, 1),
    lambda: sdf.marching_tets(np.ones((2, 2, 2)), [0, 0], 1),
    lambda: sdf.marching_tets(np.ones((2, 2, 2)), [0, 0, 0], 1, max_triangles=np.inf),
    lambda: sdf.marching_tets(np.ones((2, 2, 2)), [0, 0, 0], 1, chunk_cells=1.5),
    lambda: sdf.fit_spacing([-1] * 3, [1] * 3, 1, max_samples=np.inf),
])
def test_sampling_api_invalid_inputs_raise_atta_error(call):
    with pytest.raises(AttaError):
        call()
