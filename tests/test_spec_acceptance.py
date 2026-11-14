# -*- coding: utf-8 -*-
"""Public Spec acceptance boundaries for placement numbers and inline sources."""
import copy

import pytest

from atta import engine, features, spec
from atta.errors import AttaError


def graph():
    return {"features": [{"id": "b", "op": "box", "args": {"size": [10, 10, 10]}}]}


def assembly():
    return {"model": {"kind": "assembly", "parts": [
        {"id": "a", "graph": graph()}, {"id": "c", "graph": graph()},
    ]}}


PLACEMENT_FIELDS = [
    ("attach", "gap_mm", "bad_attach"),
    ("attach", "roll_deg", "bad_attach"),
    *[("transform", key, "bad_transform") for key in
      ("translate_mm", "translate", "rotate_deg", "rotate", "matrix")],
    *[("ports", key, "bad_port") for key in
      ("origin_mm", "axis", "normal", "x_axis", "tangent", "radius_mm")],
]


def with_placement_value(group, key, value):
    raw = assembly()
    part = raw["model"]["parts"][1]
    if group == "attach":
        part["attach"] = {"target": "a", "target_port": "center",
                          "source_port": "center", key: value}
        path = "$.model.parts[1].attach." + key
    elif group == "transform":
        if key == "matrix":
            matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            matrix[0][3] = value
            part["transform"] = {key: matrix}
        else:
            part["transform"] = {key: [value, 0, 1]}
        path = "$.model.parts[1].transform." + key
    else:
        port = {"name": "probe", key: value if key == "radius_mm" else [value, 0, 1]}
        part["ports"] = [port]
        path = "$.model.parts[1].ports[0]." + key
    return raw, path


@pytest.mark.parametrize("group,key,code", PLACEMENT_FIELDS)
@pytest.mark.parametrize("value", [10**400, -(10**400), float("inf"), float("nan"), True],
                         ids=["huge_positive", "huge_negative", "infinity", "nan", "boolean"])
def test_placement_numbers_return_errors_at_every_public_boundary(
        group, key, code, value, tmp_path, monkeypatch):
    monkeypatch.setenv("ATTA_HOME", str(tmp_path))
    raw, path = with_placement_value(group, key, value)
    validated = spec.validate_spec(raw)
    assert not validated["ok"]
    errors = [e for e in validated["errors"] if e["path"] == path]
    assert errors and errors[0]["code"] == code
    assert errors[0]["fix"]
    compiled = spec.compile_spec(raw)
    assert not compiled["ok"] and compiled["compiled"] is None
    assert compiled["errors"] == validated["errors"]
    with pytest.raises(AttaError) as exc:
        engine.model(spec=raw)
    assert exc.value.code == "spec_invalid"
    assert any(e["path"] == path for e in exc.value.details)


def test_finite_rigid_placement_and_oriented_ports_remain_valid():
    raw = assembly()
    parts = raw["model"]["parts"]
    parts[0]["transform"] = {"matrix": [
        [0, -1, 0, 20], [1, 0, 0, -3], [0, 0, 1, 10], [0, 0, 0, 1],
    ]}
    parts[0]["ports"] = [{"name": "probe", "origin_mm": [1, 2, 3],
                           "axis": [0, 0, 2], "tangent": [3, 0, 0], "radius_mm": 0}]
    parts[1]["attach"] = {"target": "a", "target_port": "probe",
                           "source_port": "center", "gap_mm": -2.5, "roll_deg": 30}
    original = copy.deepcopy(raw)
    validated = spec.validate_spec(raw)
    assert validated["ok"], validated["errors"]
    compiled = spec.compile_spec(raw)
    assert compiled["ok"], compiled["errors"]
    assert compiled["compiled"]["parts"] == validated["spec"]["model"]["parts"]
    assert raw == original


def recipe_spec(chamfer=1):
    return {"model": {"kind": "recipe", "id": "primitive.chamfer_box", "params": {
        "length": 20, "width": 12, "height": 4,
        "chamfer_w": chamfer, "chamfer_w_v": chamfer + 1 if chamfer == 3 else chamfer,
    }}}


def test_blocked_inline_recipe_fails_compile_before_geometry(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTA_HOME", str(tmp_path))
    raw = assembly()
    raw["model"]["parts"].append({"id": "bad", "spec": recipe_spec(chamfer=3)})
    assert spec.validate_spec(raw)["ok"]  # Parameter types/ranges are valid.
    compiled = spec.compile_spec(raw)
    assert not compiled["ok"] and compiled["compiled"] is None
    error = next(e for e in compiled["errors"] if e["code"] == "recipe_blocked")
    assert error["path"] == "$.model.parts[2].spec.model.params"
    assert "chamfer_fits" in error["fix"]
    with pytest.raises(AttaError) as exc:
        engine.model(spec=raw)
    assert exc.value.code == "spec_compile_failed"
    assert exc.value.details == compiled["errors"]


@pytest.mark.parametrize("kind", ["recipe", "graph"])
def test_valid_inline_sources_keep_the_normalized_spec_contract(kind, monkeypatch):
    nested = recipe_spec() if kind == "recipe" else {"model": {"kind": "graph", **graph()}}
    raw = {"model": {"kind": "assembly", "parts": [{"id": "ok", "spec": nested}]}}
    original = copy.deepcopy(raw)
    def forbid_geometry(*args, **kwargs):
        raise AssertionError("compile must not evaluate graph geometry")
    monkeypatch.setattr(features, "evaluate", forbid_geometry)
    validated = spec.validate_spec(raw)
    compiled = spec.compile_spec(raw)
    assert compiled["ok"], compiled["errors"]
    assert compiled["compiled"]["parts"] == validated["spec"]["model"]["parts"]
    assert raw == original


def test_model_id_compile_does_not_load_saved_geometry(monkeypatch):
    from atta import state
    def forbid_load(*args, **kwargs):
        raise AssertionError("compile must not load model_id records")
    monkeypatch.setattr(state, "get", forbid_load)
    raw = {"model": {"kind": "assembly", "parts": [
        {"id": "saved", "model_id": "not-loaded-during-compile"},
    ]}}
    compiled = spec.compile_spec(raw)
    assert compiled["ok"], compiled["errors"]
    assert compiled["compiled"]["parts"][0]["model_id"] == "not-loaded-during-compile"
