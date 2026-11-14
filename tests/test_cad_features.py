# -*- coding: utf-8 -*-
"""Exact edge finishing: semantic selection, graph provenance, STEP and topology."""
import json
import math

import numpy as np
import pytest

from atta import brep, features
from atta.errors import AttaError
from atta.families import load

load(strict=True)


def graph(op="fillet", edges="all", size=1, **extra):
    return [
        {"id": "base", "op": "box", "args": {"size": [30, 20, 10]}},
        {"id": "out", "op": op, "inputs": ["base"],
         "args": {"radius" if op == "fillet" else "length": size, "edges": edges, **extra}},
    ]


def spec(rows, **model):
    return {"schema": "atta.spec/v3", "name": "edge_test",
            "model": {"kind": "graph", "features": rows, "units": "mm", "output": "out", **model}}


def occt():
    return pytest.importorskip("build123d")


def assert_closed(mesh, components=1):
    topo = mesh.topology()
    assert topo["closed"] and topo["n_components"] == components
    assert topo["boundary_edges"] == topo["non_manifold_edges"] == topo["inconsistent_winding_edges"] == 0
    assert topo["degenerate_faces"] == 0
    a, b, c = mesh.verts[mesh.faces].transpose(1, 0, 2)
    assert np.all(np.linalg.norm(np.cross(b - a, c - a), axis=1) > 0)


@pytest.mark.parametrize("op", ["fillet", "chamfer"])
def test_exact_all_step_roundtrip(op, tmp_path):
    b = occt()
    rows = graph(op)
    mesh = features.evaluate(rows)
    report = brep.build_graph(spec(rows))
    assert_closed(mesh)
    assert mesh.meta["selection_receipts"][0]["selected_count"] == 12
    assert mesh.meta["analytic_volume_mm3"] == report["volume_mm3"]
    assert mesh.volume() * 1e9 == pytest.approx(report["volume_mm3"], rel=1e-4)
    assert mesh.bbox()["size"] == pytest.approx([0.03, 0.02, 0.01], abs=1e-8)
    path = tmp_path / (op + ".step")
    receipt = brep.export_step(spec(rows), str(path))
    reimport = b.import_step(str(path))
    assert reimport.is_valid and len(reimport.solids()) == 1
    assert reimport.volume == pytest.approx(report["volume_mm3"], abs=1e-5)
    assert len(reimport.faces()) == len(report["shape"].faces())
    assert receipt["volume_mm3"] == mesh.meta["analytic_volume_mm3"]
    assert receipt["source_sha256"] == mesh.meta["source_sha256"]
    assert receipt["selection_receipts"] == mesh.meta["selection_receipts"]
    assert tuple(reimport.bounding_box().min) == pytest.approx(np.asarray(mesh.bbox()["min"]) * 1000, abs=1e-5)
    assert tuple(reimport.bounding_box().max) == pytest.approx(np.asarray(mesh.bbox()["max"]) * 1000, abs=1e-5)
    json.dumps(mesh.meta, allow_nan=False)


@pytest.mark.parametrize("op", ["fillet", "chamfer"])
def test_direction_selection_has_expected_analytic_volume_and_size(op):
    occt()
    rows = graph(op, {"parallel": "z", "kind": "line", "expect_count": 4}, 2)
    result = brep.build_graph(spec(rows))
    loss_per_corner = (4 - math.pi) if op == "fillet" else 2
    assert result["volume_mm3"] == pytest.approx(6000 - 4 * loss_per_corner * 10, abs=1e-6)
    if op == "fillet":
        radii = [edge.radius for edge in result["shape"].edges() if edge.geom_type.name == "CIRCLE"]
        assert len(radii) == 8 and radii == pytest.approx([2] * 8)
    assert_closed(features.evaluate(rows))


@pytest.mark.parametrize("units, factor", [("mm", 1), ("cm", 10), ("m", 1000), ("in", 25.4)])
def test_units_parameters_forward_refs_and_world_selection(units, factor):
    occt()
    rows = [
        {"id": "out", "op": "chamfer", "inputs": ["moved"], "args": {
            "length": {"$expr": "r/2"},
            "edges": {"parallel": "x", "at": {"axis": "z", "value": 25 / factor,
                                             "tolerance": 1e-5 / factor}, "expect_count": 2}}},
        {"id": "base", "op": "box", "args": {"size": [30 / factor, 20 / factor, 10 / factor]}},
        {"id": "moved", "op": "transform", "inputs": ["base"], "args": {
            "rotate_deg": [0, 0, 90], "translate": [0, 0, 20 / factor]}},
        {"id": "unused", "op": "subdivide", "inputs": ["base"]},
    ]
    params = {"r": 2 / factor}
    mesh = features.evaluate(rows, params=params, units=units, output="out")
    report = brep.build_graph(spec(rows, params=params, units=units))
    assert_closed(mesh)
    assert mesh.volume() * 1e9 == pytest.approx(5980, abs=1e-6)
    assert report["volume_mm3"] == pytest.approx(5980, abs=1e-6)
    assert mesh.bbox()["min"] == pytest.approx([-0.01, -0.015, 0.015], abs=1e-8)
    assert mesh.meta["selection_receipts"][0]["selected_count"] == 2


@pytest.mark.parametrize("selector", [0, [0, 1], "z", {}, {"edge_index": 2},
    {"parallel": "diagonal"}, {"kind": "plane"}, {"expect_count": 4},
    {"at": {"axis": "z"}}, {"at": {"axis": "w", "value": 0}},
    {"at": {"axis": "z", "value": "foo"}},
    {"at": {"axis": "z", "value": True}},
    {"at": {"axis": "z", "value": 5, "tolerance": -1}}])
def test_bad_selection_is_never_silently_all(selector):
    occt()
    with pytest.raises(AttaError) as error:
        features.evaluate(graph(edges=selector))
    assert error.value.code in ("bad_selector", "bad_arg")
    assert error.value.fix


@pytest.mark.parametrize("selector, field", [
    ({"parallel": None}, "parallel"),
    ({"kind": None}, "kind"),
    ({"at": None}, "at"),
    ({"parallel": "z", "kind": None}, "kind"),
    ({"parallel": "z", "at": None}, "at"),
    ({"parallel": "z", "expect_count": None}, "expect_count"),
])
@pytest.mark.parametrize("op", ["fillet", "chamfer"])
def test_null_selector_conditions_never_select_all(op, selector, field):
    occt()
    with pytest.raises(AttaError) as error:
        features.evaluate(graph(op, edges=selector))
    assert error.value.code == "bad_selector"
    assert error.value.path == "args.edges." + field
    assert error.value.fix


@pytest.mark.parametrize("selector, code", [
    ({"kind": "circle"}, "brep_selection_empty"),
    ({"parallel": "z", "expect_count": 3}, "brep_selection_count"),
    ({"at": {"axis": "z", "value": 0}}, "brep_selection_empty"),
])
def test_empty_or_wrong_count_is_an_error(selector, code):
    occt()
    with pytest.raises(AttaError) as error:
        features.evaluate(graph(edges=selector))
    assert error.value.code == code


@pytest.mark.parametrize("op", ["fillet", "chamfer"])
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), 100])
def test_invalid_or_infeasible_dimensions_fail_closed(op, value):
    occt()
    with pytest.raises(AttaError) as error:
        features.evaluate(graph(op, size=value))
    assert error.value.code in ("bad_arg", "brep_operation_failed")
    assert error.value.fix


@pytest.mark.parametrize("source", ["open", "unsupported", "successor", "nonuniform"])
def test_no_unsupported_graph_fallback(source):
    occt()
    rows = graph()
    if source == "open":
        rows[0] = {"id": "base", "op": "cylinder", "args": {"radius": 5, "height": 10, "caps": False}}
    elif source in ("unsupported", "nonuniform"):
        rows.insert(1, {"id": "middle", "op": "subdivide" if source == "unsupported" else "transform",
                        "inputs": ["base"], "args": {} if source == "unsupported" else {"scale": [1, 2, 1]}})
        rows[-1]["inputs"] = ["middle"]
    else:
        rows.append({"id": "mesh_only", "op": "subdivide", "inputs": ["out"]})
    with pytest.raises(AttaError) as error:
        features.evaluate(rows)
    assert error.value.code == "brep_unsupported"


def test_plain_mesh_and_unreachable_cad_never_call_brep(monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("ordinary mesh graph reached BREP")
    monkeypatch.setattr(brep, "_require", forbid)
    monkeypatch.setattr(brep, "mesh_graph", forbid)
    rows = graph()
    assert_closed(features.evaluate(rows[:1]))
    assert_closed(features.evaluate(rows, output="base"))


def test_missing_backend_is_an_explicit_error(monkeypatch):
    def missing():
        raise AttaError("brep_unavailable", "optional backend absent", fix="install build123d")
    monkeypatch.setattr(brep, "_require", missing)
    with pytest.raises(AttaError) as error:
        features.evaluate(graph())
    assert error.value.code == "brep_unavailable"


@pytest.mark.parametrize("extra,code", [({"max_triangles": 4}, "brep_mesh_budget"),
    ({"max_triangles": 1_000_000}, "bad_arg"), ({"max_triangles": 3.5}, "bad_arg"),
    ({"linear_deflection": 1e-12}, "brep_mesh_budget"),
    ({"angular_deflection_deg": 0.1}, "bad_arg"), ({"linear_deflection": 0}, "bad_arg")])
def test_explicit_mesh_budget_errors(extra, code):
    occt()
    with pytest.raises(AttaError) as error:
        features.evaluate(graph(**extra))
    assert error.value.code == code


@pytest.mark.parametrize("primitive", ["sphere", "cylinder", "torus", "cone"])
def test_seams_and_poles_are_topological_not_welded(primitive):
    b = occt()
    shape = {"sphere": lambda: b.Sphere(5), "cylinder": lambda: b.Cylinder(5, 10),
             "torus": lambda: b.Torus(5, 1), "cone": lambda: b.Cone(5, 0, 10)}[primitive]()
    mesh = brep.tessellate(shape)
    assert_closed(mesh)
    assert mesh.topology()["genus"] == (1 if primitive == "torus" else 0)
    assert mesh.volume() * 1e9 == pytest.approx(shape.volume, rel=0.01)
    assert mesh.meta["tessellation"]["spatial_weld"] is False


def test_touching_distinct_solids_preserve_two_manifold_components():
    b = occt()
    shape = b.Compound([b.Box(10, 10, 10), b.Pos(10, 10, 0) * b.Box(10, 10, 10)])
    mesh = brep.tessellate(shape)
    assert_closed(mesh, components=2)
    assert mesh.volume() * 1e9 == pytest.approx(2000)
    # Two vertex pairs occupy the same coordinates but remain topologically distinct.
    assert len(np.unique(mesh.verts, axis=0)) < len(mesh.verts)


def test_repeated_build_and_row_order_selection_are_deterministic():
    occt()
    rows = graph(edges={"parallel": "z", "expect_count": 4})
    a = features.evaluate(rows)
    b = features.evaluate(rows[::-1], output="out")
    assert a.hash() == b.hash()
    assert a.meta["selection_receipts"] == b.meta["selection_receipts"]


def test_transform_and_boolean_after_finish_stay_exact():
    occt()
    rows = graph("chamfer", {"parallel": "z", "expect_count": 4})
    rows += [
        {"id": "scaled", "op": "transform", "inputs": ["out"], "args": {
            "scale": 2, "rotate_deg": [20, 30, 40], "pivot": [5, 1, 2], "translate": [3, 4, 5]}},
        {"id": "tiny", "op": "box", "args": {"size": [1, 1, 1]}},
        {"id": "final", "op": "difference", "inputs": ["scaled", "tiny"]},
    ]
    mesh = features.evaluate(rows)
    report = brep.build_graph(spec(rows, output="final"))
    # A fully enclosed void adds a second connected boundary shell, not a second solid.
    assert_closed(mesh, components=2)
    assert mesh.meta["backend"] == "brep"
    assert mesh.volume() * 1e9 == pytest.approx(report["volume_mm3"], abs=1e-5)
    assert report["volume_mm3"] == pytest.approx(5980 * 8 - 1, abs=1e-5)


def test_circle_selector_top_rim():
    occt()
    rows = graph("fillet", {"kind": "circle", "at": {"axis": "z", "value": "max"}, "expect_count": 1})
    rows[0] = {"id": "base", "op": "cylinder", "args": {"radius": 5, "height": 10}}
    mesh = features.evaluate(rows)
    assert_closed(mesh)
    assert mesh.meta["selection_receipts"][0]["selected_count"] == 1


@pytest.mark.parametrize("direction", [[0, 0, 1], [1, 0, 0], [0, -1, 0], [1, 2, 3]])
def test_extrude_direction_and_holes_match_existing_mesh_semantics(direction):
    occt()
    args = {"loop_xy": [[-5, -4], [5, -4], [5, 4], [-5, 4]],
            "holes_xy": [[[-1, -1], [1, -1], [1, 1], [-1, 1]]],
            "height": 7, "direction": direction}
    rows = [{"id": "out", "op": "extrude", "args": args}]
    mesh = features.evaluate(rows)
    report = brep.build_graph(spec(rows))
    assert report["volume_mm3"] == pytest.approx(532, abs=1e-6)
    assert mesh.volume() * 1e9 == pytest.approx(report["volume_mm3"], abs=1e-3)
    assert tuple(report["shape"].bounding_box().min) == pytest.approx(np.asarray(mesh.bbox()["min"]) * 1000, abs=1e-5)
    assert tuple(report["shape"].bounding_box().max) == pytest.approx(np.asarray(mesh.bbox()["max"]) * 1000, abs=1e-5)
    assert_closed(brep.tessellate(report["shape"]))


@pytest.mark.parametrize("profile", [[[5, 0], [5, 10]], [[5, 0], [5, 5], [3, 10]],
                                    [[3, 0], [5, 0], [5, 10], [3, 10], [3, 0]]])
def test_revolve_caps_and_closed_profiles_match_existing_mesh(profile):
    occt()
    rows = [{"id": "out", "op": "revolve", "args": {"profile_rz": profile, "segments": 256}}]
    mesh = features.evaluate(rows)
    report = brep.build_graph(spec(rows))
    assert mesh.volume() * 1e9 == pytest.approx(report["volume_mm3"], rel=0.0002)
    assert_closed(brep.tessellate(report["shape"]))


def test_finish_after_boolean_hole_and_mirror_pattern():
    occt()
    rows = [
        {"id": "body", "op": "box", "args": {"size": [30, 20, 10]}},
        {"id": "drill", "op": "cylinder", "args": {"radius": 3, "height": 20}},
        {"id": "hole", "op": "difference", "inputs": ["body", "drill"]},
        {"id": "round", "op": "fillet", "inputs": ["hole"], "args": {
            "radius": 0.5, "edges": {"kind": "circle", "at": {"axis": "z", "value": "max"}, "expect_count": 1}}},
        {"id": "mirror", "op": "mirror", "inputs": ["round"], "args": {"plane": "x"}},
        {"id": "out", "op": "pattern_linear", "inputs": ["mirror"], "args": {"count": 2, "spacing": 40}},
    ]
    mesh = features.evaluate(rows)
    assert_closed(mesh, components=2)
    assert mesh.topology()["genus"] == 2
    report = brep.build_graph(spec(rows))
    assert mesh.volume() * 1e9 == pytest.approx(report["volume_mm3"], rel=0.001)
    assert len(report["shape"].solids()) == 2


def test_tessellation_rejects_open_faces_and_loose_entities():
    b = occt()
    for shape in [b.Face.make_rect(10, 10),
                  b.Compound([b.Box(10, 10, 10), b.Edge.make_line((20, 0, 0), (25, 0, 0))])]:
        with pytest.raises(AttaError) as error:
            brep.tessellate(shape)
        assert error.value.code in ("brep_empty", "brep_invalid")


def test_empty_boolean_and_invalid_native_profile_raise_atta_errors():
    occt()
    rows = graph()
    rows.insert(1, {"id": "far", "op": "box", "args": {"size": [1, 1, 1], "center": [100, 0, 0]}})
    rows.insert(2, {"id": "intersection", "op": "intersection", "inputs": ["base", "far"]})
    rows[-1]["inputs"] = ["intersection"]
    with pytest.raises(AttaError) as error:
        features.evaluate(rows)
    assert error.value.code in ("brep_empty", "brep_build")
    with pytest.raises(AttaError):
        brep.build_graph(spec([{"id": "out", "op": "extrude", "args": {
            "loop_xy": [[0, 0], [1, 0], [2, 0]], "height": 10}}]))


def test_step_export_rejects_file_parent_with_structured_error(tmp_path):
    occt()
    parent = tmp_path / "not_a_directory"
    parent.write_text("keep this file", encoding="utf-8")
    output = parent / "part.step"
    with pytest.raises(AttaError) as error:
        brep.export_step(spec(graph("chamfer", {"parallel": "z"})), str(output))
    assert error.value.code == "brep_export"
    assert error.value.path == "path"
    assert error.value.fix
    assert error.value.details["path"] == str(output)
    assert isinstance(error.value.__cause__, FileExistsError)
    assert parent.read_text(encoding="utf-8") == "keep this file"


def test_step_export_does_not_relabel_non_io_errors(tmp_path, monkeypatch):
    b = occt()
    failure = RuntimeError("native exporter failed")
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(b, "export_step", fail)
    with pytest.raises(RuntimeError) as error:
        brep.export_step(spec(graph("chamfer", {"parallel": "z"})), str(tmp_path / "part.step"))
    assert error.value is failure


def test_engine_delivery_same_source(tmp_path, monkeypatch):
    occt()
    from atta import engine
    monkeypatch.setenv("ATTA_HOME", str(tmp_path / "state"))
    result = engine.model(spec=spec(graph("chamfer", {"parallel": "z", "expect_count": 4})))
    assert engine.verify(result["model_id"])["verdict"] == "pass"
    delivery = engine.deliver(result["model_id"], formats=["step"], outdir=str(tmp_path / "out"))
    assert delivery["status"] == "ok"
    assert delivery["brep"]["volume_mm3"] == pytest.approx(result["facts"]["volume_mm3"], abs=1e-4)
