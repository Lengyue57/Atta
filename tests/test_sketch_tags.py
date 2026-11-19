# -*- coding: utf-8 -*-
"""Tests for typed 2D sketch DSL (P1-7a) and face tags / edge selection (P1-5)."""

import math

import numpy as np
import pytest

from atta import brep, curves, features
from atta.errors import AttaError


def _occt():
    return pytest.importorskip("build123d")


# ----------------------------------------------------------------- P1-7a Sketch DSL

def test_eval_sketch_segments_lines_arcs_tangent_arcs():
    segs = [
        {"op": "start", "at": [0, 0]},
        {"op": "line", "to": [10, 0], "tag": "bottom"},
        {"op": "tangent_arc", "to": [15, 5], "tag": "corner_br"},
        {"op": "line", "to": [15, 10], "tag": "right"},
        {"op": "arc", "to": [10, 15], "center": [10, 10], "cw": False, "tag": "corner_tr"},
        {"op": "arc", "to": [0, 10], "via": [5, 17], "tag": "arch_top"},
        {"op": "close", "tag": "left"},
    ]
    loop, meta = curves.eval_sketch_segments(segs, samples_per_arc=16)
    assert loop.ndim == 2 and loop.shape[1] == 2
    assert len(loop) > 20
    assert [m["op"] for m in meta] == ["line", "tangent_arc", "line", "arc", "arc", "line"]
    assert [m["tag"] for m in meta] == ["bottom", "corner_br", "right", "corner_tr", "arch_top", "left"]

    # Check tangent_arc geometry: entering from (0,0)->(10,0) (+X) to (15,5) gives R=5, center=(10,5), CCW.
    tarc = meta[1]
    assert tarc["center"] == pytest.approx([10.0, 5.0], abs=1e-6)
    assert tarc["radius"] == pytest.approx(5.0, abs=1e-6)
    assert tarc["cw"] is False

    # Check 3-point via arc passes through via point [5, 17] analytically.
    via_arc = meta[4]
    c = np.asarray(via_arc["center"])
    assert np.linalg.norm(np.array([5.0, 17.0]) - c) == pytest.approx(via_arc["radius"], abs=1e-9)
    pts = np.asarray(via_arc["points"])
    dists = np.linalg.norm(pts - np.array([5.0, 17.0]), axis=1)
    assert dists.min() < 0.25


def test_eval_sketch_segments_circle_and_errors():
    loop, meta = curves.eval_sketch_segments(
        [{"op": "circle", "center": [3, -2], "radius": 5, "tag": "bore"}],
        samples_per_arc=24,
    )
    assert len(loop) == 48
    assert meta[0]["op"] == "circle"
    assert meta[0]["tag"] == "bore"
    assert np.allclose(np.linalg.norm(loop - np.array([3.0, -2.0]), axis=1), 5.0)

    bad_cases = [
        [],
        [{"op": "tangent_arc", "to": [5, 5]}],
        [{"op": "line", "to": [5, 0]}, {"op": "circle", "center": [0, 0], "radius": 2}],
        [{"op": "start", "at": [0, 0]}, {"op": "arc", "to": [10, 0], "via": [5, 0]}],
        [{"op": "start", "at": [0, 0]}, {"op": "arc", "to": [10, 0], "center": [3, 0]}],
        [{"op": "bogus", "to": [1, 1]}],
    ]
    for bad in bad_cases:
        with pytest.raises(AttaError) as err:
            curves.eval_sketch_segments(bad)
        assert err.value.code == "bad_arg"


def test_extrude_with_segments_preserves_analytic_arcs_in_step(tmp_path):
    b = _occt()
    # Rounded-corner plate: 20 x 10 rectangle with R=2 tangent arc on one corner, height=5
    segs = [
        {"op": "start", "at": [0, 0]},
        {"op": "line", "to": [18, 0], "tag": "bottom"},
        {"op": "tangent_arc", "to": [20, 2], "tag": "fillet_corner"},
        {"op": "line", "to": [20, 10], "tag": "right"},
        {"op": "line", "to": [0, 10], "tag": "top"},
    ]
    rows = [{"id": "part", "op": "extrude", "args": {"segments": segs, "height": 5}}]
    mesh = features.evaluate(rows)
    assert mesh.topology()["closed"]
    expected_area = 20 * 10 - (4 - math.pi)
    expected_vol = expected_area * 5
    assert mesh.volume() * 1e9 == pytest.approx(expected_vol, rel=0.005)

    spec = {"schema": "atta.spec/v3", "name": "arc_plate",
            "model": {"kind": "graph", "units": "mm", "features": rows, "output": "part"}}
    report = brep.build_graph(spec)
    assert report["volume_mm3"] == pytest.approx(expected_vol, abs=1e-5)

    step_path = tmp_path / "arc_plate.step"
    brep.export_step(spec, str(step_path))
    reimport = b.import_step(str(step_path))
    circle_edges = [e for e in reimport.edges() if e.geom_type.name == "CIRCLE"]
    assert len(circle_edges) == 2
    assert all(e.radius == pytest.approx(2.0, abs=1e-5) for e in circle_edges)


def test_revolve_with_segments_circle_makes_analytic_torus(tmp_path):
    b = _occt()
    segs = [{"op": "circle", "center": [10, 0], "radius": 2, "tag": "ring"}]
    rows = [{"id": "torus", "op": "revolve", "args": {"segments": segs}}]
    mesh = features.evaluate(rows)
    assert mesh.topology()["closed"]
    assert mesh.topology()["genus"] == 1
    expected_vol = 2.0 * (math.pi ** 2) * 10.0 * (2.0 ** 2)
    assert mesh.volume() * 1e9 == pytest.approx(expected_vol, rel=0.02)

    spec = {"schema": "atta.spec/v3", "name": "rev_torus",
            "model": {"kind": "graph", "units": "mm", "features": rows, "output": "torus"}}
    report = brep.build_graph(spec)
    assert report["volume_mm3"] == pytest.approx(expected_vol, abs=1e-4)
    step_path = tmp_path / "rev_torus.step"
    brep.export_step(spec, str(step_path))
    reimport = b.import_step(str(step_path))
    assert any(f.geom_type.name == "TORUS" for f in reimport.faces())


def test_extrude_segments_units_scaling():
    _occt()
    segs = [{"op": "circle", "center": [0, 0], "radius": 1}]  # 1 cm = 10 mm
    rows = [
        {"id": "cyl", "op": "extrude", "args": {"segments": segs, "height": 2}},
        {"id": "out", "op": "fillet", "inputs": ["cyl"], "args": {
            "radius": 0.1, "edges": {"kind": "circle", "at": {"axis": "z", "value": "max"}, "expect_count": 1}}},
    ]
    mesh = features.evaluate(rows, units="cm")
    assert mesh.topology()["closed"]
    assert mesh.bbox()["size"] == pytest.approx([0.02, 0.02, 0.02], abs=1e-4)


# ----------------------------------------------------------------- P1-5 Tags & Edge Selection

def test_face_tags_and_common_edge_of_on_face_selection():
    _occt()
    rows = [
        {
            "id": "body",
            "op": "extrude",
            "tags": {"end": "cap", "start": "base"},
            "args": {
                "height": 10,
                "segments": [
                    {"op": "start", "at": [-10, -5]},
                    {"op": "line", "to": [10, -5], "tag": "front_wall"},
                    {"op": "line", "to": [10, 5], "tag": "right_wall"},
                    {"op": "line", "to": [-10, 5], "tag": "back_wall"},
                    {"op": "close", "tag": "left_wall"},
                ],
            },
        },
        {
            "id": "filleted",
            "op": "fillet",
            "inputs": ["body"],
            "args": {
                "radius": 1.0,
                "edges": {"common_edge_of": ["front_wall", "cap"], "expect_count": 1},
            },
        },
        {
            "id": "out",
            "op": "chamfer",
            "inputs": ["filleted"],
            "args": {
                "length": 0.5,
                "edges": {"on_face": "base", "parallel": "y", "expect_count": 2},
            },
        },
    ]
    spec = {"schema": "atta.spec/v3", "name": "tagged_extrude",
            "model": {"kind": "graph", "units": "mm", "features": rows, "output": "out"}}
    report = brep.build_graph(spec)
    assert report["selections"][0]["selected_count"] == 1
    assert report["selections"][1]["selected_count"] == 2
    mesh = features.evaluate(rows)
    assert mesh.topology()["closed"]


def test_face_tags_survive_boolean_and_transform_and_from_feature():
    _occt()
    rows = [
        {
            "id": "plate",
            "op": "box",
            "tags": {"top": "lid", "bottom": "base"},
            "args": {"size": [30, 20, 10]},
        },
        {
            "id": "drill",
            "op": "cylinder",
            "tags": {"side": "bore"},
            "args": {"radius": 4, "height": 20},
        },
        {
            "id": "drilled",
            "op": "difference",
            "inputs": ["plate", "drill"],
        },
        {
            "id": "moved",
            "op": "transform",
            "inputs": ["drilled"],
            "args": {"translate": [15, 0, 5], "rotate_deg": [0, 0, 30]},
        },
        {
            "id": "out",
            "op": "fillet",
            "inputs": ["moved"],
            "args": {
                "radius": 0.5,
                "edges": {"common_edge_of": ["lid", "bore"], "expect_count": 1},
            },
        },
    ]
    spec = {"schema": "atta.spec/v3", "name": "tagged_bool",
            "model": {"kind": "graph", "units": "mm", "features": rows, "output": "out"}}
    report = brep.build_graph(spec)
    assert report["selections"][0]["selected_count"] == 1
    assert report["selections"][0]["edges"][0]["kind"] == "circle"

    # Also test direct brep.select_edges with from_feature + face_tags metadata
    moved_spec = {"schema": "atta.spec/v3", "name": "moved_only",
                  "model": {"kind": "graph", "units": "mm", "features": rows[:-1], "output": "moved"}}
    moved_shape = brep.build_graph(moved_spec)["shape"]
    sel_drill, _ = brep.select_edges(moved_shape, {"from_feature": "drill", "kind": "circle", "expect_count": 2})
    assert len(sel_drill) == 2
    assert all(e.geom_type.name == "CIRCLE" for e in sel_drill)


def test_revolve_segments_with_arc_and_tags_and_mirror_pattern():
    _occt()
    # Stepped shaft / flange with an arc groove and tagged outer wall
    segs = [
        {"op": "start", "at": [0, 0]},
        {"op": "line", "to": [10, 0], "tag": "bottom_face"},
        {"op": "line", "to": [10, 5], "tag": "flange_od"},
        {"op": "arc", "to": [6, 9], "center": [10, 9], "cw": True, "tag": "neck_fillet"},
        {"op": "line", "to": [6, 15], "tag": "shaft_od"},
        {"op": "line", "to": [0, 15], "tag": "top_face"},
        {"op": "close"},
    ]
    rows = [
        {"id": "shaft", "op": "revolve", "tags": {"end": "top_cap"}, "args": {"segments": segs}},
        {"id": "mirrored", "op": "mirror", "inputs": ["shaft"], "args": {"plane": "z"}},
        {"id": "out", "op": "fillet", "inputs": ["mirrored"], "args": {
            "radius": 0.4, "edges": {"on_face": "flange_od", "kind": "circle", "expect_count": 2}}},
    ]
    mesh = features.evaluate(rows)
    assert mesh.topology()["closed"]
    report = brep.build_graph({"schema": "atta.spec/v3", "name": "rev_tags",
                               "model": {"kind": "graph", "units": "mm", "features": rows, "output": "out"}})
    assert report["selections"][0]["selected_count"] == 2
    assert any(f.geom_type.name == "TORUS" for f in report["shape"].faces())


def test_direct_face_tags_passed_to_select_edges_and_pattern():
    b = _occt()
    box = b.Box(10, 10, 10)
    top_faces = [f for f in box.faces() if f.normal_at().Z > 0.9]
    front_faces = [f for f in box.faces() if f.normal_at().Y < -0.9]
    sel, desc = brep.select_edges(
        box,
        {"common_edge_of": ["top", "front"], "expect_count": 1},
        face_tags={"top": top_faces, "front": front_faces},
    )
    assert len(sel) == 1
    assert desc[0]["center_mm"] == pytest.approx([0.0, -5.0, 5.0], abs=1e-6)

    # Pattern propagation
    rows = [
        {"id": "pin", "op": "cylinder", "tags": {"end": "tip", "side": "wall"},
         "args": {"radius": 2, "height": 8}},
        {"id": "arr", "op": "pattern_linear", "inputs": ["pin"], "args": {"count": 3, "spacing": 10}},
        {"id": "out", "op": "chamfer", "inputs": ["arr"], "args": {
            "length": 0.3, "edges": {"common_edge_of": ["tip", "wall"], "expect_count": 3}}},
    ]
    mesh = features.evaluate(rows)
    assert mesh.topology()["closed"] and mesh.topology()["n_components"] == 3


def test_select_edges_tag_error_cases():
    b = _occt()
    shape = b.Box(10, 10, 10)
    with pytest.raises(AttaError) as e1:
        brep.select_edges(shape, 0)
    assert e1.value.code == "bad_selector"

    with pytest.raises(AttaError) as e2:
        brep.select_edges(shape, {"on_face": "missing_tag"})
    assert e2.value.code == "brep_selection_empty"

    with pytest.raises(AttaError) as e3:
        brep.select_edges(shape, {"common_edge_of": ["only_one"]})
    assert e3.value.code == "bad_selector"

    with pytest.raises(AttaError) as e4:
        brep.select_edges(shape, {"from_feature": None})
    assert e4.value.code == "bad_selector"
