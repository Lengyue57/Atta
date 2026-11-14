# -*- coding: utf-8 -*-
"""Multi-part assembly: spec, mates, DOF, collision, receipt and export."""

from __future__ import annotations

import json

import pytest

from atta import assembly, engine
from atta import spec as S


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTA_HOME", str(tmp_path))
    return tmp_path


def _box(pid, size, center, *, fixed=True):
    return {
        "id": pid,
        "fixed": fixed,
        "graph": {"features": [
            {"id": pid, "op": "box",
             "args": {"size": list(size), "center": list(center)}},
        ]},
    }


def _example():
    return S.schema()["examples"]["assembly"]


def test_assembly_spec_validates_proposes_and_compiles():
    raw = _example()
    v = S.validate_spec(raw)
    assert v["ok"], v["errors"]
    assert v["spec"]["model"]["kind"] == "assembly"

    c = S.compile_spec(raw)
    assert c["ok"]
    assert c["compiled"]["kind"] == "assembly"
    assert len(c["compiled"]["parts"]) == 2

    p = S.propose_spec(raw)
    # Declared mates but auto ports: propose must surface that as unresolved,
    # not silently claim the bbox frame is a real mating feature.
    assert any(u["code"] == "auto_ports" for u in p["unresolved"])
    assert p["assumption_summary"]["assumed"] >= 1


def test_assembly_build_solves_mate_and_counts_dof():
    built = assembly.build(S.validate_spec(_example())["spec"])
    assert built["status"] in ("ok", "under_constrained")
    mate = built["assembly"]["mates"][0]
    assert mate["ok"] is True
    assert mate["residual_mm"] <= 0.01
    # concentric leaves rotation about the shared z axis free.
    assert built["assembly"]["dof"]["dof"] == 1
    assert built["assembly"]["dof"]["per_part"]["pin"] == 1
    assert built["assembly"]["collision"]["collisions"] == []

    receipt = assembly.verify(built)
    assert receipt["verdict"] == "pass"
    assert receipt["score"] == 1.0
    assert {c["id"] for c in receipt["checks"]} >= {
        "part.plate.closed", "part.pin.closed", "mate.m1", "dof", "collision"}


def test_overlapping_parts_fail_verification():
    raw = {
        "schema": "atta.spec/v3", "name": "overlap",
        "model": {"kind": "assembly",
                  "parts": [_box("a", [20, 20, 20], [10, 10, 10]),
                            _box("b", [20, 20, 20], [15, 10, 10])],
                  "mates": []},
        "quality": "draft", "material": "steel",
    }
    v = S.validate_spec(raw)
    assert v["ok"], v["errors"]
    built = assembly.build(v["spec"])
    collisions = built["assembly"]["collision"]["collisions"]
    assert collisions
    assert collisions[0]["volume_mm3"] > 1000
    receipt = assembly.verify(built)
    assert receipt["verdict"] == "fail"
    assert any(i["check"] == "collision" for i in receipt["issues"])


def test_interference_backend_failure_cannot_pass(monkeypatch):
    import trimesh
    from atta import probe
    parts = [_box("a", [20, 0.2, 1], [0, 0, 0]),
             _box("b", [20, 0.2, 1], [0, 0, 0])]
    parts[0]["transform"] = {"rotate_deg": [0, 0, 45]}
    parts[1]["transform"] = {"rotate_deg": [0, 0, -45]}
    raw = {"schema": "atta.spec/v3", "name": "thin_cross",
           "model": {"kind": "assembly", "parts": parts, "mates": []},
           "quality": "draft", "material": "steel"}
    def fail(*args, **kwargs):
        raise RuntimeError("controlled boolean failure")
    monkeypatch.setattr(trimesh.boolean, "intersection", fail)
    built = assembly.build(S.validate_spec(raw)["spec"])
    collision = built["assembly"]["collision"]
    assert collision["complete"] is False
    assert len(collision["unresolved"]) == 1
    assert "unresolved" in collision["method"]
    assert assembly.verify(built)["verdict"] == "fail"
    receipt = assembly.verify(built, constraints={"interference": [
        {"a": "a", "b": "b", "max_mm3": 1e6}]})
    assert any(c["id"] == "interference.0" and not c["ok"] for c in receipt["checks"])


def test_conflicting_mates_are_reported():
    raw = {
        "schema": "atta.spec/v3", "name": "conflict",
        "model": {"kind": "assembly", "parts": [
            _box("plate", [60, 60, 10], [30, 30, 5]),
            {"id": "pin", "graph": {"features": [
                {"id": "pin", "op": "cylinder",
                 "args": {"radius": 4, "height": 30, "center": [30, 30, 25]}}]}},
        ], "mates": [
            {"id": "m1", "type": "concentric", "a": "plate", "b": "pin",
             "offset_mm": 20},
            {"id": "m2", "type": "concentric", "a": "plate", "b": "pin",
             "offset_mm": 40},
        ]},
        "quality": "draft", "material": "steel",
    }
    v = S.validate_spec(raw)
    assert v["ok"], v["errors"]
    built = assembly.build(v["spec"])
    assert built["status"] == "conflict"
    assert len(built["assembly"]["conflicts"]) == 2
    receipt = assembly.verify(built)
    assert receipt["verdict"] == "fail"
    assert any(i["check"].startswith("mate.") for i in receipt["issues"])


@pytest.mark.parametrize("op, args, prefix", [
    ("subdivide", {"scheme": "simple", "levels": 1},
     [{"id": "b", "op": "box", "args": {"size": [4, 4, 20], "center": [0, 0, 10]}}]),
    ("sweep2", {"section_xy": [[-2, -2], [2, -2], [2, 2], [-2, 2]],
                "path": [[0, 0, 0], [0, 0, 10], [0, 0, 20]]}, []),
    ("pipe", {"path": [[0, 0, 0], [0, 0, 10], [0, 0, 20]], "radius": 2,
              "inner_radius": 1, "segments": 12}, []),
    ("loft2", {"sections": [[[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]],
                              [[-2, -2, 20], [2, -2, 20], [2, 2, 20], [-2, 2, 20]]]}, []),
])
def test_new_ops_assemble_with_declared_ports_and_mm_units(op, args, prefix):
    from atta import metrology
    feature = {"id": "new", "op": op, "args": args}
    if prefix:
        feature["inputs"] = ["b"]
    a = _box("anchor", [10, 10, 10], [0, 0, 0])
    a["ports"] = [{"name": "target", "origin_mm": [30, 0, 10], "axis": [0, 0, 1]}]
    b = {"id": "new", "graph": {"features": prefix + [feature]},
         "ports": [{"name": "shaft", "origin_mm": [0, 0, 10], "axis": [0, 0, 1]}]}
    raw = {"schema": "atta.spec/v3", "name": op, "quality": "draft", "material": "steel",
           "model": {"kind": "assembly", "parts": [a, b], "mates": [
               {"id": "mate", "type": "concentric", "a": "anchor", "b": "new",
                "port_a": "target", "port_b": "shaft"}]}}
    built = assembly.build(S.validate_spec(raw)["spec"])
    assert built["placements"]["new"]["translate_mm"] == pytest.approx([30, 0, 0], abs=1e-4)
    assert metrology.bbox_mm(built["part_meshes"]["new"])["size"] == pytest.approx([4, 4, 20], abs=1e-4)
    assert assembly.verify(built)["verdict"] == "pass"


def test_graph_params_survive_inline_and_model_id_parts(monkeypatch):
    graph = {"features": [{"id": "p", "op": "pipe", "args": {
        "path": [[0, 0, 0], [0, 0, 10], [0, 0, 20]],
        "radius": {"$param": "r"}, "segments": 12}}], "params": {"r": 2}}
    raw = {"schema": "atta.spec/v3", "name": "param_part", "quality": "draft",
           "material": "steel", "model": {"kind": "assembly", "mates": [],
           "parts": [{"id": "p", "fixed": True, "graph": graph}]}}
    validated = S.validate_spec(raw)
    assert validated["ok"], validated["errors"]
    assert validated["spec"]["model"]["parts"][0]["graph"]["params"] == {"r": 2}
    inline = assembly.build(validated["spec"])
    monkeypatch.setattr(assembly.state, "get", lambda *a, **kw: {
        "spec": {"model": {"kind": "graph", **graph}}})
    raw["model"]["parts"] = [{"id": "p", "fixed": True, "model_id": "pipe"}]
    referenced = assembly.build(S.validate_spec(raw)["spec"])
    assert referenced["mesh"].hash() == inline["mesh"].hash()
    assert inline["assembly"]["parts"][0]["metrics"]["bbox_mm"]["size"] == [4, 4, 20]
    raw["model"]["parts"] = [{"id": "p", "graph": graph, "params": {"r": "bad"}}]
    assert S.validate_spec(raw)["ok"] is False


def test_placement_matrix_roundtrips_in_millimetres():
    p = _box("p", [10, 10, 10], [0, 0, 0])
    p["transform"] = {"translate_mm": [100, -25, 3], "rotate_deg": [20, 30, 40]}
    raw = {"schema": "atta.spec/v3", "name": "matrix", "quality": "draft",
           "material": "steel", "model": {"kind": "assembly", "parts": [p]}}
    built = assembly.build(S.validate_spec(raw)["spec"])
    placement = built["placements"]["p"]
    assert [r[3] for r in placement["matrix"][:3]] == [100, -25, 3]
    p["transform"] = placement
    rebuilt = assembly.build(S.validate_spec(raw)["spec"])
    assert rebuilt["part_meshes"]["p"].verts == pytest.approx(
        built["part_meshes"]["p"].verts, abs=1e-10)


@pytest.mark.parametrize("matrix", [
    [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    [[1, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]],
])
def test_nonrigid_assembly_matrix_rejected(matrix):
    from atta.errors import AttaError
    p = _box("p", [10, 10, 10], [0, 0, 0])
    p["transform"] = {"matrix": matrix}
    raw = {"schema": "atta.spec/v3", "name": "matrix", "quality": "draft",
           "material": "steel", "model": {"kind": "assembly", "parts": [p]}}
    validated = S.validate_spec(raw)
    assert not validated["ok"]
    assert any(e["code"] == "bad_transform" for e in validated["errors"])
    with pytest.raises(AttaError, match="rigid"):
        assembly.build(raw)


@pytest.mark.parametrize("mate_type, expected", [("lock", 0), ("point_on_plane", 5),
                                                 (None, 6)])
def test_port_frames_and_per_part_dof(mate_type, expected):
    a = _box("a", [10, 10, 10], [0, 0, 0])
    b = _box("b", [10, 10, 10], [0, 0, 0], fixed=False)
    b["transform"] = {"translate_mm": [30, 0, 0], "rotate_deg": [20, 30, 40]}
    mates = [{"id": "m", "type": mate_type, "a": "a", "b": "b"}] if mate_type else []
    raw = {"schema": "atta.spec/v3", "name": "dof", "quality": "draft",
           "material": "steel", "model": {"kind": "assembly", "parts": [a, b],
                                          "mates": mates}}
    built = assembly.build(S.validate_spec(raw)["spec"])
    assert built["assembly"]["dof"]["dof"] == expected
    assert built["assembly"]["dof"]["per_part"]["b"] == expected
    if mate_type == "lock":
        assert built["placements"]["b"]["rotate_deg"] == pytest.approx([20, 30, 40], abs=1e-5)


def test_no_solve_uses_requested_tolerance_and_cannot_certify_dof():
    a = _box("a", [10, 10, 10], [0, 0, 0])
    b = _box("b", [10, 10, 10], [30, 0, 0])
    raw = {"schema": "atta.spec/v3", "name": "tolerance", "quality": "draft",
           "material": "steel", "model": {"kind": "assembly", "parts": [a, b],
           "mates": [{"id": "d", "type": "distance", "a": "a", "b": "b",
                      "value_mm": 30.05}], "solver": {"tolerance_mm": 0.1}}}
    built = assembly.build(S.validate_spec(raw)["spec"], solve=False)
    assert built["assembly"]["mates"][0]["ok"] is True
    assert assembly.verify(built)["verdict"] == "pass"
    assert assembly.verify(built, constraints={"fully_constrained": True})["verdict"] == "fail"
    raw["model"]["mates"][0].update(value_mm=40, weight=0.0001)
    built = assembly.build(S.validate_spec(raw)["spec"])
    assert built["assembly"]["mates"][0]["ok"] is False
    assert built["assembly"]["mates"][0]["residual_mm"] == pytest.approx(10)
    raw["model"]["mates"][0]["weight"] = 0
    from atta.errors import AttaError
    with pytest.raises(AttaError, match="weight"):
        assembly.build(S.validate_spec(raw)["spec"])


def test_fully_constrained_is_a_gate():
    raw = S.validate_spec(_example())["spec"]
    raw["constraints"] = {"fully_constrained": True}
    built = assembly.build(raw)
    receipt = assembly.verify(built, constraints={"fully_constrained": True})
    # dof=1, so the strict gate must fail rather than call it pass.
    assert receipt["verdict"] == "fail"
    assert any(i["check"] == "dof" for i in receipt["issues"])


def test_allow_interference_suppresses_the_collision():
    raw = {
        "schema": "atta.spec/v3", "name": "press_fit",
        "model": {"kind": "assembly",
                  "parts": [_box("a", [20, 20, 20], [10, 10, 10]),
                            _box("b", [20, 20, 20], [15, 10, 10])],
                  "mates": [],
                  "allow_interference": [{"a": "a", "b": "b"}]},
        "quality": "draft", "material": "steel",
    }
    built = assembly.build(S.validate_spec(raw)["spec"])
    assert built["assembly"]["collision"]["collisions"] == []
    assert assembly.verify(built)["verdict"] == "pass"


def test_engine_model_describe_verify_assembly(home):
    out = engine.model(spec=_example())
    assert out["model_id"]
    assert out["model"]["kind"] == "assembly"
    assert out["facts"]["assembly"]["part_count"] == 2
    mid = out["model_id"]

    described = engine.describe(mid)
    assert described["assembly"]["part_count"] == 2
    assert described["assembly"]["dof"]["dof"] == 1

    receipt = engine.verify(mid)
    assert receipt["verdict"] == "pass"
    assert (home / "_atta" / "out" / mid / "verify.json").is_file()


def test_engine_assembly_deliver_writes_scene_and_parts(home, tmp_path):
    out = engine.model(spec=_example())
    delivered = engine.deliver(out["model_id"], formats=["glb", "obj"],
                               outdir=str(tmp_path / "export"))
    assert delivered["status"] == "ok"
    formats = [f["format"] for f in delivered["files"]]
    assert "glb" in formats
    assert formats.count("obj") == 2
    assert delivered["verify"]["verdict"] == "pass"
    manifest = json.loads(open(delivered["manifest_path"], encoding="utf-8").read())
    assert manifest["kind"] == "assembly"
    assert len(manifest["parts"]) == 2


def test_assembly_parameter_solve_hits_a_composite_target(home):
    raw = {
        "schema": "atta.spec/v3", "name": "one_part",
        "model": {"kind": "assembly", "parts": [
            {"id": "block", "fixed": True, "recipe": "primitive.chamfer_box",
             "params": {"length": 40, "width": 30, "height": 25}},
        ], "mates": []},
        "quality": "draft", "material": "steel",
    }
    built = engine.model(spec=raw)
    solved = engine.variant(
        base=built["model_id"], targets={"bbox_z": 50.0},
        strategy="compass", count=5, max_evaluations=60, tolerance=1e-4,
    )
    assert solved["status"] == "ok"
    assert solved["feasible"] is True
    assert solved["model_id"]
    assert solved["solution"]["metrics"]["bbox_z"] == pytest.approx(50.0, abs=1e-3)
    assert solved["solution"]["verdict"] in ("pass", "degraded")
