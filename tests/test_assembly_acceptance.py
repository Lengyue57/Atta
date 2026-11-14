"""Assembly acceptance and honest STEP delivery at public boundaries."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from atta import assembly, brep, engine
from atta import verify as verifymod
from atta.errors import AttaError


def box(pid="base", *, offset=0):
    return {"id": pid, "fixed": True, "graph": {"features": [
        {"id": "body", "op": "box", "args": {"size": [10, 12, 14]}}]},
        "transform": {"translate_mm": [offset, 0, 0]}}


def doc(parts, constraints=None, mates=None):
    return {"model": {"kind": "assembly", "parts": parts, "mates": mates or []},
            "constraints": constraints or {}}


def degraded_panel():
    return {"id": "panel", "fixed": True, "graph": {"features": [
        {"id": "skin", "op": "nurbs_surface", "args": {
            "control_points": [[[30, 0, 0], [30, 0, 50]],
                               [[30, 30, 0], [30, 30, 50]],
                               [[0, 30, 0], [0, 30, 50]]],
            "weights": [[1, 1], [2 ** -0.5, 2 ** -0.5], [1, 1]],
            "degree_u": 2, "degree_v": 1,
            "tolerance": 1e-9, "max_refinements": 0,
            "fail_on_tolerance": False}},
        {"id": "solid", "op": "thicken", "inputs": ["skin"],
         "args": {"thickness": 1}}]}}


@pytest.mark.parametrize("level", ["basic", "standard", "strict"])
@pytest.mark.parametrize("constraints,check", [
    ({"bbox": {"x": [200, 210]}}, "bbox.x"),
    ({"volume_mm3": [1, 2]}, "volume_mm3"),
    ({"genus": 5}, "genus"),
    ({"closed": False}, "closed.expected"),
    ({"n_components": 2}, "n_components"),
])
def test_whole_assembly_constraints_fail_at_every_level(level, constraints, check):
    built = assembly.build(doc([box()]))
    receipt = assembly.verify(built, constraints=constraints, level=level)
    assert receipt["verdict"] == "fail"
    assert any(c["id"] == check and not c["ok"] for c in receipt["checks"])


def test_dimensions_apply_to_world_assembly_not_each_part():
    built = assembly.build(doc([box(), box("other", offset=30)]))
    receipt = assembly.verify(built, constraints={
        "bbox": {"x": [39.99, 40.01]}, "volume_mm3": [3359, 3361],
        "n_components": 2, "fully_constrained": True})
    assert receipt["verdict"] == "pass", receipt["issues"]
    assert not any(c["id"] == "components" for c in receipt["checks"])


def test_no_general_constraints_do_not_remeasure_mesh(monkeypatch):
    built = assembly.build(doc([box(), box("other", offset=30)]))
    def forbidden(*args, **kwargs):
        raise AssertionError("unrequested generic verification")
    monkeypatch.setattr(verifymod, "verify", forbidden)
    receipt = assembly.verify(built, constraints={"fully_constrained": True,
                              "envelope": {"max": [40, 20, 20]}})
    assert receipt["verdict"] == "pass"


@pytest.mark.parametrize("level", ["basic", "standard", "strict"])
def test_wall_gate_preserves_mesh_verifier_severity_and_skips(level):
    built = assembly.build(doc([box()]))
    constraints = {"min_wall_mm": 100}
    expected = verifymod.verify(built["mesh"], constraints=constraints, level=level)
    actual = assembly.verify(built, constraints=constraints, level=level)
    reference = next(c for c in expected["checks"] if c["id"] == "min_wall_mm")
    assert next(c for c in actual["checks"] if c["id"] == "min_wall_mm") == reference
    assert actual["verdict"] == expected["verdict"]
    assert actual["skipped"] == expected["skipped"]


def test_engine_delivery_blocks_dimension_violation(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTA_HOME", str(tmp_path))
    built = engine.model(spec=doc([box()], {"bbox": {"x": [200, 210]}}))
    result = engine.deliver(built["model_id"], formats=["glb"], outdir=tmp_path / "out")
    assert result["status"] == "blocked"
    assert result["verify"]["verdict"] == "fail"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("level,verdict", [
    ("basic", "degraded"), ("standard", "degraded"), ("strict", "fail")])
def test_part_surface_tolerance_is_not_erased_by_assembly(level, verdict):
    built = assembly.build(doc([degraded_panel()]))
    receipt = assembly.verify(built, level=level)
    assert receipt["verdict"] == verdict
    assert any(c["id"] == "part.panel.surface_tolerance" and not c["ok"]
               for c in receipt["checks"])


@pytest.mark.parametrize("level,verdict", [
    ("basic", "degraded"), ("standard", "degraded"), ("strict", "fail")])
def test_inherited_surface_report_without_issue_is_checked(level, verdict):
    built = assembly.build(doc([box()]))
    built["part_meshes"]["base"].meta = {"geometry_provenance": {"sources": [
        {"nurbs": {"target_met": False, "sampled_error": 0.1, "tolerance": 0.01}}]}}
    receipt = assembly.verify(built, level=level)
    assert receipt["verdict"] == verdict
    assert any(c["id"] == "part.base.surface_tolerance" and not c["ok"]
               for c in receipt["checks"])


@pytest.mark.parametrize("severity,level,verdict", [
    ("warn", "standard", "degraded"), ("warn", "strict", "fail"),
    ("error", "standard", "fail")])
def test_part_generic_quality_issues_keep_severity(severity, level, verdict):
    built = assembly.build(doc([box()]))
    built["part_meshes"]["base"].meta["issues"] = [
        {"code": "source_budget", "severity": severity, "message": "source quality failure"}]
    receipt = assembly.verify(built, level=level)
    assert receipt["verdict"] == verdict
    assert any(c["id"].startswith("part.base.geometry_issue.") and not c["ok"]
               for c in receipt["checks"])


@pytest.mark.parametrize("field", ["gap_mm", "roll_deg", "axis", "radius_mm"])
def test_oversized_json_integers_raise_attaerror_in_direct_build(field):
    child = box("child")
    child.pop("transform")
    child["attach"] = {"target": "base", "target_port": "center", "source_port": "center"}
    if field in ("gap_mm", "roll_deg"):
        child["attach"][field] = 10 ** 400
    else:
        child["ports"] = [{"name": "dock", field: [10 ** 400, 0, 1]
                            if field == "axis" else 10 ** 400}]
    with pytest.raises(AttaError):
        assembly.build(doc([box(), child]))


def test_degraded_part_requires_explicit_delivery_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTA_HOME", str(tmp_path))
    built = engine.model(spec=doc([degraded_panel()]))
    blocked = engine.deliver(built["model_id"], formats=["glb"], outdir=tmp_path / "blocked")
    assert blocked["status"] == "blocked"
    assert blocked["verify"]["verdict"] == "degraded"
    allowed = engine.deliver(built["model_id"], formats=["glb"],
                             outdir=tmp_path / "allowed", allow_degraded=True)
    assert allowed["status"] == "ok"
    assert allowed["manifest"]["verify"]["verdict"] == "degraded"
    assert any(i["check"] == "part.panel.surface_tolerance"
               for i in allowed["manifest"]["verify"]["issues"])


@pytest.mark.parametrize("code", ["brep_build", "brep_unavailable", "bad_arg"])
@pytest.mark.parametrize("fail_all", [False, True])
def test_step_backend_errors_preserve_diagnostics_and_never_become_omissions(
        code, fail_all, tmp_path, monkeypatch):
    pytest.importorskip("build123d")
    built = assembly.build(doc([box("good"), box("bad", offset=30)]))
    original = brep.build_graph
    error = AttaError(code, "controlled backend failure", path="features.body",
                      fix="repair the backend", details={"sentinel": 17})
    def fail(spec):
        if fail_all or spec is built["parts"]["bad"].spec:
            raise error
        return original(spec)
    monkeypatch.setattr(brep, "build_graph", fail)
    result = assembly.export_scene(built, formats=["step"], outdir=tmp_path)
    assert result["files"] == []
    assert result["errors"] == [{"format": "step", "error": error.to_dict()}]
    assert not (tmp_path / "assembly.step").exists()


def test_engine_keeps_other_formats_when_step_backend_fails(tmp_path, monkeypatch):
    pytest.importorskip("build123d")
    monkeypatch.setenv("ATTA_HOME", str(tmp_path))
    built = engine.model(spec=doc([box()]))
    def fail(spec):
        raise AttaError("brep_build", "controlled OCCT failure", path="features.body",
                        fix="repair backend", details={"sentinel": 1})
    monkeypatch.setattr(brep, "build_graph", fail)
    result = engine.deliver(built["model_id"], formats=["glb", "step"], outdir=tmp_path / "out")
    assert result["status"] == "partial"
    assert [row["format"] for row in result["files"]] == ["glb"]
    assert result["errors"][0]["error"]["code"] == "brep_build"
    assert result["manifest"]["errors"] == result["errors"]


def test_all_mesh_only_step_retains_per_part_omissions(tmp_path):
    pytest.importorskip("build123d")
    part = box()
    part["graph"]["features"].append({"id": "sub", "op": "subdivide",
                                        "inputs": ["body"], "args": {"levels": 1}})
    built = assembly.build(doc([part]))
    result = assembly.export_scene(built, formats=["step"], outdir=tmp_path)
    assert result["files"] == []
    error = result["errors"][0]["error"]
    assert error["code"] == "brep_unsupported"
    omission = error["details"]["omitted"][0]
    assert omission["part"] == "base"
    assert omission["code"] == "brep_unsupported"
    assert omission["path"] and omission["fix"]


def test_mixed_step_is_partial_and_manifest_matches_real_file(tmp_path, monkeypatch):
    b = pytest.importorskip("build123d")
    monkeypatch.setenv("ATTA_HOME", str(tmp_path))
    mesh_only = box("mesh", offset=30)
    mesh_only["graph"]["features"].append({"id": "sub", "op": "subdivide",
        "inputs": ["body"], "args": {"levels": 1}})
    model = engine.model(spec=doc([box(), mesh_only]))
    result = engine.deliver(model["model_id"], formats=["step", "glb"], outdir=tmp_path / "out")
    assert result["status"] == "partial"
    row = next(f for f in result["files"] if f["format"] == "step")
    assert row["parts"] == ["base"]
    assert row["omitted"][0]["part"] == "mesh"
    assert row["omitted"][0]["code"] == "brep_unsupported"
    shape = b.import_step(row["path"])
    assert shape.is_valid and len(shape.solids()) == 1
    assert shape.volume == pytest.approx(1680)
    from pathlib import Path
    data = Path(row["path"]).read_bytes()
    assert row["bytes"] == len(data)
    assert row["sha256"] == hashlib.sha256(data).hexdigest()
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["files"] == result["files"]
    assert manifest["errors"] == [{"format": "step", **row["omitted"][0]}]


def test_attachment_chain_is_order_independent_with_independent_frame_oracle():
    root = box("z_root")
    root.pop("fixed")
    root["transform"] = {"translate_mm": [100, -20, 37], "rotate_deg": [23, -41, 72]}
    ports = [{"name": "dock", "origin_mm": [0, 0, 0],
              "axis": [0, 0, 1], "x_axis": [1, 0, 0]}]
    root["ports"] = ports
    parts = [root]
    for pid, target, gap, roll in [("middle", "z_root", 30, 37),
                                    ("a_leaf", "middle", 40, -19)]:
        part = box(pid)
        part.pop("fixed")
        part.pop("transform")
        part["ports"] = ports
        part["attach"] = {"target": target, "target_port": "dock", "source_port": "dock",
                          "facing": "same", "gap_mm": gap, "roll_deg": roll}
        parts.append(part)
    rotation = Rotation.from_euler("xyz", [23, -41, 72], degrees=True).as_matrix()
    expected_r = rotation @ Rotation.from_euler("z", 18, degrees=True).as_matrix()
    expected_t = np.array([100, -20, 37]) + rotation @ [0, 0, 70]
    for order in itertools.permutations(parts):
        built = assembly.build(doc(list(order)))
        assert built["placements"]["z_root"]["fixed"]
        assert built["assembly"]["attachments"]["order"] == ["middle", "a_leaf"]
        pose = built["parts"]["a_leaf"].pose
        np.testing.assert_allclose(pose.R, expected_r, atol=1e-12)
        np.testing.assert_allclose(pose.t * 1000, expected_t, atol=1e-10)
        assert np.linalg.det(pose.R) == pytest.approx(1)


@pytest.mark.parametrize("units,factor", [("mm", 1), ("cm", 10), ("m", 1000), ("in", 25.4)])
@pytest.mark.parametrize("facing", ["same", "opposite"])
def test_step_uses_final_solved_pose_original_graph_params_and_units(units, factor, facing, tmp_path):
    b = pytest.importorskip("build123d")
    root = box()
    root["transform"] = {"translate_mm": [100, -20, 37], "rotate_deg": [23, -41, 72]}
    root["ports"] = [{"name": "out", "origin_mm": [0, 0, 80],
                      "axis": [0, 0, 1], "x_axis": [1, 0, 0]}]
    child = {"id": "child", "graph": {"units": units, "params": {"w": 99},
        "output": "body", "features": [
            {"id": "unused", "op": "subdivide", "inputs": ["body"]},
            {"id": "body", "op": "box", "args": {
                "size": [{"$param": "w"}, {"$expr": "d*2"}, {"$param": "h"}],
                "center": [2/factor, 3/factor, 4/factor]}}]},
        "params": {"w": 4/factor, "d": 3/factor, "h": 8/factor},
        "ports": [{"name": "in", "origin_mm": [1, 2, 3],
                   "axis": [0, 0, 1], "x_axis": [1, 0, 0]}],
        "attach": {"target": "base", "target_port": "out", "source_port": "in",
                   "gap_mm": 4, "roll_deg": 37, "facing": facing}}
    raw = doc([child, root], mates=[{"id": "move", "type": "distance", "a": "base",
        "b": "child", "port_a": "out", "port_b": "in", "axis": "az", "value_mm": 17}])
    original = copy.deepcopy(raw)
    built = assembly.build(raw)
    assert raw == original
    assert built["assembly"]["mates"][0]["ok"]
    attachment = built["assembly"]["attachments"]["items"][0]
    assert attachment["final_position_error_mm"] >= 13 - 1e-4
    result = assembly.export_scene(built, formats=["step"], outdir=tmp_path)
    assert not result["errors"]
    shape = b.import_step(result["files"][0]["path"])
    assert shape.is_valid and len(shape.solids()) == 2
    solid = min(shape.solids(), key=lambda s: s.volume)
    assert solid.volume == pytest.approx(192, abs=1e-6)
    # Independent oracle: scipy Euler rotations and analytical box corners.
    rt = Rotation.from_euler("xyz", [23, -41, 72], degrees=True).as_matrix()
    turn = Rotation.from_euler("z", 37, degrees=True).as_matrix()
    if facing == "opposite":
        turn = turn @ Rotation.from_euler("x", 180, degrees=True).as_matrix()
    rotation = rt @ turn
    translation = np.array([100, -20, 37]) + rt @ [0, 0, 84] - rotation @ [1, 2, 3]
    initial = np.asarray(attachment["initial_matrix"])
    np.testing.assert_allclose(initial[:3, :3], rotation, atol=1e-12)
    np.testing.assert_allclose(initial[:3, 3], translation, atol=1e-10)
    # A distance mate leaves five DOFs: compare the export to the actual final
    # world mesh, not an invented unique final solution. Check the constrained
    # scalar independently, and confirm the final pose differs from attach.
    pose = built["parts"]["child"].pose
    source_origin = pose.R @ [1, 2, 3] + pose.t * 1000
    target_origin = np.array([100, -20, 37]) + rt @ [0, 0, 80]
    assert np.dot(source_origin - target_origin, rt[:, 2]) == pytest.approx(17, abs=1e-4)
    assert np.linalg.norm(pose.t * 1000 - initial[:3, 3]) > 1
    corners = np.array(list(itertools.product([-2, 2], [-3, 3], [-4, 4]))) + [2, 3, 4]
    expected = corners @ pose.R.T + pose.t * 1000
    world_mesh = built["part_meshes"]["child"].verts * 1000
    actual = np.array([tuple(v) for v in solid.vertices()])
    for target in (expected, world_mesh):
        assert np.max(np.min(np.linalg.norm(actual[:, None] - target, axis=2), axis=1)) < 1e-6
    assert tuple(solid.bounding_box().min) == pytest.approx(expected.min(axis=0), abs=1e-6)
    assert tuple(solid.bounding_box().max) == pytest.approx(expected.max(axis=0), abs=1e-6)
