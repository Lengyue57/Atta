"""Oriented module ports, deterministic initialization, and separate gates."""
from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest

from atta import assembly, engine
from atta import spec as S
from atta.errors import AttaError


def box(pid, *, fixed=False, ports=None, attach=None, transform=None):
    part = {"id": pid, "fixed": fixed, "graph": {"features": [
        {"id": "body", "op": "box", "args": {"size": [4, 6, 8]}}]},
        "ports": ports or []}
    if attach is not None:
        part["attach"] = attach
    if transform is not None:
        part["transform"] = transform
    return part


def port(name, origin=(0, 0, 0), axis=(0, 0, 1), x=(1, 0, 0)):
    return {"name": name, "origin_mm": list(origin), "axis": list(axis), "x_axis": list(x)}


def attach(target, source="in", destination="out", **kwargs):
    return {"target": target, "source_port": source, "target_port": destination, **kwargs}


def doc(parts, mates=None):
    return {"model": {"kind": "assembly", "parts": parts, "mates": mates or []},
            "material": "steel"}


def build(raw, **kwargs):
    validated = S.validate_spec(raw)
    assert validated["ok"], validated["errors"]
    return assembly.build(validated["spec"], **kwargs)


@pytest.mark.parametrize("facing", ["same", "opposite"])
@pytest.mark.parametrize("source_axis", [(0, 0, 1), (0, 0, -1), (1, 2, -3)])
def test_attach_frames_preserve_roll_and_follow_world_transform(facing, source_axis):
    a = box("base", fixed=True, ports=[port("out", (7, -3, 11), (1, -2, 3), (3, 2, 1))],
            transform={"translate_mm": [100, -25, 47], "rotate_deg": [23, -41, 72]})
    b = box("module", ports=[port("in", (3, 5, -4), source_axis, (2, -1, 0))],
            attach=attach("base", gap_mm=2.75, roll_deg=37, facing=facing))
    built = build(doc([b, a]), solve=False)
    target = assembly._frame_for(built["parts"]["base"], "out")
    source = assembly._frame_for(built["parts"]["module"], "in")
    turn = assembly.euler_xyz_matrix([0, 0, 37])
    if facing == "opposite":
        turn = turn @ np.diag([1, -1, -1])
    assert source.origin == pytest.approx(target.origin + target.axis * 0.00275, abs=1e-12)
    assert source.axes == pytest.approx(target.axes @ turn, abs=1e-12)
    assert np.linalg.det(built["parts"]["module"].pose.R) == pytest.approx(1)
    item = built["assembly"]["attachments"]["items"][0]
    assert item["final_position_error_mm"] < 1e-9
    assert item["final_rotation_error_deg"] < 1e-9
    assert item["initial_source_frame"]["origin_mm"] == pytest.approx(source.origin * 1000)


def test_tangent_alias_normal_inheritance_and_legacy_port_compatibility():
    p = box("p", fixed=True, ports=[
        {"name": "oriented", "normal": [0, 0, -2], "tangent": [2, 0, 3]},
        {"name": "legacy", "normal": [0, 0, -2]}])
    raw = doc([p])
    validated = S.validate_spec(raw)
    assert validated["ok"], validated["errors"]
    normalized = validated["spec"]["model"]["parts"][0]["ports"][0]
    assert "tangent" not in normalized
    assert normalized["x_axis"] == pytest.approx([1, 0, 0])
    built = assembly.build(validated["spec"])
    oriented = assembly._frame_for(built["parts"]["p"], "oriented")
    legacy = assembly._frame_for(built["parts"]["p"], "legacy")
    assert oriented.axis == pytest.approx([0, 0, -1])
    assert oriented.axes[:, 0] == pytest.approx([1, 0, 0])
    assert legacy.axis == pytest.approx([0, 0, 1])
    assert legacy.normal == pytest.approx([0, 0, -1])
    assert legacy.axes == pytest.approx(assembly._basis([0, 0, 1]))


@pytest.mark.parametrize("bad", [
    {"axis": [0, 0, 0]}, {"x_axis": [0, 0, 1]}, {"x_axis": [1e-12, 0, 1]},
    {"x_axis": [float("nan"), 0, 0]}, {"x_axis": [True, 0, 0]},
    {"normal": [0, 0, -1]}, {"axis": [1, 0, 0]},
    {"tangent": [0, 1, 0]}, {"origin_mm": [0, float("inf"), 0]},
    {"radius_mm": float("nan")},
])
def test_bad_frames_rejected_at_schema_and_direct_build(bad):
    p = port("out")
    p.update(bad)
    raw = doc([box("base", fixed=True, ports=[p])])
    assert not S.validate_spec(raw)["ok"]
    with pytest.raises(AttaError):
        assembly.build(raw)


def test_dag_forward_references_list_independent_and_mm_matrix_roundtrip():
    a = box("z_root", ports=[port("out", (0, 0, 20))],
            transform={"translate_mm": [100, -5, 2], "rotate_deg": [10, 20, 30]})
    b = box("middle", ports=[port("in"), port("out", (0, 0, 20))],
            attach=attach("z_root", gap_mm=2, roll_deg=45, facing="same"))
    c = box("a_leaf", ports=[port("in")], attach=attach("middle", facing="same"))
    reference = None
    for parts in itertools.permutations([a, b, c]):
        built = build(doc(list(parts)))
        if reference is None:
            reference = built
        for pid in ("z_root", "middle", "a_leaf"):
            assert built["placements"][pid]["matrix"] == reference["placements"][pid]["matrix"]
        assert built["placements"]["z_root"]["fixed"] is True
        assert built["assembly"]["attachments"]["order"] == ["middle", "a_leaf"]
    baked = copy.deepcopy([a, b, c])
    for part in baked:
        part.pop("attach", None)
        part["transform"] = reference["placements"][part["id"]]
    rebuilt = build(doc(baked))
    for pid, mesh in reference["part_meshes"].items():
        assert rebuilt["part_meshes"][pid].verts == pytest.approx(mesh.verts, abs=1e-11)


@pytest.mark.parametrize("mutation, code", [
    ({"target": "missing"}, "unknown_part"), ({"source_port": "missing"}, "unknown_port"),
    ({"target_port": "missing"}, "unknown_port"), ({"source_port": ""}, "bad_port_ref"),
    ({"gap_mm": float("inf")}, "bad_attach"), ({"roll_deg": True}, "bad_attach"),
    ({"facing": "flip"}, "bad_attach"), ({"unexpected": 3}, "bad_attach"),
])
def test_bad_attach_references_and_values(mutation, code):
    config = attach("base")
    config.update(mutation)
    raw = doc([box("base", ports=[port("out")]),
               box("child", ports=[port("in")], attach=config)])
    validated = S.validate_spec(raw)
    assert any(e["code"] == code for e in validated["errors"])
    with pytest.raises(AttaError) as exc:
        assembly.build(raw)
    assert exc.value.code == code


@pytest.mark.parametrize("self_cycle", [True, False])
def test_attach_cycles_are_explicit(self_cycle):
    a = box("a", ports=[port("in"), port("out")], attach=attach("a" if self_cycle else "b"))
    b = box("b", ports=[port("in"), port("out")], attach=attach("a"))
    raw = doc([a, b])
    assert any(e["code"] == "attach_cycle" for e in S.validate_spec(raw)["errors"])
    with pytest.raises(AttaError) as exc:
        assembly.build(raw)
    assert exc.value.code == "attach_cycle"


def test_attach_transform_conflict_and_propose_does_not_invent_transform():
    raw = doc([box("base", ports=[port("out")]),
               box("child", ports=[port("in")], attach=attach("base"))])
    proposed = S.propose_spec(raw)
    assert not proposed["errors"]
    assert "transform" not in proposed["spec"]["model"]["parts"][1]
    for part in raw["model"]["parts"]:
        part.pop("fixed", None)
    raw["model"]["parts"].reverse()
    proposed = S.propose_spec(raw)
    fixed = {p["id"]: p["fixed"] for p in proposed["spec"]["model"]["parts"]}
    assert fixed == {"base": True, "child": False}
    raw["model"]["parts"].reverse()
    raw["model"]["parts"][1]["transform"] = {}
    assert any(e["code"] == "bad_attach" for e in S.validate_spec(raw)["errors"])


def test_attach_composes_with_lock_and_collision_remains_separate():
    base = box("base", fixed=True, ports=[port("out", (0, 0, 20))])
    child = box("child", ports=[port("in")], attach=attach("base", gap_mm=3, roll_deg=22))
    mates = [{"id": "hold", "type": "lock", "a": "base", "b": "child",
              "port_a": "out", "port_b": "in"}]
    built = build(doc([child, base], mates))
    assert built["assembly"]["dof"]["dof"] == 0
    assert built["assembly"]["attachments"]["items"][0]["final_rotation_error_deg"] < 1e-7
    assert assembly.verify(built, constraints={"fully_constrained": True})["verdict"] == "pass"
    child["attach"]["gap_mm"] = -20
    overlap = build(doc([base, child], mates))
    assert overlap["assembly"]["attachments"]["initialization_only"] is True
    assert overlap["assembly"]["collision"]["has_collision"] is True
    assert assembly.verify(overlap)["verdict"] == "fail"


def test_attach_is_initialization_mates_can_move_the_part():
    base = box("base", fixed=True, ports=[port("out", (0, 0, 20))])
    child = box("child", ports=[port("in")], attach=attach("base", facing="same"))
    built = build(doc([base, child], [{"id": "move", "type": "distance", "a": "base",
        "b": "child", "port_a": "out", "port_b": "in", "axis": "az", "value_mm": 5}]))
    item = built["assembly"]["attachments"]["items"][0]
    assert item["final_position_error_mm"] == pytest.approx(5, abs=1e-4)
    assert built["assembly"]["mates"][0]["ok"]
    assert built["assembly"]["dof"]["dof"] == 5


def test_recipe_module_and_bbox_ports_compile_and_attach():
    base = box("base", fixed=True)
    child = {"id": "recipe", "recipe": "primitive.chamfer_box",
             "params": {"length": 20, "width": 12, "height": 4,
                        "chamfer_w": 1, "chamfer_w_v": 1},
             "attach": attach("base", source="face_z-", destination="face_z+", gap_mm=2)}
    raw = doc([child, base])
    compiled = S.compile_spec(raw)
    assert compiled["ok"], compiled["errors"]
    assert compiled["compiled"]["parts"][0]["attach"] == child["attach"]
    built = build(raw)
    source = assembly._frame_for(built["parts"]["recipe"], "face_z-")
    target = assembly._frame_for(built["parts"]["base"], "face_z+")
    assert source.origin == pytest.approx(target.origin + target.axis * 0.002)
    assert assembly.verify(built)["verdict"] == "pass"


def test_compile_rejects_blocked_recipe_part_instead_of_dropping_it():
    base = box("base", fixed=True)
    child = {"id": "recipe", "recipe": "primitive.chamfer_box",
             "params": {"length": 20, "width": 12, "height": 4,
                        "chamfer_w": 3, "chamfer_w_v": 4},
             "attach": attach("base", source="face_z-", destination="face_z+")}
    compiled = S.compile_spec(doc([child, base]))
    assert not compiled["ok"]
    assert compiled["compiled"] is None
    assert any(e["code"] == "assembly_part_blocked" for e in compiled["errors"])


@pytest.mark.parametrize("bad_ports", [7, {}, "invalid"])
def test_malformed_port_container_reports_error_with_attach(bad_ports):
    a = box("base", fixed=True)
    a["ports"] = bad_ports
    b = box("child", attach=attach("base", source="center", destination="center"))
    raw = doc([a, b])
    assert any(e["code"] == "bad_ports" for e in S.validate_spec(raw)["errors"])
    with pytest.raises(AttaError) as exc:
        assembly.build(raw)
    assert exc.value.code == "bad_ports"


def test_nurbs_graph_part_thickens_attaches_and_engine_describes(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTA_HOME", str(tmp_path))
    base = box("base", fixed=True, ports=[port("out", (0, 0, 30))])
    panel = {"id": "panel", "graph": {"features": [
        {"id": "skin", "op": "nurbs_surface", "args": {"control_points": [
            [[0, 0, 0], [0, 20, 0]], [[30, 0, 0], [30, 20, 0]]]}},
        {"id": "solid", "op": "thicken", "inputs": ["skin"], "args": {"thickness": 2}}]},
        "ports": [port("in")], "attach": attach("base", facing="same")}
    raw = doc([panel, base])
    built = build(raw)
    assert built["part_meshes"]["panel"].topology()["closed"]
    assert built["assembly"]["parts"][1]["metrics"]["bbox_mm"]["size"] == pytest.approx([30, 20, 2])
    assert assembly.verify(built)["verdict"] == "pass"
    modeled = engine.model(spec=raw)
    described = engine.describe(modeled["model_id"])
    assert described["assembly"]["attachments"]["order"] == ["panel"]
    assert engine.verify(modeled["model_id"])["verdict"] == "pass"


def test_assembly_step_exports_exact_parts_and_reports_mesh_only_omissions(tmp_path):
    occt = pytest.importorskip("build123d")
    exact = {"id": "base", "fixed": True, "graph": {"units": "mm", "features": [
        {"id": "body", "op": "box", "args": {"size": [30, 20, 10]}},
        {"id": "round", "op": "fillet", "inputs": ["body"],
         "args": {"radius": 2, "edges": {"parallel": "z", "kind": "line",
                                          "expect_count": 4}}}]},
        "ports": [port("out", (0, 0, 10))]}
    panel = {"id": "panel", "graph": {"features": [
        {"id": "skin", "op": "nurbs_surface", "args": {"control_points": [
            [[0, 0, 0], [0, 20, 0]], [[30, 0, 0], [30, 20, 0]]]}},
        {"id": "solid", "op": "thicken", "inputs": ["skin"], "args": {"thickness": 2}}]},
        "ports": [port("in")], "attach": attach("base", facing="same", gap_mm=1)}
    built = build(doc([panel, exact]))
    exported = assembly.export_scene(built, formats=["step"], outdir=tmp_path, stem="mixed")
    step = [f for f in exported["files"] if f["format"] == "step"][0]
    assert step["ok"] and step["parts"] == ["base"]
    assert any(row.get("part") == "panel" and row.get("code") == "brep_unsupported"
               for row in exported["errors"])
    imported = occt.import_step(step["path"])
    assert imported.is_valid and len(imported.solids()) == 1
    analytic = built["part_meshes"]["base"].meta["analytic_volume_mm3"]
    assert imported.volume == pytest.approx(analytic, rel=1e-6)
