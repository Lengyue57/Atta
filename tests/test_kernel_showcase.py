"""Public-engine regression for the exact CAD / NURBS / field assembly example."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from atta import engine


SPEC = Path(__file__).resolve().parents[1] / "examples/kernel_showcase/spec.json"


@pytest.fixture(scope="module")
def showcase(tmp_path_factory):
    pytest.importorskip("build123d")
    home = tmp_path_factory.mktemp("kernel-showcase")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ATTA_HOME", str(home))
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
        valid = engine.spec(spec, op="validate")
        assert valid["ok"], valid
        built = engine.model(spec=spec, quality="draft")
        assert built["status"] == "ok", built
        mid = built["model_id"]
        description = engine.describe(mid)
        receipt = engine.verify(mid, constraints={
            "fully_constrained": True,
            "bbox": {"x": [83.99, 84.01], "y": [39.99, 40.01],
                     "z": [24.99, 25.01]},
        })
        delivery = engine.deliver(mid, formats=["step", "glb"],
                                  outdir=home / "deliver")
        yield home, spec, built, description, receipt, delivery


def test_showcase_geometry_and_receipts(showcase):
    _, _, built, description, receipt, _ = showcase
    assert receipt["verdict"] == "pass", receipt["issues"]
    assert receipt["mesh_hash"] == description["model"]["mesh_hash"]
    assert receipt["spec_hash"] == description["model"]["spec_hash"]
    assert receipt["model_id"] == built["model_id"]
    assert description["facts"]["closed"] is True
    assert description["facts"]["n_components"] == 4
    assert description["facts"]["bbox_mm"]["size"] == pytest.approx([84, 40, 25])
    assembly = description["assembly"]
    assert assembly["dof"]["dof"] == 0
    assert assembly["attachments"]["order"] == ["lattice", "panel", "cap"]
    for item in assembly["attachments"]["items"]:
        assert item["final_position_error_mm"] < 1e-7
        assert item["final_rotation_error_deg"] < 1e-5
    parts = {part["part_id"]: part for part in description["geometry"]["parts"]}
    assert set(parts) == {"base", "panel", "cap", "lattice"}
    for name in ("base", "cap"):
        assert parts[name]["backend"] == "brep"
        assert parts[name]["source_sha256"]
        assert parts[name]["tessellation"]["exact"] is False
        assert parts[name]["tessellation"]["strict_surface_error_bound"] is False
    assert parts["panel"]["nurbs"]["target_met"] is True
    assert parts["panel"]["nurbs"]["surface_error_bound"] is None
    assert parts["lattice"]["field"]["distance_exact"] is False
    assert parts["lattice"]["sampling"]["coarsened"] is False


def test_showcase_delivery_roundtrip_and_manifest(showcase):
    import build123d as bd

    _, _, _, description, receipt, delivery = showcase
    assert delivery["status"] == "partial"
    assert delivery["verify"]["verdict"] == "pass"
    files = {row["format"]: row for row in delivery["files"]}
    assert set(files) == {"step", "glb"}
    assert files["step"]["parts"] == ["base", "cap"]
    assert {(row["part"], row["code"]) for row in files["step"]["omitted"]} == {
        ("panel", "brep_unsupported"), ("lattice", "brep_unsupported")}
    assert {(row["format"], row["part"], row["code"])
            for row in delivery["errors"]} == {
        ("step", "panel", "brep_unsupported"),
        ("step", "lattice", "brep_unsupported")}
    manifest = json.loads(Path(delivery["manifest_path"]).read_text())
    assert manifest["spec_hash"] == receipt["spec_hash"]
    assert manifest["mesh_hash"] == receipt["mesh_hash"]
    assert manifest["errors"] == delivery["errors"]
    assert manifest["verify"]["verdict"] == "pass"
    for row in files.values():
        data = Path(row["path"]).read_bytes()
        assert row["ok"] is True
        assert row["bytes"] == len(data)
        assert row["sha256"] == hashlib.sha256(data).hexdigest()

    shape = bd.import_step(files["step"]["path"])
    assert shape.is_valid
    solids = sorted(shape.solids(), key=lambda solid: solid.volume, reverse=True)
    assert len(solids) == 2
    analytic = {part["part_id"]: part["analytic_volume_mm3"]
                for part in description["geometry"]["parts"]
                if part["part_id"] in {"base", "cap"}}
    # Analytic rounded-rectangle extrusion and four triangular corner cuts.
    assert analytic["base"] == pytest.approx((60 * 40 - (4 - np.pi) * 3**2) * 10)
    assert analytic["cap"] == pytest.approx((20 * 10 - 4 * 0.5) * 4)
    for solid, name, lower, upper in (
        (solids[0], "base", [-30, -20, -5], [30, 20, 5]),
        (solids[1], "cap", [-20, -10, 12], [0, 0, 16]),
    ):
        assert solid.volume == pytest.approx(analytic[name], rel=1e-7)
        assert tuple(solid.bounding_box().min) == pytest.approx(lower, abs=1e-5)
        assert tuple(solid.bounding_box().max) == pytest.approx(upper, abs=1e-5)

    scene = trimesh.load(files["glb"]["path"], force="scene", process=False)
    assert len(scene.graph.nodes_geometry) == 4
    assert len(scene.geometry) == 4
    assert all(mesh.is_watertight for mesh in scene.geometry.values())
    assert scene.bounds * 1000 == pytest.approx(
        np.array([[-30, -20, -5], [54, 20, 20]]), abs=1e-4)


def test_showcase_delivery_respects_failed_verification(showcase):
    home, spec, _, _, _, _ = showcase
    bad = dict(spec, constraints={"bbox": {"x": [200, 210]}})
    built = engine.model(spec=bad, quality="draft")
    delivery = engine.deliver(built["model_id"], formats=["step", "glb"],
                              outdir=home / "blocked")
    assert delivery["status"] == "blocked"
    assert delivery["verify"]["verdict"] == "fail"
    assert not delivery.get("files")
    assert not (home / "blocked").exists()
