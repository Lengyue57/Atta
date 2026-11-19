# -*- coding: utf-8 -*-
"""CADTestBench-style mutation tests proving verify rejects geometric defects."""

from __future__ import annotations

import numpy as np
import pytest

from atta import verify
from atta.features import evaluate
from atta.mesh import Mesh


def _reference_bracket() -> Mesh:
    """Closed 40x40x20 mm plate with a centered 12 mm diameter through-hole (genus=1)."""
    return evaluate([
        {"id": "plate", "op": "box", "args": {"size": [40, 40, 20]}},
        {"id": "bore", "op": "cylinder",
         "args": {"radius": 6, "height": 30, "center": [0, 0, 0]}},
        {"id": "part", "op": "difference", "inputs": ["plate", "bore"]},
    ])


def _hollow_frame(wall_mm: float = 4.0) -> Mesh:
    outer = 40.0
    inner = outer - 2.0 * float(wall_mm)
    return evaluate([
        {"id": "outer", "op": "box", "args": {"size": [outer, outer, outer]}},
        {"id": "cavity", "op": "box",
         "args": {"size": [inner, inner, inner], "center": [0, 0, 6]}},
        {"id": "part", "op": "difference", "inputs": ["outer", "cavity"]},
    ])


def test_unmutated_reference_passes_strict_and_mesh_agreement():
    ref = _reference_bracket()
    cons = {
        "closed": True,
        "genus": 1,
        "n_components": 1,
        "holes": [{"diameter": 12.0, "tolerance": 0.6, "count": 1}],
        "mesh_agreement": {
            "chamfer_max_mm": 0.05,
            "hausdorff_p95_max_mm": 0.1,
            "volumetric_iou_min": 0.99,
        },
    }
    receipt = verify.verify(ref, constraints=cons, level="strict",
                            expected_genus=1, reference_mesh=ref)
    assert receipt["verdict"] == "pass", receipt["issues"]
    assert receipt["pass"] is True


def test_mutation_shrink_wall_rejected():
    good = _hollow_frame(wall_mm=4.0)
    cons = {"closed": True, "min_wall_mm": 3.0}
    assert verify.verify(good, constraints=cons, level="strict")["verdict"] == "pass"

    # Mutation: shrink wall from 4.0 mm to 1.0 mm
    thin = _hollow_frame(wall_mm=1.0)
    std = verify.verify(thin, constraints=cons, level="standard")
    assert std["verdict"] == "degraded"
    assert std["pass"] is False
    assert any(c["id"] == "min_wall_mm" and not c["ok"] for c in std["checks"])

    strict = verify.verify(thin, constraints=cons, level="strict")
    assert strict["verdict"] == "fail"
    assert any(c["id"] == "min_wall_mm" and not c["ok"] for c in strict["checks"])


def test_mutation_delete_hole_rejected():
    ref = _reference_bracket()
    # Mutation: delete the through-hole (solid plate, genus=0 instead of 1)
    no_hole = evaluate([
        {"id": "plate", "op": "box", "args": {"size": [40, 40, 20]}},
    ])
    cons = {
        "closed": True,
        "genus": 1,
        "holes": [{"diameter": 12.0, "tolerance": 0.6, "count": 1}],
        "mesh_agreement": {
            "chamfer_max_mm": 0.5,
            "hausdorff_p95_max_mm": 1.0,
            "volumetric_iou_min": 0.95,
        },
    }
    receipt = verify.verify(no_hole, constraints=cons, level="standard",
                            expected_genus=1, reference_mesh=ref)
    assert receipt["verdict"] == "fail"
    failed_ids = {c["id"] for c in receipt["issues"]}
    assert "genus" in failed_ids
    assert "holes.0" in failed_ids
    assert "mesh_agreement.volumetric_iou" in failed_ids


def test_mutation_flip_normal_rejected():
    ref = _reference_bracket()
    # Global inside-out inversion (mesh.volume() uses abs(), so flux_enclosure_error catches it)
    inverted = Mesh(ref.verts.copy(), ref.faces[:, ::-1].copy())
    r_inv = verify.verify(inverted, level="standard")
    assert r_inv["verdict"] == "fail"
    failed_inv = {c["id"] for c in r_inv["issues"]}
    assert "flux_enclosure_error" in failed_inv
    assert r_inv["metrics"]["flux_enclosure_error"] == pytest.approx(2.0)

    # Single face normal flipped
    bad_faces = ref.faces.copy()
    bad_faces[0] = bad_faces[0, ::-1]
    single_flip = Mesh(ref.verts.copy(), bad_faces)
    r_one = verify.verify(single_flip, level="standard")
    assert r_one["verdict"] == "fail"
    failed_one = {c["id"] for c in r_one["issues"]}
    assert "flux_enclosure_error" in failed_one
    assert "winding" in failed_one


def test_mutation_self_intersection_rejected_and_override():
    # Closed single-component U-prism whose right arm bends across the left arm:
    # valid closed topology (genus=0) and flux_enclosure_error~0, only SIR fails!
    pts2d = np.array([
        [0.0, 0.0], [0.04, 0.0], [0.04, 0.015], [0.025, 0.015],
        [0.015, 0.015], [0.0, 0.015], [0.015, 0.04], [0.0, 0.04],
        [-0.005, 0.04], [-0.015, 0.04],
    ], dtype=np.float64)
    loop = [0, 1, 2, 8, 9, 3, 4, 6, 7, 5]
    tris_2d = [
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5],
        [5, 4, 6], [5, 6, 7], [3, 2, 8], [3, 8, 9],
    ]
    n2 = len(pts2d)
    v = np.vstack([
        np.column_stack([pts2d, np.zeros(n2)]),
        np.column_stack([pts2d, np.full(n2, 0.02)]),
    ])
    faces = []
    for a, b, c in tris_2d:
        faces.append([a, c, b])
        faces.append([a + n2, b + n2, c + n2])
    for k in range(len(loop)):
        a = loop[k]
        b = loop[(k + 1) % len(loop)]
        faces.append([a, b, b + n2])
        faces.append([a, b + n2, a + n2])
    bent_u = Mesh(v, np.array(faces, dtype=np.int64))

    std = verify.verify(bent_u, level="standard")
    assert std["verdict"] == "degraded"
    assert std["pass"] is False
    assert any(c["id"] == "self_intersection_ratio" and c["severity"] == "warn"
               and not c["ok"] for c in std["checks"])

    strict = verify.verify(bent_u, level="strict")
    assert strict["verdict"] == "fail"
    assert any(c["id"] == "self_intersection_ratio" and c["severity"] == "error"
               and not c["ok"] for c in strict["checks"])

    # Overridable via constraints["max_self_intersection_ratio"]
    relaxed = verify.verify(bent_u, constraints={"max_self_intersection_ratio": 1.0},
                            level="strict")
    sir_row = next(c for c in relaxed["checks"] if c["id"] == "self_intersection_ratio")
    assert sir_row["ok"] is True
    assert relaxed["verdict"] == "pass"


def test_mutation_part_interference_rejected():
    base = evaluate([{"id": "a", "op": "box",
                      "args": {"size": [20, 20, 20], "center": [-12, 0, 0]}}])
    mate_ok = evaluate([{"id": "b", "op": "box",
                         "args": {"size": [20, 20, 20], "center": [12, 0, 0]}}])
    cons = {"interference": [{"a": "base", "b": "mate", "max_mm3": 0.0}]}
    assert verify.verify(base, constraints=cons,
                         parts={"base": base, "mate": mate_ok})["verdict"] == "pass"

    # Mutation: shift mate into base so parts overlap by 6 mm
    mate_hit = evaluate([{"id": "b", "op": "box",
                          "args": {"size": [20, 20, 20], "center": [2, 0, 0]}}])
    bad = verify.verify(base, constraints=cons,
                        parts={"base": base, "mate": mate_hit})
    assert bad["verdict"] == "fail"
    assert any(c["id"] == "interference.0" and not c["ok"] for c in bad["issues"])


def test_mutation_open_boundary_and_mesh_agreement_edge_cases():
    ref = _reference_bracket()
    # Delete the entire top face (+Z) to create a large open boundary
    z_centroids = ref.verts[ref.faces][:, :, 2].mean(axis=1)
    open_faces = ref.faces[z_centroids < 0.0099]
    open_mesh = Mesh(ref.verts.copy(), open_faces.copy())
    r_open = verify.verify(open_mesh, level="standard")
    assert r_open["verdict"] == "fail"
    failed = {c["id"] for c in r_open["issues"]}
    assert "closed" in failed
    assert "flux_enclosure_error" in failed

    # Shifted mesh violates mesh_agreement chamfer & hausdorff
    shifted = ref.transform(translate=(0.015, 0.0, 0.0))
    cons = {"mesh_agreement": {"chamfer_max_mm": 1.0,
                               "hausdorff_p95_max_mm": 2.0,
                               "volumetric_iou_min": 0.8}}
    r_shift = verify.verify(shifted, constraints=cons, level="standard",
                            reference_mesh=ref)
    assert r_shift["verdict"] == "fail"
    shift_ids = {c["id"] for c in r_shift["issues"]}
    assert "mesh_agreement.chamfer_mm" in shift_ids
    assert "mesh_agreement.hausdorff_p95_mm" in shift_ids
    assert "mesh_agreement.volumetric_iou" in shift_ids

    # Basic level skips mesh_agreement explicitly instead of silently dropping
    r_basic = verify.verify(shifted, constraints=cons, level="basic",
                            reference_mesh=ref)
    ma_row = next(c for c in r_basic["checks"] if c["id"] == "mesh_agreement")
    assert ma_row["skipped"] is True

    # Stale atta_solve string must be replaced by atta_variant
    r_bbox = verify.verify(ref, constraints={"bbox": {"x": [1, 2]}}, level="basic")
    bbox_row = next(c for c in r_bbox["checks"] if c["id"] == "bbox.x")
    assert "atta_variant" in (bbox_row["fix"] or "")
    assert "atta_solve" not in (bbox_row["fix"] or "")


def test_malformed_new_constraints_never_crash():
    ref = _reference_bracket()
    for cons in (
        {"max_self_intersection_ratio": "bad"},
        {"max_self_intersection_ratio": 1.5},
        {"max_self_intersection_ratio": True},
        {"mesh_agreement": "not_an_object"},
        {"mesh_agreement": {"chamfer_max_mm": "far"}},
        {"mesh_agreement": {"volumetric_iou_min": 2.0}},
    ):
        r = verify.verify(ref, constraints=cons, reference_mesh=ref)
        assert r["verdict"] == "fail", cons
        assert any(c["severity"] == "error" and c["fix"] for c in r["issues"])
