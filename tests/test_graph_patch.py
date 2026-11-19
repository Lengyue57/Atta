# -*- coding: utf-8 -*-
"""Tests for P1-6 locate-then-infill patching, preflight lint, and v1.2.3beta schema."""

from __future__ import annotations

import copy

import pytest

from atta import spec as S


def _graph_spec():
    return {
        "schema": S.SCHEMA,
        "name": "bracket",
        "model": {
            "kind": "graph",
            "units": "mm",
            "segments": 64,
            "params": {"r_hole": 4.0},
            "output": "cut",
            "features": [
                {"id": "plate", "op": "box",
                 "args": {"size": [80, 50, 6], "center": [40, 25, 3]},
                 "tags": {"top": "deck"}},
                {"id": "hole", "op": "cylinder",
                 "args": {"radius": 4, "height": 20, "center": [15, 25, 3]}},
                {"id": "cut", "op": "difference",
                 "args": {}, "inputs": ["plate", "hole"]},
            ],
        },
        "quality": "draft",
        "material": "steel",
        "constraints": {},
        "assumptions": [],
    }


def _assembly_spec():
    return {
        "schema": S.SCHEMA,
        "name": "pin_on_plate",
        "model": {
            "kind": "assembly",
            "parts": [
                {
                    "id": "plate",
                    "fixed": True,
                    "ports": [{"name": "bore", "origin_mm": [0, 0, 5], "axis": [0, 0, 1]}],
                    "graph": {
                        "output": "plate_box",
                        "features": [
                            {"id": "plate_box", "op": "box",
                             "args": {"size": [80, 60, 10], "center": [0, 0, 5]}},
                        ],
                    },
                },
                {
                    "id": "pin",
                    "ports": [{"name": "shank", "origin_mm": [0, 0, 0], "axis": [0, 0, 1]}],
                    "graph": {
                        "output": "pin_cyl",
                        "features": [
                            {"id": "pin_cyl", "op": "cylinder",
                             "args": {"radius": 5, "height": 30, "center": [0, 0, 15]}},
                        ],
                    },
                },
            ],
            "mates": [
                {"id": "m1", "type": "concentric", "a": "plate", "b": "pin",
                 "port_a": "bore", "port_b": "shank"},
            ],
        },
        "quality": "draft",
        "material": "steel",
    }


def _codes(res):
    return [e["code"] for e in (res.get("errors") or [])]


# ------------------------------------------------------- schema extensions

def test_validate_accepts_segments_and_tags():
    sketch_extrude = {
        "schema": S.SCHEMA,
        "name": "sketch_plate",
        "model": {
            "kind": "graph",
            "output": "round",
            "features": [
                {
                    "id": "body",
                    "op": "extrude",
                    "args": {
                        "height": 10,
                        "segments": [
                            {"type": "line", "to": [20, 0], "tag": "front"},
                            {"type": "arc", "to": [20, 20], "via": [24, 10], "tag": "fillet_wall"},
                            {"type": "tangent_arc", "to": [0, 20]},
                            {"type": "line", "to": [0, 0], "tag": "left"},
                        ],
                        "tags": {"start": "bottom", "end": "top"},
                    },
                },
                {
                    "id": "round",
                    "op": "fillet",
                    "inputs": ["body"],
                    "args": {
                        "radius": 1.0,
                        "edges": {
                            "common_edge_of": ["top", "front"],
                            "from_feature": "body",
                        },
                    },
                },
            ],
        },
    }
    out = S.validate_spec(sketch_extrude)
    assert out["ok"], out["errors"]

    sketch_revolve = {
        "schema": S.SCHEMA,
        "name": "sketch_revolve",
        "model": {
            "kind": "graph",
            "features": [
                {
                    "id": "rev",
                    "op": "revolve",
                    "args": {
                        "segments": [
                            {"type": "line", "to": [10, 0]},
                            {"type": "line", "to": [10, 20]},
                            {"type": "line", "to": [0, 20]},
                        ],
                        "tags": {"start": "base"},
                    },
                },
            ],
        },
    }
    assert S.validate_spec(sketch_revolve)["ok"]
    prop = S.propose_spec(sketch_revolve)
    assert prop["ok"], (prop["errors"], prop["unresolved"])


def test_validate_accepts_tags_on_all_primitives_and_on_face():
    for op, args in [
        ("box", {"size": [10, 10, 10]}),
        ("cylinder", {"radius": 5, "height": 10}),
        ("sphere", {"radius": 5}),
        ("cone", {"r_bottom": 5, "r_top": 2, "height": 10}),
        ("torus", {"major_radius": 10, "minor_radius": 2}),
    ]:
        spec = {
            "schema": S.SCHEMA,
            "name": "prim_" + op,
            "model": {
                "kind": "graph",
                "output": "ed",
                "features": [
                    {"id": "p", "op": op, "args": {**args, "tags": {"end": "cap"}}},
                    {"id": "ed", "op": "chamfer", "inputs": ["p"],
                     "args": {"length": 0.5, "edges": {"on_face": "cap"}}},
                ],
            },
        }
        res = S.validate_spec(spec)
        assert res["ok"], (op, res["errors"])


def test_validate_rejects_bad_segments_tags_and_selectors():
    bad_seg = _graph_spec()
    bad_seg["model"]["features"] = [
        {"id": "e", "op": "extrude", "args": {"height": 5, "segments": [{"type": "nurbs_bad"}]}},
    ]
    bad_seg["model"]["output"] = "e"
    assert not S.validate_spec(bad_seg)["ok"]

    bad_sel = _graph_spec()
    bad_sel["model"]["features"].append({
        "id": "f", "op": "fillet", "inputs": ["cut"],
        "args": {"radius": 1.0, "edges": {"common_edge_of": ["only_one"]}},
    })
    bad_sel["model"]["output"] = "f"
    assert "bad_selector" in _codes(S.validate_spec(bad_sel))

    bad_from = _graph_spec()
    bad_from["model"]["features"].append({
        "id": "f", "op": "fillet", "inputs": ["cut"],
        "args": {"radius": 1.0, "edges": {"on_face": "deck", "from_feature": "missing_feat"}},
    })
    bad_from["model"]["output"] = "f"
    assert "unknown_input" in _codes(S.validate_spec(bad_from))


def test_validate_mesh_agreement_and_self_intersection_constraints():
    spec = _graph_spec()
    spec["constraints"] = {
        "mesh_agreement": {
            "target_model_id": "mdl_ref_123",
            "chamfer_max_mm": 0.5,
            "hausdorff_p95_max_mm": 1.2,
            "volumetric_iou_min": 0.92,
        },
        "max_self_intersection_ratio": 0.01,
    }
    assert S.validate_spec(spec)["ok"]

    bad = copy.deepcopy(spec)
    bad["constraints"]["mesh_agreement"]["volumetric_iou_min"] = 1.5
    bad["constraints"]["max_self_intersection_ratio"] = -0.1
    res = S.validate_spec(bad)
    assert not res["ok"]
    assert res["errors"].count(next(e for e in res["errors"] if e["code"] == "bad_constraint")) >= 1
    assert len([e for e in res["errors"] if e["code"] == "bad_constraint"]) == 2


# ------------------------------------------------------- patch_spec

def test_patch_spec_recipe_scalar_and_params_dict():
    spec = {
        "schema": S.SCHEMA,
        "name": "mug",
        "model": {"kind": "recipe", "id": "product.mug", "params": {"height": 95}},
    }
    res = S.patch_spec(spec, {"height": 115})
    assert res["ok"], res["errors"]
    assert res["spec"]["model"]["params"]["height"] == 115.0
    assert res["delta"]["changed"] is True
    assert res["delta"]["params"]["height"]["after"] == 115.0
    assert spec["model"]["params"]["height"] == 95  # input unchanged

    res2 = S.patch_spec(spec, {"params": {"height": 120}, "material": "porcelain"})
    assert res2["ok"], res2["errors"]
    assert res2["spec"]["model"]["params"]["height"] == 120.0
    assert res2["spec"]["material"] == "porcelain"


def test_patch_spec_graph_locate_then_infill_preserves_siblings():
    orig = _graph_spec()
    snapshot = copy.deepcopy(orig)

    res = S.patch_spec(orig, {
        "features": {
            "hole": {
                "args": {"radius": 8},
                "tags": {"side": "bore"},
            },
        },
        "params": {"r_hole": 8.0},
    })
    assert res["ok"], res["errors"]
    assert orig == snapshot  # non-mutating

    feats = {f["id"]: f for f in res["spec"]["model"]["features"]}
    assert feats["hole"]["args"]["radius"] == 8
    assert feats["hole"]["args"]["height"] == 20  # untouched arg preserved
    assert feats["hole"]["args"]["center"] == [15, 25, 3]
    assert feats["hole"]["tags"] == {"side": "bore"}
    # Sibling nodes are untouched
    normalized_orig = S.validate_spec(snapshot)["spec"]["model"]["features"]
    assert feats["plate"] == normalized_orig[0]
    assert feats["cut"] == normalized_orig[2]
    assert res["spec"]["model"]["params"]["r_hole"] == 8.0
    assert res["delta"]["changed"] is True
    assert "hole" in res["delta"]["modified_features"]


def test_patch_spec_graph_add_remove_and_output():
    orig = _graph_spec()
    res = S.patch_spec(orig, {
        "add_features": [
            {"id": "hole2", "op": "cylinder",
             "args": {"radius": 3, "height": 20, "center": [65, 25, 3]}},
            {"id": "cut2", "op": "difference",
             "args": {}, "inputs": ["cut", "hole2"]},
        ],
        "output": "cut2",
    })
    assert res["ok"], res["errors"]
    assert res["spec"]["model"]["output"] == "cut2"
    assert [f["id"] for f in res["spec"]["model"]["features"]] == [
        "plate", "hole", "cut", "hole2", "cut2",
    ]
    assert res["delta"]["added_features"] == ["hole2", "cut2"]

    # Remove cut2 and revert output
    res2 = S.patch_spec(res["spec"], {
        "remove_features": ["cut2", "hole2"],
        "output": "cut",
    })
    assert res2["ok"], res2["errors"]
    assert res2["spec"]["model"]["output"] == "cut"
    assert [f["id"] for f in res2["spec"]["model"]["features"]] == ["plate", "hole", "cut"]


def test_patch_spec_graph_unknown_target_fails_explicitly():
    orig = _graph_spec()
    res = S.patch_spec(orig, {"features": {"ghost": {"args": {"radius": 5}}}})
    assert not res["ok"]
    assert "unknown_feature_id" in _codes(res)

    res_rem = S.patch_spec(orig, {"remove_features": ["ghost"]})
    assert not res_rem["ok"]
    assert "unknown_feature_id" in _codes(res_rem)

    res_dup = S.patch_spec(orig, {
        "add_features": [{"id": "plate", "op": "box", "args": {"size": [1, 1, 1]}}],
    })
    assert not res_dup["ok"]
    assert "duplicate_id" in _codes(res_dup)


def test_patch_spec_assembly_parts_and_unknown_part_error():
    asm = _assembly_spec()
    res = S.patch_spec(asm, {
        "parts": {
            "pin": {
                "material": "aluminum",
                "graph": {
                    "features": {
                        "pin_cyl": {"args": {"radius": 8}},
                    },
                },
            },
        },
    })
    assert res["ok"], res["errors"]
    parts = {p["id"]: p for p in res["spec"]["model"]["parts"]}
    assert parts["pin"]["material"] == "aluminum"
    assert parts["pin"]["graph"]["features"][0]["args"]["radius"] == 8
    assert parts["pin"]["graph"]["features"][0]["args"]["height"] == 30
    normalized_asm = S.validate_spec(asm)["spec"]["model"]["parts"]
    assert parts["plate"]["graph"] == normalized_asm[0]["graph"]

    bad = S.patch_spec(asm, {"parts": {"missing_part": {"material": "brass"}}})
    assert not bad["ok"]
    assert "unknown_part_id" in _codes(bad)

    bad_rem = S.patch_spec(asm, {"remove_parts": ["missing_part"]})
    assert not bad_rem["ok"]
    assert "unknown_part_id" in _codes(bad_rem)


# ------------------------------------------------------- preflight_spec

def test_preflight_spec_lints_graph_and_assembly():
    good = _graph_spec()
    pf = S.preflight_spec(good)
    assert pf["ok"] is True
    assert pf["checks"]["schema"] is True
    assert pf["checks"]["dependencies"] is True
    assert pf["checks"]["patchability"] is True
    assert pf["summary"]["patchable_targets"] == ["plate", "hole", "cut"]

    # Graph with unreachable node, missing $param, tiny unit scale, unresolved selector tag
    linty = {
        "schema": S.SCHEMA,
        "name": "linty",
        "model": {
            "kind": "graph",
            "params": {},
            "features": [
                {"id": "b", "op": "box", "args": {"size": [0.005, 10, 10]}},
                {"id": "dead", "op": "box", "args": {"size": [{"$param": "w"}, 5, 5]}},
                {"id": "f", "op": "fillet", "inputs": ["b"],
                 "args": {"radius": 0.5, "edges": {"on_face": "missing_face"}}},
            ],
        },
    }
    pf2 = S.preflight_spec(linty)
    assert pf2["ok"] is True  # warnings only
    w_codes = {w["code"] for w in pf2["warnings"]}
    assert {
        "implicit_output",
        "unreachable_feature",
        "missing_graph_param",
        "suspicious_unit_scale",
        "unresolved_selector_tag",
    } <= w_codes

    # Assembly unknown mate port fails preflight port check
    asm_bad_port = _assembly_spec()
    asm_bad_port["model"]["mates"][0]["port_b"] = "nonexistent_port"
    pf3 = S.preflight_spec(asm_bad_port)
    assert pf3["ok"] is False
    assert pf3["checks"]["ports"] is False
    assert "unknown_port" in _codes(pf3)

    # Assembly unconstrained part warns
    asm_float = _assembly_spec()
    asm_float["model"]["mates"] = []
    pf4 = S.preflight_spec(asm_float)
    assert pf4["ok"] is True
    assert any(w["code"] == "unconstrained_part" for w in pf4["warnings"])
