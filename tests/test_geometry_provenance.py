"""Geometry edits preserve source quality without inventing result error bounds."""
import copy
import json
import math

import pytest

from atta import boolean as B, fieldtree
from atta.errors import AttaError
from atta.features import evaluate
from atta.mesh import concat
from atta.primitives import box
from atta.verify import verify


def solid():
    return box([.04, .04, .04])


def warning(mesh, severity="warn"):
    mesh.meta["issues"] = [{"code": "source_quality_unmet", "severity": severity,
                            "message": "source quality target was not met",
                            "fix": "rebuild the source to its declared tolerance"}]
    return mesh


def failed_ids(receipt):
    return {c["id"] for c in receipt["checks"] if not c["ok"]}


@pytest.mark.parametrize("op", ["union", "difference", "intersection"])
@pytest.mark.parametrize("side", [0, 1])
def test_boolean_preserves_warnings_from_either_operand(op, side):
    operands = [solid(), solid().transform(translate=[.01, 0, 0])]
    warning(operands[side])
    before = copy.deepcopy(operands[side].meta)
    result = B.boolean(*operands, op)
    assert verify(result)["verdict"] == "degraded"
    assert verify(result, level="strict")["verdict"] == "fail"
    assert result.meta["issues"] == before["issues"]
    result.meta["issues"][0]["message"] = "changed on result only"
    assert operands[side].meta == before


@pytest.mark.parametrize("op", ["union", "difference", "intersection"])
def test_normal_boolean_remains_pass(op):
    result = B.boolean(solid(), solid().transform(translate=[.01, 0, 0]), op)
    assert verify(result)["verdict"] == "pass"
    assert not result.meta.get("issues")


def test_error_source_remains_fail_at_basic_level():
    result = B.union(warning(solid(), "error"), solid())
    assert verify(result, level="basic")["verdict"] == "fail"


def test_actual_unmet_nurbs_target_survives_transform_and_repeated_boolean():
    graph = [
        {"id": "skin", "op": "nurbs_surface", "args": {
            "control_points": [[[30, 0, 0], [30, 0, 50]],
                               [[30, 30, 0], [30, 30, 50]],
                               [[0, 30, 0], [0, 30, 50]]],
            "degree_u": 2, "degree_v": 1,
            "weights": [[1, 1], [math.sqrt(.5)] * 2, [1, 1]],
            "tolerance": 1e-9, "max_refinements": 0, "fail_on_tolerance": False}},
        {"id": "solid", "op": "thicken", "inputs": ["skin"], "args": {"thickness": 1}},
    ]
    original = evaluate(graph)
    assert original.meta["nurbs"]["target_met"] is False
    result = original
    for _ in range(5):
        result = B.union(result, result).transform(translate=[.001, 0, 0])
        receipt = verify(result)
        assert receipt["verdict"] == "degraded"
        assert "surface_tolerance" in failed_ids(receipt)
        assert "nurbs" not in result.meta
        assert "surface_error_bound" not in result.meta
        sources = result.meta["geometry_provenance"]["sources"]
        assert len(sources) == 1
        assert sources[0]["nurbs"] == original.meta["nurbs"]
        assert sources[0]["nurbs"]["surface_error_bound"] is None
        assert len(result.meta["issues"]) == 1
    assert verify(result, level="strict")["verdict"] == "fail"
    assert len(json.dumps(result.meta)) < 10000


@pytest.mark.parametrize("op", ["union", "difference", "intersection"])
@pytest.mark.parametrize("side", [0, 1])
def test_source_target_without_issue_list_still_degrades(op, side):
    operands = [solid(), solid().transform(translate=[.01, 0, 0])]
    operands[side].meta["nurbs"] = {"target_met": False, "sampled_error": .002,
                                   "tolerance": .001, "exact": False, "surface_error_bound": None}
    result = B.boolean(*operands, op)
    assert "surface_tolerance" in failed_ids(verify(result))
    assert result.meta["issues"][0]["code"] == "nurbs_tolerance_unmet"


def test_field_sampling_survives_as_source_provenance_without_result_bound():
    source = fieldtree.mesh_field(fieldtree.compile_tree({"op": "sphere", "radius": .01}),
                                  [[-.015] * 3, [.015] * 3], .003)
    moved = source.transform(translate=[.004, 0, 0])
    result = B.intersection(moved, moved)
    sources = result.meta["geometry_provenance"]["sources"]
    assert len(sources) == 1
    assert sources[0]["field"]["distance_exact"] is False
    assert sources[0]["sampling"] == source.meta["sampling"]
    assert sources[0]["extraction"] == source.meta["extraction"]
    assert "field" not in result.meta and "sampling" not in result.meta
    assert "surface_error_bound" not in result.meta
    assert verify(result)["verdict"] == "pass"


def test_concat_sources_and_provenance_limit_are_not_silently_dropped(monkeypatch):
    pair = concat([warning(solid()), solid().transform(translate=[.08, 0, 0])])
    joined = B.union(pair, box([.12, .02, .02], center=[.04, 0, 0]))
    assert any(c.startswith("geometry_issue.") for c in failed_ids(verify(joined)))
    a, b = solid(), solid()
    a.meta["geometry_provenance"] = {"method": "source_quality/v1", "sources": [
        {"field": {"nodes": i, "distance_exact": False}} for i in range(128)]}
    b.meta["field"] = {"nodes": 129, "distance_exact": False}
    def forbidden(*args):
        pytest.fail("boolean geometry ran before provenance budget validation")
    monkeypatch.setattr(B, "_run", forbidden)
    with pytest.raises(AttaError) as exc:
        B.union(a, b)
    assert exc.value.code == "geometry_provenance_budget"


def test_metadata_byte_limit_and_incomplete_provenance_cannot_pass():
    source = warning(solid())
    source.meta["issues"][0]["message"] = "x" * 262145
    with pytest.raises(AttaError, match="provenance exceeds"):
        B.union(source, solid())
    receipt = verify(source)
    assert receipt["verdict"] == "fail"
    assert "geometry_issue.provenance" in failed_ids(receipt)


@pytest.mark.parametrize("provenance", ["bad", {"sources": [None]}, {"sources": [float("nan")]}])
def test_malformed_provenance_fails_cleanly(provenance):
    source = solid()
    source.meta["geometry_provenance"] = provenance
    with pytest.raises(AttaError):
        B.union(source, solid())
    assert verify(source)["verdict"] == "fail"
