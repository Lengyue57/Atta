# -*- coding: utf-8 -*-
"""Tests for CADClamp-style static resting stability rule in atta.design."""

from __future__ import annotations

import numpy as np
import pytest

from atta import design, features, mesh as M, primitives


def _l_bracket_overhanging():
    """Vertical foot at x=[0,10], z=[0,60] with overhanging top arm x=[0,90], z=[50,60]."""
    return features.evaluate([
        {"id": "foot", "op": "box", "args": {"size": [10, 20, 60], "center": [5, 10, 30]}},
        {"id": "arm", "op": "box", "args": {"size": [90, 20, 10], "center": [45, 10, 55]}},
        {"id": "body", "op": "union", "inputs": ["foot", "arm"]},
    ], output="body")


def _l_bracket_resting_on_long_arm():
    """Long horizontal base at x=[0,90], z=[0,10] with upright leg at x=[0,10], z=[0,60]."""
    return features.evaluate([
        {"id": "base", "op": "box", "args": {"size": [90, 20, 10], "center": [45, 10, 5]}},
        {"id": "leg", "op": "box", "args": {"size": [10, 20, 60], "center": [5, 10, 30]}},
        {"id": "body", "op": "union", "inputs": ["base", "leg"]},
    ], output="body")


def test_stability_upright_block_passes():
    block = primitives.box([0.04, 0.04, 0.12])
    res = design.stability(block)
    assert res["id"] == "stability"
    assert res["ok"] is True
    assert res["value"]["com_xy_mm"] == pytest.approx([0.0, 0.0], abs=1e-3)
    assert res["value"]["clearance_to_hull_mm"] == pytest.approx(20.0, abs=1e-2)
    assert res["detail"]["method"] == "com_xy_vs_z_min_convex_hull"


def test_stability_overhanging_l_bracket_tips_over():
    bracket = _l_bracket_overhanging()
    res = design.stability(bracket)
    assert res["id"] == "stability"
    assert res["ok"] is False
    # Foot volume=10*20*50=10000 at x=5; arm volume=90*20*10=18000 at x=45 -> CoM_x=30.71mm;
    # support footprint spans x=[0,10] -> clearance = -(30.71 - 10) = -20.71 mm.
    assert res["value"]["com_xy_mm"][0] == pytest.approx(30.7143, abs=0.2)
    assert res["value"]["clearance_to_hull_mm"] == pytest.approx(-20.7143, abs=0.2)
    assert "outside" in res["advice"].lower()


def test_stability_resting_on_long_arm_is_stable():
    bracket = _l_bracket_resting_on_long_arm()
    res = design.stability(bracket)
    assert res["ok"] is True
    assert res["value"]["clearance_to_hull_mm"] == pytest.approx(10.0, abs=0.2)


def test_stability_margin_and_degenerate_footprint():
    block = primitives.box([0.02, 0.02, 0.04])  # half-width = 10 mm
    assert design.stability(block, margin_mm=5.0)["ok"] is True
    tight = design.stability(block, margin_mm=12.0)
    assert tight["ok"] is False
    assert 0.0 < tight["value"]["clearance_to_hull_mm"] < 12.0

    # Triangular prism resting on a single bottom edge (zero support area)
    verts = np.array([
        [0.0, 0.0, 0.0],
        [0.04, 0.0, 0.0],
        [0.0, 0.04, 0.04],
        [0.04, 0.04, 0.04],
        [0.0, -0.04, 0.04],
        [0.04, -0.04, 0.04],
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 3], [0, 3, 1],
        [0, 1, 5], [0, 5, 4],
        [2, 4, 5], [2, 5, 3],
        [0, 4, 2], [1, 3, 5],
    ], dtype=np.int64)
    wedge = M.Mesh(verts, faces).oriented()
    wres = design.stability(wedge)
    assert wres["ok"] is False
    assert wres["detail"]["degenerate_footprint"] is True

    empty = M.Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    assert design.stability(empty)["ok"] is None

    with pytest.raises(ValueError):
        design.stability(block, margin_mm=-0.5)


def test_stability_registered_in_design_report_and_rule_names():
    assert "stability" in design.RULE_NAMES
    block = primitives.box([0.04, 0.04, 0.08])
    full = design.design_report(block)
    ids = [c["id"] for c in full["checks"]]
    assert "stability" in ids

    only = design.design_report(block, rules=["stability"])
    assert [c["id"] for c in only["checks"]] == ["stability"]
    assert only["summary"]["passed"] == 1
    assert only["summary"]["failed"] == 0
