# -*- coding: utf-8 -*-
"""Focused regression tests for conjure handoff and design heuristics."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from atta import design, features, mesh as M, primitives
from atta.conjure import conjure


def build_plan_mesh(label, **kwargs):
    plan = conjure(label, **kwargs)
    build = plan["build"]
    assert build["kind"] == "graph"
    mesh = features.evaluate(build["features"], output=build["output"])
    return plan, mesh


@pytest.mark.parametrize("label, genus", [("mug", 1), ("bottle", 0), ("helmet", 0)])
def test_conjure_shells_are_single_closed_solids(label, genus):
    plan, mesh = build_plan_mesh(label, seed=0)
    topo = M.topology(mesh)
    assert topo["closed"]
    assert topo["n_components"] == 1
    assert topo["genus"] == genus
    assert plan["constraints"]["n_components"] == 1
    assert isinstance(plan["constraints"]["min_wall_mm"], float)


def test_conjure_scale_and_bounds_remain_derived_not_measured():
    measured, _ = build_plan_mesh("mug", size_mm=[100, 80, 120])
    assert measured["provenance"]["size_mm"] == "measured"
    scaled = conjure("mug", size_mm=[100, 80, 120], scale=2)
    assert scaled["provenance"]["size_mm"] == "derived"
    assert scaled["provenance_detail"]["size_mm"]["measured"] is False
    bounded = conjure("mug", constraints={"max_mm": [50, 50, 50]})
    assert bounded["provenance"]["size_mm"] == "derived"
    with pytest.raises(Exception):
        conjure("mug", scale=0)


def test_conjure_part_shares_are_partitions_and_not_fake_volumes():
    for label, names in (("knife", ("blade", "handle")),
                         ("sword", ("blade", "guard", "grip", "pommel")),
                         ("mushroom", ("cap", "stem"))):
        plan = conjure(label, style="organic", seed=7)
        assert sum(plan["ratios"][name] for name in names) == pytest.approx(1.0)
        assert sum(part["volume_share"] for part in plan["parts"].values()) == pytest.approx(1.0)
        assert all(part["exact"] is False for part in plan["parts"].values())


def test_conjure_style_reaches_consumed_hardware_geometry():
    neutral = conjure("sword", style="neutral", seed=7, size_mm=[1050, 210, 40])
    organic = conjure("sword", style="organic", seed=7, size_mm=[1050, 210, 40])
    assert neutral["build"] != organic["build"]
    assert neutral["ratios"]["blade"] != organic["ratios"]["blade"]


def test_form_hierarchy_is_tessellation_invariant_and_advisory():
    box = primitives.box([0.1, 0.1, 0.1])
    report = design.form_hierarchy(box)
    assert report["ok"] is None
    assert report["detail"]["heuristic"] is True
    assert report["value"]["patches"] == 6
    assert "advisory" in report["advice"]


def test_form_hierarchy_constant_area_does_not_raise():
    box = primitives.box([0.1, 0.1, 0.1])
    assert design.form_hierarchy(box, bins=12)["value"]["area_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize("ratio", [1, 5, 10, 20, 50])
def test_readability_reports_thin_views_instead_of_long_object_false_positive(ratio):
    report = design.silhouette_readability(primitives.box([0.01, 0.01, 0.01 * ratio]))
    assert report["detail"]["method"] == "shared_frame_area_sampling"
    if ratio >= 20:
        assert any(view["status"] == "unresolved" for view in report["value"]["views"].values())
    else:
        assert report["value"]["score"] is not None


def test_readability_requires_ordered_resolutions():
    with pytest.raises(ValueError):
        design.silhouette_readability(primitives.box([.1, .1, .1]), res=16, small=32)
