# -*- coding: utf-8 -*-
"""Tests for extended recipe families: architecture_ext, organic_ext,
primitive_ext, and pattern_ext.

Validates registration, category coverage (>= 3 per category), all three
presets (safe, nominal, hero) against category acceptance gates and expected
genus, parameter variants, error-level constraints with actionable fixes,
budget fail-closed behaviour, and build determinism.
"""

from __future__ import annotations

import os

import pytest

os.environ["ATTA_FAMILIES_STRICT"] = "1"

from atta import families, materials, recipes, verify as verify_mod  # noqa: E402
from atta.errors import AttaError  # noqa: E402
from atta.mesh import topology, volume  # noqa: E402
from tests.test_coverage_gate import category_of, constraints_for  # noqa: E402

families.load(strict=True)

NEW_RECIPES = [
    ("architecture.arch_vault", "architecture", 0),
    ("architecture.stair", "architecture", 0),
    ("architecture.dome_oculus", "architecture", 1),
    ("organic.tree", "organic", 0),
    ("organic.phyllotaxis", "organic", 0),
    ("organic.coral", "organic", 0),
    ("primitive.rounded_cylinder", "primitive", 0),
    ("primitive.tube", "primitive", 1),
    ("primitive.wedge", "primitive", 0),
    ("pattern.knurl", "pattern", 0),
    ("pattern.louvre_panel", "pattern", 7),
]


def test_target_categories_have_at_least_three_recipes():
    counts: dict[str, int] = {}
    for rec in recipes.all_recipes():
        cat = category_of(rec.id)
        counts[cat] = counts.get(cat, 0) + 1
    for cat in ("architecture", "organic", "primitive", "pattern"):
        assert counts.get(cat, 0) >= 3, (cat, counts.get(cat, 0))


@pytest.mark.parametrize("rid,cat,genus", NEW_RECIPES)
def test_new_recipe_metadata_and_constraints(rid, cat, genus):
    rec = recipes.get(rid)
    assert category_of(rid) == cat
    assert rec.expected_genus == genus
    assert {"safe", "nominal", "hero"} <= set(rec.presets)
    assert rec.material in materials.PRESETS
    assert len(rec.constraints) >= 3
    for con in rec.constraints:
        assert con.get("fix"), (rid, con.get("id"))
        assert con.get("message"), (rid, con.get("id"))


@pytest.mark.parametrize("rid,cat,genus", NEW_RECIPES)
@pytest.mark.parametrize("preset", ("safe", "nominal", "hero"))
def test_new_recipe_presets_verify_pass(rid, cat, genus, preset):
    rec = recipes.get(rid)
    resolved = rec.resolve(preset=preset)
    assert not resolved.blocked, (rid, preset, resolved.issues)
    assert not resolved.issues, (rid, preset, resolved.issues)

    built = rec.build_shape(resolved.values, quality="draft")
    assert isinstance(built, tuple) and len(built) == 2
    mesh, fin = built
    assert isinstance(fin, dict) and fin.get("material") in materials.PRESETS

    topo = topology(mesh)
    assert not topo["empty"], (rid, preset)
    assert topo["closed"], (rid, preset, topo)
    assert topo["n_components"] == 1, (rid, preset, topo)
    assert topo["genus"] == genus, (rid, preset, topo)
    assert volume(mesh) > 0.0, (rid, preset)

    cons = constraints_for(cat, rec)
    receipt = verify_mod.verify(mesh, constraints=cons, level="standard",
                                expected_genus=rec.expected_genus)
    failed = [c for c in receipt.get("checks", []) if not c.get("ok")]
    assert receipt["verdict"] == "pass", (rid, preset, failed)


def test_architecture_variants():
    # Pointed vs semicircular arch
    arch = recipes.get("architecture.arch_vault")
    for style in ("semicircular", "pointed"):
        res = arch.resolve({"style": style})
        assert not res.blocked
        mesh, _ = arch.build_shape(res.values, "draft")
        topo = topology(mesh)
        assert topo["closed"] and topo["n_components"] == 1 and topo["genus"] == 0

    # Dome with 0 ribs vs 20 ribs stays genus 1
    dome = recipes.get("architecture.dome_oculus")
    for rib_count in (0, 20):
        res = dome.resolve({"rib_count": rib_count})
        assert not res.blocked
        mesh, _ = dome.build_shape(res.values, "draft")
        topo = topology(mesh)
        assert topo["closed"] and topo["n_components"] == 1 and topo["genus"] == 1

    # Stair step count keeps genus 0
    stair = recipes.get("architecture.stair")
    for n in (8, 16):
        res = stair.resolve({"step_count": n, "total_height": 180.0})
        assert not res.blocked
        mesh, _ = stair.build_shape(res.values, "draft")
        topo = topology(mesh)
        assert topo["closed"] and topo["n_components"] == 1 and topo["genus"] == 0


def test_primitive_and_pattern_variants():
    rc = recipes.get("primitive.rounded_cylinder")
    for mode in ("fillet", "chamfer"):
        mesh, _ = rc.build_shape(rc.resolve({"edge_mode": mode}).values, "draft")
        assert topology(mesh)["closed"] and topology(mesh)["genus"] == 0

    tube = recipes.get("primitive.tube")
    for mode in ("fillet", "chamfer"):
        mesh, _ = tube.build_shape(tube.resolve({"edge_mode": mode}).values, "draft")
        assert topology(mesh)["closed"] and topology(mesh)["genus"] == 1

    knurl = recipes.get("pattern.knurl")
    for pat in ("diamond", "straight", "helical"):
        mesh, _ = knurl.build_shape(knurl.resolve({"pattern": pat}).values, "draft")
        topo = topology(mesh)
        assert topo["closed"] and topo["n_components"] == 1 and topo["genus"] == 0

    louvre = recipes.get("pattern.louvre_panel")
    for slats in (4, 8):
        res = louvre.resolve({"slat_count": slats, "height": 150.0})
        assert not res.blocked
        mesh, _ = louvre.build_shape(res.values, "draft")
        topo = topology(mesh)
        assert topo["closed"] and topo["n_components"] == 1
        assert topo["genus"] == slats + 1
    # Restore nominal genus on registry entry after custom slat counts.
    louvre.build_shape(louvre.resolve(preset="nominal").values, "draft")
    assert louvre.expected_genus == 7


def test_constraints_and_budgets_fail_closed():
    assert recipes.get("architecture.arch_vault").resolve({"arch_t": 8.0, "span": 200.0}).blocked
    assert recipes.get("architecture.stair").resolve({"step_count": 36, "total_height": 80.0}).blocked
    assert recipes.get("organic.tree").resolve({"levels": 5, "branching": 4, "max_branches": 64}).blocked
    assert recipes.get("organic.phyllotaxis").resolve({"count": 200, "floret_r": 8.0}).blocked
    assert recipes.get("primitive.tube").resolve({"outer_d": 30.0, "inner_d": 28.5}).blocked
    assert recipes.get("pattern.louvre_panel").resolve({"slat_count": 24, "height": 80.0}).blocked

    # Direct builder budget guard also fails closed with AttaError(code, path, fix).
    tree = recipes.get("organic.tree")
    vals = tree.resolve(preset="nominal").values
    vals["levels"] = 5
    vals["branching"] = 4
    vals["max_branches"] = 64
    with pytest.raises(AttaError) as ei:
        tree.build_shape(vals, "draft")
    assert ei.value.code == "budget_exceeded" and ei.value.path and ei.value.fix


@pytest.mark.parametrize("rid,_cat,_genus", NEW_RECIPES)
def test_build_is_deterministic(rid, _cat, _genus):
    rec = recipes.get(rid)
    vals = rec.resolve(preset="nominal").values
    m1, _ = rec.build_shape(vals, "draft")
    m2, _ = rec.build_shape(vals, "draft")
    assert m1.hash() == m2.hash(), rid
