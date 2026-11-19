# -*- coding: utf-8 -*-
"""Parametric expected_genus in registry.py and across flange, honeycomb & gyroid_slab."""

from __future__ import annotations

import math

import pytest

from atta import engine, recipes, verify as verify_mod
from atta.errors import AttaError
from atta.mesh import topology
from atta.registry import GenusRule, Recipe, evaluate_genus


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTA_HOME", str(tmp_path))
    return tmp_path


# ------------------------------------------------ registry unit tests

def test_evaluate_genus_modes_and_validation():
    assert evaluate_genus(None, {}) is None
    assert evaluate_genus(3, {}) == 3
    assert evaluate_genus("1 + bolt_holes", {"bolt_holes": 8}) == 9
    assert evaluate_genus("rows * cols", {"rows": 4, "cols": 5}) == 20
    rule = GenusRule(lambda d: 2 * int(d["n"]) + 1, text="2*n + 1")
    assert evaluate_genus(rule, {"n": 3}) == 7
    assert evaluate_genus(lambda d: None, {"n": 3}) is None

    for bad in (True, False, -1, 1.5, math.nan, math.inf, "unknown_var + 1",
                lambda d: True, lambda d: -2, lambda d: 2.5, lambda d: "bad"):
        with pytest.raises(AttaError) as exc:
            evaluate_genus(bad, {"n": 3}, "test.recipe")
        assert exc.value.code in ("bad_genus_rule", "bad_arg")
        assert exc.value.path and exc.value.fix


def test_recipe_parametric_genus_metadata_and_resolution():
    r = Recipe(
        id="test.param_genus", name="test", description="test",
        params={"holes": {"type": "int", "default": 4, "min": 1, "max": 20, "unit": "count"}},
        presets={"nominal": {"holes": 4}},
        expected_genus="1 + holes",
    )
    assert r.genus_is_parametric is True
    assert r.expected_genus == 5
    info = r.to_dict()
    assert info["expected_genus"] == 5
    assert info["expected_genus_rule"] == "1 + holes"

    res1 = r.resolve(patch={"holes": 10})
    assert res1.expected_genus == 11
    assert r.expected_genus == 11
    assert res1.to_dict()["expected_genus"] == 11

    res2 = r.resolve(patch={"holes": 2})
    assert res2.expected_genus == 3
    assert r.expected_genus == 3
    # Nominal metadata stays independent of last resolve
    assert r.to_dict()["expected_genus"] == 5
    assert r.genus_for({"holes": 7}) == 8


# ------------------------------------------------ mechanical.flange

@pytest.mark.parametrize("bolt_holes", [4, 6, 8, 12])
def test_flange_parametric_genus_across_bolt_counts(home, bolt_holes):
    """mechanical.flange expected_genus == 1 + bolt_holes and verify passes."""
    rec = recipes.get("mechanical.flange")
    assert rec.genus_is_parametric is True
    assert rec.to_dict()["expected_genus_rule"] == "1 + bolt_holes"

    patch = {"bolt_holes": bolt_holes, "bolt_hole_d": 12.0}
    out = engine.model(recipe="mechanical.flange", params=patch, quality="draft")
    expected = 1 + bolt_holes
    assert out["model"]["expected_genus"] == expected
    assert out["facts"]["genus"] == expected
    receipt = engine.verify(model_id=out["model_id"], level="standard")
    assert receipt["verdict"] == "pass", [c for c in receipt["checks"] if not c["ok"]]

    # Honesty check: asserting the wrong genus must fail verify
    wrong = engine.verify(model_id=out["model_id"], level="standard",
                          constraints={"genus": expected + 1})
    assert wrong["verdict"] == "fail"


# ------------------------------------------------ pattern.honeycomb_panel

@pytest.mark.parametrize("rows,cols", [(2, 3), (3, 4), (4, 5), (5, 7)])
def test_honeycomb_parametric_genus_across_cell_counts(home, rows, cols):
    """pattern.honeycomb_panel expected_genus == rows * cols and verify passes."""
    rec = recipes.get("pattern.honeycomb_panel")
    assert rec.genus_is_parametric is True
    assert rec.to_dict()["expected_genus_rule"] == "rows * cols"

    out = engine.model(recipe="pattern.honeycomb_panel",
                       params={"rows": rows, "cols": cols}, quality="draft")
    expected = rows * cols
    assert out["model"]["expected_genus"] == expected
    assert out["facts"]["genus"] == expected
    receipt = engine.verify(model_id=out["model_id"], level="standard")
    assert receipt["verdict"] == "pass", [c for c in receipt["checks"] if not c["ok"]]


# ------------------------------------------------ lattice.gyroid_slab

@pytest.mark.parametrize("cells,expected", [
    ((1, 1, 1), 5),
    ((2, 1, 1), 11),
    ((2, 2, 1), 25),
    ((3, 2, 1), 39),
    ((2, 2, 2), 57),
])
def test_gyroid_slab_parametric_genus_across_cell_counts(home, cells, expected):
    """lattice.gyroid_slab expected_genus == 8*nx*ny*nz - 2*nx - 2*ny + 1."""
    rec = recipes.get("lattice.gyroid_slab")
    assert rec.genus_is_parametric is True
    cell = 20.0
    patch = {
        "cell_size": cell,
        "thickness": 2.6,
        "slab_l": cells[0] * cell,
        "slab_w": cells[1] * cell,
        "slab_h": cells[2] * cell,
        "resolution": 12,
    }
    res = rec.resolve(patch=patch)
    assert not res.blocked, res.issues
    assert res.expected_genus == expected

    out = engine.model(recipe="lattice.gyroid_slab", params=patch, quality="draft")
    assert out["model"]["expected_genus"] == expected
    assert out["facts"]["genus"] == expected
    assert out["facts"]["n_components"] == 1
    assert out["facts"]["closed"] is True
    receipt = engine.verify(model_id=out["model_id"], level="standard")
    assert receipt["verdict"] == "pass", [c for c in receipt["checks"] if not c["ok"]]


def test_gyroid_slab_fractional_cells_honest_none_genus(home):
    """Fractional cell counts do not claim a bogus constant genus and still build clean."""
    rec = recipes.get("lattice.gyroid_slab")
    patch = {
        "cell_size": 20.0,
        "thickness": 2.6,
        "slab_l": 50.0,   # 2.5 cells
        "slab_w": 40.0,   # 2.0 cells
        "slab_h": 20.0,   # 1.0 cell
        "resolution": 12,
    }
    try:
        res = rec.resolve(patch=patch)
        assert not res.blocked
        assert res.expected_genus is None
        assert any(i.get("code") == "genus_envelope" and i.get("severity") == "warn"
                   for i in res.issues)
        mesh, _ = rec.build_shape(res.values, quality="draft")
        topo = topology(mesh)
        assert topo["closed"] is True
        assert topo["n_components"] == 1
        receipt = verify_mod.verify(mesh, constraints={"closed": True, "n_components": 1},
                                    level="standard", expected_genus=res.expected_genus)
        assert receipt["verdict"] == "pass"
    finally:
        # Restore nominal thread-local cache so later metadata tests see nominal int
        rec.resolve(preset="nominal")
