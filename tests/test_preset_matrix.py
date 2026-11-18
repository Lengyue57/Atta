# -*- coding: utf-8 -*-
"""Every registered recipe must survive its whole preset envelope.

`tests/test_coverage_gate.py` checks each recipe at the *nominal* preset only.
A recipe that is closed at nominal but self-intersects at `hero` (large counts,
thin walls) or degenerates at `safe` is still broken for an agent that asks for
those presets, so this test builds every recipe at **safe, nominal and hero**
and applies the same category gate:

* closed, single component, positive volume, category min-wall (from
  ``docs/coverage_matrix.json`` via the coverage-gate helpers);
* the recipe's expected genus *for the resolved parameters* -- a recipe may
  declare ``expected_genus`` as a constant, as a callable of the resolved
  values, or expose ``expected_genus_for(values)``; the count-driven ones
  (flange bolt circle, honeycomb cells, gyroid cells) must stay correct when the
  preset changes the counts.

No skip marks: a recipe that cannot build a preset it advertises is a failure.
"""

from __future__ import annotations

import os

import pytest

os.environ["ATTA_FAMILIES_STRICT"] = "1"

from atta import recipes, verify as verify_mod  # noqa: E402

from test_coverage_gate import category_of, constraints_for  # noqa: E402

PRESETS = ("safe", "nominal", "hero")


def expected_genus_for(recipe, values):
    """The genus the recipe promises for these resolved values (or None)."""
    fn = getattr(recipe, "expected_genus_for", None)
    if callable(fn):
        return fn(values)
    g = getattr(recipe, "expected_genus", None)
    if callable(g):
        return g(values)
    return g


def matrix_cases():
    out = []
    for r in recipes.all_recipes():
        for preset in PRESETS:
            out.append((r.id, preset))
    return out


def test_every_recipe_advertises_the_three_presets():
    missing = [(r.id, sorted(set(PRESETS) - set(r.presets)))
               for r in recipes.all_recipes() if not set(PRESETS) <= set(r.presets)]
    assert not missing, "recipes missing safe/nominal/hero presets: %s" % missing


@pytest.mark.parametrize("rid,preset", matrix_cases())
def test_recipe_preset_passes_category_gate(rid, preset):
    recipe = recipes.get(rid)
    category = category_of(rid)
    assert preset in recipe.presets, "%s has no %r preset" % (rid, preset)
    resolved = recipe.resolve(preset=preset)
    assert not resolved.blocked, (rid, preset, [i for i in resolved.issues
                                                if i.get("severity") == "error"])
    mesh = recipe.build_shape(resolved.values, quality="draft")
    if isinstance(mesh, tuple):
        mesh = mesh[0]
    genus = expected_genus_for(recipe, resolved.values)
    cons = constraints_for(category, recipe)
    cons.pop("genus", None)
    if genus is not None:
        cons["genus"] = int(genus)
    receipt = verify_mod.verify(mesh, constraints=cons, level="standard",
                                expected_genus=genus)
    facts = receipt.get("facts") or {}
    failed = [c for c in receipt.get("checks", []) if not c.get("ok")]
    assert receipt.get("verdict") == "pass", (
        "%s@%s (%s) verdict=%s genus=%s/%s components=%s; failing: %s"
        % (rid, preset, category, receipt.get("verdict"), facts.get("genus"), genus,
           facts.get("n_components"),
           [(c["id"], c.get("value"), c.get("expected")) for c in failed][:6]))
    assert facts.get("n_components") == 1, (rid, preset, facts.get("n_components"))
    vol = facts.get("volume_mm3")
    if vol is not None:
        assert vol > 0, (rid, preset, vol)
