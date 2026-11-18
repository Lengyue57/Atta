# -*- coding: utf-8 -*-
"""Regression tests for product.mug across safe, nominal and hero presets."""

from __future__ import annotations

import os

import pytest

os.environ["ATTA_FAMILIES_STRICT"] = "1"

from atta import metrology, recipes, verify as verify_mod  # noqa: E402
from atta.mesh import topology  # noqa: E402


@pytest.mark.parametrize("preset", ["safe", "nominal", "hero"])
def test_mug_preset_closed_wall_and_verify(preset):
    rec = recipes.get("product.mug")
    resolved = rec.resolve(preset=preset)
    assert not resolved.blocked, (preset, resolved.issues)

    mesh, _finish = rec.build_shape(resolved.values, quality="draft")
    topo = topology(mesh)
    assert not topo["empty"], (preset, topo)
    assert topo["closed"], (preset, topo)
    assert topo["non_manifold_edges"] == 0, (preset, topo)
    assert topo["n_components"] == 1, (preset, topo)
    assert topo["genus"] == 1, (preset, topo)

    wt = metrology.wall_thickness(mesh, max_samples=600)
    assert wt["applicable"] is True, (preset, wt)
    assert wt["min_mm"] is not None and wt["min_mm"] >= 2.5, (preset, wt)

    cons = {"closed": True, "min_wall_mm": {"min": 2.5}, "n_components": 1, "genus": 1}
    for level in ("standard", "strict"):
        receipt = verify_mod.verify(mesh, constraints=cons, level=level, expected_genus=1)
        assert receipt["verdict"] == "pass", (preset, level, receipt["checks"])
