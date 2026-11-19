# -*- coding: utf-8 -*-
"""Implicit TPMS topology, extraction robustness, analytic genus & metric unit guard."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atta import engine, fieldtree, metrology, recipes, sdfkit
from atta.errors import AttaError
from atta.families import implicit, lattice
from atta.mesh import topology


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTA_HOME", str(tmp_path))
    return tmp_path


def _graph(features, name="implicit_test", output="out"):
    return {
        "schema": "atta.spec/v3",
        "name": name,
        "model": {"kind": "graph", "units": "mm", "output": output,
                  "features": features},
    }


# ------------------------------------------------ analytic unit-cell & block genus

def test_tpms_unit_cell_genus_constants():
    """Verify crystallographic primitive vs conventional cubic cell genus constants."""
    p = sdfkit.tpms_unit_cell_genus("schwarz_p")
    g = sdfkit.tpms_unit_cell_genus("gyroid")
    d = sdfkit.tpms_unit_cell_genus("diamond")
    n = sdfkit.tpms_unit_cell_genus("neovius")
    assert (p["primitive_genus"], p["conventional_cubic_genus"], p["euler_per_cube"]) == (3, 3, -4)
    assert (g["primitive_genus"], g["conventional_cubic_genus"], g["euler_per_cube"]) == (3, 5, -8)
    assert (d["primitive_genus"], d["conventional_cubic_genus"], d["euler_per_cube"]) == (3, 9, -16)
    assert (n["primitive_genus"], n["conventional_cubic_genus"], n["euler_per_cube"]) == (9, 9, -16)
    assert sdfkit.tpms_block_genus("schwarz_p", 1, 1, 1, mode="network") == 0
    assert sdfkit.tpms_block_genus("schwarz_p", 2, 2, 2, mode="network") == 5
    assert sdfkit.tpms_block_genus("schwarz_p", 3, 3, 3, mode="network") == 28


@pytest.mark.parametrize("field,cells,expected", [
    ("schwarz_p", (1, 1, 1), 5),
    ("schwarz_p", (2, 1, 1), 9),
    ("schwarz_p", (2, 2, 1), 17),
    ("gyroid", (1, 1, 1), 5),
    ("gyroid", (2, 1, 1), 11),
    ("gyroid", (1, 1, 2), 13),
    ("gyroid", (2, 2, 1), 25),
    ("diamond", (1, 1, 1), 11),
    ("diamond", (2, 1, 1), 25),
    ("neovius", (1, 1, 1), 11),
    ("neovius", (2, 1, 1), 25),
])
def test_tpms_block_genus_matches_extracted_mesh(field, cells, expected):
    """Analytic Mayer-Vietoris block genus matches Marching Tetrahedra across all 4 TPMS."""
    k, l, m = cells
    assert sdfkit.tpms_block_genus(field, k, l, m) == expected
    period = 0.012
    thick = 0.0014
    size = (k * period, l * period, m * period)
    mesh, info = implicit.sdf_mesh(
        size, field=field, period=period, thickness=thick, resolution=14)
    topo = topology(mesh)
    assert topo["closed"] is True
    assert topo["n_components"] == 1
    assert topo["non_manifold_edges"] == 0
    assert topo["degenerate_faces"] == 0
    assert topo["genus"] == expected
    assert info["analytic_genus"] == expected


# ------------------------------------------------ kernel regressions from bench failures

def test_sphere_clip_larger_than_size_stays_closed_single_component(home):
    """Regression for curv_gyroid_bones_sheet: clip_radius > min(size)/2 must stay closed."""
    spec = _graph([{
        "id": "out", "op": "sdf",
        "args": {"size": [55.0, 31.0, 18.0], "field": "gyroid",
                 "resolution": 12, "thickness": 0.8,
                 "clip": "sphere", "clip_radius": 20.0},
    }])
    out = engine.model(spec=spec, quality="draft")
    facts = out["facts"]
    assert facts["closed"] is True
    assert facts["n_components"] == 1
    assert isinstance(facts["genus"], int) and facts["genus"] > 0
    # Must stay inside min(size, 2*clip_radius) = [40, 31, 18] mm
    sx, sy, sz = facts["size_mm"]
    assert sx <= 40.2 and sy <= 31.2 and sz <= 18.2


def test_thin_schwarz_p_resolves_without_fake_pinholes(home):
    """Regression for curv_schwarz_p_surface: thin 0.8 mm wall must not drop below 3 vox/wall."""
    spec = _graph([{
        "id": "out", "op": "sdf",
        "args": {"size": [47.0, 25.0, 23.0], "field": "schwarz_p",
                 "resolution": 12, "thickness": 0.8},
    }])
    out = engine.model(spec=spec, quality="draft")
    facts = out["facts"]
    assert facts["closed"] is True
    assert facts["n_components"] == 1
    # Converged true genus of this 2.04 x 1.09 x 1.00 period Schwarz P block is 9 (was 474!)
    assert facts["genus"] == 9


def test_noncommensurate_gyroid_prunes_boundary_slivers(home):
    """Regression for curv_gyroid_torus_donut: non-integer box cut yields 1 clean component."""
    spec = _graph([{
        "id": "out", "op": "sdf",
        "args": {"size": [33.0, 18.0, 15.0], "field": "gyroid",
                 "resolution": 12, "thickness": 0.8},
    }])
    out = engine.model(spec=spec, quality="draft")
    facts = out["facts"]
    assert facts["closed"] is True
    assert facts["n_components"] == 1
    assert facts["genus"] == 11


def test_prune_boundary_slivers_never_hides_two_real_bodies():
    """Honesty guard: two comparable disconnected solids are never pruned."""
    X, Y, Z, org = sdfkit.grid((-0.02, -0.01, -0.01), (0.02, 0.01, 0.01), 0.001)
    a = sdfkit.sphere(X, Y, Z, center=(-0.01, 0.0, 0.0), radius=0.005)
    b = sdfkit.sphere(X, Y, Z, center=(0.01, 0.0, 0.0), radius=0.005)
    phi = sdfkit.union(a, b)
    clip = sdfkit.box(X, Y, Z, size=(0.04, 0.02, 0.02))
    cleaned, n_pruned = sdfkit.prune_boundary_slivers(phi, clip, 0.001)
    assert n_pruned == 0
    mesh = sdfkit.extract(cleaned, org, 0.001)
    assert topology(mesh)["n_components"] == 2


def test_cap_open_boundary_closes_unclipped_field():
    """Explicit boundary capping closes a field that crosses the grid border."""
    X, Y, Z, org = sdfkit.grid((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01), 0.002)
    # Plane half-space x < 0 crosses 5 of the 6 grid borders
    open_mesh = sdfkit.extract(X, org, 0.002, cap_open_boundary=False)
    assert topology(open_mesh)["closed"] is False
    closed_mesh = sdfkit.extract(X, org, 0.002, cap_open_boundary=True)
    topo = topology(closed_mesh)
    assert topo["closed"] is True
    assert topo["n_components"] == 1
    assert topo["genus"] == 0


def test_mandelbulb_field_builds_clean_closed_manifold(home):
    spec = _graph([{
        "id": "out", "op": "sdf",
        "args": {"size": [24.0, 24.0, 24.0], "field": "mandelbulb",
                 "thickness": 2.4, "resolution": 14},
    }])
    out = engine.model(spec=spec, quality="draft")
    facts = out["facts"]
    assert facts["closed"] is True
    assert facts["n_components"] == 1
    assert facts["volume_mm3"] > 0


# ------------------------------------------------ metric wall thickness & unit guard (Task C)

@pytest.mark.parametrize("cell_mm,thick_mm,cells,t_bias", [
    (24.0, 3.6, (3, 2, 1), 0.0),   # safe preset
    (22.0, 3.0, (3, 2, 1), 0.0),   # nominal preset
    (18.0, 2.2, (3, 2, 1), 0.1),   # hero preset
    (16.0, 2.0, (2, 2, 1), 0.0),   # smaller cell
    (30.0, 3.6, (2, 1, 1), 0.0),   # larger cell
    (20.0, 2.8, (2, 2, 1), 0.15),  # non-zero bias
])
def test_gyroid_slab_metric_volume_tracks_weyl_expectation(cell_mm, thick_mm, cells, t_bias):
    """Measured volume tracks Weyl's analytic tubular volume within 4% across sizes."""
    rec = recipes.get("lattice.gyroid_slab")
    patch = {
        "cell_size": cell_mm,
        "thickness": thick_mm,
        "slab_l": cells[0] * cell_mm,
        "slab_w": cells[1] * cell_mm,
        "slab_h": cells[2] * cell_mm,
        "t_bias": t_bias,
    }
    res = rec.resolve(patch=patch)
    assert not res.blocked, res.issues
    mesh, _ = rec.build_shape(res.values, quality="draft")
    meas = metrology.measure(mesh)
    vol_mm3 = meas["volume_mm3"]
    expected_mm3 = sdfkit.tpms_sheet_volume_estimate(
        "gyroid",
        (res.values["slab_l"], res.values["slab_w"], res.values["slab_h"]),
        res.values["cell_size"],
        res.values["thickness"],
        offset=t_bias,
    ) * 1e9
    rel_err = abs(vol_mm3 - expected_mm3) / expected_mm3
    assert rel_err < 0.04, (cell_mm, thick_mm, cells, vol_mm3, expected_mm3, rel_err)


def test_gyroid_slab_scale_invariance_is_cubic():
    """Scaling all lengths by S scales extracted volume by S^3 (no field-vs-metre drift)."""
    rec = recipes.get("lattice.gyroid_slab")
    r1 = rec.resolve(patch={"cell_size": 16.0, "thickness": 2.0,
                            "slab_l": 32.0, "slab_w": 32.0, "slab_h": 16.0})
    r2 = rec.resolve(patch={"cell_size": 24.0, "thickness": 3.0,
                            "slab_l": 48.0, "slab_w": 48.0, "slab_h": 24.0})
    m1, _ = rec.build_shape(r1.values, quality="draft")
    m2, _ = rec.build_shape(r2.values, quality="draft")
    ratio = metrology.measure(m2)["volume_mm3"] / metrology.measure(m1)["volume_mm3"]
    expected_ratio = (24.0 / 16.0) ** 3
    assert abs(ratio - expected_ratio) / expected_ratio < 0.02


def test_gyroid_slab_unit_guard_rejects_raw_millimetres():
    """Passing raw mm directly to _build_gyroid_slab fails closed with unit_mismatch."""
    raw_mm = {
        "slab_l": 66.0, "slab_w": 44.0, "slab_h": 22.0,
        "cell_size": 22.0, "thickness": 3.0, "t_bias": 0.0, "resolution": 14,
    }
    with pytest.raises(AttaError) as exc:
        lattice._build_gyroid_slab(raw_mm, "draft")
    assert exc.value.code == "unit_mismatch"
    assert exc.value.path and exc.value.fix


def test_fieldtree_metric_tpms_thickness_and_exclusivity():
    """field_tree supports metric thickness on gyroid/schwarz_p/diamond/neovius."""
    tree = {
        "op": "intersection",
        "children": [
            {"op": "box", "size": [0.02, 0.02, 0.02]},
            {"op": "diamond", "period": 0.02, "thickness": 0.0024, "phase_deg": [90, 90, 90]},
        ],
    }
    compiled = fieldtree.compile_tree(tree)
    mesh = fieldtree.mesh_field(compiled, [[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]], 0.0007)
    topo = topology(mesh)
    assert topo["closed"] is True
    assert topo["n_components"] == 1
    assert topo["genus"] == 11
    with pytest.raises(AttaError) as exc:
        fieldtree.compile_tree({"op": "gyroid", "period": 0.02, "band": 0.2, "thickness": 0.002})
    assert exc.value.code == "bad_arg"


# ------------------------------------------------ determinism, budgets & adversarial inputs

def test_sdf_and_gyroid_are_deterministic():
    m1, _ = implicit.sdf_mesh((0.02, 0.02, 0.02), field="schwarz_p", period=0.02,
                              thickness=0.0024, resolution=12)
    m2, _ = implicit.sdf_mesh((0.02, 0.02, 0.02), field="schwarz_p", period=0.02,
                              thickness=0.0024, resolution=12)
    assert m1.hash() == m2.hash()


def test_budgets_and_bad_args_fail_closed():
    # Sample budget fail-closed (no silent under-sampling)
    with pytest.raises(AttaError) as exc:
        implicit.sdf_mesh((0.05, 0.05, 0.05), field="gyroid", period=0.02,
                          thickness=0.0005, max_samples=10_000)
    assert exc.value.code == "sdf_too_dense"
    assert exc.value.path and exc.value.fix

    # Triangle budget fail-closed
    with pytest.raises(AttaError) as exc:
        implicit.sdf_mesh((0.02, 0.02, 0.02), field="gyroid", period=0.02,
                          thickness=0.0024, max_triangles=100)
    assert exc.value.code == "sdf_too_dense"

    # Booleans, NaN, zero/negative, over-thick walls rejected cleanly
    for bad_kwargs in (
        {"size": (True, 0.02, 0.02)},
        {"size": (math.nan, 0.02, 0.02)},
        {"size": (0.0, 0.02, 0.02)},
        {"size": (0.02, 0.02, 0.02), "thickness": True},
        {"size": (0.02, 0.02, 0.02), "resolution": True},
        {"size": (0.02, 0.02, 0.02), "period": 0.02, "thickness": 0.011},
        {"size": (0.02, 0.02, 0.02), "clip": "cylinder"},
        {"size": (0.02, 0.02, 0.02), "field": "mandelbulb", "power": 100.0},
    ):
        with pytest.raises(AttaError) as exc:
            implicit.sdf_mesh(**bad_kwargs)
        assert exc.value.code == "bad_arg"
        assert exc.value.path and exc.value.fix

    # Out-of-range offset producing empty level set fails closed (never empty mesh)
    with pytest.raises(AttaError) as exc:
        implicit.sdf_mesh((0.02, 0.02, 0.02), field="gyroid", period=0.02,
                          thickness=0.0024, offset=5.0)
    assert exc.value.code == "empty_field"
    assert exc.value.path and exc.value.fix

    # gyroid_prepare rejects booleans
    with pytest.raises(AttaError) as exc:
        lattice._gyroid_prepare(
            {"size": [20, 20, 20], "cell_size": 10, "thickness": 1.2, "resolution": True},
            1e-3, "features[0]")
    assert exc.value.code == "bad_arg"
