# -*- coding: utf-8 -*-
"""Tests for extended recipe families: lattice_ext, text_ext, tolerance_ext, assembly_ext."""

from __future__ import annotations

import os
import time

import pytest

os.environ["ATTA_FAMILIES_STRICT"] = "1"

from atta import families, probe, recipes, verify as verify_mod  # noqa: E402
from atta.errors import AttaError  # noqa: E402
from atta.families.tolerance_ext import iso286_limits_um, snap_fit_mechanics  # noqa: E402
from atta.families.text_ext import encode_braille_cells  # noqa: E402
from atta.mesh import topology, volume  # noqa: E402
from tests.test_coverage_gate import category_of, constraints_for  # noqa: E402

families.load(strict=True)

NEW_RECIPES = [
    "lattice.strut_cell",
    "lattice.tpms_block",
    "lattice.graded_gyroid",
    "text.braille_plate",
    "text.stencil_sign",
    "text.keychain_tag",
    "tolerance.fit_pair_gauge",
    "tolerance.snap_fit",
    "tolerance.hinge_gauge",
    "assembly.hinge_pin",
    "assembly.dovetail_joint",
    "assembly.clevis_pin",
]


def test_target_categories_have_at_least_three_recipes():
    by_cat: dict[str, list[str]] = {}
    for rec in recipes.all_recipes():
        by_cat.setdefault(category_of(rec.id), []).append(rec.id)
    for cat in ("lattice", "text", "tolerance", "assembly"):
        assert len(by_cat.get(cat, [])) >= 3, (cat, by_cat.get(cat, []))
    for rid in NEW_RECIPES:
        assert rid in recipes.REGISTRY


@pytest.mark.parametrize("rid", NEW_RECIPES)
@pytest.mark.parametrize("preset", ("safe", "nominal", "hero"))
def test_new_recipes_pass_all_presets(rid, preset):
    rec = recipes.get(rid)
    resolved = rec.resolve(preset=preset)
    errors = [i for i in resolved.issues if i.get("severity") == "error"]
    assert not resolved.blocked and not errors, (rid, preset, errors)

    t0 = time.perf_counter()
    built = rec.build_shape(resolved.values, quality="draft")
    dt = time.perf_counter() - t0
    mesh = built[0] if isinstance(built, tuple) else built
    assert dt < 4.0, (rid, preset, f"build took {dt:.2f}s")

    topo = topology(mesh)
    assert not topo["empty"], (rid, preset)
    assert topo["closed"], (rid, preset, topo)
    assert topo["non_manifold_edges"] == 0, (rid, preset, topo)
    assert topo["degenerate_faces"] == 0, (rid, preset, topo)
    assert topo["n_components"] == 1, (rid, preset, topo)
    assert topo["genus"] == rec.expected_genus, (rid, preset, topo["genus"], rec.expected_genus)

    cons = constraints_for(category_of(rid), rec)
    receipt = verify_mod.verify(mesh, constraints=cons, level="standard",
                                expected_genus=rec.expected_genus)
    failed = [c for c in receipt.get("checks", []) if not c.get("ok")]
    assert receipt.get("verdict") == "pass", (rid, preset, failed)


@pytest.mark.parametrize("topo_kind,expected_g", [
    ("octet", 91),
    ("bcc", 11),
    ("fcc", 43),
])
def test_strut_cell_topologies(topo_kind, expected_g):
    rec = recipes.get("lattice.strut_cell")
    res = rec.resolve({"topology": topo_kind, "nx": 2, "ny": 2, "nz": 1,
                       "cell_size": 16.0, "strut_d": 2.0})
    mesh, _ = rec.build_shape(res.values, quality="draft")
    topo = topology(mesh)
    assert topo["closed"] and topo["n_components"] == 1
    assert topo["degenerate_faces"] == 0
    assert topo["genus"] == expected_g
    assert mesh.meta["analytic_genus"] == expected_g


@pytest.mark.parametrize("surf,nz,expected_g", [
    ("schwarz_p", 2, 33),
    ("diamond", 1, 87),
    ("gyroid", 1, 25),
])
def test_tpms_block_surfaces_and_metric_thickness(surf, nz, expected_g):
    rec = recipes.get("lattice.tpms_block")
    res_thin = rec.resolve({"surface": surf, "nx": 2, "ny": 2, "nz": nz,
                            "cell_size": 18.0, "thickness": 2.0, "resolution": 14})
    res_thick = rec.resolve({"surface": surf, "nx": 2, "ny": 2, "nz": nz,
                             "cell_size": 18.0, "thickness": 2.6, "resolution": 14})
    m_thin, _ = rec.build_shape(res_thin.values, quality="draft")
    m_thick, _ = rec.build_shape(res_thick.values, quality="draft")
    for m in (m_thin, m_thick):
        t = topology(m)
        assert t["closed"] and t["n_components"] == 1
        assert t["genus"] == expected_g
    assert volume(m_thick) > volume(m_thin) * 1.15


def test_graded_gyroid_density_gradient():
    rec = recipes.get("lattice.graded_gyroid")
    res = rec.resolve({"cell_size": 20.0, "thickness_bottom": 3.1,
                       "thickness_top": 1.9, "nx": 2, "ny": 2, "nz": 1})
    mesh, _ = rec.build_shape(res.values, quality="draft")
    topo = topology(mesh)
    assert topo["closed"] and topo["n_components"] == 1 and topo["genus"] == 25
    # Bottom Z-slice has thicker walls -> larger cross-sectional solid area than top slice
    sec_bot = probe.section(mesh, plane="z", offset_mm=-7.0, res=32)
    sec_top = probe.section(mesh, plane="z", offset_mm=+7.0, res=32)
    area_bot = sum(abs(lp["area_mm2"]) for lp in sec_bot["loops"])
    area_top = sum(abs(lp["area_mm2"]) for lp in sec_top["loops"])
    assert area_bot > area_top


def test_braille_encoding_and_caption():
    cells = encode_braille_cells("a1")
    # 'a' -> (1,), numeric indicator -> (3,4,5,6), '1' -> (1,)
    assert cells == [(1,), (3, 4, 5, 6), (1,)]
    rec = recipes.get("text.braille_plate")
    res = rec.resolve(preset="hero")
    mesh, _ = rec.build_shape(res.values, quality="draft")
    assert mesh.meta["n_cells"] == len("exit")
    assert mesh.meta["n_dots"] == 12


@pytest.mark.parametrize("txt", ["808", "B4", "OPEN"])
def test_stencil_sign_bridges_all_islands(txt):
    rec = recipes.get("text.stencil_sign")
    res = rec.resolve({"text": txt, "length": 150.0, "cap_height": 22.0})
    mesh, _ = rec.build_shape(res.values, quality="draft")
    topo = topology(mesh)
    assert topo["closed"] and topo["n_components"] == 1
    assert topo["degenerate_faces"] == 0


def test_keychain_tag_modes():
    rec = recipes.get("text.keychain_tag")
    for mode in ("emboss", "engrave"):
        res = rec.resolve({"mode": mode, "text": "LAB"})
        mesh, _ = rec.build_shape(res.values, quality="draft")
        topo = topology(mesh)
        assert topo["closed"] and topo["n_components"] == 1 and topo["genus"] == 1


def test_iso286_lookup_and_fit_pair_variants():
    g6 = iso286_limits_um(10.0, "H7/g6")
    assert g6["hole_limits_um"] == [15, 0]
    assert g6["shaft_limits_um"] == [-5, -14]
    assert g6["fit_type"] == "clearance"

    e9 = iso286_limits_um(6.0, "H8/e9")
    assert e9["hole_limits_um"] == [18, 0]
    assert e9["shaft_limits_um"] == [-20, -50]
    assert e9["min_clearance_um"] == 20 and e9["max_clearance_um"] == 68

    p6 = iso286_limits_um(16.0, "H7/p6")
    assert p6["hole_limits_um"] == [18, 0]
    assert p6["shaft_limits_um"] == [29, 18]
    assert p6["fit_type"] == "interference"
    assert p6["min_clearance_um"] == -29
    assert p6["max_clearance_um"] == 0
    assert p6["mean_clearance_um"] < 0
    p6_hi = iso286_limits_um(30.0, "H7/p6")
    assert p6_hi["shaft_limits_um"] == [35, 22]
    assert p6_hi["max_clearance_um"] == -1
    assert p6_hi["fit_type"] == "interference"

    rec = recipes.get("tolerance.fit_pair_gauge")
    for fc in ("H8/e9", "H11/h11"):
        res = rec.resolve({"fit_class": fc, "nominal_d": 12.0})
        mesh, _ = rec.build_shape(res.values, quality="draft")
        assert mesh.meta["iso286"]["fit_class"] == fc
        assert topology(mesh)["genus"] == 0


def test_snap_fit_mechanics_and_strain_constraint():
    # Uniform rectangular cantilever (taper_ratio=1.0): eps = 1.5 * t * Y / L^2
    m = snap_fit_mechanics(20.0, 2.0, 8.0, 1.5, taper_ratio=1.0)
    assert m["max_strain"] == pytest.approx(1.5 * 2.0 * 1.5 / (20.0 ** 2), rel=1e-4)

    rec = recipes.get("tolerance.snap_fit")
    bad = rec.resolve({"beam_len": 12.0, "beam_t": 3.0, "undercut": 4.0, "allow_strain_pct": 2.0})
    assert bad.blocked
    strain_issues = [i for i in bad.issues if i["id" if "id" in i else "code"] == "strain_limit"]
    assert strain_issues and strain_issues[0].get("fix")


def test_hinge_gauge_clearance_section():
    rec = recipes.get("tolerance.hinge_gauge")
    res = rec.resolve(preset="nominal")
    mesh, _ = rec.build_shape(res.values, quality="draft")
    assert mesh.meta["bore_d_mm"] - mesh.meta["pin_d_mm"] == pytest.approx(
        2.0 * mesh.meta["radial_clearance_mm"], abs=1e-4)
    sec = probe.section(mesh, plane="x", offset_mm=0.0, res=32)
    assert sec["n_loops"] == 3


def test_assembly_metadata_and_constraints():
    for rid, exp_g in (("assembly.hinge_pin", 7),
                       ("assembly.dovetail_joint", 0),
                       ("assembly.clevis_pin", 4)):
        rec = recipes.get(rid)
        res = rec.resolve(preset="nominal")
        mesh, _ = rec.build_shape(res.values, quality="draft")
        assert "assembly" in mesh.meta
        assert topology(mesh)["genus"] == exp_g
        assert topology(mesh)["n_components"] == 1

    # Verify constraints block invalid geometry with actionable fix
    dt = recipes.get("assembly.dovetail_joint")
    bad = dt.resolve({"base_h": 4.0, "gate_h": 4.0})
    assert bad.blocked
    assert any(i.get("fix") for i in bad.issues if i.get("severity") == "error")
