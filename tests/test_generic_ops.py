# -*- coding: utf-8 -*-
"""Tests for atta.families.generic_ops (hull, minkowski, polyhedron, heightmap, deform, morph, projection_extrude)."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ["ATTA_FAMILIES_STRICT"] = "1"

from atta import families, features, spec as spec_mod  # noqa: E402
from atta.errors import AttaError  # noqa: E402
from atta.mesh import canonical_hash, topology  # noqa: E402

families.load(strict=True)


def _eval(feats, **kw):
    return features.evaluate(feats, **kw)


def _assert_closed_manifold(mesh, *, expected_genus=None, expected_components=1):
    topo = topology(mesh)
    assert not topo["empty"], topo
    assert topo["closed"], topo
    assert topo["non_manifold_edges"] == 0, topo
    assert topo["inconsistent_winding_edges"] == 0, topo
    assert topo["degenerate_faces"] == 0, topo
    if expected_components is not None:
        assert topo["n_components"] == expected_components, topo
    if expected_genus is not None:
        assert topo["genus"] == expected_genus, topo
    assert mesh.volume() > 0.0
    return topo


# ---------------------------------------------------------------- hull

def test_hull_box_plus_sphere_and_capsule_analytic():
    # 1. Hull of box + separated sphere -> closed genus-0 solid, deterministic.
    feats = [
        {"id": "b", "op": "box", "args": {"size": [20, 20, 20]}},
        {"id": "s", "op": "sphere", "args": {"radius": 10, "center": [30, 0, 0], "segments": 48, "rings": 24}},
        {"id": "h", "op": "hull", "args": {}, "inputs": ["b", "s"]},
    ]
    m1 = _eval(feats)
    m2 = _eval(feats)
    _assert_closed_manifold(m1, expected_genus=0)
    assert canonical_hash(m1) == canonical_hash(m2)
    # Disjoint box (8000 mm3) and sphere (4188.79 mm3) with 10mm gap in between:
    # hull strictly exceeds sum of both volumes and encloses [ -10..40, -10..10, -10..10 ].
    vol_mm3 = m1.volume() * 1e9
    assert vol_mm3 > 8000.0 + (4.0 / 3.0) * math.pi * 1000.0
    bb = m1.bbox()
    assert abs(bb["size"][0] * 1e3 - 50.0) < 1e-6
    assert abs(bb["size"][1] * 1e3 - 20.0) < 1e-6
    assert abs(bb["size"][2] * 1e3 - 20.0) < 1e-6

    # 2. Analytic capsule volume: hull of two spheres R=10 separated by L=30.
    caps_feats = [
        {"id": "s1", "op": "sphere", "args": {"radius": 10, "center": [0, 0, 0], "segments": 64, "rings": 32}},
        {"id": "s2", "op": "sphere", "args": {"radius": 10, "center": [30, 0, 0], "segments": 64, "rings": 32}},
        {"id": "h", "op": "hull", "args": {}, "inputs": ["s1", "s2"]},
    ]
    mc = _eval(caps_feats)
    _assert_closed_manifold(mc, expected_genus=0)
    analytic_caps = (4.0 / 3.0) * math.pi * 10.0**3 + math.pi * 10.0**2 * 30.0
    assert abs(mc.volume() * 1e9 - analytic_caps) / analytic_caps < 0.005

    # 3. Single-input hull of concave torus closes the hole (genus 0).
    tor_feats = [
        {"id": "t", "op": "torus", "args": {"major_radius": 20, "minor_radius": 5}},
        {"id": "h", "op": "hull", "args": {}, "inputs": ["t"]},
    ]
    mt = _eval(tor_feats)
    _assert_closed_manifold(mt, expected_genus=0)
    assert mt.volume() > _eval([tor_feats[0]]).volume()


def test_hull_errors_and_budgets():
    with pytest.raises(AttaError) as exc_budget:
        _eval([
            {"id": "s", "op": "sphere", "args": {"radius": 10, "segments": 32, "rings": 16}},
            {"id": "h", "op": "hull", "args": {"max_vertices": 10}, "inputs": ["s"]},
        ])
    assert exc_budget.value.code == "budget_exceeded"
    assert exc_budget.value.fix

    # Flat/degenerate input fails closed.
    from atta.families.generic_ops import _hull_from_points
    with pytest.raises(AttaError) as exc_flat:
        _hull_from_points(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]))
    assert exc_flat.value.code == "hull_degenerate"
    assert exc_flat.value.fix

    # Fewer than 4 unique points fails closed.
    with pytest.raises(AttaError) as exc_few:
        _hull_from_points(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert exc_few.value.code == "hull_degenerate"


# ---------------------------------------------------------------- minkowski

def test_minkowski_box_plus_sphere_analytic_and_box_plus_box():
    L, W, H, R = 20.0, 30.0, 40.0, 5.0
    feats = [
        {"id": "b", "op": "box", "args": {"size": [L, W, H]}},
        {"id": "s", "op": "sphere", "args": {"radius": R, "segments": 48, "rings": 24}},
        {"id": "m", "op": "minkowski", "args": {}, "inputs": ["b", "s"]},
    ]
    m1 = _eval(feats)
    m2 = _eval(feats)
    _assert_closed_manifold(m1, expected_genus=0)
    assert canonical_hash(m1) == canonical_hash(m2)
    assert m1.meta["exact"] is True
    assert m1.meta["minkowski"]["method"] == "convex_hull_pairwise_sum"

    analytic = (
        L * W * H
        + 2.0 * R * (L * W + W * H + H * L)
        + math.pi * R * R * (L + W + H)
        + (4.0 / 3.0) * math.pi * R**3
    )
    assert abs(m1.volume() * 1e9 - analytic) / analytic < 0.002

    # Exact box + box = 30 x 30 x 30 = 27000 mm3.
    bb_mesh = _eval([
        {"id": "a", "op": "box", "args": {"size": [20, 20, 20]}},
        {"id": "b", "op": "box", "args": {"size": [10, 10, 10]}},
        {"id": "m", "op": "minkowski", "args": {}, "inputs": ["a", "b"]},
    ])
    _assert_closed_manifold(bb_mesh, expected_genus=0)
    assert abs(bb_mesh.volume() * 1e9 - 27000.0) < 1e-4


def test_minkowski_nonconvex_exact_and_sdf_approx():
    # L-bracket (+) box(10, 10, 10): exact analytic volume = 48000 mm3.
    l_feats = [
        {"id": "b1", "op": "box", "args": {"size": [40, 10, 20], "center": [20, 5, 10]}},
        {"id": "b2", "op": "box", "args": {"size": [10, 40, 20], "center": [5, 20, 10]}},
        {"id": "L", "op": "union", "args": {}, "inputs": ["b1", "b2"]},
        {"id": "t", "op": "box", "args": {"size": [10, 10, 10]}},
    ]
    exact_m = _eval(l_feats + [{"id": "m", "op": "minkowski", "args": {"method": "exact"}, "inputs": ["L", "t"]}])
    _assert_closed_manifold(exact_m, expected_genus=0)
    assert exact_m.meta["exact"] is True
    assert abs(exact_m.volume() * 1e9 - 48000.0) / 48000.0 < 1e-4

    sdf_m = _eval(l_feats + [{"id": "m", "op": "minkowski", "args": {"method": "sdf", "resolution": 40}, "inputs": ["L", "t"]}])
    _assert_closed_manifold(sdf_m, expected_genus=0)
    assert sdf_m.meta["exact"] is False
    assert sdf_m.meta["minkowski"]["exact"] is False
    assert abs(sdf_m.volume() * 1e9 - 48000.0) / 48000.0 < 0.15


def test_minkowski_budgets_and_bad_args():
    with pytest.raises(AttaError) as exc_b:
        _eval([
            {"id": "a", "op": "sphere", "args": {"radius": 10, "segments": 32, "rings": 16}},
            {"id": "b", "op": "sphere", "args": {"radius": 5, "segments": 32, "rings": 16}},
            {"id": "m", "op": "minkowski", "args": {"method": "exact", "max_vertices": 50}, "inputs": ["a", "b"]},
        ])
    assert exc_b.value.code == "budget_exceeded"

    with pytest.raises(AttaError) as exc_fp:
        _eval([
            {"id": "b1", "op": "box", "args": {"size": [40, 10, 20], "center": [20, 5, 10]}},
            {"id": "b2", "op": "box", "args": {"size": [10, 40, 20], "center": [5, 20, 10]}},
            {"id": "L", "op": "union", "args": {}, "inputs": ["b1", "b2"]},
            {"id": "t", "op": "box", "args": {"size": [10, 10, 10]}},
            {"id": "m", "op": "minkowski", "args": {"method": "exact", "max_face_product": 10}, "inputs": ["L", "t"]},
        ])
    assert exc_fp.value.code == "budget_exceeded"


# ---------------------------------------------------------------- polyhedron

def test_polyhedron_cube_quads_and_winding_fix():
    pts = [
        [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
        [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10],
    ]
    # Deliberately flip two quad faces to test winding repair.
    bad_faces = [
        [0, 1, 2, 3],  # wrong winding (faces +Z instead of -Z)
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    m = _eval([{"id": "p", "op": "polyhedron", "args": {"points": pts, "faces": bad_faces}}])
    _assert_closed_manifold(m, expected_genus=0)
    assert abs(m.volume() * 1e9 - 1000.0) < 1e-6
    assert m.meta["polyhedron"]["fixed_winding"] is True

    # With fix_winding=False, the same input must fail closed with actionable error.
    with pytest.raises(AttaError) as exc_w:
        _eval([{"id": "p", "op": "polyhedron", "args": {
            "points": pts, "faces": bad_faces, "fix_winding": False,
        }}])
    assert exc_w.value.code in ("polyhedron_bad_winding", "polyhedron_inside_out")
    assert exc_w.value.fix

    # Unit scaling check (cm -> m).
    m_cm = _eval([{"id": "p", "op": "polyhedron", "args": {"points": pts, "faces": bad_faces}}], units="cm")
    assert abs(m_cm.volume() * 1e6 - 1000.0) < 1e-5


def test_polyhedron_errors_and_budgets():
    pts = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]]
    # Open shell (missing 4th face? wait <4 faces fails bad_arg; 4 faces with duplicate makes open+nonmanifold)
    open_faces = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 1, 2]]
    with pytest.raises(AttaError) as exc_open:
        _eval([{"id": "p", "op": "polyhedron", "args": {"points": pts, "faces": open_faces}}])
    assert exc_open.value.code in ("polyhedron_open", "polyhedron_non_manifold")

    # Out-of-range vertex index.
    with pytest.raises(AttaError) as exc_idx:
        _eval([{"id": "p", "op": "polyhedron", "args": {
            "points": pts,
            "faces": [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 99]],
        }}])
    assert exc_idx.value.code == "bad_arg"

    # Budget exceeded.
    cube_pts = [
        [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
        [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10],
    ]
    cube_faces = [
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
        [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
    ]
    with pytest.raises(AttaError) as exc_b:
        _eval([{"id": "p", "op": "polyhedron", "args": {
            "points": cube_pts, "faces": cube_faces, "max_triangles": 6,
        }}])
    assert exc_b.value.code == "budget_exceeded"


# ---------------------------------------------------------------- heightmap

def test_heightmap_analytic_volume_and_options():
    # 1. Sloped plane h(x, y) = 4 + 0.2*x + 0.4*y over 20x30 mm, base=2 mm.
    rows, cols = 5, 6
    dx, dy, base = 4.0, 7.5, 2.0  # Lx = 5*4 = 20 mm, Ly = 4*7.5 = 30 mm
    xs = np.linspace(0.0, 20.0, cols)
    ys = np.linspace(0.0, 30.0, rows)
    gx, gy = np.meshgrid(xs, ys)
    h = 4.0 + 0.2 * gx + 0.4 * gy
    m = _eval([{"id": "hm", "op": "heightmap", "args": {
        "heights": h.tolist(), "cell_size": [dx, dy], "base": base,
    }}])
    _assert_closed_manifold(m, expected_genus=0)
    # Exact bilinear prism sum: sum_{cells} dx * dy * (cell_mean + base).
    cell_means = 0.25 * (h[:-1, :-1] + h[:-1, 1:] + h[1:, :-1] + h[1:, 1:])
    analytic_mm3 = float(np.sum(dx * dy * (cell_means + base)))
    assert abs(m.volume() * 1e9 - analytic_mm3) < 1e-6

    # 2. Non-planar wavy surface via `size=[Lx, Ly]` and `invert=True`.
    wavy = 10.0 + 2.0 * np.sin(gx * 0.2) * np.cos(gy * 0.15)
    mw = _eval([{"id": "hm", "op": "heightmap", "args": {
        "heights": wavy.tolist(), "size": [20.0, 30.0], "base": 15.0,
        "invert": True, "center": False,
    }}])
    _assert_closed_manifold(mw, expected_genus=0)
    h_inv = -wavy
    inv_means = 0.25 * (h_inv[:-1, :-1] + h_inv[:-1, 1:] + h_inv[1:, :-1] + h_inv[1:, 1:])
    expected_wavy = float(np.sum(dx * dy * (inv_means + 15.0)))
    assert abs(mw.volume() * 1e9 - expected_wavy) < 1e-6
    bb = mw.bbox()
    assert abs(bb["min"][0]) < 1e-9 and abs(bb["min"][1]) < 1e-9

    # 3. Errors & budgets.
    with pytest.raises(AttaError) as exc_deg:
        _eval([{"id": "hm", "op": "heightmap", "args": {
            "heights": [[0.0, 1.0], [1.0, 2.0]], "base": 0.0,
        }}])
    assert exc_deg.value.code == "heightmap_degenerate"

    with pytest.raises(AttaError) as exc_b:
        _eval([{"id": "hm", "op": "heightmap", "args": {
            "heights": np.full((12, 12), 5.0).tolist(), "max_triangles": 50,
        }}])
    assert exc_b.value.code == "budget_exceeded"


# ---------------------------------------------------------------- deform

def test_deform_twist_preserves_volume_and_all_modes():
    # 1. Twisted box preserves volume (det J = 1).
    feats = [
        {"id": "b", "op": "box", "args": {"size": [20, 20, 60]}},
        {"id": "d", "op": "deform", "args": {
            "mode": "twist", "axis": "z", "angle_deg": 90.0, "max_edge": 4.0,
        }, "inputs": ["b"]},
    ]
    mt1 = _eval(feats)
    mt2 = _eval(feats)
    _assert_closed_manifold(mt1, expected_genus=0)
    assert canonical_hash(mt1) == canonical_hash(mt2)
    assert len(mt1.faces) > 12
    assert abs(mt1.volume() * 1e9 - 24000.0) / 24000.0 < 0.005

    # 2. Bend: symmetric bar bent 60 deg preserves volume closely.
    mb = _eval([
        {"id": "b", "op": "box", "args": {"size": [10, 10, 60]}},
        {"id": "d", "op": "deform", "args": {
            "mode": "bend", "axis": "z", "bend_axis": "x", "angle_deg": 60.0, "max_edge": 4.0,
        }, "inputs": ["b"]},
    ])
    _assert_closed_manifold(mb, expected_genus=0)
    assert abs(mb.volume() * 1e9 - 6000.0) / 6000.0 < 0.005

    # Extreme bend folding through itself fails closed.
    with pytest.raises(AttaError) as exc_fold:
        _eval([
            {"id": "b", "op": "box", "args": {"size": [30, 10, 40]}},
            {"id": "d", "op": "deform", "args": {
                "mode": "bend", "axis": "z", "bend_axis": "x", "angle_deg": 360.0,
            }, "inputs": ["b"]},
        ])
    assert exc_fold.value.code == "deform_self_intersection"

    # 3. Taper: frustum volume formula = H/3 * (A0 + sqrt(A0*A1) + A1).
    mp = _eval([
        {"id": "b", "op": "box", "args": {"size": [20, 20, 40]}},
        {"id": "d", "op": "deform", "args": {
            "mode": "taper", "axis": "z", "start_scale": 1.0, "end_scale": 0.5, "max_edge": 5.0,
        }, "inputs": ["b"]},
    ])
    _assert_closed_manifold(mp, expected_genus=0)
    analytic_taper = (40.0 / 3.0) * (400.0 + 200.0 + 100.0)
    assert abs(mp.volume() * 1e9 - analytic_taper) / analytic_taper < 0.002

    # 4. Lattice FFD: move top 4 control points by +10 mm in Z.
    offsets = [[0, 0, 0]] * 8
    for idx in (1, 3, 5, 7):  # k=1 corresponds to +Z corner in (2, 2, 2) order
        offsets[idx] = [0, 0, 10.0]
    ml = _eval([
        {"id": "b", "op": "box", "args": {"size": [20, 20, 20]}},
        {"id": "d", "op": "deform", "args": {
            "mode": "lattice", "lattice": {"shape": [2, 2, 2], "offsets": offsets},
            "max_edge": 10.0,
        }, "inputs": ["b"]},
    ])
    _assert_closed_manifold(ml, expected_genus=0)
    assert abs(ml.bbox()["size"][2] * 1e3 - 30.0) < 1e-6
    assert abs(ml.volume() * 1e9 - 12000.0) / 12000.0 < 1e-4

    # 5. Budget check.
    with pytest.raises(AttaError) as exc_b:
        _eval([
            {"id": "b", "op": "box", "args": {"size": [20, 20, 20]}},
            {"id": "d", "op": "deform", "args": {
                "mode": "twist", "angle_deg": 45.0, "max_edge": 0.1, "max_triangles": 100,
            }, "inputs": ["b"]},
        ])
    assert exc_b.value.code == "budget_exceeded"


# ---------------------------------------------------------------- morph

def test_morph_spheres_analytic_and_errors():
    # Sphere r=10 <-> r=20 at t=0.5 -> sphere r=15.
    feats = [
        {"id": "a", "op": "sphere", "args": {"radius": 10, "segments": 32, "rings": 16}},
        {"id": "b", "op": "sphere", "args": {"radius": 20, "segments": 32, "rings": 16}},
        {"id": "m", "op": "morph", "args": {"t": 0.5, "resolution": 36}, "inputs": ["a", "b"]},
    ]
    mm1 = _eval(feats)
    mm2 = _eval(feats)
    _assert_closed_manifold(mm1, expected_genus=0)
    assert canonical_hash(mm1) == canonical_hash(mm2)
    analytic = (4.0 / 3.0) * math.pi * 15.0**3
    assert abs(mm1.volume() * 1e9 - analytic) / analytic < 0.03

    # Endpoints t=0 and t=1 match inputs exactly.
    m0 = _eval([feats[0], feats[1], {"id": "m", "op": "morph", "args": {"t": 0.0}, "inputs": ["a", "b"]}])
    assert abs(m0.volume() * 1e9 - _eval([feats[0]]).volume() * 1e9) < 1e-6

    # Far-apart disjoint shapes without align_centers fail with morph_empty; with align_centers succeed.
    far_a = {"id": "a", "op": "box", "args": {"size": [10, 10, 10], "center": [-50, 0, 0]}}
    far_b = {"id": "b", "op": "sphere", "args": {"radius": 8, "center": [50, 0, 0]}}
    with pytest.raises(AttaError) as exc_empty:
        _eval([far_a, far_b, {"id": "m", "op": "morph", "args": {"t": 0.5, "resolution": 24}, "inputs": ["a", "b"]}])
    assert exc_empty.value.code == "morph_empty"

    aligned = _eval([far_a, far_b, {"id": "m", "op": "morph", "args": {"t": 0.5, "resolution": 24, "align_centers": True}, "inputs": ["a", "b"]}])
    _assert_closed_manifold(aligned, expected_genus=0)

    # Budget and bad t.
    with pytest.raises(AttaError) as exc_t:
        _eval([feats[0], feats[1], {"id": "m", "op": "morph", "args": {"t": 1.5}, "inputs": ["a", "b"]}])
    assert exc_t.value.code == "bad_arg"

    with pytest.raises(AttaError) as exc_bud:
        _eval([feats[0], feats[1], {"id": "m", "op": "morph", "args": {"t": 0.5, "resolution": 32, "max_triangles": 10}, "inputs": ["a", "b"]}])
    assert exc_bud.value.code in ("budget_exceeded", "sdf_too_dense")


# ---------------------------------------------------------------- projection_extrude

def test_projection_extrude_torus_hole_and_cut():
    # Projecting torus(R=20, r=5) onto XY and extruding 10 mm preserves the hole (genus=1).
    feats = [
        {"id": "t", "op": "torus", "args": {"major_radius": 20, "minor_radius": 5, "seg_major": 64, "seg_minor": 24}},
        {"id": "p", "op": "projection_extrude", "args": {"thickness": 10, "cut": False}, "inputs": ["t"]},
    ]
    mp = _eval(feats)
    _assert_closed_manifold(mp, expected_genus=1)
    analytic = math.pi * (25.0**2 - 15.0**2) * 10.0
    assert abs(mp.volume() * 1e9 - analytic) / analytic < 0.01
    assert abs(mp.bbox()["size"][2] * 1e3 - 10.0) < 1e-6

    # Cut=True slice at z=0 on same torus gives the same annular cross-section.
    mc = _eval([
        feats[0],
        {"id": "p", "op": "projection_extrude", "args": {"thickness": 10, "cut": True, "cut_offset": 0.0}, "inputs": ["t"]},
    ])
    _assert_closed_manifold(mc, expected_genus=1)
    assert abs(mc.volume() * 1e9 - analytic) / analytic < 0.01

    # Cut missing the solid fails closed.
    with pytest.raises(AttaError) as exc_miss:
        _eval([
            {"id": "b", "op": "box", "args": {"size": [10, 10, 10]}},
            {"id": "p", "op": "projection_extrude", "args": {"thickness": 5, "cut": True, "cut_offset": 50}, "inputs": ["b"]},
        ])
    assert exc_miss.value.code == "projection_empty"

    # Input triangle budget fails closed.
    with pytest.raises(AttaError) as exc_bud:
        _eval([
            feats[0],
            {"id": "p", "op": "projection_extrude", "args": {"thickness": 5, "max_input_triangles": 10}, "inputs": ["t"]},
        ])
    assert exc_bud.value.code == "budget_exceeded"


# ---------------------------------------------------------------- spec validation + bad args

def test_spec_validation_and_bad_args():
    spec = {
        "spec_version": "3.0",
        "name": "genops_smoke",
        "model": {
            "kind": "graph",
            "units": "mm",
            "features": [
                {"id": "b", "op": "box", "args": {"size": [20, 20, 20]}},
                {"id": "s", "op": "sphere", "args": {"radius": 10, "center": [25, 0, 0]}},
                {"id": "h", "op": "hull", "args": {}, "inputs": ["b", "s"]},
                {"id": "mk", "op": "minkowski", "args": {"method": "auto"}, "inputs": ["b", "s"]},
                {"id": "ph", "op": "polyhedron", "args": {
                    "points": [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]],
                    "faces": [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
                }},
                {"id": "hm", "op": "heightmap", "args": {
                    "heights": [[2, 3], [4, 5]], "cell_size": 5, "base": 1,
                }},
                {"id": "d", "op": "deform", "args": {"mode": "twist", "angle_deg": 30}, "inputs": ["b"]},
                {"id": "mp", "op": "morph", "args": {"t": 0.5, "resolution": 16}, "inputs": ["b", "s"]},
                {"id": "p", "op": "projection_extrude", "args": {"thickness": 5, "cut": True}, "inputs": ["d"]},
            ],
        },
    }
    res = spec_mod.validate_spec(spec)
    assert res["ok"], res

    # Boolean passed where number/length is expected must fail.
    with pytest.raises(AttaError):
        _eval([
            {"id": "b", "op": "box", "args": {"size": [10, 10, 10]}},
            {"id": "p", "op": "projection_extrude", "args": {"thickness": True}, "inputs": ["b"]},
        ])

    # NaN in heights must fail.
    with pytest.raises(AttaError):
        _eval([
            {"id": "hm", "op": "heightmap", "args": {"heights": [[1.0, float("nan")], [1.0, 2.0]]}},
        ])
