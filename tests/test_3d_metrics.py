# -*- coding: utf-8 -*-
"""Unit & analytic tests for 3D metrology and mesh comparison metrics."""

from __future__ import annotations

import time

import numpy as np
import pytest

from atta import metrology, perception, primitives, probe
from atta.mesh import Mesh


def _box_mesh(size_mm=(100.0, 100.0, 100.0), center_mm=(0.0, 0.0, 0.0)) -> Mesh:
    sx, sy, sz = (float(x) / 1000.0 for x in size_mm)
    cx, cy, cz = (float(x) / 1000.0 for x in center_mm)
    x0, x1 = cx - sx * 0.5, cx + sx * 0.5
    y0, y1 = cy - sy * 0.5, cy + sy * 0.5
    z0, z1 = cz - sz * 0.5, cz + sz * 0.5
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)
    f = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    return Mesh(v, f)


def _open_box_mesh(size_mm: float = 100.0) -> Mesh:
    m = _box_mesh((size_mm, size_mm, size_mm))
    # Remove the top face (+Z, faces index 2 and 3) -> 4 open edges of size_mm each.
    keep = np.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    return Mesh(m.verts.copy(), m.faces[keep].copy())


def _bowtie_self_intersecting_mesh() -> Mesh:
    """Closed prism whose top face is mirrored across X=0, forming a bowtie."""
    v = np.array([
        [-0.02, -0.01, 0.0],
        [0.02, -0.01, 0.0],
        [0.02, 0.01, 0.0],
        [-0.02, 0.01, 0.0],
        [0.02, -0.01, 0.04],
        [-0.02, -0.01, 0.04],
        [-0.02, 0.01, 0.04],
        [0.02, 0.01, 0.04],
    ], dtype=np.float64)
    f = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    return Mesh(v, f)


# ---------------------------------------------------------------- SIR / DangEL / FluxEE

def test_self_intersection_clean_box_vs_bowtie():
    clean = _box_mesh()
    res_clean = metrology.self_intersection(clean)
    assert res_clean["intersecting_pairs"] == 0
    assert res_clean["self_intersection_ratio"] == 0.0
    assert res_clean["truncated"] is False
    assert res_clean["exact"] is True

    bowtie = _bowtie_self_intersecting_mesh()
    res_bad = metrology.self_intersection(bowtie)
    assert res_bad["intersecting_pairs"] > 0
    assert 0.0 < res_bad["self_intersection_ratio"] <= 1.0
    assert res_bad["truncated"] is False
    assert res_bad["exact"] is True


def test_self_intersection_coplanar_and_truncation():
    # Two coplanar overlapping triangles that share no vertices
    v = np.array([
        [0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0],
        [0.005, 0.005, 0.0], [0.025, 0.005, 0.0], [0.005, 0.025, 0.0],
    ], dtype=np.float64)
    f = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    res = metrology.self_intersection(Mesh(v, f))
    assert res["intersecting_pairs"] == 1
    assert res["self_intersection_ratio"] == pytest.approx(1.0)

    # max_pairs=0 forces truncation cleanly
    res_trunc = metrology.self_intersection(Mesh(v, f), max_pairs=0)
    assert res_trunc["checked_pairs"] == 0
    assert res_trunc["truncated"] is True
    assert res_trunc["exact"] is False


def test_self_intersection_20k_performance():
    sphere = primitives.sphere(0.05, segments=280)
    assert len(sphere.faces) >= 15000
    t0 = time.perf_counter()
    res = metrology.self_intersection(sphere)
    elapsed = time.perf_counter() - t0
    assert res["intersecting_pairs"] == 0
    assert res["self_intersection_ratio"] == 0.0
    assert elapsed < 0.3, "20k-ish sphere took %.3fs (>= 0.3s)" % elapsed


def test_dangling_edge_length_closed_vs_open_box():
    assert metrology.dangling_edge_length_mm(_box_mesh((100.0, 100.0, 100.0))) == 0.0
    # 100 mm cube missing top face has 4 boundary edges of 100 mm = 400 mm
    assert metrology.dangling_edge_length_mm(_open_box_mesh(100.0)) == pytest.approx(400.0)


def test_flux_enclosure_error_closed_open_and_flipped():
    clean = _box_mesh()
    assert metrology.flux_enclosure_error(clean) < 1e-12

    # Globally flipped normals (inside-out solid) -> orient residual 2.0
    flipped_all = Mesh(clean.verts.copy(), clean.faces[:, ::-1].copy())
    assert metrology.flux_enclosure_error(flipped_all) == pytest.approx(2.0)

    # Single flipped face -> 2 * (1/12) = 1/6 ~= 0.1667
    faces_one = clean.faces.copy()
    faces_one[0] = faces_one[0, ::-1]
    flipped_one = Mesh(clean.verts.copy(), faces_one)
    assert metrology.flux_enclosure_error(flipped_one) == pytest.approx(1.0 / 6.0, abs=1e-6)

    # Open box missing 1 of 6 faces -> 1/5 = 0.2
    open_box = _open_box_mesh()
    assert metrology.flux_enclosure_error(open_box) == pytest.approx(0.2, abs=1e-6)


def test_measure_includes_new_metrics_and_toggle():
    m = _box_mesh()
    full = metrology.measure(m)
    assert full["dangling_edge_length_mm"] == 0.0
    assert full["flux_enclosure_error"] < 1e-12
    assert full["self_intersection"]["self_intersection_ratio"] == 0.0

    fast = metrology.measure(m, self_intersect=False)
    assert "dangling_edge_length_mm" in fast
    assert "flux_enclosure_error" in fast
    assert "self_intersection" not in fast


# ---------------------------------------------------------------- 3D compare & perception.diff

def test_compare_meshes_3d_analytic_boxes():
    a = _box_mesh((100.0, 100.0, 100.0), (0.0, 0.0, 0.0))
    same = probe.compare_meshes_3d(a, a, n_samples=2000, seed=0)
    assert same["chamfer_mm"] == pytest.approx(0.0, abs=1e-6)
    assert same["chamfer_a_to_b_mm"] == pytest.approx(0.0, abs=1e-6)
    assert same["chamfer_b_to_a_mm"] == pytest.approx(0.0, abs=1e-6)
    assert same["hausdorff_p95_mm"] == pytest.approx(0.0, abs=1e-6)
    assert same["hausdorff_max_mm"] == pytest.approx(0.0, abs=1e-6)
    assert same["volumetric_iou"] == pytest.approx(1.0, abs=1e-6)
    assert same["volume_ref_mm3"] == pytest.approx(1_000_000.0)
    assert same["volume_target_mm3"] == pytest.approx(1_000_000.0)
    assert same["n_samples"] == 2000
    assert same["exact"] is False

    # Shifted by 50 mm along X: intersection=500k mm3, union=1500k mm3 -> IoU = 1/3
    b = _box_mesh((100.0, 100.0, 100.0), (50.0, 0.0, 0.0))
    shifted = probe.compare_meshes_3d(a, b, n_samples=2000, seed=0)
    assert shifted["volumetric_iou"] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert shifted["chamfer_mm"] > 5.0
    assert shifted["hausdorff_max_mm"] == pytest.approx(50.0, abs=1.0)

    # Disjoint boxes -> IoU = 0.0
    c = _box_mesh((100.0, 100.0, 100.0), (150.0, 0.0, 0.0))
    disjoint = probe.compare_meshes_3d(a, c, n_samples=1000, seed=0)
    assert disjoint["volumetric_iou"] == 0.0
    assert disjoint["hausdorff_max_mm"] == pytest.approx(150.0, abs=1.0)


def test_compare_meshes_3d_analytic_concentric_spheres():
    s1 = primitives.sphere(0.020, segments=48)  # R = 20 mm
    s2 = primitives.sphere(0.022, segments=48)  # R = 22 mm
    res = probe.compare_meshes_3d(s1, s2, n_samples=3000, seed=42)
    # Concentric spheres differ radially by exact 2.0 mm everywhere
    assert res["chamfer_mm"] == pytest.approx(2.0, abs=0.15)
    assert res["chamfer_a_to_b_mm"] == pytest.approx(2.0, abs=0.15)
    assert res["chamfer_b_to_a_mm"] == pytest.approx(2.0, abs=0.15)
    assert res["hausdorff_p95_mm"] == pytest.approx(2.0, abs=0.2)
    assert res["hausdorff_max_mm"] == pytest.approx(2.0, abs=0.25)
    expected_iou = (20.0 / 22.0) ** 3
    assert res["volumetric_iou"] == pytest.approx(expected_iou, abs=0.02)


def test_compare_meshes_3d_voxel_fallback_for_non_manifold():
    open_a = _open_box_mesh(100.0)
    open_b = _open_box_mesh(100.0)
    res = probe.compare_meshes_3d(open_a, open_b, n_samples=500, seed=0)
    assert res["chamfer_mm"] == pytest.approx(0.0, abs=1e-6)
    assert 0.8 <= res["volumetric_iou"] <= 1.0


def test_perception_diff_wires_metrics_3d():
    a = _box_mesh((40.0, 40.0, 40.0), (0.0, 0.0, 0.0))
    b = _box_mesh((40.0, 40.0, 40.0), (10.0, 0.0, 0.0))
    with_3d = perception.diff(a, b, views=("front",), res=12, compare_3d=True)
    assert "metrics_3d" in with_3d
    assert with_3d["metrics_3d"]["volumetric_iou"] == pytest.approx(0.75 / 1.25, abs=1e-3)
    assert with_3d["metrics_3d"]["exact"] is False

    without_3d = perception.diff(a, b, views=("front",), res=12, compare_3d=False)
    assert "metrics_3d" not in without_3d
