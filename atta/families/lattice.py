# -*- coding: utf-8 -*-
"""Lattice families — triply-periodic minimal surfaces sampled as real solids.

``lattice.gyroid_slab`` is a **gyroid** lattice slab: the implicit gyroid
surface ``F = sin x cos y + sin y cos z + sin z cos x = t`` is turned into a
*thick sheet* (a real printable solid, not a zero-thickness surface) and cut
out of a rectangular slab.

Why a solid field and not "mesh the surface then thicken"
---------------------------------------------------------
``surfaces.thicken`` needs an **open** patch and offsets along vertex normals;
on a periodic minimal surface that produces self-intersections wherever the
sheet's normal curvature radius is smaller than the half thickness.  Instead the
solid is defined directly by one scalar field and extracted with marching
tetrahedra::

    phi(x) = max( |F(x) - t_bias| - h ,  box(x) )

``phi < 0`` is exactly "inside the gyroid sheet **and** inside the slab", so the
isosurface of ``phi`` is the finished solid incl. the four flat cut walls, and it
is closed by construction because ``phi > 0`` on the whole sampled box border.

The physical wall thickness is a true metric quantity in metres: instead of
comparing a dimensionless field value against a metric box or dividing by a
single median ``|grad F|``, :func:`atta.sdfkit.tpms_sheet_distance` computes
the exact analytic gradient and applies 3 damped Newton closest-point
projections on the narrow band around ``F = t_bias``.  Both ``sheet`` and
``box`` enter ``max(sheet, box)`` in metres, so extracted wall thickness and
volume track Weyl's (1939) tubular volume formula
``V = N_cells * (A_cell * a^2 * t + (pi * chi / 6) * t^3)`` across sizes and
cell counts.

Analytic genus
--------------
By Schoen (1970, NASA TN D-5541) and Grosse-Brauckmann (1997), Schoen's gyroid
(space group ``Ia-3d``) has genus 3 in its primitive BCC cell and genus 5
(``chi = -8``) in the conventional cubic cell ``[0, a]^3`` (2 BCC cells).
For a ``kx * ky * kz`` integer-cell block clipped with phase ``(-0.25, 0, 0)``,
Mayer-Vietoris / Lefschetz duality yields the exact closed-mesh genus
``8*kx*ky*kz - 2*kx - 2*ky + 1`` (see :func:`atta.sdfkit.tpms_block_genus`).
``lattice.gyroid_slab`` declares this as a parametric :class:`GenusRule`.

Watertightness
--------------
Marching tetrahedra (6 tets per cell, fan around one body diagonal) is used
instead of marching cubes: every tetrahedron emits a closed polygon, and because
the 6-tet split is translation invariant the triangulation induced on each
shared cube face matches exactly on both sides, so no cracks can appear.  Shared
vertices are deduplicated by an *exact* integer edge key (the two grid node
ids), not by a distance tolerance — no weld slack, no bucket-boundary cracks.
Triangle winding is fixed per triangle against the inside→outside centroid
direction, which is globally consistent because it follows ``grad phi``.
"""

from __future__ import annotations

import math

import numpy as np

from .. import sdfkit
from ..errors import AttaError
from ..kit import finish, quality_level
from ..mesh import Mesh
from ..registry import GenusRule, recipe

__all__ = []

#: lattice pattern phase, in cells.  The slab is cut a quarter cell off the
#: gyroid's inversion centre: cutting exactly through it chips two ~0.015 mm3
#: corner slivers off the sheet (measured: 3 components), and they are only
#: held to the body through material outside the slab.
_PHASE = (0.25, 0.0, 0.0)

#: minimum samples across the sheet.  Two things were measured on the way here:
#: below ~2.2 voxels the marching surface invents/merges handles (genus 462 at
#: 1.1 voxels), and below ~2.5 voxels the extracted sheet is systematically
#: thinner than ordered (ray-sampled p50 2.51 mm for a 3.0 mm sheet, plus a bogus
#: 0.07 mm "wall" where marching cuts a corner off).  At 3 voxels the same build
#: measures p50 3.03 mm (with metric Newton distance), so 3.0 is the floor.
_MIN_VOX = 3.0
_MAX_RES = 30
_MAX_SAMPLES = 2_500_000


def marching_tets(field, origin, spacing, **budgets) -> Mesh:
    """Compatibility entry point for bounded, exact-edge field extraction."""
    from ..sdf import marching_tets as extract
    return extract(field, origin, spacing, **budgets)


def _gyroid(x, y, z, k):
    """Gyroid field and its analytic gradient magnitude."""
    sx, cx = np.sin(k * x), np.cos(k * x)
    sy, cy = np.sin(k * y), np.cos(k * y)
    sz, cz = np.sin(k * z), np.cos(k * z)
    f = sx * cy + sy * cz + sz * cx
    gx = k * (cx * cy - sz * sx)
    gy = k * (-sx * sy + cy * cz)
    gz = k * (-sy * sz + cz * cx)
    return f, np.sqrt(gx * gx + gy * gy + gz * gz)


def _num_arg(val, path, *, positive=False):
    if isinstance(val, (bool, np.bool_)):
        raise AttaError("bad_arg", "%s must be a number, not a bool" % path,
                        path=path, fix="pass a numeric length in mm")
    try:
        out = float(val)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AttaError("bad_arg", "%s must be a finite number" % path,
                        path=path, fix="pass a finite numeric value") from exc
    if not math.isfinite(out) or (positive and out <= 0.0):
        raise AttaError("bad_arg", "%s must be %s" % (path, "positive and finite" if positive else "finite"),
                        path=path, fix="pass a valid positive finite number")
    return out


def _gyroid_slab_expected_genus(display: dict) -> int | None:
    """Analytic genus of ``lattice.gyroid_slab`` for integer cell-count blocks.

    Returns ``8*nx*ny*nz - 2*nx - 2*ny + 1`` when ``(slab_l, slab_w, slab_h)``
    are integer multiples of ``cell_size`` (within ``1e-4``) and thickness /
    ``t_bias`` sit in the non-pinching sheet regime; returns ``None`` for
    fractional cell counts where no closed-form block genus holds.
    """
    cell = float(display["cell_size"])
    if cell <= 0.0:
        return None
    thick = float(display["thickness"])
    t_bias = float(display.get("t_bias", 0.0))
    if abs(t_bias) > 0.35 or thick < 0.09 * cell - 1e-9 or thick > 0.16 * cell + 1e-9:
        return None
    counts = []
    for key in ("slab_l", "slab_w", "slab_h"):
        ratio = float(display[key]) / cell
        k = int(round(ratio))
        if k < 1 or abs(ratio - k) > 1e-4:
            return None
        counts.append(k)
    return sdfkit.tpms_block_genus("gyroid", counts[0], counts[1], counts[2],
                                   offset=t_bias, mode="sheet")


def _gyroid_slab_mesh(cell, thickness, t_bias, slab, res, quality="draft", *,
                      max_samples=_MAX_SAMPLES, max_triangles=2_000_000):
    """Sample metric gyroid sheet ``d_M(x) <= thickness/2`` inside ``slab`` (m)."""
    cell = _num_arg(cell, "cell_size", positive=True)
    thickness = _num_arg(thickness, "thickness", positive=True)
    t_bias = _num_arg(t_bias, "t_bias")
    if thickness >= 0.5 * cell:
        raise AttaError("bad_arg", "gyroid thickness must be < 0.5 * cell_size",
                        path="thickness", fix="use thickness = 0.12 * cell_size")
    if abs(t_bias) > 0.8:
        raise AttaError("bad_arg", "gyroid t_bias must be within [-0.8, 0.8]",
                        path="t_bias", fix="use t_bias = 0")
    if isinstance(res, (bool, np.bool_)) or int(res) != res:
        raise AttaError("bad_arg", "gyroid resolution must be an integer",
                        path="resolution", fix="use an integer resolution >= 8")
    slab_arr = tuple(_num_arg(s, "slab", positive=True) for s in slab)
    if len(slab_arr) != 3:
        raise AttaError("bad_arg", "slab must have three positive lengths",
                        path="slab", fix="pass (L, W, H) in metres")
    L, W, H = slab_arr
    half = np.array([L, W, H], dtype=np.float64) * 0.5

    rq = int(quality_level(quality, int(res), int(res) + 2, int(res) + 5))
    sp_req = cell / float(max(8, rq))
    try:
        spacing, shape = sdfkit.feature_spacing(
            slab_arr, thickness, min_voxels=_MIN_VOX,
            requested_spacing=sp_req, max_samples=max_samples)
    except AttaError as exc:
        if exc.code == "sdf_too_dense":
            raise AttaError(
                "lattice_too_dense",
                "gyroid slab needs too many samples at spacing %.4f mm" % (
                    min(sp_req, thickness / _MIN_VOX) * 1e3),
                path="resolution",
                fix="lower resolution, use a thicker sheet or a smaller slab",
                details=exc.details,
            ) from exc
        raise
    res_eff = int(round(cell / spacing))
    margin = 2.5 * spacing

    org = -half - margin
    ax = org[0] + np.arange(shape[0]) * spacing
    ay = org[1] + np.arange(shape[1]) * spacing
    az = org[2] + np.arange(shape[2]) * spacing
    GX, GY, GZ = np.meshgrid(ax, ay, az, indexing="ij")

    # Both terms below are METRES: tpms_sheet_distance projects narrow-band
    # samples onto F(p) - t_bias = 0 with exact analytic gradients + 3 damped
    # Newton steps, so ``sheet`` is a real Euclidean signed distance in metres.
    phase = (-float(_PHASE[0]), -float(_PHASE[1]), -float(_PHASE[2]))
    sheet = sdfkit.tpms_sheet_distance(
        "gyroid", GX, GY, GZ, cell, thickness, offset=t_bias, phase=phase,
        band_margin=max(3.5 * spacing, 0.5 * thickness), newton_steps=3)
    box = np.maximum(np.maximum(np.abs(GX) - half[0], np.abs(GY) - half[1]),
                     np.abs(GZ) - half[2])
    phi = np.maximum(sheet, box)
    phi, n_pruned = sdfkit.prune_boundary_slivers(phi, box, spacing)

    boundary_min = min(float(a.min()) for a in (
        phi[0], phi[-1], phi[:, 0], phi[:, -1], phi[:, :, 0], phi[:, :, -1]))
    if boundary_min <= 0.0:
        phi = sdfkit.cap_boundary(phi, spacing)
    if not bool((phi < 0.0).any()):
        raise AttaError(
            "empty_field",
            "gyroid slab has no interior volume for these parameters",
            path="t_bias",
            fix="keep abs(t_bias) <= 0.3 and thickness >= 0.09 * cell_size",
        )

    mesh = marching_tets(phi, org, spacing, max_triangles=max_triangles)
    vol_est = sdfkit.tpms_sheet_volume_estimate("gyroid", slab_arr, cell, thickness, offset=t_bias)
    g_ref = float(2.0 * math.pi / cell * 1.158)
    return mesh, {
        "res_eff": res_eff, "shape": shape, "spacing": spacing,
        "g_ref": g_ref, "cells": [L / cell, W / cell, H / cell],
        "metric_distance": True, "slivers_pruned": int(n_pruned),
        "expected_volume_m3": float(vol_est),
    }


@recipe(
    "lattice.gyroid_slab", "陀螺晶格板",
    "隐式陀螺面 sin x cos y + sin y cos z + sin z cos x = t 经解析梯度+牛顿投影"
    "按真实公制壁厚（m）加厚，再与板坯盒求交；行进四面体提取，闭合单实体，"
    "整数单元块亏格由 8*nx*ny*nz - 2*nx - 2*ny + 1 参数化给出。",
    params={
        "slab_l": {"type": "float", "default": 66.0, "min": 20.0, "max": 200.0,
                   "unit": "mm", "group": "slab"},
        "slab_w": {"type": "float", "default": 44.0, "min": 20.0, "max": 200.0,
                   "unit": "mm", "group": "slab"},
        "slab_h": {"type": "float", "default": 22.0, "min": 8.0, "max": 120.0,
                   "unit": "mm", "group": "slab"},
        "cell_size": {"type": "float", "default": 22.0, "min": 8.0, "max": 48.0,
                      "unit": "mm", "group": "lattice",
                      "description": "晶格周期（一个完整陀螺单元）"},
        "thickness": {"type": "float", "default": 3.0, "min": 1.0, "max": 8.0,
                      "unit": "mm", "group": "lattice",
                      "description": "陀螺薄片实体壁厚"},
        "t_bias": {"type": "float", "default": 0.0, "min": -0.8, "max": 0.8,
                   "unit": "none", "group": "lattice",
                   "description": "等值面偏移（0 = 平衡陀螺面）"},
        "resolution": {"type": "int", "default": 14, "min": 8, "max": 32,
                       "unit": "count", "group": "quality", "advanced": True,
                       "description": "每个晶格周期的采样数（壁厚会再抬升它）"},
    },
    presets={
        "safe": {"cell_size": 24.0, "thickness": 3.6, "slab_l": 72.0,
                 "slab_w": 48.0, "slab_h": 24.0, "resolution": 12, "t_bias": 0.0},
        "nominal": {"cell_size": 22.0, "thickness": 3.0, "slab_l": 66.0,
                    "slab_w": 44.0, "slab_h": 22.0, "resolution": 14, "t_bias": 0.0},
        "hero": {"cell_size": 18.0, "thickness": 2.2, "slab_l": 54.0,
                 "slab_w": 36.0, "slab_h": 18.0, "resolution": 20, "t_bias": 0.1},
    },
    tags=["lattice", "gyroid", "implicit", "marching_tets", "porous"],
    expected_genus=GenusRule(
        _gyroid_slab_expected_genus,
        text="8*nx*ny*nz - 2*nx - 2*ny + 1 (integer cell counts)",
    ),
    material="nylon",
    constraints=[
        {"id": "wall_min_ratio", "expr": "thickness >= 0.09 * cell_size",
         "severity": "error", "message": "薄片壁厚小于 0.09 倍晶格周期，采样无法分辨",
         "fix": "thickness = 0.12 * cell_size"},
        {"id": "wall_max_ratio", "expr": "thickness <= 0.16 * cell_size",
         "severity": "error", "message": "薄片过厚会自熔、改变晶格拓扑（实测 genus 漂移）",
         "fix": "thickness = 0.12 * cell_size"},
        {"id": "bias_band", "expr": "abs(t_bias) <= 0.3",
         "severity": "error", "message": "等值面偏移过大会把薄片撕裂成多个实体",
         "fix": "t_bias = 0.0"},
        {"id": "slab_holds_cells", "expr": "slab_l >= 1.5 * cell_size and slab_w >= 1.5 * cell_size",
         "severity": "warn", "message": "板坯小于 1.5 个晶格周期，边界效应占主导",
         "fix": "slab_l = 3 * cell_size"},
        {"id": "genus_envelope",
         "expr": "abs(slab_l / cell_size - round(slab_l / cell_size)) <= 0.01"
                 " and abs(slab_w / cell_size - round(slab_w / cell_size)) <= 0.01"
                 " and abs(slab_h / cell_size - round(slab_h / cell_size)) <= 0.01",
         "severity": "warn",
         "message": "板坯不是整数个晶格单元：解析亏格公式 8*nx*ny*nz - 2*nx - 2*ny + 1 仅在整数单元块下定义",
         "fix": "slab_l = round(slab_l / cell_size) * cell_size"},
    ],
)
def _build_gyroid_slab(v, quality="draft"):
    cell = _num_arg(v["cell_size"], "cell_size", positive=True)
    thick = _num_arg(v["thickness"], "thickness", positive=True)
    slab = (
        _num_arg(v["slab_l"], "slab_l", positive=True),
        _num_arg(v["slab_w"], "slab_w", positive=True),
        _num_arg(v["slab_h"], "slab_h", positive=True),
    )
    # Unit guard: recipe bounds are cell_size <= 48 mm (0.048 m) and
    # slab <= 200 mm (0.20 m).  Reject raw-mm numbers passed without resolve().
    if cell > 0.5 or max(slab) > 2.0:
        raise AttaError(
            "unit_mismatch",
            "lattice.gyroid_slab builder expects metres (got cell_size=%.4g m, max_slab=%.4g m)" % (
                cell, max(slab)),
            path="cell_size",
            fix="convert millimetres to metres via recipe.resolve().values before calling build_shape",
        )
    mesh, info = _gyroid_slab_mesh(
        cell, thick, float(v["t_bias"]), slab, int(v["resolution"]), quality)
    mesh.meta.update({"lattice": "gyroid", **info})
    return mesh, finish(smooth_angle_deg=35.0, material="nylon")


# ---------------------------------------------------------------- graph op

def _gyroid_op(*, size, cell_size, thickness, t_bias=0.0, resolution=14):
    """Composable gyroid slab for feature graphs (all values in metres)."""
    mesh, info = _gyroid_slab_mesh(
        float(cell_size), float(thickness), float(t_bias),
        tuple(float(x) for x in size), int(resolution), "draft")
    mesh.meta.update({"lattice": "gyroid", **info})
    return mesh


def _gyroid_prepare(args, scale, path):
    raw_size = args.get("size")
    if isinstance(raw_size, (list, tuple, np.ndarray)):
        for idx, item in enumerate(np.asarray(raw_size, dtype=object).ravel()):
            if isinstance(item, (bool, np.bool_)):
                raise AttaError("bad_arg", "gyroid.size must not contain booleans",
                                path="%s.args.size[%d]" % (path, idx),
                                fix="pass [x,y,z] numbers in mm")
    try:
        size = np.asarray(raw_size, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise AttaError("bad_arg", "gyroid.size must be three positive lengths",
                        path=path + ".args.size", fix="pass [x,y,z] in mm") from exc
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise AttaError("bad_arg", "gyroid.size must be three positive lengths",
                        path=path + ".args.size", fix="pass [x,y,z] in mm")
    cell = _num_arg(args.get("cell_size"), path + ".args.cell_size", positive=True)
    thick = _num_arg(args.get("thickness"), path + ".args.thickness", positive=True)
    if thick >= 0.5 * cell:
        raise AttaError("bad_arg", "gyroid thickness must be < 0.5 * cell_size",
                        path=path + ".args.thickness",
                        fix="use thickness = 0.12 * cell_size")
    t_bias = _num_arg(args.get("t_bias", 0.0), path + ".args.t_bias")
    if abs(t_bias) > 0.8:
        raise AttaError("bad_arg", "gyroid t_bias must be within [-0.8, 0.8]",
                        path=path + ".args.t_bias", fix="use t_bias = 0")
    raw_res = args.get("resolution", 14)
    if isinstance(raw_res, (bool, np.bool_)):
        raise AttaError("bad_arg", "gyroid resolution must be an integer, not a bool",
                        path=path + ".args.resolution", fix="use resolution = 14")
    res_f = _num_arg(raw_res, path + ".args.resolution", positive=True)
    if abs(res_f - round(res_f)) > 1e-9:
        raise AttaError("bad_arg", "gyroid resolution must be an integer",
                        path=path + ".args.resolution", fix="use resolution = 14")
    res = int(round(res_f))
    if res < 8:
        raise AttaError("bad_arg", "gyroid resolution must be >= 8",
                        path=path + ".args.resolution", fix="use resolution = 14")
    return {"size": size * scale, "cell_size": cell * scale,
            "thickness": thick * scale, "t_bias": t_bias,
            "resolution": res}


_gyroid_op.prepare = _gyroid_prepare
try:
    from ..features import register_op as _register_op

    _register_op(
        "gyroid",
        {"required": ("size", "cell_size", "thickness"),
         "optional": ("t_bias", "resolution"), "arity": 0,
         "doc": "Gyroid lattice slab: size=[x,y,z] mm, cell_size mm, "
                "thickness mm, t_bias, resolution."},
        _gyroid_op,
    )
except Exception:  # pragma: no cover - a family must never break the engine
    pass
