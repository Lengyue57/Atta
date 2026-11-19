# -*- coding: utf-8 -*-
"""Composable implicit fields — ``sdf`` as a first-class feature-graph op.

Any field from the triply-periodic minimal surface family (``gyroid``,
``schwarz_p``, ``diamond``, ``neovius``), a Mandelbulb distance-estimator
shell, or a primitive SDF can be sampled on a bounded regular grid, clipped to
a box or sphere, and extracted as a watertight manifold mesh by the Marching
Tetrahedra kernel in :mod:`atta.sdf`.

Key invariants
--------------
* **Metric wall thickness**: TPMS sheets use
  :func:`atta.sdfkit.tpms_sheet_distance` (exact analytic gradient plus damped
  Newton closest-point projection) so ``thickness`` is a real metre length
  rather than a dimensionless level-set offset.
* **Feature-derived resolution**: voxel spacing is bounded by
  ``thickness / _MIN_VOX`` (never silently clamped below 3 voxels across a thin
  sheet) and fails closed with ``AttaError(code="sdf_too_dense")`` if the
  resulting grid exceeds ``_MAX_SAMPLES``.
* **Closed clipping domain**: ``clip="sphere"`` intersects both the declared
  ``size`` box and the sphere of radius ``clip_radius`` over their shared
  bounding box with ``2.5 * spacing`` positive padding, so ``phi < 0`` can
  never spill across the sampling-grid boundary.
* **Boundary sliver pruning**: clipped TPMS sheets suppress tiny (< 5 %)
  14-connected fragments chipped off at the clip wall before meshing.

Everything public is in millimetres; ``_sdf_prepare`` converts to metres at the
feature-graph boundary, exactly like ``gyroid``.
"""

from __future__ import annotations

import math

import numpy as np

from .. import sdf, sdfkit
from ..errors import AttaError
from ..kit import quality_level
from ..mesh import Mesh
from .lattice import _MAX_RES, _MAX_SAMPLES, _MIN_VOX, marching_tets

__all__ = ["sdf_mesh"]

_TPMS_FIELDS = ("gyroid", "schwarz_p", "diamond", "neovius")
_FIELDS = ("gyroid", "schwarz_p", "diamond", "neovius", "mandelbulb",
           "sphere", "box", "torus")
_CLIPS = ("box", "sphere")


def _num(val, path, *, positive=False):
    if isinstance(val, (bool, np.bool_)):
        raise AttaError("bad_arg", "%s must be a number, not a bool" % path,
                        path=path, fix="pass a finite numeric value")
    try:
        out = float(val)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AttaError("bad_arg", "%s must be a finite number" % path,
                        path=path, fix="pass a finite numeric value") from exc
    if not math.isfinite(out) or (positive and out <= 0.0):
        raise AttaError("bad_arg", "%s must be %s" % (path, "positive and finite" if positive else "finite"),
                        path=path, fix="pass a valid positive finite number")
    return out


def _tpms(field, X, Y, Z, period, offset):
    """Compatibility helper returning nodal TPMS field ``F - offset``."""
    F, _gx, _gy, _gz = sdfkit.tpms_eval(
        field, X, Y, Z, period, offset=offset, phase=(-0.25, 0.0, 0.0))
    return F


def _mandelbulb(X, Y, Z, power=8.0, iterations=8):
    """Distance estimate for the Mandelbulb, in normalised units (non-negative)."""
    zx, zy, zz = X.copy(), Y.copy(), Z.copy()
    dr = np.ones_like(zx)
    r = np.zeros_like(zx)
    for _ in range(int(iterations)):
        r = np.sqrt(zx * zx + zy * zy + zz * zz)
        # Escape-time iteration: stop updating points that already escaped.
        # Without this the eighth power overflows float64 on the outer grid.
        active = r <= 4.0
        safe_r = np.minimum(r, 4.0)
        theta = np.arctan2(np.sqrt(zx * zx + zy * zy), zz)
        phi = np.arctan2(zy, zx)
        zr = np.where(active, np.power(safe_r, power - 1.0), 0.0)
        dr = np.where(active, zr * power * dr + 1.0, dr)
        theta = np.where(active, theta * power, theta)
        phi = np.where(active, phi * power, phi)
        nx = np.sin(theta) * np.cos(phi) * zr * safe_r + X
        ny = np.sin(theta) * np.sin(phi) * zr * safe_r + Y
        nz = np.cos(theta) * zr * safe_r + Z
        zx = np.where(active, nx, zx)
        zy = np.where(active, ny, zy)
        zz = np.where(active, nz, zz)
    r = np.sqrt(zx * zx + zy * zy + zz * zz)
    de = 0.5 * np.log(np.maximum(r, 1e-12)) * r / np.maximum(dr, 1e-12)
    return np.nan_to_num(de, nan=1.0, posinf=1.0, neginf=1.0)


def _primitive(field, X, Y, Z, size):
    half = 0.5 * np.asarray(size, dtype=np.float64)
    if field == "sphere":
        r = 0.5 * float(min(size))
        return np.sqrt(X * X + Y * Y + Z * Z) - r
    if field == "box":
        qx = np.abs(X) - half[0]
        qy = np.abs(Y) - half[1]
        qz = np.abs(Z) - half[2]
        outside = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2
                          + np.maximum(qz, 0) ** 2)
        inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0.0)
        return outside + inside
    # torus: major radius from the two largest dimensions, minor from the smallest
    s = sorted(float(v) for v in size)
    R = 0.5 * s[0] if len(set(s)) == 1 else 0.5 * min(s[0], s[1])
    R = max(R, 1e-9)
    r = max(0.18 * s[0], 1e-9)
    return np.sqrt((np.sqrt(X * X + Y * Y) - R) ** 2 + Z * Z) - r


def sdf_mesh(size, field="gyroid", period=None, thickness=None, offset=0.0,
             clip="box", clip_radius=None, resolution=16, power=8.0,
             quality="draft", *, max_samples=_MAX_SAMPLES,
             max_triangles=2_000_000) -> tuple[Mesh, dict]:
    """Sample an implicit field and return ``(mesh, info)`` in metres."""
    raw_size = np.asarray(size, dtype=object).reshape(-1)
    if raw_size.shape != (3,):
        raise AttaError("bad_arg", "sdf.size must be three positive lengths",
                        path="sdf.size", fix="pass [x,y,z] in mm")
    size = np.array([_num(v, "sdf.size[%d]" % i, positive=True)
                     for i, v in enumerate(raw_size)], dtype=np.float64)
    if field not in _FIELDS:
        raise AttaError("bad_arg", "sdf.field must be one of %s" % (", ".join(_FIELDS),),
                        path="sdf.field", fix="use field='gyroid'")
    if clip not in _CLIPS:
        raise AttaError("bad_arg", "sdf.clip must be 'box' or 'sphere'",
                        path="sdf.clip", fix="use clip='box'")
    offset = _num(offset, "sdf.offset")
    power = _num(power, "sdf.power", positive=True)
    if power > 32.0:
        raise AttaError("bad_arg", "sdf.power must be <= 32.0",
                        path="sdf.power", fix="use power in (0, 32] (8 is standard)")
    if isinstance(resolution, (bool, np.bool_)) or int(resolution) != resolution:
        raise AttaError("bad_arg", "sdf.resolution must be an integer",
                        path="sdf.resolution", fix="use an integer resolution >= 6")
    resolution = int(resolution)
    if resolution < 6:
        raise AttaError("bad_arg", "sdf.resolution must be >= 6",
                        path="sdf.resolution", fix="use resolution = 16")

    half = 0.5 * size
    period = _num(period, "sdf.period", positive=True) if period is not None else float(min(size))
    if thickness is None:
        # A sheet thinner than ~1/10 of the period needs high voxel density;
        # this default matches the lattice recipe's 0.12 * cell rule.
        thickness = 0.12 * period
    else:
        thickness = _num(thickness, "sdf.thickness", positive=True)

    is_tpms = field in _TPMS_FIELDS
    is_shell = is_tpms or field == "mandelbulb"
    if is_tpms and thickness >= 0.5 * period:
        raise AttaError("bad_arg", "sdf.thickness must be < 0.5 * period for TPMS sheets",
                        path="sdf.thickness", fix="use thickness <= 0.16 * period")

    if clip == "sphere":
        R_clip = (_num(clip_radius, "sdf.clip_radius", positive=True)
                  if clip_radius is not None else 0.5 * float(min(size)))
        # Intersect sphere(R_clip) with box(size) so the sampled domain always
        # stays within size and strictly encloses every phi < 0 point.
        dom_half = np.minimum(half, R_clip)
    else:
        R_clip = None
        dom_half = half
    dom_size = 2.0 * dom_half

    # resolution is "samples per period" for TPMS, "samples across the part"
    # for primitives; a thin TPMS sheet always gets at least _MIN_VOX voxels
    # across its wall, checked against max_samples (fail closed, no silent cap).
    rq = int(quality_level(quality, resolution, resolution + 2, resolution + 5))
    if is_tpms:
        sp_req = period / float(max(8, rq))
        spacing, shape = sdfkit.feature_spacing(
            dom_size, thickness, min_voxels=_MIN_VOX,
            requested_spacing=sp_req, max_samples=max_samples)
        res_eff = int(round(period / spacing))
    else:
        res_eff = int(min(_MAX_RES, max(8, rq)))
        spacing = float(min(dom_size)) / float(res_eff)
        shape = sdf.grid_shape(-dom_half, dom_half, spacing)
        if math.prod(shape) > int(max_samples):
            raise AttaError(
                "sdf_too_dense",
                "sdf needs %d samples at resolution %d" % (math.prod(shape), res_eff),
                path="sdf.resolution",
                fix="lower resolution or reduce domain size",
                details={"samples": math.prod(shape), "max_samples": int(max_samples)},
            )

    margin = 2.5 * spacing
    org = -dom_half - margin
    ax = org[0] + np.arange(shape[0]) * spacing
    ay = org[1] + np.arange(shape[1]) * spacing
    az = org[2] + np.arange(shape[2]) * spacing
    X, Y, Z = np.meshgrid(ax, ay, az, indexing="ij")

    box = np.maximum(np.maximum(np.abs(X) - half[0], np.abs(Y) - half[1]),
                     np.abs(Z) - half[2])
    if clip == "sphere":
        clip_sdf = np.maximum(box, np.sqrt(X * X + Y * Y + Z * Z) - R_clip)
    else:
        clip_sdf = box

    slivers_pruned = 0
    if is_tpms:
        spec = sdfkit.TPMS_SPECS[field]
        ph = spec["canonical_phase_cells"]
        if field == "gyroid":
            # Keep world-centred coordinates matching lattice.gyroid_slab.
            SX, SY, SZ = X, Y, Z
        else:
            # Anchor phase to the lower box corner so integer k*l*m blocks have
            # congruent boundary planes for both even and odd cell counts.
            SX, SY, SZ = X + half[0], Y + half[1], Z + half[2]
        sheet = sdfkit.tpms_sheet_distance(
            field, SX, SY, SZ, period, thickness, offset=offset, phase=ph,
            band_margin=max(3.5 * spacing, 0.5 * thickness), newton_steps=3)
        phi = np.maximum(sheet, clip_sdf)
        phi, slivers_pruned = sdfkit.prune_boundary_slivers(phi, clip_sdf, spacing)
    elif field == "mandelbulb":
        R = 0.5 * float(min(dom_size))
        phi = _mandelbulb(X / max(R, 1e-9), Y / max(R, 1e-9), Z / max(R, 1e-9),
                          power=power) * R - 0.5 * thickness
        phi = np.maximum(phi, clip_sdf)
    else:
        phi = np.maximum(_primitive(field, X, Y, Z, size), clip_sdf)

    boundary_min = min(float(a.min()) for a in (
        phi[0], phi[-1], phi[:, 0], phi[:, -1], phi[:, :, 0], phi[:, :, -1]))
    if boundary_min <= 0.0:
        phi = sdfkit.cap_boundary(phi, spacing)
    if not bool((phi < 0.0).any()):
        raise AttaError(
            "empty_field",
            "implicit field %r has no interior volume in the sampled domain" % (field,),
            path="sdf.offset",
            fix="reduce |offset| or increase thickness so the level set intersects the domain",
        )

    mesh = marching_tets(phi, org, spacing, max_triangles=max_triangles)
    info = {
        "field": field, "res_eff": res_eff, "spacing": spacing,
        "period": period, "clip": clip, "samples": list(shape),
        "slivers_pruned": int(slivers_pruned),
    }
    if is_shell:
        info["thickness"] = thickness
    if is_tpms:
        info["metric_distance"] = True
        info["expected_volume_m3"] = float(sdfkit.tpms_sheet_volume_estimate(
            field, dom_size, period, thickness, offset=offset))
        if clip == "box":
            ratios = [float(size[i]) / period for i in range(3)]
            ints = [int(round(r)) for r in ratios]
            if all(k >= 1 and abs(r - k) <= 1e-4 for r, k in zip(ratios, ints)):
                info["analytic_genus"] = sdfkit.tpms_block_genus(
                    field, ints[0], ints[1], ints[2], offset=offset, mode="sheet")
    return mesh, info


def _sdf_op(*, size, field="gyroid", period=None, thickness=None, offset=0.0,
            clip="box", clip_radius=None, resolution=16, power=8.0):
    mesh, info = sdf_mesh(
        tuple(float(x) for x in size), field=field, period=period,
        thickness=thickness, offset=offset, clip=clip,
        clip_radius=clip_radius, resolution=int(resolution), power=float(power),
        quality="draft")
    mesh.meta.update({"implicit": field, **info})
    return mesh


def _sdf_prepare(args, scale, path):
    raw_size = args.get("size")
    if isinstance(raw_size, (list, tuple, np.ndarray)):
        for idx, item in enumerate(np.asarray(raw_size, dtype=object).ravel()):
            if isinstance(item, (bool, np.bool_)):
                raise AttaError("bad_arg", "sdf.size must not contain booleans",
                                path="%s.args.size[%d]" % (path, idx),
                                fix="pass [x,y,z] numbers in mm")
    try:
        size = np.asarray(raw_size, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise AttaError("bad_arg", "sdf.size must be three positive lengths",
                        path=path + ".args.size", fix="pass [x,y,z] in mm") from exc
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise AttaError("bad_arg", "sdf.size must be three positive lengths",
                        path=path + ".args.size", fix="pass [x,y,z] in mm")
    raw_field = args.get("field", "gyroid")
    field = "gyroid" if raw_field is None else raw_field
    if not isinstance(field, str) or field not in _FIELDS:
        raise AttaError("bad_arg", "sdf.field must be one of %s" % (", ".join(_FIELDS),),
                        path=path + ".args.field", fix="use field='gyroid'")
    raw_clip = args.get("clip", "box")
    clip = "box" if raw_clip is None else raw_clip
    if not isinstance(clip, str) or clip not in _CLIPS:
        raise AttaError("bad_arg", "sdf.clip must be 'box' or 'sphere'",
                        path=path + ".args.clip", fix="use clip='box'")
    period = args.get("period")
    if period is not None:
        period = _num(period, path + ".args.period", positive=True) * scale
    thickness = args.get("thickness")
    if thickness is not None:
        thickness = _num(thickness, path + ".args.thickness", positive=True) * scale
    offset = _num(args.get("offset", 0.0), path + ".args.offset")
    radius = args.get("clip_radius")
    if radius is not None:
        radius = _num(radius, path + ".args.clip_radius", positive=True) * scale
    raw_res = args.get("resolution", 16)
    if isinstance(raw_res, (bool, np.bool_)):
        raise AttaError("bad_arg", "sdf.resolution must be an integer, not a bool",
                        path=path + ".args.resolution", fix="use resolution = 16")
    res_f = _num(raw_res, path + ".args.resolution", positive=True)
    if abs(res_f - round(res_f)) > 1e-9:
        raise AttaError("bad_arg", "sdf.resolution must be an integer",
                        path=path + ".args.resolution", fix="use resolution = 16")
    res = int(round(res_f))
    if res < 6:
        raise AttaError("bad_arg", "sdf.resolution must be >= 6",
                        path=path + ".args.resolution", fix="use resolution = 16")
    power = _num(args.get("power", 8.0), path + ".args.power", positive=True)
    return {"size": size * scale, "field": field, "period": period,
            "thickness": thickness, "offset": offset, "clip": clip,
            "clip_radius": radius, "resolution": res, "power": power}


_sdf_op.prepare = _sdf_prepare
try:
    from ..features import register_op as _register_op

    _register_op(
        "sdf",
        {"required": ("size",), "arity": 0,
         "optional": ("field", "period", "thickness", "offset", "clip",
                      "clip_radius", "resolution", "power"),
         "doc": "Implicit field: size=[x,y,z] mm; field=gyroid|schwarz_p|diamond|"
                "neovius|mandelbulb|sphere|box|torus; period mm; thickness mm; "
                "clip=box|sphere; resolution samples/period."},
        _sdf_op,
    )
except Exception:  # pragma: no cover - a family must never break the engine
    pass
