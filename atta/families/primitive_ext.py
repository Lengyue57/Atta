# -*- coding: utf-8 -*-
"""Extended primitive families: rounded cylinder, hollow tube, tapered wedge.

Every primitive builds cleanly via ``surfaces.revolve`` or ``surfaces.loft``
with zero boolean fragility, returning a watertight single-component solid in
metres (public recipe parameters are in mm/deg).

Prior art / CAD conventions:
* ``primitive.rounded_cylinder`` — comparable to BOSL2 ``cyl(rounding1,
  rounding2, chamfer1, chamfer2)`` (BSD-2), with independent top/bottom corner
  radii and selectable circular fillet vs 45-degree chamfer.
* ``primitive.tube`` — comparable to BOSL2 ``tube(od, id, h, ochamfer,
  ichamfer)``, a genus-1 hollow cylindrical sleeve with independent outer and
  inner end edge treatments.
* ``primitive.wedge`` — comparable to FreeCAD ``Part.makeWedge`` / OpenSCAD
  tapered prism, supporting asymmetric X-offset top face, independent top
  width, and edge chamfering.
"""

from __future__ import annotations

import math

import numpy as np

from .. import surfaces as _S
from ..errors import AttaError
from ..kit import finish, quality_level as _q
from ..registry import recipe

__all__ = ["_build_rounded_cylinder", "_build_tube", "_build_wedge"]

_EPS = 1e-12


def _corner_arc(r_corner, z_corner, r_in, z_in, size, mode, steps, reverse=False):
    """Generate 2D profile points (r, z) for a chamfer or circular fillet corner.

    ``(r_corner, z_corner)`` is the sharp outer/inner corner; ``(r_in, z_in)``
    are the unit inward directions along r and z from the corner.
    """
    s = float(size)
    if s <= 1e-9:
        return [(float(r_corner), float(z_corner))]
    p0 = (float(r_corner + r_in * s), float(z_corner))
    p1 = (float(r_corner), float(z_corner + z_in * s))
    if str(mode).lower() == "chamfer":
        pts = [p0, p1]
        return pts[::-1] if reverse else pts
    # Circular fillet arc tangent to both edges, centre offset inward by s.
    cr = float(r_corner + r_in * s)
    cz = float(z_corner + z_in * s)
    a0 = math.atan2(p0[1] - cz, p0[0] - cr)
    a1 = math.atan2(p1[1] - cz, p1[0] - cr)
    # Take the shorter quarter-turn arc between a0 and a1.
    da = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    k = max(2, int(steps))
    angs = np.linspace(a0, a0 + da, k + 1)
    pts = [(cr + s * math.cos(a), cz + s * math.sin(a)) for a in angs]
    return pts[::-1] if reverse else pts


# ------------------------------------------------------------ rounded cylinder

@recipe(
    "primitive.rounded_cylinder", "倒圆角/倒角圆柱",
    "上下端面可独立设置圆角半径或 45° 倒角的实心圆柱（参照 BOSL2 cyl 规范），"
    "通过闭合二维截面回转一次成形，无布尔缝隙，恒为 genus 0 闭合单实体。",
    params={
        "diameter": {"type": "float", "default": 40.0, "min": 4.0, "max": 500.0,
                     "unit": "mm", "group": "body", "description": "圆柱外径"},
        "height": {"type": "float", "default": 50.0, "min": 2.0, "max": 600.0,
                   "unit": "mm", "group": "body", "description": "圆柱总高"},
        "top_radius": {"type": "float", "default": 5.0, "min": 0.0, "max": 200.0,
                       "unit": "mm", "group": "edge", "description": "顶边圆角/倒角尺寸"},
        "bottom_radius": {"type": "float", "default": 5.0, "min": 0.0, "max": 200.0,
                          "unit": "mm", "group": "edge", "description": "底边圆角/倒角尺寸"},
        "edge_mode": {"type": "enum", "default": "fillet",
                      "enum": ["fillet", "chamfer"], "unit": "none",
                      "group": "edge", "description": "fillet=圆弧过渡，chamfer=45° 倒角"},
        "segments": {"type": "int", "default": 48, "min": 12, "max": 192,
                     "unit": "count", "group": "quality", "advanced": True},
        "corner_steps": {"type": "int", "default": 8, "min": 2, "max": 32,
                         "unit": "count", "group": "quality", "advanced": True},
    },
    presets={
        "safe": {"diameter": 40.0, "height": 45.0, "top_radius": 4.0,
                 "bottom_radius": 4.0, "edge_mode": "chamfer", "segments": 40,
                 "corner_steps": 6},
        "nominal": {"diameter": 40.0, "height": 50.0, "top_radius": 5.0,
                    "bottom_radius": 5.0, "edge_mode": "fillet", "segments": 48,
                    "corner_steps": 8},
        "hero": {"diameter": 56.0, "height": 64.0, "top_radius": 10.0,
                 "bottom_radius": 4.0, "edge_mode": "fillet", "segments": 96,
                 "corner_steps": 14},
    },
    tags=["primitive", "cylinder", "fillet", "chamfer", "revolve"],
    expected_genus=0, material="abs_white",
    constraints=[
        {"id": "radial_fits",
         "expr": "max(top_radius, bottom_radius) <= 0.45 * diameter",
         "severity": "error",
         "message": "端面圆角/倒角超过半径的 90%，顶/底平面会退化为尖点",
         "fix": "top_radius = 0.25 * diameter"},
        {"id": "axial_fits",
         "expr": "top_radius + bottom_radius <= 0.90 * height",
         "severity": "error",
         "message": "上下圆角/倒角之和超过高度的 90%，圆柱侧壁会消失",
         "fix": "top_radius = 0.25 * height; bottom_radius = 0.25 * height"},
        {"id": "segments_smooth",
         "expr": "segments >= 16",
         "severity": "warn",
         "message": "环向分段不足 16，圆柱外观呈明显多边形",
         "fix": "segments = 32"},
    ],
)
def _build_rounded_cylinder(v, quality="draft"):
    r = 0.5 * float(v["diameter"])
    h = float(v["height"])
    rt = float(v["top_radius"])
    rb = float(v["bottom_radius"])
    mode = str(v.get("edge_mode") or "fillet").lower()
    seg = max(12, _q(quality, int(v["segments"]), int(v["segments"]) * 3 // 2,
                     int(v["segments"]) * 2))
    steps = max(2, _q(quality, int(v["corner_steps"]), int(v["corner_steps"]) + 4,
                      int(v["corner_steps"]) * 2))

    if rt + rb > 0.98 * h or max(rt, rb) > 0.98 * r:
        raise AttaError("edge_too_large",
                        "top/bottom edge radii leave no cylindrical wall or end cap",
                        path="top_radius", fix="reduce top_radius / bottom_radius")

    pts = [(0.0, 0.0)]
    pts.extend(_corner_arc(r, 0.0, -1.0, 1.0, rb, mode, steps, reverse=False))
    pts.extend(_corner_arc(r, h, -1.0, -1.0, rt, mode, steps, reverse=True))
    pts.append((0.0, h))
    prof = np.asarray(pts, dtype=np.float64)
    mesh = _S.revolve(prof, segments=seg, caps=False)
    mesh.meta.update({"kind": "rounded_cylinder", "edge_mode": mode})
    return mesh, finish(smooth_angle_deg=35.0, material="abs_white")


# ------------------------------------------------------------------------ tube

@recipe(
    "primitive.tube", "倒角/圆角空心管",
    "带内外径及独立内外倒角/圆角的标准空心管基元（参照 BOSL2 tube 规范），"
    "由闭合环状截面一次回转生成，恒为 genus 1 闭合单实体。",
    params={
        "outer_d": {"type": "float", "default": 50.0, "min": 6.0, "max": 500.0,
                    "unit": "mm", "group": "body", "description": "管外径"},
        "inner_d": {"type": "float", "default": 30.0, "min": 2.0, "max": 480.0,
                    "unit": "mm", "group": "body", "description": "管内孔径"},
        "height": {"type": "float", "default": 60.0, "min": 2.0, "max": 600.0,
                   "unit": "mm", "group": "body", "description": "管高"},
        "outer_edge": {"type": "float", "default": 2.5, "min": 0.0, "max": 50.0,
                       "unit": "mm", "group": "edge", "description": "外沿上下倒角/圆角尺寸"},
        "inner_edge": {"type": "float", "default": 1.5, "min": 0.0, "max": 50.0,
                       "unit": "mm", "group": "edge", "description": "内孔上下倒角/圆角尺寸"},
        "edge_mode": {"type": "enum", "default": "chamfer",
                      "enum": ["chamfer", "fillet"], "unit": "none",
                      "group": "edge", "description": "chamfer=45° 倒角，fillet=圆弧过渡"},
        "segments": {"type": "int", "default": 48, "min": 12, "max": 192,
                     "unit": "count", "group": "quality", "advanced": True},
        "corner_steps": {"type": "int", "default": 6, "min": 2, "max": 24,
                         "unit": "count", "group": "quality", "advanced": True},
    },
    presets={
        "safe": {"outer_d": 50.0, "inner_d": 28.0, "height": 50.0,
                 "outer_edge": 2.0, "inner_edge": 1.0, "edge_mode": "chamfer",
                 "segments": 40, "corner_steps": 5},
        "nominal": {"outer_d": 50.0, "inner_d": 30.0, "height": 60.0,
                    "outer_edge": 2.5, "inner_edge": 1.5, "edge_mode": "chamfer",
                    "segments": 48, "corner_steps": 6},
        "hero": {"outer_d": 64.0, "inner_d": 38.0, "height": 80.0,
                 "outer_edge": 3.5, "inner_edge": 2.5, "edge_mode": "fillet",
                 "segments": 96, "corner_steps": 10},
    },
    tags=["primitive", "tube", "pipe", "hollow", "chamfer", "revolve"],
    expected_genus=1, material="aluminium",
    constraints=[
        {"id": "wall_positive",
         "expr": "outer_d - inner_d >= 3.0",
         "severity": "error",
         "message": "单边壁厚不足 1.5 mm，空心管过薄",
         "fix": "inner_d = outer_d - 6.0"},
        {"id": "edge_fits_wall",
         "expr": "outer_edge + inner_edge <= 0.42 * (outer_d - inner_d)",
         "severity": "error",
         "message": "内外倒角/圆角之和超过壁厚的 84%，端面会退化为刃口",
         "fix": "outer_edge = 0.18 * (outer_d - inner_d); inner_edge = 0.12 * (outer_d - inner_d)"},
        {"id": "edge_fits_height",
         "expr": "2 * max(outer_edge, inner_edge) <= 0.85 * height",
         "severity": "error",
         "message": "端面倒角/圆角吃掉超过 85% 高度，管壁会消失",
         "fix": "outer_edge = 0.2 * height"},
    ],
)
def _build_tube(v, quality="draft"):
    ro = 0.5 * float(v["outer_d"])
    ri = 0.5 * float(v["inner_d"])
    h = float(v["height"])
    eo = float(v["outer_edge"])
    ei = float(v["inner_edge"])
    mode = str(v.get("edge_mode") or "chamfer").lower()
    seg = max(12, _q(quality, int(v["segments"]), int(v["segments"]) * 3 // 2,
                     int(v["segments"]) * 2))
    steps = max(2, _q(quality, int(v["corner_steps"]), int(v["corner_steps"]) + 3,
                      int(v["corner_steps"]) * 2))

    if ro - ri <= 1e-4 or eo + ei >= 0.95 * (ro - ri) or 2.0 * max(eo, ei) >= 0.95 * h:
        raise AttaError("wall_too_thin", "tube edges exceed wall thickness or height",
                        path="outer_edge", fix="reduce outer_edge/inner_edge or thicken wall")

    pts = []
    # Bottom inner -> bottom outer -> top outer -> top inner (CCW loop in RZ)
    pts.extend(_corner_arc(ri, 0.0, 1.0, 1.0, ei, mode, steps, reverse=True))
    pts.extend(_corner_arc(ro, 0.0, -1.0, 1.0, eo, mode, steps, reverse=False))
    pts.extend(_corner_arc(ro, h, -1.0, -1.0, eo, mode, steps, reverse=True))
    pts.extend(_corner_arc(ri, h, 1.0, -1.0, ei, mode, steps, reverse=False))
    pts.append(pts[0])
    prof = np.asarray(pts, dtype=np.float64)
    mesh = _S.revolve(prof, segments=seg, caps=False)
    mesh.meta.update({"kind": "tube", "edge_mode": mode})
    return mesh, finish(smooth_angle_deg=35.0, material="aluminium")


# ----------------------------------------------------------------------- wedge

@recipe(
    "primitive.wedge", "斜坡楔块基元",
    "参数化梯形/斜楔块（参照 FreeCAD Part.makeWedge）：支持顶面长度/宽度独立收窄、"
    "顶面 X 向偏置及端面导角，单次放样封闭成形，恒为 genus 0 闭合单实体。",
    params={
        "length": {"type": "float", "default": 60.0, "min": 5.0, "max": 500.0,
                   "unit": "mm", "group": "body", "description": "底面 X 向长度"},
        "width": {"type": "float", "default": 40.0, "min": 5.0, "max": 500.0,
                  "unit": "mm", "group": "body", "description": "底面 Y 向宽度"},
        "height": {"type": "float", "default": 30.0, "min": 2.0, "max": 500.0,
                   "unit": "mm", "group": "body", "description": "总高 Z"},
        "top_length": {"type": "float", "default": 24.0, "min": 2.0, "max": 500.0,
                       "unit": "mm", "group": "taper", "description": "顶面 X 向长度"},
        "top_width": {"type": "float", "default": 40.0, "min": 2.0, "max": 500.0,
                      "unit": "mm", "group": "taper", "description": "顶面 Y 向宽度"},
        "top_offset": {"type": "float", "default": -18.0, "min": -200.0, "max": 200.0,
                       "unit": "mm", "group": "taper",
                       "description": "顶面中心相对底面中心的 X 偏移（-18 表示左侧齐平直角楔）"},
        "bevel": {"type": "float", "default": 1.2, "min": 0.0, "max": 20.0,
                  "unit": "mm", "group": "edge", "description": "上下边缘倒角缩进"},
    },
    presets={
        "safe": {"length": 60.0, "width": 40.0, "height": 28.0,
                 "top_length": 28.0, "top_width": 40.0, "top_offset": -16.0, "bevel": 1.2},
        "nominal": {"length": 60.0, "width": 40.0, "height": 30.0,
                    "top_length": 24.0, "top_width": 40.0, "top_offset": -18.0, "bevel": 1.2},
        "hero": {"length": 80.0, "width": 48.0, "height": 38.0,
                 "top_length": 20.0, "top_width": 30.0, "top_offset": 12.0, "bevel": 1.8},
    },
    tags=["primitive", "wedge", "frustum", "loft", "chamfer"],
    expected_genus=0, material="powder_blue",
    constraints=[
        {"id": "top_fits_bevel",
         "expr": "min(top_length, top_width) - 2 * bevel >= 1.2",
         "severity": "error",
         "message": "顶面扣除倒角后宽度不足 1.2 mm，将退化为刀口",
         "fix": "bevel = 0.2 * min(top_length, top_width)"},
        {"id": "bevel_fits_height",
         "expr": "2 * bevel <= 0.4 * height",
         "severity": "error",
         "message": "上下倒角高度超过总高的 40%",
         "fix": "bevel = 0.15 * height"},
        {"id": "top_within_footprint",
         "expr": "abs(top_offset) + 0.5 * top_length <= 1.2 * length",
         "severity": "warn",
         "message": "顶面偏移过大，形成大幅度悬挑",
         "fix": "top_offset = 0.0"},
    ],
)
def _build_wedge(v, quality="draft"):
    L, W, H = float(v["length"]), float(v["width"]), float(v["height"])
    tl, tw = float(v["top_length"]), float(v["top_width"])
    ox = float(v["top_offset"])
    bev = float(v["bevel"])

    if min(tl, tw, L, W) - 2.0 * bev <= 1e-4 or 2.0 * bev >= 0.8 * H:
        raise AttaError("bad_wedge", "bevel exceeds top/bottom face or height",
                        path="bevel", fix="reduce bevel or increase top_length/top_width")

    def rect(cx, rx, ry, z):
        return np.array([
            [cx - rx, -ry, z],
            [cx + rx, -ry, z],
            [cx + rx,  ry, z],
            [cx - rx,  ry, z],
        ], dtype=np.float64)

    rings = []
    if bev > 1e-6:
        rings.append(rect(0.0, 0.5 * L - bev, 0.5 * W - bev, 0.0))
        rings.append(rect(ox * (bev / H), 0.5 * L, 0.5 * W, bev))
        rings.append(rect(ox * ((H - bev) / H), 0.5 * tl, 0.5 * tw, H - bev))
        rings.append(rect(ox, 0.5 * tl - bev, 0.5 * tw - bev, H))
    else:
        rings.append(rect(0.0, 0.5 * L, 0.5 * W, 0.0))
        rings.append(rect(ox, 0.5 * tl, 0.5 * tw, H))

    mesh = _S.loft(rings, closed_ring=True, caps=True)
    mesh.meta.update({"kind": "wedge"})
    return mesh, finish(smooth_angle_deg=25.0, material="powder_blue")
