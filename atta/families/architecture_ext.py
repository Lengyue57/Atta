# -*- coding: utf-8 -*-
"""Extended architectural families: arch vault, spiral stair, dome with oculus.

Real-world architectural & structural standards cited:
* ``architecture.arch_vault`` — Jacques Heyman, *The Stone Skeleton: Structural
  Engineering of Masonry Architecture* (Cambridge University Press, 1995):
  semicircular masonry arch minimum thickness-to-radius ratio ``t/R >= 0.106``
  (``t/span >= 0.053``) under self-weight, practical safety factor >= 2
  (``t/span >= 0.11``); equilateral pointed Gothic arch (arc centres at
  opposite springing points, ``R = span``, 60° apex).
* ``architecture.stair`` — International Building Code (IBC 2021 §1011.10) &
  François Blondel (*Cours d'architecture*, 1675): ergonomic step law
  ``2*R + T ~= 610..650 mm``; IBC spiral stair minimum walkline tread depth
  ``6.75 in (171 mm)`` at ``12 in (305 mm)`` from narrow edge and max riser
  ``9.5 in (241 mm)``, scaled cleanly with dimensionless pitch/slope checks.
* ``architecture.dome_oculus`` — Pantheon, Rome (126 AD): hemispherical rotunda
  with ``1:1`` interior diameter-to-height ratio, oculus opening ``~0.205 * D``,
  graduated shell thickness tapering toward the oculus compression collar, and
  stepped exterior haunch rings.

All lengths inside builders are metres; public recipe params are mm/deg.
"""

from __future__ import annotations

import math

import numpy as np

from .. import primitives as _P
from .. import surfaces as _S
from ..errors import AttaError
from ..kit import finish, quality_level as _q
from ..mesh import Mesh, oriented
from ..registry import recipe

__all__ = ["_build_arch_vault", "_build_stair", "_build_dome_oculus"]


def _m3():
    import manifold3d
    return manifold3d


def _to_manifold(mesh: Mesh):
    m3 = _m3()
    return m3.Manifold(m3.Mesh(
        vert_properties=np.asarray(mesh.verts, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))


def _batch_union(manifolds):
    m3 = _m3()
    items = [m for m in manifolds if m is not None and not m.is_empty()]
    if not items:
        raise AttaError("boolean_empty", "no non-empty parts to union",
                        path="architecture", fix="check recipe dimensions")
    out = items[0] if len(items) == 1 else m3.Manifold.batch_boolean(items, m3.OpType.Add)
    mm = out.to_mesh()
    verts = np.asarray(mm.vert_properties, dtype=np.float64)
    faces = np.asarray(mm.tri_verts, dtype=np.int64)
    if len(faces) == 0:
        raise AttaError("boolean_empty", "union produced empty mesh",
                        path="architecture", fix="check part overlaps")
    return oriented(Mesh(verts[:, :3], faces, {"kind": "architecture_union"}))


def _box_m(sx, sy, sz, cx, cy, cz):
    m3 = _m3()
    b = m3.Manifold.cube([float(sx), float(sy), float(sz)], True)
    return b.translate([float(cx), float(cy), float(cz)])


def _extrude_xz_along_y(poly_xz, y0, y1):
    """Closed prism whose cross-section ``poly_xz`` (N, 2) in XZ spans y0..y1."""
    p = np.asarray(poly_xz, dtype=np.float64)
    area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                              - np.roll(p[:, 0], -1) * p[:, 1]))
    # Ensure CCW in (x, z) so loft along +y winds outward cleanly.
    if area > 0.0:
        # In (x, z, y) order, +y = x cross (-z) -> flip so side faces wind out.
        p = p[::-1].copy()
    r0 = np.column_stack([p[:, 0], np.full(len(p), float(y0)), p[:, 1]])
    r1 = np.column_stack([p[:, 0], np.full(len(p), float(y1)), p[:, 1]])
    return _to_manifold(_S.loft([r0, r1], closed_ring=True, caps=True))


# ------------------------------------------------------------------ arch vault

def _arch_ring_points(style, span, arch_t, z_spring, overlap, seg_half):
    """Build 2D (x, z) points for a continuous arch ring from right to left."""
    half_s = 0.5 * span
    if style == "pointed":
        # Equilateral pointed Gothic arch: right arc centred at (-half_s, z_spring),
        # left arc centred at (+half_s, z_spring), both with R_in = span.
        r_in = span
        r_out = span + arch_t
        ang_apex_in = math.acos(min(1.0, half_s / r_in))
        ang_apex_out = math.acos(min(1.0, (half_s + 0.0) / r_out))
        a_r_in = np.linspace(0.0, ang_apex_in, seg_half + 1)
        a_l_in = np.linspace(math.pi - ang_apex_in, math.pi, seg_half + 1)
        # Right spring -> apex -> left spring
        intrados = (
            [(-half_s + r_in * math.cos(a), z_spring + r_in * math.sin(a)) for a in a_r_in]
            + [(half_s + r_in * math.cos(a), z_spring + r_in * math.sin(a)) for a in a_l_in[1:]]
        )
        a_r_out = np.linspace(0.0, ang_apex_out, seg_half + 1)
        a_l_out = np.linspace(math.pi - ang_apex_out, math.pi, seg_half + 1)
        # Left outer spring -> outer apex -> right outer spring
        extrados = (
            [(half_s + r_out * math.cos(a), z_spring + r_out * math.sin(a)) for a in a_l_out[::-1]]
            + [(-half_s + r_out * math.cos(a), z_spring + r_out * math.sin(a)) for a in a_r_out[::-1][1:]]
        )
    else:
        r_in = half_s
        r_out = half_s + arch_t
        ang = np.linspace(0.0, math.pi, 2 * seg_half + 1)
        intrados = [(r_in * math.cos(a), z_spring + r_in * math.sin(a)) for a in ang]
        extrados = [(r_out * math.cos(a), z_spring + r_out * math.sin(a)) for a in ang[::-1]]

    # Extend bases slightly below z_spring so the ring embeds cleanly into the piers.
    foot_r_in = (intrados[0][0], z_spring - overlap)
    foot_l_in = (intrados[-1][0], z_spring - overlap)
    foot_l_out = (extrados[0][0], z_spring - overlap)
    foot_r_out = (extrados[-1][0], z_spring - overlap)
    return np.asarray(
        [foot_r_in] + intrados + [foot_l_in, foot_l_out] + extrados + [foot_r_out],
        dtype=np.float64,
    )


def _arch_eval(style, span, z_spring, t_frac):
    """Point and outward unit normal on the intrados at parameter t in [0, 1] (right->left)."""
    half_s = 0.5 * span
    if style == "pointed":
        r_in = span
        ang_apex = math.acos(min(1.0, half_s / r_in))
        if t_frac <= 0.5:
            u = t_frac * 2.0
            a = u * ang_apex
            cx = -half_s
        else:
            u = (t_frac - 0.5) * 2.0
            a = math.pi - (1.0 - u) * ang_apex
            cx = half_s
        px = cx + r_in * math.cos(a)
        pz = z_spring + r_in * math.sin(a)
        return (px, pz), (math.cos(a), math.sin(a))
    a = t_frac * math.pi
    return (half_s * math.cos(a), z_spring + half_s * math.sin(a)), (math.cos(a), math.sin(a))


@recipe(
    "architecture.arch_vault", "石砌拱券/筒拱",
    "古典半圆拱与哥特等边尖拱（参照 Heyman 砌体拱最小厚度理论 t/span >= 0.053，"
    "工程安全系数 >= 2）：含独立墩座基脚、墩身、起拱线挑檐（impost）、"
    "拱圈筒体、放射状拱石浮雕（voussoirs）及中心拱顶石（keystone），恒为 genus 0 闭合单实体。",
    params={
        "style": {"type": "enum", "default": "semicircular",
                  "enum": ["semicircular", "pointed"], "unit": "none",
                  "group": "arch", "description": "拱形：semicircular=半圆拱，pointed=等边尖拱"},
        "span": {"type": "float", "default": 120.0, "min": 40.0, "max": 500.0,
                 "unit": "mm", "group": "arch", "description": "净跨度"},
        "vault_depth": {"type": "float", "default": 70.0, "min": 15.0, "max": 400.0,
                        "unit": "mm", "group": "arch", "description": "筒拱纵深（Y 向）"},
        "pier_h": {"type": "float", "default": 50.0, "min": 15.0, "max": 300.0,
                   "unit": "mm", "group": "pier", "description": "拱墩净高"},
        "pier_w": {"type": "float", "default": 24.0, "min": 8.0, "max": 150.0,
                   "unit": "mm", "group": "pier", "description": "拱墩厚度（X 向）"},
        "arch_t": {"type": "float", "default": 18.0, "min": 6.0, "max": 100.0,
                   "unit": "mm", "group": "arch", "description": "拱圈径向厚度"},
        "voussoirs": {"type": "int", "default": 11, "min": 5, "max": 29,
                      "unit": "count", "group": "detail", "description": "拱石数量（建议奇数以居中拱顶石）"},
        "keystone_extra": {"type": "float", "default": 5.0, "min": 0.0, "max": 30.0,
                           "unit": "mm", "group": "detail", "description": "拱顶石额外径向与立面突出量"},
        "impost_overhang": {"type": "float", "default": 3.5, "min": 0.0, "max": 20.0,
                            "unit": "mm", "group": "detail", "description": "起拱线檐口及基脚出挑量"},
        "plinth_h": {"type": "float", "default": 8.0, "min": 2.0, "max": 40.0,
                     "unit": "mm", "group": "pier", "description": "墩座基脚高度"},
    },
    presets={
        "safe": {"style": "semicircular", "span": 100.0, "vault_depth": 60.0,
                 "pier_h": 45.0, "pier_w": 24.0, "arch_t": 18.0,
                 "voussoirs": 9, "keystone_extra": 4.0, "impost_overhang": 3.0, "plinth_h": 8.0},
        "nominal": {"style": "semicircular", "span": 120.0, "vault_depth": 70.0,
                    "pier_h": 50.0, "pier_w": 24.0, "arch_t": 18.0,
                    "voussoirs": 11, "keystone_extra": 5.0, "impost_overhang": 3.5, "plinth_h": 8.0},
        "hero": {"style": "pointed", "span": 140.0, "vault_depth": 90.0,
                 "pier_h": 65.0, "pier_w": 28.0, "arch_t": 20.0,
                 "voussoirs": 13, "keystone_extra": 7.0, "impost_overhang": 4.5, "plinth_h": 10.0},
    },
    tags=["architecture", "arch", "vault", "voussoir", "gothic", "boolean"],
    expected_genus=0, material="quartz",
    constraints=[
        {"id": "heyman_min_thickness",
         "expr": "arch_t >= 0.11 * span",
         "severity": "error",
         "message": "拱圈厚度小于跨度 11%（低于 Heyman 砌体拱 2 倍安全余量下限）",
         "fix": "arch_t = 0.15 * span"},
        {"id": "pier_supports_arch",
         "expr": "pier_w >= arch_t",
         "severity": "error",
         "message": "拱墩宽度小于拱圈厚度，拱背推力悬空",
         "fix": "pier_w = arch_t * 1.25"},
        {"id": "impost_fits",
         "expr": "impost_overhang <= 0.45 * pier_w",
         "severity": "error",
         "message": "起拱线檐口出挑过大",
         "fix": "impost_overhang = 0.2 * pier_w"},
        {"id": "voussoir_odd",
         "expr": "voussoirs % 2 == 1",
         "severity": "warn",
         "message": "偶数块拱石无法形成对称中心拱顶石（keystone）",
         "fix": "voussoirs += 1"},
    ],
)
def _build_arch_vault(v, quality="draft"):
    style = str(v.get("style") or "semicircular").lower()
    span = float(v["span"])
    depth = float(v["vault_depth"])
    pier_h = float(v["pier_h"])
    pier_w = float(v["pier_w"])
    arch_t = float(v["arch_t"])
    n_vous = int(v["voussoirs"])
    k_extra = float(v["keystone_extra"])
    overhang = float(v["impost_overhang"])
    plinth_h = float(v["plinth_h"])

    seg_half = _q(quality, 16, 28, 44)
    overlap = min(1.5e-3, 0.15 * min(plinth_h, arch_t))
    z_spring = plinth_h + pier_h
    parts = []

    # Piers, plinths, and impost cornices (left and right).
    # Slight Y-recess on pier shafts gives classical reveal and avoids coplanar Y booleans.
    impost_h = min(0.28 * pier_h, 0.55 * plinth_h + 2.0e-3)
    pier_depth = max(0.85 * depth, depth - 1.0e-3)
    for side in (-1.0, 1.0):
        px = side * (0.5 * span + 0.5 * pier_w)
        parts.append(_box_m(pier_w + 2.0 * overhang, depth + 2.0 * overhang,
                            plinth_h + overlap, px, 0.0, 0.5 * (plinth_h + overlap)))
        parts.append(_box_m(pier_w, pier_depth, pier_h + 2.0 * overlap,
                            px, 0.0, plinth_h + 0.5 * pier_h))
        if overhang > 1e-5:
            parts.append(_box_m(pier_w + 2.0 * overhang, depth + 2.0 * overhang,
                                impost_h + 0.5 * overlap, px, 0.0,
                                z_spring - 0.5 * impost_h + 0.25 * overlap))

    # Main continuous barrel vault shell.
    ring_poly = _arch_ring_points(style, span, arch_t, z_spring, overlap, seg_half)
    parts.append(_extrude_xz_along_y(ring_poly, -0.5 * depth, 0.5 * depth))

    # Radial voussoir reliefs on front/back faces + proud central keystone.
    mid_idx = n_vous // 2
    proud_y = max(0.8e-3, 0.04 * depth)
    for i in range(n_vous):
        t0 = (i + 0.08) / n_vous
        t1 = (i + 0.92) / n_vous
        is_key = (i == mid_idx) and (n_vous % 2 == 1)
        extra_r = k_extra if is_key else (0.18 * arch_t if i % 2 == 0 else 0.08 * arch_t)
        extra_y = proud_y + (0.6 * k_extra if is_key else 0.0)
        p0, n0 = _arch_eval(style, span, z_spring, t0)
        p1, n1 = _arch_eval(style, span, z_spring, t1)
        pm, nm = _arch_eval(style, span, z_spring, 0.5 * (t0 + t1))
        r_in_off = 0.15 * arch_t  # embed into barrel ring
        r_out_off = arch_t + extra_r
        quad = np.array([
            [p0[0] + r_in_off * n0[0], p0[1] + r_in_off * n0[1]],
            [p1[0] + r_in_off * n1[0], p1[1] + r_in_off * n1[1]],
            [p1[0] + r_out_off * n1[0], p1[1] + r_out_off * n1[1]],
            [pm[0] + r_out_off * nm[0], pm[1] + r_out_off * nm[1]],
            [p0[0] + r_out_off * n0[0], p0[1] + r_out_off * n0[1]],
        ], dtype=np.float64)
        if is_key:
            parts.append(_extrude_xz_along_y(quad, -0.5 * depth - extra_y, 0.5 * depth + extra_y))
        else:
            # Front & back voussoir face bands (embed into main vault depth).
            band = min(0.28 * depth, max(2.5e-3, 0.16 * depth))
            parts.append(_extrude_xz_along_y(quad, 0.5 * depth - band, 0.5 * depth + extra_y))
            parts.append(_extrude_xz_along_y(quad, -0.5 * depth - extra_y, -0.5 * depth + band))

    mesh = _batch_union(parts)
    mesh.meta.update({"kind": "arch_vault", "style": style, "voussoirs": n_vous})
    return mesh, finish(smooth_angle_deg=30.0, material="quartz")


# ---------------------------------------------------------------- spiral stair

@recipe(
    "architecture.stair", "悬挑螺旋楼梯",
    "中心柱悬挑螺旋楼梯（参照 IBC §1011.10 螺旋楼梯规范及 Blondel 步距公式 2R+T）："
    "含底座法兰、中心立柱、柱顶收头及带下缘加劲肋的扇形悬挑踏步；"
    "踏步之间保持净空仅与中心柱相连，恒为 genus 0 闭合单实体。",
    params={
        "total_height": {"type": "float", "default": 150.0, "min": 40.0, "max": 500.0,
                         "unit": "mm", "group": "stair", "description": "楼梯总上升高度"},
        "outer_r": {"type": "float", "default": 60.0, "min": 25.0, "max": 250.0,
                    "unit": "mm", "group": "stair", "description": "踏步外缘半径"},
        "column_r": {"type": "float", "default": 14.0, "min": 5.0, "max": 80.0,
                     "unit": "mm", "group": "column", "description": "中心立柱半径"},
        "step_count": {"type": "int", "default": 12, "min": 4, "max": 36,
                       "unit": "count", "group": "stair", "description": "踏步级数"},
        "turn_deg": {"type": "float", "default": 270.0, "min": 90.0, "max": 540.0,
                     "unit": "deg", "group": "stair", "description": "总旋转角度"},
        "tread_t": {"type": "float", "default": 4.0, "min": 1.5, "max": 20.0,
                    "unit": "mm", "group": "tread", "description": "踏步板厚"},
        "overlap_deg": {"type": "float", "default": 8.0, "min": 0.0, "max": 25.0,
                        "unit": "deg", "group": "tread", "description": "相邻踏步平面投影搭接角"},
        "base_flange_r": {"type": "float", "default": 22.0, "min": 8.0, "max": 100.0,
                          "unit": "mm", "group": "column", "description": "底盘法兰半径"},
    },
    presets={
        "safe": {"total_height": 120.0, "outer_r": 55.0, "column_r": 15.0,
                 "step_count": 10, "turn_deg": 240.0, "tread_t": 4.2,
                 "overlap_deg": 6.0, "base_flange_r": 22.0},
        "nominal": {"total_height": 150.0, "outer_r": 60.0, "column_r": 14.0,
                    "step_count": 12, "turn_deg": 270.0, "tread_t": 4.0,
                    "overlap_deg": 8.0, "base_flange_r": 22.0},
        "hero": {"total_height": 192.0, "outer_r": 72.0, "column_r": 16.0,
                 "step_count": 16, "turn_deg": 360.0, "tread_t": 4.0,
                 "overlap_deg": 10.0, "base_flange_r": 26.0},
    },
    tags=["architecture", "stair", "spiral", "cantilever", "boolean"],
    expected_genus=0, material="wood",
    constraints=[
        {"id": "tread_width_ok",
         "expr": "outer_r - column_r >= 15.0",
         "severity": "error",
         "message": "踏步净宽不足 15 mm",
         "fix": "outer_r = column_r + 35.0"},
        {"id": "step_clearance",
         "expr": "total_height / step_count >= tread_t * 1.65 + 1.2",
         "severity": "error",
         "message": "步高过小导致上下踏步及加劲肋粘连成闭合环",
         "fix": "step_count = int(total_height / (tread_t * 2.0 + 2.0))"},
        {"id": "blondel_slope",
         "expr": "(total_height / step_count) <= 1.6 * (outer_r - column_r)",
         "severity": "warn",
         "message": "楼梯坡度过陡（超出常规建筑坡度比）",
         "fix": "outer_r = column_r + total_height / step_count"},
        {"id": "flange_covers_col",
         "expr": "base_flange_r >= column_r + 2.0",
         "severity": "error",
         "message": "底座法兰小于立柱半径",
         "fix": "base_flange_r = column_r + 6.0"},
    ],
)
def _build_stair(v, quality="draft"):
    H = float(v["total_height"])
    ro = float(v["outer_r"])
    rc = float(v["column_r"])
    n_steps = int(v["step_count"])
    turn = math.radians(float(v["turn_deg"]))
    tt = float(v["tread_t"])
    ov = math.radians(float(v["overlap_deg"]))
    rf = float(v["base_flange_r"])

    seg_col = _q(quality, 31, 47, 71)
    seg_arc = _q(quality, 6, 10, 14)
    rise = H / n_steps
    haunch = 0.55 * tt
    if rise <= tt + haunch + 0.8e-3:
        raise AttaError("step_collision", "step rise is too small for tread + haunch clearance",
                        path="step_count", fix="reduce step_count or tread_t")

    parts = []
    # Base flange + central newel column with chamfered finial.
    flange_h = min(0.6 * rise, max(4.0e-3, 1.5 * tt))
    top_ext = max(6.0e-3, 0.8 * rise)
    col_prof = np.array([
        [0.0, 0.0],
        [rf, 0.0],
        [rf, 0.7 * flange_h],
        [rc, flange_h],
        [rc, H + top_ext],
        [0.85 * rc, H + top_ext + 0.15 * rc],
        [0.0, H + top_ext + 0.25 * rc],
    ], dtype=np.float64)
    parts.append(_to_manifold(_S.revolve(col_prof, segments=seg_col, caps=False)).rotate([0.0, 0.0, 0.73]))

    # Cantilevered radial sector treads with sloped under-tread haunch at inner radius.
    dtheta = turn / n_steps
    span_a = dtheta + ov
    r_in = 0.65 * rc
    r_rib_in = 0.45 * rc
    for i in range(n_steps):
        a0 = i * dtheta
        a1 = a0 + span_a
        z_top = (i + 1) * rise
        z_bot = z_top - tt
        angs = np.linspace(a0, a1, seg_arc + 1)
        loop_xy = np.vstack([
            np.column_stack([r_in * np.cos(angs), r_in * np.sin(angs)]),
            np.column_stack([ro * np.cos(angs[::-1]), ro * np.sin(angs[::-1])]),
        ])
        tread_mesh = _S.extrude(loop_xy, tt).transform(translate=(0.0, 0.0, z_bot))
        parts.append(_to_manifold(tread_mesh))

        # Under-tread haunch gusset tapering from z_bot - haunch at column to z_bot at mid-span.
        r_mid = rc + 0.48 * (ro - rc)
        a_mid = 0.5 * (a0 + a1)
        half_w = min(0.32 * tt + 1.5e-3, 0.30 * rc * span_a)
        rib_2d = np.array([
            [r_rib_in, z_bot - haunch],
            [r_mid, z_bot - 0.1 * tt],
            [r_mid, z_bot + 0.35 * tt],
            [r_rib_in, z_bot + 0.35 * tt],
        ], dtype=np.float64)
        ca, sa = math.cos(a_mid), math.sin(a_mid)
        rib_p0 = np.column_stack([rib_2d[:, 0], np.full(4, -half_w), rib_2d[:, 1]])
        rib_p1 = np.column_stack([rib_2d[:, 0], np.full(4,  half_w), rib_2d[:, 1]])
        # Build axis-aligned in XZ then rotate by a_mid around Z -> guaranteed outward winding.
        rib_local = _extrude_xz_along_y(rib_2d, -half_w, half_w)
        parts.append(rib_local.rotate([0.0, 0.0, math.degrees(a_mid)]))

    mesh = _batch_union(parts)
    mesh.meta.update({"kind": "stair", "step_count": n_steps})
    return mesh, finish(smooth_angle_deg=35.0, material="wood")


# ---------------------------------------------------------------- dome oculus

@recipe(
    "architecture.dome_oculus", "穹顶采光天窗（万神殿式）",
    "古典柱鼓与穹顶天窗（参照罗马万神殿比例：鼓身+半球穹顶内径比、"
    "顶部采光天窗 oculus 约占内径 0.205、壁厚向天窗渐薄、外侧阶梯式环箍及子午肋），"
    "底面与天窗上下贯通，恒为 genus 1 闭合单实体。",
    params={
        "inner_d": {"type": "float", "default": 100.0, "min": 30.0, "max": 400.0,
                    "unit": "mm", "group": "dome", "description": "穹顶内净直径"},
        "wall_t": {"type": "float", "default": 8.0, "min": 2.0, "max": 40.0,
                   "unit": "mm", "group": "dome", "description": "鼓墙及穹顶起拱壁厚"},
        "oculus_ratio": {"type": "float", "default": 0.21, "min": 0.10, "max": 0.45,
                         "unit": "ratio", "group": "oculus", "description": "天窗直径/内径比（万神殿约 0.205）"},
        "drum_h": {"type": "float", "default": 40.0, "min": 10.0, "max": 250.0,
                   "unit": "mm", "group": "drum", "description": "下部圆柱鼓墙高度"},
        "rib_count": {"type": "int", "default": 16, "min": 0, "max": 36,
                      "unit": "count", "group": "detail", "description": "外表面子午加强肋数"},
        "rib_depth": {"type": "float", "default": 2.5, "min": 0.0, "max": 15.0,
                      "unit": "mm", "group": "detail", "description": "子午肋径向凸出厚度"},
        "step_rings": {"type": "int", "default": 3, "min": 0, "max": 6,
                       "unit": "count", "group": "detail", "description": "万神殿式底部阶梯环箍数"},
    },
    presets={
        "safe": {"inner_d": 90.0, "wall_t": 9.0, "oculus_ratio": 0.22,
                 "drum_h": 36.0, "rib_count": 12, "rib_depth": 2.5, "step_rings": 2},
        "nominal": {"inner_d": 100.0, "wall_t": 8.0, "oculus_ratio": 0.21,
                    "drum_h": 40.0, "rib_count": 16, "rib_depth": 2.5, "step_rings": 3},
        "hero": {"inner_d": 130.0, "wall_t": 9.0, "oculus_ratio": 0.205,
                 "drum_h": 54.0, "rib_count": 24, "rib_depth": 3.0, "step_rings": 4},
    },
    tags=["architecture", "dome", "oculus", "pantheon", "revolve", "boolean"],
    expected_genus=1, material="ceramic",
    constraints=[
        {"id": "wall_ratio",
         "expr": "wall_t <= 0.25 * inner_d",
         "severity": "error",
         "message": "壁厚相对内径过大",
         "fix": "wall_t = 0.08 * inner_d"},
        {"id": "oculus_valid",
         "expr": "oculus_ratio * inner_d >= 6.0 and oculus_ratio <= 0.45",
         "severity": "error",
         "message": "采光天窗孔径过小（< 6 mm）或过大（> 0.45 倍内径）",
         "fix": "oculus_ratio = 0.21"},
        {"id": "rib_width_ok",
         "expr": "rib_count <= 0 or rib_depth <= wall_t",
         "severity": "warn",
         "message": "子午肋突深超过墙厚",
         "fix": "rib_depth = 0.35 * wall_t"},
    ],
)
def _build_dome_oculus(v, quality="draft"):
    r_in = 0.5 * float(v["inner_d"])
    t = float(v["wall_t"])
    oc_r = float(v["oculus_ratio"]) * r_in
    drum_h = float(v["drum_h"])
    rib_n = int(v["rib_count"])
    rib_d = float(v["rib_depth"])
    n_steps = int(v["step_rings"])

    seg = _q(quality, 48, 72, 96)
    merid_n = _q(quality, 20, 32, 48)
    r_out = r_in + t

    # Oculus angle from top pole (phi = pi/2 at equator, phi_oc at oculus inner rim).
    phi_oc = math.acos(min(0.95, oc_r / r_in))
    phis = np.linspace(0.0, phi_oc, merid_n)

    # Taper wall thickness smoothly from t at springing to 0.5*t at oculus, plus compression collar.
    t_top = max(2.0e-3, 0.52 * t)
    ring_pts = []
    # 1. Base plinth foot -> drum outer -> cornice -> dome outer -> oculus rim -> dome inner -> drum inner
    foot_out = r_out + 0.22 * t
    foot_h = min(0.25 * drum_h, 0.8 * t)
    cornice_out = r_out + 0.20 * t
    cornice_h = min(0.18 * drum_h, 0.65 * t)

    ring_pts.append((r_in, 0.0))
    ring_pts.append((foot_out, 0.0))
    ring_pts.append((foot_out, foot_h))
    ring_pts.append((r_out, foot_h + 0.15 * foot_h))
    ring_pts.append((r_out, drum_h - cornice_h))
    ring_pts.append((cornice_out, drum_h - 0.5 * cornice_h))
    ring_pts.append((cornice_out, drum_h))

    # Pantheon-style stepped buttress rings on the lower 35% of the dome extrados.
    phi_step_max = min(0.55 * phi_oc, math.radians(34.0))
    for i in range(n_steps):
        p_lo = (i / max(1, n_steps)) * phi_step_max
        p_hi = ((i + 1) / max(1, n_steps)) * phi_step_max
        r_step = r_out - (i / max(1, n_steps)) * (0.25 * t)
        ring_pts.append((r_step, drum_h + r_in * math.sin(p_lo)))
        ring_pts.append((r_step, drum_h + r_in * math.sin(p_hi)))

    # Smooth tapered extrados from phi_step_max up to oculus collar.
    p_start = phi_step_max if n_steps > 0 else 0.0
    for p in np.linspace(p_start, phi_oc, merid_n):
        u = p / max(phi_oc, 1e-6)
        t_loc = t * (1.0 - u) + t_top * u
        rr = (r_in + t_loc) * math.cos(p)
        zz = drum_h + (r_in + t_loc) * math.sin(p)
        ring_pts.append((rr, zz))

    # Raised oculus compression collar at the apex.
    z_oc_in = drum_h + r_in * math.sin(phi_oc)
    collar_h = max(1.5e-3, 0.45 * t)
    collar_out = max(oc_r + 1.15 * t_top, (r_in + t_top) * math.cos(phi_oc))
    ring_pts.append((collar_out, z_oc_in + collar_h))
    ring_pts.append((oc_r, z_oc_in + collar_h))

    # Intrados descending back to drum inner wall.
    for p in phis[::-1]:
        ring_pts.append((r_in * math.cos(p), drum_h + r_in * math.sin(p)))
    ring_pts.append((r_in, 0.0))

    prof = np.asarray(ring_pts, dtype=np.float64)
    dome_mesh = _S.revolve(prof, segments=seg, caps=False)

    if rib_n <= 0 or rib_d <= 1e-6:
        dome_mesh.meta.update({"kind": "dome_oculus", "rib_count": 0})
        return dome_mesh, finish(smooth_angle_deg=35.0, material="ceramic")

    # Build meridional ribs embedded cleanly into the dome wall, then batch union.
    parts = [_to_manifold(dome_mesh)]
    rib_w = max(1.6e-3, min(0.55 * t, 0.35 * (2.0 * math.pi * oc_r / max(1, rib_n))))
    p_rib = np.linspace(0.02, phi_oc * 0.90, max(10, merid_n // 2))
    rib_r_in = []
    rib_r_out = []
    for p in p_rib:
        u = p / max(phi_oc, 1e-6)
        t_loc = t * (1.0 - u) + t_top * u
        r_mid = r_in + 0.30 * t_loc
        r_ext = r_in + t_loc + rib_d
        rib_r_in.append((r_mid * math.cos(p), drum_h + r_mid * math.sin(p)))
        rib_r_out.append((r_ext * math.cos(p), drum_h + r_ext * math.sin(p)))
    rib_poly = np.asarray(rib_r_in + rib_r_out[::-1], dtype=np.float64)
    base_rib = _extrude_xz_along_y(rib_poly, -0.5 * rib_w, 0.5 * rib_w)
    for k in range(rib_n):
        ang_deg = 360.0 * k / rib_n
        parts.append(base_rib.rotate([0.0, 0.0, ang_deg]))

    mesh = _batch_union(parts)
    mesh.meta.update({"kind": "dome_oculus", "rib_count": rib_n})
    return mesh, finish(smooth_angle_deg=35.0, material="ceramic")
