# -*- coding: utf-8 -*-
"""Extended pattern families: DIN 82 knurled grip, ASHRAE/AMCA louvre panel.

Engineering standards cited:
* ``pattern.knurl`` — DIN 82 (*Rändel* / Knurls): pattern designations
  ``RGE`` (diamond raised points, 30° helix), ``RAA`` (straight axial),
  ``RBL`` (helical 30°), metric pitch series ``P``, 90° tooth profile with
  radial depth ``~0.5 * P``.  Built as a single structured closed surface mesh
  (zero booleans, guaranteed genus 0 and watertight for any pitch/count).
* ``pattern.louvre_panel`` — ASHRAE / AMCA Standard 500-L architectural weather
  louver proportions: fixed ``30°..55°`` (standard ``45°``) sloped blades with
  ``~1:1`` blade pitch-to-depth ratio inside a perimeter flange frame.  With
  ``slat_count`` unobstructed slats, the panel has ``slat_count + 1`` through
  slots (``genus = slat_count + 1``, published via ``_publish_genus``; all
  presets hold ``slat_count = 6`` so ``expected_genus = 7``).

All lengths inside builders are metres; public recipe params are mm/deg.
"""

from __future__ import annotations

import math

import numpy as np

from .. import primitives as _P
from .. import surfaces as _S
from ..errors import AttaError
from ..kit import finish, quality_level as _q, rounded_rect
from ..mesh import Mesh, oriented
from ..registry import recipe

__all__ = ["_build_knurl", "_build_louvre_panel"]


def _publish_genus(recipe_id, genus):
    """Publish derived expected_genus (same convention as ``pattern.honeycomb_panel``)."""
    try:
        from ..registry import REGISTRY
        rec = REGISTRY.get(str(recipe_id))
        if rec is not None:
            rec.expected_genus = int(genus)
    except Exception:  # noqa: BLE001
        pass


def _m3():
    import manifold3d
    return manifold3d


def _to_manifold(mesh: Mesh):
    m3 = _m3()
    return m3.Manifold(m3.Mesh(
        vert_properties=np.asarray(mesh.verts, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))


# ----------------------------------------------------------------------- knurl

@recipe(
    "pattern.knurl", "DIN 82 滚花握柄（菱形/直纹/斜纹）",
    "依据 DIN 82 滚花标准（RGE 菱形凸点、RAA 直纹、RBL 斜纹，标准螺旋角 30°、"
    "90° 齿形角、两端光面及端面倒角）：直接按相位对齐网格构造封闭柱面，无布尔碎片，"
    "恒为 genus 0 闭合单实体。",
    params={
        "diameter": {"type": "float", "default": 32.0, "min": 10.0, "max": 150.0,
                     "unit": "mm", "group": "body", "description": "外径（齿顶圆直径）"},
        "length": {"type": "float", "default": 42.0, "min": 10.0, "max": 200.0,
                   "unit": "mm", "group": "body", "description": "握柄总长"},
        "knurl_length": {"type": "float", "default": 28.0, "min": 5.0, "max": 180.0,
                         "unit": "mm", "group": "knurl", "description": "滚花段轴向长度（居中）"},
        "pattern": {"type": "enum", "default": "diamond",
                    "enum": ["diamond", "straight", "helical"], "unit": "none",
                    "group": "knurl", "description": "滚花纹型：diamond(RGE)/straight(RAA)/helical(RBL)"},
        "pitch": {"type": "float", "default": 2.4, "min": 0.8, "max": 8.0,
                  "unit": "mm", "group": "knurl", "description": "周向节距 P"},
        "depth": {"type": "float", "default": 0.9, "min": 0.2, "max": 3.0,
                  "unit": "mm", "group": "knurl", "description": "齿深（径向切深）"},
        "helix_deg": {"type": "float", "default": 30.0, "min": 10.0, "max": 60.0,
                      "unit": "deg", "group": "knurl", "description": "螺旋角（DIN 82 标准为 30°）"},
        "end_chamfer": {"type": "float", "default": 1.6, "min": 0.2, "max": 10.0,
                        "unit": "mm", "group": "edge", "description": "两端倒角"},
    },
    presets={
        "safe": {"diameter": 34.0, "length": 40.0, "knurl_length": 26.0,
                 "pattern": "straight", "pitch": 2.8, "depth": 0.9,
                 "helix_deg": 30.0, "end_chamfer": 1.6},
        "nominal": {"diameter": 32.0, "length": 42.0, "knurl_length": 28.0,
                    "pattern": "diamond", "pitch": 2.4, "depth": 0.9,
                    "helix_deg": 30.0, "end_chamfer": 1.6},
        "hero": {"diameter": 38.0, "length": 54.0, "knurl_length": 38.0,
                 "pattern": "diamond", "pitch": 2.2, "depth": 0.95,
                 "helix_deg": 30.0, "end_chamfer": 2.0},
    },
    tags=["pattern", "knurl", "din82", "diamond", "grip", "polar"],
    expected_genus=0, material="steel",
    constraints=[
        {"id": "knurl_fits_length",
         "expr": "knurl_length + 2 * end_chamfer + 2.0 <= length",
         "severity": "error",
         "message": "滚花段长度加两端倒角超出总长（缺少光面退刀区）",
         "fix": "knurl_length = length - 2 * end_chamfer - 4.0"},
        {"id": "depth_fits_radius",
         "expr": "depth <= 0.15 * diameter",
         "severity": "error",
         "message": "齿深超过直径的 15%，芯部过细",
         "fix": "depth = 0.4 * pitch"},
        {"id": "depth_vs_pitch",
         "expr": "depth <= 0.65 * pitch",
         "severity": "warn",
         "message": "齿深超过 0.65 倍节距（DIN 82 标准约 0.5P），齿形过于尖锐",
         "fix": "depth = 0.45 * pitch"},
    ],
)
def _build_knurl(v, quality="draft"):
    R = 0.5 * float(v["diameter"])
    L = float(v["length"])
    kL = float(v["knurl_length"])
    pat = str(v.get("pattern") or "diamond").lower()
    pitch = float(v["pitch"])
    depth = float(v["depth"])
    helix = math.radians(float(v["helix_deg"]))
    cham = float(v["end_chamfer"])

    if kL + 2.0 * cham >= L or depth >= 0.45 * R:
        raise AttaError("bad_knurl", "knurl_length/chamfer exceeds length or depth exceeds radius",
                        path="knurl_length", fix="reduce knurl_length or depth")

    # Integer tooth count around circumference for seamless 2*pi periodicity.
    n_teeth = max(8, int(round(2.0 * math.pi * R / pitch)))
    sub_u = _q(quality, 4, 4, 6)
    n_theta = n_teeth * sub_u
    # Cap n_theta for reasonable draft mesh size while keeping crisp tooth peaks.
    if n_theta > 320 and quality == "draft":
        sub_u = 2
        n_theta = n_teeth * sub_u

    # Axial wave pitch Pz = pitch / tan(helix).
    pz = (2.0 * math.pi * R / n_teeth) / max(math.tan(helix), 0.1)
    z0 = 0.5 * (L - kL)
    z1 = 0.5 * (L + kL)
    ramp = min(1.5e-3, 0.12 * kL)

    if pat == "straight":
        n_kz = max(4, _q(quality, 6, 10, 14))
    else:
        sub_z = _q(quality, 4, 4, 6)
        n_kz = max(8, min(180, int(round(kL / (pz / sub_z)))))

    # Build monotonic z stations: bottom chamfer -> lower land -> knurl band -> upper land -> top chamfer.
    z_list = [0.0, cham]
    if z0 - ramp > cham + 1e-5:
        z_list.append(z0 - ramp)
    z_list.extend(np.linspace(z0, z1, n_kz + 1).tolist())
    if z1 + ramp < L - cham - 1e-5:
        z_list.append(z1 + ramp)
    z_list.extend([L - cham, L])
    zs = np.asarray(z_list, dtype=np.float64)

    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    def tri_wave(phase):
        f = phase - np.floor(phase)
        return 1.0 - np.abs(2.0 * f - 1.0)

    rings = []
    z_mid = 0.5 * L
    u_phase = n_teeth * theta / (2.0 * math.pi)
    for z in zs:
        # Envelope: 0 outside [z0 - ramp, z1 + ramp], 1 inside [z0, z1], smooth linear ramp.
        if z <= z0 - ramp or z >= z1 + ramp:
            env = 0.0
        elif z < z0:
            env = (z - (z0 - ramp)) / max(ramp, 1e-9)
        elif z > z1:
            env = ((z1 + ramp) - z) / max(ramp, 1e-9)
        else:
            env = 1.0

        # Chamfer bevel at ends.
        r_base = R
        if z < cham:
            r_base = R - (cham - z)
        elif z > L - cham:
            r_base = R - (z - (L - cham))

        if env <= 0.0:
            rr = np.full(n_theta, r_base, dtype=np.float64)
        else:
            z_rel = (z - z_mid) / pz
            if pat == "straight":
                h = tri_wave(u_phase)
            elif pat == "helical":
                h = tri_wave(u_phase + z_rel)
            else:
                h = np.minimum(tri_wave(u_phase + z_rel), tri_wave(u_phase - z_rel))
            # Outer diameter = R; valleys cut inward by depth * env * (1 - h).
            rr = r_base - depth * env * (1.0 - h)

        rings.append(np.column_stack([rr * cos_t, rr * sin_t, np.full(n_theta, z)]))

    # Construct structured cylinder mesh with bottom and top pole center vertices.
    m_rings = len(rings)
    verts_list = [np.array([[0.0, 0.0, 0.0]], dtype=np.float64)] + rings + [np.array([[0.0, 0.0, L]], dtype=np.float64)]
    verts = np.vstack(verts_list)
    faces = []
    # Bottom cap fan (center idx 0, ring 0 starts at idx 1, normal -Z).
    for k in range(n_theta):
        k1 = (k + 1) % n_theta
        faces.append([0, 1 + k1, 1 + k])
    # Side quad bands.
    for i in range(m_rings - 1):
        b0 = 1 + i * n_theta
        b1 = 1 + (i + 1) * n_theta
        for k in range(n_theta):
            k1 = (k + 1) % n_theta
            faces.append([b0 + k, b0 + k1, b1 + k1])
            faces.append([b0 + k, b1 + k1, b1 + k])
    # Top cap fan (normal +Z).
    top_center = len(verts) - 1
    top_base = 1 + (m_rings - 1) * n_theta
    for k in range(n_theta):
        k1 = (k + 1) % n_theta
        faces.append([top_center, top_base + k, top_base + k1])

    mesh = oriented(Mesh(verts, np.asarray(faces, dtype=np.int64),
                         {"kind": "knurl", "pattern": pat, "teeth": n_teeth}))
    return mesh, finish(smooth_angle_deg=25.0, material="steel")


# --------------------------------------------------------------- louvre panel

@recipe(
    "pattern.louvre_panel", "百叶通风栅板（ASHRAE/AMCA 500-L）",
    "参照 ASHRAE / AMCA 500-L 建筑防雨百叶规范（45° 标准斜叶片、约 1:1 节距/进深比、"
    "带翻边外框）：外框内均匀排布 slat_count 片斜置百叶，相邻叶片保持 ≥1.5 mm 通风间隙，"
    "贯通气道数 = slat_count + 1（genus = slat_count + 1，预设均固定 6 片即 genus = 7）。",
    params={
        "width": {"type": "float", "default": 140.0, "min": 40.0, "max": 600.0,
                  "unit": "mm", "group": "frame", "description": "百叶窗总宽（X）"},
        "height": {"type": "float", "default": 120.0, "min": 40.0, "max": 600.0,
                   "unit": "mm", "group": "frame", "description": "百叶窗总高（Y）"},
        "frame_depth": {"type": "float", "default": 24.0, "min": 8.0, "max": 120.0,
                        "unit": "mm", "group": "frame", "description": "框体进深（Z）"},
        "frame_w": {"type": "float", "default": 10.0, "min": 3.0, "max": 50.0,
                    "unit": "mm", "group": "frame", "description": "边框壁宽"},
        "slat_count": {"type": "int", "default": 6, "min": 2, "max": 24,
                       "unit": "count", "group": "louvre", "description": "百叶叶片数"},
        "blade_angle_deg": {"type": "float", "default": 45.0, "min": 20.0, "max": 65.0,
                            "unit": "deg", "group": "louvre", "description": "叶片倾角（AMCA 标准为 45°）"},
        "blade_depth": {"type": "float", "default": 18.0, "min": 5.0, "max": 100.0,
                        "unit": "mm", "group": "louvre", "description": "单片百叶斜向弦长"},
        "blade_t": {"type": "float", "default": 2.4, "min": 1.2, "max": 12.0,
                    "unit": "mm", "group": "louvre", "description": "叶片厚度"},
        "flange_w": {"type": "float", "default": 6.0, "min": 0.0, "max": 30.0,
                     "unit": "mm", "group": "frame", "description": "前缘外翻安装边宽，0=无翻边"},
        "corner_radius": {"type": "float", "default": 3.0, "min": 0.0, "max": 25.0,
                          "unit": "mm", "group": "frame", "description": "外框圆角半径"},
    },
    presets={
        "safe": {"width": 130.0, "height": 116.0, "frame_depth": 22.0, "frame_w": 10.0,
                 "slat_count": 6, "blade_angle_deg": 40.0, "blade_depth": 16.0,
                 "blade_t": 2.5, "flange_w": 5.0, "corner_radius": 3.0},
        "nominal": {"width": 140.0, "height": 120.0, "frame_depth": 24.0, "frame_w": 10.0,
                    "slat_count": 6, "blade_angle_deg": 45.0, "blade_depth": 18.0,
                    "blade_t": 2.4, "flange_w": 6.0, "corner_radius": 3.0},
        "hero": {"width": 180.0, "height": 148.0, "frame_depth": 30.0, "frame_w": 12.0,
                 "slat_count": 6, "blade_angle_deg": 45.0, "blade_depth": 24.0,
                 "blade_t": 2.6, "flange_w": 8.0, "corner_radius": 4.0},
    },
    tags=["pattern", "louvre", "vent", "ashrae", "pattern_linear", "boolean"],
    expected_genus=7, material="aluminum",
    constraints=[
        {"id": "slot_clear",
         "expr": "(height - 2 * frame_w) / (slat_count + 1) >= blade_t / cos(blade_angle_deg * pi / 180) + 1.5",
         "severity": "error",
         "message": "叶片垂直节距过小，相邻百叶贴合会堵塞气道、改变拓扑 genus",
         "fix": "slat_count = int((height - 2 * frame_w) / (blade_t / cos(blade_angle_deg * pi / 180) + 3.0)) - 1"},
        {"id": "blade_within_frame_depth",
         "expr": "frame_depth >= blade_depth * cos(blade_angle_deg * pi / 180) + 1.5",
         "severity": "error",
         "message": "叶片水平投影超出边框进深",
         "fix": "frame_depth = blade_depth * cos(blade_angle_deg * pi / 180) + 4.0"},
        {"id": "blade_fits_height",
         "expr": "blade_depth * sin(blade_angle_deg * pi / 180) <= 1.8 * (height - 2 * frame_w) / (slat_count + 1)",
         "severity": "error",
         "message": "叶片过宽，上下缘会扫出框内净高",
         "fix": "blade_depth = (height - 2 * frame_w) / (slat_count + 1)"},
        {"id": "frame_min_wall",
         "expr": "frame_w >= 3.0 and blade_t >= 1.2",
         "severity": "error",
         "message": "边框宽度 < 3 mm 或叶片厚 < 1.2 mm",
         "fix": "frame_w = 8.0; blade_t = 2.0"},
    ],
)
def _build_louvre_panel(v, quality="draft"):
    W = float(v["width"])
    H = float(v["height"])
    D = float(v["frame_depth"])
    fw = float(v["frame_w"])
    n_slats = int(v["slat_count"])
    ang_deg = float(v["blade_angle_deg"])
    bd = float(v["blade_depth"])
    bt = float(v["blade_t"])
    fl_w = float(v["flange_w"])
    cr = float(v["corner_radius"])

    inner_w = W - 2.0 * fw
    inner_h = H - 2.0 * fw
    if inner_w <= 4.0e-3 or inner_h <= 4.0e-3:
        raise AttaError("frame_too_thick", "frame_w leaves no opening", path="frame_w",
                        fix="reduce frame_w or enlarge width/height")

    per_c = _q(quality, 5, 8, 12)
    outer_xy = rounded_rect(W, H, cr, per_corner=per_c)
    inner_xy = rounded_rect(inner_w, inner_h, max(0.0, cr - 0.5 * fw), per_corner=per_c)
    frame_mesh = _S.extrude(outer_xy, D, holes_xy=[inner_xy])
    parts = [_to_manifold(frame_mesh)]

    m3 = _m3()
    if fl_w > 1e-4:
        fl_t = max(1.6e-3, min(0.28 * D, 0.5 * fw))
        fl_outer = rounded_rect(W + 2.0 * fl_w, H + 2.0 * fl_w, cr + fl_w, per_corner=per_c)
        fl_inner = rounded_rect(inner_w, inner_h, max(0.0, cr - 0.5 * fw), per_corner=per_c)
        # Overlap slightly with frame front at z = D - fl_t .. D + 0.0 (offset by -0.5e-3? No, start at D - fl_t and extend past?
        # Wait: avoid coplanar z = D face with frame by letting flange protrude 0.5*fl_t past z=D? Or sit at z=0..fl_t with slight inset!).
        fl_mesh = _S.extrude(fl_outer, fl_t, holes_xy=[fl_inner]).transform(
            translate=(0.0, 0.0, -0.35 * fl_t))
        parts.append(_to_manifold(fl_mesh))

    # Build n_slats sloped blades centred inside frame depth, embedded into left/right jambs.
    blade_span = inner_w + 0.9 * fw
    pitch_y = inner_h / (n_slats + 1)
    # Use a cube rotated around X by ang_deg (tilts normal in Y-Z plane).
    # Rotate by +ang_deg around X so upper face sheds rain outward toward -Z/+Y.
    base_blade = m3.Manifold.cube([blade_span, bt, bd], True).rotate([ang_deg, 0.0, 0.0])
    z_mid = 0.5 * D
    for i in range(n_slats):
        yc = -0.5 * inner_h + (i + 1) * pitch_y
        parts.append(base_blade.translate([0.0, yc, z_mid]))

    out = m3.Manifold.batch_boolean(parts, m3.OpType.Add)
    mm = out.to_mesh()
    verts = np.asarray(mm.vert_properties, dtype=np.float64)
    faces = np.asarray(mm.tri_verts, dtype=np.int64)
    if len(faces) == 0:
        raise AttaError("boolean_empty", "louvre union produced empty mesh",
                        path="louvre_panel", fix="check blade dimensions")
    mesh = oriented(Mesh(verts[:, :3], faces, {"kind": "louvre_panel", "slats": n_slats}))
    zmin = float(mesh.verts[:, 2].min())
    if abs(zmin) > 1e-9:
        mesh = mesh.transform(translate=(0.0, 0.0, -zmin))
    _publish_genus("pattern.louvre_panel", n_slats + 1)
    return mesh, finish(smooth_angle_deg=30.0, material="aluminum")
