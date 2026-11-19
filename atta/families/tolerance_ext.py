# -*- coding: utf-8 -*-
"""Extended tolerance & fit families — ISO 286-2 pair gauge, snap-fit, PIP hinge.

Three engineering tolerance families complement
``tolerance.clearance_ladder`` (:mod:`atta.families.mech`):

``tolerance.fit_pair_gauge``
    **ISO 286-1 / ISO 286-2:2010** hole-basis fit pair coupon (``H7/g6``,
    ``H7/f7``, ``H7/h6``, ``H8/e9``, ``H7/p6``, ``H11/h11``) across nominal
    diameters ``(1..50] mm``.  Embeds the exact ISO 286-2 upper/lower limit
    deviations in micrometres and builds a chamfered gauge block carrying a
    blind precision bore (``genus == 0``) and matching chamfered shaft pin
    at the mid-tolerance diameters (with an optional FDM scale multiplier
    ``clearance_scale`` when printing scaled demonstration gauges).

``tolerance.snap_fit``
    Cantilever snap-fit design & deflection test coupon following the
    **BASF / Bayer (Covestro) Snap-Fit Design Manual** methodology:

    .. math::

        \\varepsilon_{\\max} = \\frac{K_{\\text{taper}} \\cdot t \\cdot Y}{L^2 \\cdot Q}
        \\le \\varepsilon_{\\text{allow}}

    where ``K_taper = 1.5 - 1.16 * (1 - taper_ratio)`` (1.50 for a uniform
    beam; 0.92 for a 2:1 linearly tapered beam per Bayer/Covestro), ``t`` is
    root beam thickness, ``Y`` is undercut deflection, ``L`` is beam length,
    and ``Q`` is the root wall deflection magnification factor.  The coupon
    places the cantilevered snap arm with root chamfer and barbed hook on a
    common base opposite a mating receiver wall with a rectangular catch
    window (``expected_genus == 1``).

``tolerance.hinge_gauge``
    Print-in-place (PIP) 3-knuckle knuckle-hinge radial clearance gauge.
    Leaf A integrates the two outer knuckles and central pin (a closed frame,
    genus 1); Leaf B carries the bored central knuckle (genus 1) separated by
    explicit radial and axial clearances and joined at one end by a single
    sacrificial breakaway bridge tab so the printed gauge is a single closed
    manifold (``n_components == 1``, ``expected_genus == 2``).

References
----------
* ISO 286-1:2010 & ISO 286-2:2010. *Geometrical product specifications (GPS)
  — ISO code system for tolerances on linear sizes*.
* BASF Corporation. *Snap-Fit Design Manual*.
* Bayer / Covestro. *Snap-Fit Joints for Plastics — A Design Guide*.
"""

from __future__ import annotations

import math

import numpy as np

from .. import primitives as _P
from ..boolean import difference, union
from ..errors import AttaError
from ..kit import finish, quality_level, rounded_rect
from ..mesh import Mesh, oriented, topology, weld
from ..registry import recipe
from ..surfaces import extrude, loft

__all__ = [
    "iso286_limits_um",
    "snap_fit_mechanics",
    "build_fit_pair_gauge",
    "build_snap_fit",
    "build_hinge_gauge",
]

# ---------------------------------------------------------------- ISO 286-2 table
# Nominal diameter steps (lo_exclusive, hi_inclusive) in mm -> class -> (ES/es, EI/ei) in µm
_ISO286_STEPS = (
    (1.0, 3.0),
    (3.0, 6.0),
    (6.0, 10.0),
    (10.0, 18.0),
    (18.0, 30.0),
    (30.0, 50.0),
)

_ISO286_HOLES_UM = {
    "H7": ((10, 0), (12, 0), (15, 0), (18, 0), (21, 0), (25, 0)),
    "H8": ((14, 0), (18, 0), (22, 0), (27, 0), (33, 0), (39, 0)),
    "H11": ((60, 0), (75, 0), (90, 0), (110, 0), (130, 0), (160, 0)),
}

_ISO286_SHAFTS_UM = {
    "g6": ((-2, -8), (-4, -12), (-5, -14), (-6, -17), (-7, -20), (-9, -25)),
    "f7": ((-6, -16), (-10, -22), (-13, -28), (-16, -34), (-20, -41), (-25, -50)),
    "h6": ((0, -6), (0, -8), (0, -9), (0, -11), (0, -13), (0, -16)),
    "e9": ((-14, -39), (-20, -50), (-25, -61), (-32, -75), (-40, -92), (-50, -112)),
    "p6": ((12, 6), (20, 12), (24, 15), (29, 18), (35, 22), (42, 26)),
    "h11": ((0, -60), (0, -75), (0, -90), (0, -110), (0, -130), (0, -160)),
}

_FIT_CLASSES = ("H7/g6", "H7/f7", "H7/h6", "H8/e9", "H7/p6", "H11/h11")


def iso286_limits_um(nominal_mm: float, fit_class: str) -> dict:
    """Look up ISO 286-2 hole/shaft limit deviations in micrometres."""
    fc = str(fit_class).strip()
    if fc not in _FIT_CLASSES:
        raise AttaError("bad_arg", "unknown ISO 286 fit class %r" % (fit_class,),
                        path="fit_class", fix="use one of: " + ", ".join(_FIT_CLASSES))
    d = float(nominal_mm)
    if d <= 1.0 or d > 50.0:
        raise AttaError("bad_arg", "nominal_d %.2f mm outside ISO 286 lookup (1..50 mm]" % d,
                        path="nominal_d", fix="pass nominal_d in (1.0, 50.0] mm")
    band_idx = next(i for i, (lo, hi) in enumerate(_ISO286_STEPS) if lo < d <= hi)
    hole_cls, shaft_cls = fc.split("/")
    es_h, ei_h = _ISO286_HOLES_UM[hole_cls][band_idx]
    es_s, ei_s = _ISO286_SHAFTS_UM[shaft_cls][band_idx]
    min_clr = ei_h - es_s
    max_clr = es_h - ei_s
    if min_clr >= 0:
        fit_type = "clearance"
    elif max_clr <= 0:
        fit_type = "interference"
    else:
        fit_type = "transition"
    return {
        "fit_class": fc,
        "nominal_mm": d,
        "band_mm": list(_ISO286_STEPS[band_idx]),
        "hole_limits_um": [es_h, ei_h],
        "shaft_limits_um": [es_s, ei_s],
        "hole_mid_um": 0.5 * (es_h + ei_h),
        "shaft_mid_um": 0.5 * (es_s + ei_s),
        "min_clearance_um": min_clr,
        "max_clearance_um": max_clr,
        "mean_clearance_um": 0.5 * (min_clr + max_clr),
        "fit_type": fit_type,
    }


def snap_fit_mechanics(beam_len_mm: float, beam_t_mm: float, beam_w_mm: float,
                       undercut_mm: float, taper_ratio: float = 0.65,
                       q_factor: float = 1.0, modulus_mpa: float = 2200.0,
                       entry_deg: float = 30.0, mu: float = 0.20) -> dict:
    """Compute BASF/Bayer cantilever snap-fit root strain and mating forces."""
    L = float(beam_len_mm)
    t = float(beam_t_mm)
    b = float(beam_w_mm)
    Y = float(undercut_mm)
    r = float(taper_ratio)
    Q = float(q_factor)
    # Bayer/Covestro: K=1.5 for uniform beam (r=1), K=0.917 for 2:1 tapered beam (r=0.5)
    k_taper = 1.5 - 1.166 * (1.0 - max(0.5, min(1.0, r)))
    eps = (k_taper * t * Y) / (L * L * Q)
    # Normal deflection force P = (b * t^2 * E * eps) / (6 * L * Q)  [N when mm, MPa]
    force_p = (b * t * t * modulus_mpa * eps) / (6.0 * L * Q)
    alpha = math.radians(float(entry_deg))
    denom = max(0.05, 1.0 - mu * math.tan(alpha))
    force_w = force_p * ((mu + math.tan(alpha)) / denom)
    return {
        "k_taper": round(k_taper, 4),
        "max_strain": round(eps, 5),
        "max_strain_pct": round(eps * 100.0, 3),
        "normal_force_n": round(force_p, 2),
        "insertion_force_n": round(force_w, 2),
    }


def _chamfered_pin(radius: float, height: float, chamfer: float, segments: int,
                   center_xy=(0.0, 0.0), z0: float = 0.0) -> Mesh:
    """Lofted pin rising from ``z0`` with a 45-degree chamfered tip."""
    c = max(0.0, min(float(chamfer), 0.35 * float(radius), 0.35 * float(height)))
    a = np.linspace(0.0, 2.0 * math.pi, max(12, int(segments)), endpoint=False)
    cx, cy = float(center_xy[0]), float(center_xy[1])

    def ring(rr: float, zz: float):
        return np.column_stack([cx + rr * np.cos(a), cy + rr * np.sin(a),
                                np.full(len(a), float(zz))])

    rings = [ring(radius, z0)]
    if c > 1e-9:
        rings.append(ring(radius, z0 + height - c))
        rings.append(ring(max(1e-5, radius - c), z0 + height))
    else:
        rings.append(ring(radius, z0 + height))
    return loft(rings)


# ---------------------------------------------------------------- 1. fit_pair_gauge

@recipe(
    "tolerance.fit_pair_gauge", "ISO 286-2 孔轴配合对偶量规块",
    "依据 ISO 286-1 / ISO 286-2:2010 标准查表构建孔轴配合对偶量规试件："
    "支持基孔制常用配合 H7/g6（滑配）、H7/f7（动配合）、H7/h6（定位配合）、"
    "H8/e9（松动配合）、H7/p6（压配/过盈配合）及 H11/h11（粗配合），覆盖名义直径 (1..50] mm。"
    "基座左侧为盲孔量规座（保持 genus=0），右侧为带导入倒角的名义/中差配合销，"
    "偏差细节完整写入 mesh.meta['iso286']；可选 clearance_scale 用于 FDM 放大教学演示。",
    params={
        "fit_class": {"type": "enum", "default": "H7/g6", "enum": list(_FIT_CLASSES),
                      "group": "iso286", "description": "ISO 286-2 孔/轴配合代号"},
        "nominal_d": {"type": "float", "default": 10.0, "min": 3.0, "max": 40.0,
                      "unit": "mm", "group": "iso286", "description": "名义配合直径"},
        "clearance_scale": {"type": "float", "default": 1.0, "min": 1.0, "max": 20.0,
                            "unit": "ratio", "group": "iso286",
                            "description": "微米偏差缩放倍率（1.0 = 严格 ISO 286-2 原尺寸）"},
        "gauge_depth": {"type": "float", "default": 12.0, "min": 4.0, "max": 45.0,
                        "unit": "mm", "group": "coupon", "description": "盲孔深度与销长"},
        "boss_wall": {"type": "float", "default": 4.0, "min": 2.0, "max": 15.0,
                      "unit": "mm", "group": "coupon", "description": "孔座单边壁厚"},
        "base_t": {"type": "float", "default": 5.0, "min": 2.5, "max": 15.0,
                   "unit": "mm", "group": "coupon", "description": "量规底座厚度"},
        "spacing": {"type": "float", "default": 28.0, "min": 12.0, "max": 100.0,
                    "unit": "mm", "group": "coupon", "description": "孔中心与销中心距"},
    },
    presets={
        "safe": {"fit_class": "H7/f7", "nominal_d": 8.0, "clearance_scale": 5.0,
                 "gauge_depth": 10.0, "boss_wall": 4.0, "base_t": 5.0, "spacing": 24.0},
        "nominal": {"fit_class": "H7/g6", "nominal_d": 10.0, "clearance_scale": 1.0,
                    "gauge_depth": 12.0, "boss_wall": 4.0, "base_t": 5.0, "spacing": 28.0},
        "hero": {"fit_class": "H7/p6", "nominal_d": 16.0, "clearance_scale": 1.0,
                 "gauge_depth": 18.0, "boss_wall": 5.0, "base_t": 6.0, "spacing": 40.0},
    },
    tags=["tolerance", "iso286", "fit", "gauge", "coupon", "boolean"],
    expected_genus=0, material="steel",
    constraints=[
        {"id": "station_clear", "expr": "spacing >= nominal_d + boss_wall + 5.0",
         "severity": "error", "message": "孔座与销轴中心距太小，两工位会互相干涉",
         "fix": "spacing = nominal_d + 2 * boss_wall + 8.0"},
        {"id": "depth_ratio", "expr": "gauge_depth >= 0.8 * nominal_d",
         "severity": "error", "message": "配合长度不足 0.8×名义直径，无法稳定导向",
         "fix": "gauge_depth = nominal_d * 1.2"},
        {"id": "wall_min", "expr": "boss_wall >= 2.0",
         "severity": "error", "message": "孔座壁厚不足 2.0 mm，装配时易胀裂",
         "fix": "boss_wall = 3.5"},
        {"id": "exact_iso_note", "expr": "clearance_scale == 1.0",
         "severity": "warn",
         "message": "clearance_scale != 1.0 时为微米偏差放大演示件，非严格 ISO 286-2 实物公差",
         "fix": "clearance_scale = 1.0"},
    ],
)
def build_fit_pair_gauge(v, quality="draft"):
    """Build an ISO 286-2 hole/pin fit pair gauge coupon."""
    fc = str(v["fit_class"]).strip()
    d_m = float(v["nominal_d"])
    d_mm = d_m * 1000.0
    info = iso286_limits_um(d_mm, fc)
    scale = float(v["clearance_scale"])

    hole_d = d_m + (info["hole_mid_um"] * scale) * 1e-6
    pin_d = d_m + (info["shaft_mid_um"] * scale) * 1e-6
    depth = float(v["gauge_depth"])
    wall = float(v["boss_wall"])
    base_t = float(v["base_t"])
    spacing = float(v["spacing"])

    boss_d = hole_d + 2.0 * wall
    floor_t = max(2.0e-3, 0.4 * base_t)
    boss_h = floor_t + depth - base_t
    if boss_h < 2.0e-3:
        boss_h = 2.0e-3
        floor_t = base_t + boss_h - depth
    pad_edge = max(5.0e-3, 0.35 * boss_d)
    base_l = spacing + boss_d + 2.0 * pad_edge
    base_w = max(boss_d + 2.0 * pad_edge, 1.8 * d_m)
    corner_r = min(4.0e-3, 0.2 * base_w)

    seg = quality_level(quality, 36, 48, 72)
    base = oriented(weld(extrude(rounded_rect(base_l, base_w, corner_r,
                                              per_corner=quality_level(quality, 5, 8, 12)),
                                 base_t), 1e-9))

    x_hole = -0.5 * spacing
    x_pin = +0.5 * spacing
    embed = 0.45 * base_t

    boss = _P.cylinder(0.5 * boss_d, boss_h + embed,
                       center=(x_hole, 0.0, base_t - embed + 0.5 * (boss_h + embed)),
                       segments=seg)
    pin = _chamfered_pin(0.5 * pin_d, depth + embed, chamfer=min(1.0e-3, 0.1 * d_m),
                         segments=seg, center_xy=(x_pin, 0.0), z0=base_t - embed)
    body = union(union(base, boss), pin)

    pad_cut = max(3.0e-4, 0.03 * depth)
    z_top = base_t + boss_h
    z_floor = z_top - depth
    bore = _P.cylinder(0.5 * hole_d, depth + pad_cut,
                       center=(x_hole, 0.0, 0.5 * (z_floor + z_top + pad_cut)),
                       segments=seg)
    out = difference(body, bore)
    topo = topology(out)
    if not topo["closed"] or topo["n_components"] != 1:
        raise AttaError("gauge_topology", "fit pair gauge failed closure",
                        path="spacing", fix="increase spacing or base_t")
    out.meta.update({
        "standard": "ISO 286-2:2010",
        "iso286": {
            **info,
            "clearance_scale": scale,
            "built_hole_d_mm": round(hole_d * 1000.0, 5),
            "built_pin_d_mm": round(pin_d * 1000.0, 5),
            "built_clearance_mm": round((hole_d - pin_d) * 1000.0, 5),
        },
    })
    return out, finish(material="steel", smooth_angle_deg=32.0)


# ---------------------------------------------------------------- 2. snap_fit

@recipe(
    "tolerance.snap_fit", "悬臂卡扣与受卡窗试件（BASF / Bayer 挠度设计准则）",
    "依据 BASF / Bayer (Covestro)《塑料卡扣设计指南》建立的悬臂梁卡扣实验件："
    "悬臂根部带加强倒角/斜肋以抑制应力集中，支持等截面或线性渐变厚度（taper_ratio）；"
    "按 ε_max = K_taper·t·Y / (L²·Q) 校核最大弯曲应变 <= 许用应变（ABS/PA ~2.0%）。"
    "底座右侧对应立壁开设矩形锁扣通窗与倒扣导入间隙，构成完整单实体测试样件（genus = 1）。",
    params={
        "beam_len": {"type": "float", "default": 24.0, "min": 10.0, "max": 60.0,
                     "unit": "mm", "group": "beam", "description": "悬臂有效长度 L"},
        "beam_t": {"type": "float", "default": 2.0, "min": 0.9, "max": 5.0,
                   "unit": "mm", "group": "beam", "description": "悬臂根部厚度 t"},
        "beam_w": {"type": "float", "default": 8.0, "min": 3.0, "max": 25.0,
                   "unit": "mm", "group": "beam", "description": "悬臂宽度 b"},
        "taper_ratio": {"type": "float", "default": 0.65, "min": 0.5, "max": 1.0,
                        "unit": "ratio", "group": "beam",
                        "description": "梢/根厚度比（0.5 = 2:1 线性渐薄等应力梁；1.0 = 等厚）"},
        "undercut": {"type": "float", "default": 1.6, "min": 0.5, "max": 5.0,
                     "unit": "mm", "group": "hook", "description": "倒扣挠度 Y"},
        "entry_deg": {"type": "float", "default": 30.0, "min": 15.0, "max": 45.0,
                      "unit": "deg", "group": "hook", "description": "导入斜面角 α"},
        "retention_deg": {"type": "float", "default": 75.0, "min": 45.0, "max": 90.0,
                          "unit": "deg", "group": "hook", "description": "锁紧回退角 β"},
        "allow_strain_pct": {"type": "float", "default": 2.0, "min": 0.8, "max": 5.0,
                             "unit": "ratio", "group": "material",
                             "description": "材料许用设计应变（%）"},
        "clearance": {"type": "float", "default": 0.3, "min": 0.1, "max": 1.0,
                      "unit": "mm", "group": "fit", "description": "受卡窗配合间隙"},
        "base_t": {"type": "float", "default": 4.0, "min": 2.0, "max": 10.0,
                   "unit": "mm", "group": "base", "description": "公共底座厚度"},
    },
    presets={
        "safe": {"beam_len": 26.0, "beam_t": 2.0, "beam_w": 8.0, "taper_ratio": 0.6,
                 "undercut": 1.4, "entry_deg": 28.0, "retention_deg": 75.0,
                 "allow_strain_pct": 2.0, "clearance": 0.35, "base_t": 4.0},
        "nominal": {"beam_len": 24.0, "beam_t": 2.0, "beam_w": 8.0, "taper_ratio": 0.65,
                    "undercut": 1.6, "entry_deg": 30.0, "retention_deg": 75.0,
                    "allow_strain_pct": 2.0, "clearance": 0.3, "base_t": 4.0},
        "hero": {"beam_len": 28.0, "beam_t": 2.2, "beam_w": 10.0, "taper_ratio": 0.55,
                 "undercut": 2.2, "entry_deg": 25.0, "retention_deg": 82.0,
                 "allow_strain_pct": 2.0, "clearance": 0.25, "base_t": 4.5},
    },
    tags=["tolerance", "snap_fit", "cantilever", "deflection", "strain", "plastic"],
    expected_genus=1, material="abs_white",
    constraints=[
        {"id": "strain_limit",
         "expr": "((1.5 - 1.166 * (1.0 - taper_ratio)) * beam_t * undercut / (beam_len * beam_len)) * 100.0 <= allow_strain_pct",
         "severity": "error",
         "message": "悬臂根部弯曲应变超过材料许用值，压合时将屈服发白或断裂",
         "fix": "beam_len = sqrt((1.5 * beam_t * undercut * 100.0) / allow_strain_pct) + 4.0"},
        {"id": "slenderness_ok", "expr": "beam_len >= 5.0 * beam_t",
         "severity": "error", "message": "悬臂长厚比 L/t < 5，根部剪切与墙体旋转主导（应加长悬臂）",
         "fix": "beam_len = 8.0 * beam_t"},
        {"id": "entry_angle_ok", "expr": "entry_deg <= 42.0",
         "severity": "warn", "message": "导入角过大（>42°）导致推入力陡增且易自锁",
         "fix": "entry_deg = 30.0"},
        {"id": "hook_wall_ok", "expr": "beam_w >= 3.0",
         "severity": "error", "message": "卡扣梁宽过窄，受卡窗边壁难以成形",
         "fix": "beam_w = 6.0"},
    ],
)
def build_snap_fit(v, quality="draft"):
    """Build a cantilever snap-fit arm + mating slotted receiver wall on one base."""
    L = float(v["beam_len"])
    t_root = float(v["beam_t"])
    b = float(v["beam_w"])
    taper = float(v["taper_ratio"])
    Y = float(v["undercut"])
    alpha = math.radians(float(v["entry_deg"]))
    beta = math.radians(float(v["retention_deg"]))
    clr = float(v["clearance"])
    base_t = float(v["base_t"])

    t_tip = taper * t_root
    # Hook geometry along X (from root anchor at left toward +X, or vertical cantilever along +Z!)
    # Building the cantilever standing vertically (+Z) from the base plate:
    #   - No support needed for 3D printing!
    #   - Root fillet/chamfer is clean, hook protrudes in +X by Y!
    #   - Mating receiver wall stands on the right (+X) with a through-window catch!
    ret_h = max(0.8e-3, Y / max(math.tan(beta), 0.25))
    entry_h = max(1.2e-3, Y / max(math.tan(alpha), 0.25))
    flat_h = max(0.8e-3, 0.35 * t_tip)

    # Profile in (X, Z) plane extruded along Y in [-b/2, +b/2]:
    # Left back face at x = 0, front face tapers from x = t_root at z=base_t to x = t_tip at z=base_t+L,
    # then hook protrudes to x = t_tip + Y!
    z0 = base_t - 0.45 * base_t  # sink into base plate for overlap
    z_root = base_t
    z_hook_bot = z_root + L - ret_h - flat_h - entry_h
    if z_hook_bot < z_root + 0.4 * L:
        z_hook_bot = z_root + 0.4 * L
    t_at_hook = t_root + (t_tip - t_root) * ((z_hook_bot - z_root) / L)
    # 2D (x, z) profile of cantilever arm + root chamfer
    fillet = min(0.5 * t_root, 1.5e-3)
    prof_xz = np.array([
        [0.0, z0],
        [t_root + fillet, z0],
        [t_root + fillet, z_root],
        [t_root, z_root + fillet],
        [t_at_hook, z_hook_bot],
        [t_at_hook + Y, z_hook_bot + ret_h],
        [t_at_hook + Y, z_hook_bot + ret_h + flat_h],
        [t_tip, z_root + L],
        [0.0, z_root + L],
    ], dtype=np.float64)
    # Extrude in XY (x, z_target) along +Z by b, then rotate +90° around X so (x, y, z) -> (x, -z, y)
    arm = oriented(weld(extrude(prof_xz, b), 1e-9)).transform(
        rotate_deg=(90.0, 0.0, 0.0), translate=(0.0, 0.5 * b, 0.0))

    # Mating receiver wall placed at +X with a through-window catch slot (genus = 1)
    wall_t = max(2.5e-3, 1.2 * t_root)
    win_h = ret_h + flat_h + entry_h + 2.0 * clr
    win_w = b + 2.0 * clr
    wall_h = z_root + L + 3.0e-3
    wall_w = win_w + 2.0 * max(2.5e-3, 0.35 * b)
    gap_x = max(8.0e-3, 2.5 * (t_root + Y))
    x_arm_anchor = -0.5 * (gap_x + t_root + wall_t)
    arm = arm.transform(translate=(x_arm_anchor, 0.0, 0.0))

    x_wall = x_arm_anchor + t_root + gap_x + 0.5 * wall_t
    wall = _P.box([wall_t, wall_w, wall_h - z0],
                  center=(x_wall, 0.0, 0.5 * (z0 + wall_h)))
    z_win = z_hook_bot + 0.5 * (ret_h + flat_h + entry_h)
    window = _P.box([wall_t + 2.0e-3, win_w, win_h],
                    center=(x_wall, 0.0, z_win))
    receiver = difference(wall, window)

    base_l = (x_wall + 0.5 * wall_t) - x_arm_anchor + 8.0e-3
    base_w = max(wall_w + 6.0e-3, b + 10.0e-3)
    base_xc = 0.5 * (x_arm_anchor + x_wall + 0.5 * wall_t)
    base = oriented(weld(extrude(rounded_rect(base_l, base_w, min(3.0e-3, 0.2 * base_w),
                                              per_corner=quality_level(quality, 4, 6, 8)),
                                 base_t), 1e-9)).transform(translate=(base_xc, 0.0, 0.0))

    out = union(union(base, arm), receiver)
    topo = topology(out)
    if not topo["closed"] or topo["n_components"] != 1:
        raise AttaError("snap_topology", "snap-fit coupon failed closure",
                        path="beam_len", fix="increase beam_len or wall thickness")

    mech = snap_fit_mechanics(L * 1000.0, t_root * 1000.0, b * 1000.0, Y * 1000.0,
                              taper_ratio=taper, entry_deg=float(v["entry_deg"]))
    out.meta.update({"standard": "BASF / Bayer (Covestro) Snap-Fit Design", "snap_fit": mech})
    return out, finish(material="abs_white", smooth_angle_deg=30.0)


# ---------------------------------------------------------------- 3. hinge_gauge

@recipe(
    "tolerance.hinge_gauge", "一体打印合页径向间隙量规（Print-in-Place Hinge）",
    "面向 FDM / SLS 一体成型（Print-in-Place）铰链的径向与轴向间隙标定样件："
    "A 页片集成两端外节筒与贯穿销轴（构成闭环 U 形框架，genus=1），"
    "B 页片带中心节筒孔（内径 = pin_d + 2×radial_clearance，genus=1），"
    "两页片在一端由一道薄易折检验筋（breakaway_tab）相连，保持单连通气密流形"
    "（n_components=1，expected_genus=2）。",
    params={
        "pin_d": {"type": "float", "default": 5.0, "min": 2.5, "max": 12.0,
                  "unit": "mm", "group": "hinge", "description": "铰链销轴直径"},
        "radial_clearance": {"type": "float", "default": 0.30, "min": 0.10, "max": 0.80,
                             "unit": "mm", "group": "clearance",
                             "description": "销轴与节筒孔单边径向间隙"},
        "axial_clearance": {"type": "float", "default": 0.40, "min": 0.15, "max": 1.20,
                            "unit": "mm", "group": "clearance",
                            "description": "相邻节筒端面轴向间隙"},
        "knuckle_wall": {"type": "float", "default": 2.2, "min": 1.2, "max": 6.0,
                         "unit": "mm", "group": "hinge", "description": "节筒壁厚"},
        "knuckle_len": {"type": "float", "default": 10.0, "min": 5.0, "max": 25.0,
                        "unit": "mm", "group": "hinge", "description": "单节筒轴向长度"},
        "leaf_w": {"type": "float", "default": 18.0, "min": 10.0, "max": 45.0,
                   "unit": "mm", "group": "leaf", "description": "合页单侧翼宽"},
        "leaf_t": {"type": "float", "default": 3.0, "min": 1.5, "max": 8.0,
                   "unit": "mm", "group": "leaf", "description": "页片厚度"},
        "tab_w": {"type": "float", "default": 3.0, "min": 1.2, "max": 8.0,
                  "unit": "mm", "group": "tab", "description": "易折检验筋宽度"},
        "tab_h": {"type": "float", "default": 1.2, "min": 0.6, "max": 3.0,
                  "unit": "mm", "group": "tab", "description": "易折检验筋高度"},
    },
    presets={
        "safe": {"pin_d": 6.0, "radial_clearance": 0.40, "axial_clearance": 0.50,
                 "knuckle_wall": 2.5, "knuckle_len": 11.0, "leaf_w": 20.0,
                 "leaf_t": 3.5, "tab_w": 3.2, "tab_h": 1.4},
        "nominal": {"pin_d": 5.0, "radial_clearance": 0.30, "axial_clearance": 0.40,
                    "knuckle_wall": 2.2, "knuckle_len": 10.0, "leaf_w": 18.0,
                    "leaf_t": 3.0, "tab_w": 3.0, "tab_h": 1.2},
        "hero": {"pin_d": 4.0, "radial_clearance": 0.20, "axial_clearance": 0.30,
                 "knuckle_wall": 1.8, "knuckle_len": 9.0, "leaf_w": 16.0,
                 "leaf_t": 2.6, "tab_w": 2.4, "tab_h": 1.0},
    },
    tags=["tolerance", "hinge", "print_in_place", "clearance", "gauge", "boolean"],
    expected_genus=2, material="nylon",
    constraints=[
        {"id": "fdm_clearance", "expr": "radial_clearance >= 0.15",
         "severity": "warn", "message": "径向间隙 < 0.15 mm 时普通 FDM 极易粘死",
         "fix": "radial_clearance = 0.25"},
        {"id": "knuckle_outer_ok", "expr": "knuckle_wall >= 1.2",
         "severity": "error", "message": "节筒壁厚不足 1.2 mm，折断试验筋时易开裂",
         "fix": "knuckle_wall = 2.0"},
        {"id": "leaf_clears_knuckle",
         "expr": "leaf_w >= pin_d * 0.5 + radial_clearance + knuckle_wall + 5.0",
         "severity": "error", "message": "页片翼宽必须明显大于节筒外半径，留出手指扳动面积",
         "fix": "leaf_w = pin_d * 0.5 + radial_clearance + knuckle_wall + 8.0"},
        {"id": "tab_breakable", "expr": "tab_h <= leaf_t * 0.65",
         "severity": "error", "message": "易折检验筋高度超过页片 65% 将难以徒手掰断",
         "fix": "tab_h = leaf_t * 0.4"},
    ],
)
def build_hinge_gauge(v, quality="draft"):
    """Build a 3-knuckle print-in-place hinge gauge with one breakaway tab."""
    pin_r = 0.5 * float(v["pin_d"])
    rad_clr = float(v["radial_clearance"])
    ax_clr = float(v["axial_clearance"])
    wall = float(v["knuckle_wall"])
    k_len = float(v["knuckle_len"])
    leaf_w = float(v["leaf_w"])
    leaf_t = float(v["leaf_t"])
    tab_w = float(v["tab_w"])
    tab_h = float(v["tab_h"])

    bore_r = pin_r + rad_clr
    outer_r = bore_r + wall
    tot_l = 3.0 * k_len + 2.0 * ax_clr
    y_a1 = -k_len - ax_clr
    y_a2 = +k_len + ax_clr
    seg = quality_level(quality, 28, 40, 56)

    pin = _P.cylinder(pin_r, tot_l, center=(0.0, 0.0, outer_r), axis="y", segments=seg)
    ka1 = _P.cylinder(outer_r, k_len, center=(0.0, y_a1, outer_r), axis="y", segments=seg)
    ka2 = _P.cylinder(outer_r, k_len, center=(0.0, y_a2, outer_r), axis="y", segments=seg)
    gap_x = outer_r + rad_clr
    plate_a = _P.box([leaf_w - gap_x, tot_l, leaf_t],
                     center=(-0.5 * (leaf_w + gap_x), 0.0, 0.5 * leaf_t))
    # Ensure tangs actually reach x = 0 inside ka1/ka2
    tang_a1 = _P.box([gap_x + 1.0e-3, k_len, leaf_t],
                     center=(-0.5 * gap_x, y_a1, 0.5 * leaf_t))
    tang_a2 = _P.box([gap_x + 1.0e-3, k_len, leaf_t],
                     center=(-0.5 * gap_x, y_a2, 0.5 * leaf_t))
    leaf_a = union(union(union(plate_a, tang_a1), tang_a2), union(pin, union(ka1, ka2)))

    kb = _P.cylinder(outer_r, k_len, center=(0.0, 0.0, outer_r), axis="y", segments=seg)
    plate_b = _P.box([leaf_w - gap_x, tot_l, leaf_t],
                     center=(+0.5 * (leaf_w + gap_x), 0.0, 0.5 * leaf_t))
    tang_b = _P.box([gap_x + 1.0e-3, k_len, leaf_t],
                    center=(+0.5 * gap_x, 0.0, 0.5 * leaf_t))
    bore = _P.cylinder(bore_r, k_len + 2.0e-3, center=(0.0, 0.0, outer_r), axis="y",
                       segments=seg)
    leaf_b = difference(union(union(plate_b, tang_b), kb), bore)

    # Single breakaway tab bridging tang_a1 (-x) to plate_b (+x) at y = y_a1
    tab = _P.box([2.0 * (gap_x + 1.2e-3), min(tab_w, 0.8 * k_len), tab_h],
                 center=(0.0, y_a1, 0.5 * tab_h))
    out = union(union(leaf_a, leaf_b), tab)

    topo = topology(out)
    if not topo["closed"] or topo["n_components"] != 1:
        raise AttaError("hinge_topology", "PIP hinge gauge failed closure",
                        path="radial_clearance", fix="increase knuckle_wall or radial_clearance")
    out.meta.update({
        "pin_d_mm": round(2.0 * pin_r * 1000.0, 3),
        "bore_d_mm": round(2.0 * bore_r * 1000.0, 3),
        "radial_clearance_mm": round(rad_clr * 1000.0, 3),
        "axial_clearance_mm": round(ax_clr * 1000.0, 3),
    })
    return out, finish(material="nylon", smooth_angle_deg=32.0)
