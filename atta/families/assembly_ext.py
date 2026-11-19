# -*- coding: utf-8 -*-
"""Extended assembly families — runner-gated mating parts with derived clearance.

Three multi-part parametric assemblies complement ``assembly.bolt_nut``
(:mod:`atta.families.pattern`).  Following the same single-mesh coverage-gate
architecture, mating components are positioned on the ``z = 0`` build datum and
linked by thin cast runner gates so every build returns one watertight solid
(``n_components == 1``) while deriving real mate clearances and reporting mate
metadata in ``mesh.meta['assembly']``:

``assembly.hinge_pin``
    Two-leaf butt hinge (Leaf A with 2 outer knuckles, Leaf B with 1 central
    knuckle, 2 mounting holes per leaf) plus a matching headed hinge pin,
    joined by two cast runner gates.  All knuckle bores derive from
    ``pin_d + clearance``; axial knuckle gaps derive from ``axial_gap``.
    Exact genus = 4 mounting holes + 3 knuckle bores = **7**.

``assembly.dovetail_joint``
    Precision sliding dovetail joint (male tenon block + female mortise block)
    at a selectable dovetail angle ``angle_deg`` (e.g. 14° hardwood or 60°
    machine-tool slide).  Perpendicular flank clearance ``clearance`` widens
    the mortise walls by ``2 * clearance / cos(angle_deg)`` and adds bottom
    depth relief.  Exact genus = **0**.

``assembly.clevis_pin``
    **DIN 71752 / ISO 2340** style clevis yoke (U-fork with two ear bores),
    mating eye tang (thickness ``fork_gap - 2 * axial_clearance``), and headed
    clevis pin with transverse cotter-pin cross-hole, joined by cast runner
    gates.  Exact genus = 2 ear bores + 1 tang bore + 1 cotter hole = **4**.

References
----------
* DIN 71752 / ISO 2340. *Clevis pins — with and without head*.
* Machinery's Handbook (31st ed.). *Dovetail Slides and Tapered Gibs*;
  *Hinge Pin Clearances and Fits*.
"""

from __future__ import annotations

import math

import numpy as np

from .. import primitives as _P
from ..boolean import difference, union
from ..errors import AttaError
from ..kit import finish, quality_level
from ..mesh import Mesh, oriented, topology, weld
from ..registry import recipe
from ..surfaces import extrude, loft

__all__ = [
    "build_hinge_pin",
    "build_dovetail_joint",
    "build_clevis_pin",
]


def _chamfered_pin_z(radius: float, z0: float, z1: float, chamfer: float,
                     segments: int, center_xy=(0.0, 0.0)) -> Mesh:
    """Lofted vertical cylinder along +Z with a 45° top lead-in chamfer."""
    r = float(radius)
    lo, hi = float(z0), float(z1)
    c = max(0.0, min(float(chamfer), 0.35 * r, 0.35 * (hi - lo)))
    a = np.linspace(0.0, 2.0 * math.pi, max(12, int(segments)), endpoint=False)
    cx, cy = float(center_xy[0]), float(center_xy[1])

    def ring(rr: float, zz: float):
        return np.column_stack([cx + rr * np.cos(a), cy + rr * np.sin(a),
                                np.full(len(a), float(zz))])

    rings = [ring(r, lo)]
    if c > 1e-9:
        rings.append(ring(r, hi - c))
        rings.append(ring(max(1e-5, r - c), hi))
    else:
        rings.append(ring(r, hi))
    return loft(rings)


def _extrude_xz_along_y(poly_xz: np.ndarray, y0: float, y1: float) -> Mesh:
    """Extrude a 2D (x, z) polygon along Y from ``y0`` to ``y1``."""
    span = float(y1) - float(y0)
    m = oriented(weld(extrude(np.asarray(poly_xz, dtype=np.float64), span), 1e-9))
    return m.transform(rotate_deg=(90.0, 0.0, 0.0), translate=(0.0, float(y1), 0.0))


# ---------------------------------------------------------------- 1. hinge_pin

@recipe(
    "assembly.hinge_pin", "两页合页 + 配套台阶销轴装配体（浇道连体）",
    "平合页双页片与带帽铰链销轴的铸造连体装配：A 页片带两端外节筒与 2 个安装孔，"
    "B 页片带中央内节筒与 2 个安装孔，销轴立于浇道旁；所有节筒内孔精确派生为 "
    "pin_d + clearance，轴向配合总长派生为 3×knuckle_len + 2×axial_gap。"
    "两道薄浇道将三零件串联为单连通实体（无闭环），故 genus 恒等于 4 安装孔 + 3 节筒孔 = 7。",
    params={
        "pin_d": {"type": "float", "default": 5.0, "min": 2.5, "max": 14.0,
                  "unit": "mm", "group": "pin", "description": "铰链销轴杆径"},
        "clearance": {"type": "float", "default": 0.25, "min": 0.05, "max": 0.80,
                      "unit": "mm", "group": "fit", "description": "销孔直径配合间隙"},
        "axial_gap": {"type": "float", "default": 0.4, "min": 0.1, "max": 1.5,
                      "unit": "mm", "group": "fit", "description": "节筒端面轴向配合间隙"},
        "knuckle_wall": {"type": "float", "default": 2.5, "min": 1.2, "max": 8.0,
                         "unit": "mm", "group": "knuckle", "description": "节筒单边壁厚"},
        "knuckle_len": {"type": "float", "default": 12.0, "min": 6.0, "max": 35.0,
                        "unit": "mm", "group": "knuckle", "description": "单节筒长度"},
        "leaf_w": {"type": "float", "default": 22.0, "min": 12.0, "max": 60.0,
                   "unit": "mm", "group": "leaf", "description": "页片宽度"},
        "leaf_t": {"type": "float", "default": 3.5, "min": 1.8, "max": 10.0,
                   "unit": "mm", "group": "leaf", "description": "页片厚度"},
        "mount_hole_d": {"type": "float", "default": 4.5, "min": 2.0, "max": 10.0,
                         "unit": "mm", "group": "leaf", "description": "安装螺钉孔直径"},
        "gate_len": {"type": "float", "default": 4.0, "min": 1.5, "max": 15.0,
                     "unit": "mm", "group": "runner", "description": "铸造浇道长度"},
        "gate_h": {"type": "float", "default": 1.6, "min": 0.8, "max": 4.0,
                   "unit": "mm", "group": "runner", "description": "浇道高度"},
        "gate_w": {"type": "float", "default": 4.0, "min": 1.5, "max": 10.0,
                   "unit": "mm", "group": "runner", "description": "浇道宽度"},
    },
    presets={
        "safe": {"pin_d": 6.0, "clearance": 0.35, "axial_gap": 0.5, "knuckle_wall": 2.8,
                 "knuckle_len": 13.0, "leaf_w": 24.0, "leaf_t": 4.0, "mount_hole_d": 5.0,
                 "gate_len": 4.5, "gate_h": 1.8, "gate_w": 4.5},
        "nominal": {"pin_d": 5.0, "clearance": 0.25, "axial_gap": 0.4, "knuckle_wall": 2.5,
                    "knuckle_len": 12.0, "leaf_w": 22.0, "leaf_t": 3.5, "mount_hole_d": 4.5,
                    "gate_len": 4.0, "gate_h": 1.6, "gate_w": 4.0},
        "hero": {"pin_d": 6.0, "clearance": 0.20, "axial_gap": 0.35, "knuckle_wall": 3.0,
                 "knuckle_len": 15.0, "leaf_w": 26.0, "leaf_t": 4.0, "mount_hole_d": 5.0,
                 "gate_len": 4.5, "gate_h": 1.8, "gate_w": 4.5},
    },
    tags=["assembly", "hinge", "pin", "clearance", "mate", "runner", "boolean"],
    expected_genus=7, material="steel",
    constraints=[
        {"id": "clearance_positive", "expr": "clearance > 0.0", "severity": "error",
         "message": "销孔直径间隙必须为正，否则无法穿销",
         "fix": "clearance = 0.25"},
        {"id": "knuckle_wall_ok", "expr": "knuckle_wall >= 1.2", "severity": "error",
         "message": "节筒壁厚不足 1.2 mm，承剪易裂",
         "fix": "knuckle_wall = 2.5"},
        {"id": "mount_hole_fits",
         "expr": "mount_hole_d <= leaf_w * 0.45 and leaf_w >= pin_d + clearance + 2 * knuckle_wall + mount_hole_d + 4.0",
         "severity": "error", "message": "页片过窄，安装孔会侵占节筒或板边余量",
         "fix": "leaf_w = pin_d + clearance + 2 * knuckle_wall + mount_hole_d + 8.0"},
        {"id": "gate_fits", "expr": "gate_h <= leaf_t * 0.85", "severity": "error",
         "message": "浇道高度超过页片厚度的 85%",
         "fix": "gate_h = leaf_t * 0.5"},
    ],
)
def build_hinge_pin(v, quality="draft"):
    """Build two-leaf hinge + headed pin joined by cast runner gates."""
    pin_d = float(v["pin_d"])
    clr = float(v["clearance"])
    ax_gap = float(v["axial_gap"])
    k_wall = float(v["knuckle_wall"])
    k_len = float(v["knuckle_len"])
    leaf_w = float(v["leaf_w"])
    leaf_t = float(v["leaf_t"])
    hole_d = float(v["mount_hole_d"])
    gate_len = float(v["gate_len"])
    gate_h = float(v["gate_h"])
    gate_w = float(v["gate_w"])

    bore_d = pin_d + clr
    outer_r = 0.5 * bore_d + k_wall
    pin_r = 0.5 * pin_d
    bore_r = 0.5 * bore_d
    tot_l = 3.0 * k_len + 2.0 * ax_gap
    seg = quality_level(quality, 28, 40, 56)

    # Leaf A positioned on -X; knuckle axis along Y at x_a = -0.5 * gate_len - outer_r
    x_a_k = -0.5 * gate_len - outer_r
    y_a1 = -k_len - ax_gap
    y_a2 = +k_len + ax_gap
    plate_a = _P.box([leaf_w, tot_l, leaf_t],
                     center=(x_a_k - 0.5 * leaf_w, 0.0, 0.5 * leaf_t))
    ka1 = _P.cylinder(outer_r, k_len, center=(x_a_k, y_a1, outer_r), axis="y", segments=seg)
    ka2 = _P.cylinder(outer_r, k_len, center=(x_a_k, y_a2, outer_r), axis="y", segments=seg)
    leaf_a = union(union(plate_a, ka1), ka2)
    for y_k in (y_a1, y_a2):
        leaf_a = difference(leaf_a, _P.cylinder(bore_r, k_len + 2.0e-3,
                                                center=(x_a_k, y_k, outer_r), axis="y",
                                                segments=seg))
    hole_xa = x_a_k - 0.5 * leaf_w - 0.1 * leaf_w
    for y_h in (-0.3 * tot_l, +0.3 * tot_l):
        leaf_a = difference(leaf_a, _P.cylinder(0.5 * hole_d, leaf_t + 2.0e-3,
                                                center=(hole_xa, y_h, 0.5 * leaf_t),
                                                segments=seg))

    # Leaf B positioned on +X; knuckle axis along Y at x_b = +0.5 * gate_len + outer_r
    x_b_k = +0.5 * gate_len + outer_r
    plate_b = _P.box([leaf_w, tot_l, leaf_t],
                     center=(x_b_k + 0.5 * leaf_w, 0.0, 0.5 * leaf_t))
    kb = _P.cylinder(outer_r, k_len, center=(x_b_k, 0.0, outer_r), axis="y", segments=seg)
    leaf_b = difference(union(plate_b, kb),
                        _P.cylinder(bore_r, k_len + 2.0e-3, center=(x_b_k, 0.0, outer_r),
                                    axis="y", segments=seg))
    hole_xb = x_b_k + 0.5 * leaf_w + 0.1 * leaf_w
    for y_h in (-0.3 * tot_l, +0.3 * tot_l):
        leaf_b = difference(leaf_b, _P.cylinder(0.5 * hole_d, leaf_t + 2.0e-3,
                                                center=(hole_xb, y_h, 0.5 * leaf_t),
                                                segments=seg))

    # Hinge pin standing vertically at +X beyond Leaf B
    head_r = 0.5 * (pin_d + 2.0 * max(1.2e-3, 0.4 * k_wall))
    head_h = max(2.0e-3, 0.5 * pin_d)
    x_pin = x_b_k + leaf_w + gate_len + head_r
    pin_head = _P.cylinder(head_r, head_h, center=(x_pin, 0.0, 0.5 * head_h), segments=seg)
    pin_shank = _chamfered_pin_z(pin_r, 0.4 * head_h, head_h + tot_l,
                                 chamfer=min(1.0e-3, 0.18 * pin_r * 2.0),
                                 segments=seg, center_xy=(x_pin, 0.0))
    pin_solid = union(pin_head, pin_shank)

    # Runner gates on z=0: A <-> B and B <-> Pin (tree, no loop)
    overlap = min(0.8e-3, 0.25 * gate_len)
    gate_ab = _P.box([x_b_k - x_a_k + 2.0 * overlap, gate_w, gate_h],
                     center=(0.0, 0.0, 0.5 * gate_h))
    # Wait: make sure gate_ab meets plate_a and plate_b (which end at x_a_k and x_b_k)!
    gate_bp = _P.box([gate_len + 2.0 * overlap, gate_w, gate_h],
                     center=(x_b_k + leaf_w + 0.5 * gate_len, 0.0, 0.5 * gate_h))

    out = union(union(union(leaf_a, leaf_b), pin_solid), union(gate_ab, gate_bp))
    topo = topology(out)
    if not topo["closed"] or topo["n_components"] != 1:
        raise AttaError("assembly_topology", "hinge_pin assembly failed closure",
                        path="gate_len", fix="adjust gate_len or knuckle_wall")
    out.meta.update({
        "assembly": {
            "parts": ["leaf_a", "leaf_b", "hinge_pin", "runner_gates"],
            "pin_d_mm": round(pin_d * 1000.0, 3),
            "bore_d_mm": round(bore_d * 1000.0, 3),
            "clearance_mm": round(clr * 1000.0, 3),
            "axial_gap_mm": round(ax_gap * 1000.0, 3),
            "knuckle_stack_mm": round(tot_l * 1000.0, 3),
        }
    })
    return out, finish(material="steel", smooth_angle_deg=32.0)


# ---------------------------------------------------------------- 2. dovetail_joint

@recipe(
    "assembly.dovetail_joint", "滑动燕尾榫卯配合装配体（浇道连体）",
    "公母滑动燕尾榫卯副（支持 14° 木工燕尾与 45°–60° 机床导轨燕尾）："
    "由颈宽 neck_w、榫深 tail_depth 与燕尾角 angle_deg 精确派生底宽 "
    "base_w = neck_w + 2·tail_depth·tan(angle_deg)；母槽按法向配合间隙 clearance "
    "沿斜面法向扩宽 2·clearance / cos(angle_deg) 并留出槽底退刀隙。"
    "公榫块与母槽块并排立于基准面，由薄浇道连成单一气密实体（genus = 0）。",
    params={
        "neck_w": {"type": "float", "default": 14.0, "min": 6.0, "max": 50.0,
                   "unit": "mm", "group": "dovetail", "description": "燕尾颈部宽度"},
        "tail_depth": {"type": "float", "default": 8.0, "min": 3.0, "max": 30.0,
                       "unit": "mm", "group": "dovetail", "description": "燕尾榫高度/槽深"},
        "angle_deg": {"type": "float", "default": 18.0, "min": 8.0, "max": 35.0,
                      "unit": "deg", "group": "dovetail", "description": "燕尾单边斜角"},
        "length": {"type": "float", "default": 36.0, "min": 12.0, "max": 120.0,
                   "unit": "mm", "group": "block", "description": "滑配轴向长度（Y）"},
        "clearance": {"type": "float", "default": 0.25, "min": 0.05, "max": 0.80,
                      "unit": "mm", "group": "fit", "description": "燕尾斜面法向配合间隙"},
        "wall": {"type": "float", "default": 8.0, "min": 3.0, "max": 30.0,
                 "unit": "mm", "group": "block", "description": "母槽两侧壁最小壁厚"},
        "base_h": {"type": "float", "default": 8.0, "min": 3.0, "max": 30.0,
                   "unit": "mm", "group": "block", "description": "基座背厚"},
        "gate_len": {"type": "float", "default": 4.0, "min": 1.5, "max": 15.0,
                     "unit": "mm", "group": "runner", "description": "铸造浇道长度"},
        "gate_h": {"type": "float", "default": 2.0, "min": 0.8, "max": 5.0,
                   "unit": "mm", "group": "runner", "description": "浇道高度"},
        "gate_w": {"type": "float", "default": 6.0, "min": 2.0, "max": 20.0,
                   "unit": "mm", "group": "runner", "description": "浇道宽度"},
    },
    presets={
        "safe": {"neck_w": 16.0, "tail_depth": 8.0, "angle_deg": 15.0, "length": 30.0,
                 "clearance": 0.35, "wall": 9.0, "base_h": 8.0, "gate_len": 4.0,
                 "gate_h": 2.0, "gate_w": 6.0},
        "nominal": {"neck_w": 14.0, "tail_depth": 8.0, "angle_deg": 18.0, "length": 36.0,
                    "clearance": 0.25, "wall": 8.0, "base_h": 8.0, "gate_len": 4.0,
                    "gate_h": 2.0, "gate_w": 6.0},
        "hero": {"neck_w": 18.0, "tail_depth": 10.0, "angle_deg": 22.0, "length": 44.0,
                 "clearance": 0.18, "wall": 10.0, "base_h": 10.0, "gate_len": 5.0,
                 "gate_h": 2.2, "gate_w": 8.0},
    },
    tags=["assembly", "dovetail", "joint", "slide", "clearance", "mate", "runner"],
    expected_genus=0, material="aluminum",
    constraints=[
        {"id": "clearance_positive", "expr": "clearance > 0.0", "severity": "error",
         "message": "配合间隙必须为正，否则公母燕尾无法滑入",
         "fix": "clearance = 0.25"},
        {"id": "wall_min", "expr": "wall >= 3.0", "severity": "error",
         "message": "母槽侧壁厚不足 3 mm，夹紧易变形",
         "fix": "wall = 6.0"},
        {"id": "gate_fits", "expr": "gate_h <= base_h * 0.6 and gate_w <= length * 0.6",
         "severity": "error", "message": "浇道尺寸相对于滑块过大",
         "fix": "gate_h = base_h * 0.25; gate_w = length * 0.2"},
    ],
)
def build_dovetail_joint(v, quality="draft"):
    """Build male dovetail tenon block + female mortise block joined by a runner gate."""
    neck_w = float(v["neck_w"])
    depth = float(v["tail_depth"])
    ang = math.radians(float(v["angle_deg"]))
    length = float(v["length"])
    clr = float(v["clearance"])
    wall = float(v["wall"])
    base_h = float(v["base_h"])
    gate_len = float(v["gate_len"])
    gate_h = float(v["gate_h"])
    gate_w = float(v["gate_w"])

    tan_a = math.tan(ang)
    cos_a = math.cos(ang)
    base_w = neck_w + 2.0 * depth * tan_a
    # Normal flank offset -> horizontal widening per side = clr / cos(ang)
    dx_clr = clr / max(cos_a, 1e-4)

    # Female mortise block outer width
    mort_neck = neck_w + 2.0 * dx_clr
    mort_base = base_w + 2.0 * dx_clr
    mort_depth = depth + clr
    mort_w = mort_base + 2.0 * wall
    mort_h = base_h + mort_depth

    # Male tenon block: backing block + protruding dovetail tenon
    tenon_w = max(base_w + 6.0e-3, mort_w * 0.75)

    x_male = -0.5 * (gate_len + tenon_w)
    x_fem = +0.5 * (gate_len + mort_w)

    # Build male tenon block (XZ cross-section extruded along Y)
    male_xz = np.array([
        [x_male - 0.5 * tenon_w, 0.0],
        [x_male + 0.5 * tenon_w, 0.0],
        [x_male + 0.5 * tenon_w, base_h],
        [x_male + 0.5 * neck_w, base_h],
        [x_male + 0.5 * base_w, base_h + depth],
        [x_male - 0.5 * base_w, base_h + depth],
        [x_male - 0.5 * neck_w, base_h],
        [x_male - 0.5 * tenon_w, base_h],
    ], dtype=np.float64)
    male = _extrude_xz_along_y(male_xz, -0.5 * length, +0.5 * length)

    # Build female mortise block with open dovetail groove on top (XZ polygon extruded along Y)
    fem_xz = np.array([
        [x_fem - 0.5 * mort_w, 0.0],
        [x_fem + 0.5 * mort_w, 0.0],
        [x_fem + 0.5 * mort_w, mort_h],
        [x_fem + 0.5 * mort_neck, mort_h],
        [x_fem + 0.5 * mort_base, base_h],
        [x_fem - 0.5 * mort_base, base_h],
        [x_fem - 0.5 * mort_neck, mort_h],
        [x_fem - 0.5 * mort_w, mort_h],
    ], dtype=np.float64)
    female = _extrude_xz_along_y(fem_xz, -0.5 * length, +0.5 * length)

    overlap = min(0.8e-3, 0.25 * gate_len)
    gate = _P.box([gate_len + 2.0 * overlap, gate_w, gate_h],
                  center=(0.0, 0.0, 0.5 * gate_h))
    out = union(union(male, female), gate)
    topo = topology(out)
    if not topo["closed"] or topo["n_components"] != 1:
        raise AttaError("dovetail_topology", "dovetail joint assembly failed closure",
                        path="wall", fix="increase wall or clearance")
    out.meta.update({
        "assembly": {
            "parts": ["dovetail_tenon", "dovetail_mortise", "runner_gate"],
            "neck_w_mm": round(neck_w * 1000.0, 3),
            "base_w_mm": round(base_w * 1000.0, 3),
            "angle_deg": float(v["angle_deg"]),
            "normal_clearance_mm": round(clr * 1000.0, 3),
            "mortise_neck_mm": round(mort_neck * 1000.0, 3),
            "mortise_base_mm": round(mort_base * 1000.0, 3),
        }
    })
    return out, finish(material="aluminum", smooth_angle_deg=25.0)


# ---------------------------------------------------------------- 3. clevis_pin

@recipe(
    "assembly.clevis_pin", "U 形叉耳 + 眼柄 + 带开口销孔圆柱销装配体（DIN 71752 / ISO 2340）",
    "机械连接标准的叉耳-眼柄-销轴三件连体装配：U 形叉耳两侧耳板钻同轴销孔，"
    "配合眼柄厚度精确派生为 fork_gap - 2×axial_clearance，所有销孔内径 = pin_d + clearance；"
    "配套圆柱销（参考 DIN 71752 / ISO 2340）带限位帽头、导入倒角与尾部开口销横孔。"
    "三零件由浇道树连成单实体（无闭环），故 genus 恒为 2 耳孔 + 1 眼孔 + 1 开口销孔 = 4。",
    params={
        "pin_d": {"type": "float", "default": 8.0, "min": 4.0, "max": 20.0,
                  "unit": "mm", "group": "pin", "description": "销轴公称直径"},
        "clearance": {"type": "float", "default": 0.25, "min": 0.05, "max": 0.80,
                      "unit": "mm", "group": "fit", "description": "销轴与耳/眼孔直径间隙"},
        "axial_clearance": {"type": "float", "default": 0.35, "min": 0.1, "max": 1.2,
                            "unit": "mm", "group": "fit", "description": "叉槽内单侧轴向间隙"},
        "fork_gap": {"type": "float", "default": 12.0, "min": 5.0, "max": 36.0,
                     "unit": "mm", "group": "yoke", "description": "U 形叉槽内宽"},
        "ear_t": {"type": "float", "default": 5.0, "min": 2.5, "max": 15.0,
                  "unit": "mm", "group": "yoke", "description": "单侧叉耳厚度"},
        "ear_r": {"type": "float", "default": 10.0, "min": 5.0, "max": 30.0,
                  "unit": "mm", "group": "yoke", "description": "叉耳与眼柄端头圆弧半径"},
        "cotter_d": {"type": "float", "default": 2.2, "min": 1.2, "max": 6.0,
                     "unit": "mm", "group": "pin", "description": "开口销横孔直径"},
        "gate_len": {"type": "float", "default": 4.5, "min": 1.5, "max": 15.0,
                     "unit": "mm", "group": "runner", "description": "铸造浇道长度"},
        "gate_h": {"type": "float", "default": 2.0, "min": 0.8, "max": 5.0,
                   "unit": "mm", "group": "runner", "description": "浇道高度"},
        "gate_w": {"type": "float", "default": 4.5, "min": 2.0, "max": 12.0,
                   "unit": "mm", "group": "runner", "description": "浇道宽度"},
    },
    presets={
        "safe": {"pin_d": 8.0, "clearance": 0.35, "axial_clearance": 0.45, "fork_gap": 12.0,
                 "ear_t": 5.0, "ear_r": 10.5, "cotter_d": 2.4, "gate_len": 4.5,
                 "gate_h": 2.0, "gate_w": 4.5},
        "nominal": {"pin_d": 8.0, "clearance": 0.25, "axial_clearance": 0.35, "fork_gap": 12.0,
                    "ear_t": 5.0, "ear_r": 10.0, "cotter_d": 2.2, "gate_len": 4.5,
                    "gate_h": 2.0, "gate_w": 4.5},
        "hero": {"pin_d": 10.0, "clearance": 0.20, "axial_clearance": 0.30, "fork_gap": 14.0,
                 "ear_t": 6.0, "ear_r": 12.5, "cotter_d": 2.6, "gate_len": 5.0,
                 "gate_h": 2.2, "gate_w": 5.0},
    },
    tags=["assembly", "clevis", "yoke", "pin", "din71752", "clearance", "mate", "runner"],
    expected_genus=4, material="steel",
    constraints=[
        {"id": "ear_ring_wall", "expr": "ear_r - 0.5 * (pin_d + clearance) >= 2.5",
         "severity": "error", "message": "叉耳/眼柄孔周单边壁厚不足 2.5 mm，抗拉撕裂裕度不够",
         "fix": "ear_r = 0.5 * (pin_d + clearance) + 4.5"},
        {"id": "tang_thickness_ok", "expr": "fork_gap - 2.0 * axial_clearance >= 3.0",
         "severity": "error", "message": "眼柄派生厚度不足 3.0 mm",
         "fix": "fork_gap = 10.0; axial_clearance = 0.35"},
        {"id": "cotter_fits_pin", "expr": "cotter_d <= 0.45 * pin_d",
         "severity": "error", "message": "开口销横孔过大，会削断销轴尾部",
         "fix": "cotter_d = 0.28 * pin_d"},
        {"id": "gate_fits", "expr": "gate_h <= ear_t * 0.8",
         "severity": "error", "message": "浇道高度超过叉耳厚度的 80%",
         "fix": "gate_h = ear_t * 0.4"},
    ],
)
def build_clevis_pin(v, quality="draft"):
    """Build clevis yoke, mating eye tang, and cross-drilled pin joined by gates."""
    pin_d = float(v["pin_d"])
    clr = float(v["clearance"])
    ax_clr = float(v["axial_clearance"])
    fork_gap = float(v["fork_gap"])
    ear_t = float(v["ear_t"])
    ear_r = float(v["ear_r"])
    cotter_d = float(v["cotter_d"])
    gate_len = float(v["gate_len"])
    gate_h = float(v["gate_h"])
    gate_w = float(v["gate_w"])

    bore_d = pin_d + clr
    bore_r = 0.5 * bore_d
    tang_t = fork_gap - 2.0 * ax_clr
    yoke_w = fork_gap + 2.0 * ear_t
    stem_l = max(12.0e-3, 1.4 * ear_r)
    ear_reach = max(14.0e-3, 2.2 * ear_r)
    seg = quality_level(quality, 28, 40, 56)

    # 1. Clevis Yoke on -X (ears aligned along X, bore axis along X, base sitting on z = 0)
    x_yoke_c = -0.5 * gate_len - 0.5 * yoke_w
    # Base block in z in [0, stem_l], ears rise from stem_l to stem_l + ear_reach
    # Pin axis at (x_yoke_c, 0, z_bore) where z_bore = stem_l + ear_reach - ear_r
    z_bore = stem_l + ear_reach - ear_r
    base = _P.box([yoke_w, 2.0 * ear_r, stem_l],
                  center=(x_yoke_c, 0.0, 0.5 * stem_l))
    ear_rect_h = ear_reach - ear_r + 1.0e-3
    ears = []
    for side in (-1.0, +1.0):
        ex = x_yoke_c + side * (0.5 * fork_gap + 0.5 * ear_t)
        rect = _P.box([ear_t, 2.0 * ear_r, ear_rect_h],
                      center=(ex, 0.0, stem_l - 1.0e-3 + 0.5 * ear_rect_h))
        cyl = _P.cylinder(ear_r, ear_t, center=(ex, 0.0, z_bore), axis="x", segments=seg)
        ears.append(union(rect, cyl))
    yoke = union(union(base, ears[0]), ears[1])
    yoke = difference(yoke, _P.cylinder(bore_r, yoke_w + 4.0e-3,
                                        center=(x_yoke_c, 0.0, z_bore), axis="x",
                                        segments=seg))

    # 2. Mating Eye Tang on +X
    x_tang_c = +0.5 * gate_len + 0.5 * tang_t
    tang_base = _P.box([tang_t, 2.0 * ear_r, z_bore],
                       center=(x_tang_c, 0.0, 0.5 * z_bore))
    tang_eye = _P.cylinder(ear_r, tang_t, center=(x_tang_c, 0.0, z_bore), axis="x",
                           segments=seg)
    tang = difference(union(tang_base, tang_eye),
                      _P.cylinder(bore_r, tang_t + 4.0e-3, center=(x_tang_c, 0.0, z_bore),
                                  axis="x", segments=seg))

    # 3. Clevis Pin standing vertically at +X beyond tang
    head_r = 0.5 * (pin_d + max(3.0e-3, 0.4 * pin_d))
    head_h = max(2.5e-3, 0.4 * pin_d)
    cotter_margin = max(2.5e-3, 1.5 * cotter_d)
    pin_grip = yoke_w + 1.0e-3
    pin_tot_h = pin_grip + cotter_margin + cotter_d + 2.0e-3
    x_pin = x_tang_c + 0.5 * tang_t + gate_len + head_r
    p_head = _P.cylinder(head_r, head_h, center=(x_pin, 0.0, 0.5 * head_h), segments=seg)
    p_shank = _chamfered_pin_z(0.5 * pin_d, 0.4 * head_h, head_h + pin_tot_h,
                               chamfer=min(1.0e-3, 0.2 * pin_d),
                               segments=seg, center_xy=(x_pin, 0.0))
    pin_body = union(p_head, p_shank)
    z_cotter = head_h + pin_grip + 0.5 * cotter_d + 0.5e-3
    cotter_hole = _P.cylinder(0.5 * cotter_d, pin_d + 3.0e-3,
                              center=(x_pin, 0.0, z_cotter), axis="y", segments=max(16, seg // 2))
    pin_solid = difference(pin_body, cotter_hole)

    # 4. Cast runner gates on z = 0
    overlap = min(0.8e-3, 0.25 * gate_len)
    gate_yt = _P.box([gate_len + 2.0 * overlap, gate_w, gate_h],
                     center=(0.0, 0.0, 0.5 * gate_h))
    gate_tp = _P.box([gate_len + 2.0 * overlap, gate_w, gate_h],
                     center=(x_tang_c + 0.5 * tang_t + 0.5 * gate_len, 0.0, 0.5 * gate_h))

    out = union(union(union(yoke, tang), pin_solid), union(gate_yt, gate_tp))
    topo = topology(out)
    if not topo["closed"] or topo["n_components"] != 1:
        raise AttaError("clevis_topology", "clevis assembly failed closure",
                        path="ear_r", fix="increase ear_r or gate_len")
    out.meta.update({
        "assembly": {
            "parts": ["clevis_yoke", "eye_tang", "clevis_pin", "runner_gates"],
            "standard": "DIN 71752 / ISO 2340",
            "pin_d_mm": round(pin_d * 1000.0, 3),
            "bore_d_mm": round(bore_d * 1000.0, 3),
            "clearance_mm": round(clr * 1000.0, 3),
            "tang_t_mm": round(tang_t * 1000.0, 3),
            "cotter_d_mm": round(cotter_d * 1000.0, 3),
        }
    })
    return out, finish(material="steel", smooth_angle_deg=32.0)
