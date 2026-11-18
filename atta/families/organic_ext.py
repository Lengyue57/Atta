# -*- coding: utf-8 -*-
"""Extended organic families: botanical tree, Vogel phyllotaxis, staghorn coral.

Botanical & mathematical standards cited:
* ``organic.tree`` — Leonardo da Vinci's rule of trees (C. Eloy, *Physical Review
  Letters* 107, 258101, 2011): parent and daughter branch diameters obey
  ``d_parent ** delta = sum(d_child ** delta)`` with Leonardo exponent
  ``delta ~= 2.0`` (botanical range ``1.8..2.3``), combined with golden-angle
  azimuthal rotation (``~137.508 deg``) per level to avoid branch overlap.
* ``organic.phyllotaxis`` — H. Vogel (1979), *"A better way to construct the
  sunflower head"*, *Mathematical Biosciences* 44(3-4): 179-189: polar packing
  ``r_n = c * sqrt(n)``, ``theta_n = n * phi_g`` with the golden angle
  ``phi_g = pi * (3 - sqrt(5)) ~= 137.507764 deg``, mapped onto a domed
  receptacle so neighbouring bracts stay clear and genus remains 0.
* ``organic.coral`` — *Acropora cervicornis* (staghorn coral) dichotomous
  branching morphology: primary stems rising from an encrusting holdfast,
  bifurcating at ``35..55 deg`` with phototropic upward curvature and rounded
  corallite tips.

All lengths inside builders are metres; public recipe params are mm/deg.
Every preset maintains feature thickness >= 2.0 mm so the organic category
``min_wall_mm >= 0.8`` gate passes cleanly.
"""

from __future__ import annotations

import math

import numpy as np

from .. import surfaces as _S
from .. import sweepkit as _SK
from ..errors import AttaError
from ..kit import circle_loop, finish, quality_level as _q
from ..mesh import Mesh, oriented
from ..registry import recipe

__all__ = ["_build_tree", "_build_phyllotaxis", "_build_coral"]


def _m3():
    import manifold3d
    return manifold3d


def _to_manifold(mesh: Mesh):
    m3 = _m3()
    return m3.Manifold(m3.Mesh(
        vert_properties=np.asarray(mesh.verts, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))


def _align_z(up):
    """3x4 row-major transform taking +Z onto unit vector ``up``."""
    u = np.asarray(up, dtype=np.float64)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, u)
    c = float(np.dot(z, u))
    if np.linalg.norm(v) < 1e-12:
        r = np.eye(3) if c > 0.0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0.0, -v[2], v[1]],
                       [v[2], 0.0, -v[0]],
                       [-v[1], v[0], 0.0]])
        r = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    return np.hstack([r, np.zeros((3, 1))])


def _capsule_m(a, b, ra, rb, seg=12):
    """Closed tapered frustum from ``a`` to ``b`` with spherical end caps."""
    m3 = _m3()
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return m3.Manifold.sphere(float(max(ra, rb)), int(max(6, seg))).translate(a.tolist())
    cyl = m3.Manifold.cylinder(length, float(ra), float(rb), int(max(6, seg)), False)
    cyl = cyl.transform(_align_z(d / length)).translate(a.tolist())
    s_a = m3.Manifold.sphere(float(ra), int(max(6, seg))).translate(a.tolist())
    s_b = m3.Manifold.sphere(float(rb), int(max(6, seg))).translate(b.tolist())
    return cyl + s_a + s_b


def _batch_union(manifolds, kind="organic"):
    m3 = _m3()
    items = [m for m in manifolds if m is not None and not m.is_empty()]
    if not items:
        raise AttaError("boolean_empty", "no non-empty geometry to union",
                        path="organic", fix="check feature dimensions")
    out = items[0] if len(items) == 1 else m3.Manifold.batch_boolean(items, m3.OpType.Add)
    mm = out.to_mesh()
    verts = np.asarray(mm.vert_properties, dtype=np.float64)
    faces = np.asarray(mm.tri_verts, dtype=np.int64)
    if len(faces) == 0:
        raise AttaError("boolean_empty", "organic union produced empty mesh",
                        path="organic", fix="check branch overlap and diameters")
    return oriented(Mesh(verts[:, :3], faces, {"kind": kind}))


def _orthonormal_basis(axis):
    """Return two unit vectors (u, v) perpendicular to unit vector ``axis``."""
    w = np.asarray(axis, dtype=np.float64)
    w = w / max(float(np.linalg.norm(w)), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.85 else np.array([0.0, 1.0, 0.0])
    u = np.cross(w, ref)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(w, u)
    return u, v


# ----------------------------------------------------------------------- tree

@recipe(
    "organic.tree", "分形生长树（达芬奇法则）",
    "遵循达芬奇树木分枝面积守恒律（d_parent^Δ = Σ d_child^Δ，Δ≈2.0）与黄金角叶序方位旋转"
    "的递归分枝树：含根颈扩底与逐级收细圆滑分枝，受严格分枝预算约束，恒为 genus 0 闭合单实体。",
    params={
        "height": {"type": "float", "default": 120.0, "min": 40.0, "max": 400.0,
                   "unit": "mm", "group": "tree", "description": "参考树高"},
        "trunk_d": {"type": "float", "default": 16.0, "min": 5.0, "max": 80.0,
                    "unit": "mm", "group": "tree", "description": "主干基部直径"},
        "levels": {"type": "int", "default": 3, "min": 1, "max": 5,
                   "unit": "count", "group": "growth", "description": "递归分枝级数"},
        "branching": {"type": "int", "default": 2, "min": 2, "max": 4,
                      "unit": "count", "group": "growth", "description": "每级分叉子枝数"},
        "spread_deg": {"type": "float", "default": 34.0, "min": 15.0, "max": 60.0,
                       "unit": "deg", "group": "growth", "description": "子枝相对母枝的张开角"},
        "length_ratio": {"type": "float", "default": 0.72, "min": 0.45, "max": 0.90,
                         "unit": "ratio", "group": "growth", "description": "子枝相对母枝长度比"},
        "da_vinci_exp": {"type": "float", "default": 2.0, "min": 1.7, "max": 2.6,
                         "unit": "none", "group": "growth",
                         "description": "达芬奇指数 Δ（2.0 表示截面积守恒）"},
        "min_twig_d": {"type": "float", "default": 2.4, "min": 1.2, "max": 12.0,
                       "unit": "mm", "group": "tree", "description": "末梢最小枝径（保证可打印壁厚）"},
        "base_flare": {"type": "float", "default": 1.45, "min": 1.0, "max": 2.4,
                       "unit": "ratio", "group": "tree", "description": "根颈贴地扩展倍率"},
        "max_branches": {"type": "int", "default": 128, "min": 4, "max": 512,
                         "unit": "count", "group": "budget", "description": "硬性枝段数量预算上限"},
    },
    presets={
        "safe": {"height": 100.0, "trunk_d": 16.0, "levels": 2, "branching": 2,
                 "spread_deg": 35.0, "length_ratio": 0.72, "da_vinci_exp": 2.0,
                 "min_twig_d": 3.0, "base_flare": 1.45, "max_branches": 128},
        "nominal": {"height": 120.0, "trunk_d": 16.0, "levels": 3, "branching": 2,
                    "spread_deg": 34.0, "length_ratio": 0.72, "da_vinci_exp": 2.0,
                    "min_twig_d": 2.4, "base_flare": 1.45, "max_branches": 128},
        "hero": {"height": 148.0, "trunk_d": 20.0, "levels": 4, "branching": 2,
                 "spread_deg": 36.0, "length_ratio": 0.73, "da_vinci_exp": 2.0,
                 "min_twig_d": 2.2, "base_flare": 1.50, "max_branches": 128},
    },
    tags=["organic", "tree", "branching", "recursion", "da_vinci", "boolean"],
    expected_genus=0, material="wood",
    constraints=[
        {"id": "branch_budget",
         "expr": "(branching ** (levels + 1) - 1) / (branching - 1) <= max_branches",
         "severity": "error",
         "message": "总枝段数超过 max_branches 预算上限",
         "fix": "levels = 3, branching = 2"},
        {"id": "twig_printable",
         "expr": "min_twig_d >= 1.2",
         "severity": "error",
         "message": "末梢最小直径低于 1.2 mm（不满足有机类最小壁厚要求）",
         "fix": "min_twig_d = 2.0"},
        {"id": "trunk_larger_than_twig",
         "expr": "trunk_d >= min_twig_d * 1.5",
         "severity": "error",
         "message": "主干直径相对末梢过细",
         "fix": "trunk_d = min_twig_d * 4.0"},
        {"id": "spread_avoids_self",
         "expr": "spread_deg >= 20.0",
         "severity": "warn",
         "message": "分叉角小于 20°，子枝可能互相贴合",
         "fix": "spread_deg = 32.0"},
    ],
)
def _build_tree(v, quality="draft"):
    H = float(v["height"])
    trunk_r = 0.5 * float(v["trunk_d"])
    levels = int(v["levels"])
    branching = int(v["branching"])
    spread = math.radians(float(v["spread_deg"]))
    l_ratio = float(v["length_ratio"])
    delta = float(v["da_vinci_exp"])
    min_r = 0.5 * float(v["min_twig_d"])
    flare = float(v["base_flare"])
    max_b = int(v["max_branches"])

    total_segs = (branching ** (levels + 1) - 1) // (branching - 1)
    if total_segs > max_b:
        raise AttaError("budget_exceeded",
                        "tree needs %d branches, exceeding max_branches=%d" % (total_segs, max_b),
                        path="levels", fix="reduce levels or branching, or raise max_branches")
    if min_r < 0.5e-3:
        raise AttaError("bad_twig", "min_twig_d must be >= 1.0 mm", path="min_twig_d",
                        fix="set min_twig_d >= 1.5")

    seg = _q(quality, 10, 14, 20)
    # Normalise trunk length so total geometric height approximates H.
    geom_sum = sum(l_ratio ** k for k in range(levels + 1))
    trunk_len = H / max(1.0, geom_sum * 0.92)
    r_decay = branching ** (-1.0 / delta)
    golden_az = math.radians(137.507764)

    parts = []
    # Root flare mound sitting flat on z = 0.
    flare_h = max(3.0e-3, 0.22 * trunk_len)
    flare_prof = np.array([
        [0.0, 0.0],
        [trunk_r * flare, 0.0],
        [trunk_r * (0.55 + 0.45 * flare), 0.45 * flare_h],
        [trunk_r * 1.02, flare_h],
        [0.0, flare_h],
    ], dtype=np.float64)
    parts.append(_to_manifold(_S.revolve(flare_prof, segments=max(16, seg), caps=False)))

    def grow(p0, direction, length, r0, level, az_seed):
        r1 = max(min_r, r0 * (0.78 if level == 0 else r_decay ** 0.5))
        p1 = p0 + direction * length
        parts.append(_capsule_m(p0, p1, r0, r1, seg=seg))
        if level >= levels:
            return
        u, v_ax = _orthonormal_basis(direction)
        child_len = length * l_ratio
        child_r0 = max(min_r, r1 * (r_decay if level > 0 else r_decay))
        # Ensure child never exceeds parent tip radius so joints remain clean.
        child_r0 = min(child_r0, r1 * 0.96)
        for k in range(branching):
            az = az_seed + 2.0 * math.pi * k / branching
            radial = math.cos(az) * u + math.sin(az) * v_ax
            # Bias slightly upward (phototropism) for natural crown silhouette.
            cdir = math.cos(spread) * direction + math.sin(spread) * radial
            cdir = cdir + np.array([0.0, 0.0, 0.14])
            cdir /= max(float(np.linalg.norm(cdir)), 1e-12)
            grow(p1, cdir, child_len, child_r0, level + 1, az + golden_az)

    grow(np.array([0.0, 0.0, 0.5 * flare_h]), np.array([0.0, 0.0, 1.0]),
         trunk_len, trunk_r, 0, 0.0)

    mesh = _batch_union(parts, kind="tree")
    # Shift onto z = 0 cleanly.
    zmin = float(mesh.verts[:, 2].min())
    if abs(zmin) > 1e-9:
        mesh = mesh.transform(translate=(0.0, 0.0, -zmin))
    mesh.meta.update({"kind": "tree", "levels": levels, "branches": total_segs})
    return mesh, finish(smooth_angle_deg=45.0, material="wood")


# ---------------------------------------------------------------- phyllotaxis

@recipe(
    "organic.phyllotaxis", "黄金角叶序穹盘（Vogel 向日葵模型）",
    "基于 Vogel (1979) 黄金角叶序数学模型（r_n = c√n，θ_n = n·137.507764°）："
    "在平滑穹顶花托表面排布斐波那契螺旋鳞片/小花，各单元沿曲面法向伸展且彼此保持间隙，"
    "无邻接穿插成环，恒为 genus 0 闭合单实体。",
    params={
        "base_d": {"type": "float", "default": 90.0, "min": 30.0, "max": 300.0,
                   "unit": "mm", "group": "body", "description": "花托底座直径"},
        "dome_h": {"type": "float", "default": 32.0, "min": 10.0, "max": 150.0,
                   "unit": "mm", "group": "body", "description": "穹顶隆起高度"},
        "base_t": {"type": "float", "default": 6.0, "min": 2.0, "max": 30.0,
                   "unit": "mm", "group": "body", "description": "底盘实心厚度"},
        "count": {"type": "int", "default": 55, "min": 13, "max": 233,
                  "unit": "count", "group": "pattern", "description": "斐波那契螺旋单元数量"},
        "divergence_deg": {"type": "float", "default": 137.508, "min": 90.0, "max": 180.0,
                           "unit": "deg", "group": "pattern", "description": "发散角（默认黄金角 137.508°）"},
        "floret_r": {"type": "float", "default": 2.6, "min": 1.4, "max": 15.0,
                     "unit": "mm", "group": "floret", "description": "中心小花/鳞片基准半径"},
        "floret_h": {"type": "float", "default": 6.5, "min": 2.0, "max": 30.0,
                     "unit": "mm", "group": "floret", "description": "小花/鳞片外突高度"},
        "floret_scale_outer": {"type": "float", "default": 1.4, "min": 0.8, "max": 2.2,
                               "unit": "ratio", "group": "floret", "description": "外圈相对中心小花放大倍率"},
    },
    presets={
        "safe": {"base_d": 86.0, "dome_h": 28.0, "base_t": 6.0, "count": 34,
                 "divergence_deg": 137.508, "floret_r": 3.2, "floret_h": 6.5,
                 "floret_scale_outer": 1.3},
        "nominal": {"base_d": 90.0, "dome_h": 32.0, "base_t": 6.0, "count": 55,
                    "divergence_deg": 137.508, "floret_r": 2.6, "floret_h": 6.5,
                    "floret_scale_outer": 1.4},
        "hero": {"base_d": 116.0, "dome_h": 42.0, "base_t": 7.0, "count": 89,
                 "divergence_deg": 137.508, "floret_r": 2.4, "floret_h": 7.5,
                 "floret_scale_outer": 1.45},
    },
    tags=["organic", "phyllotaxis", "fibonacci", "golden_angle", "vogel", "boolean"],
    expected_genus=0, material="ceramic",
    constraints=[
        {"id": "floret_printable",
         "expr": "floret_r >= 1.2",
         "severity": "error",
         "message": "小花单元半径过小（直径低于 2.4 mm）",
         "fix": "floret_r = 2.0"},
        {"id": "floret_packing_clear",
         "expr": "count * (floret_r * floret_scale_outer) ** 2 <= 0.42 * (0.5 * base_d) ** 2",
         "severity": "error",
         "message": "单元面积占比过大，相邻小花可能互相接触围出微孔",
         "fix": "floret_r = 0.55 * (0.5 * base_d) / (sqrt(count) * floret_scale_outer)"},
        {"id": "dome_supports_florets",
         "expr": "base_t >= 3.0",
         "severity": "warn",
         "message": "底座厚度不足 3 mm",
         "fix": "base_t = 5.0"},
    ],
)
def _build_phyllotaxis(v, quality="draft"):
    R = 0.5 * float(v["base_d"])
    dome_h = float(v["dome_h"])
    base_t = float(v["base_t"])
    count = int(v["count"])
    div = math.radians(float(v["divergence_deg"]))
    fr0 = float(v["floret_r"])
    fh = float(v["floret_h"])
    fscale = float(v["floret_scale_outer"])

    seg_dome = _q(quality, 40, 64, 96)
    seg_floret = _q(quality, 8, 12, 16)

    # Parabolic/cosine domed receptacle sitting flat on z = 0.
    n_merid = _q(quality, 14, 22, 32)
    ts = np.linspace(0.0, 1.0, n_merid)
    prof = [(0.0, 0.0), (R, 0.0), (R, base_t)]
    for t in ts[::-1]:
        rr = R * t
        zz = base_t + dome_h * (1.0 - t * t)
        prof.append((rr, zz))
    dome = _to_manifold(_S.revolve(np.asarray(prof, dtype=np.float64), segments=seg_dome, caps=False))
    parts = [dome]

    # Place florets via Vogel's law on r_norm in [0.08, 0.90], deeply anchored in the dome.
    m3 = _m3()
    for i in range(count):
        s = math.sqrt((i + 0.5) / count) * 0.90
        r_xy = R * s
        theta = i * div
        z_surf = base_t + dome_h * (1.0 - s * s)
        # Outward surface normal to z(r) = base_t + dome_h * (1 - (r/R)^2):
        # dz/dr = -2 * dome_h * r / R^2 -> normal ~ (2*dome_h*s/R * cos, 2*dome_h*s/R * sin, 1)
        slope = 2.0 * dome_h * s / max(R, 1e-9)
        nrm = np.array([slope * math.cos(theta), slope * math.sin(theta), 1.0])
        nrm /= max(float(np.linalg.norm(nrm)), 1e-12)

        scale_i = 1.0 + (fscale - 1.0) * s
        ri = fr0 * scale_i
        hi = fh * (0.85 + 0.30 * s)
        embed = min(0.65 * base_t, 0.45 * hi)
        p_surf = np.array([r_xy * math.cos(theta), r_xy * math.sin(theta), z_surf])
        p_a = p_surf - embed * nrm
        p_b = p_surf + hi * nrm
        # Tapered bud / bract with spherical rounded tip.
        r_tip = max(0.65e-3, 0.42 * ri)
        bud = m3.Manifold.cylinder(float(np.linalg.norm(p_b - p_a)), ri, r_tip, seg_floret, False)
        bud = bud + m3.Manifold.sphere(r_tip, seg_floret).translate([0.0, 0.0, float(np.linalg.norm(p_b - p_a))])
        bud = bud.transform(_align_z(nrm)).translate(p_a.tolist())
        parts.append(bud)

    mesh = _batch_union(parts, kind="phyllotaxis")
    mesh.meta.update({"kind": "phyllotaxis", "count": count})
    return mesh, finish(smooth_angle_deg=35.0, material="ceramic")


# ----------------------------------------------------------------------- coral

@recipe(
    "organic.coral", "鹿角珊瑚群落（Acropora 分叉枝）",
    "模拟鹿角珊瑚（Acropora cervicornis）双叉生长形态：底部包壳固着基盘（holdfast）"
    "向上分出若干曲率渐升的叉状分支，末端圆润收束，利用 sweepkit 平行标架扫掠，"
    "枝条彼此散开不成闭环，恒为 genus 0 闭合单实体。",
    params={
        "base_d": {"type": "float", "default": 46.0, "min": 18.0, "max": 150.0,
                   "unit": "mm", "group": "base", "description": "固着基盘外径"},
        "height": {"type": "float", "default": 95.0, "min": 35.0, "max": 300.0,
                   "unit": "mm", "group": "coral", "description": "参考珊瑚高度"},
        "branch_d": {"type": "float", "default": 11.0, "min": 4.0, "max": 40.0,
                     "unit": "mm", "group": "coral", "description": "主枝基部直径"},
        "generations": {"type": "int", "default": 2, "min": 1, "max": 4,
                        "unit": "count", "group": "growth", "description": "双叉分枝代数"},
        "arm_count": {"type": "int", "default": 3, "min": 2, "max": 5,
                      "unit": "count", "group": "growth", "description": "基盘生出的主干数"},
        "fork_angle_deg": {"type": "float", "default": 42.0, "min": 20.0, "max": 62.0,
                           "unit": "deg", "group": "growth", "description": "双叉分叉半角"},
        "curvature": {"type": "float", "default": 0.28, "min": 0.0, "max": 0.60,
                      "unit": "ratio", "group": "coral", "description": "向光向上弯曲强度"},
        "taper": {"type": "float", "default": 0.76, "min": 0.55, "max": 0.90,
                  "unit": "ratio", "group": "coral", "description": "每代枝径收缩率"},
    },
    presets={
        "safe": {"base_d": 44.0, "height": 82.0, "branch_d": 12.0, "generations": 2,
                 "arm_count": 3, "fork_angle_deg": 44.0, "curvature": 0.25, "taper": 0.78},
        "nominal": {"base_d": 46.0, "height": 95.0, "branch_d": 11.0, "generations": 2,
                    "arm_count": 3, "fork_angle_deg": 42.0, "curvature": 0.28, "taper": 0.76},
        "hero": {"base_d": 56.0, "height": 118.0, "branch_d": 12.0, "generations": 3,
                 "arm_count": 4, "fork_angle_deg": 42.0, "curvature": 0.30, "taper": 0.76},
    },
    tags=["organic", "coral", "acropora", "sweep", "bifurcation", "boolean"],
    expected_genus=0, material="bio_bone",
    constraints=[
        {"id": "tip_printable",
         "expr": "branch_d * (taper ** generations) >= 2.2",
         "severity": "error",
         "message": "末梢珊瑚枝直径过细（低于 2.2 mm）",
         "fix": "taper = 0.80  (or reduce generations)"},
        {"id": "base_holds_arms",
         "expr": "base_d >= 2.4 * branch_d",
         "severity": "error",
         "message": "固着基盘太窄，无法承托多根主枝",
         "fix": "base_d = 3.2 * branch_d"},
        {"id": "budget_caps_branches",
         "expr": "arm_count * (2 ** (generations + 1) - 1) <= 96",
         "severity": "error",
         "message": "总分枝数超过 96 条预算上限",
         "fix": "generations = 2, arm_count = 3"},
    ],
)
def _build_coral(v, quality="draft"):
    base_r = 0.5 * float(v["base_d"])
    H = float(v["height"])
    br0 = 0.5 * float(v["branch_d"])
    gens = int(v["generations"])
    arms = int(v["arm_count"])
    fork = math.radians(float(v["fork_angle_deg"]))
    curv = float(v["curvature"])
    taper = float(v["taper"])

    if arms * (2 ** (gens + 1) - 1) > 96:
        raise AttaError("budget_exceeded", "coral branch count exceeds 96",
                        path="generations", fix="reduce generations or arm_count")

    seg_ring = _q(quality, 12, 18, 24)
    path_pts = _q(quality, 7, 10, 14)
    seg_sphere = _q(quality, 10, 14, 20)

    # Organic encrusting holdfast mound on z = 0.
    hold_h = max(4.0e-3, 0.14 * H)
    prof = np.array([
        [0.0, 0.0],
        [base_r, 0.0],
        [0.94 * base_r, 0.35 * hold_h],
        [0.65 * base_r, 0.82 * hold_h],
        [0.0, hold_h],
    ], dtype=np.float64)
    parts = [_to_manifold(_S.revolve(prof, segments=max(23, seg_ring * 2 + 1), caps=False)).rotate([0.0, 0.0, 1.37])]

    unit_circle = circle_loop(1.0, seg_ring)
    len0 = H / max(1.0, sum(0.82 ** g for g in range(gens + 1)))

    def add_curved_branch(p0, d0, length, r0, r1):
        # Phototropic curve: start along d0, bend progressively toward +Z.
        d0_u = d0 / max(float(np.linalg.norm(d0)), 1e-12)
        up = np.array([0.0, 0.0, 1.0])
        d_end = (1.0 - curv) * d0_u + curv * up
        d_end /= max(float(np.linalg.norm(d_end)), 1e-12)
        ts = np.linspace(0.0, 1.0, path_pts)
        # Quadratic Bezier with control point pulling gently upward.
        ctrl = p0 + 0.5 * length * d0_u
        end = p0 + 0.5 * length * d0_u + 0.5 * length * d_end
        pts_main = ((1.0 - ts)[:, None] ** 2) * p0 + (2.0 * (1.0 - ts) * ts)[:, None] * ctrl + (ts[:, None] ** 2) * end
        sc_main = r0 + (r1 - r0) * ts
        # Append 3 hemispherical dome-tip stations directly in the sweep (no separate sphere boolean).
        tip_angs = (math.pi / 6.0, math.pi / 3.0, 0.46 * math.pi)
        tip_pts = np.array([end + (0.95 * r1 * math.sin(a)) * d_end for a in tip_angs])
        tip_sc = np.array([r1 * math.cos(a) for a in tip_angs])
        pts = np.vstack([pts_main, tip_pts])
        scales = np.concatenate([sc_main, tip_sc])
        tube = _SK.sweep_profile(unit_circle, pts, scales=scales, caps=True)
        parts.append(_to_manifold(tube))
        return end, d_end

    def recurse(p0, d0, length, r0, gen, twist_sign):
        r1 = max(0.8e-3, r0 * taper)
        end, d_end = add_curved_branch(p0, d0, length, r0, r1)
        if gen >= gens:
            return
        u, v_ax = _orthonormal_basis(d_end)
        # Rotate fork plane by ~65 deg each generation for natural 3D spread.
        ang_plane = 1.13 * (gen + 1) * twist_sign
        plane_dir = math.cos(ang_plane) * u + math.sin(ang_plane) * v_ax
        child_start = end - 0.35 * r1 * d_end
        for s in (-1.0, 1.0):
            cdir = math.cos(fork) * d_end + s * math.sin(fork) * plane_dir
            cdir = cdir + np.array([0.0, 0.0, 0.12])
            cdir /= max(float(np.linalg.norm(cdir)), 1e-12)
            recurse(child_start, cdir, length * 0.80, r1 * 0.94, gen + 1, twist_sign * s)

    tilt = math.radians(24.0)
    for k in range(arms):
        az = 2.0 * math.pi * k / arms
        start = np.array([0.34 * base_r * math.cos(az),
                          0.34 * base_r * math.sin(az),
                          0.55 * hold_h])
        d0 = np.array([math.sin(tilt) * math.cos(az),
                       math.sin(tilt) * math.sin(az),
                       math.cos(tilt)])
        recurse(start, d0, len0, br0, 0, 1.0 if k % 2 == 0 else -1.0)

    mesh = _batch_union(parts, kind="coral")
    zmin = float(mesh.verts[:, 2].min())
    if abs(zmin) > 1e-9:
        mesh = mesh.transform(translate=(0.0, 0.0, -zmin))
    mesh.meta.update({"kind": "coral", "arms": arms, "generations": gens})
    return mesh, finish(smooth_angle_deg=45.0, material="bio_bone")
