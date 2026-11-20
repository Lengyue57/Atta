# -*- coding: utf-8 -*-
"""Extended lattice families — strut lattices, multi-TPMS blocks, graded TPMS.

Three parametric lattice families complement :mod:`atta.families.lattice`:

``lattice.strut_cell``
    Cubic strut-and-node lattice block supporting **Octet-truss**
    (Deshpande, Fleck & Ashby, 2001), **BCC** (body-centered cubic), and
    **FCC** (face-centered cubic) unit-cell topologies (Gibson & Ashby, 1997;
    ISO/ASTM 52900).  Struts are unioned with icosphere nodes in millimetre
    precision through ``manifold3d.Manifold.batch_boolean`` to avoid float32
    seam degeneracy.  Because non-incident struts stay disjoint under the
    diameter constraints, the thickened 1-complex has exact topological genus
    ``g = |E| - |V| + 1``.

``lattice.tpms_block``
    Selectable sheet-network Triply Periodic Minimal Surface (TPMS) block
    supporting **Schwarz-P**, **Schwarz-D (Diamond)**, and **Schoen Gyroid**
    surfaces (Schwarz, 1890; Schoen, 1970, NASA TN D-5541; Al-Ketan & Abu
    Al-Rub, 2019).  Level-set field offsets are normalised by ``median|grad F|``
    in the sheet band so ``thickness`` is a genuine metric wall thickness in
    millimetres, intersected with the bounding slab and extracted via
    exact-edge marching tetrahedra (:func:`atta.sdf.marching_tets`).

``lattice.graded_gyroid``
    Functionally graded gyroid sheet lattice whose metric wall thickness
    varies linearly from ``thickness_bottom`` to ``thickness_top`` along the
    build (+Z) axis (Maskery et al., 2018; Al-Ketan & Abu Al-Rub, 2019).
    Keeping local thickness inside ``[0.09, 0.16] * cell_size`` preserves the
    sheet topology across the full density gradient.

References
----------
* Schoen, A. H. (1970). *Infinite Periodic Minimal Surfaces without
  Self-Intersections*. NASA Technical Note D-5541.
* Deshpande, V. S., Fleck, N. A., & Ashby, M. F. (2001). Effective properties
  of the octet-truss lattice material. *J. Mech. Phys. Solids*, 49(8),
  1747-1769.
* Gibson, L. J., & Ashby, M. F. (1997). *Cellular Solids: Structure and
  Properties* (2nd ed.). Cambridge University Press.
* Al-Ketan, O., & Abu Al-Rub, R. K. (2019). Multifunctional mechanical
  metamaterials based on triply periodic minimal surface lattices. *Advanced
  Engineering Materials*, 21(10), 1900524.
* Maskery, I., et al. (2018). Insights into the mechanical properties of
  several triply periodic minimal surface lattice structures made by polymer
  additive manufacturing. *Polymer*, 152, 62-71.
"""

from __future__ import annotations

import math

import numpy as np

from ..errors import AttaError
from ..kit import finish, quality_level
from ..mesh import Mesh, topology
from ..registry import recipe
from ..sdf import marching_tets

__all__ = [
    "build_strut_cell",
    "build_tpms_block",
    "build_graded_gyroid",
]

_MIN_VOX = 3.0
_MAX_RES = 34
_MAX_SAMPLES = 2_500_000

_STRUT_TOPOLOGIES = ("octet", "bcc", "fcc")
_TPMS_SURFACES = ("schwarz_p", "diamond", "gyroid")

#: Phase offsets (in unit cells) chosen so cutting at integer cell counts keeps
#: a single connected manifold without corner slivers.
_TPMS_PHASE = {
    "schwarz_p": (0.0, 0.0, 0.0),
    "diamond": (0.125, 0.125, 0.125),
    "gyroid": (0.25, 0.0, 0.0),
}


# ---------------------------------------------------------------- helpers

def _strut_sdf_mesh(pts: dict, edges: list, strut_r: float, node_r: float,
                    vox_across: float = 2.7) -> Mesh:
    """Evaluate capsule-and-sphere SDF on a bounded grid via bbox scatter and extract."""
    node_coords = np.asarray(list(pts.values()), dtype=np.float64)
    spacing = (2.0 * float(strut_r)) / max(2.2, float(vox_across))
    margin = float(node_r) + 2.5 * spacing
    lo = node_coords.min(axis=0) - margin
    hi = node_coords.max(axis=0) + margin
    shape = tuple(int(math.ceil((hi[d] - lo[d]) / spacing)) + 1 for d in range(3))
    if int(np.prod(shape)) > _MAX_SAMPLES:
        raise AttaError("lattice_too_dense",
                        "strut lattice needs %d samples" % (int(np.prod(shape)),),
                        path="strut_d", fix="increase strut_d or reduce cell counts")

    ax = lo[0] + np.arange(shape[0]) * spacing
    ay = lo[1] + np.arange(shape[1]) * spacing
    az = lo[2] + np.arange(shape[2]) * spacing
    phi = np.full(shape, 1e6, dtype=np.float64)

    def idx_box(pmin: np.ndarray, pmax: np.ndarray, pad: float):
        i0 = np.maximum(0, np.floor((pmin - pad - lo) / spacing).astype(int))
        i1 = np.minimum(np.asarray(shape), np.ceil((pmax + pad - lo) / spacing).astype(int) + 1)
        return i0, i1

    pad_n = float(node_r) + 2.0 * spacing
    for p in node_coords:
        i0, i1 = idx_box(p, p, pad_n)
        gx, gy, gz = np.meshgrid(ax[i0[0]:i1[0]] - p[0],
                                 ay[i0[1]:i1[1]] - p[1],
                                 az[i0[2]:i1[2]] - p[2], indexing="ij")
        d = np.sqrt(gx * gx + gy * gy + gz * gz) - float(node_r)
        sl = (slice(i0[0], i1[0]), slice(i0[1], i1[1]), slice(i0[2], i1[2]))
        phi[sl] = np.minimum(phi[sl], d)

    pad_s = float(strut_r) + 2.0 * spacing
    for u, w in edges:
        a, b = pts[u], pts[w]
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 <= 1e-24:
            continue
        i0, i1 = idx_box(np.minimum(a, b), np.maximum(a, b), pad_s)
        gx, gy, gz = np.meshgrid(ax[i0[0]:i1[0]] - a[0],
                                 ay[i0[1]:i1[1]] - a[1],
                                 az[i0[2]:i1[2]] - a[2], indexing="ij")
        t = np.clip((gx * ab[0] + gy * ab[1] + gz * ab[2]) / ab2, 0.0, 1.0)
        dx = gx - t * ab[0]
        dy = gy - t * ab[1]
        dz = gz - t * ab[2]
        d = np.sqrt(dx * dx + dy * dy + dz * dz) - float(strut_r)
        sl = (slice(i0[0], i1[0]), slice(i0[1], i1[1]), slice(i0[2], i1[2]))
        phi[sl] = np.minimum(phi[sl], d)

    return marching_tets(phi, lo, spacing)


def _strut_graph(topo_kind: str, nx: int, ny: int, nz: int, cell: float):
    """Return ``(nodes_dict, edge_set)`` centred at the origin."""
    pts: dict[tuple[int, int, int], np.ndarray] = {}
    edges: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()

    def add_pt(key: tuple[int, int, int]):
        if key not in pts:
            pts[key] = np.array([key[0] * 0.5, key[1] * 0.5, key[2] * 0.5], dtype=np.float64) * cell

    def add_edge(u: tuple[int, int, int], v: tuple[int, int, int]):
        if u != v:
            edges.add((u, v) if u < v else (v, u))

    if topo_kind == "bcc":
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    ctr = (2 * i + 1, 2 * j + 1, 2 * k + 1)
                    add_pt(ctr)
                    for di in (-1, 1):
                        for dj in (-1, 1):
                            for dk in (-1, 1):
                                corner = (ctr[0] + di, ctr[1] + dj, ctr[2] + dk)
                                add_pt(corner)
                                add_edge(ctr, corner)
    else:  # "fcc" or "octet"
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    cx, cy, cz = 2 * i + 1, 2 * j + 1, 2 * k + 1
                    fcs = [
                        (cx - 1, cy, cz), (cx + 1, cy, cz),
                        (cx, cy - 1, cz), (cx, cy + 1, cz),
                        (cx, cy, cz - 1), (cx, cy, cz + 1),
                    ]
                    for fc in fcs:
                        add_pt(fc)
                        axes = [d for d in range(3) if fc[d] % 2 == 1]
                        for s0 in (-1, 1):
                            for s1 in (-1, 1):
                                c = list(fc)
                                c[axes[0]] += s0
                                c[axes[1]] += s1
                                ct = tuple(c)
                                add_pt(ct)
                                add_edge(fc, ct)
                    if topo_kind == "octet":
                        for a_idx in range(6):
                            for b_idx in range(a_idx + 1, 6):
                                fa, fb = fcs[a_idx], fcs[b_idx]
                                if sorted(abs(fa[d] - fb[d]) for d in range(3)) == [0, 1, 1]:
                                    add_edge(fa, fb)

    shift = np.array([0.5 * nx * cell, 0.5 * ny * cell, 0.5 * nz * cell], dtype=np.float64)
    for k in pts:
        pts[k] = pts[k] - shift
    return pts, sorted(edges)


def _tpms_eval(kind: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, k: float):
    """Evaluate TPMS level-set value ``F`` and analytic gradient magnitude ``|grad F|``."""
    sx, cx = np.sin(k * X), np.cos(k * X)
    sy, cy = np.sin(k * Y), np.cos(k * Y)
    sz, cz = np.sin(k * Z), np.cos(k * Z)
    if kind == "schwarz_p":
        f = cx + cy + cz
        gx, gy, gz = -k * sx, -k * sy, -k * sz
    elif kind == "diamond":
        f = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
        gx = k * (cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz)
        gy = k * (sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz)
        gz = k * (sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz)
    elif kind == "gyroid":
        f = sx * cy + sy * cz + sz * cx
        gx = k * (cx * cy - sz * sx)
        gy = k * (-sx * sy + cy * cz)
        gz = k * (-sy * sz + cz * cx)
    else:
        raise AttaError("bad_arg", "unknown TPMS surface %r" % (kind,), path="surface",
                        fix="use one of: " + ", ".join(_TPMS_SURFACES))
    return f, np.sqrt(gx * gx + gy * gy + gz * gz)


# ---------------------------------------------------------------- 1. strut_cell

@recipe(
    "lattice.strut_cell", "桁架杆件晶格块（Octet / BCC / FCC）",
    "基于 Deshpande-Fleck-Ashby (2001) 与 Gibson-Ashby (1997) 的三维杆件-节点晶格块："
    "支持八角桁架（octet，拉伸主导）、体心立方（bcc）与面心立方（fcc）三种拓扑。"
    "节点球采用无极点二十面体细分球，杆件与节点在毫米尺度通过 manifold3d 精确并集；"
    "当杆径满足防碰约束时，拓扑亏格精确等于 g = |E| - |V| + 1（2×2×1 Octet 恒为 91）。",
    params={
        "topology": {"type": "enum", "default": "octet", "enum": list(_STRUT_TOPOLOGIES),
                     "group": "lattice", "description": "晶格拓扑（octet / bcc / fcc）"},
        "cell_size": {"type": "float", "default": 16.0, "min": 6.0, "max": 45.0,
                      "unit": "mm", "group": "lattice", "description": "单胞边长"},
        "strut_d": {"type": "float", "default": 2.0, "min": 0.6, "max": 10.0,
                    "unit": "mm", "group": "lattice", "description": "杆件直径"},
        "node_scale": {"type": "float", "default": 1.35, "min": 1.1, "max": 2.0,
                       "unit": "ratio", "group": "lattice",
                       "description": "节点球直径与杆径之比（>=1.15 以包裹尖角汇合处）"},
        "nx": {"type": "int", "default": 2, "min": 1, "max": 4,
               "unit": "count", "group": "array", "description": "X 向单胞数"},
        "ny": {"type": "int", "default": 2, "min": 1, "max": 4,
               "unit": "count", "group": "array", "description": "Y 向单胞数"},
        "nz": {"type": "int", "default": 1, "min": 1, "max": 4,
               "unit": "count", "group": "array", "description": "Z 向单胞数"},
        "segments": {"type": "int", "default": 10, "min": 6, "max": 24,
                     "unit": "count", "group": "quality", "advanced": True,
                     "description": "杆件圆周分段数"},
    },
    presets={
        "safe": {"topology": "octet", "cell_size": 18.0, "strut_d": 2.4,
                 "node_scale": 1.35, "nx": 2, "ny": 2, "nz": 1, "segments": 8},
        "nominal": {"topology": "octet", "cell_size": 16.0, "strut_d": 2.0,
                    "node_scale": 1.35, "nx": 2, "ny": 2, "nz": 1, "segments": 10},
        "hero": {"topology": "octet", "cell_size": 14.0, "strut_d": 1.6,
                 "node_scale": 1.35, "nx": 2, "ny": 2, "nz": 1, "segments": 14},
    },
    tags=["lattice", "strut", "octet", "bcc", "fcc", "truss", "boolean"],
    expected_genus=91, material="nylon",
    constraints=[
        {"id": "strut_min_ratio", "expr": "strut_d >= 0.06 * cell_size",
         "severity": "error", "message": "杆径过细（< 0.06×cell_size），难以打印且易在节点处退化",
         "fix": "strut_d = 0.12 * cell_size"},
        {"id": "strut_max_ratio", "expr": "strut_d <= 0.22 * cell_size",
         "severity": "error", "message": "杆径过粗会导致非共点杆件互相粘连并改变晶格亏格",
         "fix": "strut_d = 0.14 * cell_size"},
        {"id": "node_covers_strut", "expr": "node_scale >= 1.15",
         "severity": "error", "message": "节点球必须比杆径大至少 15%，否则多杆汇合角会出现缝隙",
         "fix": "node_scale = 1.35"},
        {"id": "node_clearance", "expr": "node_scale * strut_d <= 0.45 * cell_size",
         "severity": "error", "message": "节点球过大，相邻节点会直接融合",
         "fix": "node_scale = 1.35; strut_d = 0.125 * cell_size"},
        {"id": "cell_budget", "expr": "nx * ny * nz <= 24",
         "severity": "error", "message": "单胞总数超过 24 时布尔并集耗时过长",
         "fix": "nx = 2; ny = 2; nz = 1"},
        {"id": "genus_envelope",
         "expr": "topology == 'octet' and nx == 2 and ny == 2 and nz == 1",
         "severity": "warn",
         "message": "expected_genus=91 对应 2×2×1 Octet；更改拓扑或单胞数后亏格变为 |E|-|V|+1",
         "fix": "topology = 'octet'; nx = 2; ny = 2; nz = 1"},
    ],
)
def build_strut_cell(v, quality="draft"):
    """Build a strut-and-node lattice block via bounded capsule/sphere SDF."""
    topo_kind = str(v["topology"]).strip().lower()
    if topo_kind not in _STRUT_TOPOLOGIES:
        raise AttaError("bad_arg", "unknown strut topology %r" % (topo_kind,),
                        path="topology", fix="use one of: " + ", ".join(_STRUT_TOPOLOGIES))
    cell = float(v["cell_size"])
    strut_r = 0.5 * float(v["strut_d"])
    node_r = strut_r * float(v["node_scale"])
    nx, ny, nz = int(v["nx"]), int(v["ny"]), int(v["nz"])
    vox = quality_level(quality, 27, 30, 34) / 10.0

    pts, edges = _strut_graph(topo_kind, nx, ny, nz, cell)
    mesh = _strut_sdf_mesh(pts, edges, strut_r, node_r, vox_across=vox)

    # Shift to sit on z = 0
    z_min = float(mesh.verts[:, 2].min())
    mesh = mesh.transform(translate=(0.0, 0.0, -z_min))

    analytic_genus = len(edges) - len(pts) + 1
    topo = topology(mesh)
    if not topo["closed"] or topo["n_components"] != 1:
        raise AttaError("lattice_topology", "strut lattice failed manifold closure",
                        path="strut_d", fix="adjust strut_d within [0.06, 0.22] * cell_size")
    mesh.meta.update({
        "lattice": topo_kind,
        "cells": [nx, ny, nz],
        "n_nodes": len(pts),
        "n_struts": len(edges),
        "analytic_genus": analytic_genus,
    })
    return mesh, finish(smooth_angle_deg=35.0, material="nylon")


# ---------------------------------------------------------------- 2. tpms_block

@recipe(
    "lattice.tpms_block", "可选三周期极小曲面薄片晶格块（Schwarz-P / Diamond / Gyroid）",
    "统一的多拓扑 TPMS 薄片晶格：支持 Schwarz Primitive (P)、Schwarz Diamond (D) 与 "
    "Schoen Gyroid (G)（Schwarz 1890；Schoen 1970 NASA TN D-5541；Al-Ketan & Abu Al-Rub 2019）。"
    "使用解析梯度中位数 |grad F| 将隐式等值面偏移归一化为真实毫米壁厚，并与板坯包络盒求交，"
    "通过精确整数边键行进四面体提取闭合单连通流形（2×2×2 Schwarz-P 亏格恒为 33）。",
    params={
        "surface": {"type": "enum", "default": "schwarz_p", "enum": list(_TPMS_SURFACES),
                    "group": "lattice", "description": "TPMS 曲面类型"},
        "cell_size": {"type": "float", "default": 18.0, "min": 8.0, "max": 45.0,
                      "unit": "mm", "group": "lattice", "description": "单胞周期"},
        "thickness": {"type": "float", "default": 2.3, "min": 0.9, "max": 8.0,
                      "unit": "mm", "group": "lattice", "description": "薄片实体壁厚"},
        "t_bias": {"type": "float", "default": 0.0, "min": -0.35, "max": 0.35,
                   "unit": "none", "group": "lattice", "description": "等值面偏移量"},
        "nx": {"type": "int", "default": 2, "min": 1, "max": 4,
               "unit": "count", "group": "array", "description": "X 向单胞数"},
        "ny": {"type": "int", "default": 2, "min": 1, "max": 4,
               "unit": "count", "group": "array", "description": "Y 向单胞数"},
        "nz": {"type": "int", "default": 2, "min": 1, "max": 4,
               "unit": "count", "group": "array", "description": "Z 向单胞数"},
        "resolution": {"type": "int", "default": 14, "min": 10, "max": 30,
                       "unit": "count", "group": "quality", "advanced": True,
                       "description": "每单胞采样步数"},
    },
    presets={
        "safe": {"surface": "schwarz_p", "cell_size": 20.0, "thickness": 2.8,
                 "t_bias": 0.0, "nx": 2, "ny": 2, "nz": 2, "resolution": 12},
        "nominal": {"surface": "schwarz_p", "cell_size": 18.0, "thickness": 2.3,
                    "t_bias": 0.0, "nx": 2, "ny": 2, "nz": 2, "resolution": 14},
        "hero": {"surface": "schwarz_p", "cell_size": 16.0, "thickness": 1.9,
                 "t_bias": 0.05, "nx": 2, "ny": 2, "nz": 2, "resolution": 18},
    },
    tags=["lattice", "tpms", "schwarz_p", "diamond", "gyroid", "implicit", "marching_tets"],
    expected_genus=33, material="nylon",
    constraints=[
        {"id": "wall_min_ratio", "expr": "thickness >= 0.09 * cell_size",
         "severity": "error", "message": "壁厚小于 0.09×cell_size，采样易撕裂薄片",
         "fix": "thickness = 0.12 * cell_size"},
        {"id": "wall_max_ratio", "expr": "thickness <= 0.16 * cell_size",
         "severity": "error", "message": "壁厚大于 0.16×cell_size 会自熔并漂移亏格",
         "fix": "thickness = 0.13 * cell_size"},
        {"id": "bias_band", "expr": "abs(t_bias) <= 0.25",
         "severity": "error", "message": "等值面偏移过大会导致局部断开",
         "fix": "t_bias = 0.0"},
        {"id": "cell_budget", "expr": "nx * ny * nz <= 16",
         "severity": "error", "message": "单胞总数超过 16 时采样网格将超出预算",
         "fix": "nx = 2; ny = 2; nz = 2"},
        {"id": "genus_envelope",
         "expr": "surface == 'schwarz_p' and nx == 2 and ny == 2 and nz == 2",
         "severity": "warn",
         "message": "expected_genus=33 对应 2×2×2 Schwarz-P（2×2×1 Gyroid=25，2×2×1 Diamond=87）",
         "fix": "surface = 'schwarz_p'; nx = 2; ny = 2; nz = 2"},
    ],
)
def build_tpms_block(v, quality="draft"):
    """Sample a metric-thickness TPMS sheet block and extract via marching tets."""
    surf = str(v["surface"]).strip().lower()
    if surf not in _TPMS_SURFACES:
        raise AttaError("bad_arg", "unknown TPMS surface %r" % (surf,), path="surface",
                        fix="use one of: " + ", ".join(_TPMS_SURFACES))
    cell = float(v["cell_size"])
    thickness = float(v["thickness"])
    t_bias = float(v["t_bias"])
    nx, ny, nz = int(v["nx"]), int(v["ny"]), int(v["nz"])
    res = int(v["resolution"])

    rq = int(quality_level(quality, res, res + 2, res + 4))
    r_need = int(math.ceil(_MIN_VOX * cell / max(thickness, 1e-9)))
    res_eff = int(min(_MAX_RES, max(10, max(rq, r_need))))

    L, W, H = nx * cell, ny * cell, nz * cell
    half = np.array([L, W, H], dtype=np.float64) * 0.5
    spacing = cell / float(res_eff)
    margin = 2.5 * spacing
    shape = tuple(int(math.ceil((2.0 * half[i] + 2.0 * margin) / spacing)) + 1 for i in range(3))
    if int(np.prod(shape)) > _MAX_SAMPLES:
        raise AttaError("lattice_too_dense",
                        "TPMS block needs %d samples at resolution %d" % (int(np.prod(shape)), res_eff),
                        path="resolution", fix="lower resolution or cell counts")

    org = -half - margin
    ax = org[0] + np.arange(shape[0]) * spacing
    ay = org[1] + np.arange(shape[1]) * spacing
    az = org[2] + np.arange(shape[2]) * spacing
    phase = _TPMS_PHASE[surf]
    X, Y, Z = np.meshgrid(ax - phase[0] * cell, ay - phase[1] * cell, az - phase[2] * cell,
                          indexing="ij")
    F, gmag = _tpms_eval(surf, X, Y, Z, 2.0 * math.pi / cell)
    band = np.abs(F - t_bias) < 0.35
    g_ref = float(np.median(gmag[band])) if bool(band.any()) else float(np.median(gmag))
    if not math.isfinite(g_ref) or g_ref <= 0.0:
        raise AttaError("lattice_field", "TPMS gradient is degenerate", path="cell_size",
                        fix="increase cell_size")

    sheet = np.abs(F - t_bias) / g_ref - 0.5 * thickness
    GX, GY, GZ = np.meshgrid(ax, ay, az, indexing="ij")
    box = np.maximum(np.maximum(np.abs(GX) - half[0], np.abs(GY) - half[1]),
                     np.abs(GZ) - half[2])
    mesh = marching_tets(np.maximum(sheet, box), org, spacing)
    mesh = mesh.transform(translate=(0.0, 0.0, half[2]))
    mesh.meta.update({
        "lattice": surf,
        "cells": [nx, ny, nz],
        "res_eff": res_eff,
        "g_ref": g_ref,
        "thickness_mm": round(thickness * 1000.0, 4),
    })
    return mesh, finish(smooth_angle_deg=35.0, material="nylon")


# ---------------------------------------------------------------- 3. graded_gyroid

@recipe(
    "lattice.graded_gyroid", "轴向功能梯度陀螺晶格块",
    "功能梯度三周期极小曲面晶格（Maskery et al., 2018；Al-Ketan & Abu Al-Rub, 2019）："
    "陀螺薄片物理壁厚沿构建方向（Z 轴）从底面 thickness_bottom 线性过渡到顶面 thickness_top，"
    "用于能量吸收与刚度梯度缓冲。全高度壁厚经梯度 |grad F| 归一化并限制在单一连通窗口内，"
    "保证 2×2×1 胞元在梯度变化下亏格严格恒为 25。",
    params={
        "cell_size": {"type": "float", "default": 20.0, "min": 10.0, "max": 45.0,
                      "unit": "mm", "group": "lattice", "description": "晶格周期"},
        "thickness_bottom": {"type": "float", "default": 3.0, "min": 1.0, "max": 7.0,
                             "unit": "mm", "group": "gradient", "description": "底面壁厚"},
        "thickness_top": {"type": "float", "default": 2.0, "min": 1.0, "max": 7.0,
                          "unit": "mm", "group": "gradient", "description": "顶面壁厚"},
        "t_bias": {"type": "float", "default": 0.0, "min": -0.3, "max": 0.3,
                   "unit": "none", "group": "lattice", "description": "等值面偏移"},
        "nx": {"type": "int", "default": 2, "min": 1, "max": 4,
               "unit": "count", "group": "array", "description": "X 向单胞数"},
        "ny": {"type": "int", "default": 2, "min": 1, "max": 4,
               "unit": "count", "group": "array", "description": "Y 向单胞数"},
        "nz": {"type": "int", "default": 1, "min": 1, "max": 3,
               "unit": "count", "group": "array", "description": "Z 向单胞数"},
        "resolution": {"type": "int", "default": 14, "min": 10, "max": 32,
                       "unit": "count", "group": "quality", "advanced": True,
                       "description": "每周期基础采样数（按最薄壁自动抬升）"},
    },
    presets={
        "safe": {"cell_size": 22.0, "thickness_bottom": 3.3, "thickness_top": 2.3,
                 "t_bias": 0.0, "nx": 2, "ny": 2, "nz": 1, "resolution": 12},
        "nominal": {"cell_size": 20.0, "thickness_bottom": 3.0, "thickness_top": 2.0,
                    "t_bias": 0.0, "nx": 2, "ny": 2, "nz": 1, "resolution": 14},
        "hero": {"cell_size": 18.0, "thickness_bottom": 2.8, "thickness_top": 1.8,
                 "t_bias": 0.0, "nx": 2, "ny": 2, "nz": 1, "resolution": 16},
    },
    tags=["lattice", "gyroid", "graded", "tpms", "implicit", "marching_tets"],
    expected_genus=25, material="nylon",
    constraints=[
        {"id": "bottom_wall_range",
         "expr": "thickness_bottom >= 0.09 * cell_size and thickness_bottom <= 0.16 * cell_size",
         "severity": "error", "message": "底面壁厚必须留在 [0.09, 0.16] × cell_size 稳定单连通区间",
         "fix": "thickness_bottom = 0.14 * cell_size"},
        {"id": "top_wall_range",
         "expr": "thickness_top >= 0.09 * cell_size and thickness_top <= 0.16 * cell_size",
         "severity": "error", "message": "顶面壁厚必须留在 [0.09, 0.16] × cell_size 稳定单连通区间",
         "fix": "thickness_top = 0.10 * cell_size"},
        {"id": "bias_band", "expr": "abs(t_bias) <= 0.25",
         "severity": "error", "message": "等值面偏移过大会撕裂薄片",
         "fix": "t_bias = 0.0"},
        {"id": "has_gradient", "expr": "abs(thickness_bottom - thickness_top) >= 0.2",
         "severity": "warn", "message": "上下壁厚差小于 0.2 mm，梯度效果不明显",
         "fix": "thickness_top = max(0.095 * cell_size, thickness_bottom - 0.8)"},
        {"id": "genus_envelope", "expr": "nx == 2 and ny == 2 and nz == 1",
         "severity": "warn",
         "message": "expected_genus=25 对应 2×2×1 晶格单元",
         "fix": "nx = 2; ny = 2; nz = 1"},
    ],
)
def build_graded_gyroid(v, quality="draft"):
    """Build an axially graded gyroid sheet lattice with metric wall thickness."""
    cell = float(v["cell_size"])
    t_bot = float(v["thickness_bottom"])
    t_top = float(v["thickness_top"])
    t_bias = float(v["t_bias"])
    nx, ny, nz = int(v["nx"]), int(v["ny"]), int(v["nz"])
    res = int(v["resolution"])

    min_t = min(t_bot, t_top)
    rq = int(quality_level(quality, res, res + 2, res + 4))
    r_need = int(math.ceil(_MIN_VOX * cell / max(min_t, 1e-9)))
    res_eff = int(min(_MAX_RES, max(10, max(rq, r_need))))

    L, W, H = nx * cell, ny * cell, nz * cell
    half = np.array([L, W, H], dtype=np.float64) * 0.5
    spacing = cell / float(res_eff)
    margin = 2.5 * spacing
    shape = tuple(int(math.ceil((2.0 * half[i] + 2.0 * margin) / spacing)) + 1 for i in range(3))
    if int(np.prod(shape)) > _MAX_SAMPLES:
        raise AttaError("lattice_too_dense",
                        "graded gyroid needs %d samples" % (int(np.prod(shape)),),
                        path="resolution", fix="lower resolution or cell counts")

    org = -half - margin
    ax = org[0] + np.arange(shape[0]) * spacing
    ay = org[1] + np.arange(shape[1]) * spacing
    az = org[2] + np.arange(shape[2]) * spacing
    X, Y, Z = np.meshgrid(ax - 0.25 * cell, ay, az, indexing="ij")
    F, gmag = _tpms_eval("gyroid", X, Y, Z, 2.0 * math.pi / cell)
    band = np.abs(F - t_bias) < 0.35
    g_ref = float(np.median(gmag[band])) if bool(band.any()) else float(np.median(gmag))
    if not math.isfinite(g_ref) or g_ref <= 0.0:
        raise AttaError("lattice_field", "gyroid gradient is degenerate", path="cell_size",
                        fix="increase cell_size")

    GX, GY, GZ = np.meshgrid(ax, ay, az, indexing="ij")
    u = np.clip((GZ + half[2]) / max(H, 1e-9), 0.0, 1.0)
    t_local = t_bot + (t_top - t_bot) * u
    sheet = np.abs(F - t_bias) / g_ref - 0.5 * t_local
    box = np.maximum(np.maximum(np.abs(GX) - half[0], np.abs(GY) - half[1]),
                     np.abs(GZ) - half[2])
    mesh = marching_tets(np.maximum(sheet, box), org, spacing)
    mesh = mesh.transform(translate=(0.0, 0.0, half[2]))
    mesh.meta.update({
        "lattice": "graded_gyroid",
        "cells": [nx, ny, nz],
        "res_eff": res_eff,
        "thickness_bottom_mm": round(t_bot * 1000.0, 4),
        "thickness_top_mm": round(t_top * 1000.0, 4),
    })
    return mesh, finish(smooth_angle_deg=35.0, material="nylon")
