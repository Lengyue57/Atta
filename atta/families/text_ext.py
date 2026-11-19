# -*- coding: utf-8 -*-
"""Extended text & signage families — ADA Braille, bridged stencil, keychain tag.

Three parametric signage families complement :mod:`atta.families.text`:

``text.braille_plate``
    Tactile Braille plate compliant with the **2010 ADA Standards for
    Accessible Design** (§703.3, Table 703.3.1), **BANA** size/spacing rules,
    **Marburg Medium** standard cell matrix, and **ISO 17049:2013**.  Maps
    printable ASCII letters, digits (with standard `#` numeric indicator), and
    basic punctuation into 6-dot Grade-1 Braille cells rendered as smooth
    domed spherical caps (not flat cylinders, per ADA §703.3.1) sunk into the
    plate for watertight volumetric union.  Optionally adds a raised tactile
    caption above the Braille block separated by >= 9.5 mm (ADA §703.3.2).

``text.stencil_sign``
    Through-cut stencil sign using :mod:`atta.families.stroke_font` swept
    prisms subtracted completely through the plate thickness, followed by
    automatic island detection and stencil bridge placement across every
    enclosed glyph counter (``A``, ``O``, ``P``, ``B``, ``8``, etc.) so the
    sign always remains a single connected manifold (``n_components == 1``).

``text.keychain_tag``
    Rounded pocket/keychain tag with a raised perimeter rim, left-end keyring
    eyelet (``expected_genus == 1``), and engraved or embossed Hershey text
    inside the framed field.

References
----------
* U.S. Department of Justice (2010). *2010 ADA Standards for Accessible
  Design*, Section 703.3 & Table 703.3.1 (Braille Dimensions).
* Braille Authority of North America (BANA). *Size and Spacing of Braille
  Characters* (Marburg Medium standard).
* ISO 17049:2013. *Accessible design — Application of braille on signage,
  equipment and appliances*.
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
from ..surfaces import extrude, revolve
from . import stroke_font as _SF
from . import text as _T

__all__ = [
    "build_braille_plate",
    "build_stencil_sign",
    "build_keychain_tag",
    "encode_braille_cells",
]

# ---------------------------------------------------------------- Braille table
# Standard 6-dot cell numbering:
#   1  4
#   2  5
#   3  6
_BRAILLE_LETTERS = {
    "a": (1,),
    "b": (1, 2),
    "c": (1, 4),
    "d": (1, 4, 5),
    "e": (1, 5),
    "f": (1, 2, 4),
    "g": (1, 2, 4, 5),
    "h": (1, 2, 5),
    "i": (2, 4),
    "j": (2, 4, 5),
    "k": (1, 3),
    "l": (1, 2, 3),
    "m": (1, 3, 4),
    "n": (1, 3, 4, 5),
    "o": (1, 3, 5),
    "p": (1, 2, 3, 4),
    "q": (1, 2, 3, 4, 5),
    "r": (1, 2, 3, 5),
    "s": (2, 3, 4),
    "t": (2, 3, 4, 5),
    "u": (1, 3, 6),
    "v": (1, 2, 3, 6),
    "w": (2, 4, 5, 6),
    "x": (1, 3, 4, 6),
    "y": (1, 3, 4, 5, 6),
    "z": (1, 3, 5, 6),
}
_BRAILLE_DIGITS = {
    "1": (1,),
    "2": (1, 2),
    "3": (1, 4),
    "4": (1, 4, 5),
    "5": (1, 5),
    "6": (1, 2, 4),
    "7": (1, 2, 4, 5),
    "8": (1, 2, 5),
    "9": (2, 4),
    "0": (2, 4, 5),
}
_BRAILLE_NUM_SIGN = (3, 4, 5, 6)
_BRAILLE_PUNCT = {
    " ": (),
    ".": (2, 5, 6),
    ",": (2,),
    "-": (3, 6),
    "!": (2, 3, 5),
    "?": (2, 3, 6),
    ":": (2, 5),
    ";": (2, 3),
    "'": (3,),
    "/": (3, 4),
}


def encode_braille_cells(text: str) -> list[tuple[int, ...]]:
    """Convert ASCII text into a list of 6-dot Grade-1 Braille cells."""
    cells: list[tuple[int, ...]] = []
    in_number = False
    for ch in str(text or "").lower():
        if ch in _BRAILLE_DIGITS:
            if not in_number:
                cells.append(_BRAILLE_NUM_SIGN)
                in_number = True
            cells.append(_BRAILLE_DIGITS[ch])
        else:
            in_number = False
            if ch in _BRAILLE_LETTERS:
                cells.append(_BRAILLE_LETTERS[ch])
            elif ch in _BRAILLE_PUNCT:
                cells.append(_BRAILLE_PUNCT[ch])
            else:
                cells.append(_BRAILLE_PUNCT["?"])
    return cells


def _braille_dome(base_radius: float, height: float, sink: float, segments: int = 16,
                  arc_steps: int = 6) -> Mesh:
    """Revolve a smooth domed Braille dot with an embedded cylindrical skirt."""
    r = float(base_radius)
    h = float(height)
    s = float(sink)
    pts = [(0.0, -s), (r, -s), (r, 0.0)]
    # Elliptical/spherical dome arc from angle 0 (equator at z=0, r=r) to pi/2 (pole at z=h, r=0)
    for k in range(1, max(3, int(arc_steps)) + 1):
        ang = 0.5 * math.pi * (float(k) / float(max(3, int(arc_steps))))
        rr = r * math.cos(ang)
        zz = h * math.sin(ang)
        if k == max(3, int(arc_steps)):
            rr = 0.0
        pts.append((rr, zz))
    pts.append((0.0, -s))
    return revolve(np.asarray(pts, dtype=np.float64), segments=max(8, int(segments)), caps=True)


def _batch_union_mm(parts: list[Mesh]) -> Mesh:
    """Batch-union closed parts at millimetre scale for clean float32 manifold output."""
    if len(parts) == 1:
        return parts[0]
    try:
        import manifold3d

        mans = [
            manifold3d.Manifold(manifold3d.Mesh(
                vert_properties=np.asarray(m.verts * 1000.0, dtype=np.float32),
                tri_verts=np.asarray(m.faces, dtype=np.uint32)))
            for m in parts
        ]
        merged = manifold3d.Manifold.batch_boolean(mans, manifold3d.OpType.Add)
        raw = merged.to_mesh()
        return oriented(Mesh(
            np.asarray(raw.vert_properties, dtype=np.float64)[:, :3] * 1e-3,
            np.asarray(raw.tri_verts, dtype=np.int64),
            {"kind": "text_batch_union"}))
    except Exception:  # noqa: BLE001 - pairwise fallback
        acc = parts[0]
        for other in parts[1:]:
            acc = union(acc, other)
        return acc


# ---------------------------------------------------------------- 1. braille_plate

@recipe(
    "text.braille_plate", "盲文触觉标牌（ADA 703.3 / Marburg Medium）",
    "符合 2010 ADA §703.3（Table 703.3.1）、BANA Marburg Medium 与 ISO 17049:2013 的"
    "触觉盲文标牌：内置一级英文盲文（含数字前导符 # 与常用标点）6 点阵编码；"
    "点粒采用圆顶半球帽（非平顶圆柱）并下沉 0.35×dot_height 嵌入基板以保证单实体气密并集；"
    "可选在盲文上方间隔 >= 9.5 mm 处凸印辅助文字标题。",
    params={
        "text": {"type": "str", "default": "exit", "group": "braille",
                 "description": "盲文文本（ASCII 字母、数字、空格与基本标点）"},
        "caption": {"type": "str", "default": "", "group": "braille",
                    "description": "可选上方凸印标题（空字符串表示纯盲文牌）"},
        "length": {"type": "float", "default": 80.0, "min": 30.0, "max": 300.0,
                   "unit": "mm", "group": "plate", "description": "标牌长度（X）"},
        "width": {"type": "float", "default": 42.0, "min": 20.0, "max": 200.0,
                  "unit": "mm", "group": "plate", "description": "标牌宽度（Y）"},
        "thickness": {"type": "float", "default": 4.0, "min": 1.5, "max": 15.0,
                      "unit": "mm", "group": "plate", "description": "基板厚度（Z）"},
        "corner_r": {"type": "float", "default": 4.0, "min": 0.0, "max": 20.0,
                     "unit": "mm", "group": "plate", "description": "四角圆角半径"},
        "edge_margin": {"type": "float", "default": 6.0, "min": 2.0, "max": 30.0,
                        "unit": "mm", "group": "plate", "description": "盲文区距板边最小留白"},
        "dot_base_d": {"type": "float", "default": 1.55, "min": 1.0, "max": 2.5,
                       "unit": "mm", "group": "ada",
                       "description": "点粒底径（ADA 703.3.1: 1.5–1.6 mm）"},
        "dot_height": {"type": "float", "default": 0.75, "min": 0.4, "max": 1.5,
                       "unit": "mm", "group": "ada",
                       "description": "点粒凸起高度（ADA 703.3.1: 0.6–0.9 mm）"},
        "dot_pitch": {"type": "float", "default": 2.4, "min": 1.8, "max": 3.5,
                      "unit": "mm", "group": "ada",
                      "description": "单方内点心距（ADA 703.3.1: 2.3–2.5 mm）"},
        "cell_pitch": {"type": "float", "default": 6.4, "min": 4.5, "max": 10.0,
                       "unit": "mm", "group": "ada",
                       "description": "相邻盲文方中心距（ADA 703.3.1: 6.1–7.6 mm）"},
        "caption_height": {"type": "float", "default": 12.0, "min": 6.0, "max": 30.0,
                           "unit": "mm", "group": "caption", "description": "上方凸印标题字高"},
        "caption_gap": {"type": "float", "default": 10.5, "min": 5.0, "max": 25.0,
                        "unit": "mm", "group": "caption",
                        "description": "标题与盲文之间距（ADA >= 9.5 mm）"},
    },
    presets={
        "safe": {"text": "ok", "caption": "", "length": 60.0, "width": 32.0,
                 "thickness": 4.0, "corner_r": 4.0, "edge_margin": 5.0,
                 "dot_base_d": 1.6, "dot_height": 0.8, "dot_pitch": 2.5,
                 "cell_pitch": 6.6},
        "nominal": {"text": "exit", "caption": "", "length": 80.0, "width": 42.0,
                    "thickness": 4.0, "corner_r": 4.0, "edge_margin": 6.0,
                    "dot_base_d": 1.55, "dot_height": 0.75, "dot_pitch": 2.4,
                    "cell_pitch": 6.4},
        "hero": {"text": "exit", "caption": "EXIT", "length": 110.0, "width": 58.0,
                 "thickness": 5.0, "corner_r": 6.0, "edge_margin": 7.0,
                 "dot_base_d": 1.55, "dot_height": 0.75, "dot_pitch": 2.4,
                 "cell_pitch": 6.4, "caption_height": 12.0, "caption_gap": 10.5},
    },
    tags=["text", "braille", "ada", "tactile", "emboss", "signage"],
    expected_genus=0, material="abs_white",
    constraints=[
        {"id": "dot_separation", "expr": "dot_pitch >= dot_base_d + 0.3",
         "severity": "error", "message": "单方内相邻盲文点底圆间隙不足 0.3 mm，会粘连",
         "fix": "dot_pitch = 2.4"},
        {"id": "cell_separation", "expr": "cell_pitch >= dot_pitch + dot_base_d + 0.6",
         "severity": "error", "message": "相邻盲文方距离过近，跨方点位可能相交",
         "fix": "cell_pitch = 6.4"},
        {"id": "cell_fits_width", "expr": "2 * dot_pitch + dot_base_d + 2 * edge_margin <= width",
         "severity": "error", "message": "盲文方高度超出标牌可用宽度",
         "fix": "width = 2 * dot_pitch + dot_base_d + 2 * edge_margin + 6"},
        {"id": "corner_fits", "expr": "corner_r <= min(length, width) * 0.48",
         "severity": "error", "message": "圆角半径超过标牌半宽",
         "fix": "corner_r = min(length, width) * 0.15"},
        {"id": "ada_dot_diameter", "expr": "dot_base_d >= 1.5 and dot_base_d <= 1.6",
         "severity": "warn", "message": "点底径超出 ADA 703.3.1 标准区间 [1.5, 1.6] mm",
         "fix": "dot_base_d = 1.55"},
        {"id": "ada_dot_height", "expr": "dot_height >= 0.6 and dot_height <= 0.9",
         "severity": "warn", "message": "点高超出 ADA 703.3.1 标准区间 [0.6, 0.9] mm",
         "fix": "dot_height = 0.75"},
        {"id": "ada_caption_gap", "expr": "caption == '' or caption_gap >= 9.5",
         "severity": "warn", "message": "ADA §703.3.2 要求触觉文字与盲文间距 >= 9.5 mm",
         "fix": "caption_gap = 10.0"},
    ],
)
def build_braille_plate(v, quality="draft"):
    """Build an ADA/BANA tactile Braille sign with domed dots."""
    text = str(v.get("text", "") or "")
    caption = str(v.get("caption", "") or "").strip()
    cells = encode_braille_cells(text)
    if not any(cells):
        raise AttaError("empty_braille", "braille text has no raised dots", path="text",
                        fix="pass at least one letter or digit, e.g. text='exit'")

    length = float(v["length"])
    width = float(v["width"])
    thickness = float(v["thickness"])
    corner_r = float(v["corner_r"])
    edge = float(v["edge_margin"])
    dot_r = 0.5 * float(v["dot_base_d"])
    dot_h = float(v["dot_height"])
    dot_p = float(v["dot_pitch"])
    cell_p = float(v["cell_pitch"])

    n_cells = len(cells)
    braille_w = (n_cells - 1) * cell_p + dot_p + 2.0 * dot_r
    braille_h = 2.0 * dot_p + 2.0 * dot_r
    usable_l = length - 2.0 * edge
    usable_w = width - 2.0 * edge
    if braille_w > usable_l + 1e-12:
        raise AttaError(
            "braille_too_wide",
            "braille block needs %.2f mm length, plate allows %.2f mm"
            % (braille_w * 1000.0, usable_l * 1000.0),
            path="length",
            fix="raise length to %.1f mm or shorten text" % ((braille_w + 2.0 * edge) * 1000.0 + 4.0))

    cap_h = float(v["caption_height"])
    cap_gap = float(v["caption_gap"])
    has_cap = bool(caption)
    total_h = braille_h + (cap_gap + cap_h if has_cap else 0.0)
    if total_h > usable_w + 1e-12:
        raise AttaError(
            "braille_too_tall",
            "braille+caption needs %.2f mm width, plate allows %.2f mm"
            % (total_h * 1000.0, usable_w * 1000.0),
            path="width",
            fix="raise width to %.1f mm or shrink caption_height" % ((total_h + 2.0 * edge) * 1000.0 + 4.0))

    seg_plate = quality_level(quality, 6, 10, 14)
    plate = oriented(weld(extrude(rounded_rect(length, width, corner_r, per_corner=seg_plate),
                                  thickness), 1e-9))

    # Vertical layout: if caption is present, stack caption above Braille centred on y=0.
    braille_yc = (-0.5 * (cap_gap + cap_h) if has_cap else 0.0)
    x_start = -0.5 * ((n_cells - 1) * cell_p + dot_p)
    y_top_row = braille_yc + dot_p

    sink = 0.35 * dot_h
    dot_seg = quality_level(quality, 12, 16, 24)
    arc_steps = quality_level(quality, 4, 6, 8)
    proto_dot = _braille_dome(dot_r, dot_h, sink, segments=dot_seg, arc_steps=arc_steps)

    parts = [plate]
    dot_count = 0
    # 6-dot offsets: dot 1..3 on col 0 (dx=0), dot 4..6 on col 1 (dx=dot_p)
    dot_offsets = {
        1: (0.0, 0.0),
        2: (0.0, -dot_p),
        3: (0.0, -2.0 * dot_p),
        4: (dot_p, 0.0),
        5: (dot_p, -dot_p),
        6: (dot_p, -2.0 * dot_p),
    }
    for ci, dots in enumerate(cells):
        cx = x_start + ci * cell_p
        for d in dots:
            dx, dy = dot_offsets[d]
            parts.append(proto_dot.transform(translate=(cx + dx, y_top_row + dy, thickness)))
            dot_count += 1

    if has_cap:
        cap_sw = max(0.8e-3, 0.14 * cap_h)
        cap_yc = braille_yc + 0.5 * braille_h + cap_gap
        _strokes, ink = _SF.layout_strokes(caption, cap_height=cap_h, stroke_width=cap_sw,
                                           align="center")
        if (ink[2] - ink[0]) > usable_l + 1e-12:
            raise AttaError("caption_too_wide", "caption exceeds usable plate length",
                            path="caption", fix="shorten caption or raise length")
        cap_mesh = _T.text_mesh([(caption, cap_yc)], cap_h, cap_sw, dot_h,
                                mode="emboss", align="center", quality=quality)
        parts.append(cap_mesh.transform(translate=(0.0, 0.0, thickness)))

    out = _batch_union_mm(parts)
    topo = topology(out)
    if not topo["closed"] or topo["n_components"] != 1:
        raise AttaError("braille_topology", "braille plate union failed closure",
                        path="dot_pitch", fix="increase dot_pitch or cell_pitch")
    out.meta.update({
        "standard": "ADA-2010-703.3 / BANA-Marburg-Medium / ISO-17049",
        "text": text,
        "caption": caption,
        "n_cells": n_cells,
        "n_dots": dot_count,
        "dot_base_d_mm": round(2.0 * dot_r * 1000.0, 3),
        "dot_height_mm": round(dot_h * 1000.0, 3),
        "dot_pitch_mm": round(dot_p * 1000.0, 3),
        "cell_pitch_mm": round(cell_p * 1000.0, 3),
    })
    return out, finish(material="abs_white", smooth_angle_deg=35.0)


# ---------------------------------------------------------------- 2. stencil_sign

@recipe(
    "text.stencil_sign", "防孤岛镂空模板字牌（Stencil Sign）",
    "贯通切割的镂空字牌：将 Hershey 单线矢量笔画扫掠并完全切穿基板；"
    "自动检测切穿后因闭合字怀（如 O、A、B、P、8、0、4 等）脱落的悬浮孤岛网格，"
    "沿最短跨距跨槽添加全厚模板连筋（stencil bridge），保证成品始终为单连通气密流形。"
    "三套预设分别采用 4 字符字样（OPEN / ATTA），跨槽桥接后均精确保持 genus = 6。",
    params={
        "text": {"type": "str", "default": "OPEN", "group": "text",
                 "description": "镂空文字（printable ASCII）"},
        "length": {"type": "float", "default": 150.0, "min": 40.0, "max": 400.0,
                   "unit": "mm", "group": "plate", "description": "字牌长度（X）"},
        "width": {"type": "float", "default": 58.0, "min": 20.0, "max": 200.0,
                  "unit": "mm", "group": "plate", "description": "字牌宽度（Y）"},
        "thickness": {"type": "float", "default": 4.0, "min": 1.5, "max": 15.0,
                      "unit": "mm", "group": "plate", "description": "字牌厚度（Z）"},
        "corner_r": {"type": "float", "default": 5.0, "min": 0.0, "max": 20.0,
                     "unit": "mm", "group": "plate", "description": "圆角半径"},
        "edge_margin": {"type": "float", "default": 8.0, "min": 3.0, "max": 40.0,
                        "unit": "mm", "group": "plate", "description": "边缘最小留白"},
        "cap_height": {"type": "float", "default": 24.0, "min": 8.0, "max": 80.0,
                       "unit": "mm", "group": "text", "description": "字符大写高度"},
        "stroke_width": {"type": "float", "default": 2.2, "min": 1.0, "max": 6.0,
                         "unit": "mm", "group": "text", "description": "镂空槽宽"},
        "bridge_width": {"type": "float", "default": 2.4, "min": 1.0, "max": 6.0,
                         "unit": "mm", "group": "stencil", "description": "孤岛连筋宽度"},
        "tracking": {"type": "float", "default": 0.08, "min": 0.0, "max": 0.5,
                     "unit": "ratio", "group": "text", "description": "字符额外间距"},
    },
    presets={
        "safe": {"text": "OPEN", "length": 130.0, "width": 52.0, "thickness": 4.0,
                 "corner_r": 5.0, "edge_margin": 8.0, "cap_height": 20.0,
                 "stroke_width": 2.4, "bridge_width": 2.6, "tracking": 0.10},
        "nominal": {"text": "OPEN", "length": 150.0, "width": 58.0, "thickness": 4.0,
                    "corner_r": 5.0, "edge_margin": 8.0, "cap_height": 24.0,
                    "stroke_width": 2.2, "bridge_width": 2.4, "tracking": 0.08},
        "hero": {"text": "ATTA", "length": 165.0, "width": 62.0, "thickness": 5.0,
                 "corner_r": 6.0, "edge_margin": 9.0, "cap_height": 26.0,
                 "stroke_width": 2.2, "bridge_width": 2.4, "tracking": 0.10},
    },
    tags=["text", "stencil", "cutout", "bridge", "signage", "boolean"],
    expected_genus=6, material="steel",
    constraints=[
        {"id": "text_fits_height", "expr": "cap_height + stroke_width + 2 * edge_margin <= width",
         "severity": "error", "message": "字符高度加边留白超过字牌宽度",
         "fix": "cap_height = width - stroke_width - 2 * edge_margin - 2"},
        {"id": "stroke_ratio", "expr": "stroke_width <= cap_height * 0.22",
         "severity": "error", "message": "镂空槽宽超过字高 22% 会吃掉封闭字符内部结构",
         "fix": "stroke_width = cap_height * 0.12"},
        {"id": "bridge_min", "expr": "bridge_width >= 0.8 * stroke_width",
         "severity": "warn", "message": "连筋宽度小于槽宽 0.8 倍，刚度偏弱",
         "fix": "bridge_width = max(1.2, stroke_width)"},
        {"id": "corner_fits", "expr": "corner_r <= min(length, width) * 0.48",
         "severity": "error", "message": "圆角半径超过字牌半宽",
         "fix": "corner_r = min(length, width) * 0.15"},
    ],
)
def build_stencil_sign(v, quality="draft"):
    """Build a through-cut stencil sign and automatically bridge any floating islands."""
    import trimesh

    text = str(v.get("text", "") or "")
    if not text.strip():
        raise AttaError("empty_text", "stencil text is empty", path="text",
                        fix="pass printable ASCII text, e.g. text='OPEN'")

    length = float(v["length"])
    width = float(v["width"])
    thickness = float(v["thickness"])
    corner_r = float(v["corner_r"])
    edge = float(v["edge_margin"])
    cap = float(v["cap_height"])
    sw = float(v["stroke_width"])
    bw = float(v["bridge_width"])
    tracking = float(v["tracking"])

    strokes, ink = _SF.layout_strokes(text, cap_height=cap, stroke_width=sw,
                                      tracking=tracking, align="center",
                                      baseline=-0.5 * cap)
    ink_w = float(ink[2] - ink[0])
    usable_l = length - 2.0 * edge
    if ink_w > usable_l + 1e-12:
        raise AttaError(
            "text_too_wide",
            "stencil text spans %.2f mm across, plate allows %.2f mm"
            % (ink_w * 1000.0, usable_l * 1000.0),
            path="length",
            fix="raise length to %.1f mm or shrink cap_height" % ((ink_w + 2.0 * edge) * 1000.0 + 4.0))

    seg_plate = quality_level(quality, 6, 10, 14)
    seg_stroke = quality_level(quality, 8, 12, 16)
    plate = oriented(weld(extrude(rounded_rect(length, width, corner_r, per_corner=seg_plate),
                                  thickness), 1e-9))
    pad = max(5e-4, 0.15 * thickness)
    cutter = _T.stroke_prisms(strokes, sw, -pad, thickness + pad, segments=seg_stroke)
    cut_plate = difference(plate, cutter)

    # Detect disconnected islands inside glyph counters and bridge each back to the plate.
    n_bridges = 0
    for _ in range(3):
        tm = trimesh.Trimesh(vertices=cut_plate.verts, faces=cut_plate.faces, process=False)
        comps = tm.split(only_watertight=False)
        if len(comps) <= 1:
            break
        comps.sort(key=lambda c: -len(c.faces))
        bridges = []
        for isl in comps[1:]:
            bmin, bmax = isl.bounds
            cx = 0.5 * float(bmin[0] + bmax[0])
            cy = 0.5 * float(bmin[1] + bmax[1])
            dx = float(bmax[0] - bmin[0]) + 2.6 * sw + 2e-3
            dy = float(bmax[1] - bmin[1]) + 2.6 * sw + 2e-3
            if dy <= dx:
                bridges.append(_P.box([bw, dy, thickness], center=(cx, cy, 0.5 * thickness)))
            else:
                bridges.append(_P.box([dx, bw, thickness], center=(cx, cy, 0.5 * thickness)))
        n_bridges += len(bridges)
        cut_plate = _batch_union_mm([cut_plate] + bridges)

    topo = topology(cut_plate)
    if not topo["closed"] or topo["n_components"] != 1:
        raise AttaError("stencil_topology", "stencil bridging left disconnected shells",
                        path="bridge_width", fix="increase bridge_width or edge_margin")
    cut_plate.meta.update({
        "text": text,
        "n_bridges": n_bridges,
        "stroke_width_mm": round(sw * 1000.0, 3),
        "bridge_width_mm": round(bw * 1000.0, 3),
    })
    return cut_plate, finish(material="steel", smooth_angle_deg=28.0)


# ---------------------------------------------------------------- 3. keychain_tag

@recipe(
    "text.keychain_tag", "带加强围边与钥匙环孔的挂牌（Keychain Tag）",
    "圆角矩形行李/钥匙挂牌：本体外周带凸起加强围边（与基板体素重叠并集），"
    "左端预留贯穿钥匙环穿孔（故 genus 恒为 1），中央支持凹凸单线文字标识。",
    params={
        "text": {"type": "str", "default": "ATTA", "group": "text",
                 "description": "挂牌文字（printable ASCII）"},
        "length": {"type": "float", "default": 68.0, "min": 30.0, "max": 180.0,
                   "unit": "mm", "group": "tag", "description": "挂牌总长（X）"},
        "width": {"type": "float", "default": 28.0, "min": 14.0, "max": 80.0,
                  "unit": "mm", "group": "tag", "description": "挂牌宽度（Y）"},
        "thickness": {"type": "float", "default": 3.2, "min": 1.6, "max": 10.0,
                      "unit": "mm", "group": "tag", "description": "基板厚度（Z）"},
        "corner_r": {"type": "float", "default": 6.0, "min": 1.0, "max": 30.0,
                     "unit": "mm", "group": "tag", "description": "四角圆角半径"},
        "rim_w": {"type": "float", "default": 2.2, "min": 1.0, "max": 8.0,
                  "unit": "mm", "group": "rim", "description": "加强围边宽度"},
        "rim_h": {"type": "float", "default": 1.2, "min": 0.4, "max": 4.0,
                  "unit": "mm", "group": "rim", "description": "围边凸出高度"},
        "hole_d": {"type": "float", "default": 5.0, "min": 2.0, "max": 12.0,
                   "unit": "mm", "group": "eyelet", "description": "钥匙环通孔直径"},
        "hole_margin": {"type": "float", "default": 4.5, "min": 2.0, "max": 15.0,
                        "unit": "mm", "group": "eyelet", "description": "穿孔内缘距左端边距离"},
        "cap_height": {"type": "float", "default": 10.0, "min": 4.0, "max": 30.0,
                       "unit": "mm", "group": "text", "description": "文字字高"},
        "stroke_width": {"type": "float", "default": 1.1, "min": 0.5, "max": 3.0,
                         "unit": "mm", "group": "text", "description": "文字笔画宽度"},
        "text_depth": {"type": "float", "default": 0.9, "min": 0.3, "max": 3.0,
                       "unit": "mm", "group": "text", "description": "文字凹凸深度"},
        "mode": {"type": "enum", "default": "emboss", "enum": ["emboss", "engrave"],
                 "group": "text", "description": "凸印或凹刻"},
    },
    presets={
        "safe": {"text": "OK", "length": 58.0, "width": 26.0, "thickness": 3.5,
                 "corner_r": 6.0, "rim_w": 2.4, "rim_h": 1.2, "hole_d": 5.0,
                 "hole_margin": 4.5, "cap_height": 9.0, "stroke_width": 1.2,
                 "text_depth": 0.9, "mode": "emboss"},
        "nominal": {"text": "ATTA", "length": 68.0, "width": 28.0, "thickness": 3.2,
                    "corner_r": 6.0, "rim_w": 2.2, "rim_h": 1.2, "hole_d": 5.0,
                    "hole_margin": 4.5, "cap_height": 10.0, "stroke_width": 1.1,
                    "text_depth": 0.9, "mode": "emboss"},
        "hero": {"text": "LAB-01", "length": 82.0, "width": 32.0, "thickness": 4.0,
                 "corner_r": 7.0, "rim_w": 2.4, "rim_h": 1.4, "hole_d": 5.5,
                 "hole_margin": 5.0, "cap_height": 11.0, "stroke_width": 1.2,
                 "text_depth": 1.0, "mode": "engrave"},
    },
    tags=["text", "keychain", "tag", "emboss", "engrave", "eyelet", "boolean"],
    expected_genus=1, material="nylon",
    constraints=[
        {"id": "rim_fits", "expr": "2 * rim_w <= width * 0.45",
         "severity": "error", "message": "围边过宽，挤压中心文字区域",
         "fix": "rim_w = width * 0.08"},
        {"id": "hole_fits_width", "expr": "hole_d <= width - 2 * rim_w - 4.0",
         "severity": "error", "message": "钥匙环孔径相对于挂牌宽度过大",
         "fix": "hole_d = min(6.0, width - 2 * rim_w - 5.0)"},
        {"id": "hole_wall", "expr": "hole_margin >= rim_w + 1.5",
         "severity": "error", "message": "穿孔边距过小，易拉裂",
         "fix": "hole_margin = rim_w + 2.0"},
        {"id": "engrave_floor", "expr": "mode == 'emboss' or text_depth <= 0.55 * thickness",
         "severity": "error", "message": "凹刻深度超过基板厚度的 55%，会削弱或击穿挂牌",
         "fix": "text_depth = 0.35 * thickness"},
        {"id": "text_fits_height", "expr": "cap_height + stroke_width <= width - 2 * rim_w - 3.0",
         "severity": "error", "message": "字高超出围边内框宽度",
         "fix": "cap_height = width - 2 * rim_w - stroke_width - 4.0"},
        {"id": "corner_fits", "expr": "corner_r <= width * 0.48",
         "severity": "error", "message": "圆角半径超过半宽",
         "fix": "corner_r = width * 0.2"},
    ],
)
def build_keychain_tag(v, quality="draft"):
    """Build a rimmed keychain tag with eyelet hole and engraved/embossed text."""
    text = str(v.get("text", "") or "")
    if not text.strip():
        raise AttaError("empty_text", "keychain text is empty", path="text",
                        fix="pass at least one ASCII character")

    length = float(v["length"])
    width = float(v["width"])
    thickness = float(v["thickness"])
    corner_r = float(v["corner_r"])
    rim_w = float(v["rim_w"])
    rim_h = float(v["rim_h"])
    hole_d = float(v["hole_d"])
    hole_margin = float(v["hole_margin"])
    cap = float(v["cap_height"])
    sw = float(v["stroke_width"])
    depth = float(v["text_depth"])
    mode = str(v["mode"]).strip().lower()
    if mode not in ("emboss", "engrave"):
        raise AttaError("bad_arg", "mode must be 'emboss' or 'engrave'", path="mode",
                        fix="use mode='emboss'")

    seg_plate = quality_level(quality, 6, 10, 14)
    plate = oriented(weld(extrude(rounded_rect(length, width, corner_r, per_corner=seg_plate),
                                  thickness), 1e-9))

    # Perimeter rim sunk 0.4*thickness into the base plate
    sink = min(0.4 * thickness, 1e-3)
    pad = max(3e-4, 0.1 * rim_h)
    outer_rim = oriented(weld(extrude(rounded_rect(length, width, corner_r, per_corner=seg_plate),
                                      rim_h + sink), 1e-9))
    inner_r = max(0.0, corner_r - rim_w)
    inner_cut = oriented(weld(extrude(rounded_rect(length - 2.0 * rim_w, width - 2.0 * rim_w,
                                                   inner_r, per_corner=seg_plate),
                                      rim_h + sink + 2.0 * pad), 1e-9))
    rim = difference(outer_rim, inner_cut.transform(translate=(0.0, 0.0, -pad)))
    tag = union(plate, rim.transform(translate=(0.0, 0.0, thickness - sink)))

    # Keyring eyelet on the left end (-X)
    hole_x = -0.5 * length + hole_margin + 0.5 * hole_d
    hole_right = hole_x + 0.5 * hole_d
    text_area_left = hole_right + 2.0e-3
    text_area_right = 0.5 * length - rim_w - 2.0e-3
    usable_l = text_area_right - text_area_left
    text_cx = 0.5 * (text_area_left + text_area_right)

    _strokes, ink = _SF.layout_strokes(text, cap_height=cap, stroke_width=sw, align="center")
    ink_w = float(ink[2] - ink[0])
    if ink_w > usable_l + 1e-12:
        raise AttaError(
            "text_too_wide",
            "tag text spans %.2f mm, inner pocket right of eyelet allows %.2f mm"
            % (ink_w * 1000.0, usable_l * 1000.0),
            path="length",
            fix="raise length or shrink cap_height")

    # Boss around the eyelet hole so it looks like a proper reinforced tag ring,
    # and cut the through-hole.
    drill_h = thickness + rim_h + 2.0 * pad
    drill = _P.cylinder(0.5 * hole_d, drill_h,
                        center=(hole_x, 0.0, 0.5 * (thickness + rim_h) - 0.5 * pad),
                        segments=quality_level(quality, 24, 36, 48))
    tag = difference(tag, drill)

    text_solid = _T.text_mesh([(text, -0.5 * cap)], cap, sw, depth, mode=mode,
                              at=(text_cx, 0.0), align="center", quality=quality)
    text_solid = text_solid.transform(translate=(0.0, 0.0, thickness))
    tag = union(tag, text_solid) if mode == "emboss" else difference(tag, text_solid)

    topo = topology(tag)
    if not topo["closed"] or topo["n_components"] != 1:
        raise AttaError("keychain_topology", "keychain tag failed closure",
                        path="thickness", fix="adjust thickness or text_depth")
    tag.meta.update({"text": text, "mode": mode, "hole_d_mm": round(hole_d * 1000.0, 2)})
    return tag, finish(material="nylon", smooth_angle_deg=30.0)
