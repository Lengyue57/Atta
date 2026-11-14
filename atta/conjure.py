# -*- coding: utf-8 -*-
"""Conjure: invent plausible geometry from a label, without looking anything up.

The problem this solves
-----------------------
An agent told "model a rifle magazine" or "model a planet" used to have two bad
options: search the web for real dimensions (impossible for a fictional
subject, wasteful for a real one) or guess numbers with no structure (which
produces parts that are locally fine and globally wrong).

``conjure`` is the third option.  It carries a small **archetype table** — the
proportions that make a class of object read as that class — and derives a full
parameter set from it deterministically.  Every number comes back tagged with
its provenance:

``measured``
    A dimension the caller supplied.
``archetype``
    A proportion from the built-in table for this object class.
``invented``
    A value derived from the seed and the archetype's variation range.
``derived``
    A transformed or constructed value; scale and bounds never turn a guess
    into a measurement. ``provenance_detail`` preserves its source and operations.

That tagging is the honest part: an agent can tell the user which numbers are
real and which were invented to look right, and ``verify`` can still gate the
geometry on its own merits.

Design rules encoded here
-------------------------
* **Human scale is the anchor.**  Anything a person holds or operates is sized
  against a 1.7 m human; anything architectural against a 2 m door.  This is
  what makes an invented object feel real.
* **Proportions, not dimensions.**  The table stores ratios (height / width,
  wall / height), so one archetype scales to any size.
* **Three sizes.**  Repeated features (buttons, greebles, ribs) come in small,
  medium and large, which is the "three sizes" rule that keeps repetition from
  looking mechanical.
* **Sub-part decomposition.**  An archetype names its parts and each part's
  share of the whole, so the result is an assembly, not one blob.

Everything is deterministic: the same ``(label, seed, style)`` gives the same
numbers.
"""

from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

from .errors import AttaError

__all__ = [
    "conjure",
    "archetypes",
    "archetype",
    "STYLES",
    "METHOD",
]

METHOD = "atta.conjure/v1"

#: Style vectors.  Each modulates proportion and detail, never topology.
#: ``bulk`` scales cross-section, ``taper`` the end-to-end narrowing,
#: ``detail`` the count of secondary features, ``sharp`` the edge treatment.
STYLES = {
    "neutral":   {"bulk": 1.00, "taper": 1.00, "detail": 1.00, "sharp": 0.50},
    "brutal":    {"bulk": 1.18, "taper": 0.85, "detail": 0.75, "sharp": 0.85},
    "elegant":   {"bulk": 0.90, "taper": 1.15, "detail": 1.20, "sharp": 0.20},
    "organic":   {"bulk": 1.05, "taper": 1.25, "detail": 1.35, "sharp": 0.10},
    "industrial": {"bulk": 1.08, "taper": 0.95, "detail": 1.10, "sharp": 0.70},
    "scifi":     {"bulk": 0.95, "taper": 1.05, "detail": 1.45, "sharp": 0.60},
    "fantasy":   {"bulk": 1.10, "taper": 1.30, "detail": 1.25, "sharp": 0.35},
    "ancient":   {"bulk": 1.12, "taper": 1.10, "detail": 0.85, "sharp": 0.30},
}

#: Archetype table.  ``size_mm`` is the nominal bounding box for a
#: human-scale instance; ``ratios`` are the proportions that make the class
#: read correctly; ``parts`` decomposes the whole and each entry's share sums
#: to ~1.0; ``variation`` is the per-axis jitter the seed may apply.
ARCHETYPES = {
    # ---- hand-held objects -------------------------------------------------
    "mug": {
        "category": "product", "size_mm": [117, 83, 95],
        "ratios": {"wall": 0.034, "handle_span": 0.55, "handle_thickness": 0.10},
        "parts": {"body": 0.82, "handle": 0.18},
        "variation": 0.12, "tags": ["holdable", "hollow"],
    },
    "bottle": {
        "category": "product", "size_mm": [70, 70, 240],
        "ratios": {"wall": 0.018, "neck": 0.28, "shoulder": 0.22},
        "parts": {"body": 0.62, "shoulder": 0.14, "neck": 0.14, "cap": 0.10},
        "variation": 0.15, "tags": ["holdable", "hollow", "sealed"],
    },
    "knife": {
        "category": "tool", "size_mm": [280, 32, 18],
        "ratios": {"blade": 0.58, "handle": 0.42, "blade_thickness": 0.09,
                   "edge_angle_deg": 18.0},
        "parts": {"blade": 0.58, "handle": 0.42},
        "variation": 0.10, "tags": ["holdable", "edge"],
    },
    "sword": {
        "category": "weapon", "size_mm": [1050, 210, 40],
        "ratios": {"blade": 0.74, "guard": 0.05, "grip": 0.17, "pommel": 0.04,
                   "blade_width": 0.045, "blade_thickness": 0.012,
                   "fuller": 0.30},
        "parts": {"blade": 0.66, "guard": 0.12, "grip": 0.16, "pommel": 0.06},
        "variation": 0.12, "tags": ["holdable", "edge", "long"],
    },
    "axe": {
        "category": "tool", "size_mm": [700, 220, 45],
        "ratios": {"haft": 0.78, "head": 0.22, "head_width": 0.30,
                   "edge_angle_deg": 25.0},
        "parts": {"haft": 0.62, "head": 0.38},
        "variation": 0.12, "tags": ["holdable", "edge", "long"],
    },
    "shield": {
        "category": "armor", "size_mm": [600, 60, 750],
        "ratios": {"thickness": 0.06, "boss": 0.18, "rim": 0.07},
        "parts": {"face": 0.74, "boss": 0.14, "rim": 0.12},
        "variation": 0.12, "tags": ["holdable", "protective", "flat"],
    },
    "magazine": {
        "category": "weapon_part", "size_mm": [26, 72, 190],
        "ratios": {"wall": 0.045, "rib_count": 0.0, "feed_lip": 0.06,
                   "floorplate": 0.05, "curve_deg": 12.0},
        "parts": {"body": 0.78, "floorplate": 0.10, "feed_lips": 0.12},
        "variation": 0.08, "tags": ["holdable", "hollow", "mechanical"],
    },
    "grip": {
        "category": "weapon_part", "size_mm": [38, 52, 110],
        "ratios": {"wall": 0.06, "palm_swell": 0.18, "finger_grooves": 3.0},
        "parts": {"shell": 0.85, "insert": 0.15},
        "variation": 0.10, "tags": ["holdable", "ergonomic"],
    },
    "helmet": {
        "category": "armor", "size_mm": [230, 280, 260],
        "ratios": {"wall": 0.022, "visor": 0.30, "crown_height": 0.42},
        "parts": {"shell": 0.72, "visor": 0.16, "rim": 0.12},
        "variation": 0.10, "tags": ["wearable", "hollow", "protective"],
    },
    "tool_box": {
        "category": "product", "size_mm": [420, 220, 180],
        "ratios": {"wall": 0.02, "lid": 0.30, "handle_span": 0.35},
        "parts": {"body": 0.62, "lid": 0.28, "handle": 0.10},
        "variation": 0.12, "tags": ["container", "hollow"],
    },
    # ---- larger hardware ---------------------------------------------------
    "rifle": {
        "category": "weapon", "size_mm": [900, 60, 220],
        "ratios": {"barrel": 0.46, "receiver": 0.22, "stock": 0.24,
                   "magazine": 0.08, "barrel_d": 0.055},
        "parts": {"barrel": 0.30, "receiver": 0.26, "stock": 0.22,
                  "magazine": 0.10, "grip": 0.12},
        "variation": 0.09, "tags": ["holdable", "mechanical", "long"],
    },
    "drone": {
        "category": "vehicle", "size_mm": [320, 320, 90],
        "ratios": {"arm_span": 0.92, "body": 0.34, "rotor_d": 0.42,
                   "landing": 0.16},
        "parts": {"body": 0.30, "arms": 0.28, "rotors": 0.30, "landing": 0.12},
        "variation": 0.14, "tags": ["mechanical", "symmetric"],
    },
    "chair": {
        "category": "furniture", "size_mm": [480, 520, 900],
        "ratios": {"seat_height": 0.50, "back_height": 0.50,
                   "seat_depth": 0.52, "leg": 0.06},
        "parts": {"seat": 0.22, "back": 0.26, "legs": 0.52},
        "variation": 0.10, "tags": ["human_scale", "structural"],
    },
    "table": {
        "category": "furniture", "size_mm": [1200, 700, 750],
        "ratios": {"top_thickness": 0.05, "leg": 0.07, "apron": 0.10},
        "parts": {"top": 0.30, "legs": 0.55, "apron": 0.15},
        "variation": 0.12, "tags": ["human_scale", "structural"],
    },
    "column": {
        "category": "architecture", "size_mm": [600, 600, 3000],
        "ratios": {"base": 0.06, "capital": 0.06, "entasis": 0.03,
                   "flutes": 20.0},
        "parts": {"base": 0.08, "shaft": 0.84, "capital": 0.08},
        "variation": 0.08, "tags": ["structural", "architectural"],
    },
    # ---- creatures and characters ------------------------------------------
    "humanoid": {
        "category": "character", "size_mm": [500, 300, 1700],
        "ratios": {"head": 0.125, "torso": 0.30, "legs": 0.47,
                   "arms": 0.44, "shoulder_w": 0.42},
        "parts": {"head": 0.10, "torso": 0.32, "arms": 0.22, "legs": 0.36},
        "variation": 0.10, "tags": ["human_scale", "bilateral", "organic"],
    },
    "quadruped": {
        "category": "character", "size_mm": [1100, 320, 800],
        "ratios": {"head": 0.18, "body": 0.46, "legs": 0.40,
                   "tail": 0.22, "shoulder_h": 0.62},
        "parts": {"head": 0.14, "body": 0.40, "legs": 0.34, "tail": 0.12},
        "variation": 0.12, "tags": ["bilateral", "organic"],
    },
    "bird": {
        "category": "character", "size_mm": [420, 180, 260],
        "ratios": {"head": 0.16, "body": 0.42, "wings": 0.34,
                   "tail": 0.20, "beak": 0.10},
        "parts": {"head": 0.14, "body": 0.40, "wings": 0.30, "tail": 0.16},
        "variation": 0.13, "tags": ["bilateral", "organic"],
    },
    "insect": {
        "category": "character", "size_mm": [90, 50, 35],
        "ratios": {"head": 0.18, "thorax": 0.28, "abdomen": 0.40,
                   "legs": 0.34, "antennae": 0.16},
        "parts": {"head": 0.16, "thorax": 0.26, "abdomen": 0.32,
                  "legs": 0.18, "antennae": 0.08},
        "variation": 0.14, "tags": ["bilateral", "organic", "many_limbs"],
    },
    "fish": {
        "category": "character", "size_mm": [300, 90, 140],
        "ratios": {"head": 0.24, "body": 0.50, "tail": 0.26,
                   "fins": 0.18, "body_depth": 0.46},
        "parts": {"head": 0.22, "body": 0.50, "tail": 0.16, "fins": 0.12},
        "variation": 0.13, "tags": ["bilateral", "organic", "streamlined"],
    },
    "tree": {
        "category": "nature", "size_mm": [3000, 3000, 6000],
        "ratios": {"trunk": 0.34, "crown": 0.62, "trunk_d": 0.06,
                   "branch_levels": 3.0},
        "parts": {"trunk": 0.22, "branches": 0.36, "crown": 0.42},
        "variation": 0.18, "tags": ["organic", "branching", "large"],
    },
    "mushroom": {
        "category": "nature", "size_mm": [120, 120, 140],
        "ratios": {"cap": 0.46, "stem": 0.54, "cap_thickness": 0.14,
                   "gill_count": 24.0},
        "parts": {"cap": 0.58, "stem": 0.42},
        "variation": 0.16, "tags": ["organic", "radial"],
    },
    # ---- vehicles and large structures ------------------------------------
    "car": {
        "category": "vehicle", "size_mm": [4400, 1800, 1450],
        "ratios": {"wheelbase": 0.62, "body_height": 0.52, "cabin": 0.34,
                   "wheel_d": 0.19},
        "parts": {"body": 0.52, "cabin": 0.22, "wheels": 0.18, "details": 0.08},
        "variation": 0.10, "tags": ["human_scale", "symmetric", "streamlined"],
    },
    "spaceship": {
        "category": "vehicle", "size_mm": [12000, 4000, 3000],
        "ratios": {"hull": 0.62, "engines": 0.18, "wings": 0.14,
                   "cockpit": 0.06},
        "parts": {"hull": 0.50, "engines": 0.20, "wings": 0.18,
                  "cockpit": 0.06, "greebles": 0.06},
        "variation": 0.15, "tags": ["symmetric", "fictional", "greebled"],
    },
    "building": {
        "category": "architecture", "size_mm": [12000, 9000, 15000],
        "ratios": {"floor_height": 0.06, "wall": 0.03, "roof": 0.10,
                   "window_ratio": 0.35},
        "parts": {"walls": 0.44, "floors": 0.30, "roof": 0.14, "openings": 0.12},
        "variation": 0.12, "tags": ["architectural", "structural", "large"],
    },
    # ---- astronomical ------------------------------------------------------
    "planet": {
        "category": "astronomical", "size_mm": [100, 100, 100],
        "ratios": {"core": 0.35, "mantle": 0.50, "crust": 0.06,
                   "atmosphere": 0.04},
        "parts": {"sphere": 1.0},
        "variation": 0.10, "tags": ["radial", "fictional_scale"],
    },
    "asteroid": {
        "category": "astronomical", "size_mm": [100, 90, 80],
        "ratios": {"crater_count": 12.0, "crater_depth": 0.08,
                   "irregularity": 0.22},
        "parts": {"body": 1.0},
        "variation": 0.25, "tags": ["fictional_scale", "irregular"],
    },
    "ringed_planet": {
        "category": "astronomical", "size_mm": [100, 100, 100],
        "ratios": {"ring_inner": 1.35, "ring_outer": 2.30, "ring_thickness": 0.04,
                   "tilt_deg": 24.0},
        "parts": {"sphere": 0.72, "rings": 0.28},
        "variation": 0.10, "tags": ["radial", "fictional_scale"],
    },
    # ---- abstract / generic ------------------------------------------------
    "generic": {
        "category": "generic", "size_mm": [100, 100, 100],
        "ratios": {"wall": 0.05, "taper": 1.0},
        "parts": {"body": 1.0},
        "variation": 0.20, "tags": ["generic"],
    },
}

#: Human-scale anchors, in mm.  Anything a person operates is sized against
#: these, which is what makes an invented object read as real.
HUMAN_SCALE_MM = {
    "human_height": 1700.0,
    "hand_length": 190.0,
    "hand_width": 90.0,
    "grip_diameter": 32.0,
    "door_height": 2000.0,
    "seat_height": 450.0,
    "table_height": 750.0,
    "step_height": 180.0,
    "reach": 750.0,
}

#: Synonym table so a free-text label in English or Chinese resolves without a
#: translation step.  Order matters only for readability; matching prefers the
#: longest synonym, so "rifle magazine" beats "rifle".
SYNONYMS = {
    "mug": ["cup", "coffee mug", "teacup", "杯子", "马克杯", "咖啡杯", "茶杯"],
    "bottle": ["flask", "canteen", "瓶子", "水瓶", "水壶"],
    "knife": ["blade", "dagger", "刀", "匕首", "小刀"],
    "sword": ["longsword", "katana", "saber", "sabre", "剑", "长剑", "刀剑"],
    "axe": ["hatchet", "斧", "斧头"],
    "shield": ["buckler", "盾", "盾牌"],
    "magazine": ["mag", "clip", "ammo magazine", "弹夹", "弹匣", "弹仓"],
    "grip": ["pistol grip", "handle grip", "握把", "枪柄"],
    "helmet": ["helm", "头盔", "盔"],
    "tool_box": ["toolbox", "tool box", "toolcase", "case", "工具箱", "箱子", "工具盒"],
    "rifle": ["gun", "firearm", "carbine", "步枪", "枪", "长枪"],
    "drone": ["quadcopter", "uav", "无人机"],
    "chair": ["seat", "椅子", "座椅"],
    "table": ["desk", "桌子", "台"],
    "column": ["pillar", "柱子", "圆柱", "立柱"],
    "humanoid": ["person", "human", "character", "creature", "alien", "figure",
                 "人", "人物", "人形", "角色", "生物", "外星人"],
    "quadruped": ["dog", "cat", "horse", "beast", "animal", "monster",
                  "四足", "兽", "狗", "猫", "马", "怪物", "动物"],
    "bird": ["avian", "鸟", "飞鸟"],
    "insect": ["bug", "beetle", "昆虫", "甲虫", "虫"],
    "fish": ["鱼", "鱼类"],
    "tree": ["plant", "树", "树木"],
    "mushroom": ["fungus", "蘑菇", "菌"],
    "car": ["automobile", "vehicle", "汽车", "车", "车辆"],
    "spaceship": ["starship", "spacecraft", "飞船", "战舰", "宇宙飞船"],
    "building": ["house", "tower", "建筑", "房子", "大楼", "塔"],
    "planet": ["world", "globe", "星球", "行星", "天体"],
    "asteroid": ["meteor", "rock", "小行星", "陨石", "岩石"],
    "ringed_planet": ["ringed planet", "ringed gas giant", "saturn", "ringed world",
                      "带环行星", "土星", "有环行星"],
    "generic": ["object", "thing", "物体", "东西"],
}


# ---------------------------------------------------------------- seed

def _seed_of(label: str, seed: int, style: str) -> np.random.Generator:
    raw = ("%s|%s|%s" % (label, style, int(seed))).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    n = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(n)


def _match(label: str):
    """Resolve a free-text label to an archetype key.

    Resolution order, most specific first:

    1. an exact archetype key;
    2. the **longest** synonym (English or Chinese) contained in the label —
       so "rifle magazine" resolves to ``magazine``, not ``rifle``;
    3. the longest archetype key contained in the label;
    4. ``generic``.

    Longest-match is the whole point: a naive token scan picks the first word
    and turns a magazine into a rifle.
    """
    text = str(label or "").strip().lower()
    if not text:
        return "generic", "empty label"
    if text in ARCHETYPES:
        return text, "exact"

    best_key, best_len, best_src = None, 0, ""
    for key, words in SYNONYMS.items():
        for word in [key] + list(words):
            w = word.lower()
            if w and w in text and len(w) > best_len:
                best_key, best_len, best_src = key, len(w), "synonym"
    if best_key:
        return best_key, best_src

    for key in ARCHETYPES:
        if key in text and len(key) > best_len:
            best_key, best_len = key, len(key)
    if best_key:
        return best_key, "substring"
    return "generic", "no match"


# ---------------------------------------------------------------- conjure

def archetypes():
    """The archetype table, for ``atta_catalog``."""
    return {
        key: {
            "category": v["category"],
            "size_mm": list(v["size_mm"]),
            "ratios": dict(v["ratios"]),
            "parts": dict(v["parts"]),
            "variation": v["variation"],
            "tags": list(v["tags"]),
        }
        for key, v in sorted(ARCHETYPES.items())
    }


def archetype(name: str):
    """One archetype, or raise with the available keys.

    An unmatched label is **not** an error: it resolves to ``generic`` and the
    caller is told so.  A fictional subject has no real-world reference to look
    up, so refusing to build it would be the wrong failure — the whole point of
    ``conjure`` is that a made-up thing still gets a plausible body.
    """
    key, how = _match(name)
    return key, copy.deepcopy(ARCHETYPES[key]), how


def conjure(label: str, *, seed: int = 0, style: str = "neutral",
            size_mm=None, scale=1.0, constraints=None) -> dict:
    """Invent a plausible parameter set for ``label``.

    Returns a dict with:

    ``archetype``    which table entry was used and how it was matched;
    ``size_mm``      the bounding box, with per-axis jitter from the seed;
    ``ratios``       the archetype's proportions, style-modulated;
    ``parts``        the decomposition, each with an absolute size;
    ``anchors``      the human-scale references that justify the size;
    ``provenance``   per-field ``measured`` / ``archetype`` / ``invented``;
    ``seed``         the effective seed, so the result is reproducible;
    ``notes``        why the numbers are what they are.

    ``size_mm`` overrides the archetype box (those fields become ``measured``).
    ``constraints`` may carry ``{"max_mm": [x,y,z], "min_mm": [...]}`` to clamp
    the invented box — the hook a caller uses to fit an existing space.
    """
    if style not in STYLES:
        raise AttaError("bad_style", "unknown style %r" % (style,),
                        path="style",
                        fix="use one of: " + ", ".join(sorted(STYLES)))
    key, arch, how = archetype(label)
    rng = _seed_of(key, seed, style)
    vec = STYLES[style]
    variation = float(arch["variation"])
    provenance = {}
    notes = ["matched %r to archetype %r (%s)" % (label, key, how)]

    # --- size
    base = np.asarray(arch["size_mm"], dtype=np.float64)
    if size_mm is not None:
        given = np.asarray(size_mm, dtype=np.float64).reshape(-1)
        if given.shape != (3,) or not np.isfinite(given).all() or np.any(given <= 0):
            raise AttaError("bad_arg", "size_mm must be three positive lengths",
                            path="size_mm", fix="pass [x,y,z] in mm")
        size = given.copy()
        provenance["size_mm"] = "measured"
        notes.append("size_mm supplied by the caller")
    else:
        # Jitter each axis independently, then apply the style's bulk factor
        # to the two cross-section axes (never to the long axis: scaling the
        # length of a rifle by "bulk" is what makes invented hardware look
        # wrong).
        jitter = 1.0 + rng.uniform(-variation, variation, size=3)
        size = base * jitter
        long_axis = int(np.argmax(size))
        for i in range(3):
            if i != long_axis:
                size[i] *= vec["bulk"]
        provenance["size_mm"] = "invented"
        notes.append("size from archetype %s jittered +-%.0f%% (seed %d)"
                     % (base.tolist(), variation * 100.0, int(seed)))

    try:
        factor = float(scale)
    except (TypeError, ValueError):
        factor = float("nan")
    if not math.isfinite(factor) or factor <= 0.0:
        raise AttaError("bad_arg", "scale must be finite and positive",
                        path="scale", fix="pass scale > 0")
    source_size = size.copy()
    size = size * factor
    operations = []
    if factor != 1.0:
        operations.append({"op": "scale", "factor": factor})
        notes.append("global scale %.4f applied; source is not a measurement"
                     % factor)

    # Bounds are constraints, never evidence that an invented length was measured.
    cons = dict(constraints or {})
    bounds = {}
    for name in ("min_mm", "max_mm"):
        if cons.get(name) is None:
            continue
        try:
            bound = np.asarray(cons[name], dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            bound = np.array([])
        if bound.shape != (3,) or not np.isfinite(bound).all() or np.any(bound <= 0):
            raise AttaError("bad_arg", "%s must be three positive finite lengths" % name,
                            path="constraints." + name,
                            fix="pass [x,y,z] in mm")
        bounds[name] = bound
    if ("min_mm" in bounds and "max_mm" in bounds
            and np.any(bounds["min_mm"] > bounds["max_mm"])):
        raise AttaError("infeasible_constraints", "min_mm exceeds max_mm",
                        path="constraints", fix="make min_mm <= max_mm on every axis")
    before_bounds = size.copy()
    if "max_mm" in bounds:
        size = np.minimum(size, bounds["max_mm"])
    if "min_mm" in bounds:
        size = np.maximum(size, bounds["min_mm"])
    if not np.array_equal(size, before_bounds):
        operations.append({"op": "constrain", "bounds": {
            k: v.tolist() for k, v in bounds.items()}})
        notes.append("box adjusted to caller bounds; values are derived, not measured")
    source = provenance["size_mm"]
    if operations:
        provenance["size_mm"] = "derived"
    provenance_detail = {"size_mm": {
        "source": source, "source_value_mm": source_size.tolist(),
        "operations": operations, "measured": bool(source == "measured" and not operations),
    }}

    # --- ratios, style-modulated
    ratios = {}
    for name, value in arch["ratios"].items():
        v = float(value)
        if name.endswith("_deg"):
            ratios[name] = round(v * (0.85 + 0.30 * vec["sharp"]), 3)
        elif "count" in name or name.endswith("_count"):
            # Counts are integers; style detail modulates them.
            ratios[name] = int(max(0, round(v * vec["detail"])))
        elif name in ("wall", "thickness", "blade_thickness", "top_thickness",
                      "cap_thickness", "ring_thickness", "crater_depth",
                      "barrel_d", "trunk_d", "wheel_d", "rotor_d", "leg",
                      "apron", "rim", "floor_height", "seat_depth"):
            ratios[name] = round(v * vec["bulk"], 4)
        else:
            ratios[name] = round(v * (0.8 + 0.2 * vec["taper"]), 4)
        provenance["ratios.%s" % name] = (
            "archetype" if ratios[name] == value else "derived")
        provenance_detail["ratios.%s" % name] = {
            "source": "archetype", "source_value": value, "style": style}
    # These are partitions of a single length, not independent scale factors.
    partitions = {"knife": ("blade", "handle"),
                  "sword": ("blade", "guard", "grip", "pommel"),
                  "mushroom": ("cap", "stem")}
    fields = partitions.get(key, ())
    if fields:
        weights = [float(arch["ratios"][k]) for k in fields]
        weights[0] *= vec["taper"]
        total_weight = sum(weights)
        for name, weight in zip(fields, weights):
            ratios[name] = weight / total_weight
            provenance["ratios.%s" % name] = (
                "archetype" if style == "neutral" else "derived")
    notes.append("style %r modulated ratios (bulk %.2f, taper %.2f, detail %.2f)"
                 % (style, vec["bulk"], vec["taper"], vec["detail"]))

    # --- parts: absolute sizes from the shares
    parts = {}
    total = float(sum(arch["parts"].values())) or 1.0
    for name, share in arch["parts"].items():
        frac = float(share) / total
        parts[name] = {
            "share": round(frac, 4),
            "size_mm": [round(float(v) * frac, 3) for v in size],
            "volume_share": round(frac, 4),
            "exact": False,
            "note": "conceptual share and size hint, not a measured part bbox or volume",
        }
        provenance["parts.%s.share" % name] = "archetype"
        provenance["parts.%s.size_mm" % name] = "derived"
        provenance["parts.%s.volume_share" % name] = "archetype"

    # --- human-scale anchors
    anchors = {}
    tags = set(arch["tags"])
    if "holdable" in tags or "wearable" in tags:
        anchors["hand_length_mm"] = HUMAN_SCALE_MM["hand_length"]
        anchors["grip_diameter_mm"] = HUMAN_SCALE_MM["grip_diameter"]
        longest = float(size.max())
        anchors["fits_hand"] = bool(longest <= HUMAN_SCALE_MM["hand_length"] * 3.2)
        notes.append("holdable: checked against a %.0f mm hand"
                     % HUMAN_SCALE_MM["hand_length"])
    if "human_scale" in tags or "architectural" in tags:
        anchors["human_height_mm"] = HUMAN_SCALE_MM["human_height"]
        anchors["door_height_mm"] = HUMAN_SCALE_MM["door_height"]
        anchors["seat_height_mm"] = HUMAN_SCALE_MM["seat_height"]
        anchors["table_height_mm"] = HUMAN_SCALE_MM["table_height"]
        notes.append("human-scale object: sized against a %.0f mm person"
                     % HUMAN_SCALE_MM["human_height"])
    if "fictional_scale" in tags:
        anchors["note"] = ("no real-world reference exists; the box is a "
                           "display scale, not a measurement")
        notes.append("fictional subject: dimensions are invented, not looked up")

    # --- derived recommendations a builder can use directly
    build = _build_hint(key, arch, size, ratios, vec, rng)
    build_constraints = dict(build.get("constraints", {}))
    if bounds:
        build_constraints["bbox"] = {
            axis: {k.replace("_mm", ""): float(v[i]) for k, v in bounds.items()}
            for i, axis in enumerate("xyz")}
    # Every numeric construction value is derived from the labelled plan, not
    # independently measured. Record leaves so agents need not infer provenance.
    def tag_build(value, path="build"):
        if isinstance(value, dict):
            for name, child in value.items():
                tag_build(child, path + "." + str(name))
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                tag_build(child, path + "[%d]" % i)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            provenance[path] = "derived"
    tag_build(build)
    assumptions = [
        "conjured from archetype %r (seed %d, style %s); size source: %s; "
        "construction geometry and conceptual part shares are derived, not measured"
        % (key, int(seed), style, provenance["size_mm"])]

    return {
        "schema": "atta.conjure/v1",
        "label": str(label),
        "archetype": {"key": key, "matched_by": how,
                      "category": arch["category"], "tags": list(arch["tags"])},
        "style": style,
        "seed": int(seed),
        "size_mm": [round(float(v), 3) for v in size],
        "ratios": ratios,
        "parts": parts,
        "anchors": anchors,
        "build": build,
        "provenance": provenance,
        "provenance_detail": provenance_detail,
        "constraints": build_constraints,
        "assumptions": assumptions,
        "notes": notes,
        "units": "mm",
        "exact": False,
        "method": METHOD,
    }


def _build_hint(key, arch, size, ratios, vec, rng):
    """A concrete construction recipe for the archetype.

    This is the bridge from "invented numbers" to a graph the kernel can
    evaluate: the caller gets op names and arguments, not a paragraph.
    """
    sx, sy, sz = (float(v) for v in size)
    long_axis = int(np.argmax(size))
    hint = {"op": "sdf_blend", "size_mm": [round(sx, 2), round(sy, 2),
                                           round(sz, 2)],
            "blend_mm": round(min(size) * 0.08, 2), "primitives": []}
    prims = hint["primitives"]

    if key in ("planet", "asteroid"):
        prims.append({"kind": "sphere", "radius_mm": round(min(size) * 0.5, 2),
                      "center_mm": [0, 0, 0]})
        if key == "asteroid":
            hint["noise"] = {"amount_mm": round(min(size) * 0.06, 2),
                             "frequency": round(40.0 / max(min(size), 1.0), 3),
                             "seed": int(rng.integers(0, 1 << 30)),
                             "octaves": 4, "kind": "ridged"}
        return hint

    if key in ("humanoid", "quadruped", "bird", "insect", "fish"):
        # A skeleton is the right representation for a creature: pose and
        # proportion live in the joints, volume in the radii.
        return _creature_hint(key, size, ratios, rng)

    if key == "tree":
        return _tree_hint(size, ratios, rng)

    if key == "mushroom":
        return {"op": "sdf_blend", "size_mm": hint["size_mm"],
                "blend_mm": round(min(size) * 0.05, 2),
                "primitives": [
                    {"kind": "cylinder", "radius_mm": round(sx * 0.09, 2),
                     "height_mm": round(sz * ratios.get("stem", 0.54), 2),
                     "center_mm": [0, 0, 0]},
                    {"kind": "ellipsoid",
                     "radii_mm": [round(sx * 0.5, 2), round(sy * 0.5, 2),
                                  round(sz * ratios.get("cap_thickness", 0.14), 2)],
                     "center_mm": [0, 0, round(sz * 0.30, 2)]},
                ]}

    if key in ("mug", "bottle", "helmet"):
        return _shell_graph(key, size, ratios, vec)

    if key in ("tool_box", "grip", "magazine"):
        # Hollow shell + one or two functional features.  The shell is made by
        # subtracting a slightly smaller inner volume, which is the honest way
        # to get a wall: `shell_mm` on an SDF field is a different operation
        # (it hollows toward the surface) and produces a genus-heavy result on
        # a rounded box.
        wall = max(1.2, min(size) * float(ratios.get("wall", 0.03)))
        outer = [round(sx * 0.94, 2), round(sy * 0.94, 2), round(sz * 0.94, 2)]
        prims.append({"kind": "box", "size_mm": outer,
                      "round_mm": round(min(size) * 0.10, 2)})
        if key == "mug":
            # Handle: a torus in the X-Z plane, attached to the +X side.
            prims.append({"kind": "torus",
                          "major_mm": round(sy * 0.30, 2),
                          "minor_mm": round(max(wall * 1.1, sy * 0.055), 2),
                          "center_mm": [round(sx * 0.50, 2), 0, 0],
                          "axis": "y"})
        if key == "bottle":
            prims.append({"kind": "cylinder",
                          "radius_mm": round(min(sx, sy) * 0.22, 2),
                          "height_mm": round(sz * 0.26, 2),
                          "center_mm": [0, 0, round(sz * 0.44, 2)]})
        if key == "magazine":
            # Feed lips: a narrow block at the top, plus a floorplate.
            prims.append({"kind": "box",
                          "size_mm": [round(sx * 0.86, 2), round(sy * 0.26, 2),
                                      round(sz * 0.08, 2)],
                          "center_mm": [0, 0, round(sz * 0.46, 2)]})
        # The cavity is open at the top: its Z extent runs past the outer shell
        # so the subtraction cuts a mouth instead of leaving a sealed bubble
        # inside the solid.  A sealed cavity is a manufacturing defect, not a
        # mug.
        hint["cavity"] = {
            "kind": "box",
            "size_mm": [round(max(outer[0] - 2 * wall, 1.0), 2),
                        round(max(outer[1] - 2 * wall, 1.0), 2),
                        round(max(outer[2] - wall, 1.0), 2)],
            "center_mm": [0.0, 0.0, round(wall * 0.5, 2)],
            "round_mm": round(min(size) * 0.06, 2),
        }
        # The blend must stay well under the wall, or the smooth union fills the
        # cavity it was supposed to leave.  Cap it at a third of the wall.
        hint["blend_mm"] = round(min(hint.get("blend_mm", 0.0), wall * 0.33), 2)
        return hint

    if key in ("rifle", "spaceship", "car", "drone", "knife", "sword", "axe",
               "shield"):
        return _hardware_hint(key, size, ratios, rng)

    if key in ("chair", "table", "column", "building"):
        return _structure_hint(key, size, ratios, rng)

    # generic fallback: a rounded box at the invented size
    prims.append({"kind": "box",
                  "size_mm": [round(sx * 0.9, 2), round(sy * 0.9, 2),
                              round(sz * 0.9, 2)],
                  "round_mm": round(min(size) * 0.08, 2)})
    return hint


def _shell_graph(key, size, ratios, vec):
    """Explicit closed wall sections: no coincident SDF mouth or sealed neck.

    Reuses the same revolve/sweep construction as the curated product recipes,
    without their fixed human-scale parameter bounds silently clamping a plan.
    ``wall`` is a nominal budget; verification still measures the resulting mesh.
    """
    sx, sy, h = (float(v) for v in size)
    wall = min(max(0.3, min(size) * ratios["wall"]), min(size) * 0.10)
    features = []
    if key == "helmet":
        # An ellipsoidal crown, closed above and open DOWN for the wearer's head.
        # Two nested profiles share only the annular rim, never a sealed bubble.
        r = 0.5 * sx
        theta = np.linspace(0.0, math.pi / 2, 33)
        outer = np.column_stack([r * np.cos(theta), h * np.sin(theta)])
        inner = np.column_stack([(r - wall) * np.cos(theta[::-1]),
                                 (h - wall) * np.sin(theta[::-1])])
        outer[-1, 0] = 0.0
        inner[0, 0] = 0.0
        profile = np.vstack([outer, inner, outer[:1]])
        features.append({"id": "crown", "op": "revolve", "args": {
            "profile_rz": np.round(profile, 6).tolist(), "segments": 64}})
        # Start with the X radius, then stretch only Y. Budget for compression.
        features.append({"id": "body", "op": "transform", "inputs": ["crown"],
                         "args": {"scale": [1.0, sy / sx, 1.0]}})
        return {"kind": "graph", "features": features, "output": "body",
                "size_mm": list(map(float, size)), "nominal_wall_mm": wall,
                "opening": "-z", "constraints": {"genus": 0, "n_components": 1,
                    "min_wall_mm": wall * min(1.0, sy / sx) * 0.7}}

    if key == "mug":
        radius = min(sy * 0.5, sx / 2.7)
        # Taper changes the silhouette while preserving the overall size.
        top = radius * (1.0 - 0.12 * (vec["taper"] - 1.0))
        z = np.linspace(0.0, h, 33)
        r = radius * (0.91 + 0.09 * np.sin(math.pi * z / h))
        r += (top - radius * 0.91) * (z / h) ** 5
        r = np.minimum(r, radius)
    else:
        radius = min(sx, sy) * 0.5
        neck_fraction = min(0.50, max(0.18, ratios["neck"]))
        shoulder = min(0.35, max(0.12, ratios["shoulder"]))
        z = h * np.array([0.0, 0.08, 0.60, 1.0 - shoulder, 0.92, 1.0])
        r = radius * np.array([0.94, 1.0, 1.0, 0.92,
                               neck_fraction, neck_fraction * 1.10])
    base_t = min(h * 0.15, max(wall * 1.5, h * 0.035))
    inner_z = np.unique(np.r_[base_t, z[z > base_t]])
    # A radial subtraction underestimates the normal thickness on shoulders.
    # Compensate by the local profile slope instead of calling wall=r_out-r_in.
    slopes = np.abs(np.diff(r) / np.diff(z))
    budget = np.maximum(np.r_[slopes[0], slopes], np.r_[slopes, slopes[-1]])
    inset = wall * np.sqrt(1.0 + budget ** 2)
    inner_r = np.interp(inner_z, z, r - inset)
    if np.any(inner_r <= wall * 0.25) or base_t >= h:
        raise AttaError("infeasible_shell", "wall budget would close the cavity",
                        path="size_mm", fix="increase the cross-section or reduce wall")
    profile = np.vstack([np.column_stack([r, z]),
                         np.column_stack([inner_r[::-1], inner_z[::-1]]),
                         [[0.0, base_t], [0.0, 0.0]]])
    features.append({"id": "vessel", "op": "revolve", "args": {
        "profile_rz": np.round(profile, 6).tolist(), "segments": 64}})
    output = "vessel"
    if key == "bottle":
        features.append({"id": "body", "op": "transform", "inputs": [output],
                         "args": {"scale": [sx / (2 * radius), sy / (2 * radius), 1.0]}})
        output = "body"
    if key == "mug":
        tube = max(wall, sy * 0.05) * (0.9 + 0.1 * vec["bulk"])
        z1, z2 = h * 0.76, h * 0.28
        r1, r2 = np.interp([z1, z2], z, r)
        reach = sx - radius - tube
        if reach <= max(r1, r2) + 2 * tube:
            raise AttaError("infeasible_shell", "mug handle has no readable gap",
                            path="size_mm", fix="increase X relative to the body diameter")
        depth = min(wall * 0.45, tube * 0.45)
        arc = np.linspace(0, math.pi, 41)
        attach = (r1 + r2) * 0.5 - depth
        path = np.column_stack([
            attach + (reach - attach) * np.sin(arc), np.zeros(len(arc)),
            (z1 + z2) * 0.5 + (z1 - z2) * 0.5 * np.cos(arc)])
        path[0, 0], path[-1, 0] = r1 - depth, r2 - depth
        angle = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        features.append({"id": "handle", "op": "sweep", "args": {
            "section_xy": np.round(np.column_stack([tube * np.cos(angle),
                                                     tube * np.sin(angle)]), 6).tolist(),
            "path": np.round(path, 6).tolist()}})
        features.append({"id": "body", "op": "union", "inputs": [output, "handle"]})
        output = "body"
    return {"kind": "graph", "features": features, "output": output,
            "size_mm": list(map(float, size)), "nominal_wall_mm": wall,
            "opening": "+z", "constraints": {
                "genus": 1 if key == "mug" else 0, "n_components": 1,
                "min_wall_mm": wall * 0.7}}


def _creature_hint(key, size, ratios, rng):
    sx, sy, sz = (float(v) for v in size)
    # A spine plus limbs, expressed as a skeleton the `skeleton` op eats.
    if key == "humanoid":
        h = sz
        joints = [
            {"id": "pelvis", "at_mm": [0, 0, h * 0.50], "radius_mm": h * 0.085},
            {"id": "chest", "at_mm": [0, 0, h * 0.72], "radius_mm": h * 0.105},
            {"id": "neck", "at_mm": [0, 0, h * 0.82], "radius_mm": h * 0.035},
            {"id": "head", "at_mm": [0, 0, h * 0.90], "radius_mm": h * 0.062},
            {"id": "l_shoulder", "at_mm": [-sx * 0.20, 0, h * 0.78], "radius_mm": h * 0.040},
            {"id": "r_shoulder", "at_mm": [sx * 0.20, 0, h * 0.78], "radius_mm": h * 0.040},
            {"id": "l_hand", "at_mm": [-sx * 0.26, 0, h * 0.42], "radius_mm": h * 0.022},
            {"id": "r_hand", "at_mm": [sx * 0.26, 0, h * 0.42], "radius_mm": h * 0.022},
            {"id": "l_foot", "at_mm": [-sx * 0.09, 0, 0.0], "radius_mm": h * 0.030},
            {"id": "r_foot", "at_mm": [sx * 0.09, 0, 0.0], "radius_mm": h * 0.030},
            {"id": "l_knee", "at_mm": [-sx * 0.09, 0, h * 0.26], "radius_mm": h * 0.038},
            {"id": "r_knee", "at_mm": [sx * 0.09, 0, h * 0.26], "radius_mm": h * 0.038},
        ]
        bones = [
            {"a": "pelvis", "b": "chest"}, {"a": "chest", "b": "neck"},
            {"a": "neck", "b": "head"},
            {"a": "chest", "b": "l_shoulder"}, {"a": "l_shoulder", "b": "l_hand"},
            {"a": "chest", "b": "r_shoulder"}, {"a": "r_shoulder", "b": "r_hand"},
            {"a": "pelvis", "b": "l_knee"}, {"a": "l_knee", "b": "l_foot"},
            {"a": "pelvis", "b": "r_knee"}, {"a": "r_knee", "b": "r_foot"},
        ]
        return {"op": "skeleton", "joints": joints, "bones": bones,
                "blend_mm": round(h * 0.02, 2), "size_mm": [sx, sy, sz]}

    if key == "quadruped":
        h = sz
        L = sx
        joints = [
            {"id": "hips", "at_mm": [-L * 0.30, 0, h * 0.60], "radius_mm": h * 0.10},
            {"id": "shoulders", "at_mm": [L * 0.26, 0, h * 0.62], "radius_mm": h * 0.11},
            {"id": "neck", "at_mm": [L * 0.40, 0, h * 0.72], "radius_mm": h * 0.055},
            {"id": "head", "at_mm": [L * 0.50, 0, h * 0.78], "radius_mm": h * 0.07},
            {"id": "tail", "at_mm": [-L * 0.46, 0, h * 0.58], "radius_mm": h * 0.025},
        ]
        for side, sign in (("l", -1), ("r", 1)):
            for name, x, z in (("front", L * 0.26, h * 0.62),
                               ("hind", -L * 0.30, h * 0.60)):
                joints.append({"id": "%s_%s_hip" % (side, name),
                               "at_mm": [x, sign * sy * 0.12, z],
                               "radius_mm": h * 0.045})
                joints.append({"id": "%s_%s_foot" % (side, name),
                               "at_mm": [x, sign * sy * 0.14, 0.0],
                               "radius_mm": h * 0.028})
        bones = [{"a": "hips", "b": "shoulders"},
                 {"a": "shoulders", "b": "neck"}, {"a": "neck", "b": "head"},
                 {"a": "hips", "b": "tail"}]
        for side in ("l", "r"):
            bones.append({"a": "shoulders", "b": "%s_front_hip" % side})
            bones.append({"a": "%s_front_hip" % side, "b": "%s_front_foot" % side})
            bones.append({"a": "hips", "b": "%s_hind_hip" % side})
            bones.append({"a": "%s_hind_hip" % side, "b": "%s_hind_foot" % side})
        return {"op": "skeleton", "joints": joints, "bones": bones,
                "blend_mm": round(h * 0.02, 2), "size_mm": [sx, sy, sz]}

    if key == "fish":
        return {"op": "skeleton",
                "joints": [
                    {"id": "nose", "at_mm": [sx * 0.46, 0, 0], "radius_mm": sy * 0.14},
                    {"id": "head", "at_mm": [sx * 0.28, 0, 0], "radius_mm": sy * 0.30},
                    {"id": "body", "at_mm": [0, 0, 0], "radius_mm": sy * 0.40},
                    {"id": "tail_base", "at_mm": [-sx * 0.34, 0, 0], "radius_mm": sy * 0.12},
                    {"id": "tail", "at_mm": [-sx * 0.48, 0, 0], "radius_mm": sy * 0.03},
                ],
                "bones": [{"a": "nose", "b": "head"}, {"a": "head", "b": "body"},
                          {"a": "body", "b": "tail_base"},
                          {"a": "tail_base", "b": "tail"}],
                "blend_mm": round(sy * 0.05, 2), "size_mm": [sx, sy, sz]}

    if key == "bird":
        return {"op": "skeleton",
                "joints": [
                    {"id": "body", "at_mm": [0, 0, sz * 0.48], "radius_mm": sz * 0.16},
                    {"id": "head", "at_mm": [sx * 0.30, 0, sz * 0.72], "radius_mm": sz * 0.09},
                    {"id": "beak", "at_mm": [sx * 0.44, 0, sz * 0.70], "radius_mm": sz * 0.025},
                    {"id": "tail", "at_mm": [-sx * 0.42, 0, sz * 0.42], "radius_mm": sz * 0.04},
                    {"id": "l_wing", "at_mm": [0, -sy * 0.44, sz * 0.52], "radius_mm": sz * 0.05},
                    {"id": "r_wing", "at_mm": [0, sy * 0.44, sz * 0.52], "radius_mm": sz * 0.05},
                ],
                "bones": [{"a": "body", "b": "head"}, {"a": "head", "b": "beak"},
                          {"a": "body", "b": "tail"},
                          {"a": "body", "b": "l_wing"}, {"a": "body", "b": "r_wing"}],
                "blend_mm": round(sz * 0.03, 2), "size_mm": [sx, sy, sz]}

    # insect
    return {"op": "skeleton",
            "joints": [
                {"id": "head", "at_mm": [sx * 0.32, 0, sz * 0.5], "radius_mm": sz * 0.16},
                {"id": "thorax", "at_mm": [sx * 0.06, 0, sz * 0.5], "radius_mm": sz * 0.22},
                {"id": "abdomen", "at_mm": [-sx * 0.28, 0, sz * 0.48], "radius_mm": sz * 0.20},
                {"id": "tail", "at_mm": [-sx * 0.46, 0, sz * 0.46], "radius_mm": sz * 0.05},
            ],
            "bones": [{"a": "head", "b": "thorax"}, {"a": "thorax", "b": "abdomen"},
                      {"a": "abdomen", "b": "tail"}],
            "blend_mm": round(sz * 0.03, 2), "size_mm": [sx, sy, sz]}


def _tree_hint(size, ratios, rng):
    sx, sy, sz = (float(v) for v in size)
    levels = int(ratios.get("branch_levels", 3))
    joints = [{"id": "root", "at_mm": [0, 0, 0], "radius_mm": sx * 0.045}]
    bones = []
    prev = "root"
    for i in range(1, 5):
        z = sz * 0.34 * (i / 4.0)
        jid = "trunk%d" % i
        joints.append({"id": jid, "at_mm": [0, 0, z],
                       "radius_mm": sx * 0.045 * (1.0 - 0.16 * i)})
        bones.append({"a": prev, "b": jid})
        prev = jid
    # A few branches, seeded so the tree is reproducible.
    for i in range(max(2, levels + 1)):
        ang = float(rng.uniform(0, 2 * math.pi))
        reach = sx * 0.30 * float(rng.uniform(0.7, 1.2))
        jid = "branch%d" % i
        joints.append({"id": jid,
                       "at_mm": [reach * math.cos(ang), reach * math.sin(ang),
                                 sz * 0.36 + sz * 0.10 * float(rng.uniform(0, 1))],
                       "radius_mm": sx * 0.018})
        bones.append({"a": "trunk3", "b": jid})
    return {"op": "skeleton", "joints": joints, "bones": bones,
            "blend_mm": round(sx * 0.02, 2), "size_mm": [sx, sy, sz]}


def _hardware_hint(key, size, ratios, rng):
    sx, sy, sz = (float(v) for v in size)
    prims = []
    if key == "rifle":
        L = sx
        prims.append({"kind": "box",
                      "size_mm": [round(L * 0.26, 2), round(sy * 0.55, 2),
                                  round(sz * 0.42, 2)],
                      "center_mm": [round(L * 0.02, 2), 0, round(sz * 0.05, 2)],
                      "round_mm": round(sy * 0.08, 2)})
        prims.append({"kind": "cylinder",
                      "radius_mm": round(sy * 0.14, 2),
                      "height_mm": round(L * 0.46, 2),
                      "center_mm": [round(L * 0.30, 2), 0, round(sz * 0.08, 2)],
                      "axis": "x"})
        prims.append({"kind": "box",
                      "size_mm": [round(L * 0.24, 2), round(sy * 0.42, 2),
                                  round(sz * 0.34, 2)],
                      "center_mm": [round(-L * 0.28, 2), 0, round(sz * 0.02, 2)]})
        prims.append({"kind": "box",
                      "size_mm": [round(L * 0.08, 2), round(sy * 0.50, 2),
                                  round(sz * 0.55, 2)],
                      "center_mm": [round(L * 0.02, 2), 0, round(-sz * 0.22, 2)]})
        return {"op": "sdf_blend", "size_mm": [round(sx, 2), round(sy, 2),
                                               round(sz, 2)],
                "blend_mm": round(sy * 0.06, 2), "primitives": prims}

    if key == "knife":
        L = sx
        blade = L * ratios["blade"]
        handle = L * ratios["handle"]
        prims.append({"kind": "box",
                      "size_mm": [round(blade, 2), round(sy * 0.9, 2),
                                  round(sy * ratios["blade_thickness"], 2)],
                      "center_mm": [round(-L * 0.5 + handle + blade * 0.5, 2), 0, 0]})
        prims.append({"kind": "box",
                      "size_mm": [round(handle, 2), round(sy * 0.7, 2),
                                  round(sz * 0.75, 2)],
                      "center_mm": [round(-L * 0.5 + handle * 0.5, 2), 0, 0],
                      "round_mm": round(sz * 0.25, 2)})
        return {"op": "sdf_blend", "size_mm": [round(sx, 2), round(sy, 2),
                                               round(sz, 2)],
                "blend_mm": round(sz * 0.10, 2), "primitives": prims}

    if key == "sword":
        L, W, T = sx, sy, sz
        # The partition controls actual stations, not just unused metadata.
        blade_len = L * ratios["blade"]
        guard_len = L * ratios["guard"]
        grip_len = L * ratios["grip"]
        guard_x = L * 0.5 - blade_len - guard_len * 0.5
        prims.append({"kind": "box",
                      "size_mm": [round(blade_len, 2), round(L * ratios["blade_width"], 2),
                                  round(L * ratios["blade_thickness"], 2)],
                      "center_mm": [round(L * 0.5 - blade_len * 0.5, 2), 0, 0]})
        prims.append({"kind": "box",
                      "size_mm": [round(guard_len, 2), round(W * 0.92, 2),
                                  round(T * 0.85, 2)],
                      "center_mm": [round(guard_x, 2), 0, 0],
                      "round_mm": round(T * 0.20, 2)})
        prims.append({"kind": "cylinder",
                      "radius_mm": round(W * 0.045, 2),
                      "height_mm": round(grip_len, 2),
                      "center_mm": [round(guard_x - guard_len * 0.5 - grip_len * 0.5, 2), 0, 0],
                      "axis": "x"})
        prims.append({"kind": "sphere",
                      "radius_mm": round(W * 0.075, 2),
                      "center_mm": [round(guard_x - guard_len * 0.5 - grip_len, 2), 0, 0]})
        return {"op": "sdf_blend", "size_mm": [round(sx, 2), round(sy, 2),
                                               round(sz, 2)],
                "blend_mm": round(T * 0.25, 2), "primitives": prims}

    if key == "axe":
        L, W, T = sx, sy, sz
        prims.append({"kind": "cylinder",
                      "radius_mm": round(W * 0.045, 2),
                      "height_mm": round(L * 0.96, 2),
                      "center_mm": [0, 0, 0], "axis": "z"})
        prims.append({"kind": "box",
                      "size_mm": [round(W * 0.55, 2), round(T * 0.9, 2),
                                  round(W * 0.42, 2)],
                      "center_mm": [round(W * 0.28, 2), 0, round(L * 0.36, 2)]})
        return {"op": "sdf_blend", "size_mm": [round(sx, 2), round(sy, 2),
                                               round(sz, 2)],
                "blend_mm": round(T * 0.20, 2), "primitives": prims}

    if key == "shield":
        W, T, H = sx, sy, sz
        prims.append({"kind": "box",
                      "size_mm": [round(W * 0.92, 2), round(T * 0.55, 2),
                                  round(H * 0.92, 2)],
                      "center_mm": [0, 0, 0],
                      "round_mm": round(min(W, H) * 0.16, 2)})
        prims.append({"kind": "sphere",
                      "radius_mm": round(min(W, H) * 0.13, 2),
                      "center_mm": [0, round(-T * 0.30, 2), 0]})
        return {"op": "sdf_blend", "size_mm": [round(sx, 2), round(sy, 2),
                                               round(sz, 2)],
                "blend_mm": round(T * 0.25, 2), "primitives": prims}

    if key == "car":
        L, W, H = sx, sy, sz
        prims.append({"kind": "box",
                      "size_mm": [round(L * 0.92, 2), round(W * 0.86, 2),
                                  round(H * 0.42, 2)],
                      "center_mm": [0, 0, round(H * 0.30, 2)],
                      "round_mm": round(H * 0.10, 2)})
        prims.append({"kind": "box",
                      "size_mm": [round(L * 0.44, 2), round(W * 0.74, 2),
                                  round(H * 0.30, 2)],
                      "center_mm": [round(-L * 0.04, 2), 0, round(H * 0.56, 2)],
                      "round_mm": round(H * 0.09, 2)})
        for sx_sign in (-1, 1):
            for sy_sign in (-1, 1):
                prims.append({"kind": "cylinder",
                              "radius_mm": round(H * 0.19, 2),
                              "height_mm": round(W * 0.10, 2),
                              "center_mm": [round(sx_sign * L * 0.31, 2),
                                            round(sy_sign * W * 0.44, 2),
                                            round(H * 0.19, 2)],
                              "axis": "y"})
        return {"op": "sdf_blend", "size_mm": [round(sx, 2), round(sy, 2),
                                               round(sz, 2)],
                "blend_mm": round(H * 0.03, 2), "primitives": prims}

    if key == "drone":
        L, W, H = sx, sy, sz
        prims.append({"kind": "box",
                      "size_mm": [round(L * 0.30, 2), round(W * 0.30, 2),
                                  round(H * 0.40, 2)],
                      "center_mm": [0, 0, round(H * 0.30, 2)],
                      "round_mm": round(H * 0.06, 2)})
        for sx_sign in (-1, 1):
            for sy_sign in (-1, 1):
                cx = sx_sign * L * 0.34
                cy = sy_sign * W * 0.34
                prims.append({"kind": "cylinder",
                              "radius_mm": round(min(L, W) * 0.20, 2),
                              "height_mm": round(H * 0.05, 2),
                              "center_mm": [round(cx, 2), round(cy, 2),
                                            round(H * 0.36, 2)]})
                prims.append({"kind": "box",
                              "size_mm": [round(abs(cx) * 0.9, 2),
                                          round(abs(cy) * 0.10, 2),
                                          round(H * 0.07, 2)],
                              "center_mm": [round(cx * 0.5, 2),
                                            round(cy * 0.5, 2),
                                            round(H * 0.30, 2)]})
        return {"op": "sdf_blend", "size_mm": [round(sx, 2), round(sy, 2),
                                               round(sz, 2)],
                "blend_mm": round(H * 0.04, 2), "primitives": prims}

    # spaceship
    L, W, H = sx, sy, sz
    prims.append({"kind": "ellipsoid",
                  "radii_mm": [round(L * 0.30, 2), round(W * 0.14, 2),
                               round(H * 0.16, 2)],
                  "center_mm": [0, 0, 0]})
    prims.append({"kind": "box",
                  "size_mm": [round(L * 0.30, 2), round(W * 0.10, 2),
                              round(H * 0.22, 2)],
                  "center_mm": [round(-L * 0.26, 2), 0, 0],
                  "round_mm": round(H * 0.04, 2)})
    for sign in (-1, 1):
        prims.append({"kind": "cylinder",
                      "radius_mm": round(W * 0.06, 2),
                      "height_mm": round(L * 0.16, 2),
                      "center_mm": [round(-L * 0.40, 2),
                                    round(sign * W * 0.07, 2), 0],
                      "axis": "x"})
        prims.append({"kind": "box",
                      "size_mm": [round(L * 0.20, 2), round(W * 0.30, 2),
                                  round(H * 0.03, 2)],
                      "center_mm": [round(-L * 0.06, 2),
                                    round(sign * W * 0.20, 2), 0]})
    return {"op": "sdf_blend", "size_mm": [round(sx, 2), round(sy, 2),
                                           round(sz, 2)],
            "blend_mm": round(H * 0.05, 2), "primitives": prims}


def _structure_hint(key, size, ratios, rng):
    sx, sy, sz = (float(v) for v in size)
    prims = []
    if key == "table":
        prims.append({"kind": "box",
                      "size_mm": [round(sx * 0.98, 2), round(sy * 0.98, 2),
                                  round(sz * ratios.get("top_thickness", 0.05), 2)],
                      "center_mm": [0, 0, round(sz * 0.48, 2)]})
        leg = max(sx * 0.04, 12.0)
        for x in (-1, 1):
            for y in (-1, 1):
                prims.append({"kind": "box",
                              "size_mm": [round(leg, 2), round(leg, 2),
                                          round(sz * 0.94, 2)],
                              "center_mm": [round(x * (sx * 0.44 - leg), 2),
                                            round(y * (sy * 0.44 - leg), 2),
                                            round(sz * 0.47, 2)]})
        return {"op": "sdf_blend", "size_mm": [round(sx, 2), round(sy, 2),
                                               round(sz, 2)],
                "blend_mm": round(min(size) * 0.01, 2), "primitives": prims}

    if key == "chair":
        seat_h = sz * float(ratios.get("seat_height", 0.5))
        prims.append({"kind": "box",
                      "size_mm": [round(sx * 0.92, 2), round(sy * 0.86, 2),
                                  round(max(8.0, sz * 0.035), 2)],
                      "center_mm": [0, 0, round(seat_h, 2)]})
        prims.append({"kind": "box",
                      "size_mm": [round(sx * 0.92, 2), round(max(8.0, sy * 0.06), 2),
                                  round(sz * 0.42, 2)],
                      "center_mm": [0, round(-sy * 0.42, 2),
                                    round(seat_h + sz * 0.21, 2)]})
        leg = max(sx * 0.05, 10.0)
        for x in (-1, 1):
            for y in (-1, 1):
                prims.append({"kind": "box",
                              "size_mm": [round(leg, 2), round(leg, 2),
                                          round(seat_h, 2)],
                              "center_mm": [round(x * (sx * 0.40 - leg), 2),
                                            round(y * (sy * 0.38 - leg), 2),
                                            round(seat_h * 0.5, 2)]})
        return {"op": "sdf_blend", "size_mm": [round(sx, 2), round(sy, 2),
                                               round(sz, 2)],
                "blend_mm": round(min(size) * 0.01, 2), "primitives": prims}

    if key == "column":
        base_h = sz * float(ratios.get("base", 0.06))
        cap_h = sz * float(ratios.get("capital", 0.06))
        prims.append({"kind": "cylinder",
                      "radius_mm": round(sx * 0.56, 2),
                      "height_mm": round(base_h, 2),
                      "center_mm": [0, 0, round(base_h * 0.5, 2)]})
        prims.append({"kind": "cylinder",
                      "radius_mm": round(sx * 0.40, 2),
                      "height_mm": round(sz - base_h - cap_h, 2),
                      "center_mm": [0, 0, round(base_h + (sz - base_h - cap_h) * 0.5, 2)]})
        prims.append({"kind": "cylinder",
                      "radius_mm": round(sx * 0.56, 2),
                      "height_mm": round(cap_h, 2),
                      "center_mm": [0, 0, round(sz - cap_h * 0.5, 2)]})
        return {"op": "sdf_blend", "size_mm": [round(sx, 2), round(sy, 2),
                                               round(sz, 2)],
                "blend_mm": round(sx * 0.02, 2), "primitives": prims}

    # building
    floor_h = max(600.0, sz * float(ratios.get("floor_height", 0.06)))
    wall = max(80.0, sx * float(ratios.get("wall", 0.03)))
    prims.append({"kind": "box",
                  "size_mm": [round(sx, 2), round(sy, 2), round(sz, 2)],
                  "center_mm": [0, 0, round(sz * 0.5, 2)]})
    hint = {"op": "sdf_blend", "size_mm": [round(sx * 1.02, 2),
                                           round(sy * 1.02, 2),
                                           round(sz * 1.02, 2)],
            "blend_mm": round(wall * 0.5, 2), "primitives": prims,
            "shell_mm": round(wall, 2)}
    return hint
