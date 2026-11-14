# -*- coding: utf-8 -*-
"""Declarative material presets — pure data, no bpy.

The host owns the vocabulary; the Blender worker owns the node graph.  A
finish references a preset by name, :func:`resolve` expands it into a flat
parameter dict, and the worker writes exactly those values into a Principled
BSDF.  One source of truth, no preset table duplicated inside Blender.

Colors are linear RGB in 0..1.  ``emission_strength`` is a multiplier, not
watts; the worker maps it onto Principled's Emission Strength.

Two layers, one dict
--------------------
A spec may carry a *scalar* layer (the :data:`BASE` BSDF keys, always present)
and an optional *texture* layer (:data:`TEXTURE_KEYS`) that points at an
image-based PBR set.  The texture layer is deliberately **not** merged into
:data:`BASE`: a scalar-only spec therefore resolves to exactly the historical
dict, byte for byte, and ``catalog()``/facts output does not grow texture keys
for materials that have no textures.  Validation and normalisation of the
texture layer lives in :func:`normalize_texture_spec`.
"""

from __future__ import annotations

import copy

__all__ = [
    "BASE", "PRESETS", "DEFAULT", "get", "names", "catalog", "resolve",
    "validate", "is_preset",
    "TEXTURE_KEYS", "UV_MODES", "UV_MODE_ALIASES", "UV_DEFAULTS",
    "TEXTURE_DEFAULTS", "normalize_texture_spec", "canonical_uv_mode",
    "has_texture", "describe_texture",
]

DEFAULT = "ceramic"

#: Every preset is this dict with per-material overrides applied.
BASE = {
    "base_color": (0.62, 0.62, 0.62),
    "metallic": 0.0,
    "roughness": 0.42,
    "ior": 1.50,
    "alpha": 1.0,
    "transmission": 0.0,
    "coat_weight": 0.0,
    "coat_roughness": 0.08,
    "sheen_weight": 0.0,
    "sheen_roughness": 0.30,
    "anisotropic": 0.0,
    "subsurface_weight": 0.0,
    "emission_color": (0.0, 0.0, 0.0),
    "emission_strength": 0.0,
}

#: ``label``/``category`` are display metadata; everything else is a BSDF key.
PRESETS: dict[str, dict] = {
    "ceramic": {
        "label": "月白陶瓷",
        "category": "ceramic",
        "base_color": (0.66, 0.62, 0.53),
        "roughness": 0.40,
        "ior": 1.50,
        "sheen_weight": 0.20,
        "sheen_roughness": 0.30,
        "coat_weight": 0.10,
    },
    "quartz": {
        "label": "白石英",
        "category": "stone",
        "base_color": (0.55, 0.57, 0.59),
        "roughness": 0.32,
        "ior": 1.54,
        "coat_weight": 0.45,
        "coat_roughness": 0.05,
    },
    "glass_frosted": {
        "label": "磨砂玻璃",
        "category": "glass",
        "base_color": (0.010, 0.020, 0.030),
        "roughness": 0.18,
        "ior": 1.45,
        "transmission": 0.92,
        "coat_weight": 0.15,
    },
    "glass_clear": {
        "label": "清玻璃",
        "category": "glass",
        "base_color": (0.006, 0.012, 0.018),
        "roughness": 0.03,
        "ior": 1.49,
        "transmission": 0.98,
    },
    "brushed_steel": {
        "label": "拉丝钢",
        "category": "metal",
        "base_color": (0.36, 0.38, 0.40),
        "metallic": 1.0,
        "roughness": 0.24,
        "anisotropic": 0.45,
        "coat_weight": 0.10,
    },
    "powder_blue": {
        "label": "粉末蓝",
        "category": "painted",
        "base_color": (0.075, 0.145, 0.260),
        "roughness": 0.55,
        "coat_weight": 0.08,
    },
    "gold": {
        "label": "熔金",
        "category": "metal",
        "base_color": (0.830, 0.620, 0.240),
        "metallic": 1.0,
        "roughness": 0.16,
        "anisotropic": 0.30,
        "coat_weight": 0.18,
    },
    "dark": {
        "label": "暗陶瓷",
        "category": "ceramic",
        "base_color": (0.012, 0.014, 0.018),
        "metallic": 0.55,
        "roughness": 0.28,
        "coat_weight": 0.15,
    },
    "clay": {
        "label": "陶土",
        "category": "ceramic",
        "base_color": (0.42, 0.21, 0.14),
        "roughness": 0.82,
        "ior": 1.47,
        "subsurface_weight": 0.05,
    },
    "resin": {
        "label": "琥珀树脂",
        "category": "polymer",
        "base_color": (0.72, 0.40, 0.10),
        "roughness": 0.14,
        "ior": 1.52,
        "transmission": 0.60,
        "coat_weight": 0.30,
        "coat_roughness": 0.04,
    },
    "emissive": {
        "label": "发光体",
        "category": "emissive",
        "base_color": (0.02, 0.02, 0.02),
        "roughness": 0.35,
        "emission_color": (1.00, 0.72, 0.35),
        "emission_strength": 6.0,
    },
    "matte_white": {
        "label": "哑光白",
        "category": "painted",
        "base_color": (0.82, 0.82, 0.80),
        "roughness": 0.72,
    },
}

# Declarative aliases kept for spec/recipe vocabulary.  They are full presets,
# not magic: every name an agent can put in a spec must resolve here.
PRESETS.update({
    "porcelain": {
        "label": "白瓷", "category": "ceramic",
        "base_color": (0.86, 0.85, 0.81), "roughness": 0.30,
        "ior": 1.52, "coat_weight": 0.22, "coat_roughness": 0.04,
    },
    "glass": {
        "label": "透明玻璃", "category": "glass",
        "base_color": (0.02, 0.03, 0.04), "roughness": 0.06,
        "ior": 1.46, "transmission": 0.98, "coat_weight": 0.10,
    },
    "steel": {
        "label": "钢", "category": "metal",
        "base_color": (0.38, 0.39, 0.41), "metallic": 1.0,
        "roughness": 0.30,
    },
    "stainless": {
        "label": "不锈钢", "category": "metal",
        "base_color": (0.46, 0.47, 0.49), "metallic": 1.0,
        "roughness": 0.20,
    },
    "aluminum": {
        "label": "铝", "category": "metal",
        "base_color": (0.62, 0.63, 0.65), "metallic": 1.0,
        "roughness": 0.34,
    },
    "aluminium": {
        "label": "铝", "category": "metal",
        "base_color": (0.62, 0.63, 0.65), "metallic": 1.0,
        "roughness": 0.34,
    },
    "brass": {
        "label": "黄铜", "category": "metal",
        "base_color": (0.58, 0.42, 0.16), "metallic": 1.0,
        "roughness": 0.24,
    },
    "copper": {
        "label": "紫铜", "category": "metal",
        "base_color": (0.55, 0.25, 0.16), "metallic": 1.0,
        "roughness": 0.28,
    },
    "plastic": {
        "label": "塑料", "category": "polymer",
        "base_color": (0.52, 0.54, 0.56), "roughness": 0.48,
        "coat_weight": 0.08,
    },
    "abs": {
        "label": "ABS 白", "category": "polymer",
        "base_color": (0.80, 0.80, 0.78), "roughness": 0.62,
    },
    "abs_white": {
        "label": "ABS 白", "category": "polymer",
        "base_color": (0.80, 0.80, 0.78), "roughness": 0.62,
    },
    "nylon": {
        "label": "尼龙", "category": "polymer",
        "base_color": (0.60, 0.58, 0.50), "roughness": 0.58,
    },
    "wood": {
        "label": "木材", "category": "organic",
        "base_color": (0.32, 0.20, 0.10), "roughness": 0.64,
    },
    "polymer_black": {
        "label": "黑聚合物", "category": "polymer",
        "base_color": (0.016, 0.017, 0.019), "roughness": 0.63,
        "coat_weight": 0.04, "coat_roughness": 0.42,
    },
    "polymer_black_grip": {
        "label": "黑聚合物握把", "category": "polymer",
        "base_color": (0.013, 0.014, 0.016), "roughness": 0.74,
        "coat_weight": 0.02, "coat_roughness": 0.50,
    },
    "nitride_black": {
        "label": "黑氮化钢", "category": "metal",
        "base_color": (0.038, 0.040, 0.045), "metallic": 1.0,
        "roughness": 0.46, "coat_weight": 0.02, "coat_roughness": 0.34,
    },
    "steel_bright": {
        "label": "亮钢", "category": "metal",
        "base_color": (0.30, 0.31, 0.33), "metallic": 1.0,
        "roughness": 0.24,
    },
    "engraving_gray": {
        "label": "刻字灰", "category": "metal",
        "base_color": (0.170, 0.174, 0.184), "metallic": 0.25,
        "roughness": 0.52,
    },
    "paint_white": {
        "label": "白色填漆", "category": "painted",
        "base_color": (0.78, 0.78, 0.75), "roughness": 0.55,
    },
    "powder_steel": {
        "label": "粉末钢", "category": "metal",
        "base_color": (0.16, 0.17, 0.19), "metallic": 0.78,
        "roughness": 0.52, "coat_weight": 0.08,
    },
    "engine_outer": {
        "label": "引擎外壳", "category": "metal",
        "base_color": (0.11, 0.12, 0.14), "metallic": 0.62,
        "roughness": 0.44, "coat_weight": 0.10,
    },
})

# Ocular / soft-tissue presets used by the anatomical assemblies.  Wet
# surfaces carry a coat layer; translucent media use transmission.
PRESETS.update({
    "bio_sclera": {
        "label": "巩膜", "category": "organic",
        "base_color": (0.355, 0.295, 0.235), "roughness": 0.54,
        "ior": 1.38, "coat_weight": 0.30, "coat_roughness": 0.10,
        "subsurface_weight": 0.18,
    },
    "bio_cornea": {
        "label": "角膜", "category": "organic",
        "base_color": (1.0, 1.0, 1.0), "roughness": 0.012,
        "ior": 1.376, "alpha": 0.030, "coat_weight": 1.0, "coat_roughness": 0.01,
    },
    "bio_iris": {
        "label": "虹膜金绿", "category": "organic",
        "base_color": (0.185, 0.088, 0.009), "roughness": 0.50,
        "ior": 1.40, "coat_weight": 0.55, "coat_roughness": 0.05,
        "sheen_weight": 0.25, "sheen_roughness": 0.35,
        "subsurface_weight": 0.10,
    },
    "bio_iris_deep": {
        "label": "虹膜深褐", "category": "organic",
        "base_color": (0.030, 0.012, 0.0025), "roughness": 0.72,
        "ior": 1.40, "coat_weight": 0.04, "sheen_weight": 0.0,
    },
    "bio_pupil": {
        "label": "瞳孔", "category": "organic",
        "base_color": (0.0035, 0.003, 0.003), "roughness": 0.58,
        "ior": 1.50, "coat_weight": 0.04,
    },
    "bio_vitreous": {
        "label": "玻璃体", "category": "organic",
        "base_color": (0.010, 0.012, 0.014), "roughness": 0.10,
        "ior": 1.336, "transmission": 0.55, "coat_weight": 0.05,
    },
    "bio_lens": {
        "label": "晶状体", "category": "organic",
        "base_color": (0.045, 0.055, 0.045), "roughness": 0.01,
        "ior": 1.42, "transmission": 0.97, "coat_weight": 0.10,
    },
    "bio_blood": {
        "label": "动脉血", "category": "organic",
        "base_color": (0.135, 0.0055, 0.0035), "roughness": 0.55,
        "ior": 1.36, "coat_weight": 0.25, "coat_roughness": 0.10,
        "subsurface_weight": 0.12,
    },
    "bio_blood_dark": {
        "label": "静脉血", "category": "organic",
        "base_color": (0.052, 0.005, 0.011), "roughness": 0.55,
        "ior": 1.36, "coat_weight": 0.20,
    },
    "bio_muscle": {
        "label": "眼外肌", "category": "organic",
        "base_color": (0.145, 0.028, 0.022), "roughness": 0.52,
        "ior": 1.40, "coat_weight": 0.35, "sheen_weight": 0.30,
        "subsurface_weight": 0.32,
    },
    "bio_tendon": {
        "label": "肌腱", "category": "organic",
        "base_color": (0.62, 0.56, 0.45), "roughness": 0.46,
        "ior": 1.41, "coat_weight": 0.30, "subsurface_weight": 0.12,
    },
    "bio_bone": {
        "label": "眶骨", "category": "organic",
        "base_color": (0.115, 0.095, 0.068), "roughness": 0.74,
        "ior": 1.52, "coat_weight": 0.12, "subsurface_weight": 0.06,
    },
    "bio_cartilage": {
        "label": "软骨", "category": "organic",
        "base_color": (0.55, 0.52, 0.42), "roughness": 0.40,
        "ior": 1.44, "coat_weight": 0.35,
    },
    "bio_fat": {
        "label": "眶脂", "category": "organic",
        "base_color": (0.38, 0.255, 0.055), "roughness": 0.58,
        "ior": 1.44, "coat_weight": 0.20, "subsurface_weight": 0.40,
    },
    "bio_nerve": {
        "label": "神经", "category": "organic",
        "base_color": (0.66, 0.60, 0.47), "roughness": 0.40,
        "ior": 1.42, "coat_weight": 0.25, "subsurface_weight": 0.18,
    },
    "bio_choroid": {
        "label": "脉络膜", "category": "organic",
        "base_color": (0.17, 0.020, 0.016), "roughness": 0.50,
        "ior": 1.40, "coat_weight": 0.30, "subsurface_weight": 0.20,
    },
    "bio_skin": {
        "label": "鳄皮", "category": "organic",
        "base_color": (0.032, 0.036, 0.024), "roughness": 0.72,
        "ior": 1.45, "coat_weight": 0.22, "coat_roughness": 0.18,
        "sheen_weight": 0.35, "sheen_roughness": 0.40,
        "subsurface_weight": 0.08,
    },
    "bio_skin_dark": {
        "label": "鳄皮暗纹", "category": "organic",
        "base_color": (0.042, 0.046, 0.034), "roughness": 0.62,
        "ior": 1.45, "coat_weight": 0.15, "sheen_weight": 0.25,
    },
    "bio_membrane": {
        "label": "瞬膜", "category": "organic",
        "base_color": (0.45, 0.40, 0.33), "roughness": 0.14,
        "ior": 1.38, "transmission": 0.62, "coat_weight": 0.20,
        "subsurface_weight": 0.15,
    },
    "bio_keratin": {
        "label": "角质鳞", "category": "organic",
        "base_color": (0.135, 0.125, 0.095), "roughness": 0.50,
        "ior": 1.48, "coat_weight": 0.30, "sheen_weight": 0.20,
    },
})


# --------------------------------------------------------------- texture layer
#
# An image-based PBR layer rides *beside* the scalar BSDF keys, never inside
# them.  Keeping it separate is what makes the extension additive: a spec with
# only scalar keys resolves to the same dict it always did, and a consumer that
# only understands scalars can ignore ``TEXTURE_KEYS`` without misreading the
# rest.

#: Spec keys that describe the texture layer.  Accepted by :func:`validate` and
#: passed through :func:`resolve` untouched (normalised, never reinterpreted).
TEXTURE_KEYS = ("texture_set", "uv", "normal_strength", "tint", "roughness_mul")

#: UV projection modes the Blender worker implements.
#:
#: ``auto``          existing UVs if the mesh has a usable layer, else a
#:                   world-scale projection chosen by triangle count.
#: ``existing``      never author UVs; use the first layer (error if absent).
#: ``smart_project`` bpy angle-based unwrap (best for organic/low-poly).
#: ``box``           world-scale planar box (triplanar) projection computed in
#:                   numpy -- the CAD default: no unwrap failure mode, O(n).
#: ``cube``          bpy ``uv.cube_project`` (screen-space cube, no world scale).
#: ``cylinder``      bpy ``uv.cylinder_project``.
#: ``sphere``        bpy ``uv.sphere_project``.
UV_MODES = ("auto", "existing", "smart_project", "box", "cube", "cylinder",
            "sphere")

#: Mode used when ``uv.mode`` is absent.  Mirrors the documented contract.
UV_DEFAULT_MODE = "auto"

#: Accepted spellings that fold onto :data:`UV_MODES`.  They exist so the
#: texture package's own vocabulary (``TextureSet.to_material(uv_mode=...)``
#: defaults to ``"uv"``) and the usual synonyms can be passed without the
#: caller having to know which module coined which word.
UV_MODE_ALIASES = {
    "uv": "existing",
    "none": "existing",
    "keep": "existing",
    "smart": "smart_project",
    "smartproject": "smart_project",
    "smart-project": "smart_project",
    "triplanar": "box",
    "planar": "box",
    "box_world": "box",
    "world": "box",
    "cube_project": "cube",
    "cylindrical": "cylinder",
    "cylinder_project": "cylinder",
    "spherical": "sphere",
    "sphere_project": "sphere",
}


def canonical_uv_mode(value: object) -> str:
    """Fold a mode spelling onto one of :data:`UV_MODES` (``""`` when unknown)."""
    text = str(value).strip().lower() if value is not None else ""
    text = UV_MODE_ALIASES.get(text, text)
    return text if text in UV_MODES else ""


#: ``uv`` defaults.  ``scale_mm=None`` means "inherit the texture set's
#: ``uv_scale_mm``", which keeps the set self-describing instead of forcing
#: every caller to repeat it.
UV_DEFAULTS = {
    "mode": UV_DEFAULT_MODE,
    "scale_mm": None,
    "rotation_deg": 0.0,
    "offset": [0.0, 0.0],
}

_UV_KEYS = ("mode", "scale_mm", "rotation_deg", "offset")

#: Texture-layer defaults for the parameters that scale a PBR set.
TEXTURE_DEFAULTS = {
    "normal_strength": 1.0,
    "tint": [1.0, 1.0, 1.0],
    "roughness_mul": 1.0,
}

#: Clamp windows, so an absurd value degrades to the edge instead of exploding.
_NORMAL_STRENGTH_RANGE = (0.0, 8.0)
_ROUGHNESS_MUL_RANGE = (0.0, 4.0)
_SCALE_MM_RANGE = (1e-3, 1e7)


def _finite(value, key: str, errors: list, path: str):
    """Parse a finite float or record an error and return ``None``."""
    if isinstance(value, bool):
        errors.append({"code": "bad_texture_param", "path": "material.%s" % path,
                       "message": "material.%s must be a number, got a bool"
                                  % path,
                       "fix": "pass a number, e.g. 1.0"})
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        errors.append({"code": "bad_texture_param", "path": "material.%s" % path,
                       "message": "material.%s must be a number, got %r"
                                  % (path, value),
                       "fix": "pass a number, e.g. 1.0"})
        return None
    if out != out or out in (float("inf"), float("-inf")):
        errors.append({"code": "bad_texture_param", "path": "material.%s" % path,
                       "message": "material.%s must be finite, got %r"
                                  % (path, value),
                       "fix": "pass a finite number"})
        return None
    return out


def _clamp(value: float, bounds) -> float:
    lo, hi = bounds
    return lo if value < lo else hi if value > hi else value


def normalize_texture_spec(spec: object) -> tuple[dict, list[dict]]:
    """Extract and normalise the texture layer of a material spec.

    Returns ``(texture, errors)``.  ``texture`` is empty for a scalar-only spec
    — that is what keeps :func:`resolve` byte-identical for old callers — and
    otherwise contains only the keys the caller supplied, plus the defaults the
    omitted ones imply (``uv`` is always filled out when a set is named,
    because every field then carries meaning).

    Nothing here touches the filesystem: a ``texture_set`` id is validated as
    an id, and resolving it to a directory is the backend's job
    (:mod:`atta.blender.textures`).
    """
    errors: list[dict] = []
    if not isinstance(spec, dict):
        return {}, errors

    present = [k for k in TEXTURE_KEYS if k in spec]
    if not present:
        return {}, errors

    out: dict = {}

    set_id = spec.get("texture_set")
    if "texture_set" in spec:
        if set_id is None:
            pass                                  # explicitly cleared
        elif not isinstance(set_id, str) or not set_id.strip():
            errors.append({
                "code": "bad_texture_set", "path": "material.texture_set",
                "message": "material.texture_set must be a non-empty string, "
                           "got %r" % (set_id,),
                "fix": "pass the id of a set under $ATTA_TEXTURES, e.g. "
                       '"tex_ab12cd34"',
            })
        else:
            out["texture_set"] = set_id.strip()

    # A modifier with nothing to modify is a caller mistake, not a no-op: say
    # so instead of dropping the key silently (a dropped ``tint`` looks like a
    # rendering bug three layers down).
    if "texture_set" not in out:
        stray = [k for k in present if k != "texture_set"]
        for key in stray:
            errors.append({
                "code": "texture_without_set", "path": "material.%s" % key,
                "message": "material.%s only applies to an image texture set, "
                           "but no material.texture_set is given" % (key,),
                "fix": "add texture_set, or drop %r for a scalar preset" % (key,),
            })
        return {}, errors

    # A named set implies the whole texture layer: fill every omitted field
    # from its documented default so the worker and the receipt never have to
    # guess, and so a caller only states what it actually cares about.
    if True:
        raw = spec.get("uv")
        uv = dict(UV_DEFAULTS)
        if raw is None:
            pass
        elif not isinstance(raw, dict):
            errors.append({
                "code": "bad_texture_param", "path": "material.uv",
                "message": "material.uv must be an object, got %r"
                           % (type(raw).__name__,),
                "fix": 'pass {"mode": "auto", "scale_mm": 250}',
            })
            raw = None
        if isinstance(raw, dict):
            unknown = [k for k in raw if k not in _UV_KEYS]
            for key in unknown:
                errors.append({
                    "code": "bad_texture_param",
                    "path": "material.uv.%s" % key,
                    "message": "unknown uv parameter %r" % (key,),
                    "fix": "use one of: %s" % ", ".join(_UV_KEYS),
                })
            mode = canonical_uv_mode(raw.get("mode", UV_DEFAULT_MODE))
            if not mode:
                errors.append({
                    "code": "bad_uv_mode", "path": "material.uv.mode",
                    "message": "unknown uv.mode %r" % (raw.get("mode"),),
                    "fix": "use one of: %s" % ", ".join(UV_MODES),
                })
            else:
                uv["mode"] = mode
            if raw.get("scale_mm") is not None:
                scale = _finite(raw.get("scale_mm"), "uv.scale_mm", errors,
                                "uv.scale_mm")
                if scale is not None:
                    if scale <= 0.0:
                        errors.append({
                            "code": "bad_texture_param",
                            "path": "material.uv.scale_mm",
                            "message": "uv.scale_mm must be > 0 (millimetres "
                                       "covered by one texture tile), got %r"
                                       % (raw.get("scale_mm"),),
                            "fix": "pass the physical tile size, e.g. 250",
                        })
                    else:
                        uv["scale_mm"] = _clamp(scale, _SCALE_MM_RANGE)
            if raw.get("rotation_deg") is not None:
                rot = _finite(raw.get("rotation_deg"), "uv.rotation_deg",
                              errors, "uv.rotation_deg")
                if rot is not None:
                    uv["rotation_deg"] = rot
            if raw.get("offset") is not None:
                off = raw.get("offset")
                if (not isinstance(off, (list, tuple)) or len(off) != 2
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               for v in off)):
                    errors.append({
                        "code": "bad_texture_param", "path": "material.uv.offset",
                        "message": "uv.offset must be [u, v], got %r" % (off,),
                        "fix": "pass [0, 0]",
                    })
                else:
                    uv["offset"] = [float(off[0]), float(off[1])]
        out["uv"] = uv

    if True:
        value = _finite(spec.get("normal_strength",
                                 TEXTURE_DEFAULTS["normal_strength"]),
                        "normal_strength", errors, "normal_strength")
        out["normal_strength"] = (TEXTURE_DEFAULTS["normal_strength"]
                                 if value is None
                                 else _clamp(value, _NORMAL_STRENGTH_RANGE))

    if True:
        raw = spec.get("tint", list(TEXTURE_DEFAULTS["tint"]))
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            value = _finite(raw, "tint", errors, "tint")
            out["tint"] = ([value] * 3 if value is not None
                           else list(TEXTURE_DEFAULTS["tint"]))
        elif isinstance(raw, (list, tuple)) and len(raw) == 3:
            values = [_finite(v, "tint[%d]" % i, errors, "tint")
                      for i, v in enumerate(raw)]
            out["tint"] = ([_clamp(v, (0.0, 1.0)) if v is not None else 1.0
                            for v in values])
        else:
            errors.append({
                "code": "bad_texture_param", "path": "material.tint",
                "message": "material.tint must be a number or [r, g, b], got %r"
                           % (raw,),
                "fix": "pass [1, 1, 1] for no tint",
            })
            out["tint"] = list(TEXTURE_DEFAULTS["tint"])

    if True:
        value = _finite(spec.get("roughness_mul",
                                 TEXTURE_DEFAULTS["roughness_mul"]),
                        "roughness_mul", errors, "roughness_mul")
        out["roughness_mul"] = (TEXTURE_DEFAULTS["roughness_mul"]
                                if value is None
                                else _clamp(value, _ROUGHNESS_MUL_RANGE))

    return out, errors


def has_texture(material: object) -> bool:
    """True when a spec (name or dict) carries a texture layer."""
    if isinstance(material, dict):
        return bool(material.get("texture_set"))
    return False


def describe_texture(material: object) -> dict | None:
    """Compact receipt for the texture layer, or ``None`` for scalar specs."""
    texture, _errors = normalize_texture_spec(material)
    if not texture or "texture_set" not in texture:
        return None
    return texture


def is_preset(name: object) -> bool:
    return isinstance(name, str) and name in PRESETS


def names() -> list[str]:
    """All preset names, sorted."""
    return sorted(PRESETS)


def get(name: str | None = None) -> dict:
    """Expand one preset into a full parameter dict.

    Unknown names fall back to :data:`DEFAULT`; use :func:`validate` when the
    caller needs to reject typos instead of silently substituting.
    """
    key = name if is_preset(name) else DEFAULT
    out = copy.deepcopy(BASE)
    out.update({k: v for k, v in PRESETS[key].items()
                if k not in ("label", "category")})
    out["name"] = key
    out["label"] = PRESETS[key].get("label", key)
    out["category"] = PRESETS[key].get("category", "generic")
    return out


def catalog() -> list[dict]:
    """Compact list for ``atta_catalog``: name, label, category, parameters."""
    return [get(n) for n in names()]


def validate(material: object) -> tuple[bool, list[dict]]:
    """Check a preset name (or an explicit spec dict) without expanding it.

    A dict may carry the scalar BSDF keys of :data:`BASE`, the identity keys
    ``name``/``label``/``category``, and the texture-layer keys of
    :data:`TEXTURE_KEYS`.  Anything else is a typo and is rejected — including
    inside ``uv``, where an unknown sub-key would otherwise be silently
    dropped by the worker and produce a default-mode projection.
    """
    if material is None:
        return True, []
    if isinstance(material, dict):
        spec = material
        allowed = set(BASE) | {"name", "label", "category"} | set(TEXTURE_KEYS)
        bad = [k for k in spec if k not in allowed]
        errors = [{"code": "unknown_material_param", "path": "material.%s" % k,
                   "message": "unknown material parameter %r" % k,
                   "fix": "use one of: %s (or a texture key: %s)"
                          % (", ".join(sorted(BASE)),
                             ", ".join(TEXTURE_KEYS))}
                  for k in bad]
        _texture, texture_errors = normalize_texture_spec(spec)
        errors.extend(texture_errors)
        return not errors, errors
    if is_preset(material):
        return True, []
    return False, [{
        "code": "unknown_material",
        "path": "finish.material",
        "message": "unknown material %r" % (material,),
        "fix": "use one of: %s" % ", ".join(names()),
    }]


def resolve(material: object = None) -> dict:
    """Expand a preset name or an inline spec dict into a full parameter dict.

    Inline dicts are merged over :data:`BASE`, so a caller can start from a
    preset with ``{"name": "gold", "roughness": 0.4}``.  Unknown names are
    rejected here; :func:`get` is the lenient variant.

    A dict that names a ``texture_set`` additionally carries the normalised
    texture layer (:data:`TEXTURE_KEYS`).  A dict without one resolves to
    exactly the historical dict — no ``uv``/``tint`` keys are invented — so
    downstream hashes and caches keyed on the resolved spec are unaffected.
    """
    if material is None:
        return get(DEFAULT)
    if isinstance(material, dict):
        base_name = material.get("name", DEFAULT)
        out = get(base_name if is_preset(base_name) else DEFAULT)
        for key, value in material.items():
            if key in ("label", "category"):
                continue
            if key in BASE or key == "name":
                out[key] = value
        texture, errors = normalize_texture_spec(material)
        if errors:
            raise ValueError(errors[0]["message"] + " — " + errors[0]["fix"])
        out.update(texture)
        if not is_preset(base_name):
            out["name"] = "custom"
            out["label"] = "custom"
        return out
    ok, errors = validate(material)
    if not ok:
        raise ValueError(errors[0]["message"] + " — " + errors[0]["fix"])
    return get(str(material))
