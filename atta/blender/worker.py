# -*- coding: utf-8 -*-
"""The bpy half of the Blender backend — runs *inside* Blender.

Started by :mod:`atta.blender.runtime` as::

    blender -b --factory-startup --python atta/blender/worker.py -- <args>

Protocol is JSON lines on stdin/stdout.  Blender is chatty (render progress,
addon notices), so every protocol message is prefixed:

    ATTA_READY                     # once, after startup

    ATTA_JSON:{"id":1,"ok":true}   # one line per response

Anything else on stdout is plain Blender noise; the host logs and skips it.

Requests carry a host-written job directory::

    mesh.npz      verts (n,3 float, metres), faces (m,3 int triangles)
    finish.json   normalised finish (see atta.blender.finish)
    material.json resolved BSDF parameters (see atta.materials)

Ops: ``ping``, ``render``, ``eval``, ``export``, ``shutdown``.

This module deliberately imports nothing from the ``atta`` package: Blender's
bundled Python does not have the repo on ``sys.path`` and a missing import at
startup would be fatal.  Data crosses the boundary as files and JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

#: Camera direction from the model centre; +Z is up.
VIEWS = {
    "iso": (1.0, -1.0, 0.75),
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
    "iso_l": (-1.0, -1.0, 0.75),
    "iso_r": (1.0, 1.0, 0.75),
    "hero": (1.15, -1.55, 0.85),
    "hero_l": (-1.35, -1.35, 0.70),
    "hero_r": (1.35, 1.35, 0.70),
}

#: Fallback only — the host always ships material.json.  Kept tiny on purpose:
#: the preset vocabulary lives in atta/materials.py, never here.
FALLBACK_MATERIAL = {
    "name": "ceramic", "base_color": (0.62, 0.62, 0.62), "metallic": 0.0,
    "roughness": 0.42, "ior": 1.5, "alpha": 1.0, "transmission": 0.0,
    "coat_weight": 0.0, "coat_roughness": 0.08, "sheen_weight": 0.0,
    "sheen_roughness": 0.3, "anisotropic": 0.0, "subsurface_weight": 0.0,
    "emission_color": (0.0, 0.0, 0.0), "emission_strength": 0.0,
}


# --------------------------------------------------------------------- utils

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _err_text(exc: BaseException) -> str:
    """Format an exception without ever raising.

    Blender raises ``RuntimeError`` carrying bytes that are not valid UTF-8
    (locale-encoded operator messages), and ``"%s" % exc`` raises
    ``UnicodeDecodeError`` on those -- which would turn a recoverable operator
    failure into a crashed job.  This is the only formatting path used for
    exceptions in the texture/UV/bake code.
    """
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 - the whole point
        try:
            text = repr(exc)
        except Exception:  # noqa: BLE001
            return "<%s: unprintable>" % type(exc).__name__
    try:
        return text.encode("utf-8", "replace").decode("utf-8")
    except Exception:  # noqa: BLE001
        return "<%s: undecodable>" % type(exc).__name__


def _load(job: Path):
    data = np.load(job / "mesh.npz")
    verts = np.asarray(data["verts"], dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(data["faces"], dtype=np.int64).reshape(-1, 3)
    finish = _read_json(job / "finish.json", {})
    material = _read_json(job / "material.json", dict(FALLBACK_MATERIAL))
    return verts, faces, finish, material


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(default) if isinstance(default, dict) else default


def _bounds(verts: np.ndarray):
    if len(verts) == 0:
        zero = np.zeros(3)
        return zero, zero, zero, zero, 0.0
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    size = hi - lo
    return lo, hi, size, (lo + hi) * 0.5, float(np.linalg.norm(size))


# -------------------------------------------------------------------- scene

def _set_engine(scene) -> tuple[str, str | None]:
    """Pick EEVEE under whichever identifier this Blender uses."""
    try:
        valid = {item.identifier
                 for item in scene.render.bl_rna.properties["engine"].enum_items}
    except Exception:  # noqa: BLE001
        valid = set()
    note = None
    for candidate in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        if candidate in valid or not valid:
            try:
                scene.render.engine = candidate
                return scene.render.engine, note
            except Exception:  # noqa: BLE001
                continue
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
        note = "EEVEE unavailable; fell back to BLENDER_WORKBENCH"
        return "BLENDER_WORKBENCH", note
    except Exception:  # noqa: BLE001
        note = "no render engine could be set; using '%s'" % scene.render.engine
        return scene.render.engine, note


def _reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    engine, note = _set_engine(scene)
    degraded = [note] if note else []
    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:  # noqa: BLE001
        try:
            scene.view_settings.view_transform = "Filmic"
        except Exception:  # noqa: BLE001
            degraded.append("AgX view transform unavailable")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    return scene, engine, degraded


# ------------------------------------------------------------------ textures
#
# A texture set reaches the worker as a resolved block inside the material
# spec, written by the host (atta.blender.textures)::
#
#   material.json {"...scalar BSDF keys...", "texture_set": "tex_a0141249",
#                  "uv": {"mode","scale_mm","rotation_deg","offset"},
#                  "normal_strength", "tint", "roughness_mul",
#                  "texture": {"id","dir","size","tileable","uv_scale_mm",
#                              "maps": {"albedo": {"path","colorspace",...}},
#                              "orm": {...}, "pbr": {...},
#                              "normal_convention","source"}}
#
# Paths are absolute, so the bpy layer never has to know where the texture root
# is.  A hand-written material.json that names only ``texture_set`` still works:
# ``_texture_block`` below re-resolves it with the standard library alone, which
# keeps this module independently runnable (`blender -b --python worker.py`)
# without the ``atta`` package on sys.path.

#: Tile size assumed when neither the spec nor the set states ``uv_scale_mm``.
DEFAULT_UV_SCALE_MM = 500.0

#: Canonical map order, so a receipt never depends on dict insertion order.
TEXTURE_MAP_ORDER = ("albedo", "roughness", "metallic", "normal", "height", "ao",
                     "orm")

#: Colour space per map.  Albedo is display-referred (sRGB); every other map
#: carries measurements and must be read raw, or roughness/normal are wrong.
TEXTURE_COLORSPACES = {
    "albedo": "sRGB", "roughness": "Non-Color", "metallic": "Non-Color",
    "normal": "Non-Color", "height": "Non-Color", "ao": "Non-Color",
    "orm": "Non-Color",
}

#: Name of the UV layer this module authors.  Fixed so a re-run overwrites its
#: own layer instead of piling up ``UVMap.001``.
ATTA_UV_LAYER = "atta_uv"

#: Triangle count above which ``auto`` stops even *trying* ``smart_project``.
#: Measured on this machine (icosphere, 1024 px set): smart_project + island
#: packing costs 390 ms at 5.1k tris and 550 ms at 20.5k tris, while the numpy
#: box projection costs 31 ms and 90 ms -- roughly 6x cheaper, and unlike the
#: unwrap it cannot fail on a boolean CAD solid.  The cap is therefore about
#: paying for an unwrap that a huge triangulation is unlikely to reward; it is
#: not a claim that smart_project is worse, and for a part under the cap the
#: three candidates are compared on measurements rather than on this number.
SMART_PROJECT_MAX_TRIS = 60000

#: Loops above which exact UV-island counting is skipped (label propagation is
#: O(loops) per pass; a 200k-tri part would spend seconds in it).
UV_ISLAND_MAX_LOOPS = 250000


def _worker_texture_root() -> Path:
    """``$ATTA_TEXTURES`` else ``<repo>/textures`` (the frozen contract)."""
    env = os.environ.get("ATTA_TEXTURES")
    if env:
        return Path(env).expanduser()
    # worker.py lives at <repo>/atta/blender/worker.py
    return Path(__file__).resolve().parents[2] / "textures"


def _texture_block_from_dir(directory: Path) -> dict:
    """Read ``set.json`` and return the same block shape the host writes."""
    raw = json.loads((directory / "set.json").read_text(encoding="utf-8"))
    size = raw.get("size") or [0, 0]
    try:
        width, height = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError):
        width = height = 0
    maps = {}
    warnings = []
    for key in TEXTURE_MAP_ORDER:
        record = (raw.get("maps") or {}).get(key)
        if not isinstance(record, dict):
            continue
        filename = str(record.get("file") or ("%s.png" % key))
        path = directory / filename
        if not path.is_file():
            warnings.append("map %r declared but missing: %s" % (key, path))
            continue
        maps[key] = {
            "file": filename,
            "path": str(path.resolve()),
            "colorspace": TEXTURE_COLORSPACES.get(key, "Non-Color"),
            "channels": int(record.get("channels")
                            or (3 if key in ("albedo", "normal", "orm") else 1)),
            "bit_depth": int(record.get("bit_depth") or 8),
        }
    pbr = {"metallic": 0.0, "roughness_mul": 1.0, "normal_strength": 1.0,
           "displacement_mm": 0.0}
    if isinstance(raw.get("pbr"), dict):
        for key in pbr:
            try:
                pbr[key] = float(raw["pbr"][key])
            except (KeyError, TypeError, ValueError):
                pass
    orm = None
    if "orm" in maps:
        orm = {"file": maps["orm"]["file"], "path": maps["orm"]["path"],
               "colorspace": "Non-Color",
               "swizzle": {"R": "ao", "G": "roughness", "B": "metallic"}}
        declared = (raw.get("pbr") or {}).get("orm") if isinstance(
            raw.get("pbr"), dict) else None
        if isinstance(declared, dict):
            for channel in ("R", "G", "B"):
                value = declared.get("channel_%s" % channel.lower())
                if isinstance(value, str):
                    orm["swizzle"][channel] = value
    try:
        uv_scale_mm = float(raw.get("uv_scale_mm") or 0.0) or None
    except (TypeError, ValueError):
        uv_scale_mm = None
    block = {
        "id": str(raw.get("id") or directory.name),
        "dir": str(directory.resolve()),
        "label": str(raw.get("label") or ""),
        "size": [width, height],
        "tileable": bool(raw.get("tileable", True)),
        "uv_scale_mm": uv_scale_mm,
        "maps": maps,
        "orm": orm,
        "pbr": pbr,
        "normal_convention": str(raw.get("normal_convention") or "opengl"),
        "source": "worker-set.json",
        "warnings": warnings,
    }
    return block


def _texture_block(spec: dict):
    """The texture block for a material spec, or ``None``.

    Prefers the host-resolved block; falls back to resolving ``texture_set``
    against the texture root by hand.  Returns ``(block, warnings)``.
    """
    if not isinstance(spec, dict):
        return None, []
    block = spec.get("texture")
    if isinstance(block, dict) and block.get("maps"):
        return dict(block), list(block.get("warnings") or [])
    set_id = spec.get("texture_set")
    if not set_id:
        return None, []
    text = str(set_id).strip()
    root = _worker_texture_root()
    candidates = [Path(text).expanduser(), root / text]
    directory = None
    for candidate in candidates:
        if candidate.name == "set.json" and candidate.is_file():
            directory = candidate.parent
            break
        if candidate.is_dir() and (candidate / "set.json").is_file():
            directory = candidate
            break
    if directory is None:
        try:
            matches = [p for p in sorted(root.iterdir())
                       if p.is_dir() and (p / "set.json").is_file()
                       and (p.name == text or p.name.startswith(text))]
        except OSError:
            matches = []
        if len(matches) == 1:
            directory = matches[0]
    if directory is None:
        return None, ["texture_set %r not found under %s" % (text, root)]
    try:
        return _texture_block_from_dir(directory), []
    except Exception as exc:                                # noqa: BLE001
        return None, ["texture set %r unreadable (%s: %s)"
                      % (text, type(exc).__name__, _err_text(exc))]


# ------------------------------------------------------------ shader graph

def _new_image_node(tree, key: str, record: dict, uv_name: str | None):
    """One Image Texture node, colour space and tiling set from the record."""
    path = record.get("path")
    if not path or not Path(str(path)).is_file():
        return None, "map %r file is gone: %r" % (key, path)
    node = tree.nodes.new("ShaderNodeTexImage")
    node.name = "atta_tex_%s" % key
    node.label = "%s  %s" % (key, Path(str(path)).name)
    try:
        image = bpy.data.images.load(str(path), check_existing=True)
    except Exception as exc:                                # noqa: BLE001
        return None, "map %r could not be loaded (%s)" % (key, exc)
    wanted = str(record.get("colorspace")
                 or TEXTURE_COLORSPACES.get(key, "Non-Color"))
    try:
        image.colorspace_settings.name = wanted
    except Exception:                                       # noqa: BLE001
        pass
    node.image = image
    # A tileable set is meant to repeat; MIRROR would hide seams but silently
    # break the "one tile == scale_mm" contract by reflecting every other tile.
    try:
        node.extension = "REPEAT"
    except Exception:                                       # noqa: BLE001
        pass
    try:
        node.interpolation = "Smart"
    except Exception:                                       # noqa: BLE001
        pass
    node["atta_map"] = key
    node["atta_colorspace"] = str(getattr(image.colorspace_settings, "name", ""))
    node.location = (-760.0, -180.0 * TEXTURE_MAP_ORDER.index(key)
                     if key in TEXTURE_MAP_ORDER else -180.0)
    return node, None


def _mix_multiply(tree, name: str, a_socket, b_socket, location):
    """Colour multiply: ``a * b`` with the factor pinned to 1.0.

    ``MixRGB`` (rather than ``ShaderNodeMix``) because its sockets are stable
    across 2.8-5.x and because the glTF exporter recognises it as a
    baseColorFactor, which is what makes ``tint`` survive a GLB round trip.
    """
    node = tree.nodes.new("ShaderNodeMixRGB")
    node.name = name
    node.blend_type = "MULTIPLY"
    node.inputs[0].default_value = 1.0
    node.location = location
    if a_socket is not None:
        try:
            tree.links.new(a_socket, node.inputs["Color1"])
        except Exception:                                   # noqa: BLE001
            pass
    if b_socket is not None:
        try:
            tree.links.new(b_socket, node.inputs["Color2"])
        except Exception:                                   # noqa: BLE001
            pass
    return node


def _build_texture_graph(tree, bsdf, texture: dict, spec: dict) -> dict:
    """Wire an image-based PBR graph onto an existing Principled BSDF.

    The rule for mixing a preset with a set is the glTF one, because it is the
    only rule that a delivery format can carry back out:

    * ``albedo`` *is* the colour; the preset's ``base_color`` is not blended in
      (a texture with a grey base would wash to grey).  ``tint`` multiplies, and
      ``ao`` multiplies, exactly like glTF ``baseColorFactor`` + occlusion.
    * ``roughness`` map * roughness_mul replaces the scalar (glTF
      ``roughnessFactor``).  Without a map the preset scalar stands.
    * ``metallic`` map * factor replaces the scalar.  With no map the preset's
      ``metallic`` stands, because that is the physical baseline of the preset.
    * ``normal`` -> Normal Map(strength), and ``height`` -> Bump chained after
      it, so a set may carry both without one silently winning.

    Returns a facts dict; a partial failure never aborts the build.
    """
    facts = {
        "ok": True, "nodes": [], "maps_used": [], "maps_skipped": [],
        "warnings": [], "uv_layer": None,
        "tint": list(spec.get("tint") or [1.0, 1.0, 1.0]),
        "roughness_mul": float(spec.get("roughness_mul") or 1.0),
        "normal_strength": float(spec.get("normal_strength") or 1.0),
        "orm_split": bool(texture.get("orm")),
        "displacement": "bump",
    }
    maps = texture.get("maps") or {}
    pbr = dict(texture.get("pbr") or {})
    facts["pbr"] = pbr
    size = texture.get("size") or [0, 0]
    facts["size_px"] = [int(size[0]), int(size[1])] if len(size) >= 2 else [0, 0]
    facts["set_id"] = texture.get("id")

    nodes = {}
    for key in TEXTURE_MAP_ORDER:
        record = maps.get(key)
        if not isinstance(record, dict):
            continue
        node, problem = _new_image_node(tree, key, record, None)
        if node is None:
            if problem:
                facts["warnings"].append(problem)
            facts["maps_skipped"].append(key)
            continue
        nodes[key] = node
        facts["nodes"].append(node.name)
        facts["maps_used"].append(key)

    if not nodes:
        facts["ok"] = False
        facts["warnings"].append("no map file could be loaded; scalar preset kept")
        return facts

    orm = texture.get("orm") or {}
    swizzle = (orm.get("swizzle") if isinstance(orm, dict) else None) or {}
    orm_node = nodes.get("orm")
    # ORM is a packing fallback: if the set ships dedicated roughness/metallic
    # images those win, because they need no channel split and cannot be
    # mis-swizzled by a producer that ordered the channels differently.
    need_split = (orm_node is not None
                  and ("roughness" not in nodes or "metallic" not in nodes
                       or "ao" not in nodes))
    separate = None
    if need_split:
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = "atta_orm_split"
        separate.location = (-420.0, -560.0)
        tree.links.new(orm_node.outputs["Color"], separate.inputs["Color"])
        facts["nodes"].append(separate.name)
        facts["orm_channels"] = {
            "R": swizzle.get("R", "ao"), "G": swizzle.get("G", "roughness"),
            "B": swizzle.get("B", "metallic")}

    def orm_channel(logical: str):
        if separate is None:
            return None
        for channel, name in (("Red", facts["orm_channels"]["R"]),
                              ("Green", facts["orm_channels"]["G"]),
                              ("Blue", facts["orm_channels"]["B"])):
            if name == logical:
                return separate.outputs[channel]
        return None

    # ---- base colour: albedo * tint(-ish) * ao ---------------------------
    base_socket = None
    albedo = nodes.get("albedo")
    if albedo is not None:
        base_socket = albedo.outputs["Color"]
        tint = [float(v) for v in (spec.get("tint") or [1.0, 1.0, 1.0])]
        if any(abs(v - 1.0) > 1e-6 for v in tint):
            mix = _mix_multiply(tree, "atta_tint", base_socket, None,
                                (-380.0, 240.0))
            mix.inputs["Color2"].default_value = (tint[0], tint[1], tint[2], 1.0)
            facts["nodes"].append(mix.name)
            base_socket = mix.outputs["Color"]
        ao_socket = (nodes["ao"].outputs["Color"] if "ao" in nodes
                     else orm_channel("ao"))
        # AO is applied only when asked for (``"apply_ao": true``).  Folding AO
        # into Base Color is commonplace for real-time looks, but it makes the
        # material's albedo no longer an albedo: a DIFFUSE bake then returns
        # albedo*AO, and reloading that set multiplies by AO a second time --
        # measured as a 2.2x darkening on a bake round trip.  The AO map is
        # still delivered in the set for consumers that want it.
        if ao_socket is not None and bool(spec.get("apply_ao")):
            mix = _mix_multiply(tree, "atta_ao", base_socket, ao_socket,
                                (-380.0, 120.0))
            facts["nodes"].append(mix.name)
            base_socket = mix.outputs["Color"]
            facts["ao_applied"] = True
    elif "orm" in nodes:
        # No albedo at all: keep the preset colour.  ORM still drives shading.
        facts["warnings"].append("no albedo map: preset base_color kept")

    if base_socket is not None:
        try:
            tree.links.new(base_socket, bsdf.inputs["Base Color"])
            facts["base_color_from"] = "texture"
        except Exception as exc:                            # noqa: BLE001
            facts["warnings"].append("could not link Base Color (%s)" % (exc,))
    else:
        facts["base_color_from"] = "preset"

    # ---- roughness -------------------------------------------------------
    rough_socket = (nodes["roughness"].outputs["Color"]
                    if "roughness" in nodes else orm_channel("roughness"))
    if rough_socket is not None:
        mul = float(spec.get("roughness_mul") or 1.0) * float(
            pbr.get("roughness_mul") or 1.0)
        target = rough_socket
        if abs(mul - 1.0) > 1e-6:
            math_node = tree.nodes.new("ShaderNodeMath")
            math_node.name = "atta_roughness_mul"
            math_node.operation = "MULTIPLY"
            math_node.inputs[1].default_value = mul
            math_node.location = (-260.0, -40.0)
            tree.links.new(rough_socket, math_node.inputs[0])
            facts["nodes"].append(math_node.name)
            target = math_node.outputs["Value"]
        try:
            tree.links.new(target, bsdf.inputs["Roughness"])
            facts["roughness_from"] = "texture"
            facts["roughness_mul_applied"] = mul
        except Exception as exc:                            # noqa: BLE001
            facts["warnings"].append("could not link Roughness (%s)" % (exc,))
    else:
        facts["roughness_from"] = "preset"

    # ---- metallic --------------------------------------------------------
    metal_socket = (nodes["metallic"].outputs["Color"]
                    if "metallic" in nodes else orm_channel("metallic"))
    if metal_socket is not None:
        # A metallic *map* exists, so the map is the signal.  pbr.metallic is
        # then a factor, and 0.0 means "unstated" (glTF's default factor is
        # 1.0): treating it as a real zero would make every metal a dielectric.
        factor = float(pbr.get("metallic") or 0.0) or 1.0
        target = metal_socket
        if abs(factor - 1.0) > 1e-6:
            math_node = tree.nodes.new("ShaderNodeMath")
            math_node.name = "atta_metallic_mul"
            math_node.operation = "MULTIPLY"
            math_node.inputs[1].default_value = factor
            math_node.location = (-260.0, -140.0)
            tree.links.new(metal_socket, math_node.inputs[0])
            facts["nodes"].append(math_node.name)
            target = math_node.outputs["Value"]
        try:
            tree.links.new(target, bsdf.inputs["Metallic"])
            facts["metallic_from"] = "texture"
            facts["metallic_factor"] = factor
        except Exception as exc:                            # noqa: BLE001
            facts["warnings"].append("could not link Metallic (%s)" % (exc,))
    else:
        facts["metallic_from"] = "preset"

    # ---- normal + height -------------------------------------------------
    normal_out = None
    if "normal" in nodes:
        strength = float(spec.get("normal_strength") or 1.0) * float(
            pbr.get("normal_strength") or 1.0)
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        normal_map.name = "atta_normal_map"
        normal_map.location = (-300.0, -420.0)
        try:
            normal_map.inputs["Strength"].default_value = strength
        except Exception:                                   # noqa: BLE001
            pass
        tree.links.new(nodes["normal"].outputs["Color"],
                       normal_map.inputs["Color"])
        facts["nodes"].append(normal_map.name)
        facts["normal_strength_applied"] = strength
        normal_out = normal_map.outputs["Normal"]
        facts["normal_from"] = "texture"
    else:
        facts["normal_from"] = "preset"

    height = nodes.get("height")
    if height is not None:
        displacement_mm = float(pbr.get("displacement_mm") or 0.0)
        tile_mm = float(spec.get("uv", {}).get("scale_mm")
                        or texture.get("uv_scale_mm")
                        or DEFAULT_UV_SCALE_MM)
        bump = tree.nodes.new("ShaderNodeBump")
        bump.name = "atta_bump"
        bump.location = (-300.0, -620.0)
        # A bump reads the *luminance* of the height map.  Distance is derived
        # from the declared displacement over one tile, so "1 mm of relief over
        # a 250 mm tile" always looks the same regardless of mesh size.
        relief = displacement_mm if displacement_mm > 0.0 else 0.25
        try:
            bump.inputs["Strength"].default_value = 0.6 if displacement_mm <= 0 \
                else 1.0
            # metres: relief millimetres over one tile, i.e. a physical slope.
            bump.inputs["Distance"].default_value = max(relief, 1e-6) / 1000.0
        except Exception:                                   # noqa: BLE001
            pass
        tree.links.new(height.outputs["Color"], bump.inputs["Height"])
        if normal_out is not None:
            tree.links.new(normal_out, bump.inputs["Normal"])
        normal_out = bump.outputs["Normal"]
        facts["nodes"].append(bump.name)
        facts["height_from"] = "bump"
        facts["displacement_mm"] = displacement_mm
        facts["bump_distance_m"] = float(
            getattr(bump.inputs["Distance"], "default_value", 0.0))
        if displacement_mm > 0.0:
            facts["displacement"] = "bump+displacement"
    else:
        facts["height_from"] = "none"

    if normal_out is not None:
        try:
            tree.links.new(normal_out, bsdf.inputs["Normal"])
            facts["normal_linked"] = True
        except Exception as exc:                            # noqa: BLE001
            facts["warnings"].append("could not link Normal (%s)" % (exc,))

    # Displacement: only wired when the set declares a physical relief.  Cycles
    # needs adaptive subdivision for it to be visible; the material method is
    # set so the intent is visible in a saved .blend either way.
    if "height" in nodes and float(pbr.get("displacement_mm") or 0.0) > 0.0:
        try:
            disp = tree.nodes.new("ShaderNodeDisplacement")
            disp.name = "atta_displacement"
            disp.location = (-120.0, -760.0)
            disp.inputs["Scale"].default_value = float(
                pbr.get("displacement_mm") or 0.0) / 1000.0
            tree.links.new(nodes["height"].outputs["Color"],
                           disp.inputs["Height"])
            tree.links.new(disp.outputs["Displacement"],
                           tree.get_output_node("CYCLES").inputs["Displacement"]
                           if tree.get_output_node("CYCLES") is not None
                           else tree.nodes["Material Output"].inputs["Displacement"])
            facts["nodes"].append(disp.name)
        except Exception as exc:                            # noqa: BLE001
            facts["warnings"].append("displacement node not wired (%s)" % _err_text(exc))
    return facts


def _make_material(spec: dict):
    name = str(spec.get("name") or "material")
    mat = bpy.data.materials.new("M_%s" % name)
    try:
        mat.use_nodes = True
    except Exception:  # noqa: BLE001
        pass
    tree = getattr(mat, "node_tree", None)
    bsdf = None
    if tree is not None:
        for node in tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bsdf = node
                break
    if bsdf is None:
        return mat, [], None

    missing = []

    def put(key: str, value) -> None:
        sock = bsdf.inputs.get(key)
        if sock is None:
            missing.append(key)
            return
        try:
            if isinstance(value, (tuple, list)):
                sock.default_value = (*[float(v) for v in value], 1.0)
            else:
                sock.default_value = float(value)
        except Exception:  # noqa: BLE001
            missing.append(key)

    color = spec.get("base_color", (0.62, 0.62, 0.62))
    put("Base Color", color)
    put("Metallic", spec.get("metallic", 0.0))
    put("Roughness", spec.get("roughness", 0.42))
    put("IOR", spec.get("ior", 1.5))
    put("Alpha", spec.get("alpha", 1.0))
    put("Transmission Weight", spec.get("transmission", 0.0))
    put("Coat Weight", spec.get("coat_weight", 0.0))
    put("Coat Roughness", spec.get("coat_roughness", 0.08))
    put("Sheen Weight", spec.get("sheen_weight", 0.0))
    put("Sheen Roughness", spec.get("sheen_roughness", 0.3))
    put("Anisotropic", spec.get("anisotropic", 0.0))
    put("Subsurface Weight", spec.get("subsurface_weight", 0.0))
    put("Emission Color", spec.get("emission_color", (0.0, 0.0, 0.0)))
    put("Emission Strength", spec.get("emission_strength", 0.0))

    try:
        mat.diffuse_color = (*[float(v) for v in color], float(spec.get("alpha", 1.0)))
    except Exception:  # noqa: BLE001
        pass
    if float(spec.get("alpha", 1.0)) < 0.999:
        for attr, value in (("surface_render_method", "BLENDED"),
                            ("blend_method", "BLEND")):
            if hasattr(mat, attr):
                try:
                    setattr(mat, attr, value)
                except Exception:  # noqa: BLE001
                    pass

    # ---- image-based layer, if the spec names one ------------------------
    # A spec without ``texture_set`` never reaches this branch, so a scalar
    # preset produces exactly the graph it always did.
    texture_facts = None
    texture, tex_warnings = _texture_block(spec)
    if texture:
        try:
            texture_facts = _build_texture_graph(tree, bsdf, texture, spec)
        except Exception as exc:  # noqa: BLE001 - never lose the render
            texture_facts = {"ok": False, "nodes": [], "maps_used": [],
                             "warnings": ["texture graph failed: %s: %s"
                                          % (type(exc).__name__, _err_text(exc))]}
        texture_facts = dict(texture_facts)
        texture_facts["warnings"] = list(texture_facts.get("warnings") or []) \
            + list(tex_warnings)
        try:
            mat["atta_texture_id"] = str(texture.get("id") or "")
            mat["atta_maps"] = ",".join(texture_facts.get("maps_used") or [])
        except Exception:  # noqa: BLE001
            pass
    return mat, missing, texture_facts


def _shade_smooth(obj, angle_deg: float) -> str:
    """Smooth-by-angle, with graceful version fallbacks. Returns method used.

    Four eras, all verified against a real Blender where possible:

    * 4.1+  ``object.shade_smooth_by_angle`` bakes sharp edges; no modifier is
      left behind, so nothing downstream has to re-apply it.  Preferred.
    * 4.1   ``object.shade_auto_smooth`` also exists but installs a
      "Smooth by Angle" geometry-nodes modifier, which would then be baked a
      second time by ``export_apply``.  Only a fallback.
    * 2.8x-4.0  ``Mesh.use_auto_smooth`` + ``Mesh.auto_smooth_angle`` is the
      *only* mechanism -- the per-angle operators do not exist yet.  Without
      this branch every face is simply flagged smooth, which silently loses the
      angle threshold and makes hard edges render soft (a real visual bug on
      those versions, not a cosmetic one).
    * anything older still exposes neither, and all-smooth is the honest best
      effort (reported as ``polygon_use_smooth``).
    """
    mesh = obj.data
    for poly in mesh.polygons:
        poly.use_smooth = True
    angle = math.radians(float(angle_deg))
    for op in ("shade_smooth_by_angle", "shade_auto_smooth"):
        func = getattr(bpy.ops.object, op, None)
        if func is None:
            continue
        try:
            func(angle=angle)
            return op
        except Exception:  # noqa: BLE001
            continue
    # 2.8x - 4.0: the legacy auto-smooth flag IS the angle threshold.
    legacy = getattr(mesh, "use_auto_smooth", None)
    if legacy is not None:
        try:
            mesh.use_auto_smooth = True
            if hasattr(mesh, "auto_smooth_angle"):
                mesh.auto_smooth_angle = angle
            return "use_auto_smooth"
        except Exception:  # noqa: BLE001
            pass
    try:
        mesh.set_sharp_from_angle(angle=angle)
        return "set_sharp_from_angle"
    except Exception:  # noqa: BLE001
        pass
    try:
        mesh.shade_smooth()
        return "mesh.shade_smooth"
    except Exception:  # noqa: BLE001
        return "polygon_use_smooth"


def _shade_flat(obj) -> None:
    for poly in obj.data.polygons:
        poly.use_smooth = False
    try:
        bpy.ops.object.shade_flat()
    except Exception:  # noqa: BLE001
        pass


def _apply_modifier(obj, name: str) -> None:
    """Apply a modifier with a context override.

    ``object.modifier_apply`` polls on the *active* object, so it raises
    "context is incorrect" whenever the active object is not the one being
    modified -- which is exactly the situation in a headless worker that
    builds several objects in a row.  ``temp_override`` is 3.2+; older builds
    (and some operator contexts) still need the active object set by hand, so
    both are attempted before giving up.
    """
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    except Exception:  # noqa: BLE001
        pass
    override = getattr(bpy.context, "temp_override", None)
    if override is not None:
        try:
            with override(object=obj, active_object=obj,
                          selected_objects=[obj],
                          selected_editable_objects=[obj]):
                bpy.ops.object.modifier_apply(modifier=name)
            return
        except Exception:  # noqa: BLE001 - fall through to the plain call
            pass
    bpy.ops.object.modifier_apply(modifier=name)


def _build(verts, faces, finish: dict, material_spec: dict):
    """Create the object and apply bevel + smoothing. Returns (obj, info)."""
    scene = bpy.context.scene
    mesh = bpy.data.meshes.new("ATTAMESH")
    raw_verts, raw_faces = len(verts), len(faces)
    mesh.from_pydata([tuple(float(x) for x in v) for v in verts], [],
                     [tuple(int(i) for i in f) for f in faces])
    mesh.validate(verbose=False)
    mesh.update()
    obj = bpy.data.objects.new("ATTAMODEL", mesh)
    scene.collection.objects.link(obj)

    view_layer = bpy.context.view_layer
    for other in list(bpy.context.selected_objects):
        try:
            other.select_set(False)
        except Exception:  # noqa: BLE001
            pass
    view_layer.objects.active = obj
    obj.select_set(True)

    info = {"applied": [], "degraded": [], "smooth_method": None,
            "bevel": None, "material": None, "missing_material_inputs": []}
    if len(mesh.vertices) != raw_verts or len(mesh.polygons) != raw_faces:
        info["degraded"].append(
            "mesh.validate dropped %d verts / %d faces"
            % (raw_verts - len(mesh.vertices), raw_faces - len(mesh.polygons)))

    bevel = finish.get("bevel") or {}
    width = float(bevel.get("width") or 0.0)
    segments = int(bevel.get("segments") or 0)
    if width > 0.0 and segments > 0:
        modifier = obj.modifiers.new("bevel", "BEVEL")
        modifier.width = width
        modifier.segments = segments
        modifier.limit_method = "ANGLE"
        modifier.angle_limit = math.radians(float(bevel.get("angle_deg", 40.0)))
        modifier.use_clamp_overlap = True
        try:
            _apply_modifier(obj, modifier.name)
            info["applied"].append("bevel")
            info["bevel"] = {"width": width, "segments": segments,
                             "angle_deg": float(bevel.get("angle_deg", 40.0))}
        except Exception as exc:  # noqa: BLE001
            info["degraded"].append("bevel modifier left live: %s" % exc)
            info["bevel"] = {"width": width, "segments": segments,
                             "angle_deg": float(bevel.get("angle_deg", 40.0)),
                             "live_modifier": True}

    resolved_spec = material_spec or dict(FALLBACK_MATERIAL)
    material, missing, texture_facts = _make_material(resolved_spec)
    mesh.materials.append(material)
    info["material"] = material.name
    info["missing_material_inputs"] = missing
    info["texture"] = texture_facts
    if texture_facts:
        for warning in texture_facts.get("warnings") or []:
            info["degraded"].append("texture: %s" % (warning,))

    if bool(finish.get("smooth", True)):
        angle = float(finish.get("smooth_angle_deg", 35.0))
        if angle <= 0.0:
            _shade_flat(obj)
            info["smooth_method"] = "flat"
        else:
            info["smooth_method"] = _shade_smooth(obj, angle)
    else:
        _shade_flat(obj)
        info["smooth_method"] = "flat"

    # UVs are authored AFTER the bevel is applied: unwrapping the pre-bevel
    # topology and then adding geometry would leave the new faces unmapped.
    info["uv"] = None
    texture = (resolved_spec or {}).get("texture") if isinstance(resolved_spec,
                                                               dict) else None
    uv_spec = (resolved_spec or {}).get("uv") if isinstance(resolved_spec,
                                                           dict) else None
    needs_uv = bool(texture) or bool(uv_spec and (uv_spec or {}).get("mode")
                                     not in (None, "existing"))
    if needs_uv:
        try:
            info["uv"] = _ensure_uv(obj, uv_spec or {}, texture,
                                    triangle_hint=len(mesh.polygons))
            for warning in info["uv"].get("warnings") or []:
                info["degraded"].append("uv: %s" % (warning,))
        except Exception as exc:  # noqa: BLE001 - a failed unwrap must not
            info["degraded"].append(      # cost the render
                "uv authoring failed: %s: %s" % (type(exc).__name__, _err_text(exc)))
            info["uv"] = {"ok": False, "error": "%s: %s"
                          % (type(exc).__name__, _err_text(exc))}

    view_layer.update()
    return obj, info


def _object_verts(obj) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        return co.reshape(-1, 3).astype(np.float64)
    finally:
        evaluated.to_mesh_clear()


# ---------------------------------------------------------------------- UVs
#
# Why this is not just `bpy.ops.uv.smart_project`
# -----------------------------------------------
# CAD meshes arriving from Atta are triangulated solids with coplanar face
# clusters and hard edges everywhere.  ``smart_project`` splits charts at every
# angle above its threshold, so a -- typical -- mechanical part yields hundreds
# of islands, an unusable texel density, and a bake that bleeds across them.
# The numpy box projection below is O(n), has no failure mode, and is the one
# mode whose ``scale_mm`` is exact by construction rather than by post-hoc
# rescaling.  ``smart_project`` stays available and is what ``auto`` picks for
# small, organic, subdivided meshes where charts are actually large.
#
# Two invariants the measurement code relies on:
#
#   * mesh coordinates are METRES in world space (Atta hands the worker world
#     geometry and the object is linked at identity, scale 1.0) -- so
#     "world space" is the mesh's own coordinate space and no object transform
#     has to be applied.  ``_uv_stats`` reports ``object_scale`` so a caller can
#     see that this assumption held.
#   * ``scale_mm`` means exactly "one texture tile spans scale_mm millimetres".
#     It is enforced by construction for ``box`` and by a measured rescale for
#     the bpy operators, and then *verified* by reading the UVs back.


def _mesh_arrays_loops(mesh):
    """Vertices (m), loop vertex indices, loop-polygon map and polygon normals."""
    n_verts = len(mesh.vertices)
    n_loops = len(mesh.loops)
    n_polys = len(mesh.polygons)
    verts = np.empty(n_verts * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3)
    loop_vert = np.empty(n_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vert)
    loop_total = np.empty(n_polys, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", loop_total)
    loop_start = np.empty(n_polys, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_start)
    normals = np.empty(n_polys * 3, dtype=np.float64)
    mesh.polygons.foreach_get("normal", normals)
    normals = normals.reshape(-1, 3)
    poly_of_loop = np.repeat(np.arange(n_polys, dtype=np.int32), loop_total)
    return verts, loop_vert, poly_of_loop, loop_start, loop_total, normals


def _rot2(u, v, degrees, pivot=(0.5, 0.5)):
    """Rotate UVs about ``pivot`` (a tile-space point), in degrees."""
    if abs(degrees) < 1e-9:
        return u, v
    theta = math.radians(float(degrees))
    ca, sa = math.cos(theta), math.sin(theta)
    du, dv = u - pivot[0], v - pivot[1]
    return (pivot[0] + du * ca - dv * sa, pivot[1] + du * sa + dv * ca)


def _uv_box_world(mesh, tile_mm: float, rotation_deg: float, offset) -> dict:
    """World-scale planar box (triplanar) projection, computed in numpy.

    Each face is projected along its dominant normal axis, so a 90 degree corner
    is still covered at full density -- which is the whole point on CAD parts.
    The 2D basis is flipped on negative-facing groups so that ``u x v`` follows
    the face normal: a mirrored texture on the back of a part is invisible in a
    noise map and glaring in a label, so the projection is orientation-correct
    by construction instead of by hope.
    """
    verts, loop_vert, poly_of_loop, _start, _total, normals = \
        _mesh_arrays_loops(mesh)
    if len(verts) == 0 or len(loop_vert) == 0:
        return {"uvs": np.zeros((0, 2)), "axis_histogram": {},
                "mirrored": {"tris": 0, "area_share": 0.0}}

    axis = np.argmax(np.abs(normals), axis=1)
    sign = np.sign(normals[np.arange(len(normals)), axis])
    sign[sign == 0] = 1.0
    loop_axis = axis[poly_of_loop]
    loop_sign = sign[poly_of_loop]
    pos = verts[loop_vert] * 1000.0                       # metres -> millimetres

    # Basis per axis, chosen so u x v == the face normal.
    u = np.zeros(len(pos))
    v = np.zeros(len(pos))
    for ax, (ui, vi, flip) in {
            0: (1, 2, "u"),        # n=+X: u=Y, v=Z   (Y x Z = X)
            1: (2, 0, "u"),        # n=+Y: u=Z, v=X   (Z x X = Y)
            2: (0, 1, "u"),        # n=+Z: u=X, v=Y   (X x Y = Z)
    }.items():
        mask = loop_axis == ax
        if not mask.any():
            continue
        uu = pos[mask, ui]
        vv = pos[mask, vi]
        negative = mask & (loop_sign < 0)
        if negative.any():
            # Flip u on back-facing groups: preserves handedness, so the map is
            # never mirrored; only the seam at the silhouette changes.
            uu = np.where(loop_sign[mask] < 0, -uu, uu)
        u[mask] = uu
        v[mask] = vv

    tile = float(tile_mm) if tile_mm and tile_mm > 0 else DEFAULT_UV_SCALE_MM
    u = u / tile + float(offset[0])
    v = v / tile + float(offset[1])
    u, v = _rot2(u, v, rotation_deg)
    uvs = np.stack([u, v], axis=1)
    # A mirrored island is one whose UV winding disagrees with its 3D winding.
    # Count them rather than claiming zero by argument.
    mirrored = _count_mirrored_tris(mesh, uvs)
    histogram = {}
    for ax, name in enumerate(("x", "y", "z")):
        count = int((loop_axis == ax).sum())
        if count:
            histogram[name] = count
    return {"uvs": uvs, "axis_histogram": histogram, "mirrored": mirrored}


def _count_mirrored_tris(mesh, uvs) -> int:
    """Triangles whose UV-space winding is opposite to their 3D winding."""
    try:
        mesh.calc_loop_triangles()
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", tris)
        tris = tris.reshape(-1, 3)
    except Exception:                                       # noqa: BLE001
        return -1
    if len(tris) == 0:
        return 0
    verts, loop_vert, _p, _s, _t, _n = _mesh_arrays_loops(mesh)
    uv = uvs[tris]                                          # (t, 3, 2)
    p = verts[loop_vert[tris]]                              # (t, 3, 3)
    du = uv[:, 1] - uv[:, 0]
    dv = uv[:, 2] - uv[:, 0]
    uv_signed = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    cross = np.cross(a, b)
    area = np.linalg.norm(cross, axis=1)
    # 3D winding sign w.r.t. the face normal is always positive for a valid
    # mesh, so the reference is simply "positive UV area".
    # A sliver triangle can have a numerically negative UV area with no visible
    # effect; counting raw triangles would report "5 mirrored tris" for a
    # projection that mirrors nothing.  Weigh by 3D area and report the share.
    mirrored_mask = (uv_signed * area) < 0
    total = float(area.sum())
    return {
        "tris": int(mirrored_mask.sum()),
        "area_share": round(float(area[mirrored_mask].sum() / total), 9)
        if total > 0 else 0.0,
    }


def _uv_stats(mesh, size_px, uv_layer=None) -> dict:
    """Measured texel density and UV coverage, in px/mm.

    This is the number that proves a mapping is sane, so it is measured from the
    stored UVs rather than recomputed from the requested scale: if the
    projection and the intent ever disagree, this reports the disagreement.
    """
    layer = uv_layer or mesh.uv_layers.active
    if layer is None or len(mesh.loops) == 0:
        return {"available": False, "reason": "no UV layer"}
    uvs = np.empty(len(mesh.loops) * 2, dtype=np.float64)
    layer.data.foreach_get("uv", uvs)
    uvs = uvs.reshape(-1, 2)
    try:
        mesh.calc_loop_triangles()
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", tris)
        tris = tris.reshape(-1, 3)
    except Exception:                                       # noqa: BLE001
        return {"available": False, "reason": "no loop triangles"}
    if len(tris) == 0:
        return {"available": False, "reason": "no triangles"}
    verts, loop_vert, _p, _s, _t, _n = _mesh_arrays_loops(mesh)
    p = verts[loop_vert[tris]] * 1000.0                     # mm
    uv = uvs[tris]
    world_area = 0.5 * np.linalg.norm(
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)   # mm^2
    uv_area = 0.5 * np.abs(
        (uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
        - (uv[:, 1, 1] - uv[:, 0, 1]) * (uv[:, 2, 0] - uv[:, 0, 0]))
    valid = world_area > 1e-12
    if not valid.any():
        return {"available": False, "reason": "degenerate geometry"}
    # px per mm = sqrt(uv_area / world_area_mm2) * px_per_uv_unit
    density = np.sqrt(uv_area[valid] / world_area[valid]) * float(size_px)
    weights = world_area[valid]
    order = np.argsort(density)
    density_sorted = density[order]
    cumulative = np.cumsum(weights[order]) / weights.sum()

    def percentile(fraction):
        return float(density_sorted[min(len(density_sorted) - 1,
                                        int(np.searchsorted(cumulative, fraction)))])

    degenerate = int((~valid).sum())
    result = {
        "available": True,
        "layer": layer.name,
        "size_px": int(size_px),
        "texel_density_px_mm": {
            "median": round(float(np.median(density)), 6),
            "area_weighted_mean": round(float((density * weights).sum()
                                              / weights.sum()), 6),
            "p5": round(percentile(0.05), 6),
            "p95": round(percentile(0.95), 6),
            "min": round(float(density.min()), 6),
            "max": round(float(density.max()), 6),
        },
        "uv_area_tiles": round(float(uv_area.sum()), 6),
        "world_area_mm2": round(float(world_area.sum()), 4),
        "tiles_needed": round(float(uv_area.sum()), 4),
        "degenerate_tris": degenerate,
        "nonzero_uv_tris": int(valid.sum()),
    }
    spread = result["texel_density_px_mm"]
    if spread["min"] > 0:
        result["density_ratio_p95_p5"] = round(spread["p95"]
                                               / max(spread["p5"], 1e-9), 4)
    return result


def _uv_overlap(mesh, grid: int = 256) -> dict:
    """Rasterise the UV triangles and measure self-overlap.

    Island count alone cannot tell a clean unwrap from a folded one: a cylinder
    projection reports **one** island on a mug *and* typically folds the handle
    onto the body, which double-covers texels and shows up as a texture that
    appears to jump.  Overlap is the property that actually matters, so it is
    measured by rasterising every UV triangle into a ``grid x grid`` occupancy
    buffer and counting texels covered more than once.

    Returns fractions of the *covered* area, so it is independent of how many
    tiles the mapping needs.
    """
    layer = mesh.uv_layers.active
    if layer is None or len(mesh.loops) == 0:
        return {"available": False, "reason": "no UV layer"}
    uvs = np.empty(len(mesh.loops) * 2, dtype=np.float64)
    layer.data.foreach_get("uv", uvs)
    uvs = uvs.reshape(-1, 2)
    try:
        mesh.calc_loop_triangles()
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", tris)
        tris = tris.reshape(-1, 3)
    except Exception:                                       # noqa: BLE001
        return {"available": False, "reason": "no loop triangles"}
    if len(tris) == 0:
        return {"available": False, "reason": "no triangles"}

    uv = uvs[tris]
    # Fold the tile count into the raster: a mapping that needs 3.2 tiles is
    # rasterised into 3.2 tiles' worth of cells, otherwise tiling would read as
    # a 100% overlap.
    lo = uv.reshape(-1, 2).min(axis=0)
    hi = uv.reshape(-1, 2).max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    # Keep the aspect: one cell == one texel-ish.
    aspect = span[0] / max(span[1], 1e-12)
    gx = int(np.clip(round(grid * math.sqrt(aspect)), 8, 4 * grid))
    gy = int(np.clip(round(grid / math.sqrt(aspect)), 8, 4 * grid))
    coverage = np.zeros((gy, gx), dtype=np.int32)
    # Barycentric rasterisation in cell space, vectorised over a fixed number of
    # interior samples per triangle (3x3 plus the centroid): cheap, and a fold
    # is never a single-sub-sample sliver.
    steps = (0.18, 0.5, 0.82)
    samples = []
    for a_ in steps:
        for b_ in steps:
            if a_ + b_ <= 1.0:
                samples.append((a_, b_))
    samples.append((1.0 / 3.0, 1.0 / 3.0))
    # Per-corner cell coordinates: (triangles, 3 corners).
    cu = (uv[..., 0] - lo[0]) / span[0] * (gx - 1)
    cv = (uv[..., 1] - lo[1]) / span[1] * (gy - 1)
    for a_, b_ in samples:
        c_ = 1.0 - a_ - b_
        px = cu[:, 0] * a_ + cu[:, 1] * b_ + cu[:, 2] * c_
        py = cv[:, 0] * a_ + cv[:, 1] * b_ + cv[:, 2] * c_
        ix = np.clip(np.rint(px).astype(np.int64), 0, gx - 1)
        iy = np.clip(np.rint(py).astype(np.int64), 0, gy - 1)
        np.add.at(coverage, (iy, ix), 1)
    covered = coverage > 0
    covered_cells = int(covered.sum())
    if covered_cells == 0:
        return {"available": False, "reason": "rasterised nothing"}
    # Samples per cell, not per triangle: a cell hit twice by the *same*
    # triangle is still a fold in texel terms, but a large triangle legitimately
    # covers a cell with several of its own samples, so normalise by the median
    # sample count of a cell.
    per_cell = coverage[covered]
    baseline = float(np.median(per_cell)) or 1.0
    overlap_cells = int((per_cell > baseline * 1.5).sum())
    return {
        "available": True,
        "grid": [gx, gy],
        "overlap_fraction": round(overlap_cells / covered_cells, 6),
        "covered_cells": covered_cells,
        "max_samples_per_cell": int(per_cell.max()),
        "median_samples_per_cell": round(baseline, 3),
        "uv_bounds": [round(float(v), 6) for v in lo] + [round(float(v), 6)
                                                        for v in hi],
    }


def _uv_island_count(mesh) -> dict:
    """Exact UV island count by label propagation over identical UV corners.

    Two loops belong to the same island when they share a mesh vertex AND the
    same UV at that vertex; islands are the connected components of that
    relation.  Pure numpy with a convergence cap, because the alternative
    (a python union-find) costs seconds on a 600k-loop part and this number is
    diagnostic, not load bearing.
    """
    layer = mesh.uv_layers.active
    if layer is None or len(mesh.loops) == 0:
        return {"count": None, "reason": "no UV layer"}
    loops = len(mesh.loops)
    if loops > UV_ISLAND_MAX_LOOPS:
        return {"count": None, "reason": "loops %d > cap %d"
                % (loops, UV_ISLAND_MAX_LOOPS)}
    uvs = np.empty(loops * 2, dtype=np.float64)
    layer.data.foreach_get("uv", uvs)
    uvs = uvs.reshape(-1, 2)
    loop_vert = np.empty(loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vert)
    quant = np.round(uvs * 1e6).astype(np.int64)
    corner = loop_vert.astype(np.int64) * 100000000 + (quant[:, 0] + 5000000) \
        * 10000 + (quant[:, 1] + 5000000)
    _unique, corner_id = np.unique(corner, return_inverse=True)
    loop_start = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_start)
    loop_total = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", loop_total)
    # Polygon edges in loop space: (i, i+1) wrapping inside each polygon.
    idx = np.arange(loops, dtype=np.int64)
    poly_of_loop = np.repeat(np.arange(len(loop_total)), loop_total)
    next_idx = idx + 1
    wrap = next_idx >= (loop_start[poly_of_loop] + loop_total[poly_of_loop])
    next_idx = np.where(wrap, loop_start[poly_of_loop], next_idx)
    a = corner_id
    b = corner_id[next_idx]
    labels = np.arange(corner_id.max() + 1, dtype=np.int64)
    converged = False
    passes = 0
    for passes in range(1, 129):
        before = labels.copy()
        np.minimum.at(labels, a, labels[b])
        np.minimum.at(labels, b, labels[a])
        # Path compression towards the representative label.
        labels = labels[labels]
        if np.array_equal(labels, before):
            converged = True
            break
    count = int(len(np.unique(labels)))
    return {"count": count, "converged": converged, "passes": passes,
            "loops": int(loops)}


def _uv_scale_to(mesh, factor: float) -> None:
    """Multiply every UV by ``factor`` in place."""
    layer = mesh.uv_layers.active
    if layer is None or abs(factor - 1.0) < 1e-12:
        return
    uvs = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    layer.data.foreach_get("uv", uvs)
    uvs = (uvs.reshape(-1, 2).astype(np.float64) * float(factor))
    layer.data.foreach_set("uv", uvs.reshape(-1).astype(np.float32))


#: UV mode -> the bpy operator name that performs it.  The modes are the
#: vocabulary callers use; the operators are what Blender calls them, and they
#: do not agree (``uv.mode='cylinder'`` is ``bpy.ops.uv.cylinder_project``).
UV_OPERATORS = {
    "smart_project": "smart_project",
    "cylinder": "cylinder_project",
    "sphere": "sphere_project",
    "cube": "cube_project",
    "unwrap": "unwrap",
}


def _bpy_uv_project(obj, mode: str, **params) -> dict:
    """Run a bpy UV operator with the edit-mode context a worker lacks."""
    operator_name = UV_OPERATORS.get(mode, mode)
    result = {"ok": False, "op": mode, "operator": operator_name}
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
    except Exception as exc:                                # noqa: BLE001
        result["error"] = "could not enter edit mode: %s" % _err_text(exc)
        return result
    try:
        try:
            bpy.ops.mesh.select_all(action="SELECT")
        except Exception:                                   # noqa: BLE001
            pass
        operator = getattr(bpy.ops.uv, operator_name, None)
        if operator is None:
            result["error"] = "bpy.ops.uv.%s is not available on this Blender" \
                % (operator_name,)
            return result
        try:
            operator(**params)
            result["ok"] = True
            result["params"] = params
        except TypeError:
            operator()
            result["ok"] = True
            result["params"] = {}
            result["note"] = "operator rejected the tuned parameters; ran defaults"
    except Exception as exc:                                # noqa: BLE001
        result["error"] = "%s: %s" % (type(exc).__name__, _err_text(exc))
    finally:
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:                                   # noqa: BLE001
            pass
    return result


def _uv_mode_score(obj, mode: str, tile_mm: float, rotation_deg: float,
                   offset, size_px: int) -> dict:
    """Apply one UV mode to ``obj`` and measure how good the result is.

    Used only by ``uv.mode='auto'``.  The caller overwrites the layer with the
    winning mode afterwards, so leaving ``obj`` in this state is intentional.
    """
    mesh = obj.data
    layer = mesh.uv_layers.get(ATTA_UV_LAYER) or mesh.uv_layers.new(name=ATTA_UV_LAYER)
    mesh.uv_layers.active = layer
    try:
        layer.active_render = True
    except Exception:  # noqa: BLE001
        pass
    if mode == "box":
        projection = _uv_box_world(mesh, tile_mm, rotation_deg, offset)
        layer.data.foreach_set(
            "uv", projection["uvs"].reshape(-1).astype(np.float32))
        mesh.update()
        applied = True
    else:
        params = {}
        if mode == "smart_project":
            params = {"angle_limit": math.radians(66.0), "island_margin": 0.02,
                      "correct_aspect": True}
        else:
            params = {"correct_aspect": True}
        result = _bpy_uv_project(obj, mode, **params)
        applied = bool(result.get("ok"))
        if applied:
            layer = mesh.uv_layers.get(ATTA_UV_LAYER) or mesh.uv_layers.active
            if layer is None:
                applied = False
            else:
                mesh.uv_layers.active = layer
                pre = _uv_stats(mesh, size_px, layer)
                if pre.get("available"):
                    current = pre["texel_density_px_mm"]["median"]
                    if current > 0:
                        _uv_scale_to(mesh, (size_px / float(tile_mm)) / current)
                if abs(rotation_deg) > 1e-9 or any(abs(v) > 1e-9
                                                   for v in offset):
                    uvs = np.empty(len(mesh.loops) * 2, dtype=np.float64)
                    layer.data.foreach_get("uv", uvs)
                    uvs = uvs.reshape(-1, 2)
                    u, v = _rot2(uvs[:, 0], uvs[:, 1], rotation_deg)
                    uvs = np.stack([u + float(offset[0]), v + float(offset[1])],
                                   axis=1)
                    layer.data.foreach_set(
                        "uv", uvs.reshape(-1).astype(np.float32))
                mesh.update()
    if not applied:
        return {"mode": mode, "score_ok": False,
                "error": "operator did not run"}
    stats = _uv_stats(mesh, size_px, mesh.uv_layers.active)
    islands = _uv_island_count(mesh)
    overlap = _uv_overlap(mesh)
    spread = (stats.get("density_ratio_p95_p5")
              if stats.get("available") else None)
    overlap_fraction = (overlap.get("overlap_fraction")
                        if overlap.get("available") else None)
    island_count = islands.get("count")
    # A folded unwrap is the worst outcome, then stretching, then seams.
    score = ((1000.0 * (overlap_fraction or 0.0))
             + 10.0 * max(0.0, (spread or 1.0) - 1.0)
             + 0.01 * (island_count or 0))
    return {"mode": mode, "score_ok": True, "score": round(score, 6),
            "overlap_fraction": overlap_fraction,
            "density_spread": spread,
            "islands": island_count,
            "px_per_mm": (stats["texel_density_px_mm"]["median"]
                          if stats.get("available") else None)}


def _safe_name(layer) -> str | None:
    """Read a UV layer's name without trusting the wrapper.

    A stale ``bpy`` wrapper (see the note in ``_ensure_uv``) can surface
    undecodable bytes; a receipt field must never be the thing that crashes a
    job, so this returns ``None`` instead.
    """
    try:
        return str(layer.name)
    except Exception:  # noqa: BLE001
        return None


def _ensure_uv(obj, uv_spec: dict, texture: dict | None,
               triangle_hint: int = 0) -> dict:
    """Guarantee a usable UV layer and report how it was obtained.

    Returns the facts dict that ends up in the job receipt: which mode ran, what
    was actually measured, and whether the requested tile size was achievable.
    """
    mesh = obj.data
    uv_spec = dict(uv_spec or {})
    size = (texture or {}).get("size") or [0, 0]
    size_px = int(size[0]) if len(size) >= 2 and size[0] else 1024
    tile_mm = uv_spec.get("scale_mm") or (texture or {}).get("uv_scale_mm") \
        or DEFAULT_UV_SCALE_MM
    tile_mm = float(tile_mm) if float(tile_mm) > 0 else DEFAULT_UV_SCALE_MM
    rotation = float(uv_spec.get("rotation_deg") or 0.0)
    offset = list(uv_spec.get("offset") or [0.0, 0.0])
    requested = str(uv_spec.get("mode") or "auto").strip().lower()
    aliases = {"uv": "existing", "none": "existing", "keep": "existing",
               "smart": "smart_project", "triplanar": "box", "planar": "box",
               "cylinder_project": "cylinder", "sphere_project": "sphere",
               "cube_project": "cube"}
    requested = aliases.get(requested, requested)

    facts = {
        "requested_mode": requested,
        "mode": None,
        "created": False,
        "layer": None,
        "scale_mm": round(tile_mm, 6),
        "rotation_deg": rotation,
        "offset": offset,
        "size_px": size_px,
        "world_space_assumption": (
            "mesh coordinates are metres in world space; object transform is "
            "identity (Atta links the object unscaled), so no object scale is "
            "applied"),
        "warnings": [],
    }

    existing = [layer.name for layer in mesh.uv_layers]
    facts["existing_layers"] = list(existing)
    has_existing = bool(existing)

    mode = requested
    candidates = []
    if mode == "auto":
        if has_existing:
            mode = "existing"
        else:
            # Measure, do not guess.  Each candidate is applied and scored on
            # the properties that decide whether a texture looks right:
            # self-overlap first (a folded unwrap is a visible defect), then
            # texel-density spread (stretching), then island count (seam cost).
            # Scoring the real mesh beats a shape heuristic, and the loser's
            # numbers are kept in the receipt so the choice is auditable.
            candidates = ["box", "cylinder"]
            if triangle_hint and triangle_hint <= SMART_PROJECT_MAX_TRIS:
                candidates.append("smart_project")
            best = None
            for candidate in candidates:
                try:
                    trial = _uv_mode_score(obj, candidate, tile_mm, rotation,
                                          offset, size_px)
                except Exception as exc:                    # noqa: BLE001
                    facts.setdefault("auto_trials", []).append(
                        {"mode": candidate,
                         "error": "%s: %s" % (type(exc).__name__,
                                              _err_text(exc))})
                    continue
                facts.setdefault("auto_trials", []).append(trial)
                if not trial.get("score_ok"):
                    continue
                if best is None or trial["score"] < best["score"]:
                    best = trial
            mode = (best or {}).get("mode") or "box"
            facts["auto_chosen"] = mode
            if best:
                facts["auto_score"] = best.get("score")
    facts["candidates"] = candidates
    if mode == "existing" and not has_existing:
        facts["warnings"].append(
            "uv.mode='existing' but the mesh has no UV layer; fell back to box")
        mode = "box"
    facts["mode"] = mode

    if mode == "existing":
        layer = mesh.uv_layers.active or mesh.uv_layers[0]
        try:
            layer.active_render = True
        except Exception:                                   # noqa: BLE001
            pass
        facts["layer"] = _safe_name(layer)
        facts["created"] = False
        stats = _uv_stats(mesh, size_px, layer)
        facts["texel"] = stats
        facts["islands"] = _uv_island_count(mesh)
        if stats.get("available"):
            achieved = stats["texel_density_px_mm"]["median"]
            facts["achieved_px_mm"] = achieved
            facts["scale_mm_effective"] = round(
                size_px / achieved, 4) if achieved > 0 else None
        return facts

    if mode == "box":
        # Author the layer ourselves; no edit-mode round trip.
        layer = mesh.uv_layers.get(ATTA_UV_LAYER)
        if layer is None:
            layer = mesh.uv_layers.new(name=ATTA_UV_LAYER)
        mesh.uv_layers.active = layer
        try:
            layer.active_render = True
        except Exception:                                   # noqa: BLE001
            pass
        projection = _uv_box_world(mesh, tile_mm, rotation, offset)
        layer.data.foreach_set(
            "uv", projection["uvs"].reshape(-1).astype(np.float32))
        mesh.update()
        facts["created"] = True
        facts["layer"] = _safe_name(layer)
        facts["axis_histogram"] = projection.get("axis_histogram")
        facts["mirrored"] = projection.get("mirrored")
        facts["projection"] = "world_box_triplanar"
    else:
        layer = mesh.uv_layers.new(name=ATTA_UV_LAYER)
        mesh.uv_layers.active = layer
        try:
            layer.active_render = True
        except Exception:                                   # noqa: BLE001
            pass
        params = {}
        if mode == "smart_project":
            # angle_limit is RADIANS: 66 degrees, the value Blender's UI uses.
            params = {"angle_limit": math.radians(66.0),
                      "island_margin": 0.02, "correct_aspect": True}
        elif mode == "cylinder":
            params = {"correct_aspect": True}
        elif mode == "sphere":
            params = {"correct_aspect": True}
        elif mode == "cube":
            params = {"correct_aspect": True}
        result = _bpy_uv_project(obj, mode, **params)
        facts["operator"] = result
        facts["created"] = True
        # NEVER reuse the ``uv_layers.new()`` reference across an edit-mode
        # operator: entering/leaving edit mode rebuilds the CustomData layers
        # and the old Python wrapper then points at freed memory -- reading
        # ``.name`` off it returns non-UTF-8 garbage (measured: UnicodeDecodeError
        # on ``layer.name``).  Re-fetch by identity from the live mesh.
        layer = mesh.uv_layers.get(ATTA_UV_LAYER) or mesh.uv_layers.active
        if layer is None:
            facts["warnings"].append(
                "uv.%s left no UV layer behind; created a box projection"
                % (mode,))
            layer = mesh.uv_layers.new(name=ATTA_UV_LAYER)
            projection = _uv_box_world(mesh, tile_mm, rotation, offset)
            layer.data.foreach_set(
                "uv", projection["uvs"].reshape(-1).astype(np.float32))
            mesh.update()
            facts["mode"] = "box_fallback"
        mesh.uv_layers.active = layer
        try:
            layer.active_render = True
        except Exception:  # noqa: BLE001
            pass
        facts["layer"] = _safe_name(layer)
        # A bpy unwrap can create its own layer as well as filling ours; a mesh
        # with two UV sets exports TEXCOORD_1 for no reason and confuses a
        # downstream consumer about which set the textures use.
        extra = [l.name for l in mesh.uv_layers
                 if l is not layer and l.name != ATTA_UV_LAYER]
        for name in extra:
            try:
                mesh.uv_layers.remove(mesh.uv_layers[name])
                facts.setdefault("removed_uv_layers", []).append(name)
            except Exception:  # noqa: BLE001
                pass
        if not result.get("ok"):
            facts["warnings"].append(
                "uv.%s failed (%s); fell back to the numpy box projection"
                % (mode, result.get("error")))
            projection = _uv_box_world(mesh, tile_mm, rotation, offset)
            layer.data.foreach_set(
                "uv", projection["uvs"].reshape(-1).astype(np.float32))
            mesh.update()
            facts["mode"] = "box_fallback"
            facts["axis_histogram"] = projection.get("axis_histogram")
            facts["mirrored"] = projection.get("mirrored")
        else:
            # A bpy unwrap is normalised, not world-scaled: rescale to the
            # requested tile size by measurement, then verify the measurement.
            pre = _uv_stats(mesh, size_px, layer)
            if pre.get("available"):
                current = pre["texel_density_px_mm"]["median"]
                if current > 0:
                    target = size_px / float(tile_mm)
                    _uv_scale_to(mesh, target / current)
                    mesh.update()
                    facts["rescaled_from_px_mm"] = round(current, 6)
                    facts["rescale_factor"] = round(target / current, 9)
            # Pack the islands inside the unit square.  Without this a
            # smart_project layout can extend past [0,1] and a bake of it would
            # capture only part of the surface; packing also makes the layout
            # reproducible between a bake and a later reload of the baked set.
            if mode == "smart_project":
                try:
                    bpy.context.view_layer.objects.active = obj
                    obj.select_set(True)
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.uv.select_all(action="SELECT")
                    bpy.ops.uv.pack_islands(rotate=True, margin=0.004)
                    bpy.ops.object.mode_set(mode="OBJECT")
                    mesh.update()
                    layer = mesh.uv_layers.get(ATTA_UV_LAYER) or mesh.uv_layers.active
                    if layer is not None:
                        mesh.uv_layers.active = layer
                        try:
                            layer.active_render = True
                        except Exception:  # noqa: BLE001
                            pass
                    facts["packed_islands"] = True
                except Exception as exc:  # noqa: BLE001
                    try:
                        bpy.ops.object.mode_set(mode="OBJECT")
                    except Exception:  # noqa: BLE001
                        pass
                    facts.setdefault("pack_warning", _err_text(exc))
            if abs(rotation) > 1e-9 or any(abs(v) > 1e-9 for v in offset):
                uvs = np.empty(len(mesh.loops) * 2, dtype=np.float64)
                layer.data.foreach_get("uv", uvs)
                uvs = uvs.reshape(-1, 2)
                u, v = _rot2(uvs[:, 0], uvs[:, 1], rotation)
                uvs = np.stack([u + float(offset[0]), v + float(offset[1])],
                               axis=1)
                layer.data.foreach_set(
                    "uv", uvs.reshape(-1).astype(np.float32))
                mesh.update()

    stats = _uv_stats(mesh, size_px, mesh.uv_layers.active)
    facts["texel"] = stats
    facts["islands"] = _uv_island_count(mesh)
    facts["overlap"] = _uv_overlap(mesh)
    if stats.get("available"):
        achieved = stats["texel_density_px_mm"]["median"]
        facts["achieved_px_mm"] = achieved
        facts["scale_mm_effective"] = (round(size_px / achieved, 4)
                                       if achieved > 0 else None)
        facts["tile_mm_intended"] = round(tile_mm, 6)
        intended = size_px / float(tile_mm)
        if intended > 0 and achieved > 0:
            facts["density_error"] = round(abs(achieved - intended)
                                           / intended, 6)
            # A packed single-tile unwrap and a world-scale tile size are
            # mutually exclusive: packing rescales the whole layout so it fits
            # in [0,1], which sets ONE global texel density from the surface
            # area.  Say so instead of reporting the requested number as if it
            # had been honoured -- the achieved value is the truthful one.
            if facts["density_error"] > 0.05:
                if facts.get("packed_islands"):
                    # Only a packed unwrap *causes* the shortfall; it rescales
                    # the whole layout into [0,1], which sets one global texel
                    # density from the surface area and overrides scale_mm.
                    facts["warnings"].append(
                        "uv.mode=%s packs the unwrap into one UV tile, which "
                        "fixes the texel density at %.4f px/mm (one tile spans "
                        "%.2f mm) instead of the requested %.2f mm; scale_mm is "
                        "a request that island packing overrides -- use a "
                        "world-scale mode (box/cylinder) when a physical tile "
                        "size is required"
                        % (facts.get("mode"), achieved,
                           facts.get("scale_mm_effective") or 0.0, tile_mm))
                else:
                    # A planar/triplanar projection foreshortens on a face that
                    # is not perpendicular to its projection axis -- inherent,
                    # and reported rather than "corrected" by rescaling, which
                    # would silently break the world-scale contract elsewhere.
                    facts["warnings"].append(
                        "uv.mode=%s achieves %.4f px/mm (one tile spans %.2f mm) "
                        "where %.2f mm was requested: a planar projection "
                        "foreshortens on faces oblique to their dominant axis, "
                        "so the median is below the nominal density by %.1f%%"
                        % (facts.get("mode"), achieved,
                           facts.get("scale_mm_effective") or 0.0, tile_mm,
                           100.0 * facts["density_error"]))
    return facts


# ------------------------------------------------------- engine / device presets
#
# Blender 5.3 facts this code depends on (all verified on this machine):
#
#   * ``scene.render.engine`` accepts 'CYCLES', 'BLENDER_EEVEE' (5.x name --
#     'BLENDER_EEVEE_NEXT' is gone), 'BLENDER_WORKBENCH'.
#   * the Cycles addon is not registered under ``--factory-startup``, so
#     ``preferences.addons['cycles']`` raises KeyError until it is enabled.
#   * ``preferences.devices`` is EMPTY until ``refresh_devices()`` runs; skipping
#     it silently renders on CPU while reporting ``cycles.device == 'GPU'``.
#   * HIP is selected with ``compute_device_type = 'HIP'`` (gfx1200 on this box)
#     and ``cycles.device = 'GPU'`` -- there is no 'HIP' value for
#     ``cycles.device``.
#   * the only accepted denoiser here is 'OPENIMAGEDENOISE'.

#: Engines this worker will hand back, mapped to their 5.3 identifiers.
ENGINE_IDS = {
    "cycles": ("CYCLES",),
    "eevee": ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"),
    "workbench": ("BLENDER_WORKBENCH",),
}


def _ensure_cycles_addon() -> str | None:
    """Enable the Cycles addon; returns an error string or ``None``."""
    try:
        import addon_utils
        addon_utils.enable("cycles", default_set=False, persistent=False)
    except Exception as exc:                                # noqa: BLE001
        pass
    try:
        _ = bpy.context.preferences.addons["cycles"].preferences
        return None
    except Exception as exc:                                # noqa: BLE001
        return "Cycles addon unavailable (%s)" % (exc,)


def _available_engines(scene) -> set:
    """Engine identifiers this build actually accepts (probed by assignment)."""
    current = scene.render.engine
    out = set()
    for names in ENGINE_IDS.values():
        for name in names:
            try:
                scene.render.engine = name
                out.add(name)
                break
            except Exception:                               # noqa: BLE001
                continue
    try:
        scene.render.engine = current
    except Exception:                                       # noqa: BLE001
        pass
    return out


def _cycles_devices(wanted_device: str) -> dict:
    """Configure Cycles' compute device.  Returns facts + warnings.

    ``wanted_device``: ``auto`` | ``hip`` | ``cpu``.  ``auto`` prefers a HIP GPU
    and falls back to CPU.  Every failure is reported, never raised: a render on
    the wrong device is a performance fact, not a correctness failure.
    """
    facts = {"requested": wanted_device, "device": "CPU", "device_name": None,
             "compute_device_type": None, "gpu_available": False,
             "warnings": []}
    problem = _ensure_cycles_addon()
    if problem:
        facts["warnings"].append(problem)
        return facts
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except Exception as exc:                                # noqa: BLE001
        facts["warnings"].append("no Cycles preferences (%s)" % (exc,))
        return facts

    types = ["HIP"] if wanted_device in ("hip", "auto") else []
    for device_type in types:
        try:
            prefs.compute_device_type = device_type
            prefs.refresh_devices()        # mandatory: devices is [] before it
        except Exception as exc:                            # noqa: BLE001
            facts["warnings"].append("could not select %s (%s)"
                                     % (device_type, exc))
            continue
        try:
            gpus = [d for d in prefs.devices if d.type not in ("CPU", "NONE")]
        except Exception as exc:                            # noqa: BLE001
            facts["warnings"].append("could not read device list (%s)" % (exc,))
            continue
        facts["devices_seen"] = [{"name": d.name, "type": d.type,
                                  "use": bool(d.use)} for d in prefs.devices]
        if not gpus:
            facts["warnings"].append("%s selected but no GPU was enumerated"
                                     % (device_type,))
            continue
        for d in prefs.devices:
            try:
                d.use = (d.type not in ("CPU", "NONE"))
            except Exception:                               # noqa: BLE001
                pass
        facts.update({"device": "GPU", "device_name": gpus[0].name,
                      "compute_device_type": device_type, "gpu_available": True})
        return facts
    if wanted_device == "hip":
        facts["warnings"].append(
            "device='hip' was requested but no HIP GPU could be enabled; "
            "rendering on CPU")
    return facts


def _apply_engine(scene, settings: dict, samples: int,
                  allow_cycles: bool = True) -> dict:
    """Resolve ``engine``/``device``/``samples``/``denoise`` into a real setup.

    ``auto`` picks Cycles for final-quality renders (>=768 px or >=32 samples)
    and EEVEE for drafts, because a draft is judged for composition and a
    second of EEVEE beats a minute of Cycles for that judgement.  Workbench is
    never chosen by ``auto``: it cannot show a texture at all.
    """
    s = dict(settings or {})
    requested = str(s.get("engine") or "auto").strip().lower()
    facts = {"requested_engine": requested, "engine": None, "device": "CPU",
             "samples": int(samples), "denoise": False, "warnings": [],
             "fallback": None}
    res = max(int(scene.render.resolution_x), int(scene.render.resolution_y))
    if requested == "auto":
        requested = "cycles" if (res >= 768 or int(samples) >= 32) else "eevee"
        facts["auto_picked"] = requested
    if requested not in ENGINE_IDS:
        facts["warnings"].append("unknown engine %r; used cycles"
                                 % (s.get("engine"),))
        requested = "cycles"

    if requested == "cycles" and not allow_cycles:
        requested = "eevee"
        facts["fallback"] = "cycles disallowed for this pass"

    chosen = None
    if requested == "cycles":
        problem = _ensure_cycles_addon()
        if problem:
            facts["warnings"].append(problem)
            requested = "eevee"
            facts["fallback"] = problem
        else:
            for name in ENGINE_IDS["cycles"]:
                try:
                    scene.render.engine = name
                    chosen = name
                    break
                except Exception:                           # noqa: BLE001
                    continue
            if chosen is None:
                facts["warnings"].append("this build refused engine 'CYCLES'")
                requested = "eevee"
                facts["fallback"] = "Cycles not assignable"
    if chosen is None:
        for name in ENGINE_IDS.get(requested, ENGINE_IDS["eevee"]):
            try:
                scene.render.engine = name
                chosen = name
                break
            except Exception:                               # noqa: BLE001
                continue
    if chosen is None:
        facts["warnings"].append("no requested engine could be set; kept %r"
                                 % (scene.render.engine,))
        chosen = scene.render.engine
    facts["engine"] = chosen
    facts["engine_family"] = ("cycles" if chosen == "CYCLES"
                              else "workbench"
                              if chosen == "BLENDER_WORKBENCH" else "eevee")

    denoise = s.get("denoise")
    if denoise is None:
        denoise = facts["engine_family"] == "cycles"
    facts["denoise"] = bool(denoise)

    if facts["engine_family"] == "cycles":
        device_facts = _cycles_devices(str(s.get("device") or "auto")
                                      .strip().lower())
        facts.update({"device": device_facts["device"],
                      "device_name": device_facts.get("device_name"),
                      "compute_device_type": device_facts.get(
                          "compute_device_type"),
                      "gpu_available": device_facts.get("gpu_available"),
                      "devices_seen": device_facts.get("devices_seen")})
        facts["warnings"].extend(device_facts.get("warnings") or [])
        try:
            scene.cycles.device = "GPU" if facts["device"] == "GPU" else "CPU"
        except Exception as exc:                            # noqa: BLE001
            facts["warnings"].append("could not set cycles.device (%s)" % (exc,))
        try:
            scene.cycles.samples = int(samples)
        except Exception:                                   # noqa: BLE001
            pass
        try:
            scene.cycles.use_denoising = bool(denoise)
            if denoise:
                scene.cycles.denoiser = "OPENIMAGEDENOISE"
        except Exception:                                   # noqa: BLE001
            pass
        # A camera ray limit keeps a stray OOM/slow render from hanging a job.
        try:
            scene.cycles.use_adaptive_sampling = True
        except Exception:                                   # noqa: BLE001
            pass
    else:
        eevee = getattr(scene, "eevee", None)
        if eevee is not None:
            for attr, value in (("taa_render_samples", int(samples)),
                                ("use_gtao", True), ("use_raytracing", True),
                                ("use_shadows", True)):
                if hasattr(eevee, attr):
                    try:
                        setattr(eevee, attr, value)
                    except Exception:                       # noqa: BLE001
                        pass
    return facts


def _apply_color_management(scene, settings: dict) -> dict:
    """AgX by default; ``look``/``exposure``/``world_strength`` honoured."""
    s = dict(settings or {})
    facts = {"view_transform": None, "look": None, "exposure": 0.0,
             "warnings": []}
    wanted = str(s.get("view_transform") or "AgX")
    for candidate in (wanted, "AgX", "Filmic", "Standard"):
        try:
            scene.view_settings.view_transform = candidate
            facts["view_transform"] = candidate
            break
        except Exception:                                   # noqa: BLE001
            continue
    else:
        facts["warnings"].append("no view transform could be set")
    look = str(s.get("look") or "None")
    try:
        scene.view_settings.look = look
        facts["look"] = look
    except Exception:                                       # noqa: BLE001
        if look not in ("None", ""):
            facts["warnings"].append("look %r not available on this build"
                                     % (look,))
    if "exposure" in s:
        try:
            scene.view_settings.exposure = float(s.get("exposure") or 0.0)
            facts["exposure"] = float(scene.view_settings.exposure)
        except Exception:                                   # noqa: BLE001
            pass
    return facts


# --------------------------------------------------------------------- bake
#
# The Blender 5.3 trap this code works around, verified on this machine:
# ``scene.render.bake.type`` only accepts ('NORMALS','DISPLACEMENT',
# 'VECTOR_DISPLACEMENT') -- those are the multires ``object.bake_image``
# types.  The real knob for a Cycles surface bake is ``scene.cycles.bake_type``,
# whose enum is the familiar (COMBINED, AO, SHADOW, NORMAL, UV, ROUGHNESS,
# EMIT, DIFFUSE, GLOSSY, TRANSMISSION, ...).  Both are set to the same value so
# the state is consistent for anyone who opens the scene.
#
# Everything else that makes a bake come out black, in the order it happens:
#   * every material on the object needs an Image Texture node whose image is
#     the bake target, selected, and active -- otherwise Cycles has nowhere to
#     write and silently produces nothing;
#   * data maps (normal/roughness/AO/metallic/height) must be Non-Color BEFORE
#     the bake, or the 8-bit write applies an sRGB curve to a measurement;
#   * ``use_clear`` plus a small margin leaves black seams: margin 16 +
#     EXTEND bleed;
#   * engine must be CYCLES (the bake operators are Cycles-only).

#: Bake types this worker implements, mapped to the Cycles bake enum.
BAKE_TYPES = {
    "diffuse": "DIFFUSE",
    "normal": "NORMAL",
    "roughness": "ROUGHNESS",
    "metallic": "EMIT",          # via the emission trick, see EMIT_CHANNELS
    "ao": "AO",
    "combined": "COMBINED",
    "emit": "EMIT",
}

#: Bake types that produce measurements, not colours.
BAKE_DATA_TYPES = ("normal", "roughness", "ao", "metallic")


def _bake_target_node(tree, image, label: str):
    """Make ``image`` the active Image Texture node of one material tree.

    Returns the node.  Any previous bake node is removed first: two candidate
    targets make Cycles pick one by an internal rule, and the wrong image then
    looks like a bake that "did nothing".
    """
    for node in list(tree.nodes):
        if node.type == "TEX_IMAGE" and node.name.startswith("atta_bake_target"):
            tree.nodes.remove(node)
    node = tree.nodes.new("ShaderNodeTexImage")
    node.name = "atta_bake_target_%s" % label
    node.label = "BAKE TARGET %s" % label
    node.image = image
    node.location = (620.0, 420.0)
    for other in tree.nodes:
        try:
            other.select = (other == node)
        except Exception:                                   # noqa: BLE001
            pass
    tree.nodes.active = node
    return node


def _bake_context(objects):
    """The override Cycles' bake operator needs to see its target mesh."""
    active = objects[0]
    return dict(object=active, active_object=active, selected_objects=list(objects),
                selected_editable_objects=list(objects))


def _image_channel_stats(image) -> dict:
    """mean/std/min/max of the baked buffer, plus the non-black fraction.

    Read back from the image datablock (not the PNG) so the numbers describe
    exactly what Cycles wrote, before any 8-bit round trip.
    """
    try:
        width, height = int(image.size[0]), int(image.size[1])
    except Exception:                                       # noqa: BLE001
        return {"width": 0, "height": 0}
    if width <= 0 or height <= 0:
        return {"width": 0, "height": 0}
    buf = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)
    px = buf.reshape(-1, 4)[:, :3]
    stats = {
        "width": width, "height": height,
        "mean": [round(float(v), 6) for v in px.mean(axis=0)],
        "std": [round(float(v), 6) for v in px.std(axis=0)],
        "min": round(float(px.min()), 6),
        "max": round(float(px.max()), 6),
        "nonzero_fraction": round(float((px.max(axis=1) > 1e-6).mean()), 6),
    }
    return stats


#: Channels Cycles has no bake pass for, baked instead through an Emission
#: shader (the standard trick): the target socket's *value* becomes emitted
#: light, which ``type='EMIT'`` bakes exactly.  Without this a delivery set has
#: no metallic map and a reloaded material falls back to the preset's scalar
#: metallic -- measured as a 2.2x brightness error on a mug, because the source
#: was 65% dielectric by texture and the fallback was 100% metal.
EMIT_CHANNELS = {
    # Albedo is baked through the same trick, and this is not cosmetic: Cycles'
    # DIFFUSE colour pass bakes *diffuse reflectance*, which on a metallic
    # surface is base_color * (1 - metallic).  Measured on a mug that is ~30%
    # metallic by texture: the round-tripped render came back uniformly x0.821
    # darker, i.e. exactly the missing metallic share.  Emission of Base Color
    # is the foundation colour itself, with no lighting and no metallic term.
    "diffuse": "Base Color",
    "metallic": "Metallic",
}


def _bake_emit_channel(obj, socket_name: str, image, sample_type: str,
                       margin: int) -> dict:
    """Bake one Principled socket as emission.  Restores the graph afterwards."""
    results = []
    for material in obj.data.materials:
        tree = getattr(material, "node_tree", None)
        if tree is None:
            continue
        bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        output = next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if bsdf is None or output is None:
            continue
        socket = bsdf.inputs.get(socket_name)
        if socket is None:
            results.append({"material": material.name, "ok": False,
                            "error": "no Principled socket %r" % (socket_name,)})
            continue
        saved_links = []
        surface = output.inputs.get("Surface")
        if surface is not None and surface.is_linked:
            link = surface.links[0]
            saved_links.append((link.from_socket, surface))
            tree.links.remove(link)
        emit = tree.nodes.new("ShaderNodeEmission")
        emit.name = "atta_emit_probe"
        emit.location = (300.0, -400.0)
        # Carry the channel value itself, which may be a constant or a link.
        if socket.is_linked:
            source = socket.links[0].from_socket
            tree.links.new(source, emit.inputs["Color"])
        else:
            try:
                value = float(socket.default_value)
            except (TypeError, ValueError):
                value = 0.0
            emit.inputs["Color"].default_value = (value, value, value, 1.0)
        emit.inputs["Strength"].default_value = 1.0
        tree.links.new(emit.outputs["Emission"], surface)
        target = _bake_target_node(tree, image, sample_type)
        try:
            bpy.context.scene.cycles.bake_type = "EMIT"
            with bpy.context.temp_override(**_bake_context([obj])):
                bpy.ops.object.bake(type="EMIT", pass_filter={"NONE"},
                                    margin=margin, use_clear=True)
            results.append({"material": material.name, "ok": True})
        except Exception as exc:                        # noqa: BLE001
            results.append({"material": material.name, "ok": False,
                            "error": "%s: %s" % (type(exc).__name__,
                                                 _err_text(exc))})
        finally:
            try:
                tree.nodes.remove(target)
            except Exception:                           # noqa: BLE001
                pass
            try:
                tree.nodes.remove(emit)
            except Exception:                           # noqa: BLE001
                pass
            for from_socket, to_socket in saved_links:
                try:
                    tree.links.new(from_socket, to_socket)
                except Exception:                       # noqa: BLE001
                    pass
    ok = bool(results) and all(r["ok"] for r in results)
    return {"ok": ok, "materials": results}


def _ensure_bake_uv(obj, req: dict, material_spec: dict) -> dict:
    """Guarantee a UV layout a bake can actually address.

    Requirements: UVs inside ``[0, 1]`` (one image), and no two surface points
    on the same texel (otherwise the last one baked wins).  ``smart_project``
    plus Blender's island packing is the standard way to get that from a CAD
    triangulation, and it is deterministic for a fixed mesh.
    """
    mesh = obj.data
    facts = {"mode": None, "warnings": [], "tiles": None, "overlap": None}
    requested = req.get("uv")
    if isinstance(requested, dict) and requested.get("mode"):
        facts.update(_ensure_uv(obj, requested, material_spec.get("texture"),
                                triangle_hint=len(mesh.polygons)))
        facts["mode"] = "%s (requested)" % facts.get("mode")
        bounds = _uv_bounds(mesh)
        facts["tiles"] = bounds.get("tiles")
        if bounds.get("tiles") and bounds["tiles"] > 1.01:
            facts["warnings"].append(
                "uv.mode=%r spans %.3f tiles, so a bake only captures the first "
                "tile; pass a packed mode (smart_project) for bake coverage"
                % (requested.get("mode"), bounds["tiles"]))
        return facts

    bounds = _uv_bounds(mesh)
    overlap = _uv_overlap(mesh)
    facts["tiles"] = bounds.get("tiles")
    facts["overlap"] = overlap.get("overlap_fraction")
    needs_repack = (not bounds.get("available")
                    or (bounds.get("tiles") or 0) > 1.01
                    or (overlap.get("overlap_fraction") or 0) > 0.02)
    if not needs_repack:
        facts["mode"] = "existing"
        return facts

    try:
        mesh.uv_layers.active = (mesh.uv_layers.get(ATTA_UV_LAYER)
                                or mesh.uv_layers.new(name=ATTA_UV_LAYER))
        result = _bpy_uv_project(obj, "smart_project", angle_limit=math.radians(66.0),
                                 island_margin=0.02, correct_aspect=True)
        if not result.get("ok"):
            facts["warnings"].append("bake unwrap failed (%s)"
                                     % (result.get("error"),))
            return facts
        layer = mesh.uv_layers.get(ATTA_UV_LAYER) or mesh.uv_layers.active
        mesh.uv_layers.active = layer
        try:
            layer.active_render = True
        except Exception:  # noqa: BLE001
            pass
        # Pack islands into [0,1] with a margin, then confirm.
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.select_all(action="SELECT")
            bpy.ops.uv.pack_islands(rotate=True, margin=0.004)
            bpy.ops.object.mode_set(mode="OBJECT")
            facts["packed"] = True
        except Exception as exc:  # noqa: BLE001
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:  # noqa: BLE001
                pass
            facts["warnings"].append("island packing failed (%s)"
                                     % _err_text(exc))
        mesh.update()
        after = _uv_bounds(mesh)
        facts.update({"mode": "smart_project_packed",
                      "tiles": after.get("tiles"),
                      "overlap": (_uv_overlap(mesh) or {}).get("overlap_fraction"),
                      "islands": (_uv_island_count(mesh) or {}).get("count")})
        facts["warnings"].append(
            "bake re-unwrapped the mesh into one UV tile (was %.3f tiles): a "
            "world-scale tiling projection cannot be baked into a single image"
            % (bounds.get("tiles") or 0.0))
    except Exception as exc:  # noqa: BLE001
        facts["warnings"].append("bake unwrap raised (%s)" % _err_text(exc))
    return facts


def _uv_bounds(mesh) -> dict:
    """UV-space extent in tiles, and whether it fits inside one."""
    layer = mesh.uv_layers.active
    if layer is None or len(mesh.loops) == 0:
        return {"available": False}
    uvs = np.empty(len(mesh.loops) * 2, dtype=np.float64)
    layer.data.foreach_get("uv", uvs)
    uvs = uvs.reshape(-1, 2)
    lo = uvs.min(axis=0)
    hi = uvs.max(axis=0)
    return {"available": True,
            "min": [round(float(v), 6) for v in lo],
            "max": [round(float(v), 6) for v in hi],
            "tiles": round(float(max(hi[0] - lo[0], hi[1] - lo[1])), 6),
            "inside_unit_square": bool(lo.min() >= -1e-6 and hi.max() <= 1 + 1e-6)}


class _BakeDone(Exception):
    """Internal control flow: the per-channel baker already wrote the image."""


def _bake(job: Path, req: dict) -> dict:
    """Bake the built+textured object to PNG maps and (optionally) a set.

    ``req`` keys: ``bake`` (list of types) or ``bake_type``; ``size_px``;
    ``margin``; ``samples``; ``device``; ``engine``; ``stem``; ``write_set``
    (+ ``set_label``) to emit an ``atta.texture/v1`` directory.
    """
    verts, faces, finish, material_spec = _load(job)
    scene, base_engine, degraded = _reset()
    settings = dict(req.get("settings") or {})
    for key in ("engine", "device", "samples", "denoise", "look", "exposure",
                "world_strength", "view_transform"):
        if key in req:
            settings[key] = req[key]
    samples = max(1, int(req.get("samples") or 8))
    engine_facts = _apply_engine(scene, dict(settings, engine=settings.get(
        "engine") or "cycles"), samples)
    degraded.extend(engine_facts.get("warnings") or [])
    _apply_color_management(scene, settings)

    obj, build_info = _build(verts, faces, finish, material_spec)
    degraded.extend(build_info["degraded"])

    # A bake writes ONE image covering UV [0,1]^2, so the mapping must be a
    # packed, unique, in-range unwrap.  A world-scale tiling projection (the
    # right thing for rendering a tileable set) spans as many UV units as the
    # object is wide in tiles -- measured at 3.18 tiles for this mug -- and a
    # bake of it silently captures only the first tile, leaving the rest of the
    # surface black.  Enforce a bake layout and say that it happened.
    bake_uv = _ensure_bake_uv(obj, req, material_spec)
    degraded.extend(bake_uv.get("warnings") or [])

    raw = req.get("bake") or req.get("bake_type") or ["diffuse", "normal",
                                                      "roughness", "ao"]
    if isinstance(raw, str):
        raw = [raw]
    wanted = []
    for item in raw:
        key = str(item).strip().lower()
        if key == "albedo":
            key = "diffuse"
        if key == "diagnostic":
            key = "all"
        if key == "all":
            for name in ("diffuse", "normal", "roughness", "metallic", "ao"):
                if name not in wanted:
                    wanted.append(name)
            continue
        if key not in BAKE_TYPES:
            raise ValueError("unknown bake type %r (use %s)"
                             % (item, ", ".join(sorted(BAKE_TYPES))))
        if key not in wanted:
            wanted.append(key)

    size_px = max(16, int(req.get("size_px") or req.get("size") or 1024))
    margin = max(0, int(req.get("margin") if req.get("margin") is not None else 16))

    # AO and COMBINED are lighting-dependent: AO is *the* ambient occlusion, so
    # it needs a world to occlude against, and COMBINED needs the scene lights.
    # Without this an AO bake of an open surface is uniformly zero (measured).
    if any(k in ("ao", "combined") for k in wanted):
        world = bpy.data.worlds.new("AttaBakeWorld")
        world.use_nodes = True
        background = world.node_tree.nodes.get("Background")
        if background is not None:
            background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
            background.inputs[1].default_value = float(
                settings.get("world_strength") or 1.0)
        scene.world = world
    if "combined" in wanted:
        world_verts = _object_verts(obj)
        lo, hi, size, center, diag = _bounds(world_verts)
        _setup_world_and_lights(scene, lo, hi, size, center, diag, settings)
        # The helper may have replaced the world; re-assert it.
        if scene.world is not None:
            background = scene.world.node_tree.nodes.get("Background")
            if background is not None:
                background.inputs[1].default_value = float(
                    settings.get("world_strength") or 1.0)

    outdir = job / "bake"
    outdir.mkdir(parents=True, exist_ok=True)
    bake_settings = scene.render.bake
    bake_settings.margin = margin
    try:
        bake_settings.margin_type = "EXTEND"
    except Exception:                                       # noqa: BLE001
        pass
    try:
        bake_settings.use_clear = True
    except Exception:                                       # noqa: BLE001
        pass
    try:
        bake_settings.normal_space = "TANGENT"
    except Exception:                                       # noqa: BLE001
        pass
    try:
        bake_settings.target = "IMAGE_TEXTURES"
    except Exception:                                       # noqa: BLE001
        pass
    results = []
    for key in wanted:
        cycles_type = BAKE_TYPES[key]
        image_name = "atta_bake_%s" % key
        image = bpy.data.images.get(image_name)
        if image is not None:
            bpy.data.images.remove(image)
        image = bpy.data.images.new(image_name, size_px, size_px, alpha=False)
        is_data = key in BAKE_DATA_TYPES
        try:
            image.colorspace_settings.name = "Non-Color" if is_data else "sRGB"
        except Exception:                                   # noqa: BLE001
            pass
        targets = []
        for material in obj.data.materials:
            tree = getattr(material, "node_tree", None)
            if tree is not None:
                targets.append(_bake_target_node(tree, image, key))
        started = time.perf_counter()
        error = None
        try:
            if key in EMIT_CHANNELS:
                channel = _bake_emit_channel(obj, EMIT_CHANNELS[key], image,
                                             key, margin)
                if not channel.get("ok"):
                    raise RuntimeError("emission-trick bake failed: %s"
                                       % json.dumps(channel.get("materials")))
                raise _BakeDone()
            # Blender 5.3: the surface-bake selector is scene.cycles.bake_type.
            try:
                scene.cycles.bake_type = cycles_type
            except Exception:                               # noqa: BLE001
                pass
            try:
                bake_settings.type = cycles_type
            except Exception:                               # noqa: BLE001
                pass                          # legacy-only property on 5.3
            filters = {"NONE"}
            if key == "diffuse":
                # Kept in sync for anyone inspecting the scene; the actual
                # albedo goes through EMIT_CHANNELS above.
                filters = {"COLOR"}
                bake_settings.use_pass_direct = False
                bake_settings.use_pass_indirect = False
                bake_settings.use_pass_color = True
            with bpy.context.temp_override(**_bake_context([obj])):
                bpy.ops.object.bake(type=cycles_type, pass_filter=filters,
                                    margin=margin, use_clear=True)
        except _BakeDone:
            pass
        except Exception as exc:                            # noqa: BLE001
            error = "%s: %s" % (type(exc).__name__, _err_text(exc))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        for node in targets:
            try:
                node.id_data.nodes.remove(node)
            except Exception:                               # noqa: BLE001
                pass
        if error:
            results.append({"type": key, "ok": False, "error": error,
                            "bake_ms": round(elapsed_ms, 1)})
            degraded.append("bake %r failed: %s" % (key, error))
            continue
        target_path = outdir / ("%s.png" % key)
        try:
            image.file_format = "PNG"
            image.filepath_raw = str(target_path)
            image.save()
            saved = True
        except Exception as exc:                            # noqa: BLE001
            saved = False
            degraded.append("bake %r could not be saved (%s)" % (key, exc))
        stats = _image_channel_stats(image)
        row = {"type": key, "ok": bool(saved), "path": str(target_path),
               "abs_path": str(target_path.resolve()),
               "colorspace": "Non-Color" if is_data else "sRGB",
               "bake_ms": round(elapsed_ms, 1),
               "stats": stats,
               "mean": stats.get("mean"),
               "nonzero_fraction": stats.get("nonzero_fraction")}
        if saved:
            row["bytes"] = target_path.stat().st_size
            row["sha256"] = _sha256(target_path)
            if not stats.get("nonzero_fraction"):
                row["warning"] = "bake wrote an all-black image"
                degraded.append("bake %r wrote an all-black image" % (key,))
        results.append(row)
    set_dir = None
    if bool(req.get("write_set")) and any(r.get("ok") for r in results):
        set_dir = _write_texture_set(outdir, results, req, material_spec,
                                     obj, degraded)
    return {"ok": True, "blender": bpy.app.version_string, "uv": bake_uv,
            "engine": engine_facts.get("engine"), "device": engine_facts.get(
                "device"), "device_name": engine_facts.get("device_name"),
            "samples": samples, "size_px": size_px, "margin": margin,
            "bakes": results, "set_dir": set_dir,
            "finish_applied": build_info, "degraded": degraded,
            "engine_facts": engine_facts, "exact": False}


def _pack_bake_orm(target: Path, maps: dict, metallic: float,
                   degraded: list) -> None:
    """Write orm.png from baked ao/roughness and a constant metallic channel.

    Done through Blender's image API rather than numpy-on-disk so the PNG is
    encoded exactly like the other maps (same bit depth, same colour space
    handling) and the set is internally consistent.
    """
    ao_image = bpy.data.images.load(str(target / maps["ao"]["file"]),
                                    check_existing=True)
    rough_image = bpy.data.images.load(str(target / maps["roughness"]["file"]),
                                       check_existing=True)
    try:
        if tuple(ao_image.size) != tuple(rough_image.size):
            degraded.append("orm packing skipped: ao and roughness differ in size")
            return
        width, height = int(ao_image.size[0]), int(ao_image.size[1])
        count = width * height
        ao_buf = np.empty(count * 4, dtype=np.float32)
        rough_buf = np.empty(count * 4, dtype=np.float32)
        ao_image.pixels.foreach_get(ao_buf)
        rough_image.pixels.foreach_get(rough_buf)
        orm = np.empty(count * 4, dtype=np.float32)
        orm[0::4] = ao_buf[0::4]
        orm[1::4] = rough_buf[0::4]
        orm[2::4] = metallic
        orm[3::4] = 1.0
        packed = bpy.data.images.new("atta_orm", width, height, alpha=False)
        try:
            packed.colorspace_settings.name = "Non-Color"
        except Exception:  # noqa: BLE001
            pass
        packed.pixels.foreach_set(orm)
        dest = target / "orm.png"
        packed.file_format = "PNG"
        packed.filepath_raw = str(dest)
        packed.save()
        bpy.data.images.remove(packed)
        maps["orm"] = {"file": "orm.png", "colorspace": "Non-Color",
                       "channels": 3, "bit_depth": 8, "sha256": _sha256(dest),
                       "bytes": dest.stat().st_size}
    finally:
        for image in (ao_image, rough_image):
            try:
                bpy.data.images.remove(image)
            except Exception:  # noqa: BLE001
                pass


def _write_texture_set(outdir: Path, results, req: dict, material_spec: dict,
                       obj, degraded: list):
    """Assemble the baked PNGs into an ``atta.texture/v1`` directory.

    The set is written by hand (not by ``atta.texture``) because this code runs
    inside Blender where the package is not importable; it therefore writes the
    same JSON the host reads back, and the host is what validates it against
    ``atta.texture`` on the way out.
    """
    try:
        set_id = str(req.get("set_id") or req.get("texture_id") or "baked")
        target = outdir / ("set_%s" % set_id)
        target.mkdir(parents=True, exist_ok=True)
        maps = {}
        # The bake pass is called ``diffuse`` (that is what Cycles calls an
        # albedo bake); the texture-set contract calls the file ``albedo``.  A
        # set that shipped ``diffuse.png`` would reload with no albedo at all,
        # so the rename happens here, once, at the boundary.
        map_name = {"diffuse": "albedo"}
        for row in results:
            if not row.get("ok"):
                continue
            source = Path(row["path"])
            name = map_name.get(row["type"], row["type"])
            dest = target / ("%s.png" % name)
            try:
                dest.write_bytes(source.read_bytes())
            except OSError as exc:
                degraded.append("could not copy %s (%s)" % (name, exc))
                continue
            maps[name] = {
                "file": dest.name,
                "colorspace": row["colorspace"],
                "channels": 3 if row["colorspace"] == "sRGB" else 1,
                "bit_depth": 8,
                "sha256": _sha256(dest),
                "bytes": dest.stat().st_size,
            }
        for name in list(maps):
            if name in ("normal", "orm"):
                maps[name]["channels"] = 3
        # ORM (R=AO, G=roughness, B=metallic) is what glTF consumers expect.
        # A bake has no metallic pass, so B is written as the material's own
        # metallic factor -- stated in the provenance rather than guessed.
        if "ao" in maps and "roughness" in maps:
            try:
                metallic_value = float(material_spec.get("metallic") or 0.0)
                _pack_bake_orm(target, maps, metallic_value, degraded)
            except Exception as exc:                    # noqa: BLE001
                degraded.append("orm packing failed (%s)" % _err_text(exc))
        # With a baked metallic map the map carries the data, so the factor must
        # be 1.0 (our convention: 0.0 means "unstated" -> 1.0).  Without a map,
        # fall back to the source material's scalar metallic and say so.
        has_metallic_map = "metallic" in maps
        pbr = {"metallic": 1.0 if has_metallic_map
               else float(material_spec.get("metallic") or 0.0),
               "roughness_mul": 1.0, "normal_strength": 1.0,
               "displacement_mm": 0.0}
        payload = {
            "schema": "atta.texture/v1",
            "id": set_id,
            "label": str(req.get("set_label") or "baked from %s"
                         % (material_spec.get("name") or "material")),
            "size": [int(req.get("size_px") or 1024),
                     int(req.get("size_px") or 1024)],
            "tileable": False,
            "maps": maps,
            "uv_scale_mm": float((material_spec.get("uv") or {}).get(
                "scale_mm") or DEFAULT_UV_SCALE_MM),
            "pbr": pbr,
            "provenance": {"source": "atta bake", "backend": "cycles",
                           "bake_types": [r["type"] for r in results
                                          if r.get("ok")]},
            "dir": set_id,
        }
        (target / "set.json").write_text(json.dumps(payload, indent=2),
                                         encoding="utf-8")
        return {"dir": str(target), "abs_path": str(target.resolve()),
                "set_json": str(target / "set.json"), "maps": sorted(maps),
                "id": set_id}
    except Exception as exc:                                # noqa: BLE001
        degraded.append("texture set write failed: %s" % (exc,))
        return None


# ---------------------------------------------------------- diagnostic maps

def _mesh_arrays(obj):
    """Final mesh vertices, triangles and vertex normals (modifiers applied)."""
    mesh = obj.data
    n = len(mesh.vertices)
    verts = np.empty(n * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3).astype(np.float64)
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3).astype(np.int64)
    normals = np.empty(n * 3, dtype=np.float32)
    try:
        mesh.vertex_normals.foreach_get("vector", normals)
        normals = normals.reshape(-1, 3).astype(np.float64)
    except Exception:  # noqa: BLE001 - older Blender
        normals = np.zeros_like(verts)
    return verts, tris, normals


def _vertex_curvature(verts, tris, diag):
    """Uniform-Laplacian mean curvature magnitude, scaled to dimensionless."""
    n = len(verts)
    if n == 0 or len(tris) == 0:
        return np.zeros(n)
    i, j, k = tris[:, 0], tris[:, 1], tris[:, 2]
    edges = np.concatenate([
        np.stack([i, j], axis=1), np.stack([j, k], axis=1),
        np.stack([k, i], axis=1),
    ], axis=0)
    d = verts[edges[:, 1]] - verts[edges[:, 0]]
    acc = np.zeros_like(verts)
    cnt = np.zeros(n)
    np.add.at(acc, edges[:, 0], d)
    np.add.at(acc, edges[:, 1], -d)
    np.add.at(cnt, edges[:, 0], 1.0)
    np.add.at(cnt, edges[:, 1], 1.0)
    lap = acc / np.maximum(cnt, 1.0)[:, None]
    area = np.linalg.norm(np.cross(verts[j] - verts[i], verts[k] - verts[i]),
                          axis=1)
    va = np.zeros(n)
    np.add.at(va, i, area / 3.0)
    np.add.at(va, j, area / 3.0)
    np.add.at(va, k, area / 3.0)
    h = np.linalg.norm(lap, axis=1) / np.maximum(2.0 * va, 1e-18)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0)
    return np.log1p(h * max(diag, 1e-9))


def _normalize01(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return v
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def _ramp(values, stops):
    """Piecewise-linear colormap; stops = [(t, (r,g,b)), ...]."""
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    out = np.zeros((v.size, 3))
    for idx in range(len(stops) - 1):
        t0, c0 = stops[idx]
        t1, c1 = stops[idx + 1]
        m = (v >= t0) & (v <= t1)
        if not m.any():
            continue
        f = (v[m] - t0) / max(t1 - t0, 1e-12)
        out[m] = (np.asarray(c0)[None, :] * (1 - f)[:, None]
                  + np.asarray(c1)[None, :] * f[:, None])
    out[v <= stops[0][0]] = stops[0][1]
    out[v >= stops[-1][0]] = stops[-1][1]
    return out


def _id_palette(count):
    import colorsys
    out = []
    for i in range(max(1, count)):
        h = (i * 0.61803398875) % 1.0
        out.append(colorsys.hsv_to_rgb(h, 0.62, 0.92))
    return out


def _display_to_linear(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4)


def _set_vertex_colors(obj, colors):
    mesh = obj.data
    n = len(mesh.vertices)
    if n == 0:
        return
    cols = np.asarray(colors, dtype=np.float64)
    if cols.ndim != 2 or cols.shape[0] != n:
        return
    # Color attributes are scene-linear; the map values are display-referred.
    cols = _display_to_linear(cols)
    rgba = np.concatenate([cols, np.ones((n, 1))], axis=1).astype(np.float32)
    attr = mesh.color_attributes.get("atta_diag")
    if attr is None:
        attr = mesh.color_attributes.new(name="atta_diag", type="FLOAT_COLOR",
                                         domain="POINT")
    attr.data.foreach_set("color", rgba.ravel())
    try:
        mesh.color_attributes.active_color = attr
    except Exception:  # noqa: BLE001
        pass


def _workbench(scene, samples=8):
    scene.render.engine = "BLENDER_WORKBENCH"
    try:
        aa = min((5, 8, 11, 16, 32), key=lambda v: abs(v - int(samples)))
        scene.display.render_aa = str(aa)
    except Exception:  # noqa: BLE001
        pass
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
    except Exception:  # noqa: BLE001
        pass
    sh = scene.display.shading
    sh.light = "FLAT"
    sh.color_type = "VERTEX"
    sh.show_object_outline = False
    sh.show_cavity = False
    sh.show_shadows = False
    sh.show_specular_highlight = False
    sh.show_xray = False
    sh.background_type = "WORLD"


def _reference_grid(reference):
    """Voxel-hash nearest-surface samples for a deviation map (numpy only)."""
    verts, tris, _ = reference
    if len(verts) == 0:
        return None
    if len(tris):
        centroids = verts[tris].mean(axis=1)
        samples = np.concatenate([verts, centroids], axis=0)
    else:
        samples = verts
    lo, hi = samples.min(axis=0), samples.max(axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    cell = max(diag / 64.0, 1e-6)
    keys = np.floor((samples - lo) / cell).astype(np.int64)
    grid = {}
    for key, point in zip(map(tuple, keys), samples):
        grid.setdefault(key, []).append(point)
    return {"grid": grid, "lo": lo, "cell": cell, "samples": samples,
            "diag": diag}


def _nearest_from_grid(ref, points):
    grid, lo, cell = ref["grid"], ref["lo"], ref["cell"]
    out = np.full(len(points), np.inf)
    for idx, point in enumerate(points):
        base = np.floor((point - lo) / cell).astype(np.int64)
        best = np.inf
        for radius in (0, 1, 2):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        bucket = grid.get((int(base[0] + dx),
                                           int(base[1] + dy),
                                           int(base[2] + dz)))
                        if bucket:
                            arr = np.asarray(bucket)
                            d = float(np.min(np.linalg.norm(arr - point, axis=1)))
                            if d < best:
                                best = d
            if best < np.inf:
                break
        if not np.isfinite(best):
            best = float(np.min(np.linalg.norm(ref["samples"] - point, axis=1)))
        out[idx] = best
    return out


# ------------------------------------------------------------------- render

def _setup_world_and_lights(scene, lo, hi, size, center, diag, settings=None):
    """Studio presets.  ``settings`` is optional so the old single-mesh path
    keeps its original look exactly."""
    s = dict(settings or {})
    lighting = str(s.get("lighting") or "studio3")
    background = str(s.get("background") or "transparent")
    diag = max(diag, 1e-3)

    world = bpy.data.worlds.new("AttaWorld")
    world.use_nodes = True
    tree = world.node_tree
    bg = tree.nodes.get("Background")
    world_color = {
        "transparent": (0.020, 0.024, 0.030, 1.0),
        "dark": (0.006, 0.007, 0.009, 1.0),
        "white": (0.900, 0.900, 0.900, 1.0),
        "gray": (0.180, 0.180, 0.180, 1.0),
        "gradient": (0.080, 0.090, 0.105, 1.0),
    }.get(background, (0.045, 0.050, 0.060, 1.0))
    strength = {
        "transparent": 0.45, "dark": 1.0, "white": 0.85,
        "gray": 1.0, "gradient": 1.0,
    }.get(background, 1.0)
    if bg is not None:
        bg.inputs[0].default_value = world_color
        bg.inputs[1].default_value = float(strength)
    if background == "gradient":
        try:
            coord = tree.nodes.new("ShaderNodeTexCoord")
            grad = tree.nodes.new("ShaderNodeTexGradient")
            grad.gradient_type = "SPHERICAL"
            ramp = tree.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].position = 0.10
            ramp.color_ramp.elements[0].color = (0.012, 0.015, 0.020, 1.0)
            ramp.color_ramp.elements[1].position = 0.90
            ramp.color_ramp.elements[1].color = (0.16, 0.18, 0.22, 1.0)
            tree.links.new(coord.outputs["Generated"], grad.inputs["Vector"])
            tree.links.new(grad.outputs["Color"], ramp.inputs["Fac"])
            tree.links.new(ramp.outputs["Color"], bg.inputs["Color"])
        except Exception:  # noqa: BLE001
            pass
    scene.world = world
    scene.render.film_transparent = background == "transparent"

    presets = {
        # Energies are calibrated so a 0.18-linear ceramic reads near Zone V
        # at exposure 0; the old 110 W key was ~6 stops hot.
        "studio3": [((1.2, -1.4, 1.8), 1.72, 1.4),
                    ((-1.6, -1.0, 0.9), 0.70, 1.8),
                    ((-0.2, 1.8, 1.2), 1.41, 1.2)],
        "studio": [((1.4, -1.6, 1.9), 3.44, 1.8),
                   ((-1.7, -1.1, 0.8), 1.09, 2.4),
                   ((-0.3, 1.9, 1.3), 2.50, 1.5)],
        "catalog": [((1.0, -1.8, 2.0), 4.06, 2.6),
                    ((-1.8, -1.2, 1.0), 2.97, 3.0),
                    ((0.0, 1.6, 1.4), 0.78, 2.0)],
        "dark": [((1.5, -1.3, 1.6), 1.25, 1.2),
                 ((-1.6, -0.8, 0.6), 0.28, 2.0),
                 ((-0.2, 1.8, 1.2), 3.75, 0.9)],
        "clay": [((1.2, -1.4, 1.8), 2.50, 2.0),
                 ((-1.6, -1.0, 0.9), 1.72, 2.4),
                 ((-0.2, 1.8, 1.2), 1.09, 1.6)],
        "none": [],
    }
    lights = presets.get(lighting, presets["studio3"])

    def area(name, offset, base_energy, size_factor):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = float(base_energy * (diag / 0.2) ** 2)
        data.size = float(diag * size_factor)
        data.shape = "DISK"
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.location = Vector(center + np.asarray(offset, dtype=np.float64) * diag)
        light.rotation_euler = (Vector(center) - light.location).to_track_quat(
            "-Z", "Y").to_euler()
        return light

    for i, row in enumerate(lights):
        area("atta_light_%d" % i, *row)

    if bool(s.get("ground")):
        z = float(lo[2] - 0.002 * diag)
        bpy.ops.mesh.primitive_plane_add(size=float(diag * 30.0),
                                         location=(float(center[0]),
                                                   float(center[1]), z))
        plane = bpy.context.object
        plane.name = "ATTA_GROUND"
        tone = 0.62 if background == "white" else 0.16
        gm, _missing, _gm_tex = _make_material({
            "name": "ground", "base_color": (tone, tone, tone),
            "metallic": 0.0, "roughness": 0.78, "ior": 1.45,
        })
        plane.data.materials.append(gm)

    camera_data = bpy.data.cameras.new("AttaCam")
    camera_data.lens = float(s.get("lens_mm") or 50.0)
    camera_data.sensor_width = 36.0
    camera_data.clip_start = max(1e-4, diag * 0.01)
    camera_data.clip_end = max(10.0, diag * 200.0)
    if bool(s.get("ortho")):
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = float(diag * 1.45)
    camera = bpy.data.objects.new("AttaCam", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return camera, diag


def _configure_render(scene, settings, samples):
    s = dict(settings or {})
    if "exposure" in s:
        try:
            scene.view_settings.exposure = float(s.get("exposure") or 0.0)
        except Exception:  # noqa: BLE001
            pass
    look = str(s.get("look") or "None")
    try:
        scene.view_settings.look = look
    except Exception:  # noqa: BLE001
        pass
    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        for attr, value in (("taa_render_samples", int(samples)),
                            ("use_gtao", True),
                            ("use_raytracing", True),
                            ("use_shadows", True)):
            if hasattr(eevee, attr):
                try:
                    setattr(eevee, attr, value)
                except Exception:  # noqa: BLE001
                    pass
    if hasattr(scene, "render") and hasattr(scene.render, "use_file_extension"):
        scene.render.use_file_extension = True


def _aim(camera, center, direction, distance):
    vec = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        vec = np.array([0.0, -1.0, 0.0])
        norm = 1.0
    vec = vec / norm
    camera.location = Vector(np.asarray(center) + vec * distance)
    camera.rotation_euler = (Vector(center) - camera.location).to_track_quat(
        "-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def _worst_ndc(lo, hi) -> float:
    from bpy_extras.object_utils import world_to_camera_view

    scene = bpy.context.scene
    camera = scene.camera
    worst = 0.0
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                view = world_to_camera_view(scene, camera, Vector((x, y, z)))
                worst = max(worst, abs(view.x - 0.5) * 2.0, abs(view.y - 0.5) * 2.0)
    return float(worst)


def _png_stats(path: Path) -> dict:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = int(image.size[0]), int(image.size[1])
        if width <= 0 or height <= 0:
            return {"width": 0, "height": 0, "coverage": 0.0,
                    "content_bbox_px": None, "empty_frame": True}
        buf = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(buf)
        alpha = buf.reshape(height, width, 4)[:, :, 3]
        mask = alpha > 0.02
        coverage = float(mask.mean()) if mask.size else 0.0
        bbox = None
        if mask.any():
            ys, xs = np.nonzero(mask)
            # Blender pixel rows run bottom-up; report top-left origin.
            bbox = [int(xs.min()), int(height - 1 - ys.max()),
                    int(xs.max()), int(height - 1 - ys.min())]
        return {"width": width, "height": height,
                "coverage": round(coverage, 6), "content_bbox_px": bbox,
                "empty_frame": coverage < 0.005}
    finally:
        try:
            bpy.data.images.remove(image)
        except Exception:  # noqa: BLE001
            pass


def _render(job: Path, req: dict):
    verts, faces, finish, material_spec = _load(job)
    scene, engine, degraded = _reset()
    res = max(16, int(req.get("res") or 512))
    samples = max(1, int(req.get("samples") or 8))
    obj, build_info = _build(verts, faces, finish, material_spec)
    degraded.extend(build_info["degraded"])

    scene.render.resolution_x = res
    scene.render.resolution_y = res
    # Engine/device selection is a preset here, not a hardcode: ``auto`` keeps
    # the historical EEVEE draft behaviour, and cycles/hip is opt-in.
    engine_facts = _apply_engine(scene, dict(req), samples)
    degraded.extend(engine_facts.get("warnings") or [])
    engine = engine_facts.get("engine") or engine
    cm_facts = _apply_color_management(scene, dict(req))
    degraded.extend(cm_facts.get("warnings") or [])
    eevee = getattr(scene, "eevee", None)
    if eevee is not None and float((material_spec or {}).get(
            "transmission", 0.0) or 0.0) > 0.05:
        for attr in ("use_raytracing",):
            if hasattr(eevee, attr):
                try:
                    setattr(eevee, attr, True)
                except Exception:  # noqa: BLE001
                    pass

    world_verts = _object_verts(obj)
    lo, hi, size, center, diag = _bounds(world_verts)
    camera, diag = _setup_world_and_lights(scene, lo, hi, size, center, diag)

    outdir = job / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    images = []
    for view in [str(v) for v in (req.get("views") or ["iso"])]:
        direction = VIEWS.get(view)
        if direction is None:
            raise ValueError("unknown view %r (known: %s)"
                             % (view, ", ".join(sorted(VIEWS))))
        distance = max(diag * 1.9, 0.02)
        ndc = 0.0
        for _ in range(6):
            _aim(camera, center, direction, distance)
            ndc = _worst_ndc(lo, hi) if len(world_verts) else 0.0
            if ndc <= 0.92:
                break
            distance *= max(1.02, ndc / 0.85)
        _aim(camera, center, direction, distance)

        target = outdir / ("%s.png" % view)
        scene.render.filepath = str(target)
        started = time.perf_counter()
        render_error = None
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as exc:  # noqa: BLE001 - degrade once, then report
            if not str(engine).startswith("BLENDER_WORKBENCH"):
                degraded.append("render failed on %s (%s); retried with "
                                "BLENDER_WORKBENCH" % (engine, exc))
                try:
                    scene.render.engine = "BLENDER_WORKBENCH"
                    engine = "BLENDER_WORKBENCH"
                    bpy.ops.render.render(write_still=True)
                except Exception as retry_exc:  # noqa: BLE001
                    render_error = retry_exc
            else:
                render_error = exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if render_error is not None:
            raise render_error
        stats = _png_stats(target)
        images.append({
            "view": view,
            "path": str(target),
            "abs_path": str(target.resolve()),
            "width": stats["width"],
            "height": stats["height"],
            "coverage": stats["coverage"],
            "content_bbox_px": stats["content_bbox_px"],
            "worst_ndc": round(float(ndc), 4),
            "empty_frame": bool(stats["empty_frame"]),
            "render_ms": round(elapsed_ms, 1),
            "sha256": _sha256(target),
            "bytes": target.stat().st_size,
            "res": res,
            "samples": samples,
            "engine": engine,
            "exact": False,
        })

    return {"ok": True, "blender": bpy.app.version_string, "engine": engine,
            "images": images, "views": [img["view"] for img in images],
            "res": res, "samples": samples, "degraded": degraded,
            "device": engine_facts.get("device"),
            "device_name": engine_facts.get("device_name"),
            "engine_facts": engine_facts, "color_management": cm_facts,
            "texture": build_info.get("texture"),
            "uv": build_info.get("uv"),
            "finish_applied": build_info, "exact": False}


# --------------------------------------------------------------- multi-part

def _bbox_aspect(camera, lo, hi):
    """Width/height of the bbox projected on the camera's right/up axes."""
    R = camera.matrix_world.to_3x3()
    right = np.array(R @ Vector((1.0, 0.0, 0.0)), dtype=np.float64)
    up = np.array(R @ Vector((0.0, 1.0, 0.0)), dtype=np.float64)
    pts = np.array([[x, y, z] for x in (lo[0], hi[0])
                    for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = float(np.ptp(pts @ right))
    h = float(np.ptp(pts @ up))
    if w <= 1e-9 or h <= 1e-9:
        return 1.0
    return w / h


def _load_parts(job: Path):
    spec = json.loads((job / "parts.json").read_text(encoding="utf-8"))
    parts = []
    for row in spec.get("parts") or []:
        data = np.load(job / str(row["mesh"]))
        parts.append({
            "id": str(row.get("id") or "part"),
            "verts": np.asarray(data["verts"], dtype=np.float64).reshape(-1, 3),
            "faces": np.asarray(data["faces"], dtype=np.int64).reshape(-1, 3),
            "finish": row.get("finish") or {},
            "material": row.get("material") or dict(FALLBACK_MATERIAL),
        })
    reference = None
    ref_name = spec.get("reference")
    if ref_name:
        data = np.load(job / str(ref_name))
        reference = (np.asarray(data["verts"], dtype=np.float64).reshape(-1, 3),
                     np.asarray(data["faces"], dtype=np.int64).reshape(-1, 3),
                     None)
    return parts, dict(spec.get("settings") or {}), reference


def _render_parts(job: Path, req: dict):
    """Render a named-part scene with per-part materials, studio presets and
    diagnostic passes (normal / depth / id / curvature / deviation)."""
    parts, settings, reference = _load_parts(job)
    settings = dict(settings)
    settings.update({k: v for k, v in req.items()
                     if k in ("lighting", "background", "ground", "lens_mm",
                              "exposure", "look", "ortho", "passes",
                              "distance_factor", "deviation_range_mm",
                              "focus", "aspect", "focus_padding")})
    scene, engine, degraded = _reset()
    base_engine = engine
    res = max(16, int(req.get("res") or 512))
    samples = max(1, int(req.get("samples") or 8))
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    _configure_render(scene, settings, samples)
    # ``auto`` keeps this op on EEVEE (its historical behaviour, and the right
    # choice for a diagnostic pass); cycles/hip must be asked for.
    engine_facts = _apply_engine(scene, dict(settings, engine=settings.get(
        "engine") or "eevee"), samples, allow_cycles=True)
    degraded.extend(engine_facts.get("warnings") or [])
    base_engine = engine_facts.get("engine") or base_engine
    cm_facts = _apply_color_management(scene, settings)
    degraded.extend(cm_facts.get("warnings") or [])

    objs = []
    applied = []
    for part in parts:
        obj, info = _build(part["verts"], part["faces"],
                           part["finish"], part["material"])
        obj.name = "ATTA_" + str(part["id"])[:48]
        objs.append(obj)
        applied.append({"id": part["id"], **info})
        degraded.extend(info.get("degraded") or [])

    world_verts = (np.concatenate([_object_verts(o) for o in objs], axis=0)
                   if objs else np.zeros((0, 3)))
    lo, hi, size, center, diag = _bounds(world_verts)
    focus = str(settings.get("focus") or "")
    if focus:
        target = next((o for o in objs if o.name == "ATTA_" + focus[:48]), None)
        if target is None:
            raise ValueError("focus part %r not found; parts: %s"
                             % (focus, ", ".join(o.name[5:] for o in objs)))
        flo, fhi, fsize, fcenter, fdiag = _bounds(_object_verts(target))
        if fdiag > 0:
            pad = float(settings.get("focus_padding", 0.3)) * max(fsize.max(), 1e-9)
            flo = flo - pad
            fhi = fhi + pad
            lo, hi = flo, fhi
            size = fhi - flo
            center = (flo + fhi) * 0.5
            diag = float(np.linalg.norm(size))
    camera, diag = _setup_world_and_lights(
        scene, lo, hi, size, center, diag, settings)
    factor = float(settings.get("distance_factor") or 1.0)
    aspect_setting = settings.get("aspect", "square")
    allowed = ("beauty", "normal", "depth", "id", "curvature", "deviation")
    passes = []
    for raw in (settings.get("passes") or ["beauty"]):
        name = str(raw).strip().lower()
        seq = (["normal", "depth", "id", "curvature"]
               if name == "diagnostic" else [name])
        for item in seq:
            if item not in allowed:
                raise ValueError("unknown render pass %r" % item)
            if item not in passes:
                passes.append(item)
    meshes = [_mesh_arrays(o) for o in objs]
    curv = [_vertex_curvature(v, t, diag) for v, t, _ in meshes]
    curv_all = np.concatenate(curv) if curv else np.zeros(0)
    curv_t = _normalize01(curv_all)
    curv_slices = []
    _off = 0
    for h in curv:
        curv_slices.append(curv_t[_off:_off + len(h)])
        _off += len(h)
    palette = _id_palette(len(objs))
    ref_grid = _reference_grid(reference) if reference is not None else None

    outdir = job / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    images = []
    for view in [str(v) for v in (req.get("views") or ["iso"])]:
        direction = VIEWS.get(view)
        if direction is None:
            raise ValueError("unknown view %r (known: %s)"
                             % (view, ", ".join(sorted(VIEWS))))
        distance = max(diag * 1.9, 0.02)
        for _ in range(6):
            _aim(camera, center, direction, distance)
            ndc = _worst_ndc(lo, hi) if len(world_verts) else 0.0
            if ndc <= 0.92:
                break
            distance *= max(1.02, ndc / 0.85)

        # Orientation-aware auto aspect, independent of the current resolution
        # (world_to_camera_view depends on the camera frame, so it cannot be
        # used here without feedback).
        aspect = 1.0
        if isinstance(aspect_setting, (int, float)):
            aspect = float(aspect_setting)
        elif str(aspect_setting).lower() == "auto":
            aspect = _bbox_aspect(camera, lo, hi)
        elif str(aspect_setting).lower() != "square":
            try:
                aspect = float(aspect_setting)
            except (TypeError, ValueError):
                aspect = 1.0
        aspect = min(2.2, max(0.45, float(aspect)))
        if aspect >= 1.0:
            rx, ry = int(res), max(16, int(round(res / aspect)))
        else:
            rx, ry = max(16, int(round(res * aspect))), int(res)
        scene.render.resolution_x = rx
        scene.render.resolution_y = ry

        # Re-fit with the final frame, then apply the explicit distance
        # multiplier so a close-up is not undone by the fit loop.
        for _ in range(4):
            _aim(camera, center, direction, distance)
            ndc = _worst_ndc(lo, hi) if len(world_verts) else 0.0
            if ndc <= 0.92:
                break
            distance *= max(1.02, ndc / 0.85)
        distance *= factor
        _aim(camera, center, direction, distance)
        cam_loc = np.array(camera.location, dtype=np.float64)
        forward = np.asarray(center, dtype=np.float64) - cam_loc
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        depths = [((v - cam_loc) @ forward) for v, _, _ in meshes]
        dmin = min((float(d.min()) for d in depths if len(d)), default=0.0)
        dmax = max((float(d.max()) for d in depths if len(d)), default=1.0)

        for pass_name in passes:
            if pass_name == "beauty":
                scene.render.engine = base_engine
                scene.render.film_transparent = (
                    str(settings.get("background") or "transparent") == "transparent")
                try:
                    scene.view_settings.view_transform = "AgX"
                except Exception:  # noqa: BLE001
                    pass
                _configure_render(scene, settings, samples)
                for obj in objs:
                    obj.hide_render = False
                target = outdir / ("%s.png" % view)
            elif pass_name in ("normal", "depth", "id", "curvature", "deviation"):
                _workbench(scene, samples)
                scene.render.film_transparent = True
                target = outdir / ("%s_%s.png" % (view, pass_name))
                if pass_name == "normal":
                    for obj, (_, _, normals) in zip(objs, meshes):
                        _set_vertex_colors(obj, np.clip(normals * 0.5 + 0.5, 0, 1))
                elif pass_name == "depth":
                    for obj, depth in zip(objs, depths):
                        t = (depth - dmin) / max(dmax - dmin, 1e-12)
                        gray = np.clip(1.0 - t, 0.0, 1.0)
                        _set_vertex_colors(obj, np.stack([gray, gray, gray], axis=1))
                elif pass_name == "curvature":
                    for obj, t in zip(objs, curv_slices):
                        _set_vertex_colors(obj, _ramp(t, [
                            (0.0, (0.03, 0.10, 0.35)),
                            (0.35, (0.05, 0.55, 0.65)),
                            (0.7, (0.95, 0.80, 0.20)),
                            (1.0, (0.75, 0.08, 0.06)),
                        ]))
                elif pass_name == "id":
                    for obj, color in zip(objs, palette):
                        _set_vertex_colors(obj, np.tile(color, (len(obj.data.vertices), 1)))
                else:  # deviation
                    if ref_grid is None:
                        images.append({
                            "view": view, "pass": "deviation", "ok": False,
                            "skipped": "no reference mesh",
                            "path": None, "abs_path": None,
                        })
                        continue
                    dist = _nearest_from_grid(ref_grid, np.concatenate(
                        [v for v, _, _ in meshes], axis=0))
                    offset = 0
                    range_mm = settings.get("deviation_range_mm")
                    p95 = float(np.percentile(dist, 95)) * 1000.0 if len(dist) else 0.0
                    scale = float(range_mm) if range_mm else max(1.0, p95 * 2.0)
                    for obj, (v, _, _) in zip(objs, meshes):
                        d = dist[offset:offset + len(v)] * 1000.0
                        offset += len(v)
                        t = np.clip(d / max(scale, 1e-9), 0.0, 1.0)
                        _set_vertex_colors(obj, _ramp(t, [
                            (0.0, (0.04, 0.10, 0.55)),
                            (0.25, (0.05, 0.55, 0.85)),
                            (0.5, (0.10, 0.75, 0.35)),
                            (0.75, (0.95, 0.80, 0.15)),
                            (1.0, (0.80, 0.06, 0.05)),
                        ]))
            else:
                raise ValueError("unknown render pass %r" % pass_name)

            scene.render.filepath = str(target)
            started = time.perf_counter()
            render_error = None
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as exc:  # noqa: BLE001
                if not str(scene.render.engine).startswith("BLENDER_WORKBENCH"):
                    degraded.append("render failed on %s (%s); retried with "
                                    "BLENDER_WORKBENCH" % (scene.render.engine, exc))
                    try:
                        scene.render.engine = "BLENDER_WORKBENCH"
                        bpy.ops.render.render(write_still=True)
                    except Exception as retry_exc:  # noqa: BLE001
                        render_error = retry_exc
                else:
                    render_error = exc
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if render_error is not None:
                raise render_error
            stats = _png_stats(target)
            row = {
                "view": view,
                "pass": pass_name,
                "path": str(target),
                "abs_path": str(target.resolve()),
                "width": stats["width"],
                "height": stats["height"],
                "coverage": stats["coverage"],
                "content_bbox_px": stats["content_bbox_px"],
                "worst_ndc": round(float(ndc), 4),
                "empty_frame": bool(stats["empty_frame"]),
                "render_ms": round(elapsed_ms, 1),
                "sha256": _sha256(target),
                "bytes": target.stat().st_size,
                "res": res,
                "resolution": [rx, ry],
                "aspect": round(float(aspect), 4),
                "samples": samples,
                "engine": scene.render.engine,
                "exact": False,
            }
            if pass_name == "depth":
                row["range_m"] = [round(dmin, 9), round(dmax, 9)]
                row["note"] = "near = white, far = black"
            elif pass_name == "curvature":
                row["note"] = "log mean-curvature magnitude (blue low, red high)"
            elif pass_name == "normal":
                row["note"] = "world-space normal encoded as RGB = n*0.5+0.5"
            elif pass_name == "deviation":
                row["range_mm"] = round(scale, 6)
                row["note"] = "absolute nearest-surface distance (blue=0, red=range)"
            images.append(row)

    return {"ok": True, "blender": bpy.app.version_string, "engine": base_engine,
            "images": images, "views": [img["view"] for img in images],
            "passes": passes,
            "res": res, "samples": samples, "degraded": degraded,
            "device": engine_facts.get("device"),
            "device_name": engine_facts.get("device_name"),
            "engine_facts": engine_facts, "color_management": cm_facts,
            "texture": [{"id": p["id"], "texture": (a.get("texture")),
                         "uv": (a.get("uv"))} for p, a in zip(parts, applied)
                        if isinstance(a, dict)
                        and (a.get("texture") or a.get("uv"))],
            "parts": [{"id": p["id"], "material": (p["material"] or {}).get("name")}
                      for p in parts],
            "finish_applied": applied, "settings": settings, "exact": False}


# ----------------------------------------------------------- eval / export

def _eval(job: Path, req: dict):
    verts, faces, finish, material_spec = _load(job)
    _reset()
    obj, build_info = _build(verts, faces, finish, material_spec)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        n_vertices = len(mesh.vertices)
        n_tris = len(mesh.loop_triangles)
        positions = np.empty(n_vertices * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", positions)
        indices = np.empty(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", indices)
    finally:
        evaluated.to_mesh_clear()

    out = job / "evaluated.npz"
    np.savez_compressed(out,
                        verts=positions.reshape(-1, 3).astype(np.float64),
                        faces=indices.reshape(-1, 3).astype(np.int64))
    return {"ok": True, "path": str(out), "abs_path": str(out.resolve()),
            "n_vertices": int(n_vertices), "n_tris": int(n_tris),
            "live_modifiers": [m.type for m in obj.modifiers],
            "finish_applied": build_info, "exact": False,
            "degraded": build_info["degraded"]}


def _export_normal_chain_straighten() -> list:
    """Temporarily make the Principled Normal input the Normal Map's output.

    The glTF exporter walks the ``Normal`` input: if it meets a ``Bump`` node it
    exports that node's *height* image as ``normalTexture``, which is a
    tangent-space normal map in name only -- grayscale height data delivered as
    a normal map, i.e. visibly wrong on any glTF viewer.  Rendering keeps the
    bump (it is a real improvement on top of the normal map), so the rewire is
    done for the export only and undone immediately after.
    """
    rewired = []
    for material in bpy.data.materials:
        tree = getattr(material, "node_tree", None)
        if tree is None:
            continue
        bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        socket = bsdf.inputs.get("Normal")
        if socket is None or not socket.is_linked:
            continue
        link = socket.links[0]
        source = link.from_node
        if source.type != "BUMP":
            continue
        normal_source = None
        bump_normal = source.inputs.get("Normal")
        if bump_normal is not None and bump_normal.is_linked:
            candidate = bump_normal.links[0].from_node
            if candidate.type == "NORMAL_MAP":
                normal_source = candidate.outputs.get("Normal")
        if normal_source is None:
            continue
        from_socket = link.from_socket
        tree.links.remove(link)
        tree.links.new(normal_source, socket)
        rewired.append((tree, socket, from_socket))
    return rewired


def _restore_normal_chain(rewired) -> None:
    for tree, socket, from_socket in rewired or []:
        try:
            for link in list(socket.links):
                tree.links.remove(link)
            tree.links.new(from_socket, socket)
        except Exception:  # noqa: BLE001
            pass


def _pack_images_for_export() -> list:
    """Pack every image that has a file, so a GLB can embed it.

    The glTF exporter only writes an ``images`` entry for an image it can read:
    a packed image or one whose file is reachable.  Our maps come from the
    texture set on disk, so they are reachable -- but the bake targets and any
    generated image are not, and a half-embedded GLB is worse than a loud
    failure.  Packing is idempotent and cheap.
    """
    packed = []
    for image in bpy.data.images:
        if image.source == "VIEWER":
            continue
        try:
            if image.packed_file is None and image.filepath:
                image.pack()
                packed.append(image.name)
        except Exception:  # noqa: BLE001 - unreachable file: report, keep going
            pass
    return packed


def _glb_json_chunk(path: Path) -> dict:
    """Parse a GLB container's JSON chunk (header + chunk table, no library)."""
    import struct

    blob = Path(path).read_bytes()
    if len(blob) < 20:
        raise ValueError("GLB is %d bytes, too short for a container" % len(blob))
    magic, version, length = struct.unpack_from("<III", blob, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB (magic 0x%08x)" % magic)
    chunk_length, chunk_type = struct.unpack_from("<II", blob, 12)
    if chunk_type != 0x4E4F534A:
        raise ValueError("first GLB chunk is not JSON (type 0x%08x)"
                         % chunk_type)
    payload = blob[20:20 + chunk_length].decode("utf-8", "replace")
    gltf = json.loads(payload)
    return {"gltf": gltf, "version": version,
            "declared_length": int(length), "actual_length": len(blob),
            "json_chunk_bytes": int(chunk_length)}


def _glb_texture_report(target: Path) -> dict:
    """Assert the GLB really carries its textures, and say what they are."""
    report = {"ok": False, "images": 0, "textures": 0, "materials": 0,
              "materials_with_textures": 0, "warnings": []}
    try:
        parsed = _glb_json_chunk(target)
    except Exception as exc:  # noqa: BLE001
        report["warnings"].append("could not parse the GLB: %s" % (exc,))
        return report
    gltf = parsed["gltf"]
    images = gltf.get("images") or []
    textures = gltf.get("textures") or []
    materials = gltf.get("materials") or []
    report.update({
        "container_version": parsed["version"],
        "declared_length": parsed["declared_length"],
        "actual_length": parsed["actual_length"],
        "images": len(images),
        "textures": len(textures),
        "materials": len(materials),
        "image_names": [str(i.get("name") or "") for i in images],
        "mime_types": [str(i.get("mimeType") or "") for i in images],
        "embedded_images": sum(1 for i in images if "bufferView" in i),
        "external_uris": [i.get("uri") for i in images if "uri" in i],
    })
    for index, image in enumerate(images):
        if "bufferView" not in image:
            report["warnings"].append(
                "image %d is not embedded (uri=%r)"
                % (index, image.get("uri")))
        mime = str(image.get("mimeType") or "")
        if not mime.startswith("image/"):
            report["warnings"].append("image %d has mimeType %r" % (index, mime))
    with_textures = 0
    referenced = set()
    for material in materials:
        used = False
        pbr = material.get("pbrMetallicRoughness") or {}
        for holder, key in ((pbr, "baseColorTexture"),
                            (pbr, "metallicRoughnessTexture"),
                            (material, "normalTexture"),
                            (material, "occlusionTexture"),
                            (material, "emissiveTexture")):
            entry = holder.get(key)
            if isinstance(entry, dict) and "index" in entry:
                used = True
                referenced.add(int(entry["index"]))
        if used:
            with_textures += 1
    report["materials_with_textures"] = with_textures
    report["textures_referenced"] = len(referenced)
    report["material_names"] = [str(m.get("name") or "") for m in materials]
    report["material_texture_slots"] = [
        {key: (int(value["index"]) if isinstance(value, dict)
               and "index" in value else None)
         for key, value in (
             ("baseColorTexture",
              (m.get("pbrMetallicRoughness") or {}).get("baseColorTexture")),
             ("metallicRoughnessTexture",
              (m.get("pbrMetallicRoughness") or {}).get("metallicRoughnessTexture")),
             ("normalTexture", m.get("normalTexture")),
             ("occlusionTexture", m.get("occlusionTexture")))}
        for m in materials]
    report["ok"] = bool(images and textures and with_textures
                        and report["embedded_images"] == len(images)
                        and not report["external_uris"])
    if not images:
        report["warnings"].append(
            "GLB has no images: the material is not texture-backed, or the "
            "exporter could not read the map files")
    elif not with_textures:
        report["warnings"].append(
            "GLB has images but no material references one")
    return report


#: glTF operator keyword ladder, richest first.  ``bpy`` operators reject an
#: unknown keyword with ``TypeError``, and the accepted set genuinely moves
#: between Blender releases (``export_texcoords``/``export_image_format`` are
#: 3.6+; ``use_selection`` predates them), so the export must retry with fewer
#: keywords rather than die.  Kept as a module constant so the retry loop stays
#: the first thing in ``_export_glb``.
GLB_KWARG_LADDER = (
    {"use_selection": True, "export_apply": True, "export_yup": True,
     "export_image_format": "AUTO", "export_materials": "EXPORT",
     "export_texcoords": True, "export_normals": True,
     "export_unused_images": False, "export_unused_textures": False},
    {"use_selection": True, "export_apply": True, "export_yup": True,
     "export_image_format": "AUTO", "export_materials": "EXPORT",
     "export_texcoords": True},
    {"use_selection": True, "export_apply": True, "export_yup": True},
    {"use_selection": True, "export_apply": True},
    {"use_selection": True},
    {},
)


def _glb_prepare_export() -> tuple[list, list]:
    """Pack images and temporarily straighten the Normal chain (see below)."""
    return _pack_images_for_export(), _export_normal_chain_straighten()


def _export_glb(target: Path, expect_textures: bool = False) -> dict:
    """glTF export with images embedded, verified by re-parsing the container."""
    packed, rewired = _glb_prepare_export()
    last = None
    used = None
    for extra in GLB_KWARG_LADDER:
        try:
            bpy.ops.export_scene.gltf(filepath=str(target),
                                      export_format="GLB", **extra)
            used = extra
            break
        except TypeError as exc:   # unknown kwarg on this Blender: drop it
            last = exc
            continue
        except Exception as exc:   # noqa: BLE001
            _restore_normal_chain(rewired)
            raise RuntimeError("glTF export failed: %s" % _err_text(exc)) from exc
    else:
        _restore_normal_chain(rewired)
        raise RuntimeError("glTF export failed with every keyword set: %s"
                           % _err_text(last))
    _restore_normal_chain(rewired)
    extra_uv = {}
    for obj in bpy.context.selected_objects:
        if obj.type == "MESH" and obj.data.uv_layers:
            layer = obj.data.uv_layers.active
            extra_uv["uv_layer"] = _safe_name(layer) if layer else None
            break
    report = _glb_texture_report(target)
    # A scalar material legitimately embeds no images; saying "ok" for that case
    # is the difference between an honest receipt and a spurious degradation.
    if not expect_textures and not report.get("images"):
        report["not_applicable"] = ("the material declares no texture set, so "
                                    "an image-free GLB is correct")
        report["warnings"] = [w for w in report.get("warnings") or []
                              if "no images" not in w]
        report["ok"] = True
    report["expect_textures"] = bool(expect_textures)
    report["packed_images"] = packed
    report["kwargs_used"] = sorted(used) if used is not None else []
    report.update(extra_uv)
    return report


def _export_obj(target: Path) -> None:
    """OBJ export.  ``wm.obj_export`` landed in 3.2; before that the operator
    was ``export_scene.obj`` with different keyword names."""
    try:
        bpy.ops.wm.obj_export(
            filepath=str(target), export_selected_objects=True,
            apply_modifiers=True, export_materials=True,
            export_normals=True)
        return
    except AttributeError:
        pass                      # operator absent on this Blender
    except TypeError:
        pass                      # keyword mismatch
    except Exception as exc:      # noqa: BLE001
        raise RuntimeError("OBJ export failed: %s" % (exc,)) from exc
    # legacy path (<= 3.1)
    try:
        bpy.ops.export_scene.obj(
            filepath=str(target), use_selection=True,
            use_mesh_modifiers=True, use_materials=True,
            use_normals=True)
        return
    except TypeError as exc:
        try:
            bpy.ops.export_scene.obj(filepath=str(target), use_selection=True)
            return
        except Exception as exc2:  # noqa: BLE001
            raise RuntimeError("OBJ export failed: %s / %s" % (exc, exc2)) from exc2
    except Exception as exc:      # noqa: BLE001
        raise RuntimeError("OBJ export failed: %s" % (exc,)) from exc


#: MTL statements this module guarantees for a textured material.  Blender's
#: OBJ exporter writes ``map_Kd`` for a base-colour image and nothing for
#: metallic/roughness, so those lines are appended here rather than hoped for.
MTL_TEXTURE_LINES = (
    ("map_Kd", "albedo"),
    ("map_Ks", "metallic"),
    ("map_Bump", "normal"),
)


def _copy_texture_sidecars(target: Path, texture: dict | None) -> dict:
    """Copy a set's PNGs next to the OBJ and return the mapping used.

    A ``.mtl`` that points at ``/home/.../textures/tex_x/albedo.png`` is not a
    portable deliverable, so the files are copied into ``<stem>_textures/`` and
    every map line is rewritten to that relative directory.
    """
    out = {"dir": None, "copied": [], "missing": [], "maps": {}}
    if not isinstance(texture, dict):
        return out
    maps = texture.get("maps") or {}
    if not maps:
        return out
    dest = target.parent / ("%s_textures" % target.stem)
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        out["error"] = "could not create %s (%s)" % (dest, exc)
        return out
    out["dir"] = str(dest)
    for name in TEXTURE_MAP_ORDER:
        record = maps.get(name)
        if not isinstance(record, dict):
            continue
        source = Path(str(record.get("path") or ""))
        if not source.is_file():
            out["missing"].append(name)
            continue
        try:
            blob = source.read_bytes()
            copy = dest / source.name
            copy.write_bytes(blob)
        except OSError as exc:
            out["missing"].append(name)
            out["error"] = "%s: %s" % (source, exc)
            continue
        out["copied"].append(name)
        out["maps"][name] = {
            "file": str(copy.relative_to(target.parent)),
            "abs_path": str(copy.resolve()),
            "bytes": copy.stat().st_size,
            "sha256": _sha256(copy),
        }
    return out


def _patch_mtl(mtl_path: Path, sidecar: dict) -> dict:
    """Ensure ``map_Kd``/``map_Ks``/``map_Bump`` exist and point at the copies.

    Rewrites existing lines in place (so the exporter's choice of filename is
    honoured) and appends whatever is missing.  Returns a receipt of the final
    statements, because "the .mtl has a map_Bump line" is a checkable claim.
    """
    receipt = {"exists": False, "path": str(mtl_path), "statements": {},
               "appended": [], "rewritten": []}
    if not mtl_path.is_file():
        receipt["error"] = "the exporter did not write an .mtl"
        return receipt
    receipt["exists"] = True
    try:
        text = mtl_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        receipt["error"] = str(exc)
        return receipt
    lines = text.splitlines()
    wanted = {}
    for keyword, map_name in MTL_TEXTURE_LINES:
        record = (sidecar.get("maps") or {}).get(map_name)
        if record:
            wanted[keyword] = record["file"].replace("\\", "/")
    present = set()
    for index, line in enumerate(lines):
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        keyword = parts[0]
        if keyword in wanted:
            present.add(keyword)
            if parts[1].strip() != wanted[keyword]:
                lines[index] = "%s %s" % (keyword, wanted[keyword])
                receipt["rewritten"].append(keyword)
            receipt["statements"][keyword] = lines[index].split(None, 1)[1]
    for keyword in ("map_Kd", "map_Ks", "map_Bump"):
        if keyword in wanted and keyword not in present:
            lines.append("%s %s" % (keyword, wanted[keyword]))
            receipt["appended"].append(keyword)
            receipt["statements"][keyword] = wanted[keyword]
    # ``map_Bump`` is spelled ``map_bump``/``bump`` by some readers; add the
    # canonical alias so more importers pick the normal map up.
    if "map_Bump" in wanted and not any(
            line.split(None, 1)[0] in ("bump", "map_bump")
            for line in lines if line.strip()):
        lines.append("map_bump %s" % wanted["map_Bump"])
        receipt["appended"].append("map_bump")
    if receipt["rewritten"] or receipt["appended"]:
        try:
            mtl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            receipt["error"] = str(exc)
    receipt["lines"] = len(lines)
    receipt["map_lines"] = sum(1 for line in lines
                               if line.split(None, 1)[:1] and
                               line.split(None, 1)[0].startswith("map_"))
    return receipt


def _export_obj_textured(target: Path, texture: dict | None) -> dict:
    """Write the OBJ, then make the MTL and the PNGs actually self-contained."""
    _export_obj(target)                       # the actual geometry + MTL
    if not target.is_file():
        raise RuntimeError("OBJ export reported success but wrote no file at %s"
                           % (target,))
    sidecar = _copy_texture_sidecars(target, texture)
    mtl = target.with_suffix(".mtl")
    receipt = _patch_mtl(mtl, sidecar) if sidecar.get("maps") else {
        "exists": mtl.is_file(), "path": str(mtl), "statements": {},
        "appended": [], "rewritten": [], "note": "no texture set on this material"}
    receipt["sidecar"] = sidecar
    if sidecar.get("copied"):
        receipt["textures"] = sorted(sidecar["copied"])
    return receipt


def _export_stl(target: Path) -> None:
    """STL export.  ``wm.stl_export`` is 3.2+; older builds use
    ``export_mesh.stl`` which takes a different keyword set."""
    try:
        bpy.ops.wm.stl_export(
            filepath=str(target), export_selected_objects=True,
            apply_modifiers=True, ascii_format=False)
        return
    except AttributeError:
        pass
    except TypeError:
        pass
    except Exception as exc:      # noqa: BLE001
        raise RuntimeError("STL export failed: %s" % (exc,)) from exc
    try:
        bpy.ops.export_mesh.stl(
            filepath=str(target), use_selection=True,
            use_mesh_modifiers=True, ascii=False)
        return
    except TypeError:
        try:
            bpy.ops.export_mesh.stl(filepath=str(target), use_selection=True)
            return
        except Exception as exc2:  # noqa: BLE001
            raise RuntimeError("STL export failed: %s" % (exc2,)) from exc2
    except Exception as exc:      # noqa: BLE001
        raise RuntimeError("STL export failed: %s" % (exc,)) from exc


def _export(job: Path, req: dict):
    verts, faces, finish, material_spec = _load(job)
    _reset()
    obj, build_info = _build(verts, faces, finish, material_spec)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    outdir = job / "export"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = str(req.get("stem") or "model")
    formats = [str(f).lower() for f in (req.get("formats") or ["glb"])]
    texture = (material_spec or {}).get("texture")
    if not isinstance(texture, dict):
        texture = None
    files = []
    #: Sidecar files (MTL, copied PNGs) so the host can hash a complete list.
    extra_files = []
    for fmt in formats:
        suffix = "blend" if fmt == "blend" else fmt
        target = outdir / ("%s.%s" % (stem, suffix))
        entry = {"format": fmt, "path": str(target),
                 "abs_path": str(target.resolve()),
                 "exact": fmt == "blend"}
        try:
            if fmt == "glb":
                gltf_report = _export_glb(target, expect_textures=bool(texture))
                entry["gltf"] = gltf_report
                if not gltf_report.get("ok"):
                    for warning in gltf_report.get("warnings") or []:
                        build_info["degraded"].append("glb: %s" % (warning,))
            elif fmt == "obj":
                obj_report = _export_obj_textured(target, texture)
                entry["mtl"] = obj_report
                mtl_path = Path(obj_report.get("path") or "")
                if mtl_path.is_file():
                    extra_files.append({
                        "role": "mtl", "format": "mtl",
                        "path": str(mtl_path),
                        "abs_path": str(mtl_path.resolve()),
                        "bytes": mtl_path.stat().st_size,
                        "sha256": _sha256(mtl_path),
                    })
                for name, record in ((obj_report.get("sidecar") or {})
                                     .get("maps") or {}).items():
                    extra_files.append({
                        "role": "texture", "map": name, "format": "png",
                        "path": record["abs_path"],
                        "abs_path": record["abs_path"],
                        "bytes": record["bytes"], "sha256": record["sha256"],
                    })
                if (obj_report.get("sidecar") or {}).get("missing"):
                    build_info["degraded"].append(
                        "obj: map(s) %s were declared but not found"
                        % (", ".join((obj_report["sidecar"] or {})["missing"]),))
            elif fmt == "stl":
                _export_stl(target)
            elif fmt == "blend":
                bpy.ops.wm.save_as_mainfile(filepath=str(target), copy=True)
            else:
                raise ValueError("unsupported format %r (use glb, obj, stl, blend)"
                                 % (fmt,))
            entry.update({"ok": True, "bytes": target.stat().st_size,
                          "sha256": _sha256(target)})
            summary = _uv_summary(build_info.get("uv"), texture)
            if summary:
                entry["texture"] = summary
        except Exception as exc:  # noqa: BLE001
            entry.update({"ok": False,
                          "error": "%s: %s" % (type(exc).__name__, _err_text(exc))})
        files.append(entry)

    return {"ok": all(f.get("ok", False) for f in files), "files": files,
            "extra_files": extra_files,
            "dir": str(outdir), "n_vertices": int(len(obj.data.vertices)),
            "n_tris": int(len(obj.data.polygons)),
            "texture": _texture_receipt(texture),
            "texture_summary": _uv_summary(build_info.get("uv"), texture),
            "finish_applied": build_info, "exact": False,
            "blender": bpy.app.version_string,
            "degraded": build_info["degraded"]}


def _uv_summary(uv: dict | None, texture: dict | None) -> dict | None:
    """The compact, threshold-able answer to "is this texture stretched?".

    Deliberately flat and small: ``px_per_mm.median`` and ``overlap_fraction``
    are the two numbers a caller can put a tolerance on, so they are first.
    """
    uv = uv or {}
    texel = (uv.get("texel") or {})
    density = texel.get("texel_density_px_mm") or {}
    islands = (uv.get("islands") or {})
    overlap = (uv.get("overlap") or {})
    maps = sorted((texture or {}).get("maps") or {})
    if not uv and not texture:
        return None
    return {
        "uv_mode": uv.get("mode"),
        "uv_requested_mode": uv.get("requested_mode"),
        "uv_layer": uv.get("layer"),
        "uv_islands": islands.get("count"),
        "uv_islands_measured": bool(islands.get("count") is not None),
        "px_per_mm": {
            "median": density.get("median"),
            "area_weighted_mean": density.get("area_weighted_mean"),
            "p5": density.get("p5"),
            "p95": density.get("p95"),
        },
        "texel_density_px_mm": density.get("median"),
        "tile_mm": uv.get("tile_mm_intended") or uv.get("scale_mm"),
        "scale_mm_effective": uv.get("scale_mm_effective"),
        "density_error": uv.get("density_error"),
        "density_spread_p95_p5": texel.get("density_ratio_p95_p5"),
        "overlap_fraction": overlap.get("overlap_fraction"),
        "uv_tiles": texel.get("uv_area_tiles"),
        "mirrored": uv.get("mirrored"),
        "world_area_mm2": texel.get("world_area_mm2"),
        "maps": maps,
        "set_id": (texture or {}).get("id"),
        "size_px": (texture or {}).get("size"),
        "world_space_assumption": uv.get("world_space_assumption"),
        "warnings": list(uv.get("warnings") or []),
    }


def _texture_receipt(texture: dict | None) -> dict | None:
    """What texture the exported asset carries, independent of the format."""
    if not isinstance(texture, dict):
        return None
    maps = texture.get("maps") or {}
    return {
        "set_id": texture.get("id"),
        "dir": texture.get("dir"),
        "size": texture.get("size"),
        "maps": sorted(maps),
        "map_sha256": {name: rec.get("sha256") for name, rec in maps.items()
                       if isinstance(rec, dict)},
    }


# ------------------------------------------------------------------- protocol

def _handle(req: dict) -> dict:
    op = req.get("op")
    if op == "ping":
        return {"ok": True, "pong": True, "blender": bpy.app.version_string,
                "pid": os.getpid(), "engine": bpy.context.scene.render.engine}
    if op == "shutdown":
        return {"ok": True, "shutdown": True}
    job = Path(str(req.get("dir") or ".")).expanduser().resolve()
    if op == "render":
        return _render(job, req)
    if op in ("render_parts", "render_scene"):
        return _render_parts(job, req)
    if op in ("eval", "evaluate"):
        return _eval(job, req)
    if op == "export":
        return _export(job, req)
    if op == "shoot":
        return _shoot(job, req)
    if op in ("bake", "bake_maps"):
        return _bake(job, req)
    return {"ok": False, "error": "unknown op %r" % (op,),
            "fix": "use ping, render, render_parts, shoot, eval, export, bake "
                   "or shutdown"}


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None)
    parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass

    sys.stdout.write("ATTA_READY\n")
    sys.stdout.flush()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("request must be a JSON object")
            req_id = req.get("id")
            response = _handle(req)
        except Exception as exc:  # noqa: BLE001
            response = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                        "traceback": traceback.format_exc(limit=8)}
        response["id"] = req_id
        sys.stdout.write("ATTA_JSON:" + json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        if response.get("shutdown"):
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
