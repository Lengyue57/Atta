# -*- coding: utf-8 -*-
"""Host-side resolution of an ``atta.texture/v1`` texture set for the worker.

Why this module exists
----------------------
The Blender worker runs inside Blender's bundled Python and deliberately
imports nothing from the ``atta`` package (see ``worker.py``): a missing import
at worker startup is fatal, and Blender would have to be told about the repo on
``sys.path``.  So *this* module — which runs in the host process, where the
package is importable — is where a ``texture_set`` id becomes absolute file
paths.  It enriches the material spec that gets written into the job directory
as ``material.json``, under a single ``"texture"`` key::

    {"name": "steel", ..., "texture_set": "tex_a0141249",
     "uv": {...}, "normal_strength": 1.0, "tint": [1, 1, 1],
     "roughness_mul": 1.0,
     "texture": {
        "id": "tex_a0141249",
        "dir": "/abs/path/textures/tex_a0141249",
        "size": [512, 512],
        "tileable": true,
        "maps": {"albedo": {"path": "/abs/.../albedo.png",
                            "colorspace": "sRGB", "channels": 3}, ...},
        "orm": {"path": "/abs/.../orm.png",
                "swizzle": {"R": "ao", "G": "roughness", "B": "metallic"}},
        "pbr": {"metallic": 1.0, "roughness_mul": 1.0,
                "normal_strength": 1.0, "displacement_mm": 0.0},
        "normal_convention": "opengl",
        "source": "atta.texture"}}

The worker also carries a small stdlib resolver of its own, so a hand-written
``material.json`` that names only ``texture_set`` still works; this module is
the intended path and the one that can report rich warnings.

The ``atta.texture`` package (``pbr-texture-core``) owns the on-disk format.  It
is used when importable — including its id/prefix resolution and its
``TextureSet.to_material()`` shape — and this module falls back to reading
``set.json`` directly when it is not, so the bridge works either way and the
package stays a refinement rather than a load-bearing dependency.

Nothing here imports ``bpy``; it is safe to import from the MCP server.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

__all__ = [
    "TEXTURE_ROOT_ENV", "MAP_ORDER", "MAP_COLORSPACES",
    "textures_root", "load_set", "enrich", "enrich_finish", "texture_facts",
    "sha256_of",
]

#: Environment override for the texture root (frozen contract).
TEXTURE_ROOT_ENV = "ATTA_TEXTURES"

#: Canonical map order; the worker builds its graph in this order so a receipt
#: is reproducible rather than dict-insertion dependent.
MAP_ORDER = ("albedo", "roughness", "metallic", "normal", "height", "ao", "orm")

#: Colour space each map must be read in.  Wrong = washed-out or over-shiny,
#: which is why the set record states it and this module never guesses.
MAP_COLORSPACES = {
    "albedo": "sRGB",
    "roughness": "Non-Color",
    "metallic": "Non-Color",
    "normal": "Non-Color",
    "height": "Non-Color",
    "ao": "Non-Color",
    "orm": "Non-Color",
}

#: Default ``pbr`` block, matching ``atta.texture.set.DEFAULT_PBR``.
DEFAULT_PBR = {
    "metallic": 0.0,
    "roughness_mul": 1.0,
    "normal_strength": 1.0,
    "displacement_mm": 0.0,
}


def _atta_root() -> Path:
    """Package root (``<repo>`` for an editable checkout)."""
    return Path(__file__).resolve().parents[2]


def textures_root(root=None) -> Path:
    """``$ATTA_TEXTURES`` else ``<ATTA_ROOT>/textures``.

    Deliberately does *not* create the directory: a missing texture root is a
    fact to report, not a side effect to hide.
    """
    if root:
        return Path(str(root)).expanduser()
    env = os.environ.get(TEXTURE_ROOT_ENV)
    if env:
        return Path(env).expanduser()
    return _atta_root() / "textures"


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------ loading

def _locate_fallback(set_id: str, root: Path) -> Path | None:
    """Resolve a set id / directory / ``set.json`` path without ``atta.texture``.

    Accepts exactly what the package's ``locate()`` accepts for the common
    cases: a directory, a ``set.json`` path, a full id, or a unique id prefix.
    """
    text = str(set_id).strip()
    direct = Path(text).expanduser()
    if direct.name == "set.json" and direct.is_file():
        return direct.parent
    if direct.is_dir() and (direct / "set.json").is_file():
        return direct
    candidate = root / text
    if (candidate / "set.json").is_file():
        return candidate
    try:
        entries = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return None
    matches = [p for p in entries if (p / "set.json").is_file()
               and (p.name == text or p.name.startswith(text))]
    if len(matches) == 1:
        return matches[0]
    return None


def _read_set_json(directory: Path) -> dict:
    path = Path(directory) / "set.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_blocks(raw: dict, directory: Path) -> tuple[dict, list, list]:
    """Turn a raw ``set.json`` record into the worker's texture block.

    Returns ``(block, warnings, missing)``.  Missing map files are dropped from
    ``maps`` *and* reported, because silently rendering without the normal map
    would look like a shading bug rather than a build problem.
    """
    warnings: list[str] = []
    missing: list[str] = []

    size = raw.get("size") or [0, 0]
    try:
        width, height = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError):
        width = height = 0
        warnings.append("set.json size %r is not [w, h]" % (size,))

    raw_maps = raw.get("maps") or {}
    if not isinstance(raw_maps, dict):
        warnings.append("set.json maps is %r, not an object"
                        % (type(raw_maps).__name__,))
        raw_maps = {}

    out_maps: dict[str, dict] = {}
    for name in MAP_ORDER:
        record = raw_maps.get(name)
        if record is None:
            continue
        if not isinstance(record, dict):
            warnings.append("map %r record is %r, not an object" % (name, record))
            continue
        filename = str(record.get("file") or ("%s.png" % name))
        path = (Path(directory) / filename)
        if not path.is_file():
            missing.append(name)
            warnings.append("map %r is declared but missing on disk: %s"
                            % (name, path))
            continue
        stated = str(record.get("colorspace") or "").strip()
        colorspace = MAP_COLORSPACES.get(name, "Non-Color")
        if stated and stated.lower() != colorspace.lower():
            warnings.append(
                "map %r declares colorspace %r; %r is correct for that map and "
                "is what will be used" % (name, stated, colorspace))
        entry = {
            "file": filename,
            "path": str(path.resolve()),
            "colorspace": colorspace,
            "channels": int(record.get("channels") or
                            (3 if name in ("albedo", "normal", "orm") else 1)),
            "bit_depth": int(record.get("bit_depth") or 8),
        }
        if width <= 0 or height <= 0:
            entry["size"] = None
        out_maps[name] = entry
        declared = str(record.get("sha256") or "")
        entry["sha256"] = sha256_of(path) if declared else None
        if declared and entry["sha256"] and declared != entry["sha256"]:
            warnings.append("map %r sha256 changed on disk (%s != %s)"
                            % (name, entry["sha256"][:12], declared[:12]))

    pbr = dict(DEFAULT_PBR)
    raw_pbr = raw.get("pbr") or {}
    if isinstance(raw_pbr, dict):
        for key in DEFAULT_PBR:
            if key in raw_pbr:
                try:
                    pbr[key] = float(raw_pbr[key])
                except (TypeError, ValueError):
                    warnings.append("pbr.%s is %r, not a number; used %r"
                                    % (key, raw_pbr[key], DEFAULT_PBR[key]))
    else:
        warnings.append("set.json pbr is %r, not an object"
                        % (type(raw_pbr).__name__,))

    orm = None
    if "orm" in out_maps:
        orm = {
            "file": out_maps["orm"]["file"],
            "path": out_maps["orm"]["path"],
            "colorspace": "Non-Color",
            "packed": ["ao", "roughness", "metallic"],
            "swizzle": {"R": "ao", "G": "roughness", "B": "metallic"},
        }
        declared = raw_pbr.get("orm") if isinstance(raw_pbr, dict) else None
        if isinstance(declared, dict):
            for channel in ("R", "G", "B"):
                value = declared.get("channel_%s" % channel.lower())
                if isinstance(value, str):
                    orm["swizzle"][channel] = value

    try:
        uv_scale_mm = float(raw.get("uv_scale_mm") or 0.0) or None
    except (TypeError, ValueError):
        uv_scale_mm = None
        warnings.append("uv_scale_mm is %r, not a number"
                        % (raw.get("uv_scale_mm"),))

    block = {
        "id": str(raw.get("id") or Path(directory).name),
        "dir": str(Path(directory).resolve()),
        "label": str(raw.get("label") or ""),
        "size": [width, height],
        "tileable": bool(raw.get("tileable", True)),
        "uv_scale_mm": uv_scale_mm,
        "maps": out_maps,
        "orm": orm,
        "pbr": pbr,
        "normal_convention": str(raw.get("normal_convention") or "opengl"),
        "source": "set.json",
        "schema": str(raw.get("schema") or ""),
    }
    return block, warnings, missing


def _load_via_package(set_id: str, root) -> tuple[dict | None, list]:
    """Preferred path: let ``atta.texture`` resolve and validate the set."""
    try:                                   # lazy: the package is another lane
        from . import _texture_package     # type: ignore
    except Exception:                      # noqa: BLE001 - absence is expected
        return None, []
    return _texture_package.load(set_id, root=root)


def load_set(set_id, root=None, *, verify_hashes: bool = False) -> dict:
    """Resolve ``set_id`` to a normalised texture block.

    Raises :class:`FileNotFoundError` when no set can be found at all; every
    *partial* problem (a missing map, a stale hash) is reported inside
    ``block["warnings"]`` instead, so one bad file does not lose the whole
    material.
    """
    if isinstance(set_id, dict) and "maps" in set_id:
        # Already a resolved block (e.g. straight from TextureSet.to_material).
        return dict(set_id)
    text = str(set_id or "").strip()
    if not text:
        raise FileNotFoundError("empty texture_set id")
    root_dir = textures_root(root)

    block = None
    warnings: list[str] = []
    via = "set.json"
    try:
        from . import _texture_package
        block, warnings = _texture_package.load(text, root=root)
        via = "atta.texture"
    except ImportError:
        block = None
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from None

    if block is None:
        directory = _locate_fallback(text, root_dir)
        if directory is None:
            raise FileNotFoundError(
                "no texture set %r under %s" % (text, root_dir))
        raw = _read_set_json(directory)
        block, warnings, _missing = _normalize_blocks(raw, directory)
        block["source"] = "set.json"
    else:
        block["source"] = via

    if verify_hashes:
        for name, record in (block.get("maps") or {}).items():
            path = record.get("path")
            if not path or not Path(path).is_file():
                continue
            got = sha256_of(path)
            if record.get("sha256") and got != record["sha256"]:
                warnings.append("map %r hash mismatch after load" % (name,))
    block["warnings"] = warnings
    if not block.get("maps"):
        raise FileNotFoundError(
            "texture set %r has no usable map files (declared but missing)"
            % (text,))
    return block


# ---------------------------------------------------------------- enrichment

def enrich(material_spec: dict, root=None) -> tuple[dict, list]:
    """Attach the resolved ``texture`` block to a resolved material spec.

    Returns ``(spec, warnings)``.  A spec without ``texture_set`` is returned
    unchanged — the function is the identity for scalar materials.  A spec that
    names a set that cannot be resolved keeps its scalar baseline and gains a
    warning rather than raising, so a render still happens and the reason is in
    the receipt.
    """
    if not isinstance(material_spec, dict):
        return material_spec, []
    spec = dict(material_spec)
    set_id = spec.get("texture_set")
    if not set_id:
        spec.pop("texture", None)
        return spec, []
    try:
        block = load_set(set_id, root=root)
    except Exception as exc:                            # noqa: BLE001
        return spec, ["texture_set %r could not be resolved (%s: %s); the "
                      "material falls back to its scalar preset"
                      % (set_id, type(exc).__name__, exc)]

    warnings = list(block.pop("warnings", []) or [])

    # Fill the layer's documented defaults first, so a caller may hand in a
    # bare {"texture_set": ...} and still get a complete receipt.
    try:
        from .. import materials as _materials
        normalised, errors = _materials.normalize_texture_spec(spec)
        for error in errors:
            warnings.append("%s: %s" % (error.get("path"), error.get("message")))
        for key, value in normalised.items():
            if key != "texture_set":
                spec.setdefault(key, value)
    except Exception as exc:                            # noqa: BLE001
        warnings.append("could not normalise the texture layer (%s)" % (exc,))

    uv = dict(spec.get("uv") or {})
    if uv.get("scale_mm") is None:
        uv["scale_mm"] = block.get("uv_scale_mm")
    spec["uv"] = uv
    spec["texture"] = block

    if "albedo" not in (block.get("maps") or {}):
        warnings.append("texture set %r has no albedo map; the preset's "
                        "base_color is used as the colour baseline"
                        % (block.get("id"),))
    if block.get("normal_convention", "opengl").lower() not in ("opengl", "gl"):
        warnings.append("texture set %r declares normal convention %r; OpenGL "
                        "(+Y up) is what Blender expects — the normal map may "
                        "be inverted" % (block.get("id"),
                                         block.get("normal_convention")))
    return spec, warnings


def enrich_finish(finish: dict, root=None) -> tuple[dict, list]:
    """Enrich a finish's ``material`` if it is an inline spec, else pass through.

    A finish whose material is a preset *name* carries no texture and is
    returned unchanged, hash for hash.
    """
    if not isinstance(finish, dict):
        return finish, []
    material = finish.get("material")
    if not isinstance(material, dict):
        return finish, []
    resolved, warnings = enrich(material, root=root)
    out = dict(finish)
    out["material"] = resolved
    return out, warnings


# ------------------------------------------------------------------- receipts

def texture_facts(spec: dict) -> dict | None:
    """Compact receipt for a textured material spec, or ``None``.

    Never raises and never stats the filesystem: the worker may serialise this
    into a response and the host may forward it verbatim.
    """
    if not isinstance(spec, dict):
        return None
    block = spec.get("texture")
    if not isinstance(block, dict):
        if not spec.get("texture_set"):
            return None
        block = {"id": spec.get("texture_set"), "maps": {}, "source": "unresolved"}
    maps = block.get("maps") or {}
    return {
        "id": block.get("id"),
        "dir": block.get("dir"),
        "size": block.get("size"),
        "tileable": block.get("tileable"),
        "maps": sorted(maps),
        "source": block.get("source"),
        "normal_convention": block.get("normal_convention"),
        "pbr": dict(block.get("pbr") or {}),
        "uv": dict(spec.get("uv") or {}),
        "normal_strength": spec.get("normal_strength"),
        "tint": spec.get("tint"),
        "roughness_mul": spec.get("roughness_mul"),
    }
