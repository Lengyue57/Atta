# -*- coding: utf-8 -*-
"""Spec v3 -- the single declarative input to Atta.

A spec says *what* to build, never *how*: either a curated ``recipe`` family or
a general feature ``graph``.  This module owns four verbs:

    validate_spec(spec)  -> {ok, spec, errors}
    propose_spec(spec)   -> {ok, spec, assumption_ledger, ...}
    compile_spec(spec)   -> {ok, compiled, errors}
    spec_hash(spec)      -> sha256 of canonical JSON

Units are millimetres and degrees at this boundary.  Recipe parameters are
resolved by ``Recipe.resolve`` (values in metres); feature graph args are not
evaluated here.

Assumptions are **design defaults**, never measurements.  Every inferred value
lands in an explicit ledger and is also written as a human-readable string in
``spec.assumptions``.  ``validate`` never invents assumptions.

``atta.recipes`` and ``atta.features`` are unfinished neighbours.  They are
imported lazily, inside functions, and a missing module becomes an actionable
``AttaError`` (surfaced as an ordinary validation error by public verbs).
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import re
import sys
from collections.abc import Mapping
from typing import Any

from .errors import AttaError
from .version import SCHEMA_SPEC

SCHEMA = SCHEMA_SPEC  # "atta.spec/v3"

KIND_RECIPE = "recipe"
KIND_GRAPH = "graph"
KIND_ASSEMBLY = "assembly"

MATE_TYPES = (
    "coincident", "concentric", "axis", "plane", "distance", "angle",
    "tangent", "gear", "insert", "parallel", "perpendicular", "lock",
    "point_on_plane",
)

QUALITY = ("draft", "review", "deliver")

#: The single vocabulary is the material preset registry; spec validation must
#: not drift from what the Blender worker can actually resolve.
try:
    from . import materials as _materials
    MATERIALS = frozenset(_materials.names())
except Exception:  # pragma: no cover - materials is a sibling module
    MATERIALS = frozenset({
        "ceramic", "porcelain", "quartz", "glass", "glass_frosted",
        "steel", "brushed_steel", "stainless", "aluminum", "aluminium",
        "brass", "copper", "gold", "plastic", "abs", "nylon", "wood",
        "powder_blue", "engine_outer", "dark",
    })

CONSTRAINT_DOCS: dict[str, str] = {
    "bbox": "{x:[min,max], y:[min,max], z:[min,max]} or a target value per axis (mm)",
    "volume_mm3": "[min,max] or a target volume (mm^3)",
    "min_wall_mm": "number; ray-sampled wall thickness (mm)",
    "genus": "integer; expected topology genus",
    "closed": "bool; mesh must be watertight",
    "n_components": "integer; expected number of connected components",
    "dihedral_max_deg": "number; maximum face-angle jump (degrees)",
    "curvature_p95_deg": "number; 95th percentile curvature (degrees)",
    "clearance": "[{a,b,min}] minimum distance between parts (mm)",
    "interference": "[{a,b,max}] allowed overlap volume (mm^3)",
    "interfaces": "[{a,b,max_mm3}] designed press-fit/overlap allowance",
    "envelope": "{min:[x,y,z], max:[x,y,z]} world envelope",
    "holes": "[{diameter, tolerance, count}] expected cylindrical holes",
    "render": "{res:[w,h], samples:n, require_ndc:true} render contract",
    "silhouette_iou": "[{view, outline_mm|target_model_id, min_iou}] form gate: "
                      "built silhouette vs a declared outline (0..1 IoU)",
    "mesh_agreement": "{target_model_id, chamfer_max_mm, hausdorff_p95_max_mm, "
                      "volumetric_iou_min} 3D mesh agreement vs reference model",
    "max_self_intersection_ratio": "number in 0..1; maximum fraction of "
                                   "self-intersecting triangle pairs",
}

#: Human-facing finish keys.  ``*_mm`` is millimetres, ``*_deg`` degrees.
FINISH_DOCS: dict[str, str] = {
    "smooth": "bool; enable smooth shading",
    "smooth_deg": "number 0..180; smooth shading angle",
    "smooth_angle_deg": "number 0..180; alias of smooth_deg",
    "bevel_mm": "number >= 0; bevel width in mm",
    "bevel_segments": "integer >= 1; bevel segments",
    "bevel_angle_deg": "number 0..180; bevel angle limit",
    "material": "optional material override; prefer spec.material",
}

_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_MAX_LIST_ROWS = {"clearance", "interference", "interfaces", "holes"}

#: Keys that may sit either in a feature's ``args`` or at its top level.
_TOP_LEVEL_FEATURE_KEYS = {"id", "op", "args", "inputs", "tags"}

_SKETCH_OPS = frozenset({"extrude", "revolve"})
_TAGGABLE_OPS = frozenset({
    "extrude", "revolve", "box", "cylinder", "sphere", "cone", "torus",
})
_SEGMENT_TYPES = frozenset({
    "line", "arc", "tangent_arc", "circle", "start", "move", "close",
})
_EDGE_SELECTOR_KEYS = frozenset({
    "parallel", "kind", "at", "expect_count",
    "common_edge_of", "on_face", "from_feature",
})
_EDGE_CRITERIA_KEYS = frozenset({
    "parallel", "kind", "at", "common_edge_of", "on_face", "from_feature",
})
_BREP_CAPABLE_OPS = frozenset({
    "box", "cylinder", "sphere", "cone", "torus",
    "revolve", "extrude",
    "union", "difference", "intersection",
    "transform", "mirror", "pattern_linear", "pattern_polar",
    "fillet", "chamfer",
})


# --------------------------------------------------------------------- helpers

def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_num(value: Any) -> bool:
    """Whether a JSON number is representable as a finite kernel float."""
    if not _is_num(value):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError, TypeError):
        return False


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a dataclass-like object or a dict."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _err(errors: list[dict], code: str, path: str, message: str, fix: str) -> None:
    errors.append({"code": str(code), "path": str(path),
                   "message": str(message), "fix": str(fix)})


def _err_from_atta(exc: AttaError, *, path: str | None = None) -> dict:
    return {"code": exc.code,
            "path": path or exc.path or "$",
            "message": exc.message,
            "fix": exc.fix or "Fix the underlying module or input and retry."}


def _load_recipes():
    """Lazy import so a missing ``atta.recipes`` never breaks import time."""
    mod = sys.modules.get("atta.recipes")
    if mod is not None:
        return mod
    try:
        return importlib.import_module("atta.recipes")
    except Exception as exc:  # noqa: BLE001 - any import failure is the message
        raise AttaError(
            "recipes_unavailable",
            "atta.recipes is not available: %s" % exc,
            path="$.model.id",
            fix="Install/implement atta.recipes (get(recipe_id)), "
                "or use model.kind='graph'.",
        ) from exc


def _get_recipe(recipe_id: str):
    mod = _load_recipes()
    getter = getattr(mod, "get", None)
    if not callable(getter):
        raise AttaError(
            "recipes_unavailable",
            "atta.recipes has no callable get(recipe_id)",
            path="$.model.id",
            fix="Expose get(recipe_id) -> Recipe from atta.recipes.",
        )
    try:
        return getter(recipe_id)
    except Exception as exc:  # noqa: BLE001 - registry may raise its own error
        raise AttaError(
            "unknown_recipe",
            "unknown recipe %r: %s" % (recipe_id, exc),
            path="$.model.id",
            fix="Call atta_catalog for recipe ids, or fix model.id.",
        ) from exc


def _load_feature_ops() -> dict:
    """Lazy import of the feature registry."""
    mod = sys.modules.get("atta.features")
    if mod is None:
        try:
            mod = importlib.import_module("atta.features")
        except Exception as exc:  # noqa: BLE001
            raise AttaError(
                "features_unavailable",
                "atta.features is not available: %s" % exc,
                path="$.model.features",
                fix="Install/implement atta.features.FEATURE_OPS, "
                    "or use model.kind='recipe'.",
            ) from exc
    ops = getattr(mod, "FEATURE_OPS", None)
    if not isinstance(ops, Mapping):
        raise AttaError(
            "features_unavailable",
            "atta.features.FEATURE_OPS is missing or not a mapping",
            path="$.model.features",
            fix="Expose FEATURE_OPS: {op: {required, optional, doc}}.",
        )
    return dict(ops)


# ---------------------------------------------------------------- JSON / hash

def canonical_json(value: Any) -> str:
    """Sorted keys, no insignificant whitespace, no NaN/Infinity."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def spec_hash(spec: Any) -> str:
    """sha256 of canonical JSON.  ``assumptions`` are part of the document."""
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------- validation

def _validate_finish(raw: Any, errors: list[dict]) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        _err(errors, "bad_finish", "$.finish", "finish must be an object",
             "Use {} or {bevel_mm: 0.8, smooth_deg: 35}.")
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        path = "$.finish.%s" % key
        if key not in FINISH_DOCS:
            _err(errors, "unknown_finish_key", path,
                 "unknown finish key %r" % (key,),
                 "Allowed keys: %s." % ", ".join(sorted(FINISH_DOCS)))
            continue
        if key == "smooth":
            if not isinstance(value, bool):
                _err(errors, "bad_finish", path, "smooth must be a bool",
                     "Use true or false.")
                continue
        elif key == "bevel_segments":
            if not _is_num(value) or not float(value).is_integer() or value < 1:
                _err(errors, "bad_finish", path,
                     "bevel_segments must be an integer >= 1",
                     "Use e.g. bevel_segments: 2.")
                continue
            value = int(value)
        elif key == "bevel_mm":
            if not _is_num(value) or not math.isfinite(float(value)) or value < 0:
                _err(errors, "bad_finish", path,
                     "bevel_mm must be a finite number >= 0",
                     "Use e.g. bevel_mm: 0.8.")
                continue
            value = float(value)
        elif key == "material":
            if not isinstance(value, str) or not value.strip():
                _err(errors, "bad_finish", path,
                     "finish.material must be a non-empty string",
                     "Prefer spec.material; otherwise give a material name.")
                continue
        else:  # smooth_deg / smooth_angle_deg / bevel_angle_deg
            if not _is_num(value) or not math.isfinite(float(value)) or not 0 <= value <= 180:
                _err(errors, "bad_finish", path,
                     "%s must be a number in 0..180" % key,
                     "Use e.g. %s: 35." % key)
                continue
            value = float(value)
        out[key] = value
    return out


def _validate_constraint_value(key: str, value: Any, errors: list[dict]) -> None:
    path = "$.constraints.%s" % key
    if key == "bbox":
        if not isinstance(value, Mapping):
            _err(errors, "bad_constraint", path, "bbox must be an object",
                 "Use {x:[min,max], y:[min,max], z:[min,max]}.")
        return
    if key in _MAX_LIST_ROWS:
        if not isinstance(value, (list, tuple)):
            _err(errors, "bad_constraint", path, "%s must be an array" % key,
                 "Wrap rows in a list.")
            return
        for i, row in enumerate(value):
            if not isinstance(row, Mapping):
                _err(errors, "bad_constraint", "%s[%d]" % (path, i),
                     "%s row must be an object" % key,
                     "Use {a:..., b:..., min|max|max_mm3:...}.")
                continue
            if key != "holes" and ("a" not in row or "b" not in row):
                _err(errors, "bad_constraint", "%s[%d]" % (path, i),
                     "%s row needs a and b" % key,
                     "Name the two parts as a and b.")
        return
    if key == "envelope":
        if not isinstance(value, Mapping):
            _err(errors, "bad_constraint", path, "envelope must be an object",
                 "Use {min:[x,y,z], max:[x,y,z]}.")
        return
    if key in {"genus", "n_components"}:
        if not _is_num(value) or not float(value).is_integer() or value < 0:
            _err(errors, "bad_constraint", path,
                 "%s must be a non-negative integer" % key,
                 "Use e.g. %s: 0." % key)
        return
    if key == "closed":
        if not isinstance(value, bool):
            _err(errors, "bad_constraint", path, "closed must be a bool",
                 "Use true or false.")
        return
    if key == "volume_mm3":
        if _is_num(value):
            return
        if not isinstance(value, (list, tuple)) or len(value) != 2 or \
                not all(_is_num(v) for v in value):
            _err(errors, "bad_constraint", path,
                 "volume_mm3 must be a number or [min,max]",
                 "Use e.g. [100000, 900000].")
        return
    if key == "render":
        if not isinstance(value, Mapping):
            _err(errors, "bad_constraint", path, "render must be an object",
                 "Use {res:[w,h], samples:n, require_ndc:true}.")
            return
        return
    if key == "silhouette_iou":
        rows = value if isinstance(value, (list, tuple)) else [value]
        if not rows:
            _err(errors, "bad_constraint", path,
                 "silhouette_iou must have at least one row",
                 "Use [{view:'front', outline_mm:[[x,z],...], min_iou:0.9}].")
            return
        for i, row in enumerate(rows):
            where = path if len(rows) == 1 else "%s[%d]" % (path, i)
            if not isinstance(row, Mapping):
                _err(errors, "bad_constraint", where,
                     "silhouette_iou row must be an object",
                     "Use {view:'front', outline_mm:[[x,z],...], min_iou:0.9}.")
                continue
            view = row.get("view", "front")
            if not isinstance(view, str) or view.lower() not in (
                    "front", "top", "right"):
                _err(errors, "bad_constraint", where + ".view",
                     "view must be front, top or right",
                     "Use view: 'front' (x-z), 'top' (x-y) or 'right' (y-z).")
            if row.get("outline_mm") is None and not row.get("target_model_id"):
                _err(errors, "bad_constraint", where,
                     "row needs outline_mm or target_model_id",
                     "Pass outline_mm=[[u,v],...] in mm, or target_model_id.")
                continue
            outline = row.get("outline_mm")
            if outline is not None:
                if not _is_mat(outline, 2, 3):
                    _err(errors, "bad_constraint", where + ".outline_mm",
                         "outline_mm must be a closed 2D loop of >= 3 points",
                         "Use [[u,v], ...] in mm.")
            need = row.get("min_iou", 0.9)
            if not _is_num(need) or not (0.0 <= float(need) <= 1.0):
                _err(errors, "bad_constraint", where + ".min_iou",
                     "min_iou must be a number in 0..1",
                     "Use min_iou: 0.90.")
        return
    if key == "mesh_agreement":
        if not isinstance(value, Mapping) or not value:
            _err(errors, "bad_constraint", path,
                 "mesh_agreement must be a non-empty object",
                 "Use {target_model_id:'mdl_...', chamfer_max_mm:0.5, "
                 "hausdorff_p95_max_mm:1.0, volumetric_iou_min:0.9}.")
            return
        allowed = {
            "target_model_id", "chamfer_max_mm",
            "hausdorff_p95_max_mm", "volumetric_iou_min",
        }
        for k, v in value.items():
            sub_path = "%s.%s" % (path, k)
            if k not in allowed:
                _err(errors, "bad_constraint", sub_path,
                     "unknown mesh_agreement field %r" % (k,),
                     "Allowed fields: %s." % ", ".join(sorted(allowed)))
            elif k == "target_model_id":
                if not isinstance(v, str) or not v.strip():
                    _err(errors, "bad_constraint", sub_path,
                         "target_model_id must be a non-empty string",
                         "Pass the reference model_id.")
            elif k in {"chamfer_max_mm", "hausdorff_p95_max_mm"}:
                if not _is_finite_num(v) or float(v) < 0.0:
                    _err(errors, "bad_constraint", sub_path,
                         "%s must be a finite number >= 0" % k,
                         "Use e.g. %s: 0.5." % k)
            elif k == "volumetric_iou_min":
                if not _is_finite_num(v) or not (0.0 <= float(v) <= 1.0):
                    _err(errors, "bad_constraint", sub_path,
                         "volumetric_iou_min must be a number in 0..1",
                         "Use e.g. volumetric_iou_min: 0.9.")
        return
    if key == "max_self_intersection_ratio":
        if not _is_finite_num(value) or not (0.0 <= float(value) <= 1.0):
            _err(errors, "bad_constraint", path,
                 "max_self_intersection_ratio must be a number in 0..1",
                 "Use e.g. max_self_intersection_ratio: 0.01.")
        return
    if key in {"min_wall_mm", "dihedral_max_deg", "curvature_p95_deg"}:
        if not _is_num(value) or not math.isfinite(float(value)):
            _err(errors, "bad_constraint", path, "%s must be a finite number" % key,
                 "Use a number; angles are degrees.")


def _validate_constraints(raw: Any, errors: list[dict]) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        _err(errors, "bad_constraints", "$.constraints",
             "constraints must be an object",
             "Use {} or a map of constraint keys.")
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in CONSTRAINT_DOCS:
            _err(errors, "unknown_constraint", "$.constraints.%s" % key,
                 "unknown constraint %r" % (key,),
                 "Allowed keys: %s." % ", ".join(sorted(CONSTRAINT_DOCS)))
            continue
        _validate_constraint_value(str(key), value, errors)
        out[str(key)] = value
    return out


def _validate_assumptions(raw: Any, errors: list[dict]) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        _err(errors, "bad_assumptions", "$.assumptions",
             "assumptions must be an array of strings",
             'Use [] or ["one human-readable statement"].')
        return []
    out: list[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            _err(errors, "bad_assumption", "$.assumptions[%d]" % i,
                 "assumption must be a non-empty string",
                 "Assumptions are human-readable strings; provenance lives "
                 "in the propose ledger.")
            continue
        out.append(item)
    return out


def _validate_recipe_params(model: dict, recipe: Any, errors: list[dict]) -> None:
    raw = model.get("params")
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        _err(errors, "bad_params", "$.model.params",
             "model.params must be an object",
             "Use {} or {param_name: value}.")
        model["params"] = {}
        return
    specs = _attr(recipe, "params", {})
    if not isinstance(specs, Mapping):
        _err(errors, "bad_recipe", "$.model.id",
             "recipe.params must be a mapping of ParamSpec",
             "Fix the recipe definition.")
        model["params"] = dict(raw)
        return

    norm: dict[str, Any] = {}
    for name, value in raw.items():
        path = "$.model.params.%s" % name
        if not isinstance(name, str) or not _ID_RE.match(name):
            _err(errors, "bad_param_name", path,
                 "invalid parameter name %r" % (name,),
                 "Use letters/digits/underscore, not starting with a digit.")
            continue
        ps = specs.get(name)
        if ps is None:
            _err(errors, "unknown_param", path,
                 "unknown parameter %r for recipe %r"
                 % (name, _attr(recipe, "id", "?")),
                 "Known params: %s." % (", ".join(sorted(specs)) or "(none)"))
            continue
        typ = str(_attr(ps, "type", "float") or "float")
        ok, norm_value, why = _coerce_param(value, ps, typ)
        if not ok:
            _err(errors, "bad_param", path, why,
                 "Give a %s value; see atta_catalog for the recipe schema." % typ)
            continue
        lo, hi = _attr(ps, "min", None), _attr(ps, "max", None)
        if _is_num(norm_value):
            if _is_num(lo) and norm_value < lo:
                _err(errors, "param_out_of_range", path,
                     "%s=%s is below minimum %s" % (name, norm_value, lo),
                     "Use %s >= %s." % (name, lo))
                continue
            if _is_num(hi) and norm_value > hi:
                _err(errors, "param_out_of_range", path,
                     "%s=%s is above maximum %s" % (name, norm_value, hi),
                     "Use %s <= %s." % (name, hi))
                continue
        norm[name] = norm_value
    model["params"] = norm


def _coerce_param(value: Any, ps: Any, typ: str) -> tuple[bool, Any, str]:
    if typ == "bool":
        if not isinstance(value, bool):
            return False, None, "expected a bool, got %r" % (value,)
        return True, value, ""
    if typ == "int":
        if not _is_num(value) or not float(value).is_integer():
            return False, None, "expected an integer, got %r" % (value,)
        return True, int(value), ""
    if typ == "enum":
        allowed = list(_attr(ps, "enum", []) or [])
        if value not in allowed:
            return False, None, "expected one of %r, got %r" % (allowed, value)
        return True, value, ""
    if typ == "str":
        if not isinstance(value, str):
            return False, None, "expected a string, got %r" % (value,)
        return True, value, ""
    if not _is_num(value) or not math.isfinite(float(value)):
        return False, None, "expected a finite number, got %r" % (value,)
    return True, float(value), ""


def _validate_recipe(model: dict, errors: list[dict]) -> None:
    rid = model.get("id")
    if not isinstance(rid, str) or not rid.strip():
        _err(errors, "bad_recipe_id", "$.model.id",
             "model.id must be a non-empty string",
             "Use a recipe id from atta_catalog, e.g. product.mug.")
        return

    preset = model.get("preset", "nominal")
    if preset is None:
        preset = "nominal"
    model["preset"] = preset
    if not isinstance(preset, str) or not preset.strip():
        _err(errors, "bad_preset", "$.model.preset",
             "model.preset must be a non-empty string",
             "Use 'nominal' or a preset listed by atta_catalog.")
        return

    try:
        recipe = _get_recipe(rid)
    except AttaError as exc:
        errors.append(_err_from_atta(exc))
        return

    _validate_recipe_params(model, recipe, errors)

    presets = _attr(recipe, "presets", {}) or {}
    if preset != "nominal" and preset not in presets:
        _err(errors, "unknown_preset", "$.model.preset",
             "unknown preset %r for recipe %r" % (preset, rid),
             "Known presets: %s." % (", ".join(sorted(presets)) or "(none)"))


def _cycle_nodes(ids: list[str], deps: dict[str, list[str]]) -> set[str]:
    state: dict[str, int] = {}
    found: set[str] = set()

    def visit(node: str, stack: list[str]) -> None:
        state[node] = 1
        stack.append(node)
        for dep in deps.get(node, []):
            if dep not in deps:
                continue
            if state.get(dep) == 1:
                found.update(stack[stack.index(dep):])
            elif state.get(dep) is None:
                visit(dep, stack)
        stack.pop()
        state[node] = 2

    for node in ids:
        if state.get(node) is None:
            visit(node, [])
    return found


def _is_deferred(value: Any) -> bool:
    """True for ``{"$param": ...}`` / ``{"$expr": ...}`` placeholders."""
    return isinstance(value, Mapping) and ("$param" in value or "$expr" in value)


def _has_deferred(value: Any) -> bool:
    """True when any scalar inside a nested arg is a deferred placeholder.

    A ``size: [{"$expr": "w"}, 10, 10]`` is legal but its numbers are unknown
    until evaluation, so type validation has to stand down for that argument.
    """
    if _is_deferred(value):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_deferred(v) for v in value)
    if isinstance(value, Mapping):
        return any(_has_deferred(v) for v in value.values())
    return False


def _is_num_list(value: Any, n: int | None = None) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if n is not None and len(value) != n:
        return False
    return all(_is_num(v) and math.isfinite(float(v)) for v in value)


def _is_mat(value: Any, cols: int, min_rows: int) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) >= min_rows
            and all(_is_num_list(row, cols) for row in value))


def _arg_kind_ok(value: Any, kind: Any) -> tuple[bool, str]:
    """Check one feature arg against a declared kind.

    Returns ``(ok, human-readable expectation)``.  ``{"$param": ...}`` and
    ``{"$expr": ...}`` are opaque here: their value is only known after
    evaluation, so they are accepted and left to ``features.evaluate``.
    """
    if isinstance(value, Mapping) and ("$param" in value or "$expr" in value):
        return True, "a number, expression or $param reference"
    if _has_deferred(value):
        # Contains $param/$expr somewhere: its concrete type is only known
        # after evaluate() resolves params, so defer rather than false-fail.
        return True, ""
    if isinstance(kind, (tuple, list)):  # ("int", min)
        base = str(kind[0])
        if base == "int":
            minimum = int(kind[1]) if len(kind) > 1 else None
            if isinstance(value, bool) or not _is_num(value):
                return False, "an integer%s" % (
                    "" if minimum is None else " >= %d" % minimum)
            if not float(value).is_integer():
                return False, "a whole number"
            if minimum is not None and int(value) < minimum:
                return False, "an integer >= %d" % minimum
            return True, ""
        return _arg_kind_ok(value, base)
    kind = str(kind)
    if kind in ("anything", ""):
        return True, ""
    if kind == "number":
        if isinstance(value, bool) or not _is_num(value) \
                or not math.isfinite(float(value)):
            return False, "a finite number"
        return True, ""
    if kind == "length":
        if isinstance(value, bool) or not _is_num(value) \
                or not math.isfinite(float(value)):
            return False, "a finite length (model units)"
        return True, ""
    if kind == "bool":
        return (isinstance(value, bool), "true or false")
    if kind == "vec3":
        return (_is_num_list(value, 3), "3 finite numbers [x, y, z]")
    if kind == "vec3_free":
        return (_is_num_list(value, 3), "3 finite numbers [x, y, z]")
    if kind == "mat2":
        return (_is_mat(value, 2, 2), "a list of >= 2 [u, v] points")
    if kind == "mat3":
        return (_is_mat(value, 3, 2), "a list of >= 2 [x, y, z] points")
    if kind == "rings":
        return (isinstance(value, (list, tuple)) and len(value) >= 2
                and all(_is_mat(r, 3, 3) for r in value),
                "a list of >= 2 rings, each >= 3 [x, y, z] points")
    if kind == "loops":
        return (isinstance(value, (list, tuple))
                and all(_is_mat(loop, 2, 3) for loop in value),
                "a list of closed 2D loops [[x, y], ...]")
    if kind == "curves":
        return (isinstance(value, (list, tuple)) and len(value) >= 2
                and all(_is_mat(c, 3, 2) for c in value),
                "a list of >= 2 curves of [x, y, z] points")
    if kind == "curves4":
        return (isinstance(value, (list, tuple)) and len(value) == 4
                and all(_is_mat(c, 3, 2) for c in value),
                "exactly 4 curves (bottom, right, top, left)")
    if kind.startswith("enum"):
        allowed = kind.split(":", 1)[1].split("|") if ":" in kind else None
        if not isinstance(value, str):
            return False, ("one of: %s" % "|".join(allowed) if allowed
                           else "a string")
        if allowed is not None and value not in allowed:
            return False, "one of: %s" % ", ".join(allowed)
        return True, ""
    return True, ""


def _validate_tags(raw: Any, path: str, errors: list[dict]) -> None:
    if raw is None or _has_deferred(raw):
        return
    if not isinstance(raw, Mapping):
        _err(errors, "bad_arg", path, "tags must be an object {slot: tag_name}",
             "Use e.g. {\"end\": \"cap\", \"start\": \"base\"}.")
        return
    for k, v in raw.items():
        if not isinstance(k, str) or not k.strip():
            _err(errors, "bad_arg", path, "tag keys must be non-empty strings",
                 "Use string face/feature tag slot names.")
        elif not isinstance(v, str) or not v.strip():
            _err(errors, "bad_arg", "%s.%s" % (path, k),
                 "tag value for %r must be a non-empty string" % k,
                 "Give a non-empty tag name string.")


def _validate_segments(raw: Any, path: str, errors: list[dict]) -> Any:
    if _has_deferred(raw):
        return raw
    if not isinstance(raw, (list, tuple)) or not raw:
        _err(errors, "bad_arg", path,
             "segments must be a non-empty array of segment objects",
             "Use [{op:'line', to:[x,y]}, {op:'arc', ...}] or circle.")
        return raw
    normalized: list[Any] = []
    for i, seg in enumerate(raw):
        sp = "%s[%d]" % (path, i)
        if not isinstance(seg, Mapping):
            _err(errors, "bad_arg", sp, "segment must be an object",
                 "Use {op:'line'|'arc'|'tangent_arc'|'circle'|'start'|'move'|'close', ...}.")
            normalized.append(seg)
            continue
        seg_out = copy.deepcopy(dict(seg))
        seg_type = seg_out.get("op") or seg_out.get("type") or seg_out.get("kind")
        seg_norm = seg_type.strip().lower() if isinstance(seg_type, str) else seg_type
        if not isinstance(seg_norm, str) or seg_norm not in _SEGMENT_TYPES:
            _err(errors, "bad_arg", sp,
                 "segment op must be one of %s, got %r"
                 % (", ".join(sorted(_SEGMENT_TYPES)), seg_type),
                 "Set op to line, arc, tangent_arc, circle, start, move, or close.")
        else:
            if not seg_out.get("op"):
                seg_out["op"] = seg_norm
                seg_out.pop("type", None)
                seg_out.pop("kind", None)
            else:
                seg_out["op"] = seg_norm
        if "tag" in seg_out and seg_out["tag"] is not None:
            if not isinstance(seg_out["tag"], str) or not seg_out["tag"].strip():
                _err(errors, "bad_arg", sp + ".tag",
                     "segment tag must be a non-empty string",
                     "Give a non-empty tag string or omit tag.")
        normalized.append(seg_out)
    return normalized


def _validate_edge_selector(raw: Any, path: str, errors: list[dict]) -> None:
    if _has_deferred(raw):
        return
    if raw == "all":
        return
    if not isinstance(raw, Mapping) or not raw:
        _err(errors, "bad_selector", path,
             "edges must be 'all' or a geometric selector object",
             "Use 'all' or {parallel|kind|at|common_edge_of|on_face|from_feature}.")
        return
    unknown = set(raw) - _EDGE_SELECTOR_KEYS
    if unknown or not (set(raw) & _EDGE_CRITERIA_KEYS):
        _err(errors, "bad_selector", path,
             "selector needs one of %s without unknown keys"
             % "/".join(sorted(_EDGE_CRITERIA_KEYS)),
             "Use 'all' or {parallel|kind|at|common_edge_of|on_face|from_feature}.")
        return
    for key, val in raw.items():
        sp = "%s.%s" % (path, key)
        if val is None:
            _err(errors, "bad_selector", sp,
                 "selector condition %s must not be null" % key,
                 "Remove %s or supply a valid value." % key)
            continue
        if key == "parallel" and val not in ("x", "y", "z"):
            _err(errors, "bad_selector", sp,
                 "parallel must be 'x', 'y', or 'z'",
                 "Use parallel: 'z'.")
        elif key == "kind" and val not in ("line", "circle"):
            _err(errors, "bad_selector", sp,
                 "kind must be 'line' or 'circle'",
                 "Use kind: 'line' or 'circle'.")
        elif key == "expect_count":
            if not _is_num(val) or not float(val).is_integer() or int(val) < 1:
                _err(errors, "bad_selector", sp,
                     "expect_count must be an integer >= 1",
                     "Use e.g. expect_count: 4.")
        elif key == "at":
            if (not isinstance(val, Mapping)
                    or set(val) - {"axis", "value", "tolerance"}
                    or not {"axis", "value"} <= set(val)
                    or val.get("axis") not in ("x", "y", "z")):
                _err(errors, "bad_selector", sp,
                     "at needs axis=x/y/z and value=number|min|max",
                     "Use {axis:'z', value:'max'}.")
            else:
                av = val["value"]
                if av not in ("min", "max") and not _is_finite_num(av):
                    _err(errors, "bad_selector", sp + ".value",
                         "at.value must be a finite number or 'min'/'max'",
                         "Pass a coordinate in model units or 'min'/'max'.")
                if "tolerance" in val and (
                        not _is_finite_num(val["tolerance"])
                        or float(val["tolerance"]) <= 0):
                    _err(errors, "bad_selector", sp + ".tolerance",
                         "at.tolerance must be a finite number > 0",
                         "Use e.g. tolerance: 1e-6.")
        elif key == "common_edge_of":
            if (not isinstance(val, (list, tuple)) or len(val) != 2
                    or not all(isinstance(t, str) and bool(t.strip()) for t in val)):
                _err(errors, "bad_selector", sp,
                     "common_edge_of must be [tagA, tagB] of two non-empty strings",
                     "Use e.g. common_edge_of: ['cap', 'side'].")
        elif key == "on_face":
            if not isinstance(val, str) or not val.strip():
                _err(errors, "bad_selector", sp,
                     "on_face must be a non-empty tag string",
                     "Use e.g. on_face: 'top'.")
        elif key == "from_feature":
            if not isinstance(val, str) or not val.strip():
                _err(errors, "bad_selector", sp,
                     "from_feature must be a non-empty feature id string",
                     "Use e.g. from_feature: 'plate'.")


def _validate_feature_args(op: str, args: Mapping, base: str,
                           docs: Mapping, errors: list[dict]) -> None:
    """Type-check feature args against the registry's declared kinds.

    DEFECT 4: validation used to stop at "the required key exists", so
    ``{"op": "box", "args": {"size": "big"}}`` validated ok and then exploded
    inside ``features.evaluate``.  Unknown extra args were accepted too, only
    to be rejected much later by the evaluator.  Catching both here means a
    spec can be trusted to at least *start* evaluating.
    """
    required = tuple(docs.get("required") or ())
    optional = tuple(docs.get("optional") or ())
    allowed = set(required) | set(optional)
    if op in _SKETCH_OPS:
        allowed.add("segments")
    if op in _TAGGABLE_OPS:
        allowed.add("tags")

    for key in args:
        if not isinstance(key, str):
            _err(errors, "bad_arg", "%s.args" % base,
                 "feature arg names must be strings, got %r" % (key,),
                 "Use a mapping of string keys; see atta_catalog for %s." % op)
            continue
        if key in _TOP_LEVEL_FEATURE_KEYS:
            if key == "tags" and op in _TAGGABLE_OPS:
                _validate_tags(args[key], "%s.args.tags" % base, errors)
            continue  # legal here: these live at the feature's top level
        if allowed and key not in allowed:
            _err(errors, "unknown_arg", "%s.args.%s" % (base, key),
                 "%s has no argument %r" % (op, key),
                 "Use one of: %s." % (", ".join(sorted(allowed)) or "(none)"))
            continue
        if key == "segments" and op in _SKETCH_OPS and args[key] is not None:
            if not (op == "revolve" and _is_num(args[key])):
                norm_segs = _validate_segments(args[key], "%s.args.segments" % base, errors)
                if isinstance(args, dict):
                    args["segments"] = norm_segs
        elif op in ("fillet", "chamfer") and key == "edges":
            _validate_edge_selector(args[key], "%s.args.edges" % base, errors)

    try:
        kinds = _load_arg_kinds(op)
    except AttaError:
        return
    for key, value in args.items():
        if not isinstance(key, str) or key not in kinds:
            continue
        if key == "segments" and op in _SKETCH_OPS and isinstance(value, (list, tuple)):
            continue
        ok, expected = _arg_kind_ok(value, kinds[key])
        if not ok:
            _err(errors, "bad_arg", "%s.args.%s" % (base, key),
                 "%s.%s must be %s, got %r" % (op, key, expected, value),
                 "Give %s a %s; see atta_catalog for %s." % (key, expected, op))


def _load_arg_kinds(op: str) -> dict:
    """Lazy read of ``features.arg_kinds``, so spec.py keeps its lazy import.

    ``atta.features`` is an unfinished neighbour: if it will not import, the
    caller already reports ``features_unavailable`` and arg kinds are simply
    unavailable rather than a second error.
    """
    feats = sys.modules.get("atta.features")
    if feats is None:
        importlib.import_module("atta.features")
        feats = sys.modules.get("atta.features")
    getter = getattr(feats, "arg_kinds", None)
    if not callable(getter):
        return {}
    return dict(getter(str(op)) or {})


def _validate_graph(model: dict, errors: list[dict]) -> None:
    raw_params = model.get("params", {})
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, Mapping):
        _err(errors, "bad_params", "$.model.params",
             "graph.params must be an object of numeric values",
             "Use {} or {w: 40, h: 20}; feature args can use {\"$param\":\"w\"}.")
        raw_params = {}
    params = {}
    for name, value in raw_params.items():
        if not isinstance(name, str) or not _ID_RE.match(name):
            _err(errors, "bad_param_name", "$.model.params",
                 "invalid param name %r" % (name,),
                 "Use letters/digits/underscore, not starting with a digit.")
            continue
        if not _is_num(value) or not math.isfinite(float(value)):
            _err(errors, "bad_param", "$.model.params.%s" % name,
                 "param must be a finite number", "Use a number in mm/degrees.")
            continue
        params[name] = value
    model["params"] = params

    units = model.get("units", "mm")
    if units != "mm":
        _err(errors, "bad_units", "$.model.units",
             "graph units must be 'mm', got %r" % (units,),
             "Convert feature arguments to mm and set units='mm'.")
    model["units"] = "mm"

    segments = model.get("segments", 64)
    if not _is_num(segments) or not float(segments).is_integer() or segments < 3:
        _err(errors, "bad_segments", "$.model.segments",
             "segments must be an integer >= 3",
             "Use e.g. segments: 64.")
    else:
        model["segments"] = int(segments)

    raw_feats = model.get("features")
    if not isinstance(raw_feats, (list, tuple)) or not raw_feats:
        _err(errors, "bad_features", "$.model.features",
             "graph.features must be a non-empty array",
             "Add at least one primitive feature, e.g. box.")
        raw_feats = []

    ops: dict | None = None
    try:
        ops = _load_feature_ops()
    except AttaError as exc:
        errors.append(_err_from_atta(exc, path="$.model.features"))

    features: list[dict] = []
    ids: list[str] = []
    known: set[str] = set()
    for i, raw in enumerate(raw_feats):
        base = "$.model.features[%d]" % i
        if not isinstance(raw, Mapping):
            _err(errors, "bad_feature", base, "feature must be an object",
                 'Use {"id":..., "op":..., "args":{...}}.')
            continue
        feat = copy.deepcopy(dict(raw))
        fid = feat.get("id")
        if not isinstance(fid, str) or not _ID_RE.match(fid):
            _err(errors, "bad_id", base + ".id",
                 "feature id is missing or invalid: %r" % (fid,),
                 "Use a unique id like 'plate' or 'cut_01'.")
        elif fid in known:
            _err(errors, "duplicate_id", base + ".id",
                 "duplicate feature id %r" % (fid,),
                 "Every feature id must be unique.")
        else:
            known.add(fid)
            ids.append(fid)

        op = feat.get("op")
        if not isinstance(op, str) or not op:
            _err(errors, "unknown_feature", base + ".op",
                 "feature op is missing",
                 "Use one of: %s." % (", ".join(sorted(ops or {})) or "(unavailable)"))
        elif ops is not None and op not in ops:
            _err(errors, "unknown_feature", base + ".op",
                 "unknown feature op %r" % (op,),
                 "Use one of: %s." % ", ".join(sorted(ops)))
        elif ops is not None:
            docs = ops.get(op) or {}
            required = docs.get("required", []) if isinstance(docs, Mapping) else []
            args = feat.get("args")
            args_map = args if isinstance(args, Mapping) else {}
            arg_keys = set(args_map)
            for key in required:
                if key in {"id", "op"}:
                    continue
                if op == "extrude" and key == "loop_xy" and isinstance(args_map.get("segments"), (list, tuple)):
                    continue
                if op == "revolve" and key == "profile_rz" and isinstance(args_map.get("segments"), (list, tuple)):
                    continue
                if key in _TOP_LEVEL_FEATURE_KEYS:
                    if key not in feat:
                        _err(errors, "bad_arg", base + "." + key,
                             "%s requires %r" % (op, key),
                             "Add %r; see atta_catalog for %s." % (key, op))
                elif key not in arg_keys:
                    _err(errors, "bad_arg", base + ".args.%s" % key,
                         "%s requires arg %r" % (op, key),
                         "Add %r to args; see atta_catalog for %s." % (key, op))
            if isinstance(args, Mapping):
                _validate_feature_args(op, args, base, docs, errors)

        if "tags" in feat and feat["tags"] is not None:
            _validate_tags(feat["tags"], base + ".tags", errors)

        args = feat.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, Mapping):
            _err(errors, "bad_arg", base + ".args",
                 "feature args must be an object",
                 "Use {} or {key: value}.")
            args = {}
        feat["args"] = copy.deepcopy(dict(args))

        inputs = feat.get("inputs", [])
        if inputs is None:
            inputs = []
        if not isinstance(inputs, (list, tuple)):
            _err(errors, "bad_arg", base + ".inputs",
                 "feature inputs must be an array of ids",
                 "Use [] or [\"earlier_feature_id\"].")
            inputs = []
        feat["inputs"] = list(inputs)
        features.append(feat)

    # References (checked after all ids are known, so forward refs can cycle).
    for i, feat in enumerate(features):
        base = "$.model.features[%d]" % i
        for j, ref in enumerate(feat.get("inputs", [])):
            if not isinstance(ref, str):
                _err(errors, "bad_arg", "%s.inputs[%d]" % (base, j),
                     "input reference must be a feature id string",
                     "Use an existing feature id.")
            elif ref not in known:
                _err(errors, "unknown_input", "%s.inputs[%d]" % (base, j),
                     "input refers to unknown feature %r" % (ref,),
                     "Known ids: %s." % (", ".join(sorted(known)) or "(none)"))
        feat_args = feat.get("args")
        edges_sel = feat_args.get("edges") if isinstance(feat_args, Mapping) else None
        if isinstance(edges_sel, Mapping):
            from_fid = edges_sel.get("from_feature")
            if isinstance(from_fid, str) and from_fid.strip() and from_fid not in known:
                _err(errors, "unknown_input", base + ".args.edges.from_feature",
                     "from_feature refers to unknown feature %r" % (from_fid,),
                     "Known ids: %s." % (", ".join(sorted(known)) or "(none)"))

    deps = {feat["id"]: [r for r in feat.get("inputs", []) if isinstance(r, str)]
            for feat in features
            if isinstance(feat.get("id"), str) and feat["id"] in known}
    cycle = _cycle_nodes(ids, deps)
    if cycle:
        _err(errors, "cycle", "$.model.features",
             "feature graph has a cycle through: %s" % ", ".join(sorted(cycle)),
             "Break the cycle; a feature graph must be a DAG.")

    output = model.get("output")
    if output is None:
        if ids:
            model["output"] = ids[-1]
    elif not isinstance(output, str) or output not in known:
        _err(errors, "unknown_output_ref", "$.model.output",
             "output refers to unknown feature %r" % (output,),
             "Known ids: %s." % (", ".join(sorted(known)) or "(none)"))
    model["features"] = features


def _prefix_errors(errors: list[dict], prefix: str) -> list[dict]:
    out = []
    for err in errors:
        row = dict(err)
        path = str(row.get("path") or "$")
        row["path"] = prefix + path[1:] if path.startswith("$") else prefix + "." + path
        out.append(row)
    return out


def _vec3(raw) -> bool:
    return (isinstance(raw, (list, tuple)) and len(raw) == 3
            and all(_is_finite_num(x) for x in raw))


def _validate_transform(raw: Any, path: str, errors: list[dict]) -> None:
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        _err(errors, "bad_transform", path, "transform must be an object",
             'Use {"translate_mm":[0,0,0], "rotate_deg":[0,0,0]}.')
        return
    for key in ("translate_mm", "translate"):
        if key in raw and not _vec3(raw[key]):
            _err(errors, "bad_transform", path + "." + key,
                 "%s must be three finite numbers (mm)" % key,
                 "Use [x, y, z] in millimetres.")
    for key in ("rotate_deg", "rotate"):
        if key in raw and not _vec3(raw[key]):
            _err(errors, "bad_transform", path + "." + key,
                 "%s must be three finite numbers (degrees)" % key,
                 "Use [rx, ry, rz] in degrees.")
    if "matrix" in raw:
        m = raw["matrix"]
        ok = (isinstance(m, (list, tuple)) and len(m) == 4
              and all(isinstance(r, (list, tuple)) and len(r) == 4 for r in m)
              and all(_is_finite_num(v) for row in m for v in row))
        if not ok:
            _err(errors, "bad_transform", path + ".matrix",
                 "matrix must be a 4x4 finite array (translation in mm)",
                 "Provide a row-major 4x4 matrix, or use translate_mm/rotate_deg.")
        else:
            # Assembly poses are SE(3): scaling, shear, reflection and
            # projective matrices would invalidate volumes and port frames.
            r = [[float(v) for v in row[:3]] for row in m[:3]]
            orthogonal = all(abs(sum(r[k][i] * r[k][j] for k in range(3))
                                 - (1.0 if i == j else 0.0)) <= 1e-7
                             for i in range(3) for j in range(3))
            det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                   - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                   + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
            affine = all(abs(float(m[3][i]) - v) <= 1e-9
                         for i, v in enumerate((0, 0, 0, 1)))
            if not orthogonal or abs(det - 1.0) > 1e-7 or not affine:
                _err(errors, "bad_transform", path + ".matrix",
                     "assembly matrix must be a rigid transform (orthonormal "
                     "rotation, determinant +1, last row [0,0,0,1])",
                     "Apply scale/mirror inside the part graph; use a rigid "
                     "placement with translation in millimetres.")


def _validate_ports(raw: Any, path: str, errors: list[dict]) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        _err(errors, "bad_ports", path, "ports must be an array",
             "Use [] or [{name, origin_mm, axis, radius_mm}].")
        return []
    out = []
    seen = set()
    for i, item in enumerate(raw):
        base = "%s[%d]" % (path, i)
        if not isinstance(item, Mapping):
            _err(errors, "bad_port", base, "port must be an object",
                 'Use {"name":"axis_z","origin_mm":[0,0,0],"axis":[0,0,1]}.')
            continue
        port = copy.deepcopy(dict(item))
        name = port.get("name")
        if not isinstance(name, str) or not _ID_RE.match(name):
            _err(errors, "bad_port_name", base + ".name",
                 "port name is missing or invalid: %r" % (name,),
                 "Use letters/digits/underscore, e.g. 'axis_z'.")
        elif name in seen:
            _err(errors, "duplicate_port", base + ".name",
                 "duplicate port name %r" % name, "Use a unique port name.")
        else:
            seen.add(name)
        if "origin_mm" in port and not _vec3(port["origin_mm"]):
            _err(errors, "bad_port", base + ".origin_mm",
                 "origin_mm must be three finite numbers (mm)",
                 "Use [x, y, z] in millimetres.")
        oriented = "x_axis" in port or "tangent" in port
        if "x_axis" in port and "tangent" in port:
            _err(errors, "bad_port_frame", base,
                 "x_axis and tangent are aliases; provide only one",
                 "Use x_axis to declare the local frame X direction.")
        if oriented and "axis" not in port and "normal" in port:
            # Only the new oriented contract inherits normal as Z.  Legacy
            # normal-only ports keep their historical +Z axis.
            port["axis"] = copy.deepcopy(port["normal"])
        directions = {}
        for key in ("axis", "normal", "x_axis", "tangent"):
            if key in port:
                v = port[key]
                length = math.hypot(*v) if _vec3(v) else 0.0
                if not math.isfinite(length) or length <= 1e-9:
                    _err(errors, "bad_port", base + "." + key,
                         "%s must be a finite non-zero 3-vector" % key,
                         "Use a direction such as [0,0,1].")
                else:
                    directions[key] = [float(x) / length for x in v]
        if oriented:
            z = directions.get("axis", [0.0, 0.0, 1.0])
            normal = directions.get("normal")
            if normal is not None and ("axis" not in port or "axis" in directions):
                if sum(a * b for a, b in zip(z, normal)) < 1.0 - 1e-8:
                    _err(errors, "bad_port_frame", base + ".normal",
                         "an oriented port's normal must point along its axis (frame Z)",
                         "Use the same direction for axis and normal; facing belongs in attach.")
            x_key = "x_axis" if "x_axis" in port else "tangent"
            x = directions.get(x_key)
            if x is not None and ("axis" not in port or "axis" in directions):
                dot = sum(a * b for a, b in zip(x, z))
                projected = [a - dot * b for a, b in zip(x, z)]
                length = math.hypot(*projected)
                if length <= 1e-8:
                    _err(errors, "bad_port_frame", base + "." + x_key,
                         "%s is parallel or too close to the frame Z axis" % x_key,
                         "Give a transverse X direction; it is projected perpendicular to axis.")
                elif not ("x_axis" in port and "tangent" in port):
                    port["x_axis"] = [v / length for v in projected]
                    port.pop("tangent", None)
        if "radius_mm" in port and (not _is_finite_num(port["radius_mm"])
                                    or float(port["radius_mm"]) < 0):
            _err(errors, "bad_port", base + ".radius_mm",
                 "radius_mm must be a finite number >= 0", "Use e.g. 5.0.")
        out.append(port)
    return out


def _validate_part_source(part: dict, base: str, errors: list[dict]) -> None:
    sources = [k for k in ("recipe", "graph", "spec", "model_id")
               if part.get(k) is not None]
    if len(sources) != 1:
        _err(errors, "bad_part_source", base,
             "part needs exactly one of recipe/graph/spec/model_id; got %s"
             % (", ".join(sources) or "none"),
             "Use recipe+params, an inline graph, an inline spec, or model_id.")
        return
    src = sources[0]
    if src == "recipe":
        rid = part.get("recipe")
        if not isinstance(rid, str) or not rid.strip():
            _err(errors, "bad_recipe_id", base + ".recipe",
                 "recipe must be a non-empty recipe id",
                 "Use an id from atta_catalog.")
            return
        try:
            recipe = _get_recipe(rid)
        except AttaError as exc:
            errors.append(_err_from_atta(exc, path=base + ".recipe"))
            return
        sub = {"params": dict(part.get("params") or {}),
               "preset": part.get("preset", "nominal")}
        before = len(errors)
        _validate_recipe_params(sub, recipe, errors)
        for err in errors[before:]:
            err["path"] = base + str(err.get("path") or "").replace("$.model", "")
        part["params"] = sub["params"]
        presets = _attr(recipe, "presets", {}) or {}
        preset = sub.get("preset")
        if not isinstance(preset, str) or not preset.strip():
            _err(errors, "bad_preset", base + ".preset",
                 "preset must be a non-empty string", "Use 'nominal'.")
        elif preset != "nominal" and preset not in presets:
            _err(errors, "unknown_preset", base + ".preset",
                 "unknown preset %r for recipe %r" % (preset, rid),
                 "Known presets: %s." % (", ".join(sorted(presets)) or "(none)"))
    elif src == "graph":
        graph = part.get("graph")
        if not isinstance(graph, Mapping):
            _err(errors, "bad_graph", base + ".graph",
                 "graph must be an object with features",
                 'Use {"features":[{"id":"b","op":"box","args":{"size":[1,1,1]}}]}.')
            return
        sub = {"kind": "graph", "features": copy.deepcopy(graph.get("features")),
               "units": graph.get("units", "mm"),
               "segments": graph.get("segments", 64),
               "output": graph.get("output"),
               "params": part.get("params", graph.get("params", {}))}
        before = len(errors)
        _validate_graph(sub, errors)
        for err in errors[before:]:
            err["path"] = base + ".graph" + str(err.get("path") or "").replace(
                "$.model", "")
        part["graph"] = {
            "features": sub.get("features"),
            "units": sub.get("units"),
            "segments": sub.get("segments"),
            "output": sub.get("output"),
            "params": sub.get("params") or {},
        }
        if "params" in part:
            part["params"] = sub.get("params") or {}
    elif src == "spec":
        nested = part.get("spec")
        if not isinstance(nested, Mapping):
            _err(errors, "bad_part_spec", base + ".spec",
                 "spec must be an object", "Inline a model spec or use a graph.")
            return
        if (nested.get("model") or {}).get("kind") == KIND_ASSEMBLY \
                or nested.get("kind") == KIND_ASSEMBLY:
            _err(errors, "nested_assembly", base + ".spec",
                 "an assembly part cannot itself be an assembly",
                 "Flatten the parts into this assembly.")
            return
        v = validate_spec(nested)
        errors.extend(_prefix_errors(v.get("errors") or [], base + ".spec"))
        part["spec"] = v.get("spec") or nested
    else:
        mid = part.get("model_id")
        if not isinstance(mid, str) or not mid.strip():
            _err(errors, "bad_model_id", base + ".model_id",
                 "model_id must be a non-empty string",
                 "Use a model_id built in this workspace.")


_AUTO_PORT_NAMES = frozenset({"center", "axis_x", "axis_y", "axis_z"} | {
    "face_%s%s" % (axis, sign) for axis in "xyz" for sign in "+-"
})
_ATTACH_KEYS = frozenset({"target", "target_port", "source_port", "gap_mm",
                          "roll_deg", "facing"})


def _validate_attachments(parts: list[dict], path: str,
                          errors: list[dict]) -> list[str]:
    """Validate the small placement contract and return a stable DAG order.

    No geometry is built here: bbox port names are known independently of the
    mesh, and declared ports override those names.  Direct assembly callers
    use the same check before building any part.
    """
    by_id = {p["id"]: p for p in parts
             if isinstance(p, Mapping) and isinstance(p.get("id"), str)}
    deps = {pid: [] for pid in by_id}
    port_names = {}
    for pid, part in by_id.items():
        raw_ports = part.get("ports")
        raw_ports = raw_ports if isinstance(raw_ports, (list, tuple)) else []
        port_names[pid] = _AUTO_PORT_NAMES | {
            p.get("name") for p in raw_ports
            if isinstance(p, Mapping) and isinstance(p.get("name"), str)
        }
    for i, part in enumerate(parts):
        if not isinstance(part, Mapping) or part.get("attach") is None:
            continue
        base = "%s[%d].attach" % (path, i)
        attach = part["attach"]
        if not isinstance(attach, Mapping):
            _err(errors, "bad_attach", base, "attach must be an object",
                 "Use {target, target_port, source_port, gap_mm, roll_deg, facing}.")
            continue
        if part.get("transform") is not None:
            _err(errors, "bad_attach", base,
                 "attach and transform cannot both place one part",
                 "Choose attach, or bake the placement into transform.")
        for key in attach:
            if key not in _ATTACH_KEYS:
                _err(errors, "bad_attach", base + "." + str(key),
                     "unknown attach field %r" % key,
                     "Allowed fields: %s." % ", ".join(sorted(_ATTACH_KEYS)))
        target = attach.get("target")
        if not isinstance(target, str) or target not in by_id:
            _err(errors, "unknown_part", base + ".target",
                 "attach references unknown target part %r" % target,
                 "Known parts: %s." % ", ".join(sorted(by_id)))
        elif isinstance(part.get("id"), str) and part["id"] in deps:
            deps[part["id"]] = [target]
        for key, pid in (("target_port", target), ("source_port", part.get("id"))):
            name = attach.get(key)
            if not isinstance(name, str) or not name:
                _err(errors, "bad_port_ref", base + "." + key,
                     "%s must explicitly name a port" % key,
                     "Use a declared port or a bbox port such as face_z+.")
            elif isinstance(pid, str) and pid in port_names and name not in port_names[pid]:
                _err(errors, "unknown_port", base + "." + key,
                     "part %r has no port %r" % (pid, name),
                     "Available ports: %s." % ", ".join(sorted(port_names[pid])))
        for key in ("gap_mm", "roll_deg"):
            if key in attach and not _is_finite_num(attach[key]):
                _err(errors, "bad_attach", base + "." + key,
                     "%s must be a finite number" % key,
                     "Use millimetres for gap_mm and degrees for roll_deg.")
        if attach.get("facing", "opposite") not in ("same", "opposite"):
            _err(errors, "bad_attach", base + ".facing",
                 "facing must be 'same' or 'opposite'",
                 "Use opposite for facing interfaces, same for aligned frame Z axes.")
    cycle = _cycle_nodes(sorted(deps), deps)
    if cycle:
        _err(errors, "attach_cycle", path,
             "attach reference graph has a cycle through: %s" % ", ".join(sorted(cycle)),
             "Break the cycle; each attached part needs a path to a root placement.")
    # Lexical tie-breaking means part-list reordering cannot change placement
    # or the implicit solver anchor of an assembly using this new contract.
    remaining = set(deps)
    order = []
    while remaining:
        ready = sorted(pid for pid in remaining
                       if all(ref not in remaining for ref in deps[pid]))
        if not ready:
            break
        order.extend(ready)
        remaining.difference_update(ready)
    return order


def _validate_assembly(model: dict, errors: list[dict]) -> None:
    parts = model.get("parts")
    if not isinstance(parts, (list, tuple)) or not parts:
        _err(errors, "bad_parts", "$.model.parts",
             "assembly.parts must be a non-empty array",
             "Add at least one part with an id and a recipe/graph.")
        parts = []
    if len(parts) > 256:
        _err(errors, "too_many_parts", "$.model.parts",
             "assembly has %d parts; the limit is 256" % len(parts),
             "Split the assembly into subassemblies.")
    ids: set[str] = set()
    norm_parts = []
    for i, raw in enumerate(parts):
        base = "$.model.parts[%d]" % i
        if not isinstance(raw, Mapping):
            _err(errors, "bad_part", base, "part must be an object",
                 'Use {"id":"base","recipe":"part.bracket","params":{...}}.')
            continue
        part = copy.deepcopy(dict(raw))
        pid = part.get("id")
        if not isinstance(pid, str) or not _ID_RE.match(pid):
            _err(errors, "bad_part_id", base + ".id",
                 "part id is missing or invalid: %r" % (pid,),
                 "Use a unique id like 'base' or 'shaft_01'.")
        elif pid in ids:
            _err(errors, "duplicate_part_id", base + ".id",
                 "duplicate part id %r" % pid, "Every part id must be unique.")
        else:
            ids.add(pid)
        _validate_part_source(part, base, errors)
        _validate_transform(part.get("transform"), base + ".transform", errors)
        if "fixed" in part and not isinstance(part["fixed"], bool):
            _err(errors, "bad_fixed", base + ".fixed", "fixed must be a bool",
                 "Use true or false.")
        if "material" in part:
            m = part["material"]
            if not isinstance(m, str) or m.lower() not in MATERIALS:
                _err(errors, "bad_material", base + ".material",
                     "unknown material %r" % (m,),
                     "Use one of: %s." % ", ".join(sorted(MATERIALS)))
        if "quality" in part:
            q = part["quality"]
            if not isinstance(q, str) or q.lower() not in QUALITY:
                _err(errors, "bad_quality", base + ".quality",
                     "quality must be one of %s" % ", ".join(QUALITY),
                     "Use draft.")
        part["ports"] = _validate_ports(part.get("ports"), base + ".ports", errors)
        norm_parts.append(part)
    model["parts"] = norm_parts
    _validate_attachments(norm_parts, "$.model.parts", errors)

    mates = model.get("mates", [])
    if mates is None:
        mates = []
    if not isinstance(mates, (list, tuple)):
        _err(errors, "bad_mates", "$.model.mates", "mates must be an array",
             "Use [] or [{id,type,a,b}].")
        mates = []
    seen = set()
    norm_mates = []
    for i, raw in enumerate(mates):
        base = "$.model.mates[%d]" % i
        if not isinstance(raw, Mapping):
            _err(errors, "bad_mate", base, "mate must be an object",
                 'Use {"id":"m1","type":"concentric","a":"base","b":"shaft"}.')
            continue
        mate = copy.deepcopy(dict(raw))
        mid = mate.get("id")
        if not isinstance(mid, str) or not _ID_RE.match(mid):
            _err(errors, "bad_mate_id", base + ".id",
                 "mate id is missing or invalid: %r" % (mid,),
                 "Use a unique id like 'm1'.")
        elif mid in seen:
            _err(errors, "duplicate_mate_id", base + ".id",
                 "duplicate mate id %r" % mid, "Every mate id must be unique.")
        else:
            seen.add(mid)
        typ = mate.get("type")
        if not isinstance(typ, str) or typ.lower() not in MATE_TYPES:
            _err(errors, "bad_mate_type", base + ".type",
                 "unknown mate type %r" % (typ,),
                 "Use one of: %s." % ", ".join(MATE_TYPES))
        else:
            mate["type"] = typ.lower()
        for key in ("a", "b"):
            ref = mate.get(key)
            if not isinstance(ref, str) or ref not in ids:
                _err(errors, "unknown_part", base + "." + key,
                     "mate references unknown part %r" % (ref,),
                     "Known parts: %s." % (", ".join(sorted(ids)) or "(none)"))
        if mate.get("a") == mate.get("b") and mate.get("a") is not None:
            _err(errors, "self_mate", base, "a mate cannot reference one part twice",
                 "Use two distinct parts; use a lock mate for a rigid pair.")
        for key in ("port_a", "port_b"):
            if key in mate and not isinstance(mate[key], str):
                _err(errors, "bad_port_ref", base + "." + key,
                     "%s must be a port name string" % key,
                     "Use e.g. 'axis_z' or omit it for the default port.")
        for key in ("offset_mm", "value_mm", "distance_mm", "angle_deg",
                    "weight"):
            if key in mate and not _is_num(mate[key]):
                _err(errors, "bad_mate_value", base + "." + key,
                     "%s must be a number" % key, "Use a number.")
        if "align" in mate and not isinstance(mate["align"], str):
            _err(errors, "bad_align", base + ".align", "align must be a string",
                 "Use auto | parallel | antiparallel.")
        if "allow_interference" in mate and not isinstance(
                mate["allow_interference"], bool):
            _err(errors, "bad_allow_interference", base + ".allow_interference",
                 "allow_interference must be a bool", "Use true or false.")
        norm_mates.append(mate)
    model["mates"] = norm_mates

    solver = model.get("solver")
    if solver is not None:
        if not isinstance(solver, Mapping):
            _err(errors, "bad_solver", "$.model.solver",
                 "solver must be an object", "Use {} or {max_nfev, tolerance_mm}.")
        else:
            for key in ("max_nfev", "tolerance_mm"):
                if key in solver and not _is_num(solver[key]):
                    _err(errors, "bad_solver", "$.model.solver." + key,
                         "%s must be a number" % key, "Use a number.")
    clearance = model.get("clearance")
    if clearance is not None and not isinstance(clearance, Mapping):
        _err(errors, "bad_clearance", "$.model.clearance",
             "clearance options must be an object",
             "Use {\"threshold_mm\": 5}.")
    allow = model.get("allow_interference")
    if allow is not None:
        if not isinstance(allow, (list, tuple)):
            _err(errors, "bad_allow", "$.model.allow_interference",
                 "allow_interference must be an array",
                 'Use [{"a":"base","b":"shaft"}] or omit it.')
        else:
            for i, row in enumerate(allow):
                if not isinstance(row, Mapping) or row.get("a") not in ids \
                        or row.get("b") not in ids:
                    _err(errors, "bad_allow", "$.model.allow_interference[%d]" % i,
                         "allow_interference row needs known a/b parts",
                         "Use {\"a\":\"part1\",\"b\":\"part2\"}.")


def validate_spec(spec: Any) -> dict:
    """Validate and normalise a spec.  Accumulates errors; never raises."""
    errors: list[dict] = []
    if not isinstance(spec, Mapping):
        _err(errors, "bad_spec", "$", "spec must be a JSON object",
             "Pass {schema, name, model, ...}.")
        return {"ok": False, "spec": None, "errors": errors}

    out = copy.deepcopy(dict(spec))

    schema = out.get("schema")
    if schema is None:
        out["schema"] = SCHEMA
    elif schema != SCHEMA:
        _err(errors, "bad_schema", "$.schema",
             "unsupported schema %r" % (schema,),
             "Set schema to %r." % SCHEMA)

    name = out.get("name")
    if name is None:
        out["name"] = "part"
    elif not isinstance(name, str) or not name.strip():
        _err(errors, "bad_name", "$.name", "name must be a non-empty string",
             "Use e.g. name: 'mug-01'.")

    quality = out.get("quality")
    if quality is None:
        out["quality"] = "draft"
    elif not isinstance(quality, str) or quality.lower() not in QUALITY:
        _err(errors, "bad_quality", "$.quality",
             "quality must be one of %s" % (", ".join(QUALITY),),
             "Use quality: 'draft'.")
    else:
        out["quality"] = quality.lower()

    material = out.get("material")
    if material is None:
        out["material"] = "ceramic"
    elif not isinstance(material, str) or material.lower() not in MATERIALS:
        _err(errors, "bad_material", "$.material",
             "unknown material %r" % (material,),
             "Use one of: %s." % ", ".join(sorted(MATERIALS)))
    else:
        out["material"] = material.lower()

    model = out.get("model")
    if not isinstance(model, Mapping):
        _err(errors, "bad_model", "$.model", "model must be an object",
             "Use {kind:'recipe', id:...} or {kind:'graph', features:[...]}.")
        model = {}
    else:
        model = copy.deepcopy(dict(model))
    out["model"] = model

    kind = model.get("kind")
    if kind is None:
        _err(errors, "bad_model_kind", "$.model.kind",
             "model.kind is required",
             "Use 'recipe', 'graph' or 'assembly'.")
    elif kind == KIND_RECIPE:
        model["kind"] = KIND_RECIPE
        _validate_recipe(model, errors)
    elif kind == KIND_GRAPH:
        model["kind"] = KIND_GRAPH
        _validate_graph(model, errors)
    elif kind == KIND_ASSEMBLY:
        model["kind"] = KIND_ASSEMBLY
        _validate_assembly(model, errors)
    else:
        _err(errors, "bad_model_kind", "$.model.kind",
             "unknown model.kind %r" % (kind,),
             "Use 'recipe', 'graph' or 'assembly'.")

    out["finish"] = _validate_finish(out.get("finish"), errors)
    out["constraints"] = _validate_constraints(out.get("constraints"), errors)
    out["assumptions"] = _validate_assumptions(out.get("assumptions"), errors)

    return {"ok": not errors, "spec": out, "errors": errors}


# --------------------------------------------------------------------- propose

def _assume(ledger: list[dict], path: str, value: Any, unit: str, why: str) -> None:
    ledger.append({
        "id": "A%d" % (len(ledger) + 1),
        "path": path,
        "value": value,
        "unit": unit,
        "basis": "design_default",
        "status": "assumed",
        "requires_confirmation": True,
        "why": why,
    })


def _statement(entry: dict) -> str:
    unit = (" " + entry["unit"]) if entry.get("unit") else ""
    return ("%s was not supplied; assumed %s%s (%s; design default, "
            "requires confirmation)."
            % (entry["path"], entry["value"], unit, entry["why"]))


def _unit_of(ps: Any) -> str:
    unit = _attr(ps, "unit", "none")
    return "" if unit in (None, "none", "") else str(unit)


def _propose_recipe(s: dict, model: dict, ledger: list[dict],
                    unresolved: list[dict], errors: list[dict]) -> None:
    rid = model.get("id")
    if not isinstance(rid, str) or not rid:
        return
    try:
        recipe = _get_recipe(rid)
    except AttaError as exc:
        errors.append(_err_from_atta(exc))
        return

    if model.get("preset") is None:
        model["preset"] = "nominal"
        _assume(ledger, "$.model.preset", "nominal", "",
                "recipe %r default preset" % rid)

    if s.get("quality") is None:
        s["quality"] = "draft"
        _assume(ledger, "$.quality", "draft", "", "default build quality")
    if s.get("material") is None:
        material = _attr(recipe, "material", "ceramic") or "ceramic"
        s["material"] = material
        _assume(ledger, "$.material", material, "",
                "recipe %r declared material" % rid)

    params = model.get("params")
    if params is None:
        params = {}
        model["params"] = params
    if not isinstance(params, dict):
        return  # validate reports bad_params
    specs = _attr(recipe, "params", {}) or {}
    if not isinstance(specs, Mapping):
        return
    for name, ps in specs.items():
        if name in params:
            continue
        default = _attr(ps, "default", None)
        if default is None:
            unresolved.append({
                "code": "missing_param",
                "path": "$.model.params.%s" % name,
                "message": "param %r has no default and was not supplied" % name,
                "fix": "Give an explicit %s value; no design default exists." % name,
            })
            continue
        params[name] = default
        _assume(ledger, "$.model.params.%s" % name, default,
                _unit_of(ps),
                "recipe ParamSpec.default for %r" % name)


def _feature_has(feat: Mapping, args: Mapping, key: str) -> bool:
    return bool(args.get(key)) or bool(feat.get(key))


#: Arguments that are irreducibly *design intent*: propose must report them in
#: ``unresolved`` rather than invent a default.  Keyed by feature op, valued by
#: ``(arg name, unresolved code, what is missing, fix)``.
#:
#: The arg names here must be the ones ``features.FEATURE_OPS`` actually
#: accepts.  ``_propose_graph`` cross-checks every entry against the live
#: registry and ignores the ones the registry does not declare, so a mismatch
#: can never again make a legal spec look unresolved (v1.1 read ``profile`` and
#: ``sections`` while the ops were ``profile_rz`` and ``rings``).
#:
#: There is deliberately no ``hole`` entry: Atta has no ``hole`` op (a hole is
#: ``cylinder`` + ``difference``), and ``cylinder.center`` has a real default,
#: so flagging it would fire on every legitimately centred cylinder.
_DESIGN_INTENT = {
    "revolve": ("profile_rz", "missing_profile", "the axis profile",
                "Give args.profile_rz=[[r,z],...]."),
    "loft": ("rings", "missing_sections", "at least two rings",
             "Give args.rings as a list of >= 2 rings of [x,y,z] points."),
    "sweep": ("path", "missing_path", "the sweep path",
              "Give args.path as a point list."),
}


def _intent_missing(op: str, feat: Mapping, args: Mapping, key: str) -> bool:
    """True when the design-intent arg ``key`` is absent or unusable."""
    if op == "revolve" and key == "profile_rz":
        segs = args.get("segments", feat.get("segments"))
        if isinstance(segs, (list, tuple)) and len(segs) >= 1:
            return False
    value = args.get(key, feat.get(key))
    if op == "loft":  # loft needs at least two rings, not merely one
        return not (isinstance(value, list) and len(value) >= 2)
    return not bool(value)


def _propose_graph(s: dict, model: dict, ledger: list[dict],
                   unresolved: list[dict]) -> None:
    if model.get("units") is None:
        model["units"] = "mm"
        _assume(ledger, "$.model.units", "mm", "",
                "public spec units are millimetres")
    if model.get("segments") is None:
        model["segments"] = 64
        _assume(ledger, "$.model.segments", 64, "",
                "general default tessellation for graph primitives")
    if s.get("quality") is None:
        s["quality"] = "draft"
        _assume(ledger, "$.quality", "draft", "", "default build quality")
    if s.get("material") is None:
        s["material"] = "ceramic"
        _assume(ledger, "$.material", "ceramic", "", "default material")

    feats = model.get("features")
    if not isinstance(feats, list):
        return

    try:
        ops = _load_feature_ops()
    except AttaError:
        ops = {}  # validate_spec reports features_unavailable itself

    last_id = None
    for i, feat in enumerate(feats):
        if not isinstance(feat, dict):
            continue
        base = "$.model.features[%d]" % i
        if feat.get("id") is None:
            fid = "f%d" % i
            feat["id"] = fid
            _assume(ledger, base + ".id", fid, "",
                    "graph feature id derived from its position")
        if isinstance(feat.get("id"), str):
            last_id = feat["id"]

        op = feat.get("op")
        args = feat.get("args") if isinstance(feat.get("args"), Mapping) else {}
        intent = _DESIGN_INTENT.get(op)
        if intent is None:
            continue
        key, code, what, fix = intent
        declared = ops.get(op)
        if isinstance(declared, Mapping):
            allowed = set(declared.get("required") or ()) | \
                set(declared.get("optional") or ())
            if key not in allowed:  # registry disagrees: trust the registry
                continue
        if _intent_missing(op, feat, args, key):
            unresolved.append({
                "code": code,
                "path": base + ".args." + key,
                "message": "%s has no %s; it is design intent" % (op, what),
                "fix": fix,
            })

    if model.get("output") is None and last_id is not None:
        model["output"] = last_id
        _assume(ledger, "$.model.output", last_id, "",
                "output defaults to the last feature in the graph")


def _propose_assembly(s: dict, model: dict, ledger: list[dict],
                      unresolved: list[dict], errors: list[dict]) -> None:
    if s.get("quality") is None:
        s["quality"] = "draft"
        _assume(ledger, "$.quality", "draft", "", "default build quality")
    if s.get("material") is None:
        s["material"] = "steel"
        _assume(ledger, "$.material", "steel", "",
                "default material for a mechanical assembly")

    parts = model.get("parts")
    if not isinstance(parts, list):
        return
    any_auto_ports = False
    has_attachments = any(isinstance(p, Mapping) and p.get("attach") is not None for p in parts)
    attach_order = _validate_attachments(parts, "$.model.parts", []) if has_attachments else []
    root_id = attach_order[0] if attach_order else None
    for i, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        base = "$.model.parts[%d]" % i
        if part.get("transform") is None and part.get("attach") is None:
            part["transform"] = {"translate_mm": [0, 0, 0],
                                 "rotate_deg": [0, 0, 0]}
            _assume(ledger, base + ".transform",
                    {"translate_mm": [0, 0, 0], "rotate_deg": [0, 0, 0]}, "mm/deg",
                    "identity initial pose; mates may move a free part")
        if part.get("fixed") is None:
            fixed = (part.get("id") == root_id) if has_attachments else (i == 0)
            part["fixed"] = fixed
            _assume(ledger, base + ".fixed", fixed, "",
                    ("anchor the stable attachment root so the assembly has a world frame"
                     if has_attachments else "anchor the first part so the assembly has a world frame")
                    if fixed else "leave this part free for the mate solver")
        ports = part.get("ports")
        if not ports:
            any_auto_ports = True
            _assume(ledger, base + ".ports", "auto: bbox center/axes/faces", "",
                    "no ports declared; the bbox frame is a design default and "
                    "must be confirmed for a real mate")
    if model.get("mates") is None:
        model["mates"] = []
        _assume(ledger, "$.model.mates", [], "",
                "no mates supplied; the assembly keeps its initial transforms")
    if any_auto_ports:
        unresolved.append({
            "code": "auto_ports",
            "path": "$.model.parts",
            "message": "one or more parts use auto-derived bbox ports",
            "fix": "Declare ports (origin_mm, axis, normal, radius_mm) for every "
                   "mating feature; bbox ports are only a starting frame.",
        })
    if model.get("solver") is None:
        model["solver"] = {"max_nfev": 200, "tolerance_mm": 0.01}
        _assume(ledger, "$.model.solver",
                {"max_nfev": 200, "tolerance_mm": 0.01}, "",
                "numerical solver defaults; not a measurement")


def _dedupe_errors(*groups: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for group in groups:
        for err in group:
            key = (err.get("code"), err.get("path"), err.get("message"))
            if key in seen:
                continue
            seen.add(key)
            out.append(err)
    return out


def propose_spec(spec: Any) -> dict:
    """Fill explainable design defaults; never invent measurements.

    Recipe parameters come from ``ParamSpec.default``.  Graph specs get
    ``units``/``segments``/``quality``/``material`` and a default ``output``.
    Critical positioning (hole ``at``, ``revolve`` profile, ``loft`` sections,
    ``sweep`` path) is left in ``unresolved`` for the caller to supply.
    """
    if not isinstance(spec, Mapping):
        errors = [{"code": "bad_spec", "path": "$",
                   "message": "spec must be a JSON object",
                   "fix": "Pass {schema, name, model, ...}."}]
        return {"ok": False, "spec": None, "assumption_ledger": [],
                "assumption_summary": {"n": 0, "assumed": 0, "unresolved": 0,
                                       "total": 0, "requires_confirmation": 0},
                "unresolved": [], "errors": errors}

    s = copy.deepcopy(dict(spec))
    ledger: list[dict] = []
    unresolved: list[dict] = []
    errors: list[dict] = []

    if s.get("schema") is None:
        s["schema"] = SCHEMA
    if s.get("name") is None:
        s["name"] = "part"

    model = s.get("model")
    if isinstance(model, dict):
        kind = model.get("kind")
        if kind == KIND_RECIPE:
            _propose_recipe(s, model, ledger, unresolved, errors)
        elif kind == KIND_GRAPH:
            _propose_graph(s, model, ledger, unresolved)
        elif kind == KIND_ASSEMBLY:
            _propose_assembly(s, model, ledger, unresolved, errors)

    # Make prototype assumptions visible in the human-readable field.
    existing = s.get("assumptions")
    if not isinstance(existing, list):
        existing = []
    statements = [x for x in existing if isinstance(x, str)]
    for entry in ledger:
        text = _statement(entry)
        if text not in statements:
            statements.append(text)
    if statements:
        s["assumptions"] = statements

    v = validate_spec(s)
    errors = _dedupe_errors(errors, v["errors"])
    proposed = v["spec"] if v["spec"] is not None else s
    summary = {
        "n": len(ledger),
        "assumed": sum(1 for e in ledger if e["status"] == "assumed"),
        "unresolved": len(unresolved),
        "total": len(ledger),
        "requires_confirmation": sum(1 for e in ledger
                                     if e.get("requires_confirmation")),
    }
    return {
        "ok": not errors and not unresolved,
        "spec": proposed,
        "assumption_ledger": ledger,
        "assumption_summary": summary,
        "unresolved": unresolved,
        "errors": errors,
    }


# --------------------------------------------------------------------- compile

def _json_copy(value: Any) -> Any:
    """Deep copy through canonical JSON so compiled output is deterministic."""
    return json.loads(canonical_json(value))


def compile_spec(spec: Any) -> dict:
    """Validate then compile to the geometry-facing contract.

    Recipe parameters are resolved through ``Recipe.resolve``: ``params_display``
    stays in mm, ``params_m`` is in metres.  A graph is **not** evaluated here;
    its features, output, units and segments are carried through untouched.
    """
    v = validate_spec(spec)
    if not v["ok"]:
        return {"ok": False, "compiled": None, "errors": v["errors"]}

    s = v["spec"]
    model = s["model"]
    base = {
        "name": s["name"],
        "kind": model["kind"],
        "quality": s["quality"],
        "material": s["material"],
        "finish": s["finish"],
        "constraints": s["constraints"],
        "assumptions": s["assumptions"],
        "spec_hash": spec_hash(s),
    }

    errors: list[dict] = []
    if model["kind"] == KIND_ASSEMBLY:
        compiled = dict(base)
        parts_out = []
        for i, part in enumerate(model.get("parts") or []):
            row = copy.deepcopy(dict(part))
            if part.get("spec") is not None:
                nested = compile_spec(part["spec"])
                if not nested["ok"]:
                    errors.extend(_prefix_errors(
                        nested["errors"], "$.model.parts[%d].spec" % i))
                    continue
                # Keep the normalized inline spec as the public part source;
                # compilation checks its constraints without replacing it by
                # a different contract or loading model_id-backed parts.
            if part.get("recipe"):
                try:
                    recipe = _get_recipe(str(part["recipe"]))
                    resolved = recipe.resolve(
                        patch=dict(part.get("params") or {}),
                        preset=str(part.get("preset") or "nominal"))
                except Exception as exc:  # noqa: BLE001 - compile returns values
                    _err(errors, "assembly_part_resolve", "$.model.parts",
                         "part %r could not resolve: %s" % (part.get("id"), exc),
                         "Fix the part parameters; see atta_catalog.")
                    continue
                if bool(_attr(resolved, "blocked", False)):
                    _err(errors, "assembly_part_blocked", "$.model.parts",
                         "part %r recipe constraints block this parameter set"
                         % (part.get("id"),),
                         "Fix the part issues: %s"
                         % json.dumps(_attr(resolved, "issues", []) or [],
                                      ensure_ascii=False))
                    continue
                row["params_display"] = dict(_attr(resolved, "display", {}) or {})
                row["params_m"] = dict(_attr(resolved, "values", {}) or {})
                row["preset"] = str(part.get("preset") or "nominal")
            parts_out.append(row)
        compiled.update({
            "parts": parts_out,
            "mates": model.get("mates") or [],
            "solver": model.get("solver") or {},
            "clearance": model.get("clearance") or {},
            "allow_interference": model.get("allow_interference") or [],
            "params_display": {},
            "params_m": {},
            "preset": None,
        })
    elif model["kind"] == KIND_GRAPH:
        compiled = dict(base)
        compiled.update({
            "features": model["features"],
            "output": model["output"],
            "units": model["units"],
            "segments": model["segments"],
            # Graph params feed $param/$expr in feature args; they are already
            # in public units, so they are passed through unchanged.
            "params_display": dict(model.get("params") or {}),
            "params_m": dict(model.get("params") or {}),
            "preset": None,
        })
    else:
        try:
            recipe = _get_recipe(model["id"])
        except AttaError as exc:
            return {"ok": False, "compiled": None,
                    "errors": [_err_from_atta(exc)]}
        try:
            resolved = recipe.resolve(patch=dict(model["params"]),
                                      preset=model["preset"])
        except Exception as exc:  # noqa: BLE001 - recipe errors are values
            _err(errors, "resolve_failed", "$.model.params",
                 "Recipe.resolve failed: %s" % exc,
                 "Check parameter values and preset against atta_catalog.")
            return {"ok": False, "compiled": None, "errors": errors}

        if bool(_attr(resolved, "blocked", False)):
            issues = _attr(resolved, "issues", None) or []
            _err(errors, "recipe_blocked", "$.model.params",
                 "recipe constraints block this parameter set",
                 "Resolve the recipe issues: %s"
                 % (json.dumps(issues, ensure_ascii=False) if issues else "see issues"))
            return {"ok": False, "compiled": None, "errors": errors}

        display = dict(_attr(resolved, "display", None) or {})
        values = dict(_attr(resolved, "values", None) or {})
        compiled = dict(base)
        compiled.update({
            "recipe": model["id"],
            "preset": model["preset"],
            "params_display": display,
            "params_m": values,
        })

    if errors:
        # A part that failed to resolve was skipped above; returning it as a
        # successful compile would silently drop geometry from the assembly.
        return {"ok": False, "compiled": None, "errors": errors}
    try:
        compiled = _json_copy(compiled)
    except Exception as exc:  # noqa: BLE001 - non-JSON recipe values are a value
        _err(errors, "compile_failed", "$",
             "compiled spec is not JSON-serialisable: %s" % exc,
             "Use plain JSON numbers/strings in params, finish and constraints.")
        return {"ok": False, "compiled": None, "errors": errors}
    return {"ok": True, "compiled": compiled, "errors": errors}


# ---------------------------------------------------------------------- patch

_TOP_SPEC_PATCH_KEYS = frozenset({
    "name", "quality", "material", "finish", "constraints", "assumptions",
})
_GRAPH_PATCH_KEYS = frozenset({
    "features", "add_features", "remove_features",
    "output", "params", "segments", "units",
})
_ASSEMBLY_PATCH_KEYS = frozenset({
    "parts", "add_parts", "remove_parts",
    "mates", "solver", "clearance", "allow_interference",
})
_FEATURE_PATCH_KEYS = frozenset({"op", "args", "inputs", "tags"})


def _safe_spec_hash(spec_obj: Any) -> str | None:
    try:
        return spec_hash(spec_obj)
    except Exception:  # noqa: BLE001
        return None


def _apply_top_spec_fields(s: dict, patches: Mapping,
                           applied: list[dict], delta: dict) -> None:
    fields_delta = delta.setdefault("fields", {})
    for key in sorted(_TOP_SPEC_PATCH_KEYS):
        if key not in patches:
            continue
        val = patches[key]
        before = copy.deepcopy(s.get(key))
        if key in ("finish", "constraints") and isinstance(val, Mapping):
            merged = dict(s.get(key) or {}) if isinstance(s.get(key), Mapping) else {}
            merged.update(copy.deepcopy(dict(val)))
            s[key] = merged
        else:
            s[key] = copy.deepcopy(val)
        fields_delta[key] = {"before": before, "after": copy.deepcopy(s[key])}
        applied.append({"op": "patch_field", "path": "$." + key})


def _patch_graph_model(model: dict, patches: Mapping, base_path: str,
                       errors: list[dict], applied: list[dict],
                       delta: dict) -> None:
    features = copy.deepcopy(list(model.get("features") or []))
    if "params" in patches:
        p_patch = patches["params"]
        if not isinstance(p_patch, Mapping):
            _err(errors, "bad_patch", base_path + ".params",
                 "params patch must be an object",
                 "Pass {param_name: finite_number}.")
        else:
            cur_params = dict(model.get("params") or {})
            p_delta = delta.setdefault("params", {})
            for k, v in p_patch.items():
                old = cur_params.get(k)
                cur_params[k] = copy.deepcopy(v)
                p_delta[k] = {"before": old, "after": copy.deepcopy(v)}
                applied.append({"op": "patch_param", "name": k,
                                "before": old, "after": copy.deepcopy(v)})
            model["params"] = cur_params

    if "remove_features" in patches:
        rem = patches["remove_features"]
        if not isinstance(rem, (list, tuple)):
            _err(errors, "bad_patch", base_path + ".remove_features",
                 "remove_features must be an array of feature ids",
                 "Pass [\"feature_id\", ...].")
        else:
            rem_delta = delta.setdefault("removed_features", [])
            for i, fid in enumerate(rem):
                rp = "%s.remove_features[%d]" % (base_path, i)
                if not isinstance(fid, str) or not any(
                        isinstance(f, Mapping) and f.get("id") == fid
                        for f in features):
                    known = sorted(f["id"] for f in features
                                   if isinstance(f, Mapping)
                                   and isinstance(f.get("id"), str))
                    _err(errors, "unknown_feature_id", rp,
                         "cannot remove unknown feature id %r" % (fid,),
                         "Known feature ids: %s." % (", ".join(known) or "(none)"))
                    continue
                features = [f for f in features
                            if not (isinstance(f, Mapping) and f.get("id") == fid)]
                rem_delta.append(fid)
                applied.append({"op": "remove_feature", "id": fid})

    if "features" in patches:
        f_patches = patches["features"]
        if not isinstance(f_patches, Mapping):
            _err(errors, "bad_patch", base_path + ".features",
                 "features patch must be an object {feature_id: {args, inputs, op, tags}}",
                 "Key patches by target feature id.")
        else:
            by_id = {f["id"]: idx for idx, f in enumerate(features)
                     if isinstance(f, Mapping) and isinstance(f.get("id"), str)}
            mod_delta = delta.setdefault("modified_features", {})
            for fid, fpatch in f_patches.items():
                fp = "%s.features.%s" % (base_path, fid)
                if fid not in by_id:
                    _err(errors, "unknown_feature_id", fp,
                         "unknown target feature id %r" % (fid,),
                         "Known feature ids: %s."
                         % (", ".join(sorted(by_id)) or "(none)"))
                    continue
                if not isinstance(fpatch, Mapping):
                    _err(errors, "bad_patch", fp,
                         "feature patch must be an object",
                         "Use {args, inputs, op, tags}.")
                    continue
                unknown = set(fpatch) - _FEATURE_PATCH_KEYS
                if unknown:
                    _err(errors, "bad_patch", fp,
                         "unknown feature patch field(s): %s"
                         % ", ".join(sorted(unknown)),
                         "Allowed fields: %s."
                         % ", ".join(sorted(_FEATURE_PATCH_KEYS)))
                    continue
                feat = dict(features[by_id[fid]])
                before_feat = copy.deepcopy(feat)
                if "op" in fpatch:
                    feat["op"] = copy.deepcopy(fpatch["op"])
                if "args" in fpatch:
                    ap = fpatch["args"]
                    if not isinstance(ap, Mapping):
                        _err(errors, "bad_patch", fp + ".args",
                             "feature args patch must be an object",
                             "Pass {arg_name: value}.")
                    else:
                        cur_args = dict(feat.get("args") or {})
                        for ak, av in ap.items():
                            cur_args[ak] = copy.deepcopy(av)
                        feat["args"] = cur_args
                if "inputs" in fpatch:
                    ip = fpatch["inputs"]
                    if not isinstance(ip, (list, tuple)):
                        _err(errors, "bad_patch", fp + ".inputs",
                             "feature inputs patch must be an array",
                             "Pass [\"input_id\", ...].")
                    else:
                        feat["inputs"] = list(ip)
                if "tags" in fpatch:
                    tp = fpatch["tags"]
                    if tp is None:
                        feat.pop("tags", None)
                    elif not isinstance(tp, Mapping):
                        _err(errors, "bad_patch", fp + ".tags",
                             "feature tags patch must be an object or null",
                             "Pass {slot: tag_name}.")
                    else:
                        cur_tags = (dict(feat["tags"])
                                    if isinstance(feat.get("tags"), Mapping) else {})
                        for tk, tv in tp.items():
                            cur_tags[tk] = copy.deepcopy(tv)
                        feat["tags"] = cur_tags
                features[by_id[fid]] = feat
                mod_delta[fid] = {"before": before_feat,
                                  "after": copy.deepcopy(feat)}
                applied.append({"op": "patch_feature", "id": fid,
                                "fields": sorted(fpatch)})

    if "add_features" in patches:
        adds = patches["add_features"]
        if not isinstance(adds, (list, tuple)):
            _err(errors, "bad_patch", base_path + ".add_features",
                 "add_features must be an array of feature objects",
                 "Pass [{id, op, args, inputs}, ...].")
        else:
            existing = {f["id"] for f in features
                        if isinstance(f, Mapping) and isinstance(f.get("id"), str)}
            add_delta = delta.setdefault("added_features", [])
            for i, raw in enumerate(adds):
                ap = "%s.add_features[%d]" % (base_path, i)
                if not isinstance(raw, Mapping):
                    _err(errors, "bad_patch", ap,
                         "added feature must be an object",
                         "Pass {id, op, args, inputs}.")
                    continue
                nf = copy.deepcopy(dict(raw))
                nfid = nf.get("id")
                if isinstance(nfid, str) and nfid in existing:
                    _err(errors, "duplicate_id", ap + ".id",
                         "cannot add feature with existing id %r" % (nfid,),
                         "Use a unique id or patch it under patches.features.")
                    continue
                if isinstance(nfid, str):
                    existing.add(nfid)
                features.append(nf)
                add_delta.append(nfid)
                applied.append({"op": "add_feature", "id": nfid})

    for key in ("output", "segments", "units"):
        if key in patches:
            before = copy.deepcopy(model.get(key))
            model[key] = copy.deepcopy(patches[key])
            delta.setdefault("fields", {})["model." + key] = {
                "before": before, "after": copy.deepcopy(model[key]),
            }
            applied.append({"op": "patch_field", "path": "$.model." + key})

    model["features"] = features


def _patch_assembly_model(model: dict, patches: Mapping, base_path: str,
                          errors: list[dict], applied: list[dict],
                          delta: dict) -> None:
    parts = copy.deepcopy(list(model.get("parts") or []))

    if "remove_parts" in patches:
        rem = patches["remove_parts"]
        if not isinstance(rem, (list, tuple)):
            _err(errors, "bad_patch", base_path + ".remove_parts",
                 "remove_parts must be an array of part ids",
                 "Pass [\"part_id\", ...].")
        else:
            rem_delta = delta.setdefault("removed_parts", [])
            for i, pid in enumerate(rem):
                rp = "%s.remove_parts[%d]" % (base_path, i)
                if not isinstance(pid, str) or not any(
                        isinstance(p, Mapping) and p.get("id") == pid
                        for p in parts):
                    known = sorted(p["id"] for p in parts
                                   if isinstance(p, Mapping)
                                   and isinstance(p.get("id"), str))
                    _err(errors, "unknown_part_id", rp,
                         "cannot remove unknown part id %r" % (pid,),
                         "Known part ids: %s." % (", ".join(known) or "(none)"))
                    continue
                parts = [p for p in parts
                         if not (isinstance(p, Mapping) and p.get("id") == pid)]
                rem_delta.append(pid)
                applied.append({"op": "remove_part", "id": pid})

    if "parts" in patches:
        p_patches = patches["parts"]
        if not isinstance(p_patches, Mapping):
            _err(errors, "bad_patch", base_path + ".parts",
                 "parts patch must be an object {part_id: {...}}",
                 "Key patches by target part id.")
        else:
            by_id = {p["id"]: idx for idx, p in enumerate(parts)
                     if isinstance(p, Mapping) and isinstance(p.get("id"), str)}
            mod_parts = delta.setdefault("modified_parts", {})
            for pid, ppatch in p_patches.items():
                pp = "%s.parts.%s" % (base_path, pid)
                if pid not in by_id:
                    _err(errors, "unknown_part_id", pp,
                         "unknown target part id %r" % (pid,),
                         "Known part ids: %s."
                         % (", ".join(sorted(by_id)) or "(none)"))
                    continue
                if not isinstance(ppatch, Mapping):
                    _err(errors, "bad_patch", pp,
                         "part patch must be an object",
                         "Pass fields to update on the part.")
                    continue
                part = dict(parts[by_id[pid]])
                before_part = copy.deepcopy(part)
                for key, val in ppatch.items():
                    if key == "params" and isinstance(val, Mapping):
                        cur = dict(part.get("params") or {})
                        cur.update(copy.deepcopy(dict(val)))
                        part["params"] = cur
                    elif (key == "graph" and isinstance(val, Mapping)
                          and isinstance(part.get("graph"), Mapping)
                          and not isinstance(val.get("features"), (list, tuple))):
                        sub_graph = copy.deepcopy(dict(part["graph"]))
                        sub_delta: dict = {}
                        _patch_graph_model(sub_graph, val, pp + ".graph",
                                           errors, applied, sub_delta)
                        part["graph"] = sub_graph
                    elif (key == "spec" and isinstance(val, Mapping)
                          and isinstance(part.get("spec"), Mapping)
                          and "model" not in val and "schema" not in val):
                        sub_res = patch_spec(part["spec"], val)
                        if not sub_res["ok"]:
                            errors.extend(_prefix_errors(
                                sub_res["errors"], pp + ".spec"))
                        part["spec"] = sub_res["spec"] or part["spec"]
                    elif val is None and key in ("attach", "transform", "ports"):
                        part.pop(key, None)
                    else:
                        part[key] = copy.deepcopy(val)
                parts[by_id[pid]] = part
                mod_parts[pid] = {"before": before_part,
                                  "after": copy.deepcopy(part)}
                applied.append({"op": "patch_part", "id": pid,
                                "fields": sorted(ppatch)})

    if "add_parts" in patches:
        adds = patches["add_parts"]
        if not isinstance(adds, (list, tuple)):
            _err(errors, "bad_patch", base_path + ".add_parts",
                 "add_parts must be an array of part objects",
                 "Pass [{id, recipe|graph|spec|model_id, ...}, ...].")
        else:
            existing = {p["id"] for p in parts
                        if isinstance(p, Mapping) and isinstance(p.get("id"), str)}
            add_delta = delta.setdefault("added_parts", [])
            for i, raw in enumerate(adds):
                ap = "%s.add_parts[%d]" % (base_path, i)
                if not isinstance(raw, Mapping):
                    _err(errors, "bad_patch", ap,
                         "added part must be an object",
                         "Pass {id, recipe|graph|spec|model_id, ...}.")
                    continue
                np_obj = copy.deepcopy(dict(raw))
                npid = np_obj.get("id")
                if isinstance(npid, str) and npid in existing:
                    _err(errors, "duplicate_part_id", ap + ".id",
                         "cannot add part with existing id %r" % (npid,),
                         "Use a unique id or patch it under patches.parts.")
                    continue
                if isinstance(npid, str):
                    existing.add(npid)
                parts.append(np_obj)
                add_delta.append(npid)
                applied.append({"op": "add_part", "id": npid})

    for key in ("mates", "solver", "clearance", "allow_interference"):
        if key in patches:
            before = copy.deepcopy(model.get(key))
            model[key] = copy.deepcopy(patches[key])
            delta.setdefault("fields", {})["model." + key] = {
                "before": before, "after": copy.deepcopy(model[key]),
            }
            applied.append({"op": "patch_field", "path": "$.model." + key})

    model["parts"] = parts


def patch_spec(spec: Any, patches: Any = None) -> dict:
    """Apply locate-then-infill patches to a recipe, graph, or assembly spec."""
    if not isinstance(spec, Mapping):
        errors = [{"code": "bad_spec", "path": "$",
                   "message": "spec must be a JSON object",
                   "fix": "Pass {schema, name, model, ...}."}]
        return {"ok": False, "spec": None, "applied_patches": [],
                "delta": {}, "errors": errors}
    if patches is None:
        patches = {}
    if not isinstance(patches, Mapping):
        errors = [{"code": "bad_patches", "path": "$.patches",
                   "message": "patches must be a JSON object",
                   "fix": "Pass a dict of patches."}]
        return {"ok": False, "spec": copy.deepcopy(dict(spec)),
                "applied_patches": [], "delta": {}, "errors": errors}

    s = copy.deepcopy(dict(spec))
    before_hash = _safe_spec_hash(s)
    model = s.get("model")
    if not isinstance(model, dict):
        v = validate_spec(s)
        return {"ok": False, "spec": v["spec"], "applied_patches": [],
                "delta": {"kind": None, "spec_hash_before": before_hash,
                          "spec_hash_after": before_hash, "changed": False},
                "errors": v["errors"]}

    kind = model.get("kind")
    errors: list[dict] = []
    applied: list[dict] = []
    delta: dict[str, Any] = {
        "kind": kind,
        "spec_hash_before": before_hash,
        "spec_hash_after": before_hash,
        "changed": False,
    }

    if kind == KIND_RECIPE:
        recipe_param_names: set[str] = set()
        rid = model.get("id")
        if isinstance(rid, str) and rid.strip():
            try:
                recipe = _get_recipe(rid)
                recipe_param_names = set(_attr(recipe, "params", {}) or {})
            except AttaError:
                recipe_param_names = set()

        cur_params = dict(model.get("params") or {}) if isinstance(
            model.get("params"), Mapping) else {}
        p_delta = delta.setdefault("params", {})

        def _set_recipe_param(k: str, v: Any) -> None:
            old = cur_params.get(k)
            cur_params[k] = copy.deepcopy(v)
            p_delta[k] = {"before": old, "after": copy.deepcopy(v)}
            applied.append({"op": "patch_param", "name": k,
                            "before": old, "after": copy.deepcopy(v)})

        top_patches: dict[str, Any] = {}
        for key, val in patches.items():
            if key in recipe_param_names:
                _set_recipe_param(key, val)
            elif key == "params":
                if not isinstance(val, Mapping):
                    _err(errors, "bad_patch", "$.patches.params",
                         "params patch must be an object",
                         "Pass {param_name: value}.")
                else:
                    for pk, pv in val.items():
                        _set_recipe_param(pk, pv)
            elif key == "preset":
                before = model.get("preset")
                model["preset"] = copy.deepcopy(val)
                delta.setdefault("fields", {})["model.preset"] = {
                    "before": before, "after": copy.deepcopy(val),
                }
                applied.append({"op": "patch_field", "path": "$.model.preset"})
            elif key in _TOP_SPEC_PATCH_KEYS:
                top_patches[key] = val
            else:
                _set_recipe_param(key, val)
        model["params"] = cur_params
        if top_patches:
            _apply_top_spec_fields(s, top_patches, applied, delta)

    elif kind == KIND_GRAPH:
        unknown = set(patches) - _GRAPH_PATCH_KEYS - _TOP_SPEC_PATCH_KEYS
        if unknown:
            _err(errors, "bad_patch", "$.patches",
                 "unknown graph patch key(s): %s" % ", ".join(sorted(unknown)),
                 "Allowed keys: %s."
                 % ", ".join(sorted(_GRAPH_PATCH_KEYS | _TOP_SPEC_PATCH_KEYS)))
        _apply_top_spec_fields(s, patches, applied, delta)
        _patch_graph_model(model, patches, "$.patches", errors, applied, delta)

    elif kind == KIND_ASSEMBLY:
        unknown = set(patches) - _ASSEMBLY_PATCH_KEYS - _TOP_SPEC_PATCH_KEYS
        if unknown:
            _err(errors, "bad_patch", "$.patches",
                 "unknown assembly patch key(s): %s" % ", ".join(sorted(unknown)),
                 "Allowed keys: %s."
                 % ", ".join(sorted(_ASSEMBLY_PATCH_KEYS | _TOP_SPEC_PATCH_KEYS)))
        _apply_top_spec_fields(s, patches, applied, delta)
        _patch_assembly_model(model, patches, "$.patches", errors, applied, delta)

    else:
        v = validate_spec(s)
        return {"ok": False, "spec": v["spec"], "applied_patches": [],
                "delta": delta, "errors": v["errors"]}

    v = validate_spec(s)
    combined_errors = _dedupe_errors(errors, v["errors"])
    out_spec = v["spec"] if v["spec"] is not None else s
    after_hash = _safe_spec_hash(out_spec)
    delta["spec_hash_after"] = after_hash
    delta["changed"] = (before_hash != after_hash)
    return {
        "ok": not combined_errors,
        "spec": out_spec,
        "applied_patches": applied,
        "delta": delta,
        "errors": combined_errors,
    }


# ------------------------------------------------------------------ preflight

def _warn(warnings: list[dict], code: str, path: str,
          message: str, fix: str) -> None:
    warnings.append({"code": str(code), "path": str(path),
                     "message": str(message), "fix": str(fix)})


def _collect_param_refs(value: Any, out: set[str]) -> None:
    if isinstance(value, Mapping):
        if "$param" in value and isinstance(value["$param"], str):
            out.add(value["$param"])
        for v in value.values():
            _collect_param_refs(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _collect_param_refs(v, out)


def _collect_feature_tags(feat: Mapping) -> set[str]:
    tags: set[str] = set()
    args = feat.get("args")
    args_map = args if isinstance(args, Mapping) else {}
    for container in (feat.get("tags"), args_map.get("tags")):
        if isinstance(container, Mapping):
            for v in container.values():
                if isinstance(v, str) and v.strip():
                    tags.add(v.strip())
    segs = args_map.get("segments")
    if isinstance(segs, (list, tuple)):
        for seg in segs:
            if isinstance(seg, Mapping) and isinstance(seg.get("tag"), str):
                if seg["tag"].strip():
                    tags.add(seg["tag"].strip())
    return tags


def _lint_graph_model(model: Mapping, prefix: str, ops: dict | None,
                      errors: list[dict], warnings: list[dict]) -> dict:
    features = list(model.get("features") or [])
    by_id = {f["id"]: f for f in features
             if isinstance(f, Mapping) and isinstance(f.get("id"), str)}
    output = model.get("output")
    declared_params = set((model.get("params") or {}).keys())

    # 1. Dependencies & reachability from output
    reachable: set[str] = set()
    if isinstance(output, str) and output in by_id:
        stack = [output]
        while stack:
            fid = stack.pop()
            if fid in reachable or fid not in by_id:
                continue
            reachable.add(fid)
            f = by_id[fid]
            for inp in f.get("inputs") or ():
                if isinstance(inp, str):
                    stack.append(inp)
            f_args = f.get("args")
            edges_sel = f_args.get("edges") if isinstance(f_args, Mapping) else None
            if isinstance(edges_sel, Mapping) and isinstance(edges_sel.get("from_feature"), str):
                stack.append(edges_sel["from_feature"])

        for i, feat in enumerate(features):
            if isinstance(feat, Mapping) and isinstance(feat.get("id"), str):
                if feat["id"] not in reachable:
                    _warn(warnings, "unreachable_feature",
                          "%s.features[%d]" % (prefix, i),
                          "feature %r is not reachable from output %r"
                          % (feat["id"], output),
                          "Wire it into %r or remove the unused feature." % output)

    # Ancestor cache for from_feature & selector tag checks
    ancestors_cache: dict[str, set[str]] = {}

    def _ancestors(fid: str, seen: set[str] | None = None) -> set[str]:
        if fid in ancestors_cache:
            return ancestors_cache[fid]
        seen = set(seen or ())
        if fid in seen or fid not in by_id:
            return set()
        seen.add(fid)
        res: set[str] = set()
        for inp in by_id[fid].get("inputs") or ():
            if isinstance(inp, str) and inp in by_id:
                res.add(inp)
                res |= _ancestors(inp, seen)
        ancestors_cache[fid] = res
        return res

    # 2. Arity, $param refs, unit scales, selectors
    dep_ok = True
    unit_ok = model.get("units", "mm") == "mm"
    sel_ok = True
    has_cad_op = any(isinstance(f, Mapping) and f.get("op") in ("fillet", "chamfer")
                     for f in features)
    for i, feat in enumerate(features):
        if not isinstance(feat, Mapping):
            continue
        fp = "%s.features[%d]" % (prefix, i)
        fid = feat.get("id")
        op = feat.get("op")
        inputs = feat.get("inputs") or []
        args = feat.get("args") if isinstance(feat.get("args"), Mapping) else {}

        if ops is not None and isinstance(op, str) and op in ops:
            meta = ops[op] or {}
            arity = meta.get("arity", 0)
            min_inputs = meta.get("min_inputs")
            if arity == 0 and len(inputs) > 0:
                _warn(warnings, "unused_inputs", fp + ".inputs",
                      "%s takes 0 inputs but got %d" % (op, len(inputs)),
                      "Remove inputs from %r." % fid)
            elif arity == 1 and min_inputs is None and len(inputs) != 1:
                dep_ok = False
                _err(errors, "bad_arity", fp + ".inputs",
                     "%s requires exactly 1 input, got %d" % (op, len(inputs)),
                     "Supply inputs: ['source_id'].")
            elif arity >= 2 and len(inputs) < (min_inputs or 2):
                _warn(warnings, "few_inputs", fp + ".inputs",
                      "%s normally takes >= %d inputs, got %d"
                      % (op, int(min_inputs or 2), len(inputs)),
                      "Add all operands to inputs.")

        refs: set[str] = set()
        _collect_param_refs(args, refs)
        for ref in sorted(refs - declared_params):
            _warn(warnings, "missing_graph_param", fp + ".args",
                  "arg references $param %r which is not declared in model.params" % ref,
                  "Add %r to model.params or supply params at build time." % ref)

        for dim_key in ("size", "radius", "height", "r_bottom", "r_top"):
            val = args.get(dim_key)
            nums = val if isinstance(val, (list, tuple)) else [val]
            for num in nums:
                if _is_finite_num(num):
                    fv = abs(float(num))
                    if 0.0 < fv < 0.01:
                        _warn(warnings, "suspicious_unit_scale",
                              "%s.args.%s" % (fp, dim_key),
                              "%s=%g mm is < 0.01 mm; check mm vs m units"
                              % (dim_key, float(num)),
                              "Graph dimensions are in millimetres, not metres.")
                        break
                    if fv > 100000.0:
                        _warn(warnings, "suspicious_unit_scale",
                              "%s.args.%s" % (fp, dim_key),
                              "%s=%g mm exceeds 100 m" % (dim_key, float(num)),
                              "Verify dimensions are in millimetres.")
                        break

        if op in ("fillet", "chamfer"):
            edges_sel = args.get("edges")
            anc = _ancestors(fid) if isinstance(fid, str) else set()
            if has_cad_op:
                for afid in anc:
                    aop = by_id[afid].get("op")
                    if isinstance(aop, str) and aop not in _BREP_CAPABLE_OPS:
                        _warn(warnings, "non_brep_with_cad_op", fp,
                              "ancestor %r (%s) is outside the OCCT B-rep subset"
                              % (afid, aop),
                              "fillet/chamfer require B-rep-capable ancestors.")
            if isinstance(edges_sel, Mapping):
                from_fid = edges_sel.get("from_feature")
                if isinstance(from_fid, str) and from_fid in by_id and from_fid not in anc:
                    _warn(warnings, "non_ancestor_from_feature",
                          fp + ".args.edges.from_feature",
                          "from_feature %r is not an ancestor of %r"
                          % (from_fid, fid),
                          "Ensure from_feature feeds into this fillet/chamfer.")
                upstream_ids = ({from_fid} if isinstance(from_fid, str)
                                and from_fid in by_id else anc)
                upstream_tags: set[str] = set()
                for uid in upstream_ids:
                    upstream_tags |= _collect_feature_tags(by_id[uid])
                wanted_tags: list[str] = []
                if isinstance(edges_sel.get("on_face"), str):
                    wanted_tags.append(edges_sel["on_face"])
                if isinstance(edges_sel.get("common_edge_of"), (list, tuple)):
                    wanted_tags.extend(t for t in edges_sel["common_edge_of"]
                                       if isinstance(t, str))
                for wt in wanted_tags:
                    if wt not in upstream_tags:
                        _warn(warnings, "unresolved_selector_tag",
                              fp + ".args.edges",
                              "edge selector tag %r is not declared on upstream features"
                              % wt,
                              "Declare tags={...} or segment tag=%r on the target feature, "
                              "or verify it is an implicit primitive tag." % wt)

    return {
        "dep_ok": dep_ok,
        "unit_ok": unit_ok,
        "sel_ok": sel_ok,
        "n_features": len(features),
        "n_reachable": len(reachable),
        "patchable_feature_ids": [f["id"] for f in features
                                  if isinstance(f, Mapping)
                                  and isinstance(f.get("id"), str)],
    }


def preflight_spec(spec: Any) -> dict:
    """Fast non-building validation plus dependency/unit/port/selector/patchability lint."""
    v = validate_spec(spec)
    errors = list(v.get("errors") or [])
    warnings: list[dict] = []
    s = v.get("spec")

    if not isinstance(spec, Mapping) or s is None:
        return {
            "ok": False,
            "valid": False,
            "kind": None,
            "spec": None,
            "errors": errors,
            "warnings": warnings,
            "checks": {
                "schema": False, "dependencies": False, "units": False,
                "ports": False, "selectors": False, "patchability": False,
            },
            "summary": {"n_errors": len(errors), "n_warnings": 0,
                        "patchable_targets": []},
        }

    model = s.get("model") or {}
    kind = model.get("kind")
    dep_ok = True
    unit_ok = True
    port_ok = True
    sel_ok = True
    patch_ok = bool(v.get("ok"))
    summary: dict[str, Any] = {"kind": kind}

    ma = (s.get("constraints") or {}).get("mesh_agreement")
    if isinstance(ma, Mapping) and not ma.get("target_model_id"):
        _warn(warnings, "missing_target_model_id",
              "$.constraints.mesh_agreement",
              "mesh_agreement has no target_model_id",
              "Set target_model_id or pass reference=... to atta_verify.")

    if kind == KIND_RECIPE:
        rid = model.get("id")
        patchable_params: list[str] = []
        if isinstance(rid, str) and rid.strip():
            try:
                recipe = _get_recipe(rid)
                specs = _attr(recipe, "params", {}) or {}
                patchable_params = sorted(specs)
                resolved = recipe.resolve(
                    patch=dict(model.get("params") or {}),
                    preset=str(model.get("preset") or "nominal"))
                if bool(_attr(resolved, "blocked", False)):
                    issues = _attr(resolved, "issues", None) or []
                    _err(errors, "recipe_blocked", "$.model.params",
                         "recipe constraints block this parameter set",
                         "Resolve issues: %s"
                         % json.dumps(issues, ensure_ascii=False))
                    dep_ok = False
            except AttaError as exc:
                errors.append(_err_from_atta(exc))
                dep_ok = False
        summary["patchable_targets"] = patchable_params

    elif kind == KIND_GRAPH:
        raw_model = spec.get("model") if isinstance(spec.get("model"), Mapping) else {}
        if "output" not in raw_model or raw_model.get("output") is None:
            _warn(warnings, "implicit_output", "$.model.output",
                  "graph.output is omitted and defaults to the last feature",
                  "Set model.output explicitly before appending features via patch.")
        try:
            ops = _load_feature_ops()
        except AttaError:
            ops = None
        ginfo = _lint_graph_model(model, "$.model", ops, errors, warnings)
        dep_ok = dep_ok and ginfo["dep_ok"]
        unit_ok = unit_ok and ginfo["unit_ok"]
        sel_ok = sel_ok and ginfo["sel_ok"]
        summary["n_features"] = ginfo["n_features"]
        summary["n_reachable"] = ginfo["n_reachable"]
        summary["patchable_targets"] = ginfo["patchable_feature_ids"]

    elif kind == KIND_ASSEMBLY:
        parts = model.get("parts") or []
        mates = model.get("mates") or []
        port_map: dict[str, set[str]] = {}
        has_declared_ports = False
        for part in parts:
            if not isinstance(part, Mapping) or not isinstance(part.get("id"), str):
                continue
            pid = part["id"]
            declared = {p["name"] for p in (part.get("ports") or [])
                        if isinstance(p, Mapping) and isinstance(p.get("name"), str)}
            if declared:
                has_declared_ports = True
            port_map[pid] = _AUTO_PORT_NAMES | declared
        if mates and not has_declared_ports:
            _warn(warnings, "auto_ports_only", "$.model.parts",
                  "assembly has mates but no declared ports (using bbox ports)",
                  "Declare explicit ports for repeatable mating.")

        for i, mate in enumerate(mates):
            if not isinstance(mate, Mapping):
                continue
            for pfield, part_key in (("port_a", "a"), ("port_b", "b")):
                pname = mate.get(pfield)
                pid = mate.get(part_key)
                if isinstance(pname, str) and isinstance(pid, str) and pid in port_map:
                    if pname not in port_map[pid]:
                        port_ok = False
                        _err(errors, "unknown_port",
                             "$.model.mates[%d].%s" % (i, pfield),
                             "part %r has no port %r" % (pid, pname),
                             "Available ports: %s."
                             % ", ".join(sorted(port_map[pid])))

        constrained = set()
        for part in parts:
            if isinstance(part, Mapping) and isinstance(part.get("id"), str):
                if part.get("fixed") or part.get("attach") is not None:
                    constrained.add(part["id"])
        for mate in mates:
            if isinstance(mate, Mapping):
                for key in ("a", "b"):
                    if isinstance(mate.get(key), str):
                        constrained.add(mate[key])
        for i, part in enumerate(parts):
            if isinstance(part, Mapping) and isinstance(part.get("id"), str):
                if part["id"] not in constrained:
                    _warn(warnings, "unconstrained_part",
                          "$.model.parts[%d]" % i,
                          "part %r is not fixed, attached, or mated" % part["id"],
                          "Set fixed=true, attach, or add a mate.")

        try:
            ops = _load_feature_ops()
        except AttaError:
            ops = None
        for i, part in enumerate(parts):
            if isinstance(part, Mapping) and isinstance(part.get("graph"), Mapping):
                ginfo = _lint_graph_model(
                    part["graph"], "$.model.parts[%d].graph" % i,
                    ops, errors, warnings)
                dep_ok = dep_ok and ginfo["dep_ok"]
                unit_ok = unit_ok and ginfo["unit_ok"]
                sel_ok = sel_ok and ginfo["sel_ok"]

        summary["n_parts"] = len(parts)
        summary["n_mates"] = len(mates)
        summary["patchable_targets"] = [p["id"] for p in parts
                                        if isinstance(p, Mapping)
                                        and isinstance(p.get("id"), str)]

    errors = _dedupe_errors(errors)
    schema_ok = bool(v.get("ok"))
    patch_ok = len(errors) == 0
    summary["n_errors"] = len(errors)
    summary["n_warnings"] = len(warnings)
    return {
        "ok": len(errors) == 0,
        "valid": schema_ok,
        "kind": kind,
        "spec": s,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "schema": schema_ok,
            "dependencies": dep_ok and schema_ok,
            "units": unit_ok and schema_ok,
            "ports": port_ok and schema_ok,
            "selectors": sel_ok and schema_ok,
            "patchability": patch_ok,
        },
        "summary": summary,
    }


# ---------------------------------------------------------------------- schema

_RECIPE_EXAMPLE = {
    "schema": SCHEMA,
    "name": "mug-01",
    "model": {
        "kind": "recipe",
        "id": "product.mug",
        "preset": "nominal",
        "params": {"height": 95},
    },
    "quality": "draft",
    "material": "ceramic",
    "finish": {"bevel_mm": 0.8, "smooth_deg": 35},
    "constraints": {
        "bbox": {"x": [60, 140]},
        "min_wall_mm": 2.5,
        "volume_mm3": [100000, 900000],
    },
    "assumptions": [],
}

_GRAPH_EXAMPLE = {
    "schema": SCHEMA,
    "name": "bracket",
    "model": {
        "kind": "graph",
        "units": "mm",
        "segments": 64,
        "output": "cut",
        "features": [
            {"id": "plate", "op": "box", "args": {"size": [80, 50, 6]}},
            {"id": "holes", "op": "cylinder",
             "args": {"radius": 4, "height": 20, "center": [15, 25, 3]}},
            {"id": "cut", "op": "difference",
             "args": {}, "inputs": ["plate", "holes"]},
        ],
    },
    "quality": "draft",
    "material": "steel",
    "finish": {"bevel_mm": 0.5, "smooth_deg": 30},
    "constraints": {},
    "assumptions": [],
}


_ASSEMBLY_EXAMPLE = {
    "schema": SCHEMA,
    "name": "pin_on_plate",
    "model": {
        "kind": "assembly",
        "parts": [
            {"id": "plate", "fixed": True,
             "graph": {"features": [
                 {"id": "plate", "op": "box",
                  "args": {"size": [80, 60, 10], "center": [40, 30, 5]}}]}},
            {"id": "pin",
             "graph": {"features": [
                 {"id": "pin", "op": "cylinder",
                  "args": {"radius": 5, "height": 30,
                           "center": [40, 30, 25]}}]}},
        ],
        "mates": [
            {"id": "m1", "type": "concentric", "a": "plate", "b": "pin",
             "port_a": "axis_z", "port_b": "axis_z", "offset_mm": 20},
        ],
        "solver": {"max_nfev": 200, "tolerance_mm": 0.01},
    },
    "quality": "draft",
    "material": "steel",
    "constraints": {},
    "assumptions": [],
}


def schema() -> dict:
    """Field documentation for agents; three runnable examples."""
    return {
        "schema": SCHEMA,
        "model_kinds": [KIND_RECIPE, KIND_GRAPH, KIND_ASSEMBLY],
        "fields": {
            "schema": "string; must be %r" % SCHEMA,
            "name": "non-empty string",
            "model": "object with kind='recipe', 'graph' or 'assembly'",
            "quality": "one of %s" % ", ".join(QUALITY),
            "material": "one of %s" % ", ".join(sorted(MATERIALS)),
            "finish": dict(FINISH_DOCS),
            "constraints": dict(CONSTRAINT_DOCS),
            "assumptions": "array of human-readable strings (provenance text)",
        },
        "recipe_model": {
            "kind": "recipe",
            "id": "recipe id from atta_catalog",
            "preset": "preset name; defaults to 'nominal'",
            "params": "known recipe params in mm/degrees/counts",
        },
        "graph_model": {
            "kind": "graph",
            "units": "'mm' (fixed)",
            "segments": "integer >= 3 tessellation default",
            "features": "array of {id, op, args, inputs, tags}; a DAG in mm; "
                        "extrude/revolve also accept args.segments (line/arc/tangent_arc/circle); "
                        "fillet/chamfer edges accept common_edge_of, on_face, from_feature",
            "output": "feature id to build; defaults to the last feature",
        },
        "patch_contract": {
            "recipe": "{params, preset, ...} or scalar param keys",
            "graph": "{features: {fid: {args, inputs, op, tags}}, "
                     "add_features: [...], remove_features: [...], output, params}",
            "assembly": "{parts: {pid: {...}}, add_parts, remove_parts, mates, solver}",
        },
        "assembly_model": {
            "kind": "assembly",
            "parts": "non-empty array of {id, recipe|graph|spec|model_id, "
                     "transform:{translate_mm,rotate_deg,matrix}|attach, fixed, ports, "
                     "material, quality, preset, params}",
            "mates": "array of {id, type, a, b, port_a, port_b, offset_mm, "
                     "value_mm, angle_deg, align, weight, allow_interference}",
            "mate_types": list(MATE_TYPES),
            "ports": "declared ports override auto bbox ports: "
                     "{name, origin_mm, axis, normal, x_axis|tangent, radius_mm, kind}; "
                     "axis is frame Z; explicit X is projected perpendicular to Z; "
                     "normal-only oriented ports inherit normal as Z; otherwise "
                     "normal and axis must agree; omitted X keeps legacy roll. "
                     "auto ports are center, axis_x/y/z, face_x-/x+/...",
            "attach": "part field {target, target_port, source_port, gap_mm:0, "
                      "roll_deg:0, facing:'opposite'|'same'}; port names required. "
                      "Mutually exclusive with transform. Stable target DAG order; "
                      "gap along target +Z; roll about target +Z after Rx(180) for "
                      "opposite. Initialization only: add mates to constrain; "
                      "collision/clearance/DOF remain separate checks.",
            "attachments_receipt": "assembly.attachments contains order, items "
                                   "with initial world frames/mm matrix and final "
                                   "position/rotation diagnostics; never a clearance proof",
            "solver": "{max_nfev, tolerance_mm}; least-squares mate solve",
            "clearance": "{threshold_mm} for sampled clearance checks",
            "allow_interference": "array of {a,b} pairs allowed to overlap "
                                  "(press fits, threads)",
        },
        "constraints": dict(CONSTRAINT_DOCS),
        "finish": dict(FINISH_DOCS),
        "quality": list(QUALITY),
        "material": sorted(MATERIALS),
        "examples": {
            "recipe": copy.deepcopy(_RECIPE_EXAMPLE),
            "graph": copy.deepcopy(_GRAPH_EXAMPLE),
            "assembly": copy.deepcopy(_ASSEMBLY_EXAMPLE),
        },
        "assumption_ledger_entry": {
            "id": "A1",
            "path": "$.model.params.height",
            "value": 95,
            "unit": "mm",
            "basis": "design_default",
            "status": "assumed",
            "requires_confirmation": True,
            "why": "recipe ParamSpec.default for 'height'",
        },
        "assumption_rule": (
            "propose may fill design defaults, never measurements. Critical "
            "positioning (hole at, revolve profile, loft sections, sweep path) "
            "is returned in unresolved and must be supplied explicitly."
        ),
        "units": "mm and degrees at this boundary; graph args are mm",
    }


__all__ = [
    "SCHEMA",
    "KIND_RECIPE",
    "KIND_GRAPH",
    "KIND_ASSEMBLY",
    "MATE_TYPES",
    "QUALITY",
    "MATERIALS",
    "CONSTRAINT_DOCS",
    "FINISH_DOCS",
    "canonical_json",
    "spec_hash",
    "validate_spec",
    "propose_spec",
    "compile_spec",
    "patch_spec",
    "preflight_spec",
    "schema",
]
