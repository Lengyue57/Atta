# -*- coding: utf-8 -*-
"""Recipe registry: typed parameters, resolution, presets, constraints.

Split out of :mod:`atta.recipes` so that family modules (``atta/families/*``)
can register themselves without importing the built-in builders.  Everything
here is re-exported by :mod:`atta.recipes` for backwards compatibility.
"""

from __future__ import annotations

import copy
import math
import threading

from .errors import AttaError
from .features import _eval_expr

__all__ = ["ParamSpec", "Recipe", "Resolved", "GenusRule", "evaluate_genus", "recipe",
           "register", "get", "all_recipes", "REGISTRY"]

_LENGTH = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}
_UNITLESS = ("none", "ratio", "count", "deg")


# ---------------------------------------------------------------- registry

class ParamSpec:
    """One typed recipe parameter."""

    __slots__ = ("type", "default", "min", "max", "unit", "description", "group",
                 "advanced", "enum")

    def __init__(self, type="float", default=None, min=None, max=None, unit="none",
                 description="", group="", advanced=False, enum=None):
        self.type = str(type)
        self.default = default
        self.min = min
        self.max = max
        self.unit = str(unit)
        self.description = str(description)
        self.group = str(group)
        self.advanced = bool(advanced)
        self.enum = list(enum) if enum is not None else None

    @classmethod
    def from_dict(cls, data):
        if isinstance(data, cls):
            return data
        data = dict(data or {})
        allowed = set(cls.__slots__)
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_dict(self):
        out = {"type": self.type, "default": self.default, "unit": self.unit,
               "description": self.description, "group": self.group,
               "advanced": self.advanced}
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if self.enum is not None:
            out["enum"] = list(self.enum)
        return out

    def coerce(self, value):
        if self.type == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if self.type == "int":
            if isinstance(value, bool):
                raise ValueError("expected an integer, got a bool")
            x = float(value)
            if not math.isfinite(x) or abs(x - round(x)) > 1e-9:
                raise ValueError("expected an integer")
            return int(round(x))
        if self.type == "float":
            x = float(value)
            if not math.isfinite(x):
                raise ValueError("expected a finite number")
            return x
        if self.type == "enum":
            if value not in (self.enum or []):
                raise ValueError("expected one of %r" % (self.enum or [],))
            return value
        if self.type == "str":
            return str(value)
        return value


class Resolved:
    """Resolved values: ``display`` in recipe units, ``values`` in metres.

    ``expected_genus`` is the recipe's genus rule evaluated on *these* values
    (``None`` when the recipe declares no genus).
    """

    __slots__ = ("values", "display", "issues", "applied_patches", "blocked",
                 "expected_genus")

    def __init__(self, values, display, issues=None, applied_patches=None, blocked=False,
                 expected_genus=None):
        self.values = dict(values)
        self.display = dict(display)
        self.issues = list(issues or [])
        self.applied_patches = list(applied_patches or [])
        self.blocked = bool(blocked)
        self.expected_genus = expected_genus

    def to_dict(self):
        return {"values": self.values, "display": self.display, "issues": self.issues,
                "applied_patches": self.applied_patches, "blocked": self.blocked,
                "expected_genus": self.expected_genus}


# ---------------------------------------------------------------- genus rules

class GenusRule:
    """A genus that is a function of the resolved parameters.

    ``fn`` receives the *display* values (recipe units: mm, counts, deg) and
    returns an integer genus, or ``None`` when no claim can be made for that
    parameter set (verify then skips the genus check instead of asserting a
    number nobody derived).  ``text`` is the human/LLM-readable formula that
    ``to_dict`` publishes, so an agent can see *why* the genus moves.
    """

    __slots__ = ("fn", "text")

    def __init__(self, fn, text=""):
        if not callable(fn):
            raise AttaError("bad_arg", "GenusRule needs a callable", path="expected_genus",
                            fix="pass a function of the display params")
        self.fn = fn
        self.text = str(text or getattr(fn, "__doc__", "") or getattr(fn, "__name__", "fn"))

    def __call__(self, display):
        return self.fn(dict(display))


def _is_static_genus(rule):
    return rule is None or (isinstance(rule, (int, float)) and not isinstance(rule, bool))


def evaluate_genus(rule, display, recipe_id="recipe"):
    """Evaluate a genus rule (int | expression string | callable) on display values.

    * ``int``      -> itself (the legacy constant).
    * ``str``      -> a safe arithmetic expression over the param names, e.g.
                      ``"1 + bolt_holes"`` or ``"rows * cols"`` (same evaluator
                      and function set as recipe constraints; no ``eval``).
    * callable     -> ``fn(display) -> int | None``.

    The result must be a non-negative integer (or ``None`` from a callable);
    anything else is a recipe bug and raises ``AttaError(code="bad_genus_rule")``
    rather than silently disabling the genus check.
    """
    path = "%s.expected_genus" % recipe_id
    if rule is None:
        return None
    if _is_static_genus(rule):
        value = rule
    elif isinstance(rule, str):
        value = _eval_expr(rule, dict(display), path)
    elif callable(rule):
        try:
            value = rule(dict(display))
        except AttaError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AttaError("bad_genus_rule", "genus rule failed: %s" % exc,
                            path=path, fix="fix the expected_genus callable") from exc
        if value is None:
            return None
    else:
        raise AttaError("bad_genus_rule", "expected_genus must be an int, expression or callable",
                        path=path, fix="use an int, a string like '1 + holes' or a GenusRule")
    if isinstance(value, bool):
        raise AttaError("bad_genus_rule", "genus rule returned a bool %r, not an integer" % (value,),
                        path=path, fix="return a non-negative integer genus")
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise AttaError("bad_genus_rule", "genus rule returned a non-number %r" % (value,),
                        path=path, fix="return an integer genus") from None
    if not math.isfinite(f) or f < 0 or abs(f - round(f)) > 1e-9:
        raise AttaError("bad_genus_rule", "genus rule returned %r, not a non-negative integer" % (value,),
                        path=path, fix="return an integer genus (or None for 'no claim')")
    return int(round(f))


class Recipe:
    """A named parametric family with a deterministic builder.

    ``expected_genus`` may be a constant, an expression string over the
    parameter names or a callable/:class:`GenusRule` of the display values.
    Reading ``recipe.expected_genus`` returns the genus for the parameter set
    most recently resolved or built **on the calling thread** (the engine and
    the coverage gate always ``resolve`` -> ``build_shape`` -> read), falling
    back to the nominal preset.  Race-free callers should use
    ``Resolved.expected_genus`` or :meth:`genus_for`.
    """

    def __init__(self, id, name, description, params, presets=None, constraints=None,
                 expected_genus=0, material="ceramic", tags=None, build=None):
        self.id = str(id)
        self.name = str(name)
        self.description = str(description)
        self.params = {str(k): ParamSpec.from_dict(v) for k, v in dict(params).items()}
        self.presets = copy.deepcopy(dict(presets or {}))
        self.constraints = copy.deepcopy(list(constraints or []))
        self._local = threading.local()
        self._genus_rule = expected_genus
        self.material = str(material)
        self.tags = list(tags or [])
        self.build = build

    # -- genus ---------------------------------------------------------------
    @property
    def genus_rule(self):
        """The declared rule (int, expression string, or callable)."""
        return self._genus_rule

    @property
    def genus_is_parametric(self):
        return not _is_static_genus(self._genus_rule)

    def genus_for(self, display):
        """Genus for a full or partial set of display values (defaults fill gaps)."""
        full = self._nominal_display()
        full.update(dict(display or {}))
        return evaluate_genus(self._genus_rule, full, self.id)

    def _nominal_display(self):
        raw = {k: spec.default for k, spec in self.params.items()}
        raw.update(self.presets.get("nominal", {}))
        out = {}
        for key, spec in self.params.items():
            try:
                out[key] = spec.coerce(raw.get(key, spec.default))
            except (TypeError, ValueError):
                out[key] = spec.default
        return out

    def _display_from_values(self, values):
        """Invert ``resolve``'s unit conversion (metres -> recipe units)."""
        out = {}
        for key, val in dict(values or {}).items():
            spec = self.params.get(key)
            if (spec is not None and spec.unit in _LENGTH and not isinstance(val, bool)
                    and isinstance(val, (int, float))):
                out[key] = val / _LENGTH[spec.unit]
            else:
                out[key] = val
        return out

    @property
    def expected_genus(self):
        if not self.genus_is_parametric:
            return self._genus_rule
        if getattr(self._local, "has", False):
            return self._local.genus
        return self.genus_for({})

    @expected_genus.setter
    def expected_genus(self, value):
        # Legacy builders publish a derived genus by assignment.  For a static
        # recipe that replaces the constant (old behaviour); for a parametric
        # one it only records the value for this thread's current build, so a
        # publish can never overwrite the rule for the next parameter set.
        if self.genus_is_parametric and _is_static_genus(value):
            self._remember(value)
            return
        self._genus_rule = value
        self._local.has = False

    def _remember(self, genus):
        self._local.genus = genus
        self._local.has = True

    def resolve(self, patch=None, preset="nominal") -> Resolved:
        patch = dict(patch or {})
        raw = {k: spec.default for k, spec in self.params.items()}
        raw.update(self.presets.get("nominal", {}))
        if preset and preset != "nominal":
            if preset not in self.presets:
                raise AttaError("unknown_preset", "unknown preset %r for %s" % (preset, self.id),
                                path="preset",
                                fix="use one of: " + ", ".join(sorted(self.presets)))
            raw.update(self.presets[preset])

        issues, applied, blocked = [], [], False
        for key in sorted(set(patch) - set(self.params)):
            issues.append({"code": "unknown_param", "severity": "warn", "path": key,
                           "message": "%s has no parameter %r; it was ignored" % (self.id, key),
                           "fix": "use one of: " + ", ".join(sorted(self.params))})
        raw.update({k: v for k, v in patch.items() if k in self.params})

        display, values = {}, {}
        for key, spec in self.params.items():
            value = raw.get(key, spec.default)
            try:
                value = spec.coerce(value)
            except (TypeError, ValueError) as exc:
                issues.append({"code": "bad_param", "severity": "error", "path": key,
                               "message": "%s: %s" % (key, exc), "actual": repr(value),
                               "expected": spec.type,
                               "fix": "pass a %s for %s" % (spec.type, key)})
                blocked = True
                try:
                    value = spec.coerce(spec.default)
                except (TypeError, ValueError):
                    continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                display[key] = value
                values[key] = value
                continue
            if spec.min is not None and value < spec.min:
                issues.append({"code": "param_out_of_range", "severity": "warn", "path": key,
                               "message": "%s clamped to minimum %s" % (key, spec.min),
                               "actual": value, "expected": ">= %s" % spec.min,
                               "fix": "%s = %s" % (key, spec.min)})
                value = spec.min
                applied.append({"op": "clamp", "path": key, "value": spec.min})
            if spec.max is not None and value > spec.max:
                issues.append({"code": "param_out_of_range", "severity": "warn", "path": key,
                               "message": "%s clamped to maximum %s" % (key, spec.max),
                               "actual": value, "expected": "<= %s" % spec.max,
                               "fix": "%s = %s" % (key, spec.max)})
                value = spec.max
                applied.append({"op": "clamp", "path": key, "value": spec.max})
            display[key] = value
            values[key] = value * _LENGTH[spec.unit] if spec.unit in _LENGTH else value

        for con in self.constraints:
            expr = con.get("expr")
            if not expr:
                continue
            try:
                ok = bool(_eval_expr(expr, dict(display), "constraint." + str(con.get("id", "?")),
                                     allow_bool=True))
            except AttaError as exc:
                issues.append({"code": "constraint_error", "severity": "error",
                               "path": str(con.get("id", "constraint")),
                               "message": exc.message, "fix": exc.fix})
                blocked = True
                continue
            if not ok:
                sev = str(con.get("severity", "error"))
                issues.append({"code": str(con.get("id", "constraint_failed")),
                               "severity": sev,
                               "path": str(con.get("id", "constraint")),
                               "message": str(con.get("message")
                                              or "constraint failed: %s" % expr),
                               "fix": con.get("fix")})
                if sev == "error":
                    blocked = True
        genus = None
        if self._genus_rule is not None:
            try:
                genus = evaluate_genus(self._genus_rule, display, self.id)
            except AttaError as exc:
                issues.append({"code": exc.code, "severity": "error", "path": exc.path,
                               "message": exc.message, "fix": exc.fix})
                blocked = True
        if self.genus_is_parametric:
            self._remember(genus)
        return Resolved(values, display, issues, applied, blocked, expected_genus=genus)

    def build_shape(self, values, quality="draft"):
        if self.build is None:
            raise AttaError("no_builder", "recipe %s has no builder" % self.id,
                            path="recipe", fix="choose a registered recipe")
        if self.genus_is_parametric:
            # the genus that verify will compare against is the one for the
            # values actually being built, not whatever was resolved last
            self._remember(self.genus_for(self._display_from_values(values)))
        return self.build(dict(values), str(quality))

    def to_dict(self, verbose=False):
        out = {"id": self.id, "name": self.name, "description": self.description,
               "tags": list(self.tags), "presets": list(self.presets),
               "n_params": len(self.params),
               "expected_genus": (self.genus_for({}) if self.genus_is_parametric
                                  else self._genus_rule),
               "material": self.material}
        if self.genus_is_parametric:
            rule = self._genus_rule
            out["expected_genus_rule"] = (rule if isinstance(rule, str)
                                          else getattr(rule, "text", None)
                                          or getattr(rule, "__name__", "callable"))
        if verbose:
            out["params"] = {k: v.to_dict() for k, v in self.params.items()}
            out["presets"] = copy.deepcopy(self.presets)
            out["constraints"] = copy.deepcopy(self.constraints)
        return out


REGISTRY: dict[str, Recipe] = {}


def register(rec) -> Recipe:
    if not isinstance(rec, Recipe):
        raise AttaError("bad_arg", "register expects a Recipe", path="recipe",
                        fix="construct Recipe(...)")
    if rec.id in REGISTRY:
        raise AttaError("duplicate_recipe", "recipe id %r is already registered" % rec.id,
                        path="recipe.id", fix="choose a unique recipe id")
    REGISTRY[rec.id] = rec
    return rec


def get(recipe_id) -> Recipe:
    key = str(recipe_id)
    if key not in REGISTRY:
        raise AttaError("unknown_recipe", "unknown recipe %r" % key, path="recipe",
                        fix="available recipes: " + ", ".join(sorted(REGISTRY)))
    return REGISTRY[key]


def all_recipes():
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def recipe(recipe_id, name, description, params, presets=None, constraints=None,
           expected_genus=0, material="ceramic", tags=None):
    def deco(fn):
        register(Recipe(id=recipe_id, name=name, description=description, params=params,
                        presets=presets, constraints=constraints,
                        expected_genus=expected_genus, material=material,
                        tags=tags, build=fn))
        return fn
    return deco


