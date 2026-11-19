# -*- coding: utf-8 -*-
"""Shape-grammar graph ops: ``shape_grammar`` (CFDG / Structure Synth style)
and ``lsystem`` (ABOP turtle).  Public lengths are model units (mm); the
returned mesh is in metres.  Unit conversion happens ONLY in ``prepare``.

Both ops are *declarative*: rules are JSON data, parametric L-system
expressions go through the same safe arithmetic evaluator as feature
``$expr`` values (no ``eval``).  Every expansion is bounded by hard budgets
that fail closed with an ``AttaError(code, path, fix)`` before any geometry
is built; the result is one boolean union through manifold3d and must be a
closed manifold.  See docs/GRAMMAR.md for the language reference.
"""
from __future__ import annotations

import ast
import math
import random

import numpy as np

from ..errors import AttaError
from ..features import _eval_expr, register_op
from ..mesh import Mesh, oriented, topology, weld

__all__ = ["compile_shape_grammar", "compile_lsystem", "build_shape_grammar",
           "build_lsystem", "expand_shape_grammar", "expand_lsystem",
           "interpret_lsystem", "PRIMITIVES"]

_UNITS = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}

# ------------------------------------------------------------------ limits

#: Absolute ceilings: a caller may lower a budget, never raise it past these.
CEILING = {"max_depth": 10000, "max_instances": 50000, "max_expansions": 1000000,
           "max_triangles": 5000000, "max_symbols": 1000000, "max_segments": 50000,
           "iterations": 32, "loop_count": 100000, "segments": 128}
DEFAULTS = {"max_depth": 256, "max_instances": 4096, "max_expansions": 200000,
            "max_triangles": 1000000, "max_symbols": 200000, "max_segments": 5000}

#: Unit primitives, centred on the origin, 1 grammar unit across (CFDG style).
PRIMITIVES = ("box", "sphere", "cylinder", "cone", "torus", "square", "circle", "triangle")
#: CFDG colour/alpha adjustments: accepted so a port is faithful, no geometry.
_COLOUR_KEYS = ("hue", "h", "sat", "saturation", "b", "brightness", "a", "alpha", "color")
_TRANSFORM_KEYS = ("x", "y", "z", "t", "r", "rx", "ry", "rz", "s", "flip", "fx", "fy", "fz")


def _err(code, message, path, fix, **details):
    return AttaError(code, message, path=path, fix=fix, details=details or None)


# ------------------------------------------------------------------ scalars

def _number(value, path, *, positive=False, nonneg=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _err("bad_arg", "expected a number", path, "pass a finite number")
    x = float(value)
    if not math.isfinite(x):
        raise _err("bad_arg", "number must be finite", path, "pass a finite number")
    if positive and x <= 0:
        raise _err("bad_arg", "number must be > 0", path, "pass a positive number")
    if nonneg and x < 0:
        raise _err("bad_arg", "number must be >= 0", path, "pass a non-negative number")
    return x


def _integer(value, path, minimum, maximum):
    x = _number(value, path)
    if abs(x - round(x)) > 1e-9:
        raise _err("bad_arg", "expected a whole number", path, "pass an integer")
    n = int(round(x))
    if n < minimum:
        raise _err("bad_arg", "must be >= %d" % minimum, path, "pass an integer >= %d" % minimum)
    if n > maximum:
        raise _err("grammar_budget_ceiling", "must be <= %d (hard ceiling)" % maximum, path,
                   "lower the value to <= %d or simplify the grammar" % maximum)
    return n


def _budget(args, key, path, minimum=1):
    if key not in args:
        return DEFAULTS[key]
    return _integer(args[key], path + ".args." + key, minimum, CEILING[key])


def _seed(args, path):
    if "seed" not in args:
        return 0
    return _integer(args["seed"], path + ".args.seed", 0, 2 ** 63 - 1)


def _segments(args, path, default):
    if "segments" not in args:
        return default
    return _integer(args["segments"], path + ".args.segments", 3, CEILING["segments"])


def _max_components(args, path):
    if args.get("max_components") is None:
        return None
    return _integer(args["max_components"], path + ".args.max_components", 1, 10 ** 9)


# ------------------------------------------------------------------ matrices

def _rot(axis, deg):
    a = math.radians(math.fmod(deg, 360.0))
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


def _translate(v):
    m = np.eye(4)
    m[:3, 3] = v
    return m


def _scale(v):
    return np.diag([v[0], v[1], v[2], 1.0])


def _flip(deg):
    """Reflection across the line through the origin at ``deg`` in the xy-plane."""
    a = math.radians(2.0 * math.fmod(deg, 180.0))
    m = np.eye(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = math.cos(a), math.sin(a), math.sin(a), -math.cos(a)
    return m


def _mirror(axis):
    m = np.eye(4)
    m["xyz".index(axis), "xyz".index(axis)] = -1.0
    return m


def _scale_bounds(m):
    """(min_singular_value, max_singular_value) of the 3x3 linear block."""
    sv = np.linalg.svd(m[:3, :3], compute_uv=False)
    return float(sv[2]), float(sv[0])


# ------------------------------------------------------------------ values

class _Rand:
    """``{"rand": [lo, hi]}``: uniform draw from the seeded stream at expansion."""
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def draw(self, rng):
        return self.lo + (self.hi - self.lo) * rng.random()


def _value(v, path, *, positive=False):
    if isinstance(v, dict):
        if set(v) != {"rand"}:
            raise _err("bad_arg", "a transform value object must be {\"rand\": [lo, hi]}", path,
                       "use a number or {\"rand\": [lo, hi]}")
        r = v["rand"]
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            raise _err("bad_arg", "rand needs [lo, hi]", path + ".rand", "pass two numbers")
        lo = _number(r[0], path + ".rand[0]", positive=positive)
        hi = _number(r[1], path + ".rand[1]", positive=positive)
        if hi < lo:
            raise _err("bad_arg", "rand range must be ordered", path + ".rand", "use [lo, hi] with lo <= hi")
        return _Rand(lo, hi)
    return _number(v, path, positive=positive)


def _real(v, rng):
    return v.draw(rng) if isinstance(v, _Rand) else v


def _has_rand(obj):
    if isinstance(obj, _Rand):
        return True
    if isinstance(obj, (list, tuple)):
        return any(_has_rand(x) for x in obj)
    return False


# ------------------------------------------------------------------ transforms

def _compile_step(spec, path):
    """One canonical CFDG ``[ ]`` adjustment -> list of (kind, payload)."""
    if not isinstance(spec, dict):
        raise _err("bad_arg", "transform must be an object (or a list of objects)", path,
                   "use e.g. {\"x\": 1, \"r\": 45, \"s\": 0.9}")
    ops, ignored = [], []
    for key in spec:
        if key in _COLOUR_KEYS:
            ignored.append(key)
        elif key not in _TRANSFORM_KEYS:
            raise _err("bad_arg", "unknown transform key %r" % key, path + "." + str(key),
                       "use one of: " + ", ".join(_TRANSFORM_KEYS) + " (colour keys are ignored)")
    if "r" in spec and "rz" in spec:
        raise _err("bad_arg", "r and rz both rotate about z", path + ".r", "use only one of r / rz")
    if "t" in spec and any(k in spec for k in "xyz"):
        raise _err("bad_arg", "t conflicts with x/y/z", path + ".t", "use either t or x/y/z")
    if "t" in spec:
        t = spec["t"]
        if not isinstance(t, (list, tuple)) or len(t) not in (2, 3):
            raise _err("bad_arg", "t must be [x, y] or [x, y, z]", path + ".t", "pass 2 or 3 numbers")
        tv = [_value(c, path + ".t[%d]" % i) for i, c in enumerate(t)] + [0.0] * (3 - len(t))
        ops.append(("t", tv))
    elif any(k in spec for k in "xyz"):
        ops.append(("t", [_value(spec[k], path + "." + k) if k in spec else 0.0 for k in "xyz"]))
    # M = T * Rz * Ry * Rx : x rotation is applied to the shape first.
    for key, axis in (("rz", "z"), ("r", "z"), ("ry", "y"), ("rx", "x")):
        if key in spec:
            ops.append(("r", (axis, _value(spec[key], path + "." + key))))
    if "s" in spec:
        s = spec["s"]
        if isinstance(s, (list, tuple)):
            if len(s) not in (2, 3):
                raise _err("bad_arg", "s must be a number, [sx, sy] or [sx, sy, sz]", path + ".s",
                           "pass 1, 2 or 3 positive numbers")
            sv = [_value(c, path + ".s[%d]" % i, positive=True) for i, c in enumerate(s)] + [1.0] * (3 - len(s))
        else:
            v = _value(s, path + ".s", positive=True)
            sv = [v, v, v]
        ops.append(("s", sv))
    if "flip" in spec:
        ops.append(("flip", _value(spec["flip"], path + ".flip")))
    for key in ("fx", "fy", "fz"):
        if key in spec:
            if not isinstance(spec[key], bool):
                raise _err("bad_arg", "%s must be true/false" % key, path + "." + key, "pass true")
            if spec[key]:
                ops.append(("m", key[1]))
    return ops, ignored


class _Transform:
    """Compiled transform; constant ones are folded to a matrix once."""
    __slots__ = ("ops", "const")

    def __init__(self, ops):
        self.ops = ops
        self.const = None if _has_rand([p for _, p in ops if not isinstance(p, str)]
                                       + [p[1] for k, p in ops if k == "r"]) else self._matrix(None)

    def _matrix(self, rng):
        m = np.eye(4)
        for kind, p in self.ops:
            if kind == "t":
                m = m @ _translate([_real(c, rng) for c in p])
            elif kind == "r":
                m = m @ _rot(p[0], _real(p[1], rng))
            elif kind == "s":
                m = m @ _scale([_real(c, rng) for c in p])
            elif kind == "flip":
                m = m @ _flip(_real(p, rng))
            elif kind == "m":
                m = m @ _mirror(p)
        return m

    def matrix(self, rng):
        return self.const if self.const is not None else self._matrix(rng)


def _compile_transform(spec, path, ignored):
    if spec is None:
        return _Transform([])
    steps = spec if isinstance(spec, (list, tuple)) else [spec]
    ops = []
    for i, step in enumerate(steps):
        sub, ign = _compile_step(step, path + ("[%d]" % i if isinstance(spec, (list, tuple)) else ""))
        ops.extend(sub)
        ignored.update(ign)
    return _Transform(ops)


# ------------------------------------------------------------------ symmetry

def _axis_frame(axis):
    """Permutation taking the symmetry axis to z (so groups are built about z)."""
    p = np.eye(4)
    if axis == "x":
        p[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    elif axis == "y":
        p[:3, :3] = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    return p


def _parse_cf_string(text, path):
    parts = [p.strip() for p in str(text).split(",")]
    head = parts[0].lower().replace(" ", "")
    kinds = {"cf::cyclic": "cyclic", "cf::dihedral": "dihedral"}
    if head not in kinds or len(parts) != 2:
        raise _err("bad_arg", "symmetry string must be 'CF::Cyclic, n' or 'CF::Dihedral, n'", path,
                   "use the object form {\"type\": \"dihedral\", \"n\": 6, \"center\": [x, y, z]} "
                   "for a centre or mirror angle")
    try:
        n = float(parts[1])
    except ValueError:
        raise _err("bad_arg", "symmetry order must be an integer", path, "use e.g. 'CF::Dihedral, 6'")
    return {"type": kinds[head], "n": n}


def _symmetry_group(spec, path, max_order):
    """List of 4x4 matrices (identity first) for the directive list."""
    if spec is None:
        return [np.eye(4)], []
    items = spec if isinstance(spec, (list, tuple)) else [spec]
    group, labels = [np.eye(4)], []
    for i, item in enumerate(items):
        where = path + ("[%d]" % i if isinstance(spec, (list, tuple)) else "")
        if isinstance(item, str):
            item = _parse_cf_string(item, where)
        if not isinstance(item, dict):
            raise _err("bad_arg", "symmetry directive must be an object or a CF:: string", where,
                       "use {\"type\": \"cyclic\", \"n\": 6}")
        kind = str(item.get("type", "")).lower()
        allowed = {"cyclic": {"type", "n", "axis", "center"},
                   "dihedral": {"type", "n", "axis", "center", "mirror_angle"},
                   "reflect": {"type", "axis", "axes", "center"}}
        if kind not in allowed:
            raise _err("bad_arg", "symmetry type must be cyclic, dihedral or reflect", where + ".type",
                       "use type cyclic|dihedral|reflect")
        extra = set(item) - allowed[kind]
        if extra:
            raise _err("bad_arg", "unexpected key(s) %s in %s symmetry" % (sorted(extra), kind),
                       where, "allowed: " + ", ".join(sorted(allowed[kind])))
        center = item.get("center", [0.0, 0.0, 0.0])
        if not isinstance(center, (list, tuple)) or len(center) not in (2, 3):
            raise _err("bad_arg", "center must be [x, y] or [x, y, z]", where + ".center",
                       "pass grammar-unit coordinates")
        c = [_number(v, where + ".center[%d]" % j) for j, v in enumerate(center)] + [0.0] * (3 - len(center))
        to_c, from_c = _translate(c), _translate([-v for v in c])
        if kind == "reflect":
            axes = item.get("axes", [item.get("axis", "x")])
            if isinstance(axes, str):
                axes = [axes]
            if not isinstance(axes, (list, tuple)) or not axes or any(a not in ("x", "y", "z") for a in axes):
                raise _err("bad_arg", "reflect axes must be a list of x|y|z", where + ".axes",
                           "use {\"type\": \"reflect\", \"axes\": [\"x\"]}")
            mats = [np.eye(4)]
            for a in dict.fromkeys(axes):
                mats = mats + [to_c @ _mirror(a) @ from_c @ m for m in mats]
            labels.append("reflect(%s)" % ",".join(dict.fromkeys(axes)))
        else:
            n = _integer(item.get("n"), where + ".n", 1, CEILING["max_instances"])
            axis = item.get("axis", "z")
            if axis not in ("x", "y", "z"):
                raise _err("bad_arg", "axis must be x, y or z", where + ".axis", "use axis 'z'")
            p = _axis_frame(axis)
            rots = [_rot("z", 360.0 * k / n) for k in range(n)]
            local = list(rots)
            if kind == "dihedral":
                ang = _number(item.get("mirror_angle", 90.0), where + ".mirror_angle")
                local += [r @ _flip(ang) for r in rots]
            mats = [to_c @ p @ m @ p.T @ from_c for m in local]
            labels.append("%s(%d,%s)" % (kind, n, axis))
        group = [g @ m for g in group for m in mats]
        if len(group) > max_order:
            raise _err("grammar_instance_budget", "symmetry group order %d exceeds max_instances" % len(group),
                       where, "lower n or raise max_instances")
    # Deduplicate coincident group elements (e.g. cyclic 2 + reflect x,y).
    seen, unique = set(), []
    for g in group:
        key = tuple(np.round(g, 9).ravel())
        if key not in seen:
            seen.add(key)
            unique.append(g)
    return unique, labels


# ------------------------------------------------------------------ primitives

def _proto(name, segments, tile):
    import manifold3d as mf
    M = mf.Manifold
    if name == "box":
        return M.cube((1.0, 1.0, 1.0), True)
    if name == "square":
        return M.cube((1.0, 1.0, tile), True)
    if name == "sphere":
        return M.sphere(0.5, segments)
    if name == "cylinder":
        return M.cylinder(1.0, 0.5, 0.5, segments, True)
    if name == "circle":
        return M.cylinder(tile, 0.5, 0.5, segments, True)
    if name == "cone":
        return M.cylinder(1.0, 0.5, 0.0, segments, True)
    if name == "torus":
        cs = mf.CrossSection.circle(0.15, max(3, segments // 2)).translate((0.35, 0.0))
        return M.revolve(cs, segments)
    if name == "triangle":  # CFDG TRIANGLE: unit-side equilateral about its centroid
        h = math.sqrt(3.0) / 2.0
        pts = [(-0.5, -h / 3.0), (0.5, -h / 3.0), (0.0, 2.0 * h / 3.0)]
        return M.extrude(mf.CrossSection([pts]), tile).translate((0.0, 0.0, -tile / 2.0))
    raise KeyError(name)


def _to_4x3(m):
    return np.ascontiguousarray(m[:3, :4], dtype=np.float64)


def _union(protos, instances, scale_m, max_triangles, max_components, path, kind, meta):
    """Union transformed prototypes through manifold3d; enforce budgets first."""
    import manifold3d as mf
    estimate = sum(int(protos[name].num_tri()) for name, _ in instances)
    if estimate > max_triangles:
        raise _err("grammar_triangle_budget", "%s input would have %d triangles > max_triangles=%d"
                   % (kind, estimate, max_triangles), path + ".args.max_triangles",
                   "lower recursion/iterations/segments or raise max_triangles", estimate=estimate)
    if not instances:
        raise _err("grammar_empty", "%s produced no geometry" % kind, path + ".args",
                   "make sure the start rule reaches a primitive (check min_scale and weights)")
    world = _scale([scale_m] * 3)
    parts = [protos[name].transform(_to_4x3(world @ m)) for name, m in instances]
    solid = mf.Manifold.batch_boolean(parts, mf.OpType.Add) if len(parts) > 1 else parts[0]
    status = solid.status()
    if status != mf.Error.NoError or solid.is_empty():
        raise _err("grammar_boolean_failed", "%s union failed (%s)" % (kind, status), path,
                   "check for degenerate (zero-scale) instances or lower the instance count")
    if int(solid.num_tri()) > max_triangles:
        raise _err("grammar_triangle_budget", "%s output has %d triangles > max_triangles=%d"
                   % (kind, solid.num_tri(), max_triangles), path + ".args.max_triangles",
                   "lower segments/instances or raise max_triangles")
    raw = solid.to_mesh64() if hasattr(solid, "to_mesh64") else solid.to_mesh()
    verts = np.asarray(raw.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(raw.tri_verts, dtype=np.int64)
    mesh = oriented(Mesh(verts, faces, {"kind": kind, "op": kind}))
    topo = topology(mesh)
    if not topo["closed"] or topo["degenerate_faces"] > 0:
        retry = oriented(weld(mesh, 1e-12))
        rtopo = topology(retry)
        if rtopo["closed"] and rtopo["degenerate_faces"] <= topo["degenerate_faces"]:
            mesh, topo = retry, rtopo
    if not topo["closed"]:
        raise _err("grammar_not_closed", "%s union is not a closed manifold" % kind, path,
                   "raise segments, avoid zero-thickness tiles, or perturb coincident instances")
    if max_components is not None and topo["n_components"] > max_components:
        raise _err("grammar_components", "%s produced %d components > max_components=%d"
                   % (kind, topo["n_components"], max_components), path + ".args.max_components",
                   "make successive instances overlap (smaller steps / larger shapes) or "
                   "raise max_components", components=int(topo["n_components"]))
    meta.update({"triangles": int(topo["n_tris"]), "components": int(topo["n_components"]),
                 "genus": topo["genus"], "backend": "manifold3d.batch_boolean"})
    mesh.meta[kind] = meta
    return mesh


# ================================================================== shape_grammar

class _Item:
    __slots__ = ("shape", "call", "transform", "loop_count", "loop_transform")


class _Alt:
    __slots__ = ("weight", "body", "max_depth", "then")


def _compile_item(item, path, rules, ignored):
    if not isinstance(item, dict):
        raise _err("bad_arg", "body item must be an object", path,
                   "use {\"shape\": \"box\", \"transform\": {...}} or {\"call\": \"RULE\"}")
    extra = set(item) - {"shape", "call", "transform", "loop"}
    if extra:
        raise _err("bad_arg", "unexpected body key(s) %s" % sorted(extra), path,
                   "allowed: shape | call, transform, loop")
    out = _Item()
    if ("shape" in item) == ("call" in item):
        raise _err("bad_arg", "body item needs exactly one of shape / call", path,
                   "use shape for a primitive, call for a rule")
    out.shape = item.get("shape")
    out.call = item.get("call")
    if out.shape is not None and out.shape not in PRIMITIVES:
        raise _err("bad_arg", "unknown primitive %r" % (out.shape,), path + ".shape",
                   "use one of: " + ", ".join(PRIMITIVES))
    if out.call is not None and out.call not in rules:
        raise _err("bad_arg", "call to undefined rule %r" % (out.call,), path + ".call",
                   "define it in rules; available: " + ", ".join(sorted(map(str, rules))))
    out.transform = _compile_transform(item.get("transform"), path + ".transform", ignored)
    out.loop_count, out.loop_transform = 1, None
    if "loop" in item:
        loop = item["loop"]
        if isinstance(loop, dict):
            if set(loop) - {"count", "transform"} or "count" not in loop:
                raise _err("bad_arg", "loop must be {count, transform}", path + ".loop",
                           "use {\"count\": 12, \"transform\": {\"r\": 30}}")
            out.loop_count = _integer(loop["count"], path + ".loop.count", 0, CEILING["loop_count"])
            out.loop_transform = _compile_transform(loop.get("transform"), path + ".loop.transform", ignored)
        else:
            raise _err("bad_arg", "loop must be an object", path + ".loop",
                       "use {\"count\": 12, \"transform\": {\"r\": 30}}")
    return out


def _compile_alts(name, value, path, rules, ignored):
    def is_alt(x):
        return isinstance(x, dict) and "body" in x
    if is_alt(value):
        raw = [value]
    elif isinstance(value, (list, tuple)) and value and all(is_alt(x) for x in value):
        raw = list(value)
    elif isinstance(value, (list, tuple)) and not any(is_alt(x) for x in value):
        raw = [{"body": list(value)}]
    else:
        raise _err("bad_arg", "rule %r must be a body list, an alternative {weight, body} or a list "
                   "of alternatives" % name, path, "use [{\"weight\": 1, \"body\": [...]}]")
    alts = []
    for i, alt in enumerate(raw):
        where = path + "[%d]" % i
        extra = set(alt) - {"weight", "body", "max_depth", "then"}
        if extra:
            raise _err("bad_arg", "unexpected rule key(s) %s" % sorted(extra), where,
                       "allowed: weight, body, max_depth, then")
        a = _Alt()
        a.weight = _number(alt.get("weight", 1.0), where + ".weight", nonneg=True)
        body = alt.get("body")
        if not isinstance(body, (list, tuple)):
            raise _err("bad_arg", "body must be a list (may be empty)", where + ".body",
                       "use [] for a terminating rule")
        a.body = [_compile_item(it, where + ".body[%d]" % j, rules, ignored) for j, it in enumerate(body)]
        a.max_depth = None
        if "max_depth" in alt:
            a.max_depth = _integer(alt["max_depth"], where + ".max_depth", 0, CEILING["max_depth"])
        a.then = alt.get("then")
        if a.then is not None and a.then not in rules:
            raise _err("bad_arg", "then names undefined rule %r" % (a.then,), where + ".then",
                       "available: " + ", ".join(sorted(map(str, rules))))
        if a.then is not None and a.max_depth is None:
            raise _err("bad_arg", "then needs max_depth", where + ".then", "add max_depth")
        alts.append(a)
    if sum(a.weight for a in alts) <= 0:
        raise _err("bad_arg", "rule %r has zero total weight" % name, path, "give one alternative weight > 0")
    return alts


def compile_shape_grammar(args, scale=1e-3, path="shape_grammar"):
    """Validate ``args`` and return the keyword dict the op builder takes."""
    if not isinstance(args, dict):
        raise _err("bad_arg", "args must be an object", path + ".args", "pass a dict")
    rules_in = args.get("rules")
    if not isinstance(rules_in, dict) or not rules_in:
        raise _err("bad_arg", "rules must be a non-empty object name -> alternatives",
                   path + ".args.rules", "use {\"R\": [{\"weight\": 1, \"body\": [...]}]}")
    for name in rules_in:
        if not isinstance(name, str) or not name:
            raise _err("bad_arg", "rule names must be non-empty strings", path + ".args.rules",
                       "rename the rule")
    start = args.get("startshape")
    if not isinstance(start, str) or start not in rules_in:
        raise _err("bad_arg", "startshape must name a rule", path + ".args.startshape",
                   "available: " + ", ".join(sorted(rules_in)))
    ignored = set()
    rules = {name: _compile_alts(name, value, path + ".args.rules." + name, rules_in, ignored)
             for name, value in rules_in.items()}
    factor = _number(scale, path + ".units", positive=True)
    unit_m = _number(args.get("unit", 1.0), path + ".args.unit", positive=True) * factor
    min_scale = _number(args.get("min_scale", 1e-3), path + ".args.min_scale", positive=True)
    tile_thickness = _number(args.get("tile_thickness", 0.1), path + ".args.tile_thickness", positive=True)
    if unit_m * min_scale < 1e-12:
        raise _err("bad_arg", "min_scale * unit is below kernel precision (1e-12 m)",
                   path + ".args.min_scale", "raise min_scale or unit")
    if unit_m * tile_thickness < 1e-9 or tile_thickness > 1e6:
        raise _err("bad_arg", "tile_thickness is out of bounds",
                   path + ".args.tile_thickness", "use tile_thickness between 1e-4 and 100")
    max_instances = _budget(args, "max_instances", path)
    out = {
        "rules": rules, "startshape": start,
        "start_transform": _compile_transform(args.get("start_transform"), path + ".args.start_transform", ignored),
        "seed": _seed(args, path),
        "max_depth": _budget(args, "max_depth", path, 0),
        "max_instances": max_instances,
        "max_expansions": _budget(args, "max_expansions", path),
        "max_triangles": _budget(args, "max_triangles", path),
        "max_components": _max_components(args, path),
        "segments": _segments(args, path, 24),
        "min_scale": min_scale,
        "tile_thickness": tile_thickness,
        "unit_m": unit_m,
        "path": path,
    }
    policy = args.get("depth_policy", "error")
    if policy not in ("error", "prune"):
        raise _err("bad_arg", "depth_policy must be error or prune", path + ".args.depth_policy",
                   "use 'prune' to stop silently at max_depth (Structure Synth maxdepth)")
    out["depth_policy"] = policy
    out["symmetry"], out["symmetry_labels"] = _symmetry_group(args.get("symmetry"),
                                                              path + ".args.symmetry", max_instances)
    out["ignored_keys"] = sorted(ignored)
    return out


def _choose(alts, rng):
    if len(alts) == 1:
        return alts[0]
    total = sum(a.weight for a in alts)
    x = rng.random() * total
    acc = 0.0
    for a in alts:
        acc += a.weight
        if x < acc:
            return a
    return [a for a in alts if a.weight > 0][-1]


def expand_shape_grammar(g):
    """Expand a compiled grammar to ``[(primitive, 4x4 matrix in grammar units)]``."""
    rng = random.Random(g["seed"])
    path = g["path"]
    instances, seen = [], set()
    stats = {"expansions": 0, "pruned_scale": 0, "pruned_depth": 0, "depth_reached": 0,
             "duplicates": 0, "rule_counts": {}}
    def check_matrix(w, where):
        if not np.isfinite(w).all() or float(np.max(np.abs(w[:3, 3]))) > 1e6 \
                or float(np.max(np.abs(w[:3, 3]))) * g["unit_m"] > 1e6:
            raise _err("grammar_overflow", "transform coordinates diverged", where,
                       "keep translations bounded and recursive scales <= 1")
        s_min, s_max = _scale_bounds(w)
        if not (math.isfinite(s_min) and math.isfinite(s_max)) or s_max > 1e6 \
                or s_max * g["unit_m"] > 1e6:
            raise _err("grammar_overflow", "transform scale diverged", where,
                       "keep recursive scale <= 1")
        return s_min

    root_m = g["start_transform"].matrix(rng)
    check_matrix(root_m, path + ".args.start_transform")
    stack = [(g["startshape"], root_m, 0, ())]
    group = g["symmetry"]

    def emit(name, m):
        for s in group:
            w = s @ m
            check_matrix(w, path + ".args.symmetry")
            key = (name, tuple(np.round(w, 9).ravel()))
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)
            instances.append((name, w))
            if len(instances) > g["max_instances"]:
                raise _err("grammar_instance_budget", "grammar emitted more than max_instances=%d "
                           "primitives (after symmetry)" % g["max_instances"], path + ".args.max_instances",
                           "add a terminating rule, raise min_scale, lower loop counts, or raise "
                           "max_instances", limit=g["max_instances"])

    while stack:
        name, m, depth, md = stack.pop()
        stats["expansions"] += 1
        if stats["expansions"] > g["max_expansions"]:
            raise _err("grammar_expansion_budget", "more than max_expansions=%d rule expansions"
                       % g["max_expansions"], path + ".args.max_expansions",
                       "the grammar does not terminate fast enough: add a weighted empty rule, a "
                       "scale decay with min_scale, or max_depth with depth_policy='prune'")
        if depth > g["max_depth"]:
            if g["depth_policy"] == "prune":
                stats["pruned_depth"] += 1
                continue
            raise _err("grammar_depth_budget", "rule %r exceeded max_depth=%d" % (name, g["max_depth"]),
                       path + ".args.max_depth",
                       "set depth_policy='prune' to stop silently, add a terminating rule, or "
                       "raise max_depth", rule=name)
        stats["depth_reached"] = max(stats["depth_reached"], depth)
        stats["rule_counts"][name] = stats["rule_counts"].get(name, 0) + 1
        alt = _choose(g["rules"][name], rng)
        if alt.max_depth is not None:
            counts = dict(md)
            used = counts.get(name, 0)
            if used >= alt.max_depth:
                if alt.then is not None:
                    stack.append((alt.then, m, depth + 1, tuple(sorted((k, v) for k, v in counts.items()
                                                                    if k != name))))
                continue
            counts[name] = used + 1
            md = tuple(sorted(counts.items()))
        children = []
        for item in alt.body:
            lm = np.eye(4)
            for _ in range(item.loop_count):
                with np.errstate(over="ignore", invalid="ignore"):
                    w = m @ lm @ item.transform.matrix(rng)
                    if item.loop_transform is not None:
                        lm = lm @ item.loop_transform.matrix(rng)
                s_min = check_matrix(w, path + ".args.rules." + name)
                if s_min < g["min_scale"]:
                    stats["pruned_scale"] += 1
                    continue
                if item.shape is not None:
                    emit(item.shape, w)
                else:
                    children.append((item.call, w, depth + 1, md))
        stack.extend(reversed(children))
    return instances, stats


def _shape_grammar_op(**g):
    instances, stats = expand_shape_grammar(g)
    names = sorted({n for n, _ in instances})
    protos = {n: _proto(n, g["segments"], g["tile_thickness"]) for n in names}
    meta = {"instances": len(instances), "seed": g["seed"], "symmetry": g["symmetry_labels"],
            "symmetry_order": len(g["symmetry"]), "ignored_keys": g["ignored_keys"],
            **{k: v for k, v in stats.items()}}
    return _union(protos, instances, g["unit_m"], g["max_triangles"], g["max_components"],
                  g["path"], "shape_grammar", meta)


def _unit_scale(units):
    scale = _UNITS.get(str(units).lower())
    if scale is None:
        raise _err("bad_arg", "units must be mm, cm, m or in", "units", "use units='mm'")
    return scale


def build_shape_grammar(args, units="mm"):
    """Direct (non-graph) entry point: ``args`` as in the feature op, returns metres."""
    return _shape_grammar_op(**compile_shape_grammar(dict(args), _unit_scale(units), "shape_grammar"))


_shape_grammar_op.prepare = compile_shape_grammar

register_op("shape_grammar", {
    "required": ("startshape", "rules"), "arity": 0,
    "optional": ("symmetry", "seed", "unit", "segments", "tile_thickness", "start_transform",
                 "min_scale", "max_depth", "depth_policy", "max_instances", "max_expansions",
                 "max_triangles", "max_components"),
    "args": {"startshape": "enum", "rules": "anything", "symmetry": "anything",
             "seed": ("int", 0), "unit": "length", "segments": ("int", 3),
             "tile_thickness": "number", "start_transform": "anything", "min_scale": "number",
             "max_depth": ("int", 0), "depth_policy": "enum:error|prune",
             "max_instances": ("int", 1), "max_expansions": ("int", 1),
             "max_triangles": ("int", 1), "max_components": ("int", 1)},
    "doc": "Declarative CFDG / Structure Synth shape grammar (JSON, no eval). "
           "rules: {NAME: [{weight, body: [{shape: box|sphere|cylinder|cone|torus|square|circle|"
           "triangle | call: NAME, transform: {x,y,z,r,rx,ry,rz,s,flip,fx,fy,fz} or ordered list, "
           "loop: {count, transform}}], max_depth?, then?}]}; values may be {rand:[lo,hi]}. "
           "Unit primitives are 1 grammar unit across; unit = mm per grammar unit. "
           "symmetry: 'CF::Dihedral, n' | {type: cyclic|dihedral|reflect, n, axis, center, "
           "mirror_angle}. Weighted choice seeded by seed. min_scale prunes; max_depth/"
           "max_instances/max_expansions/max_triangles fail closed. Output is one manifold union.",
}, _shape_grammar_op)


# ================================================================== lsystem

_TURTLE = set("+-&^\\/|[]!\"$;")


def _parse_modules(text, path, allow_params=True):
    """``"F(1,2)[+A]"`` -> [("F", ["1","2"]), ("[", []), ("+", []), ("A", []), ("]", [])]."""
    if not isinstance(text, str):
        raise _err("bad_arg", "expected a module string", path, "pass e.g. \"F[+F]F\"")
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "(),":
            raise _err("bad_arg", "unexpected %r at %d" % (ch, i), path,
                       "parameters follow a symbol: F(10)")
        args = []
        i += 1
        if i < n and text[i] == "(":
            if not allow_params:
                raise _err("bad_arg", "parameters not allowed here", path, "remove the (...)")
            depth, start, j = 1, i + 1, i + 1
            parts = []
            while j < n and depth:
                c = text[j]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        parts.append(text[start:j])
                elif c == "," and depth == 1:
                    parts.append(text[start:j])
                    start = j + 1
                j += 1
            if depth:
                raise _err("bad_arg", "unbalanced parentheses", path, "close every (")
            args = [p.strip() for p in parts]
            if any(not a for a in args):
                raise _err("bad_arg", "empty parameter", path, "write F(10), not F() or F(1,)")
            i = j
        out.append((ch, args))
    return out


def _check_expr_ast(expr, path):
    s = str(expr)
    if len(s) > 512:
        raise _err("bad_arg", "expression exceeds 512 characters", path, "simplify the expression")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as exc:
        raise _err("bad_arg", "expression %r is not valid arithmetic: %s" % (expr, exc), path,
                   "use + - * / ** % and sqrt/sin/cos/tan/min/max/abs/round") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 128:
        raise _err("bad_arg", "expression has more than 128 AST nodes", path, "simplify the expression")
    for node in nodes:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if any(isinstance(sub, ast.Call) and getattr(sub.func, "id", None) == "round"
                   for sub in ast.walk(node)):
                raise _err("bad_arg", "exponentiation with round() is not allowed", path,
                           "use float operands in **")


def _eval_num(expr, scope, path):
    try:
        return _number(float(expr), path)
    except ValueError:
        pass
    _check_expr_ast(expr, path)
    try:
        v = _eval_expr(expr, scope, path)
    except AttaError:
        raise
    except (TypeError, RecursionError, MemoryError, OverflowError) as exc:
        raise _err("bad_arg", "expression %r failed: %s" % (expr, exc), path,
                   "use numeric arithmetic over the bound parameters and constants") from exc
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise _err("bad_arg", "expression %r is not a finite number" % expr, path,
                   "use numeric arithmetic only")
    return float(v)


def _eval_cond(expr, scope, path):
    _check_expr_ast(expr, path)
    try:
        v = _eval_expr(expr, scope, path, allow_bool=True)
    except AttaError:
        raise
    except (TypeError, RecursionError, MemoryError, OverflowError) as exc:
        raise _err("bad_arg", "condition %r failed: %s" % (expr, exc), path,
                   "compare numbers, e.g. \"l > 1 and w < 3\"") from exc
    return bool(v)


class _Prod:
    __slots__ = ("symbol", "params", "cond", "succ", "weight", "path")


def _compile_productions(rules, path):
    """Accepts {pred: succ | [succ...] | [{succ, weight}]} or [{pred, succ, weight?, cond?}]."""
    rows = []
    if isinstance(rules, dict):
        for pred, value in rules.items():
            if isinstance(value, str):
                rows.append(({"pred": pred, "succ": value}, path + "." + str(pred)))
            elif isinstance(value, (list, tuple)) and value:
                for i, v in enumerate(value):
                    where = path + ".%s[%d]" % (pred, i)
                    if isinstance(v, str):
                        rows.append(({"pred": pred, "succ": v}, where))
                    elif isinstance(v, dict):
                        rows.append(({"pred": pred, **v}, where))
                    else:
                        raise _err("bad_arg", "successor must be a string or {succ, weight, cond}",
                                   where, "use \"F[+F]F\"")
            else:
                raise _err("bad_arg", "rule value must be a successor string or a list", path + "." + str(pred),
                           "use {\"F\": \"F[+F]F\"}")
    elif isinstance(rules, (list, tuple)) and rules:
        rows = [(r, path + "[%d]" % i) for i, r in enumerate(rules)]
    else:
        raise _err("bad_arg", "rules must be a non-empty object or list", path,
                   "use {\"F\": \"F[+F]F[-F]F\"}")
    prods = {}
    for row, where in rows:
        if not isinstance(row, dict) or "pred" not in row or "succ" not in row:
            raise _err("bad_arg", "production needs pred and succ", where,
                       "use {\"pred\": \"A(l)\", \"succ\": \"F(l)[+A(l*0.7)]\", \"cond\": \"l > 1\"}")
        extra = set(row) - {"pred", "succ", "weight", "cond"}
        if extra:
            raise _err("bad_arg", "unexpected production key(s) %s" % sorted(extra), where,
                       "allowed: pred, succ, weight, cond")
        pred = _parse_modules(str(row["pred"]), where + ".pred")
        if len(pred) != 1:
            raise _err("bad_arg", "predecessor must be one module (context-free L-system)", where + ".pred",
                       "use a single symbol such as A or A(l,w)")
        p = _Prod()
        p.symbol, params = pred[0]
        for name in params:
            if not name.isidentifier():
                raise _err("bad_arg", "predecessor parameter %r is not a name" % name, where + ".pred",
                           "use formal names: A(l,w)")
        p.params = params or None
        p.cond = row.get("cond")
        if p.cond is not None and not isinstance(p.cond, str):
            raise _err("bad_arg", "cond must be an expression string", where + ".cond", "use \"l > 1\"")
        p.succ = _parse_modules(row["succ"], where + ".succ")
        p.weight = _number(row.get("weight", 1.0), where + ".weight", nonneg=True)
        p.path = where
        prods.setdefault(p.symbol, []).append(p)
    return prods


def compile_lsystem(args, scale=1e-3, path="lsystem"):
    if not isinstance(args, dict):
        raise _err("bad_arg", "args must be an object", path + ".args", "pass a dict")
    a = path + ".args."
    consts = args.get("constants") or {}
    if not isinstance(consts, dict):
        raise _err("bad_arg", "constants must be an object name -> number", a + "constants",
                   "use {\"r\": 0.7}")
    constants = {}
    for k, v in consts.items():
        if not isinstance(k, str) or not k.isidentifier():
            raise _err("bad_arg", "constant name %r is not an identifier" % (k,), a + "constants", "rename it")
        constants[k] = _number(v, a + "constants." + k)
    factor = _number(scale, path + ".units", positive=True)
    out = {
        "axiom": _parse_modules(args.get("axiom"), a + "axiom"),
        "productions": _compile_productions(args.get("rules"), a + "rules"),
        "iterations": _integer(args.get("iterations"), a + "iterations", 0, CEILING["iterations"]),
        "constants": constants,
        "angle": _number(args.get("angle", 25.0), a + "angle"),
        "step": _number(args.get("step", 10.0), a + "step", positive=True),
        "width": _number(args.get("width", 1.0), a + "width", positive=True),
        "step_decay": _number(args.get("step_decay", 0.9), a + "step_decay", positive=True),
        "width_decay": _number(args.get("width_decay", 0.707), a + "width_decay", positive=True),
        "angle_decay": _number(args.get("angle_decay", 1.0), a + "angle_decay", positive=True),
        "min_width": _number(args.get("min_width", 1e-3), a + "min_width", nonneg=True),
        "seed": _seed(args, path),
        "segments": _segments(args, path, 8),
        "joints": args.get("joints", True),
        "draw": args.get("draw_symbols", "F"), "move": args.get("move_symbols", "f"),
        "max_symbols": _budget(args, "max_symbols", path),
        "max_segments": _budget(args, "max_segments", path),
        "max_depth": _budget(args, "max_depth", path, 0),
        "max_triangles": _budget(args, "max_triangles", path),
        "max_components": _max_components(args, path),
        "scale": factor, "path": path,
    }
    if not out["axiom"]:
        raise _err("bad_arg", "axiom is empty", a + "axiom", "pass e.g. \"F\"")
    if not isinstance(out["joints"], bool):
        raise _err("bad_arg", "joints must be true/false", a + "joints", "pass true")
    for key in ("draw", "move"):
        v = out[key]
        if not isinstance(v, str) or not v or any(c in _TURTLE or c.isspace() or c in "()," for c in v):
            raise _err("bad_arg", "%s_symbols must be a string of symbol characters" % key,
                       a + key + "_symbols", "use e.g. \"FG\" (turtle commands are reserved)")
    if set(out["draw"]) & set(out["move"]):
        raise _err("bad_arg", "a symbol cannot both draw and move", a + "move_symbols",
                   "make draw_symbols and move_symbols disjoint")
    trop = args.get("tropism")
    out["tropism"] = None
    if trop is not None:
        if not isinstance(trop, dict) or set(trop) - {"vector", "e"} or "vector" not in trop:
            raise _err("bad_arg", "tropism must be {vector: [x,y,z], e}", a + "tropism",
                       "use {\"vector\": [0, 0, -1], \"e\": 0.2}")
        vec = trop["vector"]
        if not isinstance(vec, (list, tuple)) or len(vec) != 3:
            raise _err("bad_arg", "tropism.vector must be 3 numbers", a + "tropism.vector", "use [0, 0, -1]")
        v = np.array([_number(c, a + "tropism.vector[%d]" % i) for i, c in enumerate(vec)])
        vnorm = math.hypot(float(v[0]), float(v[1]), float(v[2]))
        if vnorm == 0 or not math.isfinite(vnorm) or vnorm > 1e6:
            raise _err("bad_arg", "tropism.vector must be non-zero with norm <= 1e6",
                       a + "tropism.vector", "use a unit vector such as [0, 0, -1]")
        e = _number(trop.get("e", 0.2), a + "tropism.e")
        if not math.isfinite(e * vnorm) or abs(e * vnorm) > 1e6:
            raise _err("bad_arg", "tropism.e * |vector| is out of bounds",
                       a + "tropism.e", "use a small susceptibility e such as 0.2")
        out["tropism"] = (v, e)
    leaf = args.get("leaf")
    out["leaf"] = None
    if leaf is not None:
        if not isinstance(leaf, dict) or set(leaf) - {"symbol", "shape", "size"}:
            raise _err("bad_arg", "leaf must be {symbol, shape, size}", a + "leaf",
                       "use {\"symbol\": \"L\", \"shape\": \"sphere\", \"size\": [4, 1, 6]}")
        sym = leaf.get("symbol", "L")
        if not isinstance(sym, str) or len(sym) != 1 or sym in _TURTLE or sym in out["draw"] + out["move"]:
            raise _err("bad_arg", "leaf.symbol must be one free character", a + "leaf.symbol", "use \"L\"")
        shape = leaf.get("shape", "sphere")
        if shape not in PRIMITIVES:
            raise _err("bad_arg", "unknown leaf shape %r" % (shape,), a + "leaf.shape",
                       "use one of: " + ", ".join(PRIMITIVES))
        size = leaf.get("size", [out["width"] * 3] * 3)
        if isinstance(size, (int, float)) and not isinstance(size, bool):
            size = [size] * 3
        if not isinstance(size, (list, tuple)) or len(size) != 3:
            raise _err("bad_arg", "leaf.size must be [along L, along U, along H] mm", a + "leaf.size",
                       "use [4, 1, 6]")
        out["leaf"] = (sym, shape, [_number(c, a + "leaf.size[%d]" % i, positive=True)
                                    for i, c in enumerate(size)])
    return out


def expand_lsystem(g):
    """Rewrite the axiom ``iterations`` times -> list of (symbol, [float args])."""
    rng = random.Random(g["seed"])
    path, consts = g["path"], g["constants"]
    current = [(s, [_eval_num(e, consts, path + ".args.axiom") for e in args]) for s, args in g["axiom"]]
    prods = g["productions"]
    for it in range(g["iterations"]):
        nxt = []
        for sym, vals in current:
            cands = []
            for p in prods.get(sym, ()):
                if p.params is not None and len(p.params) != len(vals):
                    continue
                scope = dict(consts)
                if p.params is not None:
                    scope.update(zip(p.params, vals))
                if p.cond is not None and not _eval_cond(p.cond, scope, p.path + ".cond"):
                    continue
                if p.weight > 0:
                    cands.append((p, scope))
            if not cands:
                nxt.append((sym, vals))
            else:
                if len(cands) == 1:
                    p, scope = cands[0]
                else:
                    x = rng.random() * sum(c[0].weight for c in cands)
                    acc = 0.0
                    for p, scope in cands:
                        acc += p.weight
                        if x < acc:
                            break
                for s, exprs in p.succ:
                    nxt.append((s, [_eval_num(e, scope, p.path + ".succ") for e in exprs]))
            if len(nxt) > g["max_symbols"]:
                raise _err("lsystem_symbol_budget", "string exceeds max_symbols=%d at iteration %d"
                           % (g["max_symbols"], it + 1), path + ".args.max_symbols",
                           "lower iterations, use stochastic/conditional termination, or raise "
                           "max_symbols", iteration=it + 1)
        current = nxt
    return current


def _rotate(v, axis, deg):
    a = math.radians(math.fmod(deg, 360.0))
    return v * math.cos(a) + np.cross(axis, v) * math.sin(a) + axis * float(np.dot(axis, v)) * (1 - math.cos(a))


def interpret_lsystem(g, modules):
    """3D turtle (ABOP 1.5): returns (segments, leaves, stats) in model units.

    Frame: heading H=+z, left L=+x, up U=H x L=+y.  ``+``/``-`` yaw about U,
    ``&``/``^`` pitch about L, ``\\``/``/`` roll about H, ``|`` turns 180.
    """
    path = g["path"]
    pos = np.zeros(3)
    H, L, U = np.array([0.0, 0, 1]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    width, step, angle = g["width"], g["step"], g["angle"]
    stack, segs, leaves = [], [], []
    stats = {"pruned_thin": 0, "max_nesting": 0}
    floor_w = max(g["min_width"], 1e-9 / g["scale"])
    floor_len = 1e-9 / g["scale"]
    bound = 1e6 / g["scale"]
    draw, move, leaf = set(g["draw"]), set(g["move"]), g["leaf"]
    for sym, vals in modules:
        arg = vals[0] if vals else None
        if sym in draw or sym in move:
            length = step if arg is None else arg
            if not math.isfinite(length) or abs(length) > bound:
                raise _err("grammar_overflow", "turtle step length diverged", path + ".args.step",
                           "keep step and step_decay bounded")
            nxt = pos + H * length
            if not np.isfinite(nxt).all() or float(np.max(np.abs(nxt))) > bound:
                raise _err("grammar_overflow", "turtle position diverged", path + ".args.step",
                           "keep step and iterations bounded")
            if sym in draw and abs(length) >= floor_len:
                if width <= 0:
                    raise _err("lsystem_width", "segment width %.6g <= 0" % width, path + ".args.width",
                               "keep !(w) and width_decay positive")
                if not math.isfinite(width) or width > bound:
                    raise _err("grammar_overflow", "segment width diverged", path + ".args.width",
                               "keep width and width_decay bounded")
                if width < floor_w:
                    stats["pruned_thin"] += 1
                else:
                    segs.append((pos.copy(), nxt.copy(), width))
                    if len(segs) > g["max_segments"]:
                        raise _err("lsystem_segment_budget", "more than max_segments=%d tube segments"
                                   % g["max_segments"], path + ".args.max_segments",
                                   "lower iterations or raise max_segments")
            pos = nxt
            if g["tropism"] is not None and sym in draw:
                tv, e = g["tropism"]
                hxt = np.cross(H, tv)
                nrm = float(np.linalg.norm(hxt))
                if nrm > 1e-12:
                    axis = hxt / nrm
                    deg = math.degrees(e * nrm)
                    H, L, U = (_rotate(H, axis, deg), _rotate(L, axis, deg), _rotate(U, axis, deg))
        elif sym in "+-":
            d = (angle if arg is None else arg) * (1 if sym == "+" else -1)
            if not math.isfinite(d):
                raise _err("grammar_overflow", "turtle angle diverged", path + ".args.angle",
                           "keep angle and angle_decay bounded")
            H, L = _rotate(H, U, d), _rotate(L, U, d)
        elif sym in "&^":
            d = (angle if arg is None else arg) * (1 if sym == "&" else -1)
            if not math.isfinite(d):
                raise _err("grammar_overflow", "turtle angle diverged", path + ".args.angle",
                           "keep angle and angle_decay bounded")
            H, U = _rotate(H, L, d), _rotate(U, L, d)
        elif sym in "\\/":
            d = (angle if arg is None else arg) * (1 if sym == "\\" else -1)
            if not math.isfinite(d):
                raise _err("grammar_overflow", "turtle angle diverged", path + ".args.angle",
                           "keep angle and angle_decay bounded")
            L, U = _rotate(L, H, d), _rotate(U, H, d)
        elif sym == "|":
            H, L = -H, -L
        elif sym == "$":
            lv = np.cross(np.array([0.0, 0, 1]), H)
            n = float(np.linalg.norm(lv))
            if n > 1e-9:
                L = lv / n
                U = np.cross(H, L)
        elif sym == "!":
            width = arg if arg is not None else width * g["width_decay"]
            if width <= 0:
                raise _err("lsystem_width", "segment width %.6g <= 0" % width, path + ".args.width",
                           "keep !(w) and width_decay positive")
            if not math.isfinite(width) or width > bound:
                raise _err("grammar_overflow", "segment width diverged", path + ".args.width",
                           "keep width and width_decay bounded")
        elif sym == '"':
            if arg is not None and arg <= 0:
                raise _err("bad_arg", "step multiplier \"(k) must be > 0", path + ".args.rules",
                           "pass a positive factor in \"(k)")
            step *= arg if arg is not None else g["step_decay"]
            if not math.isfinite(step) or step > bound:
                raise _err("grammar_overflow", "turtle step diverged", path + ".args.step",
                           "keep step_decay bounded")
        elif sym == ";":
            angle *= arg if arg is not None else g["angle_decay"]
            if not math.isfinite(angle):
                raise _err("grammar_overflow", "turtle angle diverged", path + ".args.angle",
                           "keep angle_decay bounded")
        elif sym == "[":
            stack.append((pos.copy(), H, L, U, width, step, angle))
            stats["max_nesting"] = max(stats["max_nesting"], len(stack))
            if len(stack) > g["max_depth"]:
                raise _err("lsystem_depth_budget", "bracket nesting exceeds max_depth=%d" % g["max_depth"],
                           path + ".args.max_depth", "lower iterations or raise max_depth")
        elif sym == "]":
            if not stack:
                raise _err("lsystem_unbalanced", "']' without matching '['", path + ".args.rules",
                           "balance the brackets in the axiom and successors")
            pos, H, L, U, width, step, angle = stack.pop()
        elif leaf is not None and sym == leaf[0]:
            k = 1.0 if arg is None else arg
            if not math.isfinite(k) or k <= 0:
                raise _err("bad_arg", "leaf scale parameter must be > 0", path + ".args.leaf",
                           "pass a positive factor in L(k)")
            scaled = [c * k for c in leaf[2]]
            if any(not math.isfinite(v) or v > bound for v in scaled):
                raise _err("grammar_overflow", "leaf size diverged", path + ".args.leaf",
                           "keep leaf scale bounded")
            if min(scaled) >= floor_len:
                leaves.append((pos.copy(), H.copy(), L.copy(), U.copy(), scaled))
        # Re-orthonormalise so long strings do not drift.
        H = H / np.linalg.norm(H)
        L = L - H * float(np.dot(L, H))
        L = L / np.linalg.norm(L)
        U = np.cross(H, L)
    return segs, leaves, stats


def _frame_matrix(origin, x, y, z):
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, origin
    return m


def _lsystem_op(**g):
    modules = expand_lsystem(g)
    segs, leaves, stats = interpret_lsystem(g, modules)
    instances = []
    joints = {}
    for p0, p1, w in segs:
        d = p1 - p0
        length = float(np.linalg.norm(d))
        h = d / length
        ref = np.array([1.0, 0, 0]) if abs(h[0]) < 0.9 else np.array([0.0, 1, 0])
        x = np.cross(ref, h)
        x /= np.linalg.norm(x)
        y = np.cross(h, x)
        mid = (p0 + p1) / 2.0
        instances.append(("cylinder", _frame_matrix(mid, x * w, y * w, h * length)))
        if g["joints"]:
            for p in (p0, p1):
                key = tuple(np.round(p, 9))
                joints[key] = (p, max(joints[key][1], w) if key in joints else w)
    for p, w in joints.values():
        instances.append(("sphere", _frame_matrix(p, *(np.eye(3) * w))))
    leaf_shape = g["leaf"][1] if g["leaf"] else None
    for pos, H, L, U, size in leaves:
        centre = pos + H * size[2] / 2.0
        instances.append((leaf_shape, _frame_matrix(centre, L * size[0], U * size[1], H * size[2])))
    names = sorted({n for n, _ in instances})
    protos = {n: _proto(n, g["segments"], 0.1) for n in names}
    meta = {"symbols": len(modules), "segments": len(segs), "joints": len(joints),
            "leaves": len(leaves), "iterations": g["iterations"], "seed": g["seed"], **stats}
    return _union(protos, instances, g["scale"], g["max_triangles"], g["max_components"],
                  g["path"], "lsystem", meta)


def build_lsystem(args, units="mm"):
    return _lsystem_op(**compile_lsystem(dict(args), _unit_scale(units), "lsystem"))


_lsystem_op.prepare = compile_lsystem

register_op("lsystem", {
    "required": ("axiom", "rules", "iterations"), "arity": 0,
    "optional": ("angle", "step", "width", "step_decay", "width_decay", "angle_decay", "constants",
                 "seed", "tropism", "leaf", "segments", "joints", "draw_symbols", "move_symbols",
                 "min_width", "max_symbols", "max_segments", "max_depth", "max_triangles",
                 "max_components"),
    "args": {"axiom": "enum", "rules": "anything", "iterations": ("int", 0), "angle": "number",
             "step": "length", "width": "length", "step_decay": "number", "width_decay": "number",
             "angle_decay": "number", "constants": "anything", "seed": ("int", 0),
             "tropism": "anything", "leaf": "anything", "segments": ("int", 3), "joints": "bool",
             "draw_symbols": "enum", "move_symbols": "enum", "min_width": "length",
             "max_symbols": ("int", 1), "max_segments": ("int", 1), "max_depth": ("int", 0),
             "max_triangles": ("int", 1), "max_components": ("int", 1)},
    "doc": "ABOP L-system with a 3D turtle (H=+z). rules: {pred: succ | [succ...] | [{succ, weight, "
           "cond}]} or [{pred: 'A(l,w)', succ: 'F(l)[&A(l*r,w)]', cond: 'l > 1', weight}] — "
           "stochastic and parametric (safe arithmetic, constants). Turtle: F draw, f move, + - yaw, "
           "& ^ pitch, \\ / roll, | turn, $ level, [ ] branch, !(w) width (bare: *width_decay), "
           "\"(k) step*k, ;(k) angle*k. step/width (tube diameter) in mm. Output: unioned tube "
           "segments with sphere joints + optional leaf primitive; budgets fail closed.",
}, _lsystem_op)
