# -*- coding: utf-8 -*-
"""Bounded, composable implicit-field trees. Kernel coordinates are metres.

This is a scalar-field modeller, not an exact distance oracle. CSG, TPMS,
noise and warps do not in general preserve Euclidean distance. Tree order is
literal: a noise child of a transform moves with that transform. Sampling
uses fixed world coordinates in bounded chunks; no Python expression eval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import sdf, sdfkit
from .errors import AttaError

MAX_NODES = 128
MAX_DEPTH = 24
MAX_SAMPLES = 2_500_000
MAX_EVALUATIONS = 200_000_000
MAX_TRIANGLES = 2_000_000


def _bad(message, path, fix="use the documented field_tree schema"):
    raise AttaError("bad_arg", message, path=path, fix=fix)


def number(value, path, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_, str)):
        _bad("expected a finite number", path)
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        _bad("expected a finite number", path)
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        _bad("number is non-finite or outside its allowed range", path)
    return value


def integer(value, path, lo, hi):
    value = number(value, path)
    if value != int(value) or not lo <= value <= hi:
        _bad("expected an integer in [%d, %d]" % (lo, hi), path)
    return int(value)


def vector(value, path, *, positive=False):
    if (not isinstance(value, (list, tuple, np.ndarray))
            or (isinstance(value, np.ndarray) and value.ndim != 1)
            or len(value) != 3):
        _bad("expected three finite coordinates", path)
    return np.array([number(v, path, positive=positive) for v in value])


def _norm(points):
    return np.hypot.reduce(points, axis=-1)


def _checked(evaluate, path):
    def checked(points):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                result = evaluate(points)
        except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
            raise AttaError("field_invalid", "field arithmetic is not representable",
                            path=path, fix="use a local coordinate frame and representable feature scales") from exc
        if result.shape != (len(points),) or not np.isfinite(result).all():
            raise AttaError("field_invalid", "field returned non-finite or incorrectly shaped values",
                            path=path, fix="reduce the parameter range or simplify the tree")
        return result
    return checked


def _rotation(deg):
    x, y, z = np.deg2rad(deg)
    sx, sy, sz = np.sin([x, y, z])
    cx, cy, cz = np.cos([x, y, z])
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @
            np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @
            np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


@dataclass(frozen=True)
class Field:
    evaluate: object
    nodes: int
    cost: int
    depth: int

    def __call__(self, points):
        try:
            points = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AttaError("bad_arg", "points must be finite (N,3) coordinates",
                            path="points", fix="provide numeric metre-based coordinates") from exc
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            _bad("points must be finite (N,3) coordinates", "points")
        return self.evaluate(points)


def compile_tree(tree, *, scale=1.0, path="tree"):
    """Validate/compile a JSON tree, scaling all length values exactly once.

    ``scale=1`` for the metre-based Python API; the feature boundary supplies
    model-unit-to-metre scale. No allocation proportional to a sample grid.
    """
    scale = number(scale, "scale", positive=True)
    seen_nodes = 0
    max_depth = 0

    def visit(node, where, depth):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                evaluate, cost = compile_node(node, where, depth)
        except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
            raise AttaError("bad_arg", "field parameters are not representable",
                            path=where, fix="use representable lengths in a local coordinate frame") from exc
        return _checked(evaluate, where), cost

    def compile_node(node, where, depth):
        nonlocal seen_nodes, max_depth
        seen_nodes += 1
        max_depth = max(depth, max_depth)
        if seen_nodes > MAX_NODES or depth > MAX_DEPTH:
            raise AttaError("field_budget", "field tree exceeds node/depth budget",
                            path=where, fix="split the model into smaller field trees")
        if (not isinstance(node, dict) or not isinstance(node.get("op"), str)
                or any(not isinstance(key, str) for key in node)):
            _bad("field node needs an op string and string argument names", where)
        op = node["op"]
        binary = {"union", "intersection", "difference", "smooth_union",
                  "smooth_intersection", "smooth_difference"}
        allowed = {
            "sphere": {"radius", "center"}, "box": {"size", "center", "round"},
            "cylinder": {"radius", "height", "center", "axis"},
            "torus": {"major_radius", "minor_radius", "center", "axis"},
            "capsule": {"a", "b", "radius"},
            "gyroid": {"period", "band", "phase_deg"},
            "schwarz_p": {"period", "band", "phase_deg"},
            "transform": {"child", "translate", "rotate_deg", "scale"},
            "shell": {"child", "thickness"}, "offset": {"child", "distance"},
            "noise": {"child", "amplitude", "wavelength", "seed", "octaves"},
            "twist": {"child", "angle_deg", "length", "axis"},
        }
        if op in binary:
            keys = {"children", "radius"} if op.startswith("smooth_") else {"children"}
        else:
            keys = allowed.get(op)
        if keys is None:
            _bad("unknown field op %r" % op, where + ".op")
        extra = set(node) - keys - {"op"}
        if extra:
            _bad("unknown field arguments: " + ", ".join(sorted(extra)), where)

        def num(key, default=None, **kwargs):
            return number(node.get(key, default), where + "." + key, **kwargs)

        def length(key, default=None, **kwargs):
            source = num(key, default, **kwargs)
            value = source * scale
            if not math.isfinite(value):
                _bad("length overflows after unit scaling", where + "." + key,
                     "reduce the declared length or use a smaller model unit")
            if source != 0 and value == 0:
                _bad("length underflows to zero after unit scaling", where + "." + key,
                     "build the feature in local units or increase the scaling factor")
            return value

        def vec(key, default=None, *, scaled=True, positive=False):
            value = vector(node.get(key, default), where + "." + key,
                           positive=positive) * (scale if scaled else 1.0)
            if not np.isfinite(value).all():
                _bad("vector overflows after unit scaling", where + "." + key,
                     "reduce the declared coordinates or use a smaller model unit")
            if positive and np.any(value <= 0):
                _bad("vector underflows to zero after unit scaling", where + "." + key,
                     "build the feature in local units or lower the scaling factor")
            return value

        def axis():
            result = node.get("axis", "z")
            if not isinstance(result, str) or result not in ("x", "y", "z"):
                _bad("axis must be x, y or z", where + ".axis")
            return result

        if op in binary:
            children = node.get("children")
            if not isinstance(children, list) or not 2 <= len(children) <= MAX_NODES:
                _bad("children must contain 2..128 fields", where + ".children")
            compiled = [visit(c, where + ".children[%d]" % i, depth + 1)
                        for i, c in enumerate(children)]
            k = length("radius", positive=True) if op.startswith("smooth_") else 0
            methods = {"union": sdfkit.union, "intersection": sdfkit.intersect,
                       "difference": sdfkit.subtract, "smooth_union": sdfkit.smooth_union,
                       "smooth_intersection": sdfkit.smooth_intersect,
                       "smooth_difference": sdfkit.smooth_subtract}
            fn = methods[op]
            def combined(p):
                result = compiled[0][0](p)
                for child, _ in compiled[1:]:
                    other = child(p)
                    result = fn(result, other, k) if k else fn(result, other)
                return result
            return combined, 1 + sum(c[1] for c in compiled)

        if "child" in keys:
            child, cost = visit(node.get("child"), where + ".child", depth + 1)
            if op == "transform":
                shift = vec("translate", [0, 0, 0])
                rotation = _rotation(vec("rotate_deg", [0, 0, 0], scaled=False))
                if not np.isfinite(rotation).all():
                    _bad("rotation is not representable", where + ".rotate_deg",
                         "use a finite angle in degrees")
                factor = num("scale", 1, positive=True)
                def transformed(p):
                    q = p - shift
                    # Explicit fixed-order reductions avoid BLAS selecting a
                    # different dot-product kernel for a one-point chunk.
                    local = np.column_stack([q[:, 0] * rotation[0, k] +
                                             q[:, 1] * rotation[1, k] +
                                             q[:, 2] * rotation[2, k] for k in range(3)])
                    return factor * child(local / factor)
                return transformed, cost + 1
            if op == "shell":
                thickness = length("thickness", positive=True)
                return lambda p: np.abs(child(p)) - thickness / 2, cost + 1
            if op == "offset":
                distance = length("distance")
                return lambda p: child(p) - distance, cost + 1
            if op == "twist":
                angle_deg = num("angle_deg")
                rate = math.radians(angle_deg) / length("length", positive=True)
                if not math.isfinite(rate) or (angle_deg != 0 and rate == 0):
                    _bad("twist rate is not representable", where,
                         "use a finite angle and a non-degenerate length")
                ax = {"x": 0, "y": 1, "z": 2}[axis()]
                i, j = (ax + 1) % 3, (ax + 2) % 3
                def twisted(p):
                    angle = -rate * p[:, ax]
                    c, s = np.cos(angle), np.sin(angle)
                    q = p.copy()
                    q[:, i] = c * p[:, i] - s * p[:, j]
                    q[:, j] = s * p[:, i] + c * p[:, j]
                    return child(q)
                return twisted, cost + 1
            amplitude = length("amplitude", nonnegative=True)
            wavelength = length("wavelength", positive=True)
            seed = integer(node.get("seed", 0), where + ".seed", 0, 2**31 - 1)
            octaves = integer(node.get("octaves", 3), where + ".octaves", 1, 8)
            def noisy(p):
                q = p / wavelength
                # Noise lattice uses integer hashing; refuse coordinates that
                # would overflow it rather than silently changing the pattern.
                if np.max(np.abs(q), initial=0) * 2**octaves > 2**30:
                    _bad("noise coordinates exceed the lattice range", where,
                         "move the field into a local coordinate frame")
                return child(p) + amplitude * sdfkit.fbm(
                    q[:, 0], q[:, 1], q[:, 2], octaves=octaves, seed=seed)
            return noisy, cost + 8 * octaves

        if op in ("gyroid", "schwarz_p"):
            period = length("period", positive=True)
            band = num("band", .25, positive=True)
            phase = np.deg2rad(vec("phase_deg", [0, 0, 0], scaled=False))
            if not np.isfinite(phase).all():
                _bad("phase is not representable", where + ".phase_deg",
                     "use a finite angle in degrees")
            def periodic(p):
                q = p * (2 * math.pi / period) + phase
                sx, sy, sz = np.sin(q).T
                cx, cy, cz = np.cos(q).T
                value = sx * cy + sy * cz + sz * cx if op == "gyroid" else cx + cy + cz
                # band is a dimensionless level-set interval, NOT wall thickness.
                return (np.abs(value) - band) * period / (2 * math.pi)
            return periodic, 3
        if op == "capsule":
            a, b = vec("a"), vec("b")
            radius = length("radius", positive=True)
            edge = b - a
            edge_len = float(_norm(edge))
            if edge_len == 0:
                return lambda p: _norm(p - a) - radius, 1
            direction = edge / edge_len
            def capsule(p):
                local = p - a
                projection = (local[:, 0] * direction[0] + local[:, 1] * direction[1]
                              + local[:, 2] * direction[2])
                t = np.clip(projection, 0, edge_len)
                return _norm(local - t[:, None] * direction) - radius
            return capsule, 1
        center = vec("center", [0, 0, 0])
        if op == "sphere":
            radius = length("radius", positive=True)
            return lambda p: _norm(p - center) - radius, 1
        if op == "box":
            size = vec("size", positive=True)
            rounding = length("round", 0, nonnegative=True)
            if rounding > min(size) / 2:
                _bad("round must be <= half the shortest box side", where + ".round")
            return lambda p: sdfkit.box(*p.T, size=size, center=center, round_r=rounding), 1
        ax = axis()
        if op == "cylinder":
            radius, height = length("radius", positive=True), length("height", positive=True)
            return lambda p: sdfkit.cylinder(*p.T, radius=radius, height=height,
                                             center=center, axis=ax), 1
        major, minor = length("major_radius", positive=True), length("minor_radius", positive=True)
        if minor >= major:
            _bad("torus needs minor_radius < major_radius", where)
        return lambda p: sdfkit.torus(*p.T, major=major, minor=minor, center=center, axis=ax), 1

    fn, cost = visit(tree, path, 1)
    return Field(fn, seen_nodes, cost, max_depth)


def mesh_field(field, bounds, spacing, *, max_samples=1_200_000,
               max_evaluations=100_000_000, max_triangles=1_000_000,
               chunk_points=32768, budget_policy="error", min_feature=None):
    """Sample one compiled field then extract once, with allocation/work caps.

    Bounds are the requested sampling domain, not a clipping solid. The field
    must be positive on all six padded boundary faces. Coarsening is opt-in;
    declared min_feature must retain >=3 voxels even with coarsening enabled.
    """
    if not isinstance(field, Field):
        _bad("compile the tree before sampling", "field")
    try:
        bounds = np.asarray(bounds, dtype=float)
    except (TypeError, ValueError, OverflowError):
        _bad("bounds must be [min_xyz,max_xyz]", "bounds")
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        _bad("bounds must be finite and strictly ordered", "bounds")
    with np.errstate(over="ignore", invalid="ignore"):
        span = bounds[1] - bounds[0]
    if not np.isfinite(span).all():
        _bad("bounds span overflows floating point", "bounds",
             "use a local coordinate frame or smaller bounds")
    spacing = number(spacing, "spacing", positive=True)
    max_samples = integer(max_samples, "max_samples", 343, MAX_SAMPLES)
    max_evaluations = integer(max_evaluations, "max_evaluations", 343, MAX_EVALUATIONS)
    max_triangles = integer(max_triangles, "max_triangles", 4, MAX_TRIANGLES)
    chunk_points = integer(chunk_points, "chunk_points", 1, 262144)
    if not isinstance(budget_policy, str) or budget_policy not in ("error", "coarsen"):
        _bad("budget_policy must be error or coarsen", "budget_policy")
    effective_budget = min(max_samples, max_evaluations // field.cost)
    requested_spacing = spacing
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            if budget_policy == "coarsen":
                spacing = sdf.fit_spacing(*bounds, spacing, effective_budget)
            shape = sdf.grid_shape(*bounds, spacing)
    except AttaError:
        raise
    except (OverflowError, ValueError, ZeroDivisionError, TypeError) as exc:
        raise AttaError("bad_arg", "sampling grid is not representable: %s" % exc,
                        path="bounds", fix="use a local coordinate frame and a positive spacing") from exc
    samples = math.prod(shape)
    if samples > effective_budget:
        raise AttaError("field_budget", "field exceeds sampling or evaluation budget",
                        path="spacing", fix="increase spacing or explicitly choose budget_policy='coarsen'",
                        details={"samples": samples, "estimated_evaluations": samples * field.cost,
                                 "max_samples": max_samples, "max_evaluations": max_evaluations})
    if min_feature is not None:
        min_feature = number(min_feature, "min_feature", positive=True)
        if min_feature / spacing < 3:
            raise AttaError("field_under_resolved", "declared minimum feature has fewer than three voxels",
                            path="min_feature", fix="decrease spacing and increase the sampling budget",
                            details={"feature_samples": min_feature / spacing})
    origin = bounds[0] - 2.5 * spacing
    # Guard against a spacing smaller than representable increments at origin.
    endpoint = origin + (np.asarray(shape) - 1) * spacing
    if not np.isfinite(origin).all() or not np.isfinite(endpoint).all():
        _bad("sampling grid coordinates overflow floating point", "bounds",
             "build in local coordinates, then position the completed module")
    ulp = np.maximum(np.abs(np.spacing(origin)), np.abs(np.spacing(endpoint)))
    # Tetrahedral interpolation retains features down to 1e-3 voxel. Merely
    # representing one whole voxel does not preserve those surface vertices.
    if np.any(ulp > spacing * 1e-4):
        _bad("surface interpolation is below floating-point resolution at this origin", "bounds",
             "build in local coordinates, then position the completed module")
    phi = np.empty(shape, dtype=np.float64)
    flat = phi.ravel()
    ny, nz = shape[1:]
    for start in range(0, samples, chunk_points):
        ids = np.arange(start, min(samples, start + chunk_points), dtype=np.int64)
        coords = np.column_stack([ids // (ny * nz), (ids // nz) % ny, ids % nz])
        values = np.asarray(field(origin + coords * spacing), dtype=float)
        if values.shape != (len(ids),) or not np.isfinite(values).all():
            raise AttaError("field_invalid", "field returned non-finite or incorrectly shaped values",
                            path="tree", fix="reduce the parameter range or simplify the tree")
        flat[start:start + len(ids)] = values
    boundary_min = min(float(a.min()) for a in (phi[0], phi[-1], phi[:, 0],
                       phi[:, -1], phi[:, :, 0], phi[:, :, -1]))
    if boundary_min <= 0:
        raise AttaError("sdf_open", "field intersects the sampling domain boundary",
                        path="bounds", fix="increase bounds or intersect the tree with a clipping primitive")
    mesh = sdf.marching_tets(phi, origin, spacing, max_triangles=max_triangles)
    if not mesh.topology()["closed"] or not np.isfinite(mesh.verts).all():
        raise AttaError("field_invalid", "extracted field is not a finite closed manifold",
                        path="tree", fix="increase sampling precision or use a local coordinate frame")
    sampling = {"samples": samples, "shape": list(shape), "chunk_points": chunk_points,
                "max_samples": max_samples, "estimated_evaluations": samples * field.cost,
                "max_evaluations": max_evaluations, "requested_spacing_mm": requested_spacing * 1000,
                "spacing_mm": spacing * 1000, "coarsened": spacing > requested_spacing,
                "min_feature_samples": min_feature / spacing if min_feature else None}
    mesh.meta.update({"op": "field_tree", "field": {"nodes": field.nodes, "depth": field.depth,
                     "distance_exact": False, "method": "composable_scalar_field"},
                     "sampling": sampling})
    return mesh
