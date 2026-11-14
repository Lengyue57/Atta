# -*- coding: utf-8 -*-
"""Bounded rational tensor-product B-spline patches. Kernel lengths are metres.

No external NURBS implementation is copied here. Local Cox--de Boor basis
recurrence plus the rational quotient rule supplies positions/first partials.
Tessellation is a conforming tensor grid, NOT a certified error-bound algorithm.
"""
from __future__ import annotations

import math

import numpy as np

from .errors import AttaError
from .mesh import Mesh

__all__ = ["NURBSSurface", "tessellate"]

# Hard ceilings apply even when callers ask for larger budgets. Arrays are
# preflighted before conversion/allocation; evaluator scratch is chunk-local.
MAX_CONTROL_POINTS = 65_536
MAX_AXIS_CONTROLS = 1_024
MAX_DEGREE = 8
MAX_VERTICES = 250_000
MAX_FACES = 500_000
MAX_EVALUATIONS = 2_000_000
MAX_BASIS_PRODUCTS = 64_000_000
CHUNK = 1_024
_EPS = np.finfo(np.float64).eps


def _bad(message, path, fix="pass finite numeric values with the documented shape"):
    raise AttaError("bad_arg", message, path=path, fix=fix)


def _budget(message, path):
    raise AttaError("nurbs_budget", message, path=path,
                    fix="reduce controls/sampling or use budgets within the hard limits")


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        _bad(name + " must be an integer", name)
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError):
        finite = False
    if not finite or value != int(value) or not low <= value <= high:
        _bad("%s must be an integer in [%d, %d]" % (name, low, high), name)
    return int(value)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        _bad(name + " must be a positive finite number", name)
    try:
        out = float(value)
    except (OverflowError, ValueError):
        _bad(name + " is not representable", name)
    if not math.isfinite(out) or out <= 0:
        _bad(name + " must be a positive finite number", name)
    return out


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        _bad(name + " must be boolean", name)
    return bool(value)


def _array(value, name, shape=None, max_values=MAX_CONTROL_POINTS * 3):
    """Check nested container size/types BEFORE numpy makes a dense allocation."""
    visited = 0

    def inspect(x, depth=0):
        nonlocal visited
        visited += 1
        if visited > 2 * max_values + 16:
            _budget(name + " exceeds the input traversal budget", name)
        if isinstance(x, np.ndarray):
            if x.size > max_values:
                _budget(name + " exceeds the input allocation budget", name)
            if x.dtype.kind not in "iuf" or x.ndim + depth > 4:
                _bad(name + " must contain real numeric values", name)
            return x.shape, x.size
        if isinstance(x, (list, tuple)):
            if depth >= 4 or len(x) > max_values:
                _budget(name + " exceeds the input allocation budget", name)
            subshape, count = None, 0
            for item in x:
                sh, n = inspect(item, depth + 1)
                if subshape is not None and subshape != sh:
                    _bad(name + " must be rectangular", name)
                subshape = sh
                count += n
                if count > max_values:
                    _budget(name + " exceeds the input allocation budget", name)
            return (len(x),) + (subshape or ()), count
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating)):
            _bad(name + " must contain real numbers, not strings/bools/objects", name)
        return (), 1

    actual, _ = inspect(value)
    if shape is not None and actual != shape:
        _bad("%s must have shape %r, got %r" % (name, shape, actual), name)
    try:
        arr = np.array(value, dtype=np.float64, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        _bad(name + " cannot be represented as float64: " + str(exc), name)
    if not np.isfinite(arr).all():
        _bad(name + " contains NaN or infinity", name)
    return arr


def _knots(value, count, degree, name):
    if value is None:
        interior = np.arange(1, count - degree, dtype=np.float64) / (count - degree)
        knots = np.concatenate((np.zeros(degree + 1), interior, np.ones(degree + 1)))
    else:
        knots = _array(value, name, (count + degree + 1,), max_values=MAX_AXIS_CONTROLS + MAX_DEGREE + 1)
    if np.any(knots[1:] < knots[:-1]):
        _bad(name + " must be nondecreasing", name)
    lo, hi = float(knots[degree]), float(knots[count])
    with np.errstate(over="ignore", invalid="ignore"):
        span = hi - lo
        normalized = (knots - lo) / span if span > 0 else knots.copy()
    if not math.isfinite(span) or span <= 0 or not np.isfinite(normalized).all():
        _bad(name + " must have a finite, positive active domain", name)
    unique, multiplicity = np.unique(knots, return_counts=True)
    if np.any(multiplicity > degree + 1) or np.any(multiplicity[(unique > lo) & (unique < hi)] > degree):
        _bad(name + " has excessive knot multiplicity (discontinuous patches unsupported)", name,
             "use endpoint multiplicity <= degree+1 and interior multiplicity <= degree")
    active = np.unique(normalized[degree:count + 1])
    if np.any(np.diff(active) < 64 * _EPS):
        _bad(name + " has numerically unresolved knot spans", name,
             "separate active knot spans by at least 64 machine eps relative to the domain")
    return knots, normalized, (lo, hi), span


def _basis(t, degree, count, x, derivative):
    """Nonzero basis values/derivatives only, at paired normalized parameters.

    Right-hand spans at interior knots; left-hand at the final active endpoint.
    The saved degree-(p-1) basis uses the SAME span as degree p.
    """
    span = np.clip(np.searchsorted(t, x, side="right") - 1, degree, count - 1)
    # At a non-clamped endpoint, the last active span can precede count-1.
    endpoint_span = max(degree, min(count - 1, int(np.searchsorted(t, 1.0, side="left")) - 1))
    span = np.where(x == 1.0, endpoint_span, span)
    n = len(x)
    b = np.zeros((n, degree + 1))
    b[:, 0] = 1.0
    left, right = np.zeros_like(b), np.zeros_like(b)
    lower = b[:, :1].copy() if degree == 1 and derivative else None
    for j in range(1, degree + 1):
        left[:, j] = x - t[span + 1 - j]
        right[:, j] = t[span + j] - x
        saved = np.zeros(n)
        for r in range(j):
            denom = right[:, r + 1] + left[:, j - r]
            temp = np.divide(b[:, r], denom, out=np.zeros(n), where=denom != 0)
            b[:, r] = saved + right[:, r + 1] * temp
            saved = left[:, j - r] * temp
        b[:, j] = saved
        if derivative and j == degree - 1:
            lower = b[:, :degree].copy()
    ids = span[:, None] - degree + np.arange(degree + 1)[None, :]
    db = None
    if derivative:
        db = np.zeros_like(b)
        den1 = t[ids[:, 1:] + degree] - t[ids[:, 1:]]
        den2 = t[ids[:, :-1] + degree + 1] - t[ids[:, :-1] + 1]
        db[:, 1:] += np.divide(degree * lower, den1, out=np.zeros_like(lower), where=den1 != 0)
        db[:, :-1] -= np.divide(degree * lower, den2, out=np.zeros_like(lower), where=den2 != 0)
    return ids, b, db


class NURBSSurface:
    """A positive-weight, continuous rational tensor-product spline patch.

    ``control_points[u][v][xyz]`` are in metres. Knots and evaluation arguments
    are dimensionless and retain their original domain. Default knots are open
    clamped uniform; omitted weights give a polynomial B-spline. Degrees 1..8.
    Evaluation broadcasts u and v as *paired* coordinates, not an outer grid.
    """
    def __init__(self, control_points, *, weights=None, degree_u=None, degree_v=None,
                 knots_u=None, knots_v=None):
        cp = _array(control_points, "control_points")
        if cp.ndim != 3 or cp.shape[2] != 3 or min(cp.shape[:2]) < 2:
            _bad("control_points must have shape (nu>=2,nv>=2,3)", "control_points")
        nu, nv, _ = cp.shape
        if max(nu, nv) > MAX_AXIS_CONTROLS or nu * nv > MAX_CONTROL_POINTS:
            _budget("control net exceeds the control point budget", "control_points")
        pu = min(3, nu - 1) if degree_u is None else _integer(degree_u, "degree_u", 1, MAX_DEGREE)
        pv = min(3, nv - 1) if degree_v is None else _integer(degree_v, "degree_v", 1, MAX_DEGREE)
        if pu >= nu or pv >= nv:
            _bad("degree must be smaller than its axis control count", "degree")
        w = np.ones((nu, nv)) if weights is None else _array(weights, "weights", (nu, nv), MAX_CONTROL_POINTS)
        if np.any(w <= 0):
            _bad("weights must be strictly positive", "weights")
        w /= float(w.max())  # rational invariance to a global weight scale
        if float(w.min()) < 1e-12:
            _bad("weight ratio exceeds 1e12; rational denominator is ill-conditioned", "weights",
                 "use positive weights with max(weights)/min(weights) <= 1e12")
        ku, tu, du, su = _knots(knots_u, nu, pu, "knots_u")
        kv, tv, dv, sv = _knots(knots_v, nv, pv, "knots_v")
        self.origin = cp[0, 0].copy()
        with np.errstate(over="ignore", invalid="ignore"):
            local = cp - self.origin
        if not np.isfinite(local).all():
            _bad("control point extent overflows float64", "control_points")
        self.scale = float(np.max(np.abs(local)))
        if self.scale == 0:
            _bad("control net has zero extent", "control_points")
        self._local = local / self.scale
        self.control_points, self.weights = cp, w
        self.degree_u, self.degree_v = pu, pv
        self.knots_u, self.knots_v = ku, kv
        self.domain_u, self.domain_v = du, dv
        self._tu, self._tv, self._su, self._sv = tu, tv, su, sv
        for a in (self.origin, self._local, cp, w, ku, kv, tu, tv):
            a.setflags(write=False)

    def _parameters(self, u, v, max_evaluations):
        cap = _integer(max_evaluations, "max_evaluations", 1, MAX_EVALUATIONS)
        a = _array(u, "u", max_values=cap)
        b = _array(v, "v", max_values=cap)
        try:
            shape = np.broadcast_shapes(a.shape, b.shape)
        except ValueError:
            _bad("u and v must have broadcast-compatible shapes", "parameters")
        count = math.prod(shape)
        if count > cap or count * (self.degree_u + 1) * (self.degree_v + 1) > MAX_BASIS_PRODUCTS:
            _budget("evaluation exceeds paired-sample/work budget", "parameters")
        if np.any(a < self.domain_u[0]) or np.any(a > self.domain_u[1]):
            _bad("u is outside the active knot domain", "u")
        if np.any(b < self.domain_v[0]) or np.any(b > self.domain_v[1]):
            _bad("v is outside the active knot domain", "v")
        a, b = np.broadcast_arrays(a, b)
        return ((a - self.domain_u[0]) / self._su).ravel(), ((b - self.domain_v[0]) / self._sv).ravel(), shape

    def _evaluate_normalized(self, u, v, derivative=False):
        n = len(u)
        result = np.empty((n, 3))
        du = np.empty_like(result) if derivative else None
        dv = np.empty_like(result) if derivative else None
        nu, nv = self.weights.shape
        for start in range(0, n, CHUNK):
            sl = slice(start, min(n, start + CHUNK))
            iu, bu, dbu = _basis(self._tu, self.degree_u, nu, u[sl], derivative)
            iv, bv, dbv = _basis(self._tv, self.degree_v, nv, v[sl], derivative)
            w = self.weights[iu[:, :, None], iv[:, None, :]]
            cp = self._local[iu[:, :, None], iv[:, None, :]]
            coeff = bu[:, :, None] * bv[:, None, :] * w
            den = coeff.sum(axis=(1, 2))
            p = np.einsum("bij,bijk->bk", coeff, cp) / den[:, None]
            result[sl] = p
            if derivative:
                for basis, dest in ((dbu[:, :, None] * bv[:, None, :], du),
                                    (bu[:, :, None] * dbv[:, None, :], dv)):
                    dcoeff = basis * w
                    dest[sl] = (np.einsum("bij,bijk->bk", dcoeff, cp)
                                - p * dcoeff.sum(axis=(1, 2))[:, None]) / den[:, None]
        if not np.isfinite(result).all() or (derivative and (not np.isfinite(du).all() or not np.isfinite(dv).all())):
            _bad("spline evaluation became non-finite", "parameters", "reduce knot/weight conditioning")
        return (result, du, dv) if derivative else result

    def evaluate(self, u, v, *, max_evaluations=MAX_EVALUATIONS):
        """Positions in metres with shape broadcast(u,v).shape + (3,)."""
        a, b, shape = self._parameters(u, v, max_evaluations)
        with np.errstate(over="ignore", invalid="ignore"):
            result = self._evaluate_normalized(a, b) * self.scale + self.origin
        if not np.isfinite(result).all():
            _bad("world positions overflow float64", "control_points")
        return result.reshape(shape + (3,))

    def derivatives(self, u, v, *, max_evaluations=MAX_EVALUATIONS):
        """Return analytic (dS/du,dS/dv), per ORIGINAL knot parameter unit."""
        a, b, shape = self._parameters(u, v, max_evaluations)
        _, du, dv = self._evaluate_normalized(a, b, True)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            du = (du * self.scale) / self._su
            dv = (dv * self.scale) / self._sv
        if not np.isfinite(du).all() or not np.isfinite(dv).all():
            _bad("derivatives overflow float64", "parameters", "rescale the knot parameter domain")
        return du.reshape(shape + (3,)), dv.reshape(shape + (3,))

    def normal(self, u, v, *, max_evaluations=MAX_EVALUATIONS):
        """Unit dS/du cross dS/dv; singular/near-parallel tangents fail clearly."""
        a, b, shape = self._parameters(u, v, max_evaluations)
        _, du, dv = self._evaluate_normalized(a, b, True)
        # Per-tangent rescaling avoids area underflow on microscopic geometry.
        au, av = np.max(np.abs(du), axis=1), np.max(np.abs(dv), axis=1)
        if np.any(au == 0) or np.any(av == 0):
            raise AttaError("nurbs_singular", "surface normal is undefined", path="parameters",
                            fix="remove collapsed control rows/columns or avoid singular parameters")
        du, dv = du / au[:, None], dv / av[:, None]
        cross = np.cross(du, dv)
        length = np.linalg.norm(cross, axis=1)
        if np.any(length <= 64 * _EPS * np.linalg.norm(du, axis=1) * np.linalg.norm(dv, axis=1)):
            raise AttaError("nurbs_singular", "surface tangents are parallel", path="parameters",
                            fix="use a regular nondegenerate patch")
        return (cross / length[:, None]).reshape(shape + (3,))


# Correspondence samples on each triangle and all four boundary edges. The
# diagonal is 00->11; a point (a,b) has barycentric weights (1-a,a-b,b) if a>=b.
_PROBES = ((0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5), (0.5, 0.5),
           (2 / 3, 1 / 3), (1 / 3, 2 / 3), (0.25, 0.25), (0.75, 0.75))


def _axis_grid(knots, degree, count, samples):
    active = np.unique(knots[degree:count + 1])
    # Budget already checked by the caller before constructing these arrays.
    return np.concatenate([np.linspace(a, b, samples)[:-1]
                           for a, b in zip(active[:-1], active[1:])] + [active[-1:]])


def _grid_positions(surface, u, v):
    out = np.empty((len(u), len(v), 3))
    flat = out.reshape(-1, 3)
    for start in range(0, len(flat), CHUNK):
        ids = np.arange(start, min(len(flat), start + CHUNK))
        flat[start:start + len(ids)] = surface._evaluate_normalized(u[ids // len(v)], v[ids % len(v)])
    return out


def _sample_error(surface, u, v, positions):
    maximum = 0.0
    cells = (len(u) - 1) * (len(v) - 1)
    for start in range(0, cells, CHUNK):
        ids = np.arange(start, min(cells, start + CHUNK))
        i, j = ids // (len(v) - 1), ids % (len(v) - 1)
        p00, p10 = positions[i, j], positions[i + 1, j]
        p01, p11 = positions[i, j + 1], positions[i + 1, j + 1]
        # Reject locally collapsed triangles without absolute world-unit EPS.
        for a, b in ((p10 - p00, p11 - p00), (p11 - p00, p01 - p00)):
            sa, sb = np.max(np.abs(a), axis=1), np.max(np.abs(b), axis=1)
            if np.any(sa == 0) or np.any(sb == 0):
                raise AttaError("nurbs_singular", "tessellation has collapsed edges", path="control_points",
                                fix="remove singularities or increase initial sampling")
            aa, bb = a / sa[:, None], b / sb[:, None]
            if np.any(np.linalg.norm(np.cross(aa, bb), axis=1) <= 64 * _EPS):
                raise AttaError("nurbs_singular", "tessellation contains degenerate triangles", path="control_points",
                                fix="use a regular patch or increase initial sampling")
        for a, b in _PROBES:
            pu = u[i] + (u[i + 1] - u[i]) * a
            pv = v[j] + (v[j + 1] - v[j]) * b
            exact = surface._evaluate_normalized(pu, pv)
            approx = ((1 - a) * p00 + (a - b) * p10 + b * p11 if a >= b else
                      (1 - b) * p00 + (b - a) * p01 + a * p11)
            maximum = max(maximum, float(np.max(np.linalg.norm(exact - approx, axis=1))))
    result = maximum * surface.scale
    if not math.isfinite(result):
        _bad("sampled error overflows float64", "control_points", "reduce the geometric extent")
    return result


def tessellate(surface, *, tolerance=5e-5, u_samples=3, v_samples=3,
               max_vertices=100_000, max_faces=200_000, max_evaluations=1_000_000,
               max_refinements=8, fail_on_tolerance=True, flip=False):
    """Return one open, crack-free patch Mesh with honest sampled-error metadata.

    Samples count per nonzero knot span. All spans enter the initial tensor grid;
    each unsuccessful level bisects every interval. Shared indices guarantee no
    T-junctions WITHIN this patch. Independent patches are not sewn. Nine probes
    per cell measure parametric correspondence to the piecewise linear triangles;
    this is neither a Hausdorff bound nor a global extremum/error guarantee.
    """
    if not isinstance(surface, NURBSSurface):
        _bad("surface must be a NURBSSurface", "surface")
    tolerance = _positive(tolerance, "tolerance")
    us = _integer(u_samples, "u_samples", 2, 1_025)
    vs = _integer(v_samples, "v_samples", 2, 1_025)
    mv = _integer(max_vertices, "max_vertices", 4, MAX_VERTICES)
    mf = _integer(max_faces, "max_faces", 2, MAX_FACES)
    me = _integer(max_evaluations, "max_evaluations", 1, MAX_EVALUATIONS)
    mr = _integer(max_refinements, "max_refinements", 0, 16)
    strict, flip = _flag(fail_on_tolerance, "fail_on_tolerance"), _flag(flip, "flip")
    nu, nv = surface.weights.shape
    spans_u = len(np.unique(surface._tu[surface.degree_u:nu + 1])) - 1
    spans_v = len(np.unique(surface._tv[surface.degree_v:nv + 1])) - 1
    nx, ny = spans_u * (us - 1) + 1, spans_v * (vs - 1) + 1
    used = 0
    support = (surface.degree_u + 1) * (surface.degree_v + 1)

    def reason(x, y):
        nverts, cells = x * y, (x - 1) * (y - 1)
        if nverts > mv:
            return "max_vertices"
        if 2 * cells > mf:
            return "max_faces"
        proposed = used + nverts + len(_PROBES) * cells
        if proposed > me:
            return "max_evaluations"
        if proposed * support > MAX_BASIS_PRODUCTS:
            return "max_basis_products"
        return None

    stop = reason(nx, ny)
    if stop:
        _budget("initial tensor grid exceeds " + stop, stop)
    u = _axis_grid(surface._tu, surface.degree_u, nu, us)
    v = _axis_grid(surface._tv, surface.degree_v, nv, vs)
    for level in range(mr + 1):
        ideal_positions = _grid_positions(surface, u, v)
        with np.errstate(over="ignore", invalid="ignore"):
            world_positions = ideal_positions * surface.scale + surface.origin
        if not np.isfinite(world_positions).all():
            _bad("tessellated world positions overflow float64", "control_points")
        # Compare against the ACTUAL representable output triangles. At large
        # translations, adding the origin can round away local surface detail.
        positions = (world_positions - surface.origin) / surface.scale
        world_roundoff = float(np.max(np.linalg.norm(positions - ideal_positions, axis=2))) * surface.scale
        sampled = _sample_error(surface, u, v, positions)
        used += len(u) * len(v) + len(_PROBES) * (len(u) - 1) * (len(v) - 1)
        target = sampled <= tolerance
        if target:
            stop = "sampled_target_met"
            break
        if level == mr:
            stop = "max_refinements"
            break
        stop = reason(2 * len(u) - 1, 2 * len(v) - 1)
        if stop:
            break
        # Keep span endpoints exact; no spatial deduplication or tolerance weld.
        def bisect(axis):
            out = np.empty(2 * len(axis) - 1)
            out[::2] = axis
            out[1::2] = axis[:-1] + np.diff(axis) * 0.5
            if np.any(np.diff(out) <= 0):
                return None
            return out
        next_u, next_v = bisect(u), bisect(v)
        if next_u is None or next_v is None:
            stop = "parameter_precision"
            break
        u, v = next_u, next_v
    report = {
        "method": "knot_span_tensor_grid_global_bisection", "exact": False,
        "error_metric": "sampled_parametric_correspondence_to_triangles",
        "sampled_error": sampled, "tolerance": tolerance, "length_unit": "m",
        "max_vertex_roundoff": world_roundoff,
        "surface_error_bound": None, "target_met": bool(target), "stop_reason": stop,
        "refinements": level, "grid_shape": [len(u), len(v)],
        "evaluations": used, "probes_per_cell": len(_PROBES),
        "max_vertices": mv, "max_faces": mf, "max_evaluations": me,
        "max_basis_products": MAX_BASIS_PRODUCTS, "scratch_chunk": CHUNK,
        "crack_free_within_patch": True, "cross_patch_sewing": False,
        "self_intersection_checked": False,
    }
    if not target and strict:
        raise AttaError("nurbs_tolerance_unmet", "sampled approximation target was not reached: " + stop,
                        path="tolerance", fix="relax tolerance, increase bounded budgets, or explicitly set fail_on_tolerance=false",
                        details=report)
    nx, ny = len(u), len(v)
    cells = (nx - 1) * (ny - 1)
    faces = np.empty((2 * cells, 3), dtype=np.int64)
    ids = np.arange(cells)
    a = (ids // (ny - 1)) * ny + ids % (ny - 1)
    faces[0::2] = np.column_stack((a, a + ny, a + ny + 1))
    faces[1::2] = np.column_stack((a, a + ny + 1, a + 1))
    if flip:
        faces = faces[:, ::-1].copy()
    verts = world_positions.reshape(-1, 3)
    # Detect loss of representable detail during translation (no silent merging).
    for k in (1, 2):
        if np.any(np.all(verts[faces[:, 0]] == verts[faces[:, k]], axis=1)):
            raise AttaError("nurbs_precision", "world coordinates collapse mesh edges", path="control_points",
                            fix="move the object nearer the origin or increase its representable feature scale")
    meta = {"kind": "nurbs_surface", "open": True, "nurbs": report,
            "degree_u": surface.degree_u, "degree_v": surface.degree_v,
            "domain_u": list(surface.domain_u), "domain_v": list(surface.domain_v)}
    if not target:
        meta["issues"] = [{"code": "nurbs_tolerance_unmet", "severity": "warn",
                           "message": "sampled approximation target not reached; no global bound is claimed",
                           "fix": "increase bounded budgets or relax tolerance", "details": report}]
    return Mesh(verts, faces, meta)
