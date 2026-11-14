# -*- coding: utf-8 -*-
"""Form-first construction: skeleton, silhouette and visual agreement.

Why this module exists
----------------------
The v1.2 kernel is *parametric*: you name a recipe, tune numbers, and geometry
appears.  That works when the numbers already encode the shape.  It fails when
the shape is the thing you are still deciding — every parameter is locally
correct and the whole is wrong, because nothing in the loop ever looked at the
form as a whole.

Three mechanisms close that gap without asking a model to read a render:

``skeleton_solid``
    A **skeleton** is a graph of joints and bones (a "mathematical bone").
    The solid is grown as a *field*: every point's signed distance is the
    distance to the nearest bone minus that bone's local radius.  One field,
    one isosurface, no booleans — so a limb never fails to fuse at a joint.
    Radius may vary along a bone, so a limb tapers, and ``blend`` rounds the
    crotch where two bones meet, which is what flesh actually does.

``silhouette_solid``
    Three orthographic outlines (front / top / right) *intersected* into a
    solid — the classical visual hull.  It is the direct answer to "I know what
    it looks like from these three sides, not what its parameters are".  Also
    a field: ``max`` of the per-view distances, so the intersection is exact
    and watertight by construction.

``form_agreement``
    Compares a built mesh against a target outline or skeleton and reports
    per-view IoU plus signed area error, so "the numbers matched but it looks
    wrong" becomes a number an agent can act on.

Everything here is pure numpy and metre-based, like the rest of the kernel.
Public entry points take millimetres and convert at the boundary.
"""

from __future__ import annotations

import math

import numpy as np

from . import mesh as M
from . import probe
from .errors import AttaError

__all__ = [
    "skeleton_solid",
    "silhouette_solid",
    "form_agreement",
    "skeleton_from_points",
    "METHOD",
]

METHOD = "atta.form/v1"

#: A field is only trustworthy when the thinnest feature spans several voxels.
_MIN_VOX = 3.0
_MAX_SAMPLES = 2_500_000
_MAX_RES = 220


# ---------------------------------------------------------------- field grid

def _field_grid(lo, hi, spacing):
    """Sample a padded grid, coarsening until the exact shape fits the budget."""
    from .sdf import fit_spacing
    from .sdfkit import grid

    spacing = fit_spacing(lo, hi, spacing, _MAX_SAMPLES)
    X, Y, Z, org = grid(lo, hi, spacing, max_samples=_MAX_SAMPLES)
    return (X, Y, Z), org, spacing


def _isosurface(phi, org, spacing):
    from .sdf import marching_tets
    return marching_tets(phi, org, spacing)


# ---------------------------------------------------------------- skeleton

def _as_vec3(value, name, path, scale=1.0):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise AttaError("bad_arg", "%s must be 3 finite numbers" % name,
                        path=path, fix="pass [x, y, z] in mm")
    return arr * float(scale)


def _seg_distance(X, Y, Z, a, b):
    """Distance from every grid point to segment ``a-b``."""
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-18:
        return np.sqrt((X - a[0]) ** 2 + (Y - a[1]) ** 2 + (Z - a[2]) ** 2), None
    t = ((X - a[0]) * ab[0] + (Y - a[1]) * ab[1] + (Z - a[2]) * ab[2]) / denom
    t = np.clip(t, 0.0, 1.0)
    dx = X - (a[0] + t * ab[0])
    dy = Y - (a[1] + t * ab[1])
    dz = Z - (a[2] + t * ab[2])
    return np.sqrt(dx * dx + dy * dy + dz * dz), t


def _smin(a, b, k):
    """Polynomial smooth minimum: rounds the crease where two bones meet."""
    if k <= 0.0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def skeleton_solid(joints, bones, *, segments=32, blend=0.0, resolution=0):
    """Grow a closed solid around a skeleton.

    ``joints`` is a list of ``{"id", "at", "radius"}``; ``bones`` is a list of
    ``{"a", "b", "radius_a", "radius_b"}`` naming joint ids.
    Every bone contributes ``distance_to_segment - radius(t)`` with the radius
    interpolated along the bone; every joint with a radius contributes a ball.
    The pieces are combined with a smooth minimum, so they fuse into one body
    instead of intersecting as separate lumps.

    **Units:** metres throughout, and the field names carry no unit suffix so
    there is nothing to misread.  :mod:`atta.families.form_ops` converts the
    spec's ``*_mm`` keys once at the feature-graph boundary.

    Returns ``(mesh, info)`` in metres.

    Proportion and pose live in the skeleton, not in a parameter list: move a
    joint and the body follows; change a radius and only the flesh changes.
    That is the difference between designing a form and tuning a shape.
    """
    if not isinstance(joints, (list, tuple)) or not joints:
        raise AttaError("bad_arg", "skeleton needs at least one joint",
                        path="joints", fix='pass [{"id":"hip","at":[0,0,0]}]')
    index = {}
    placed = {}
    for i, row in enumerate(joints):
        if not isinstance(row, dict):
            raise AttaError("bad_arg", "joint must be an object",
                            path="joints[%d]" % i,
                            fix='use {"id":..., "at":[...], "radius":...}')
        jid = row.get("id")
        if not isinstance(jid, str) or not jid:
            raise AttaError("bad_arg", "joint needs a non-empty id",
                            path="joints[%d].id" % i, fix="give each joint a unique id")
        if jid in index:
            raise AttaError("duplicate_id", "joint id %r is used twice" % jid,
                            path="joints[%d].id" % i, fix="rename one of them")
        index[jid] = i
        at = _as_vec3(row.get("at"), "at", "joints[%d].at" % i)
        r = row.get("radius", 0.0)
        r = float(r) if r is not None else 0.0
        if not math.isfinite(r) or r < 0.0:
            raise AttaError("bad_arg", "joint radius must be >= 0",
                            path="joints[%d].radius" % i, fix="use radius >= 0")
        placed[jid] = (at, r)

    bones = list(bones or [])
    for i, row in enumerate(bones):
        if not isinstance(row, dict):
            raise AttaError("bad_arg", "bone must be an object",
                            path="bones[%d]" % i, fix='use {"a":"hip","b":"knee"}')
        for key in ("a", "b"):
            ref = row.get(key)
            if not isinstance(ref, str) or ref not in placed:
                raise AttaError("unknown_input", "bone %s names unknown joint %r"
                                % (key, ref), path="bones[%d].%s" % (i, key),
                                fix="available joints: " + ", ".join(index))
        if row.get("a") == row.get("b"):
            raise AttaError("bad_arg", "bone endpoints must differ",
                            path="bones[%d]" % i, fix="name two different joints")

    if not bones and not any(r > 0 for _, r in placed.values()):
        raise AttaError("bad_arg",
                        "skeleton has no bones and no joint radii, so it has no volume",
                        path="bones", fix="add a bone or give joints a radius")

    # ---- resolve per-bone radii, and the thinnest feature for the grid
    resolved = []
    radii = [r for _, r in placed.values() if r > 0.0]
    for i, row in enumerate(bones):
        a, ra0 = placed[row["a"]]
        b, rb0 = placed[row["b"]]
        ra = row.get("radius_a")
        rb = row.get("radius_b")
        ra = float(ra) if ra is not None else ra0
        rb = float(rb) if rb is not None else rb0
        if not (math.isfinite(ra) and math.isfinite(rb)):
            raise AttaError("bad_arg", "bone %d radius is not finite" % i,
                            path="bones[%d]" % i, fix="use finite radii")
        if ra < 0.0 or rb < 0.0:
            raise AttaError("bad_arg", "bone %d radius must be >= 0" % i,
                            path="bones[%d]" % i, fix="use non-negative radii")
        if ra <= 0.0 and rb <= 0.0:
            raise AttaError("bad_arg",
                            "bone %d has no radius at either end" % i,
                            path="bones[%d]" % i,
                            fix="set radius_a/radius_b or a joint radius")
        # A zero end is a tip: taper to a small cap rather than a knife edge.
        ra = max(ra, 1e-5)
        rb = max(rb, 1e-5)
        resolved.append((a, b, ra, rb))
        radii.extend([ra, rb])

    r_min = min(radii) if radii else 1e-3
    pts = np.array([at for at, _ in placed.values()], dtype=np.float64)
    r_max = max(radii) if radii else 0.0
    lo = pts.min(axis=0) - r_max
    hi = pts.max(axis=0) + r_max

    blend = float(blend)
    if blend < 0.0:
        raise AttaError("bad_arg", "blend must be >= 0", path="blend",
                        fix="use blend: 0 (or a few mm for an organic fillet)")

    # Grid spacing: enough voxels across the thinnest limb, capped for sanity.
    res = int(resolution) if resolution else 0
    if res > 0:
        span = float(np.linalg.norm(hi - lo)) or 1e-3
        spacing = span / float(max(8, min(_MAX_RES, res)))
    else:
        spacing = max(r_min / _MIN_VOX, 1e-5)
        span = float(np.linalg.norm(hi - lo)) or 1e-3
        spacing = max(spacing, span / float(_MAX_RES))

    (X, Y, Z), org, spacing = _field_grid(lo, hi, spacing)

    phi = None
    for jid, (at, r) in placed.items():
        if r <= 0.0:
            continue
        d = np.sqrt((X - at[0]) ** 2 + (Y - at[1]) ** 2 + (Z - at[2]) ** 2) - r
        phi = d if phi is None else _smin(phi, d, blend)
    for (a, b, ra, rb) in resolved:
        d, t = _seg_distance(X, Y, Z, a, b)
        if t is None:
            rloc = np.maximum(ra, rb)
        else:
            rloc = ra + (rb - ra) * t
        d = d - rloc
        phi = d if phi is None else _smin(phi, d, blend)

    # The isosurface is only closed if the field is positive on the grid
    # boundary; the margin above guarantees that, but assert it so a silently
    # open mesh can never leave this function.
    if not (phi[0, :, :].min() > 0 and phi[-1, :, :].min() > 0
            and phi[:, 0, :].min() > 0 and phi[:, -1, :].min() > 0
            and phi[:, :, 0].min() > 0 and phi[:, :, -1].min() > 0):
        raise AttaError("form_open", "the skeleton field touches the grid boundary",
                        path="joints",
                        fix="increase the radii margin or move the skeleton inward")

    samples = int(phi.size)
    del X, Y, Z, d
    if resolved:
        del t, rloc
    out = _isosurface(phi, org, spacing)
    info = {
        "joints": len(placed),
        "bones": len(bones),
        "pieces": len(placed) + len(bones),
        "blend_mm": round(blend * 1000.0, 4),
        "spacing_mm": round(spacing * 1000.0, 4),
        "samples": samples,
        "method": METHOD,
        "exact": False,
    }
    return out, info


def skeleton_from_points(points, radius, *, closed=False, segments=32,
                         blend=0.0):
    """Convenience: a chain of joints through ``points`` (metres).

    Give the centre-line and one radius and the flesh follows.  ``closed``
    joins the last point to the first (a ring).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise AttaError("bad_arg", "points must be (n>=2, 3)",
                        path="points", fix="pass a centre-line of [x,y,z] points")
    if not np.isfinite(pts).all():
        raise AttaError("bad_arg", "points contains NaN or Inf",
                        path="points", fix="remove non-finite coordinates")
    joints = [{"id": "p%d" % i, "at": pts[i].tolist(), "radius": radius}
              for i in range(len(pts))]
    bones = [{"a": "p%d" % i, "b": "p%d" % (i + 1)} for i in range(len(pts) - 1)]
    if closed and len(pts) > 2:
        bones.append({"a": "p%d" % (len(pts) - 1), "b": "p0"})
    return skeleton_solid(joints, bones, segments=segments, blend=blend)


# ---------------------------------------------------------------- silhouette

def _loop_area(loop):
    p = np.asarray(loop, dtype=np.float64)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                              - p[:, 1] * np.roll(p[:, 0], -1)))


def _clean_loop(loop, name, path):
    p = np.asarray(loop, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        raise AttaError("bad_arg", "%s must be a closed 2D loop of >= 3 points"
                        % name, path=path, fix="pass [[x,y], ...] in mm")
    if not np.isfinite(p).all():
        raise AttaError("bad_arg", "%s contains NaN or Inf" % name, path=path,
                        fix="remove non-finite coordinates")
    if len(p) > 3 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    if abs(_loop_area(p)) <= 1e-12:
        raise AttaError("bad_arg", "%s has zero area" % name, path=path,
                        fix="give the outline a non-degenerate polygon")
    if _loop_area(p) < 0.0:
        p = p[::-1]
    return p


def _poly_distance(u, v, loop):
    """Unsigned distance from each (u,v) grid point to a closed polygon."""
    n = len(loop)
    best = np.full(u.shape, np.inf)
    for i in range(n):
        ax, ay = float(loop[i, 0]), float(loop[i, 1])
        bx, by = float(loop[(i + 1) % n, 0]), float(loop[(i + 1) % n, 1])
        ex, ey = bx - ax, by - ay
        den = ex * ex + ey * ey
        if den <= 1e-18:
            d = np.sqrt((u - ax) ** 2 + (v - ay) ** 2)
        else:
            t = np.clip(((u - ax) * ex + (v - ay) * ey) / den, 0.0, 1.0)
            d = np.sqrt((u - (ax + t * ex)) ** 2 + (v - (ay + t * ey)) ** 2)
        best = np.minimum(best, d)
    return best


def _poly_inside(u, v, loop):
    """Even-odd point-in-polygon, vectorised over the grid."""
    inside = np.zeros(u.shape, dtype=bool)
    n = len(loop)
    for i in range(n):
        x1, y1 = float(loop[i, 0]), float(loop[i, 1])
        x2, y2 = float(loop[(i + 1) % n, 0]), float(loop[(i + 1) % n, 1])
        cond = (y1 > v) != (y2 > v)
        if not cond.any():
            continue
        xint = (x2 - x1) * (v - y1) / ((y2 - y1) or 1e-15) + x1
        inside ^= cond & (u < xint)
    return inside


def _view_axes(view):
    """(u_axis, v_axis, depth_axis) for a named orthographic view."""
    return {
        "front": (0, 2, 1),   # u=x, v=z, depth=y
        "top": (0, 1, 2),     # u=x, v=y, depth=z
        "right": (1, 2, 0),   # u=y, v=z, depth=x
    }[view]


def silhouette_solid(views, *, height=None, resolution=0):
    """Intersect orthographic outlines into a solid (visual hull).

    ``views`` maps a view name to ``{"outline": [[u,v], ...], "depth":
    [min,max]}``.  ``front`` is (u=x, v=z), ``top`` is (u=x, v=y), ``right`` is
    (u=y, v=z) — the same convention as :mod:`atta.probe`.

    The hull is built as a field: ``max`` over views of
    ``max(polygon_distance, depth_slab_distance)``.  Intersecting fields is
    exact and watertight, where intersecting meshes can fail on a coincident
    face.

    The result is the **visual hull**: the largest solid consistent with every
    view.  It cannot represent a concavity no view reveals, so treat it as a
    first body to refine, not a finished part.
    """
    if not isinstance(views, dict) or not views:
        raise AttaError("bad_arg", "views must be a non-empty object",
                        path="views",
                        fix='use {"front":{"outline":[...]}, "top":{...}}')
    known = ("front", "top", "right")
    unknown = [k for k in views if str(k).lower() not in known]
    if unknown:
        raise AttaError("bad_arg", "unknown view(s): %s" % ", ".join(map(str, unknown)),
                        path="views",
                        fix="use front (x-z), top (x-y) and/or right (y-z)")

    prepared = []
    extents = {}
    for name, raw in views.items():
        key = str(name).lower()
        if not isinstance(raw, dict):
            raise AttaError("bad_arg", "view %r must be an object" % key,
                            path="views.%s" % key,
                            fix='use {"outline":[[u,v],...]}')
        loop = _clean_loop(raw.get("outline"), "views.%s.outline" % key,
                           "views.%s.outline" % key)
        depth = raw.get("depth")
        if depth is None:
            if height is None:
                raise AttaError("bad_arg",
                                "view %r needs depth (or pass height)" % key,
                                path="views.%s.depth" % key,
                                fix="pass [min,max] along the view axis")
            depth = [0.0, float(height)]
        if not isinstance(depth, (list, tuple)) or len(depth) != 2:
            raise AttaError("bad_arg", "views.%s.depth must be [min,max]" % key,
                            path="views.%s.depth" % key, fix="use [0, 0.04]")
        d0, d1 = float(depth[0]), float(depth[1])
        if not (math.isfinite(d0) and math.isfinite(d1)) or d1 <= d0:
            raise AttaError("bad_arg", "view %r depth must be finite with max > min"
                            % key, path="views.%s.depth" % key,
                            fix="use [0, 0.04] in metres")
        prepared.append((key, loop, d0, d1))
        lo = loop.min(axis=0)
        hi = loop.max(axis=0)
        extents[key] = {"u_mm": [round(float(lo[0]) * 1000.0, 3),
                                 round(float(hi[0]) * 1000.0, 3)],
                        "v_mm": [round(float(lo[1]) * 1000.0, 3),
                                 round(float(hi[1]) * 1000.0, 3)],
                        "depth_mm": [round(d0 * 1000.0, 3), round(d1 * 1000.0, 3)]}

    # World bbox: the intersection of every view's implied slab.  Lower bounds
    # start at -inf and upper at +inf, then each view tightens them.
    lo = np.full(3, -np.inf)
    hi = np.full(3, np.inf)
    for key, loop, d0, d1 in prepared:
        ua, va, da = _view_axes(key)
        ulo, uhi = float(loop[:, 0].min()), float(loop[:, 0].max())
        vlo, vhi = float(loop[:, 1].min()), float(loop[:, 1].max())
        lo[ua] = max(lo[ua], ulo)
        hi[ua] = min(hi[ua], uhi)
        lo[va] = max(lo[va], vlo)
        hi[va] = min(hi[va], vhi)
        lo[da] = max(lo[da], d0)
        hi[da] = min(hi[da], d1)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() \
            or np.any(hi <= lo):
        raise AttaError("form_inconsistent",
                        "the views do not overlap in a common bounding box",
                        path="views",
                        fix="check that every outline uses the same mm origin and "
                            "that depth_mm spans the same axis")

    res = int(resolution) if resolution else 0
    if res > 0:
        span = float(np.linalg.norm(hi - lo)) or 1e-3
        spacing = span / float(max(8, min(_MAX_RES, res)))
    else:
        span = float(np.linalg.norm(hi - lo)) or 1e-3
        spacing = span / 96.0

    (X, Y, Z), org, spacing = _field_grid(lo, hi, spacing)
    coords = (X, Y, Z)

    phi = None
    for key, loop, d0, d1 in prepared:
        ua, va, da = _view_axes(key)
        u = coords[ua]
        v = coords[va]
        w = coords[da]
        d2 = _poly_distance(u, v, loop)
        inside = _poly_inside(u, v, loop)
        d2 = np.where(inside, -d2, d2)
        # Slab along the depth axis, as a signed distance.
        slab = np.maximum(d0 - w, w - d1)
        view_phi = np.maximum(d2, slab)
        phi = view_phi if phi is None else np.maximum(phi, view_phi)

    if not (phi[0, :, :].min() > 0 and phi[-1, :, :].min() > 0
            and phi[:, 0, :].min() > 0 and phi[:, -1, :].min() > 0
            and phi[:, :, 0].min() > 0 and phi[:, :, -1].min() > 0):
        raise AttaError("form_open", "the silhouette field touches the grid boundary",
                        path="views",
                        fix="check depth_mm and that the outlines enclose an area")

    samples = int(phi.size)
    del X, Y, Z, coords, u, v, w, d2, inside, slab, view_phi
    out = _isosurface(phi, org, spacing)
    info = {
        "views": [k for k, _, _, _ in prepared],
        "hull": "visual",
        "extents": extents,
        "spacing_mm": round(spacing * 1000.0, 4),
        "samples": samples,
        "method": METHOD,
        "exact": False,
    }
    return out, info


# ---------------------------------------------------------------- agreement

def _rasterize_loop_like(loop_mm, mesh, view, res):
    """Rasterise ``loop_mm`` on the same pixel frame as ``mesh``'s silhouette.

    ``probe.silhouette`` frames each view from the mesh's own projected extent
    (centred, square span).  To compare pixel-for-pixel the target loop has to
    use that same frame, so project the mesh, derive the frame, and map the
    loop into it.
    """
    right, up = probe._basis(view)
    v = np.asarray(mesh.verts, dtype=np.float64)
    u = v @ right
    vv = v @ up
    umin, umax = float(u.min()), float(u.max())
    vmin, vmax = float(v.min()), float(v.max())
    uc, vc = (umin + umax) * 0.5, (vmin + vmax) * 0.5
    span = max(umax - umin, vmax - vmin, 1e-12)

    p = np.asarray(loop_mm, dtype=np.float64) * 1e-3
    px = (p[:, 0] - uc) / span * (res - 1) + (res - 1) * 0.5
    # probe flips the vertical axis when rasterising.
    py = -(p[:, 1] - vc) / span * (res - 1) + (res - 1) * 0.5

    occ = np.zeros((res, res), dtype=bool)
    n = len(px)
    ys = np.arange(res, dtype=np.float64)
    for i in range(n):
        x1, y1 = float(px[i]), float(py[i])
        x2, y2 = float(px[(i + 1) % n]), float(py[(i + 1) % n])
        cond = (y1 > ys) != (y2 > ys)
        if not cond.any():
            continue
        xint = (x2 - x1) * (ys - y1) / ((y2 - y1) or 1e-15) + x1
        for row in np.nonzero(cond)[0]:
            lo = int(math.ceil(xint[row] - 0.5))
            if lo < 0:
                lo = 0
            if lo < res:
                occ[row, lo:] ^= True
    return occ


def form_agreement(mesh, target, *, res=32, views=("front", "top", "right")):
    """Measure how well a built mesh matches a target form.

    ``target`` may be:

    * ``{"views": {name: {"outline_mm": [[u,v],...]}}}`` — compare silhouettes
      to the given outlines (IoU per view, plus a signed occupancy error);
    * ``{"skeleton": {"joints": [...], "bones": [...]}}`` — grow the skeleton
      and compare the two solids;
    * a ``Mesh`` — compare directly.

    Returns per-view ``iou``, a ``score`` (mean IoU) and ``worst_view``, so "it
    looks wrong from the side" is a locatable fact rather than a feeling.  The
    ASCII overlay marks ``+`` where the model is fatter than the target and
    ``-`` where it is thinner.  This is a judgement aid, not a verdict.
    """
    res = max(8, min(64, int(res)))
    view_list = [str(v).lower() for v in (views or ("front", "top", "right"))]

    if isinstance(target, M.Mesh):
        ref = target
        target_kind = "mesh"
        target_views = None
    elif isinstance(target, dict) and "skeleton" in target:
        sk = target.get("skeleton") or {}
        ref, _ = skeleton_solid(sk.get("joints") or [], sk.get("bones") or [],
                                segments=int(sk.get("segments", 32)),
                                blend=float(sk.get("blend", 0.0)))
        target_kind = "skeleton"
        target_views = None
    elif isinstance(target, dict) and "views" in target:
        ref = None
        target_kind = "views"
        target_views = target.get("views") or {}
    else:
        raise AttaError("bad_arg",
                        "target must be a Mesh, {views:...} or {skeleton:...}",
                        path="target",
                        fix='pass {"views":{"front":{"outline":[...]}}}')

    got = probe.silhouette(mesh, views=tuple(view_list), res=res)
    out_views = {}
    for name in view_list:
        mine = got["views"].get(name) or {}
        mb = np.array([[c == "1" for c in row] for row in (mine.get("bitmap") or [])],
                      dtype=bool)
        if target_views is not None:
            raw = target_views.get(name)
            if not isinstance(raw, dict) or raw.get("outline") is None:
                out_views[name] = {"iou": None,
                                   "note": "target has no %s outline" % name}
                continue
            loop = _clean_loop(raw.get("outline"), "views.%s.outline" % name,
                               "views.%s.outline" % name)
            tb = _rasterize_loop_like(loop, mesh, name, res)
        else:
            rb = probe.silhouette(ref, views=(name,), res=res)
            rv = rb["views"].get(name) or {}
            tb = np.array([[c == "1" for c in row] for row in (rv.get("bitmap") or [])],
                          dtype=bool)
        if mb.shape != tb.shape:
            out_views[name] = {"iou": None, "note": "bitmap shape mismatch"}
            continue
        inter = int((mb & tb).sum())
        union = int((mb | tb).sum())
        iou = round(inter / union, 4) if union else 1.0
        rows = []
        for i in range(res):
            row = []
            for j in range(res):
                if mb[i, j] and tb[i, j]:
                    row.append("#")
                elif mb[i, j]:
                    row.append("+")
                elif tb[i, j]:
                    row.append("-")
                else:
                    row.append(".")
            rows.append("".join(row))
        out_views[name] = {
            "iou": iou,
            "pixels_model": int(mb.sum()),
            "pixels_target": int(tb.sum()),
            "occupancy_delta": int(mb.sum()) - int(tb.sum()),
            "verdict": ("matches" if iou >= 0.90
                        else "close" if iou >= 0.75
                        else "differs"),
            "ascii": rows,
            "legend": "# both, + model only (too fat), - target only (too thin)",
        }

    scores = [v["iou"] for v in out_views.values() if v.get("iou") is not None]
    score = round(float(np.mean(scores)), 4) if scores else None
    worst = None
    if scores:
        worst = min((kv for kv in out_views.items() if kv[1].get("iou") is not None),
                    key=lambda kv: kv[1]["iou"])[0]

    b_got = M.bbox(mesh)
    payload = {
        "schema": "atta.form_agreement/v1",
        "target_kind": target_kind,
        "views": out_views,
        "score": score,
        "worst_view": worst,
        "bbox_model_mm": [round(float(x) * 1000.0, 3) for x in b_got["size"]],
        "volume_model_mm3": round(float(mesh.volume()) * 1e9, 3),
        "res": res,
        "units": "mm",
        "exact": False,
        "method": METHOD,
    }
    if ref is not None:
        b_ref = M.bbox(ref)
        payload["bbox_target_mm"] = [round(float(x) * 1000.0, 3) for x in b_ref["size"]]
        payload["volume_target_mm3"] = round(float(ref.volume()) * 1e9, 3)
        vr = float(ref.volume())
        payload["volume_ratio"] = (round(float(mesh.volume()) / vr, 4)
                                   if vr > 0 else None)
        payload["bbox_delta_mm"] = [
            round(float(a - b) * 1000.0, 3)
            for a, b in zip(b_got["size"], b_ref["size"])]
    return payload
