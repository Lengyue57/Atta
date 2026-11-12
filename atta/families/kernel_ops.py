# -*- coding: utf-8 -*-
"""Kernel ops: subdivision, sweep, pipe, loft, SDF, repair.

These register the modules added in 1.21b1 as first-class feature-graph ops, so
they compose with the original 25 and work inside a Spec v3 ``graph`` model
without touching the engine.

The point of this family is *geometric freedom*: before it, a model could only
be what the primitives described.  Now a coarse cage can be subdivided into an
organic body, a profile can be swept along a twisting path, a form can be
blended smoothly into another, and a broken mesh can be repaired instead of
discarded.
"""

from __future__ import annotations

import math

import numpy as np

from ..errors import AttaError

__all__ = []


def _need_mesh(meshes, op, path):
    if not meshes:
        raise AttaError("bad_arg", "%s needs an input mesh" % op,
                        path=path + ".inputs", fix='pass inputs=["<id>"]')
    return meshes[0]


# ---------------------------------------------------------------- subdivision

def _subdivide_op(*, mesh, scheme="catmull_clark", levels=2,
                  crease_angle_deg=None):
    from .. import subdiv
    out = subdiv.subdivide(mesh, scheme=scheme, levels=int(levels),
                           crease_angle_deg=(float(crease_angle_deg)
                                             if crease_angle_deg is not None
                                             else None))
    out.meta.update({"op": "subdivide", "scheme": scheme, "levels": int(levels)})
    return out


def _subdivide_prepare(args, scale, path):
    scheme = str(args.get("scheme") or "catmull_clark").lower()
    if scheme not in ("catmull_clark", "catmull-clark", "cc", "quad",
                      "loop", "tri", "triangle", "simple", "linear"):
        raise AttaError("bad_arg", "unknown subdivision scheme %r" % scheme,
                        path=path + ".args.scheme",
                        fix="use catmull_clark, loop or simple")
    levels = int(round(float(args.get("levels", 2))))
    if levels < 1 or levels > 5:
        raise AttaError("bad_arg", "levels must be 1..5", path=path + ".args.levels",
                        fix="use levels: 2 (face count grows ~4x per level)")
    crease = args.get("crease_angle_deg")
    crease = float(crease) if crease is not None else None
    if crease is not None and not (0.0 < crease <= 180.0):
        raise AttaError("bad_arg", "crease_angle_deg must be 0..180",
                        path=path + ".args.crease_angle_deg", fix="use 30")
    return {"scheme": scheme, "levels": levels, "crease_angle_deg": crease}


# ---------------------------------------------------------------- sweep

def _sweep_op(*, section_xy, path_, closed=False, twist_deg=0.0,
              roll_deg=0.0, caps=True, scales=None, up_hint=(0.0, 0.0, 1.0)):
    from .. import sweepkit
    out = sweepkit.sweep_profile(
        section_xy, path_, closed=bool(closed), twist_deg=float(twist_deg),
        roll_deg=float(roll_deg), caps=bool(caps), scales=scales,
        up_hint=up_hint)
    out.meta.update({"op": "sweep2", "twist_deg": float(twist_deg)})
    return out


def _sweep_prepare(args, scale, path):
    prof = np.asarray(args.get("section_xy"), dtype=np.float64)
    if prof.ndim != 2 or prof.shape[1] != 2 or len(prof) < 3:
        raise AttaError("bad_arg", "section_xy must be (n>=3, 2)",
                        path=path + ".args.section_xy",
                        fix="pass a closed 2D profile in mm")
    pts = np.asarray(args.get("path"), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise AttaError("bad_arg", "path must be (n>=2, 3)",
                        path=path + ".args.path",
                        fix="pass a polyline of [x,y,z] points in mm")
    if not np.isfinite(pts).all():
        raise AttaError("bad_arg", "path contains NaN or Inf",
                        path=path + ".args.path", fix="remove non-finite points")
    scales = args.get("scales")
    if scales is not None:
        sc = np.asarray(scales, dtype=np.float64).reshape(-1)
        if len(sc) not in (1, len(pts)):
            raise AttaError("bad_arg", "scales must have 1 or %d values" % len(pts),
                            path=path + ".args.scales",
                            fix="pass one scale per path point")
        if np.any(sc <= 0.0) or not np.isfinite(sc).all():
            raise AttaError("bad_arg", "scales must be finite and positive",
                            path=path + ".args.scales", fix="use scales >= 0.001")
        scales = (sc[0] if len(sc) == 1 else sc).tolist() if len(sc) == 1 else sc.tolist()
    return {
        "section_xy": prof * scale,
        "path_": pts * scale,
        "closed": bool(args.get("closed", False)),
        "twist_deg": float(args.get("twist_deg", 0.0)),
        "roll_deg": float(args.get("roll_deg", 0.0)),
        "caps": bool(args.get("caps", True)),
        "scales": scales,
        "up_hint": tuple(float(v) for v in (args.get("up_hint") or (0, 0, 1))),
    }


def _pipe_op(*, path_, radius, inner_radius=None, closed=False, segments=24,
             caps=True):
    from .. import sweepkit
    out = sweepkit.pipe(path_, radius, inner_radius=inner_radius,
                        closed=bool(closed), segments=int(segments),
                        caps=bool(caps))
    out.meta.update({"op": "pipe", "radius": float(radius)})
    return out


def _pipe_prepare(args, scale, path):
    pts = np.asarray(args.get("path"), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise AttaError("bad_arg", "path must be (n>=2, 3)",
                        path=path + ".args.path",
                        fix="pass a polyline of [x,y,z] points in mm")
    r = float(args.get("radius"))
    if not math.isfinite(r) or r <= 0.0:
        raise AttaError("bad_arg", "radius must be positive",
                        path=path + ".args.radius", fix="use radius > 0 mm")
    inner = args.get("inner_radius")
    inner = float(inner) if inner is not None else None
    if inner is not None and not (0.0 < inner < r):
        raise AttaError("bad_arg", "inner_radius must be positive and < radius",
                        path=path + ".args.inner_radius",
                        fix="use inner_radius < radius")
    seg = int(round(float(args.get("segments", 24))))
    if seg < 3:
        raise AttaError("bad_arg", "segments must be >= 3",
                        path=path + ".args.segments", fix="use segments: 24")
    return {"path_": pts * scale, "radius": r * scale,
            "inner_radius": (inner * scale if inner else None),
            "closed": bool(args.get("closed", False)), "segments": seg,
            "caps": bool(args.get("caps", True))}


# ---------------------------------------------------------------- loft

def _loft_op(*, sections, closed_ring=True, caps=True, align=True):
    from .. import sweepkit
    out = sweepkit.loft_sections(sections, closed_ring=bool(closed_ring),
                                 caps=bool(caps), align=bool(align))
    out.meta.update({"op": "loft2", "sections": len(sections)})
    return out


def _loft_prepare(args, scale, path):
    raw = args.get("sections")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise AttaError("bad_arg", "sections must hold at least two rings",
                        path=path + ".args.sections",
                        fix="pass two or more closed rings")
    rings = []
    for i, s in enumerate(raw):
        arr = np.asarray(s, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] not in (2, 3) or len(arr) < 3:
            raise AttaError("bad_arg", "section %d must be (n>=3, 2 or 3)" % i,
                            path=path + ".args.sections[%d]" % i,
                            fix="pass a closed ring of points in mm")
        if not np.isfinite(arr).all():
            raise AttaError("bad_arg", "section %d contains NaN or Inf" % i,
                            path=path + ".args.sections[%d]" % i,
                            fix="remove non-finite coordinates")
        rings.append((arr * scale).tolist())
    return {"sections": rings,
            "closed_ring": bool(args.get("closed_ring", True)),
            "caps": bool(args.get("caps", True)),
            "align": bool(args.get("align", True))}


# ---------------------------------------------------------------- sdf blend

def _extent_of_primitives(primitives, half):
    """Half-extent that contains every primitive, never smaller than ``half``."""
    out = np.asarray(half, dtype=np.float64).copy()
    for prim in primitives or []:
        if not isinstance(prim, dict):
            continue
        c = np.asarray(prim.get("center_mm") or (0.0, 0.0, 0.0), dtype=np.float64)
        kind = str(prim.get("kind") or "sphere").lower()
        if kind == "sphere":
            r = float(prim.get("radius_mm", 0.0))
            ext = np.abs(c) + r
        elif kind == "box":
            s = np.asarray(prim.get("size_mm") or (0.0, 0.0, 0.0),
                           dtype=np.float64) * 0.5
            ext = np.abs(c) + s + float(prim.get("round_mm", 0.0))
        elif kind == "cylinder":
            r = float(prim.get("radius_mm", 0.0))
            h = float(prim.get("height_mm", 0.0)) * 0.5
            axis = str(prim.get("axis", "z"))
            ext = np.abs(c) + np.array([r, r, r])
            ext[{"x": 0, "y": 1, "z": 2}.get(axis, 2)] += h
        elif kind == "cone":
            r = max(float(prim.get("r_bottom_mm", 0.0)),
                    float(prim.get("r_top_mm", 0.0)))
            h = float(prim.get("height_mm", 0.0)) * 0.5
            ext = np.abs(c) + np.array([r, r, h])
        elif kind == "torus":
            ext = np.abs(c) + np.full(3, float(prim.get("major_mm", 0.0))
                                      + float(prim.get("minor_mm", 0.0)))
        elif kind == "capsule":
            a = np.asarray(prim.get("a_mm") or (0, 0, 0), dtype=np.float64)
            b = np.asarray(prim.get("b_mm") or (0, 0, 0), dtype=np.float64)
            r = float(prim.get("radius_mm", 0.0))
            ext = np.maximum(np.abs(a), np.abs(b)) + r
        elif kind == "ellipsoid":
            ext = np.abs(c) + np.asarray(prim.get("radii_mm") or (0, 0, 0),
                                         dtype=np.float64)
        else:
            continue
        out = np.maximum(out, ext)
    return out


def _wall_between(primitives, cavity):
    """Smallest gap between the outer primitives and the cavity, in metres.

    Only the axis-aligned box case is measured exactly; for anything else the
    smallest half-extent difference is used as a conservative proxy.  This is
    what the sampling spacing has to resolve, not the overall part size.
    """
    best = None
    cav = cavity or {}
    csize = cav.get("size_mm")
    if csize is None:
        return None
    csize = np.asarray(csize, dtype=np.float64)
    for prim in primitives or []:
        if not isinstance(prim, dict):
            continue
        psize = prim.get("size_mm")
        if psize is None:
            continue
        psize = np.asarray(psize, dtype=np.float64)
        if psize.shape != csize.shape:
            continue
        gap = (psize - csize) * 0.5
        g = float(np.min(np.abs(gap)))
        if g > 0.0 and (best is None or g < best):
            best = g
    return best


def _fit_spacing(half, spacing, max_samples=1_200_000):
    """Coarsen ``spacing`` until the sample grid fits ``max_samples``.

    Marching tets holds the field plus several per-cell index arrays, so the
    real memory cost is roughly 50-80 bytes per sample *and* the extracted mesh
    can be several times larger than the grid.  1.2 M samples keeps the peak
    around 100 MB and the mesh near 200-400 k triangles.  There is no cheap
    decimation fallback (the pure-Python quadric decimator is far too slow at
    this size), so the sampling budget is the only real control.
    Returns the (possibly coarsened) spacing.
    """
    half = np.asarray(half, dtype=np.float64)
    margin = 2.5 * float(spacing)
    span = 2.0 * half + 2.0 * margin
    samples = float(np.prod(np.ceil(span / float(spacing)) + 1.0))
    if samples <= max_samples:
        return float(spacing)
    # Volume scales as spacing^-3, so one cube-root step is exact.
    factor = (samples / float(max_samples)) ** (1.0 / 3.0)
    return float(spacing) * factor


def _sdf_blend_op(*, size, primitives, blend_mm=0.0, resolution=48,
                  noise=None, warp=None, shell_mm=None, cavity=None):
    """Compose primitives as a signed-distance field and extract one surface.

    ``primitives`` is a list of ``{"kind": "sphere"|"box"|"cylinder"|"cone"|
    "torus"|"capsule"|"ellipsoid", ...}``.  ``blend_mm`` smooth-unions them, so
    overlapping forms fuse into one mass instead of showing a seam.
    """
    from .. import sdfkit

    size = np.asarray(size, dtype=np.float64).reshape(-1)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise AttaError("bad_arg", "size must be three positive lengths",
                        path="size", fix="pass [x,y,z] in mm")
    half = 0.5 * size
    # A primitive may legitimately reach past the nominal box (a mug handle
    # sticks out).  Grow the sampling box to the primitives' real extent plus a
    # margin, so the field is positive on the grid boundary and the extracted
    # surface is closed.  Without this the op fails with "field touches the
    # grid boundary" for any form whose parts exceed the declared size.
    half = _extent_of_primitives(primitives, half)
    spacing = float(min(size)) / float(max(8, min(160, int(resolution))))
    # A cavity leaves a wall; if the wall is thinner than ~2.5 samples the grid
    # can miss it entirely and the field comes back with no negative region at
    # all ("the field has no negative region").  Tighten the spacing to the
    # wall, not to the overall size.
    if isinstance(cavity, dict):
        wall = _wall_between(primitives, cavity)
        if wall and math.isfinite(wall) and wall > 0.0:
            # Three samples across the wall is the minimum that resolves it;
            # more than that only multiplies triangles (a 1 mm wall sampled at
            # 0.4 mm produced 1.4 M triangles for a mug).
            spacing = min(spacing, wall / 3.0)
    # Hard sample budget.  A long thin part (a sword is 1070x197x32 mm) sized
    # by its smallest axis produces tens of millions of samples, and marching
    # tets allocates several arrays per cell — that is an out-of-memory kill,
    # not a slow build.  Coarsen the spacing until the grid fits the budget;
    # the caller sees the effective spacing in the result meta.
    spacing = _fit_spacing(half, spacing)
    X, Y, Z, org = sdfkit.grid(-half, half, spacing)

    phi = None
    for i, prim in enumerate(primitives or []):
        d = _primitive_field(sdfkit, X, Y, Z, prim, i)
        if phi is None:
            phi = d
        elif float(blend_mm) > 0.0:
            phi = sdfkit.smooth_union(phi, d, float(blend_mm))
        else:
            phi = sdfkit.union(phi, d)

    if phi is None:
        raise AttaError("bad_arg", "primitives must not be empty",
                        path="primitives", fix="add at least one primitive")

    if noise:
        phi = sdfkit.displace(
            phi, X, Y, Z,
            amount=float(noise.get("amount_mm", 0.0)),
            frequency=float(noise.get("frequency", 1.0)) / 1000.0,
            seed=int(noise.get("seed", 0)),
            octaves=int(noise.get("octaves", 3)),
            kind=str(noise.get("kind", "fbm")))
    if warp:
        axis = str(warp.get("axis", "z"))
        if warp.get("twist_deg"):
            # Re-evaluate the composed field in twisted coordinates.
            phi = sdfkit.twist(lambda a, b, c: _compose(
                sdfkit, a, b, c, primitives, float(blend_mm)), X, Y, Z,
                angle_deg=float(warp["twist_deg"]), axis=axis)
    if shell_mm:
        phi = sdfkit.shell(phi, float(shell_mm))
    if cavity:
        # Subtract an inner volume to leave a real wall.  This is the honest
        # way to hollow a form: `shell_mm` hollows toward the surface and turns
        # a rounded box into a genus-heavy shell, while a cavity keeps the
        # outer silhouette and gives one open mouth.
        inner = _primitive_field(sdfkit, X, Y, Z, cavity, -1)
        phi = sdfkit.subtract(phi, inner)

    # The isosurface is only closed when the field is positive on the grid
    # boundary.  An open cavity carves through the top face on purpose, so the
    # check applies to the four side walls and the bottom, not the +Z face.
    open_top = isinstance(cavity, dict)
    sides = [phi[0, :, :], phi[-1, :, :], phi[:, 0, :], phi[:, -1, :],
             phi[:, :, 0]]
    if open_top:
        sides.append(phi[1:-1, 1:-1, -1])
    if not all(s.min() > 0 for s in sides):
        raise AttaError("sdf_open", "the field touches the grid boundary",
                        path="size",
                        fix="increase size or reduce the primitive extents")

    out = sdfkit.extract(phi, org, spacing)
    out.meta.update({"op": "sdf_blend", "primitives": len(primitives or []),
                     "blend_mm": float(blend_mm), "spacing_mm": spacing * 1000.0})
    return out


def _compose(sdfkit, X, Y, Z, primitives, blend):
    phi = None
    for i, prim in enumerate(primitives or []):
        d = _primitive_field(sdfkit, X, Y, Z, prim, i)
        if phi is None:
            phi = d
        elif blend > 0.0:
            phi = sdfkit.smooth_union(phi, d, blend)
        else:
            phi = sdfkit.union(phi, d)
    return phi


def _primitive_field(sdfkit, X, Y, Z, prim, i):
    if not isinstance(prim, dict):
        raise AttaError("bad_arg", "primitive %d must be an object" % i,
                        path="primitives[%d]" % i,
                        fix='use {"kind":"sphere","radius_mm":10}')
    kind = str(prim.get("kind") or "sphere").lower()
    center = tuple(float(v) for v in (prim.get("center_mm") or (0, 0, 0)))
    if kind == "sphere":
        return sdfkit.sphere(X, Y, Z, center=center,
                             radius=float(prim.get("radius_mm", 10.0)))
    if kind == "box":
        return sdfkit.box(X, Y, Z, size=tuple(float(v) for v in
                                              (prim.get("size_mm") or (20, 20, 20))),
                          center=center,
                          round_r=float(prim.get("round_mm", 0.0)))
    if kind == "cylinder":
        return sdfkit.cylinder(X, Y, Z, radius=float(prim.get("radius_mm", 10.0)),
                               height=float(prim.get("height_mm", 20.0)),
                               center=center, axis=str(prim.get("axis", "z")),
                               round_r=float(prim.get("round_mm", 0.0)))
    if kind == "cone":
        return sdfkit.cone(X, Y, Z,
                           r_bottom=float(prim.get("r_bottom_mm", 10.0)),
                           r_top=float(prim.get("r_top_mm", 0.0)),
                           height=float(prim.get("height_mm", 20.0)),
                           center=center)
    if kind == "torus":
        return sdfkit.torus(X, Y, Z, major=float(prim.get("major_mm", 15.0)),
                            minor=float(prim.get("minor_mm", 4.0)),
                            center=center, axis=str(prim.get("axis", "z")))
    if kind == "capsule":
        return sdfkit.capsule(X, Y, Z,
                              a=tuple(float(v) for v in
                                      (prim.get("a_mm") or (0, 0, -10))),
                              b=tuple(float(v) for v in
                                      (prim.get("b_mm") or (0, 0, 10))),
                              radius=float(prim.get("radius_mm", 4.0)))
    if kind == "ellipsoid":
        return sdfkit.ellipsoid(X, Y, Z,
                                radii=tuple(float(v) for v in
                                            (prim.get("radii_mm") or (10, 8, 6))),
                                center=center)
    raise AttaError("bad_arg", "unknown primitive kind %r" % kind,
                    path="primitives[%d].kind" % i,
                    fix="use sphere, box, cylinder, cone, torus, capsule or ellipsoid")


def _sdf_blend_prepare(args, scale, path):
    """Convert every mm length in the primitive list to metres."""
    size = np.asarray(args.get("size"), dtype=np.float64).reshape(-1)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise AttaError("bad_arg", "size must be three positive lengths",
                        path=path + ".args.size", fix="pass [x,y,z] in mm")
    prims = args.get("primitives")
    if not isinstance(prims, (list, tuple)) or not prims:
        raise AttaError("bad_arg", "primitives must be a non-empty array",
                        path=path + ".args.primitives",
                        fix='pass [{"kind":"sphere","radius_mm":10}]')
    out_prims = []
    for i, prim in enumerate(prims):
        if not isinstance(prim, dict):
            raise AttaError("bad_arg", "primitive %d must be an object" % i,
                            path=path + ".args.primitives[%d]" % i,
                            fix='use {"kind":"sphere","radius_mm":10}')
        out_prims.append(_scale_primitive(prim, scale, path, i))
    noise = args.get("noise")
    if noise is not None:
        if not isinstance(noise, dict):
            raise AttaError("bad_arg", "noise must be an object",
                            path=path + ".args.noise",
                            fix='use {"amount_mm":0.5,"frequency":20,"seed":1}')
        noise = dict(noise)
        noise["amount_mm"] = float(noise.get("amount_mm", 0.0)) * scale
    return {
        "size": size * scale,
        "primitives": out_prims,
        "blend_mm": float(args.get("blend_mm", 0.0)) * scale,
        "resolution": int(round(float(args.get("resolution", 48)))),
        "noise": noise,
        "warp": args.get("warp"),
        "shell_mm": (float(args["shell_mm"]) * scale
                     if args.get("shell_mm") else None),
        "cavity": (_scale_primitive(dict(args["cavity"]), scale, path, -1)
                   if isinstance(args.get("cavity"), dict) else None),
    }


_LENGTH_KEYS = ("radius_mm", "height_mm", "round_mm", "r_bottom_mm",
                "r_top_mm", "major_mm", "minor_mm")
_VEC_KEYS = ("center_mm", "size_mm", "radii_mm", "a_mm", "b_mm")


def _scale_primitive(prim, scale, path, i):
    out = dict(prim)
    for key in _LENGTH_KEYS:
        if out.get(key) is not None:
            out[key] = float(out[key]) * scale
    for key in _VEC_KEYS:
        if out.get(key) is not None:
            arr = np.asarray(out[key], dtype=np.float64).reshape(-1)
            if not np.isfinite(arr).all():
                raise AttaError("bad_arg", "%s contains NaN or Inf" % key,
                                path=path + ".args.primitives[%d].%s" % (i, key),
                                fix="pass finite coordinates")
            out[key] = (arr * scale).tolist()
    return out


# ---------------------------------------------------------------- repair

def _repair_op(*, mesh, fill=True, orient=True, smooth=0, min_faces=8):
    from .. import repair as _repair
    out, report = _repair.repair(mesh, fill=bool(fill), orient=bool(orient),
                                 smooth=int(smooth), min_faces=int(min_faces))
    out.meta.update({"op": "repair", "repair": report})
    return out


def _repair_prepare(args, scale, path):
    return {
        "fill": bool(args.get("fill", True)),
        "orient": bool(args.get("orient", True)),
        "smooth": int(round(float(args.get("smooth", 0)))),
        "min_faces": int(round(float(args.get("min_faces", 8)))),
    }


# ---------------------------------------------------------------- registration

def register():
    from ..features import register_op

    register_op(
        "subdivide",
        {"required": (), "arity": 1,
         "optional": ("scheme", "levels", "crease_angle_deg"),
         "doc": "Subdivision surface: scheme=catmull_clark|loop|simple, "
                "levels 1..5, crease_angle_deg keeps sharp edges sharp.",
         "args": {"scheme": "anything", "levels": ("int", 1),
                  "crease_angle_deg": "number"}},
        _subdivide_op,
    )
    _subdivide_op.prepare = _subdivide_prepare

    register_op(
        "sweep2",
        {"required": ("section_xy", "path"), "arity": 0,
         "optional": ("closed", "twist_deg", "roll_deg", "caps", "scales",
                      "up_hint"),
         "doc": "Sweep a 2D profile along a 3D path with rotation-minimising "
                "frames (no twist, no flip). section_xy mm, path mm.",
         "args": {"section_xy": "mat2", "path": "mat3", "closed": "bool",
                  "twist_deg": "number", "roll_deg": "number", "caps": "bool",
                  "scales": "anything", "up_hint": "vec3_free"}},
        _sweep_op,
    )
    _sweep_op.prepare = _sweep_prepare

    register_op(
        "pipe",
        {"required": ("path", "radius"), "arity": 0,
         "optional": ("inner_radius", "closed", "segments", "caps"),
         "doc": "Circular tube along a path; inner_radius makes it hollow. mm.",
         "args": {"path": "mat3", "radius": "length",
                  "inner_radius": "length", "closed": "bool",
                  "segments": ("int", 3), "caps": "bool"}},
        _pipe_op,
    )
    _pipe_op.prepare = _pipe_prepare

    register_op(
        "loft2",
        {"required": ("sections",), "arity": 0,
         "optional": ("closed_ring", "caps", "align"),
         "doc": "Loft through sections with correspondence matching, so a "
                "rotated section does not twist the result.",
         "args": {"sections": "anything", "closed_ring": "bool", "caps": "bool",
                  "align": "bool"}},
        _loft_op,
    )
    _loft_op.prepare = _loft_prepare

    register_op(
        "sdf_blend",
        {"required": ("size", "primitives"), "arity": 0,
         "optional": ("blend_mm", "resolution", "noise", "warp", "shell_mm",
                      "cavity"),
         "doc": "Compose SDF primitives with smooth blends, noise displacement, "
                "twist, shelling and an optional subtracted cavity; extract one "
                "watertight surface.",
         "args": {"size": "vec3", "primitives": "anything",
                  "blend_mm": "length", "resolution": ("int", 8),
                  "noise": "anything", "warp": "anything",
                  "shell_mm": "length", "cavity": "anything"}},
        _sdf_blend_op,
    )
    _sdf_blend_op.prepare = _sdf_blend_prepare

    register_op(
        "repair",
        {"required": (), "arity": 1,
         "optional": ("fill", "orient", "smooth", "min_faces"),
         "doc": "Weld, orient, cap holes and drop debris; returns a solid.",
         "args": {"fill": "bool", "orient": "bool", "smooth": ("int", 0),
                  "min_faces": ("int", 0)}},
        _repair_op,
    )
    _repair_op.prepare = _repair_prepare


try:
    register()
except Exception:  # pragma: no cover - a family must never break the engine
    pass
