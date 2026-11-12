# -*- coding: utf-8 -*-
"""Sweeps, lofts and pipes with rotation-minimising frames.

v1.2's ``sweep`` used a fixed up-vector, so a path that turned more than 90°
flipped the profile and a closed loop twisted shut.  That is why a cable, a
handle or a hose could not be modelled: the primitive existed but the frame
maths did not.

This module supplies the missing maths:

``sweep_frames``
    Rotation-minimising frames by the **double reflection** method (Wang,
    Jüttler, Zheng & Liu 2008).  The frame is parallel-transported along the
    path, so the profile never spins and a closed path closes without twist.

``sweep_profile``
    Sweep an arbitrary 2D profile along a 3D path, with per-station scale,
    twist and roll.  Caps are generated for open paths.

``pipe``
    A circular tube along a path — the cable/hose/handle case, with optional
    inner radius for a hollow tube.

``loft_sections``
    Loft through sections with **correspondence matching**: sections are
    resampled to a common count, aligned by minimising the total squared
    distance over a cyclic shift, and oriented consistently.  That is what
    stops a loft from pinching into a bow-tie when two sections are rotated
    relative to each other.

All lengths are metres; the feature-graph boundary converts from mm.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import AttaError
from .mesh import Mesh

__all__ = [
    "sweep_frames",
    "sweep_profile",
    "pipe",
    "loft_sections",
    "resample_loop",
    "align_loops",
]


# ---------------------------------------------------------------- frames

def sweep_frames(path, *, closed=False, up_hint=(0.0, 0.0, 1.0)):
    """Rotation-minimising frames along ``path`` (double reflection).

    Returns ``(tangents, normals, binormals)``, each ``(n, 3)``.  The frame is
    parallel-transported, so there is no accumulated twist — the classic Frenet
    frame spins at inflection points and flips at straight segments, which is
    exactly what made v1.2's sweep unusable on real paths.
    """
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2:
        raise AttaError("bad_arg", "path must be (n>=2, 3)", path="path",
                        fix="pass a polyline of [x,y,z] points")
    if not np.isfinite(p).all():
        raise AttaError("bad_arg", "path contains NaN or Inf", path="path",
                        fix="remove non-finite coordinates")
    n = len(p)

    # Tangents: central differences, forward/backward at the ends.
    t = np.empty_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]
    if closed:
        t[0] = p[1] - p[-1]
        t[-1] = p[0] - p[-2]
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    if np.any(norm < 1e-15):
        raise AttaError("bad_arg", "path has a zero-length segment", path="path",
                        fix="remove duplicate consecutive points")
    t = t / norm

    # Initial normal: any vector perpendicular to t[0].
    hint = np.asarray(up_hint, dtype=np.float64)
    r = hint - t[0] * float(hint @ t[0])
    if np.linalg.norm(r) < 1e-9:
        alt = np.array([1.0, 0.0, 0.0])
        if abs(float(t[0] @ alt)) > 0.9:
            alt = np.array([0.0, 1.0, 0.0])
        r = alt - t[0] * float(alt @ t[0])
    r = r / np.linalg.norm(r)

    normals = np.empty_like(p)
    normals[0] = r
    for i in range(n - 1):
        # Double reflection (Wang et al. 2008): two reflections map frame i to
        # frame i+1 with the minimum rotation.
        v1 = p[i + 1] - p[i]
        c1 = float(v1 @ v1)
        if c1 < 1e-18:
            normals[i + 1] = normals[i]
            continue
        rL = normals[i] - (2.0 / c1) * float(v1 @ normals[i]) * v1
        tL = t[i] - (2.0 / c1) * float(v1 @ t[i]) * v1
        v2 = t[i + 1] - tL
        c2 = float(v2 @ v2)
        if c2 < 1e-18:
            nxt = rL
        else:
            nxt = rL - (2.0 / c2) * float(v2 @ rL) * v2
        ln = float(np.linalg.norm(nxt))
        normals[i + 1] = nxt / ln if ln > 1e-15 else normals[i]

    binormals = np.cross(t, normals)
    return t, normals, binormals


# ---------------------------------------------------------------- loops

def resample_loop(loop, count):
    """Resample a closed 2D loop to ``count`` points by arc length."""
    p = np.asarray(loop, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        raise AttaError("bad_arg", "loop must be (n>=3, 2)", path="loop",
                        fix="pass a closed 2D polygon")
    count = max(3, int(count))
    closed = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-15:
        raise AttaError("bad_arg", "loop has zero length", path="loop",
                        fix="give the loop a non-zero perimeter")
    cum = np.concatenate([[0.0], np.cumsum(seg)]) / total
    targets = np.linspace(0.0, 1.0, count, endpoint=False)
    out = np.empty((count, 2), dtype=np.float64)
    for i, u in enumerate(targets):
        k = int(np.searchsorted(cum, u, side="right") - 1)
        k = min(max(k, 0), len(seg) - 1)
        span = cum[k + 1] - cum[k]
        f = 0.0 if span <= 1e-15 else (u - cum[k]) / span
        out[i] = closed[k] + f * (closed[k + 1] - closed[k])
    return out


def align_loops(a, b):
    """Rotate ``b`` (cyclically) to best match ``a``; returns ``b`` reordered.

    Correspondence matters: two identical circles sampled from different start
    angles produce a twisted loft.  Minimising the sum of squared distances
    over all cyclic shifts is O(n²) and exact, which is fine for the section
    counts a person actually uses.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise AttaError("bad_arg", "align_loops needs equal point counts",
                        path="loop", fix="resample both loops to the same count")
    n = len(a)
    best_shift, best_cost = 0, np.inf
    for s in range(n):
        cost = float(np.sum((a - np.roll(b, s, axis=0)) ** 2))
        if cost < best_cost:
            best_cost, best_shift = cost, s
    shifted = np.roll(b, best_shift, axis=0)
    # Also try the reversed winding: a mirrored section must not twist.
    rev = shifted[::-1]
    if float(np.sum((a - rev) ** 2)) < float(np.sum((a - shifted) ** 2)):
        return rev
    return shifted


# ---------------------------------------------------------------- sweep

def sweep_profile(section_xy, path, *, closed=False, scales=None,
                  twist_deg=0.0, roll_deg=0.0, caps=True,
                  up_hint=(0.0, 0.0, 1.0), closed_section=True) -> Mesh:
    """Sweep a 2D profile along a 3D path with rotation-minimising frames.

    ``section_xy`` is the profile in the frame's (normal, binormal) plane.
    ``scales`` is an optional per-station scale (a list matching the path, or a
    single number).  ``twist_deg`` is the *total* twist from start to end;
    ``roll_deg`` rotates the profile inside its own plane.

    Open paths with ``caps=True`` produce a closed solid.
    """
    prof = np.asarray(section_xy, dtype=np.float64)
    if prof.ndim != 2 or prof.shape[1] != 2 or len(prof) < 3:
        raise AttaError("bad_arg", "section_xy must be (n>=3, 2)", path="section_xy",
                        fix="pass a closed 2D profile")
    if not np.isfinite(prof).all():
        raise AttaError("bad_arg", "section_xy contains NaN or Inf",
                        path="section_xy", fix="remove non-finite coordinates")
    p = np.asarray(path, dtype=np.float64)
    t, nrm, bnm = sweep_frames(p, closed=closed, up_hint=up_hint)
    m = len(p)
    k = len(prof)

    if scales is None:
        sc = np.ones(m)
    elif np.isscalar(scales):
        sc = np.full(m, float(scales))
    else:
        sc = np.asarray(scales, dtype=np.float64).reshape(-1)
        if len(sc) != m:
            raise AttaError("bad_arg", "scales must have one value per path point",
                            path="scales", fix="pass %d values" % m)
    if np.any(sc <= 0.0) or not np.isfinite(sc).all():
        raise AttaError("bad_arg", "scales must be finite and positive",
                        path="scales", fix="use scales >= 0.001")

    total_twist = math.radians(float(twist_deg))
    roll = math.radians(float(roll_deg))

    verts = np.empty((m * k, 3), dtype=np.float64)
    for i in range(m):
        # Section frame at station i: (normal, binormal), rotated by twist+roll.
        ang = (total_twist * (i / float(m - 1)) if m > 1 else 0.0) + roll
        ca, sa = math.cos(ang), math.sin(ang)
        e1 = nrm[i] * ca + bnm[i] * sa
        e2 = -nrm[i] * sa + bnm[i] * ca
        base = p[i]
        local = base[None, :] + (prof[:, 0][:, None] * e1[None, :]
                                 + prof[:, 1][:, None] * e2[None, :]) * sc[i]
        verts[i * k:(i + 1) * k] = local

    faces = []
    for i in range(m - 1):
        for j in range(k):
            j2 = (j + 1) % k if closed_section else j + 1
            if not closed_section and j2 >= k:
                continue
            a = i * k + j
            b = i * k + j2
            c = (i + 1) * k + j2
            d = (i + 1) * k + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    if closed and m > 2:
        i = m - 1
        for j in range(k):
            j2 = (j + 1) % k
            faces.append((i * k + j, i * k + j2, j2))
            faces.append((i * k + j, j2, j))
    faces = np.asarray(faces, dtype=np.int64)

    if caps and not closed and closed_section:
        # Cap the two ends with a fan from the section centroid.
        out_verts = [verts]
        extra = []
        base_idx = len(verts)
        for end, sign in ((0, -1.0), (m - 1, 1.0)):
            ang = (total_twist * (end / float(m - 1)) if m > 1 else 0.0) + roll
            ca, sa = math.cos(ang), math.sin(ang)
            e1 = nrm[end] * ca + bnm[end] * sa
            e2 = -nrm[end] * sa + bnm[end] * ca
            c = p[end] + (prof.mean(axis=0)[0] * e1 + prof.mean(axis=0)[1] * e2) * sc[end]
            extra.append(c)
        cap_verts = np.asarray(extra, dtype=np.float64)
        all_verts = np.vstack([verts, cap_verts])
        cap_faces = []
        for j in range(k):
            j2 = (j + 1) % k
            cap_faces.append((base_idx, j2, j))
            cap_faces.append((base_idx + 1, (m - 1) * k + j, (m - 1) * k + j2))
        faces = np.vstack([faces, np.asarray(cap_faces, dtype=np.int64)])
        return _finish_sweep(all_verts, faces, mesh_meta(closed, m, k))

    return _finish_sweep(verts, faces, mesh_meta(closed, m, k))


def mesh_meta(closed, stations, section_pts):
    return {"sweep": {"closed_path": bool(closed), "stations": int(stations),
                      "section_points": int(section_pts)}}


def _finish_sweep(verts, faces, meta):
    from .mesh import oriented, weld
    m = Mesh(verts, faces, dict(meta))
    m = weld(m, 1e-12)
    return oriented(m)


# ---------------------------------------------------------------- pipe

def pipe(path, radius, *, inner_radius=None, closed=False, segments=24,
         caps=True, up_hint=(0.0, 0.0, 1.0)) -> Mesh:
    """A circular tube along ``path``.

    ``inner_radius`` makes it hollow (a real tube with a wall), which is the
    difference between a cable and a pipe.
    """
    r = float(radius)
    if not math.isfinite(r) or r <= 0.0:
        raise AttaError("bad_arg", "radius must be positive", path="radius",
                        fix="use radius > 0")
    segments = max(3, int(segments))
    ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    outer = np.column_stack([r * np.cos(ang), r * np.sin(ang)])

    if inner_radius is None:
        return sweep_profile(outer, path, closed=closed, caps=caps,
                             up_hint=up_hint)

    ri = float(inner_radius)
    if not math.isfinite(ri) or ri <= 0.0 or ri >= r:
        raise AttaError("bad_arg", "inner_radius must be positive and < radius",
                        path="inner_radius", fix="use inner_radius < radius")
    # Hollow tube: outer wall, inner wall (reversed), and annular caps.
    p = np.asarray(path, dtype=np.float64)
    t, nrm, bnm = sweep_frames(p, closed=closed, up_hint=up_hint)
    m = len(p)
    k = segments
    verts = np.empty((m * k * 2, 3), dtype=np.float64)
    for i in range(m):
        e1, e2 = nrm[i], bnm[i]
        for j in range(k):
            ca, sa = math.cos(ang[j]), math.sin(ang[j])
            d = e1 * ca + e2 * sa
            verts[i * k + j] = p[i] + d * r
            verts[m * k + i * k + j] = p[i] + d * ri
    faces = []
    for i in range(m - 1 if not closed else m):
        i2 = (i + 1) % m
        for j in range(k):
            j2 = (j + 1) % k
            a, b = i * k + j, i * k + j2
            c, d = i2 * k + j2, i2 * k + j
            faces.extend([(a, b, c), (a, c, d)])
            # The inner wall faces the other way: its outward normal points at
            # the tube axis, so its triangles are wound in reverse.  Without
            # this the two walls disagree on 2*segments edges and the solid is
            # not closed.
            a2, b2 = m * k + i * k + j, m * k + i * k + j2
            c2, d2 = m * k + i2 * k + j2, m * k + i2 * k + j
            faces.extend([(a2, c2, b2), (a2, d2, c2)])
    if caps and not closed:
        # Annular caps: the outer ring and the inner ring, bridged.  The two
        # ends need opposite winding so the whole surface stays consistently
        # outward-facing; getting this backwards leaves 2*segments edges with
        # inconsistent winding and `closed` reports False.
        for i, start in ((0, True), (m - 1, False)):
            for j in range(k):
                j2 = (j + 1) % k
                o1, o2 = i * k + j, i * k + j2
                n1, n2 = m * k + i * k + j, m * k + i * k + j2
                if start:
                    # The start annulus faces -Z, so wind it the other way.
                    faces.extend([(o1, n2, o2), (o1, n1, n2)])
                else:
                    faces.extend([(o1, o2, n2), (o1, n2, n1)])
    return _finish_sweep(verts, np.asarray(faces, dtype=np.int64),
                         {"pipe": {"hollow": True, "stations": m,
                                   "segments": k}})


# ---------------------------------------------------------------- loft

def loft_sections(sections, *, closed_ring=True, caps=True,
                  resample=None, align=True) -> Mesh:
    """Loft through 2D or 3D sections with correspondence matching.

    Sections may be:

    * ``(n, 2)`` — a planar loop, placed at ``z = index`` is *not* assumed;
      pass 3D sections if you need placement;
    * ``(n, 3)`` — a 3D ring.

    ``resample`` forces every section to the same point count (default: the
    max count present).  ``align`` rotates each section to minimise twist.
    """
    if not isinstance(sections, (list, tuple)) or len(sections) < 2:
        raise AttaError("bad_arg", "loft needs at least two sections",
                        path="sections", fix="pass two or more rings")
    rings = []
    for i, s in enumerate(sections):
        arr = np.asarray(s, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] not in (2, 3) or len(arr) < 3:
            raise AttaError("bad_arg", "section %d must be (n>=3, 2 or 3)" % i,
                            path="sections[%d]" % i,
                            fix="pass a closed ring of points")
        if not np.isfinite(arr).all():
            raise AttaError("bad_arg", "section %d contains NaN or Inf" % i,
                            path="sections[%d]" % i,
                            fix="remove non-finite coordinates")
        if arr.shape[1] == 2:
            arr = np.column_stack([arr, np.full(len(arr), float(i))])
        rings.append(arr)

    count = int(resample) if resample else max(len(r) for r in rings)
    count = max(3, count)
    prepared = []
    for r in rings:
        if len(r) == count:
            prepared.append(r)
        else:
            # Resample in 3D by arc length, keeping the section's own shape.
            prepared.append(_resample_ring_3d(r, count))

    if align:
        for i in range(1, len(prepared)):
            # Align in the section's own plane: project to 2D, shift, reapply.
            a2 = _project_2d(prepared[i - 1])
            b2 = _project_2d(prepared[i])
            shifted = align_loops(a2, b2)
            prepared[i] = _apply_2d_shift(prepared[i], b2, shifted)

    k = count
    m = len(prepared)
    verts = np.vstack(prepared)
    faces = []
    for i in range(m - 1):
        for j in range(k):
            j2 = (j + 1) % k if closed_ring else j + 1
            if not closed_ring and j2 >= k:
                continue
            a = i * k + j
            b = i * k + j2
            c = (i + 1) * k + j2
            d = (i + 1) * k + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    faces = np.asarray(faces, dtype=np.int64)

    if caps and closed_ring:
        all_verts = np.vstack([verts, prepared[0].mean(axis=0)[None, :],
                               prepared[-1].mean(axis=0)[None, :]])
        base = len(verts)
        cap = []
        for j in range(k):
            j2 = (j + 1) % k
            cap.append((base, j2, j))
            cap.append((base + 1, (m - 1) * k + j, (m - 1) * k + j2))
        faces = np.vstack([faces, np.asarray(cap, dtype=np.int64)])
        verts = all_verts
    return _finish_sweep(verts, faces, {"loft": {"sections": m,
                                                 "section_points": k}})


def _resample_ring_3d(ring, count):
    """Resample a closed 3D ring to ``count`` points by arc length."""
    closed = np.vstack([ring, ring[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-15:
        raise AttaError("bad_arg", "a loft section has zero perimeter",
                        path="sections", fix="give every section a non-zero size")
    cum = np.concatenate([[0.0], np.cumsum(seg)]) / total
    targets = np.linspace(0.0, 1.0, count, endpoint=False)
    out = np.empty((count, 3), dtype=np.float64)
    for i, u in enumerate(targets):
        kk = int(np.searchsorted(cum, u, side="right") - 1)
        kk = min(max(kk, 0), len(seg) - 1)
        span = cum[kk + 1] - cum[kk]
        f = 0.0 if span <= 1e-15 else (u - cum[kk]) / span
        out[i] = closed[kk] + f * (closed[kk + 1] - closed[kk])
    return out


def _project_2d(ring):
    """Project a 3D ring to its best-fit plane (PCA), for shift matching."""
    c = ring.mean(axis=0)
    q = ring - c
    # Two dominant principal axes
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    return q @ vt[:2].T


def _apply_2d_shift(ring, projected, target):
    """Reorder ``ring`` so its projection matches ``target``.

    ``target`` is a permutation of ``projected``'s rows; recover the index map
    by nearest-neighbour matching (exact when the rows are distinct).
    """
    idx = []
    used = np.zeros(len(projected), dtype=bool)
    for row in target:
        d = np.linalg.norm(projected - row[None, :], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        used[j] = True
        idx.append(j)
    return ring[np.asarray(idx, dtype=np.int64)]
