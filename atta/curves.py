# -*- coding: utf-8 -*-
"""Polylines, Catmull-Rom sampling, arc-length resampling, transport frames.

Units are metres; no angle input here.  All functions return ``(n, k)``
float64 arrays with the same dimension as their input (2D or 3D).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .errors import AttaError

__all__ = [
    "polyline", "catmull_rom", "resample_arclength", "parallel_frames",
    "eval_sketch_segments",
]

_EPS = 1e-12


def _points(value, name="points"):
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] not in (2, 3) or len(p) < 2:
        raise AttaError("bad_arg", "%s must be an (n,2) or (n,3) array with n>=2" % name,
                        path=name, fix="pass a list of two or more coordinate pairs/triples")
    if not np.isfinite(p).all():
        raise AttaError("bad_arg", "%s contains NaN or Inf" % name, path=name,
                        fix="remove non-finite coordinates")
    return p


def _compress(p, closed=False):
    keep = [0]
    for i in range(1, len(p)):
        if np.linalg.norm(p[i] - p[keep[-1]]) > _EPS:
            keep.append(i)
    q = p[keep]
    if closed and len(q) > 1 and np.linalg.norm(q[0] - q[-1]) <= _EPS:
        q = q[:-1]
    return q


def polyline(points, closed=False) -> np.ndarray:
    """Return the points as an array; optionally append the first point."""
    p = _points(points)
    if closed and np.linalg.norm(p[0] - p[-1]) > _EPS:
        p = np.vstack([p, p[:1]])
    return np.ascontiguousarray(p, dtype=np.float64)


def catmull_rom(points, samples_per_seg=16, closed=False) -> np.ndarray:
    """Sample a Catmull-Rom spline through the points.

    A closed curve returns ``len(points) * samples_per_seg`` samples without
    a duplicated endpoint; an open curve returns ``(n-1)*spp + 1`` samples.
    """
    p = _compress(_points(points), closed=bool(closed))
    spp = int(samples_per_seg)
    if spp < 1:
        raise AttaError("bad_arg", "samples_per_seg must be >= 1", path="samples_per_seg",
                        fix="use samples_per_seg >= 1, e.g. 16")
    out = []
    if closed:
        if len(p) < 3:
            raise AttaError("bad_arg", "closed catmull_rom needs >= 3 distinct points",
                            path="points", fix="provide at least three distinct points")
        m = len(p)
        for i in range(m):
            p0, p1, p2, p3 = p[(i - 1) % m], p[i], p[(i + 1) % m], p[(i + 2) % m]
            for k in range(spp):
                t = k / float(spp)
                t2, t3 = t * t, t * t * t
                out.append(0.5 * ((2.0 * p1) + (-p0 + p2) * t
                                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3))
        return np.asarray(out, dtype=np.float64)
    if len(p) < 2:
        raise AttaError("bad_arg", "catmull_rom needs >= 2 distinct points", path="points",
                        fix="provide at least two distinct points")
    ext = np.vstack([p[:1], p, p[-1:]])
    for i in range(len(p) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for k in range(spp):
            t = k / float(spp)
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2.0 * p1) + (-p0 + p2) * t
                              + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                              + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3))
    out.append(p[-1])
    return np.asarray(out, dtype=np.float64)


def resample_arclength(points, n, closed=False) -> np.ndarray:
    """Resample a polyline to ``n`` points evenly spaced by arc length."""
    p = _compress(_points(points), closed=bool(closed))
    n = int(n)
    if n < 2:
        raise AttaError("bad_arg", "n must be >= 2", path="n", fix="use n >= 2")
    if len(p) < 2:
        raise AttaError("bad_arg", "resample needs >= 2 distinct points", path="points",
                        fix="provide at least two distinct points")
    q = np.vstack([p, p[:1]]) if closed else p
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= _EPS:
        raise AttaError("bad_arg", "polyline has zero length", path="points",
                        fix="provide points that are not all coincident")
    s = np.linspace(0.0, total, n, endpoint=not closed)
    out = np.empty((n, q.shape[1]), dtype=np.float64)
    for k in range(q.shape[1]):
        out[:, k] = np.interp(s, cum, q[:, k])
    return out


def parallel_frames(path, up=(0, 0, 1)):
    """Rotation-minimising frames along a 3D path.

    Returns ``(tangents, normals, binormals)``, each ``(n,3)`` and unit length,
    with ``binormal = tangent x normal``.  Consecutive duplicate points are an
    error because a tangent is undefined there.
    """
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2:
        raise AttaError("bad_arg", "path must be (n,3) with n>=2", path="path",
                        fix="pass a 3D polyline with at least two points")
    if not np.isfinite(p).all():
        raise AttaError("bad_arg", "path contains NaN or Inf", path="path",
                        fix="remove non-finite coordinates")
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    if np.any(seg <= _EPS):
        raise AttaError("bad_arg", "path has duplicate consecutive points", path="path",
                        fix="remove duplicate consecutive points before sweeping")
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), _EPS)

    ref = np.asarray(up, dtype=np.float64).reshape(3)
    n0 = ref - t[0] * float(ref @ t[0])
    if np.linalg.norm(n0) <= 1e-9:
        alt = np.array([1.0, 0.0, 0.0]) if abs(t[0, 0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        n0 = alt - t[0] * float(alt @ t[0])
    n = np.empty_like(p)
    n[0] = n0 / np.linalg.norm(n0)
    for i in range(1, len(p)):
        axis = np.cross(t[i - 1], t[i])
        sn = float(np.linalg.norm(axis))
        cs = float(np.clip(t[i - 1] @ t[i], -1.0, 1.0))
        v = n[i - 1]
        if sn < 1e-12:
            n[i] = v
        else:
            axis /= sn
            ang = np.arctan2(sn, cs)
            n[i] = (v * np.cos(ang) + np.cross(axis, v) * np.sin(ang)
                    + axis * float(axis @ v) * (1.0 - np.cos(ang)))
        n[i] -= t[i] * float(n[i] @ t[i])
        ln = np.linalg.norm(n[i])
        if ln <= 1e-9:
            raise AttaError("bad_arg", "transport frame collapsed at index %d" % i,
                            path="path", fix="avoid a path that folds back on itself")
        n[i] /= ln
    b = np.cross(t, n)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), _EPS)
    return t, n, b


def _xy(value, path):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise AttaError("bad_arg", "%s must be [x, y]" % path, path=path,
                        fix="pass a 2D coordinate [x, y]")
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError) as exc:
        raise AttaError("bad_arg", "%s must contain numeric [x, y]" % path,
                        path=path, fix="pass finite numbers") from exc
    if isinstance(value[0], bool) or isinstance(value[1], bool) or not (math.isfinite(x) and math.isfinite(y)):
        raise AttaError("bad_arg", "%s must contain finite numbers" % path,
                        path=path, fix="pass finite [x, y] coordinates")
    return np.array([x, y], dtype=np.float64)


def _arc_from_via(p0, via, p1, path):
    x1, y1 = float(p0[0]), float(p0[1])
    x2, y2 = float(via[0]), float(via[1])
    x3, y3 = float(p1[0]), float(p1[1])
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    scale = max(float(np.linalg.norm(p1 - p0)), float(np.linalg.norm(via - p0)), 1.0)
    if abs(d) <= 1e-9 * scale * scale:
        raise AttaError("bad_arg", "arc via point is collinear with start and end",
                        path=path, fix="pick a non-collinear via point on the arc")
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d
    center = np.array([ux, uy], dtype=np.float64)
    radius = float(np.linalg.norm(p0 - center))
    a0 = math.atan2(y1 - uy, x1 - ux)
    am = math.atan2(y2 - uy, x2 - ux)
    a1 = math.atan2(y3 - uy, x3 - ux)
    dm = (am - a0) % (2.0 * math.pi)
    d1 = (a1 - a0) % (2.0 * math.pi)
    if d1 <= 1e-12:
        raise AttaError("bad_arg", "arc start and end coincide", path=path,
                        fix="use op='circle' for a full circle")
    if dm < d1:
        sweep = d1
        cw = False
    else:
        sweep = d1 - 2.0 * math.pi
        cw = True
    return center, radius, a0, sweep, cw


def _arc_from_center(p0, p1, center, cw, path):
    r0 = float(np.linalg.norm(p0 - center))
    r1 = float(np.linalg.norm(p1 - center))
    if r0 <= 1e-9 or r1 <= 1e-9:
        raise AttaError("bad_arg", "arc radius must be positive", path=path,
                        fix="move center away from arc endpoints")
    if abs(r0 - r1) > max(1e-4, 1e-3 * max(r0, r1)):
        raise AttaError("bad_arg", "arc start and end are not equidistant from center (r0=%g, r1=%g)" % (r0, r1),
                        path=path, fix="adjust center or endpoint so ||to - center|| == ||from - center||")
    radius = 0.5 * (r0 + r1)
    a0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
    a1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
    if cw:
        sweep = -((a0 - a1) % (2.0 * math.pi))
        if abs(sweep) <= 1e-12:
            sweep = -2.0 * math.pi
    else:
        sweep = (a1 - a0) % (2.0 * math.pi)
        if abs(sweep) <= 1e-12:
            sweep = 2.0 * math.pi
    return radius, a0, sweep


def _arc_tangent(angle, cw):
    return np.array([math.sin(angle), -math.cos(angle)], dtype=np.float64) if cw else np.array(
        [-math.sin(angle), math.cos(angle)], dtype=np.float64)


def eval_sketch_segments(segments, samples_per_arc: int = 24) -> tuple[np.ndarray, list[dict]]:
    """Compile declarative 2D sketch segments into a closed ``(N, 2)`` loop and metadata.

    Supported segment dicts:
    - ``{"op": "start", "at": [x, y]}`` / ``{"op": "move", "to": [x, y]}``
    - ``{"op": "line", "to": [x, y], "tag": ...}``
    - ``{"op": "arc", "to": [x, y], "center": [cx, cy] | "via": [mx, my], "cw": bool, "tag": ...}``
    - ``{"op": "tangent_arc", "to": [x, y], "tag": ...}``
    - ``{"op": "circle", "center": [cx, cy], "radius": r, "tag": ...}``
    - ``{"op": "close", "tag": ...}``
    """
    if not isinstance(segments, (list, tuple)) or len(segments) == 0:
        raise AttaError("bad_arg", "segments must be a non-empty list of segment dicts",
                        path="segments", fix="pass a list of {op, ...} segment objects")
    if isinstance(samples_per_arc, bool):
        raise AttaError("bad_arg", "samples_per_arc must be an integer >= 2",
                        path="samples_per_arc", fix="use samples_per_arc=24")
    spp = int(samples_per_arc)
    if spp < 2:
        raise AttaError("bad_arg", "samples_per_arc must be >= 2",
                        path="samples_per_arc", fix="use samples_per_arc=24")

    # Handle standalone circle loop first.
    ops = [str(s.get("op", "")).lower() if isinstance(s, Mapping) else "" for s in segments]
    if "circle" in ops:
        if len(segments) != 1:
            raise AttaError("bad_arg", "circle segment must be the sole segment in a loop",
                            path="segments", fix="use a single circle segment per loop")
        seg = segments[0]
        if not isinstance(seg, Mapping):
            raise AttaError("bad_arg", "segment must be an object", path="segments[0]",
                            fix="pass a dict")
        center = _xy(seg.get("center"), "segments[0].center")
        r = seg.get("radius")
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(float(r)) or float(r) <= 0:
            raise AttaError("bad_arg", "circle radius must be a positive finite number",
                            path="segments[0].radius", fix="pass radius > 0")
        radius = float(r)
        n_circ = max(8, 2 * spp)
        theta = np.linspace(0.0, 2.0 * math.pi, n_circ, endpoint=False)
        pts = np.column_stack([center[0] + radius * np.cos(theta),
                               center[1] + radius * np.sin(theta)])
        tag = seg.get("tag")
        if tag is not None:
            tag = str(tag)
        p_start = [float(pts[0, 0]), float(pts[0, 1])]
        meta = [{
            "op": "circle",
            "tag": tag,
            "from": p_start,
            "to": p_start,
            "center": [float(center[0]), float(center[1])],
            "radius": radius,
            "point_range": [0, n_circ],
            "points": [[float(x), float(y)] for x, y in pts],
        }]
        return np.ascontiguousarray(pts, dtype=np.float64), meta

    cursor = None
    start_pt = None
    prev_tangent = None
    loop_pts: list[np.ndarray] = []
    meta: list[dict] = []
    closed_explicitly = False

    for idx, seg in enumerate(segments):
        spath = "segments[%d]" % idx
        if not isinstance(seg, Mapping):
            raise AttaError("bad_arg", "%s must be an object" % spath, path=spath,
                            fix="pass a segment dict like {'op': 'line', 'to': [x, y]}")
        op = str(seg.get("op") or "").lower()
        tag = seg.get("tag")
        if tag is not None:
            tag = str(tag)

        if op in ("start", "move"):
            if meta:
                raise AttaError("bad_arg", "%s can only appear at the beginning of a loop" % op,
                                path=spath, fix="place start/move before drawing segments")
            raw = seg.get("at") if "at" in seg else seg.get("to")
            cursor = _xy(raw, spath + (".at" if "at" in seg else ".to"))
            start_pt = cursor.copy()
            continue

        if closed_explicitly:
            raise AttaError("bad_arg", "no segments allowed after close", path=spath,
                            fix="put close at the end of the loop")

        if cursor is None:
            if "from" in seg:
                cursor = _xy(seg["from"], spath + ".from")
            else:
                cursor = np.array([0.0, 0.0], dtype=np.float64)
            start_pt = cursor.copy()
        elif "from" in seg:
            frm = _xy(seg["from"], spath + ".from")
            if np.linalg.norm(frm - cursor) > 1e-6:
                raise AttaError("bad_arg", "segment from %s does not match current cursor %s" % (frm.tolist(), cursor.tolist()),
                                path=spath + ".from", fix="omit 'from' or match the previous segment's 'to'")

        if op == "close":
            closed_explicitly = True
            if np.linalg.norm(cursor - start_pt) <= 1e-9:
                continue
            p0 = cursor.copy()
            p1 = start_pt.copy()
            d = p1 - p0
            if not loop_pts:
                loop_pts.append(p0.copy())
            start_idx = len(loop_pts) - 1
            meta.append({
                "op": "line",
                "tag": tag,
                "from": [float(p0[0]), float(p0[1])],
                "to": [float(p1[0]), float(p1[1])],
                "point_range": [start_idx, start_idx + 1],
                "points": [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]],
            })
            cursor = p1.copy()
            prev_tangent = d / float(np.linalg.norm(d))
            continue

        if op == "line":
            p0 = cursor.copy()
            p1 = _xy(seg.get("to"), spath + ".to")
            d = p1 - p0
            ln = float(np.linalg.norm(d))
            if ln <= 1e-12:
                raise AttaError("bad_arg", "line segment has zero length", path=spath,
                                fix="provide a distinct 'to' coordinate")
            if not loop_pts:
                loop_pts.append(p0.copy())
            start_idx = len(loop_pts) - 1
            closing_here = np.linalg.norm(p1 - start_pt) <= 1e-9 and idx == len(segments) - 1
            if not closing_here:
                loop_pts.append(p1.copy())
            end_idx = len(loop_pts) - 1 if not closing_here else 0
            meta.append({
                "op": "line",
                "tag": tag,
                "from": [float(p0[0]), float(p0[1])],
                "to": [float(p1[0]), float(p1[1])],
                "point_range": [start_idx, end_idx],
                "points": [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]],
            })
            cursor = p1.copy()
            prev_tangent = d / ln
            continue

        if op == "arc":
            p0 = cursor.copy()
            p1 = _xy(seg.get("to"), spath + ".to")
            has_center = seg.get("center") is not None
            has_via = seg.get("via") is not None
            if has_center == has_via:
                raise AttaError("bad_arg", "arc requires exactly one of 'center' or 'via'",
                                path=spath, fix="provide center=[cx,cy] or via=[mx,my]")
            if has_via:
                via = _xy(seg["via"], spath + ".via")
                center, radius, a0, sweep, cw = _arc_from_via(p0, via, p1, spath)
            else:
                center = _xy(seg["center"], spath + ".center")
                cw = bool(seg.get("cw", False))
                radius, a0, sweep = _arc_from_center(p0, p1, center, cw, spath)
                a_mid = a0 + 0.5 * sweep
                via = np.array([center[0] + radius * math.cos(a_mid),
                                center[1] + radius * math.sin(a_mid)], dtype=np.float64)

            if not loop_pts:
                loop_pts.append(p0.copy())
            start_idx = len(loop_pts) - 1
            closing_here = np.linalg.norm(p1 - start_pt) <= 1e-9 and idx == len(segments) - 1
            angles = np.linspace(a0, a0 + sweep, spp + 1)
            seg_pts = np.column_stack([center[0] + radius * np.cos(angles),
                                       center[1] + radius * np.sin(angles)])
            seg_pts[0] = p0
            seg_pts[-1] = p1
            interior = seg_pts[1:-1] if closing_here else seg_pts[1:]
            for pt in interior:
                loop_pts.append(pt.copy())
            end_idx = 0 if closing_here else len(loop_pts) - 1
            a1 = a0 + sweep
            meta.append({
                "op": "arc",
                "tag": tag,
                "from": [float(p0[0]), float(p0[1])],
                "to": [float(p1[0]), float(p1[1])],
                "center": [float(center[0]), float(center[1])],
                "radius": float(radius),
                "via": [float(via[0]), float(via[1])],
                "cw": bool(cw),
                "start_deg": math.degrees(a0),
                "end_deg": math.degrees(a1),
                "sweep_deg": math.degrees(sweep),
                "point_range": [start_idx, end_idx],
                "points": [[float(x), float(y)] for x, y in seg_pts],
            })
            cursor = p1.copy()
            prev_tangent = _arc_tangent(a1, cw)
            continue

        if op == "tangent_arc":
            if prev_tangent is None:
                raise AttaError("bad_arg", "tangent_arc requires a preceding segment for its entry tangent",
                                path=spath, fix="precede tangent_arc with a line or arc segment")
            p0 = cursor.copy()
            p1 = _xy(seg.get("to"), spath + ".to")
            d = p1 - p0
            ln = float(np.linalg.norm(d))
            if ln <= 1e-12:
                raise AttaError("bad_arg", "tangent_arc endpoint coincides with start",
                                path=spath, fix="provide a distinct 'to' coordinate")
            t0 = prev_tangent / float(np.linalg.norm(prev_tangent))
            n0 = np.array([-t0[1], t0[0]], dtype=np.float64)
            denom = float(np.dot(d, n0))
            if abs(denom) <= 1e-9 * ln:
                if float(np.dot(d, t0)) <= 0.0:
                    raise AttaError("bad_arg", "tangent_arc cannot reverse 180 degrees collinearly",
                                    path=spath, fix="use an explicit arc with center or via")
                # Collinear forward: degenerate straight segment.
                if not loop_pts:
                    loop_pts.append(p0.copy())
                start_idx = len(loop_pts) - 1
                closing_here = np.linalg.norm(p1 - start_pt) <= 1e-9 and idx == len(segments) - 1
                if not closing_here:
                    loop_pts.append(p1.copy())
                end_idx = 0 if closing_here else len(loop_pts) - 1
                meta.append({
                    "op": "line",
                    "tag": tag,
                    "from": [float(p0[0]), float(p0[1])],
                    "to": [float(p1[0]), float(p1[1])],
                    "point_range": [start_idx, end_idx],
                    "points": [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]],
                })
                cursor = p1.copy()
                prev_tangent = d / ln
                continue

            s = (ln * ln) / (2.0 * denom)
            center = p0 + s * n0
            cw = bool(s < 0.0)
            radius = abs(s)
            a0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
            a1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
            if cw:
                sweep = -((a0 - a1) % (2.0 * math.pi))
                if abs(sweep) <= 1e-12:
                    sweep = -2.0 * math.pi
            else:
                sweep = (a1 - a0) % (2.0 * math.pi)
                if abs(sweep) <= 1e-12:
                    sweep = 2.0 * math.pi
            a_mid = a0 + 0.5 * sweep
            via = np.array([center[0] + radius * math.cos(a_mid),
                            center[1] + radius * math.sin(a_mid)], dtype=np.float64)
            if not loop_pts:
                loop_pts.append(p0.copy())
            start_idx = len(loop_pts) - 1
            closing_here = np.linalg.norm(p1 - start_pt) <= 1e-9 and idx == len(segments) - 1
            angles = np.linspace(a0, a0 + sweep, spp + 1)
            seg_pts = np.column_stack([center[0] + radius * np.cos(angles),
                                       center[1] + radius * np.sin(angles)])
            seg_pts[0] = p0
            seg_pts[-1] = p1
            interior = seg_pts[1:-1] if closing_here else seg_pts[1:]
            for pt in interior:
                loop_pts.append(pt.copy())
            end_idx = 0 if closing_here else len(loop_pts) - 1
            a1_eff = a0 + sweep
            meta.append({
                "op": "tangent_arc",
                "tag": tag,
                "from": [float(p0[0]), float(p0[1])],
                "to": [float(p1[0]), float(p1[1])],
                "center": [float(center[0]), float(center[1])],
                "radius": float(radius),
                "via": [float(via[0]), float(via[1])],
                "cw": cw,
                "start_deg": math.degrees(a0),
                "end_deg": math.degrees(a1_eff),
                "sweep_deg": math.degrees(sweep),
                "point_range": [start_idx, end_idx],
                "points": [[float(x), float(y)] for x, y in seg_pts],
            })
            cursor = p1.copy()
            prev_tangent = _arc_tangent(a1_eff, cw)
            continue

        raise AttaError("bad_arg", "unknown sketch segment op %r" % op, path=spath + ".op",
                        fix="use line, arc, tangent_arc, circle, start, move, or close")

    if not meta or start_pt is None or cursor is None:
        raise AttaError("bad_arg", "segments produced no geometric edges", path="segments",
                        fix="add line/arc/circle segments")

    # Auto-close if not already closed back to start_pt.
    if np.linalg.norm(cursor - start_pt) > 1e-9:
        p0 = cursor.copy()
        p1 = start_pt.copy()
        start_idx = len(loop_pts) - 1
        meta.append({
            "op": "line",
            "tag": None,
            "implicit_close": True,
            "from": [float(p0[0]), float(p0[1])],
            "to": [float(p1[0]), float(p1[1])],
            "point_range": [start_idx, 0],
            "points": [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]],
        })

    arr = _compress(np.asarray(loop_pts, dtype=np.float64), closed=True)
    if len(arr) < 3:
        raise AttaError("bad_arg", "sketch loop must enclose >= 3 distinct points",
                        path="segments", fix="add more non-collinear segments")
    return np.ascontiguousarray(arr, dtype=np.float64), meta
