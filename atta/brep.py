# -*- coding: utf-8 -*-
"""Optional B-rep / STEP backend behind the same Spec v3 graph.

Atta's primary kernel is a numpy mesh kernel because it is fast, dependency-free
and good enough for draft iteration and measured verification.  A mesh is not a
B-rep: it has no analytic faces, so it cannot carry a STEP hand-off to a CAD
system or a CAM tool.  This module adds that second backend for the subset of
feature graphs that map cleanly onto OpenCASCADE via ``build123d``.

The contract is deliberately narrow and honest:

* the mesh build and :func:`atta.engine.verify` still decide the verdict;
* STEP is a *delivery* format, produced from the same mm-valued spec;
* an unsupported op fails with ``brep_unsupported`` and a concrete ``fix`` —
  it never silently falls back to a tessellated STEP, which would look like a
  B-rep while carrying only triangles.

Supported graph ops: box, cylinder, sphere, cone, torus, revolve, extrude,
union, difference, intersection, transform, mirror, pattern_linear,
pattern_polar, fillet, chamfer. Graphs with edge finishing use OCCT for their
whole reachable closure: the mesh is a tessellation of the same analytic
construction exported to STEP. Other graphs retain the optional STEP backend.
Unsupported/open ancestors and successors fail closed.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping

from .errors import AttaError

__all__ = [
    "available", "build_graph", "mesh_graph", "tessellate", "export_step",
    "select_edges", "get_face_tags",
]

SUPPORTED = frozenset({
    "box", "cylinder", "sphere", "cone", "torus", "revolve", "extrude",
    "union", "difference", "intersection", "transform", "mirror",
    "pattern_linear", "pattern_polar", "fillet", "chamfer",
})

CAD_OPS = frozenset({"fillet", "chamfer"})
_UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}
MAX_TRIANGLES = 500_000



def available() -> bool:
    try:
        import build123d  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _require():
    try:
        import build123d as b
    except Exception as exc:  # noqa: BLE001
        raise AttaError(
            "brep_unavailable",
            "the B-rep backend needs build123d/OCCT, which is not importable: %s" % exc,
            path="backend",
            fix="pip install build123d, or deliver glb/obj/stl instead",
        ) from exc
    return b


def _vec(value, default=(0.0, 0.0, 0.0), path="args"):
    raw = default if value is None else value
    if raw is None:
        raise AttaError("bad_arg", "%s is required" % path, path=path,
                        fix="use [x,y,z] in mm")
    try:
        out = [float(v) for v in raw]
    except (TypeError, ValueError) as exc:
        raise AttaError("bad_arg", "%s must be a number or 3-vector" % path,
                        path=path, fix="use [x,y,z] in mm") from exc
    if len(out) == 1:
        out = out * 3
    if len(out) != 3 or not all(math.isfinite(v) for v in out):
        raise AttaError("bad_arg", "%s must be three finite numbers" % path,
                        path=path, fix="use [x,y,z] in mm")
    return out


def _axis(name):
    raw = str(name or "z").lower()
    if raw in ("x", "y", "z"):
        return raw
    raise AttaError("brep_unsupported", "axis %r is not supported by the B-rep backend" % raw,
                    path="axis", fix="use axis='x', 'y' or 'z'")


def _rotation(b, axis, center=(0.0, 0.0, 0.0)):
    """Place a Z-axis primitive on x/y/z through ``center``."""
    axis = _axis(axis)
    if axis == "z":
        return b.Pos(*center)
    if axis == "x":
        return b.Pos(*center) * b.Rot(Y=90)
    return b.Pos(*center) * b.Rot(X=-90)


def get_face_tags(shape) -> dict | None:
    """Return the face-tag registry attached to a ``build123d`` shape, if any."""
    return getattr(shape, "_atta_face_tags", None)


def _face_key(face):
    from OCP.TopAbs import TopAbs_FORWARD
    from OCP.TopoDS import TopoDS
    cast_face = getattr(TopoDS, "Face_s", None) or TopoDS.Face
    return face.wrapped.TShape(), hash(cast_face(face.wrapped.Oriented(TopAbs_FORWARD)))


def _dedupe_faces(b, faces):
    out, seen = [], set()
    for f in faces:
        ff = f if isinstance(f, b.Face) else b.Face(f)
        if not any(ff.wrapped.IsSame(prev.wrapped) for prev in out):
            out.append(ff)
    return out


def _merged_tags(feat, path="tags"):
    raw = {}
    for src in (feat.get("tags"), (feat.get("args") or {}).get("tags")):
        if src is None:
            continue
        if not isinstance(src, Mapping):
            raise AttaError("bad_arg", "tags must be an object mapping role -> tag name",
                            path=path, fix="use e.g. {'end': 'cap', 'start': 'base'}")
        for k, v in src.items():
            if not isinstance(k, str) or not k or not isinstance(v, str) or not v:
                raise AttaError("bad_arg", "tags entries must map non-empty string -> non-empty string",
                                path=path, fix="use e.g. {'end': 'cap'}")
            raw[k] = v
    return raw


def _tag_faces_by_role(b, shape, roles_map, tags_dict, extra_valid_roles=()):
    """Map semantic role keys in ``tags_dict`` to matching faces."""
    out_tags = {}
    valid = set(roles_map) | set(extra_valid_roles)
    for role, tag_name in tags_dict.items():
        if role not in valid:
            raise AttaError(
                "bad_arg",
                "unknown face role %r in tags" % role,
                path="tags",
                fix="valid roles: %s" % (", ".join(sorted(valid)) or "none"),
            )
        matched = roles_map.get(role) or []
        if matched:
            out_tags.setdefault(tag_name, []).extend(matched)
    return {k: _dedupe_faces(b, v) for k, v in out_tags.items() if v}


def _axis_unit(axis):
    ax = _axis(axis)
    return {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[ax]


def _primitive_roles(op, shape, args):
    faces = list(shape.faces())
    if op == "box":
        roles = {
            "top": [], "+z": [], "end": [],
            "bottom": [], "-z": [], "start": [],
            "front": [], "-y": [],
            "back": [], "+y": [],
            "left": [], "-x": [],
            "right": [], "+x": [],
            "side": [], "sides": [], "wall": [], "walls": [],
            "all": list(faces),
        }
        for f in faces:
            nx, ny, nz = tuple(f.normal_at())
            if nz > 0.9:
                for k in ("top", "+z", "end"):
                    roles[k].append(f)
            elif nz < -0.9:
                for k in ("bottom", "-z", "start"):
                    roles[k].append(f)
            elif ny < -0.9:
                for k in ("front", "-y", "side", "sides", "wall", "walls"):
                    roles[k].append(f)
            elif ny > 0.9:
                for k in ("back", "+y", "side", "sides", "wall", "walls"):
                    roles[k].append(f)
            elif nx < -0.9:
                for k in ("left", "-x", "side", "sides", "wall", "walls"):
                    roles[k].append(f)
            elif nx > 0.9:
                for k in ("right", "+x", "side", "sides", "wall", "walls"):
                    roles[k].append(f)
        return roles
    if op in ("cylinder", "cone"):
        ax = _axis_unit(args.get("axis", "z"))
        roles = {
            "start": [], "bottom": [], "-axis": [],
            "end": [], "top": [], "cap": [], "+axis": [],
            "side": [], "wall": [], "walls": [], "lateral": [],
            "all": list(faces),
        }
        for f in faces:
            if f.geom_type.name == "PLANE":
                n = tuple(f.normal_at())
                dot = n[0] * ax[0] + n[1] * ax[1] + n[2] * ax[2]
                if dot > 0.9:
                    for k in ("end", "top", "cap", "+axis"):
                        roles[k].append(f)
                elif dot < -0.9:
                    for k in ("start", "bottom", "-axis"):
                        roles[k].append(f)
            else:
                for k in ("side", "wall", "walls", "lateral"):
                    roles[k].append(f)
        return roles
    return {"side": list(faces), "surface": list(faces), "all": list(faces)}


def _primitive(b, feat):
    op = feat["op"]
    args = feat.get("args") or {}
    center = _vec(args.get("center"), (0, 0, 0), "args.center")
    if op == "box":
        size = _vec(args.get("size"), None, "args.size")
        if any(v <= 0 for v in size):
            raise AttaError("bad_arg", "box.size must be positive",
                            path="args.size", fix="pass [x,y,z] in mm")
        align = str(args.get("align", "center")).lower()
        if align in ("min", "corner"):
            center = [c + s * 0.5 for c, s in zip(center, size)]
        elif align != "center":
            raise AttaError("bad_arg", "box.align must be center or min",
                            path="args.align", fix="use align='center' or 'min'")
        return b.Pos(*center) * b.Box(*size)
    if op == "cylinder":
        r = float(args.get("radius"))
        h = float(args.get("height"))
        if r <= 0 or h <= 0:
            raise AttaError("bad_arg", "cylinder radius/height must be positive",
                            path="args", fix="use radius > 0 and height > 0")
        return _rotation(b, args.get("axis", "z"), center) * b.Cylinder(r, h)
    if op == "sphere":
        r = float(args.get("radius"))
        if r <= 0:
            raise AttaError("bad_arg", "sphere.radius must be positive",
                            path="args.radius", fix="use radius > 0")
        return b.Pos(*center) * b.Sphere(r)
    if op == "cone":
        rb = float(args.get("r_bottom"))
        rt = float(args.get("r_top", 0.0))
        h = float(args.get("height"))
        if h <= 0 or rb < 0 or rt < 0 or (rb <= 0 and rt <= 0):
            raise AttaError("bad_arg", "cone needs positive height and a radius",
                            path="args", fix="use r_bottom, r_top, height in mm")
        return _rotation(b, args.get("axis", "z"), center) * b.Cone(rb, rt, h)
    if op == "torus":
        major = float(args.get("major_radius"))
        minor = float(args.get("minor_radius"))
        if major <= 0 or minor <= 0 or minor >= major:
            raise AttaError("bad_arg", "torus needs 0 < minor_radius < major_radius",
                            path="args", fix="use major_radius > minor_radius > 0")
        return _rotation(b, args.get("axis", "z"), center) * b.Torus(major, minor)
    raise AttaError("brep_unsupported", "op %r is not a B-rep primitive" % op,
                    path="op", fix="use a supported primitive")


def _segments_wire(b, segments, plane=None):
    """Compile 2D sketch segments into analytic OCCT edges + face-tag specs."""
    from .curves import eval_sketch_segments
    _, seg_meta = eval_sketch_segments(segments)
    edges = []
    seg_records = []
    for seg in seg_meta:
        sop = seg["op"]
        if sop == "circle":
            cx, cy = seg["center"]
            edge = b.Pos(cx, cy, 0.0) * b.Edge.make_circle(seg["radius"])
        elif sop == "line":
            x0, y0 = seg["from"]
            x1, y1 = seg["to"]
            edge = b.Edge.make_line((x0, y0, 0.0), (x1, y1, 0.0))
        else:
            sweep = abs(float(seg.get("sweep_deg", 0.0)))
            if abs(sweep - 360.0) < 1e-6:
                cx, cy = seg["center"]
                edge = b.Pos(cx, cy, 0.0) * b.Edge.make_circle(seg["radius"])
            else:
                x0, y0 = seg["from"]
                xm, ym = seg["via"]
                x1, y1 = seg["to"]
                edge = b.Edge.make_three_point_arc((x0, y0, 0.0), (xm, ym, 0.0), (x1, y1, 0.0))
        local_mid = edge.position_at(0.5)
        if plane is not None:
            edge = plane * edge
        edges.append(edge)
        seg_records.append((edge, seg.get("tag"), local_mid))
    wire = b.Wire(edges)
    return b.make_face(wire), seg_records


def _propagate_partner(b, old_shape, new_shape, tag_map, loc=None):
    """Propagate tags across a rigid location change using OCCT IsPartner + transformed center."""
    new_faces = list(new_shape.faces())
    trsf = None
    if loc is not None:
        raw_loc = loc.location if hasattr(loc, "location") else loc
        trsf = raw_loc.wrapped.Transformation()
    from OCP.gp import gp_Pnt
    old_to_new = {}
    for f_old in old_shape.faces():
        partners = [f_new for f_new in new_faces if f_new.wrapped.IsPartner(f_old.wrapped)]
        if trsf is not None and len(partners) > 1:
            c0 = f_old.center()
            ct = gp_Pnt(float(c0.X), float(c0.Y), float(c0.Z)).Transformed(trsf)
            best = min(
                partners,
                key=lambda fn: (
                    (float(fn.center().X) - ct.X()) ** 2
                    + (float(fn.center().Y) - ct.Y()) ** 2
                    + (float(fn.center().Z) - ct.Z()) ** 2
                ),
            )
            old_to_new[f_old] = [best]
        elif partners:
            old_to_new[f_old] = partners
    return _remap_tag_map(b, old_shape, old_to_new, tag_map)


def _remap_tag_map(b, old_shape, old_to_new, tag_map):
    def map_list(faces):
        mapped = []
        for f in faces:
            for k_old, vals in old_to_new.items():
                if f.wrapped.IsSame(k_old.wrapped):
                    mapped.extend(vals)
                    break
        return _dedupe_faces(b, mapped)

    out_tags = {}
    for tname, flist in (tag_map.get("tags") or {}).items():
        res = map_list(flist)
        if res:
            out_tags[tname] = res
    out_feats = {}
    for fid, flist in (tag_map.get("features") or {}).items():
        res = map_list(flist)
        if res:
            out_feats[fid] = res
    return {"tags": out_tags, "features": out_feats}


def _merge_tag_maps(b, maps):
    tags = {}
    feats = {}
    for m in maps:
        for k, v in (m.get("tags") or {}).items():
            tags.setdefault(k, []).extend(v)
        for k, v in (m.get("features") or {}).items():
            feats.setdefault(k, []).extend(v)
    return {
        "tags": {k: _dedupe_faces(b, v) for k, v in tags.items() if v},
        "features": {k: _dedupe_faces(b, v) for k, v in feats.items() if v},
    }


def _propagate_history(b, inputs_with_tags, result_shape):
    """Propagate face tags across OCCT boolean/fillet/chamfer using ShapeHistory."""
    history = getattr(result_shape, "_history", None)
    res_faces = list(result_shape.faces())
    mapped_list = []
    for in_shape, in_tags in inputs_with_tags:
        old_to_new = {}
        for f_old in in_shape.faces():
            surviving = [rf for rf in res_faces if rf.wrapped.IsSame(f_old.wrapped)]
            if not surviving and history is not None:
                mods = history.modified(f_old.wrapped)
                if mods:
                    surviving = [
                        rf for rf in res_faces
                        if any(rf.wrapped.IsSame(m) for m in mods)
                    ]
                elif not history.is_removed(f_old.wrapped):
                    # Check geometric match if cleaned/rebuilt without explicit modified entry.
                    c0 = tuple(f_old.center())
                    n0 = tuple(f_old.normal_at())
                    g0 = f_old.geom_type.name
                    a0 = float(f_old.area)
                    for rf in res_faces:
                        if rf.geom_type.name == g0 and abs(float(rf.area) - a0) <= max(1e-4, 1e-3 * a0):
                            c1 = tuple(rf.center())
                            n1 = tuple(rf.normal_at())
                            dist = math.sqrt(sum((a - c) ** 2 for a, c in zip(c0, c1)))
                            dot = abs(n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2])
                            if dist <= 1e-4 and dot > 0.999:
                                surviving.append(rf)
            if surviving:
                old_to_new[f_old] = surviving
        mapped_list.append(_remap_tag_map(b, in_shape, old_to_new, in_tags))
    return _merge_tag_maps(b, mapped_list)


def _propagate_transform(b, old_shape, new_shape, tag_map, pt_fn, dir_fn, scale=1.0):
    """Propagate face tags across transform/mirror/pattern via IsPartner or geometry."""
    new_faces = list(new_shape.faces())
    diag = max(float(new_shape.bounding_box().diagonal), 1.0)
    tol = max(1e-3, 1e-3 * diag)
    area_scale = float(scale) * float(scale)
    old_to_new = {}
    for f_old in old_shape.faces():
        if abs(area_scale - 1.0) <= 1e-9:
            partners = [fn for fn in new_faces if fn.wrapped.IsPartner(f_old.wrapped)]
            if len(partners) == 1:
                old_to_new[f_old] = partners
                continue
        c_tgt = pt_fn(tuple(f_old.center()))
        n_tgt = dir_fn(tuple(f_old.normal_at()))
        g_old = f_old.geom_type.name
        a_tgt = float(f_old.area) * area_scale
        best, best_dist = None, float("inf")
        for f_new in new_faces:
            if f_new.geom_type.name != g_old:
                continue
            if abs(float(f_new.area) - a_tgt) > max(1e-3, 5e-3 * max(a_tgt, 1.0)):
                continue
            c1 = tuple(f_new.center())
            n1 = tuple(f_new.normal_at())
            dot = abs(n_tgt[0] * n1[0] + n_tgt[1] * n1[1] + n_tgt[2] * n1[2])
            if dot < 0.98:
                continue
            dist = math.sqrt(sum((a - c) ** 2 for a, c in zip(c_tgt, c1)))
            if dist <= tol and dist < best_dist:
                best = f_new
                best_dist = dist
        if best is None:
            # Fallback using transformed bounding box center for periodic seams (cylinder/torus).
            bb_old = f_old.bounding_box()
            bc_old = tuple(0.5 * (lo + hi) for lo, hi in zip(tuple(bb_old.min), tuple(bb_old.max)))
            bc_tgt = pt_fn(bc_old)
            for f_new in new_faces:
                if f_new.geom_type.name != g_old:
                    continue
                if abs(float(f_new.area) - a_tgt) > max(1e-3, 5e-3 * max(a_tgt, 1.0)):
                    continue
                bb_new = f_new.bounding_box()
                bc_new = tuple(0.5 * (lo + hi) for lo, hi in zip(tuple(bb_new.min), tuple(bb_new.max)))
                dist = math.sqrt(sum((a - c) ** 2 for a, c in zip(bc_tgt, bc_new)))
                if dist <= tol and dist < best_dist:
                    best = f_new
                    best_dist = dist
        if best is not None:
            old_to_new[f_old] = [best]
    return _remap_tag_map(b, old_shape, old_to_new, tag_map)


def _revolve(b, args, tags_dict=None):
    raw_segs = args.get("segments")
    seg_records = []
    if isinstance(raw_segs, (list, tuple)):
        face, seg_records = _segments_wire(b, raw_segs, plane=b.Plane.XZ)
        bb = face.bounding_box()
        if bb.min.X < -1e-6:
            raise AttaError("bad_arg", "revolve radii must be non-negative",
                            path="args.segments", fix="use r >= 0 in [r, z] sketch coordinates")
    else:
        profile = args.get("profile_rz")
        if not isinstance(profile, (list, tuple)) or len(profile) < 2:
            raise AttaError("bad_arg", "revolve.profile_rz needs >= 2 [r,z] points",
                            path="args.profile_rz", fix="pass [[r,z], ...] in mm")
        pts = []
        for row in profile:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise AttaError("bad_arg", "revolve points must be [r,z]",
                                path="args.profile_rz", fix="pass [[r,z], ...] in mm")
            point = (float(row[0]), float(row[1]))
            if point[0] < 0:
                raise AttaError("bad_arg", "revolve radii must be non-negative",
                                path="args.profile_rz", fix="use radii >= 0")
            if not pts or pts[-1] != point:
                pts.append(point)
        if pts[0] != pts[-1]:
            # Match surfaces.revolve: an open profile with caps closes to the
            # axis with end discs, not by drawing a slanted line to its start.
            if pts[0][0] != 0:
                pts.insert(0, (0.0, pts[0][1]))
            if pts[-1][0] != 0:
                pts.append((0.0, pts[-1][1]))
        face = b.make_face(b.Plane.XZ * b.Polyline(*pts, close=True))
    body = b.revolve(face, axis=b.Axis.Z, revolution_arc=360).clean()

    faces = list(body.faces())
    z_vals = [float(f.center().Z) for f in faces if f.geom_type.name == "PLANE"]
    z_min = min(z_vals) if z_vals else None
    z_max = max(z_vals) if z_vals else None
    start_faces, end_faces, side_faces = [], [], []
    for f in faces:
        n = tuple(f.normal_at())
        if f.geom_type.name == "PLANE" and abs(n[2]) > 0.9:
            cz = float(f.center().Z)
            if n[2] < -0.9 and z_min is not None and abs(cz - z_min) < 1e-4:
                start_faces.append(f)
                continue
            if n[2] > 0.9 and z_max is not None and abs(cz - z_max) < 1e-4:
                end_faces.append(f)
                continue
        side_faces.append(f)
    roles = {
        "start": start_faces, "bottom": start_faces, "base": start_faces,
        "end": end_faces, "top": end_faces, "cap": end_faces,
        "side": side_faces or faces, "wall": side_faces or faces, "walls": side_faces or faces,
        "all": list(faces),
    }
    seg_tags = {}
    for _, stag, mid in seg_records:
        if stag and float(mid.X) > 1e-5:
            p = b.Vertex(float(mid.X), 0.0, float(mid.Y))
            matched = [f for f in faces if f.distance(p) < 1e-4]
            if matched:
                seg_tags.setdefault(stag, []).extend(matched)
                roles.setdefault(stag, []).extend(matched)
    tag_map = {"tags": {**seg_tags, **_tag_faces_by_role(b, body, roles, tags_dict or {}, seg_tags)},
               "features": {}}
    rot_loc = _rotation(b, args.get("axis", "z"))
    rotated = rot_loc * body
    tag_map = _propagate_partner(b, body, rotated, tag_map, loc=rot_loc)
    return rotated, tag_map


def _extrude(b, args, tags_dict=None):
    raw_segs = args.get("segments")
    seg_records = []
    if raw_segs is not None:
        face, seg_records = _segments_wire(b, raw_segs)
    else:
        loop = args.get("loop_xy")
        if not isinstance(loop, (list, tuple)) or len(loop) < 3:
            raise AttaError("bad_arg", "extrude.loop_xy needs >= 3 points",
                            path="args.loop_xy", fix="pass [[x,y], ...] in mm")
        pts = []
        for row in loop:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise AttaError("bad_arg", "extrude points must be [x,y]",
                                path="args.loop_xy", fix="pass [[x,y], ...] in mm")
            pts.append((float(row[0]), float(row[1])))
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        face = b.make_face(b.Polyline(*pts, close=True))
    height = float(args.get("height"))
    if height <= 0:
        raise AttaError("bad_arg", "extrude.height must be positive",
                        path="args.height", fix="use height in mm")
    body = b.extrude(face, amount=height)

    faces = list(body.faces())
    start_faces, end_faces, side_faces = [], [], []
    for f in faces:
        n = tuple(f.normal_at())
        if f.geom_type.name == "PLANE" and n[2] < -0.9 and abs(float(f.center().Z)) < 1e-4:
            start_faces.append(f)
        elif f.geom_type.name == "PLANE" and n[2] > 0.9 and abs(float(f.center().Z) - height) < 1e-4:
            end_faces.append(f)
        else:
            side_faces.append(f)
    roles = {
        "start": start_faces, "bottom": start_faces, "base": start_faces,
        "end": end_faces, "top": end_faces, "cap": end_faces,
        "side": side_faces, "wall": side_faces, "walls": side_faces,
        "all": list(faces),
    }
    seg_tags = {}
    for _, stag, mid in seg_records:
        if stag:
            p = b.Vertex(float(mid.X), float(mid.Y), height * 0.5)
            matched = [f for f in side_faces if f.distance(p) < 1e-4]
            if matched:
                seg_tags.setdefault(stag, []).extend(matched)
                roles.setdefault(stag, []).extend(matched)
    tag_map = {"tags": {**seg_tags, **_tag_faces_by_role(b, body, roles, tags_dict or {}, seg_tags)},
               "features": {}}

    direction = _vec(args.get("direction"), (0, 0, 1), "args.direction")
    # Match surfaces._direction_frame: loop_xy is expressed in a plane
    # perpendicular to direction, not sheared from the global XY plane.
    from .surfaces import _direction_frame
    direction, x_dir, _ = _direction_frame(direction)
    plane = b.Plane(origin=(0, 0, 0), x_dir=tuple(x_dir), z_dir=tuple(direction))
    holes = args.get("holes_xy") or []
    if holes:
        for hole in holes:
            hpts = [(float(p[0]), float(p[1])) for p in hole]
            if hpts[0] != hpts[-1]:
                hpts.append(hpts[0])
            hface = b.make_face(b.Polyline(*hpts, close=True))
            # Over-long cutter: coplanar end faces are the classic boolean
            # failure mode, so the cutter starts below and ends above the body.
            cutter = b.extrude(hface, amount=height * 1.2)
            prev_body = body
            body = body - (b.Pos(0, 0, -0.1 * height) * cutter)
            tag_map = _propagate_history(b, [(prev_body, tag_map)], body)
    placed = plane * body
    tag_map = _propagate_partner(b, body, placed, tag_map, loc=plane)
    return placed, tag_map


def _transform_fns(args):
    translate = _vec(args.get("translate"), (0, 0, 0), "args.translate")
    rot = args.get("rotate_deg") or [0.0, 0.0, 0.0]
    rot = [float(v) for v in rot]
    pivot = _vec(args.get("pivot"), (0, 0, 0), "args.pivot")
    scale = args.get("scale", 1.0)
    if isinstance(scale, (list, tuple)):
        scale = float(scale[0])
    scale = float(scale)
    rx, ry, rz = (math.radians(v) for v in rot)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    def rot_vec(v):
        x, y, z = v
        # X, then Y, then Z (matches b.Rot(Z=rz) * b.Rot(Y=ry) * b.Rot(X=rx))
        y, z = y * cx - z * sx, y * sx + z * cx
        x, z = x * cy + z * sy, -x * sy + z * cy
        x, y = x * cz - y * sz, x * sz + y * cz
        return (x, y, z)

    def pt_fn(p):
        dx = (p[0] - pivot[0]) * scale
        dy = (p[1] - pivot[1]) * scale
        dz = (p[2] - pivot[2]) * scale
        rx_p, ry_p, rz_p = rot_vec((dx, dy, dz))
        return (rx_p + pivot[0] + translate[0],
                ry_p + pivot[1] + translate[1],
                rz_p + pivot[2] + translate[2])

    return pt_fn, rot_vec, scale


def _transform(b, shape, args):
    translate = _vec(args.get("translate"), (0, 0, 0), "args.translate")
    rot = args.get("rotate_deg") or [0.0, 0.0, 0.0]
    rot = [float(v) for v in rot]
    if len(rot) != 3:
        raise AttaError("bad_arg", "transform.rotate_deg must be [rx,ry,rz]",
                        path="args.rotate_deg", fix="use [rx,ry,rz] in degrees")
    pivot = _vec(args.get("pivot"), (0, 0, 0), "args.pivot")
    scale = args.get("scale", 1.0)
    if isinstance(scale, (list, tuple)):
        s = [float(v) for v in scale]
        if len(s) != 3 or len(set(round(v, 12) for v in s)) != 1:
            raise AttaError(
                "brep_unsupported",
                "the B-rep backend supports uniform scale only",
                path="args.scale", fix="use a single scale factor, or the mesh backend")
        scale = s[0]
    shape = b.Pos(-pivot[0], -pivot[1], -pivot[2]) * shape
    scale = _positive(scale, "args.scale")
    if abs(float(scale) - 1.0) > 1e-12:
        shape = shape.scale(float(scale), about=(0, 0, 0))
    if any(abs(v) > 1e-12 for v in rot):
        shape = b.Rot(Z=rot[2]) * b.Rot(Y=rot[1]) * b.Rot(X=rot[0]) * shape
    shape = b.Pos(pivot[0], pivot[1], pivot[2]) * shape
    return b.Pos(*translate) * shape


def _pattern(b, shape, feat, in_tags=None):
    args = feat.get("args") or {}
    op = feat["op"]
    count = int(args.get("count", 1))
    if count < 1:
        raise AttaError("bad_arg", "pattern count must be >= 1",
                        path="args.count", fix="use count >= 2")
    copies = []
    copy_tags = []
    if op == "pattern_linear":
        spacing = float(args.get("spacing"))
        axis = _axis(args.get("axis", "x"))
        unit = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
        for i in range(count):
            dx, dy, dz = (spacing * i * v for v in unit)
            loc = b.Pos(dx, dy, dz)
            cshape = loc * shape
            copies.append(cshape)
            if in_tags is not None:
                copy_tags.append(_propagate_partner(b, shape, cshape, in_tags, loc=loc))
    else:
        angle = float(args.get("angle_deg", 360.0))
        center = _vec(args.get("center"), (0, 0, 0), "args.center")
        axis = _axis(args.get("axis", "z"))
        step = angle / count if abs(abs(angle) - 360.0) <= 1e-9 else (
            angle / (count - 1) if count > 1 else 0.0)
        for i in range(count):
            rot = {"x": b.Rot(X=step * i), "y": b.Rot(Y=step * i),
                   "z": b.Rot(Z=step * i)}[axis]
            loc = b.Pos(*center) * rot * b.Pos(-center[0], -center[1], -center[2])
            cshape = loc * shape
            copies.append(cshape)
            if in_tags is not None:
                copy_tags.append(_propagate_partner(b, shape, cshape, in_tags, loc=loc))
    out = copies[0]
    out_tags = copy_tags[0] if copy_tags else {"tags": {}, "features": {}}
    for idx, copy in enumerate(copies[1:], start=1):
        prev = out
        out = out + copy
        if in_tags is not None:
            out_tags = _propagate_history(b, [(prev, out_tags), (copy, copy_tags[idx])], out)
    if in_tags is not None:
        return out, out_tags
    return out


def _positive(value, path):
    if isinstance(value, bool):
        raise AttaError("bad_arg", "expected a positive finite number", path=path,
                        fix="supply a positive numeric dimension")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AttaError("bad_arg", "expected a positive finite number", path=path,
                        fix="supply a positive numeric dimension") from exc
    if not math.isfinite(result) or result <= 0:
        raise AttaError("bad_arg", "expected a positive finite number", path=path,
                        fix="supply a positive numeric dimension")
    return result


def _integer(value, path, low, high):
    result = _positive(value, path)
    if result != int(result) or not low <= result <= high:
        raise AttaError("bad_arg", "expected integer in [%d,%d]" % (low, high),
                        path=path, fix="use an integer within the documented budget")
    return int(result)


def _scale_segments(segments, factor):
    out = []
    for seg in segments:
        if not isinstance(seg, Mapping):
            out.append(seg)
            continue
        item = dict(seg)
        for key in ("to", "from", "at", "center", "via"):
            if key in item and isinstance(item[key], (list, tuple)) and len(item[key]) == 2:
                item[key] = [float(item[key][0]) * factor, float(item[key][1]) * factor]
        if "radius" in item and isinstance(item["radius"], (int, float)) and not isinstance(item["radius"], bool):
            item["radius"] = float(item["radius"]) * factor
        out.append(item)
    return out


def _to_mm(op, args, factor):
    """Scale only declared dimensions, never angles, directions or parameters."""
    from .features import arg_kinds, _num
    def times(value):
        if isinstance(value, (list, tuple)):
            return [times(v) for v in value]
        return _num(value, "dimension", "args") * factor
    kinds = arg_kinds(op)
    out = {}
    for k, v in args.items():
        if k == "segments" and op in ("extrude", "revolve") and isinstance(v, (list, tuple)):
            out[k] = _scale_segments(v, factor)
        elif kinds.get(k) in ("length", "vec3", "mat2", "mat3", "loops", "rings", "curves", "curves4"):
            out[k] = times(v)
        else:
            out[k] = v
    if op in CAD_OPS and isinstance(out.get("edges"), Mapping):
        selector = dict(out["edges"])
        if isinstance(selector.get("at"), Mapping):
            at = dict(selector["at"])
            if "value" in at and not isinstance(at["value"], str):
                at["value"] = times(at["value"])
            if "tolerance" in at:
                at["tolerance"] = times(at["tolerance"])
            selector["at"] = at
        out["edges"] = selector
    return out


def _valid_solid(shape, path):
    from OCP.BRep import BRep_Tool
    if shape is None or not math.isfinite(float(shape.volume)) or shape.volume <= 0:
        raise AttaError("brep_empty", "operation produced no positive solid volume",
                        path=path, fix="check dimensions and boolean operands")
    solids = shape.solids()
    solid_faces = {face for solid in solids for face in solid.faces()}
    solid_edges = {edge for solid in solids for edge in solid.edges()}
    solid_vertices = {vertex for solid in solids for vertex in solid.vertices()}
    # is_manifold in build123d flags sphere poles as non-manifold. Use the
    # OCCT closed-shell check, shape validity and explicit solid coverage.
    if (not solids or not shape.is_valid or set(shape.faces()) != solid_faces
            or set(shape.edges()) != solid_edges or set(shape.vertices()) != solid_vertices
            or any(not BRep_Tool.IsClosed_s(shell.wrapped)
                   for solid in solids for shell in solid.shells())):
        raise AttaError("brep_invalid", "expected only valid, closed OCCT solids",
                        path=path, fix="remove open shells/invalid ancestors; no mesh fallback exists")


def _edge_signature(edge):
    bb = edge.bounding_box()
    return {"kind": edge.geom_type.name.lower(), "length_mm": round(float(edge.length), 10),
            "bbox_min_mm": [round(v, 10) for v in tuple(bb.min)],
            "bbox_max_mm": [round(v, 10) for v in tuple(bb.max)],
            "center_mm": [round(v, 10) for v in tuple(edge.center())]}


def _forward_edge_key(b, edge):
    from OCP.TopAbs import TopAbs_FORWARD
    from OCP.TopoDS import TopoDS
    cast_edge = getattr(TopoDS, "Edge_s", None) or TopoDS.Edge
    return b.Edge(cast_edge(edge.wrapped.Oriented(TopAbs_FORWARD)))


def _face_edge_keys(b, faces):
    keys = set()
    for f in faces:
        ff = f if isinstance(f, b.Face) else b.Face(f)
        for e in ff.edges():
            keys.add(_forward_edge_key(b, e))
    return keys


def _resolve_faces_from_tags(b, face_tags, key, *, kind="tag"):
    if not isinstance(face_tags, Mapping):
        return []
    if kind == "feature":
        feats = face_tags.get("features")
        if isinstance(feats, Mapping):
            raw = feats.get(key) or []
        else:
            raw = face_tags.get("feature:" + key) or face_tags.get(key) or []
    else:
        tags = face_tags.get("tags")
        if isinstance(tags, Mapping):
            raw = tags.get(key) or []
        else:
            raw = face_tags.get(key) or []
    return [f if isinstance(f, b.Face) else b.Face(f) for f in raw]


def select_edges(shape, selector, face_tags=None):
    """Geometric and topological set selection in the current world frame; never edge indices."""
    import json
    b = _require()
    if face_tags is None:
        face_tags = get_face_tags(shape)
    cond_keys = {"parallel", "kind", "at", "common_edge_of", "on_face", "from_feature"}
    allowed_keys = cond_keys | {"expect_count"}
    if selector == "all":
        options = {}
    elif isinstance(selector, Mapping) and selector:
        options = dict(selector)
        unknown = set(options) - allowed_keys
        if unknown or not (set(options) & cond_keys):
            raise AttaError("bad_selector", "selector needs parallel/kind/at/common_edge_of/on_face/from_feature, without unknown keys",
                            path="args.edges", fix="use 'all' or {parallel:'z', expect_count:4}")
    else:
        raise AttaError("bad_selector", "edges must be 'all' or a geometric selector object",
                        path="args.edges", fix="numeric edge indices are intentionally unsupported")
    for key, value in options.items():
        if value is None:
            raise AttaError("bad_selector", "selector condition %s must not be null" % key,
                            path="args.edges." + key,
                            fix="remove the condition or supply its documented value; use 'all' explicitly")
    parallel = options.get("parallel")
    kind = options.get("kind")
    if parallel is not None and parallel not in ("x", "y", "z"):
        raise AttaError("bad_selector", "parallel must be x/y/z", path="args.edges.parallel",
                        fix="select a principal world direction; only straight edges match")
    if kind is not None and kind not in ("line", "circle"):
        raise AttaError("bad_selector", "kind must be line or circle", path="args.edges.kind",
                        fix="use kind='line' or kind='circle'")
    at = options.get("at")
    if at is not None:
        if (not isinstance(at, Mapping) or set(at) - {"axis", "value", "tolerance"}
                or not {"axis", "value"} <= set(at) or at["axis"] not in ("x", "y", "z")):
            raise AttaError("bad_selector", "at needs axis=x/y/z and value=number|min|max",
                            path="args.edges.at", fix="use {axis:'z',value:'max',tolerance:0.000001}")
        axis = ("x", "y", "z").index(at["axis"])
        value = at["value"]
        if isinstance(value, str) and value in ("min", "max"):
            target = tuple(getattr(shape.bounding_box(), value))[axis]
        else:
            try:
                if isinstance(value, (bool, str)):
                    raise ValueError()
                target = float(value)
                if not math.isfinite(target):
                    raise ValueError()
            except (ValueError, TypeError) as exc:
                raise AttaError("bad_selector", "at.value must be finite or min/max",
                                path="args.edges.at.value", fix="supply a coordinate in model units") from exc
        tolerance = _positive(at.get("tolerance", 1e-6), "args.edges.at.tolerance")

    on_face_keys = None
    if "on_face" in options:
        tag = options["on_face"]
        if not isinstance(tag, str) or not tag:
            raise AttaError("bad_selector", "on_face must be a non-empty tag string",
                            path="args.edges.on_face", fix="use on_face='cap'")
        on_face_keys = _face_edge_keys(b, _resolve_faces_from_tags(b, face_tags, tag, kind="tag"))

    common_keys = None
    if "common_edge_of" in options:
        pair = options["common_edge_of"]
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(t, str) and t for t in pair)):
            raise AttaError("bad_selector", "common_edge_of must be [tagA, tagB]",
                            path="args.edges.common_edge_of", fix="use common_edge_of=['cap', 'side']")
        tag_a, tag_b = pair
        faces_a = _resolve_faces_from_tags(b, face_tags, tag_a, kind="tag")
        if tag_a == tag_b:
            counts = {}
            for f in faces_a:
                for ek in {_forward_edge_key(b, e) for e in f.edges()}:
                    counts[ek] = counts.get(ek, 0) + 1
            common_keys = {ek for ek, n in counts.items() if n >= 2}
        else:
            faces_b = _resolve_faces_from_tags(b, face_tags, tag_b, kind="tag")
            common_keys = _face_edge_keys(b, faces_a) & _face_edge_keys(b, faces_b)

    feat_keys = None
    if "from_feature" in options:
        fid = options["from_feature"]
        if not isinstance(fid, str) or not fid:
            raise AttaError("bad_selector", "from_feature must be a non-empty feature id string",
                            path="args.edges.from_feature", fix="use from_feature='<feature_id>'")
        feat_keys = _face_edge_keys(b, _resolve_faces_from_tags(b, face_tags, fid, kind="feature"))

    selected = []
    for edge in shape.edges():
        edge_kind = edge.geom_type.name.lower()
        if kind is not None and kind != edge_kind:
            continue
        if parallel is not None:
            if edge_kind != "line":
                continue
            tangent = tuple(edge.tangent_at(0.5))
            if abs(abs(tangent[("x", "y", "z").index(parallel)]) - 1.0) > 1e-10:
                continue
        if at is not None:
            bb = edge.bounding_box()
            # Whole edge lies on the coordinate plane, not just its midpoint.
            if max(abs(tuple(bb.min)[axis] - target), abs(tuple(bb.max)[axis] - target)) > tolerance:
                continue
        if on_face_keys is not None or common_keys is not None or feat_keys is not None:
            ek = _forward_edge_key(b, edge)
            if on_face_keys is not None and ek not in on_face_keys:
                continue
            if common_keys is not None and ek not in common_keys:
                continue
            if feat_keys is not None and ek not in feat_keys:
                continue
        selected.append(edge)
    def order_key(edge):
        samples = tuple(tuple(round(v, 10) for v in edge.position_at(t))
                        for t in (0.0, 0.25, 0.5, 0.75, 1.0))
        return json.dumps(_edge_signature(edge), sort_keys=True), min(samples, samples[::-1])
    selected.sort(key=order_key)
    if not selected:
        raise AttaError("brep_selection_empty", "no edge matches the geometric selector",
                        path="args.edges", fix="check world coordinates, edge kind, direction, or face tags")
    expected = options.get("expect_count")
    if expected is not None and len(selected) != _integer(expected, "args.edges.expect_count", 1, 100_000):
        raise AttaError("brep_selection_count", "selected %d edges, not expect_count=%s" % (len(selected), expected),
                        path="args.edges.expect_count", fix="refine the selector or correct expect_count")
    return selected, [_edge_signature(edge) for edge in selected]


def _finish_edges(shape, op, args, face_tags=None):
    parameter = "radius" if op == "fillet" else "length"
    size = _positive(args.get(parameter), "args." + parameter)
    selected, descriptors = select_edges(shape, args.get("edges"), face_tags=face_tags)
    try:
        result = (shape.fillet(size, selected) if op == "fillet"
                  else shape.chamfer(size, None, selected))
        _valid_solid(result, "args." + parameter)
    except Exception as exc:
        raise AttaError("brep_operation_failed",
                        "%s %g mm failed on %d selected edges: %s" % (op, size, len(selected), exc),
                        path="args." + parameter,
                        fix="reduce the size or select fewer edges; no radius/length works on every shape") from exc
    return result, {"op": op, parameter + "_mm": size,
                    "selected_count": len(selected), "edges": descriptors}


def _mesh_settings(args):
    linear = _positive(args.get("linear_deflection", 0.05), "args.linear_deflection")
    angle = _positive(args.get("angular_deflection_deg", 12.0), "args.angular_deflection_deg")
    if angle < 1 or angle > 45:
        raise AttaError("bad_arg", "angular_deflection_deg must be in [1,45]",
                        path="args.angular_deflection_deg", fix="use 12 degrees for the default mesh")
    budget = _integer(args.get("max_triangles", 100_000), "args.max_triangles", 4, MAX_TRIANGLES)
    return linear, angle, budget


def build_graph(spec: Mapping) -> dict:
    """Build the supported graph in mm, with fail-closed native errors."""
    try:
        return _build_graph(spec)
    except AttaError:
        raise
    except Exception as exc:
        raise AttaError("brep_build", "OCCT graph build failed: %s" % exc,
                        path="model.features", fix="check dimensions/profiles and simplify the failing solid") from exc


def _build_graph(spec: Mapping) -> dict:
    """Build a B-rep from a validated Spec v3 graph; return a small report."""
    if not available():
        _require()
    b = _require()
    model = spec.get("model") if isinstance(spec.get("model"), Mapping) else spec
    kind = str((model or {}).get("kind") or "graph")
    if kind != "graph":
        raise AttaError(
            "brep_unsupported",
            "the B-rep backend currently accepts feature graphs only (kind=%r)" % kind,
            path="model.kind",
            fix="deliver the mesh (glb/obj/stl), or express the part as a graph",
        )
    feats = list((model or {}).get("features") or [])
    if not feats:
        raise AttaError("bad_spec", "graph has no features", path="model.features",
                        fix="add at least one primitive feature")

    # Match the mesh evaluator's DAG contract: forward references are legal,
    # and only the output's dependency closure is evaluated.  Resolve the
    # same numeric placeholders rather than feeding dicts to OCCT constructors.
    from .features import FEATURE_OPS, _resolve, _prepare
    units = str((model or {}).get("units", "mm")).lower()
    if units not in _UNIT_TO_MM:
        raise AttaError("bad_arg", "units must be mm/cm/m/in", path="model.units", fix="use mm")
    factor = _UNIT_TO_MM[units]
    params = dict((model or {}).get("params") or {})
    indexed = {}
    for feat in feats:
        fid = str(feat.get("id") or "")
        if not fid or fid in indexed:
            raise AttaError("duplicate_id" if fid in indexed else "bad_feature",
                            "feature ids must be non-empty and unique", path="features",
                            fix="give every feature a unique id")
        indexed[fid] = feat
    output = str((model or {}).get("output") or feats[-1].get("id"))
    if output not in indexed:
        raise AttaError("bad_output", "graph.output %r is not a feature id" % output,
                        path="model.output", fix="use one of: %s" % ", ".join(indexed))
    ordered, active, done = [], set(), set()

    def visit(fid):
        if fid in done:
            return
        if fid in active:
            raise AttaError("cycle", "feature graph contains a cycle", path="features",
                            fix="remove a cyclic dependency")
        if fid not in indexed:
            raise AttaError("unknown_input", "unknown feature input %r" % fid,
                            path="features.inputs", fix="reference an existing feature")
        feat = indexed[fid]
        op = str(feat.get("op") or "").lower()
        if op not in SUPPORTED:
            raise AttaError("brep_unsupported", "op %r has no B-rep mapping (mesh-only)" % op,
                            path="features.%s" % fid,
                            fix="remove it for STEP delivery, or use the mesh backend")
        inputs = list(feat.get("inputs") or [])
        arity = FEATURE_OPS[op]["arity"]
        if ((arity == 0 and inputs) or (arity == 1 and len(inputs) != 1)
                or (arity == 2 and len(inputs) < 2)):
            raise AttaError("bad_feature", "%s has invalid input count" % op,
                            path="features.%s.inputs" % fid,
                            fix="use the same input arity as the mesh feature")
        active.add(fid)
        for ref in inputs:
            visit(str(ref))
        active.remove(fid)
        resolved = dict(feat)
        resolved["op"] = op
        resolved["args"] = _resolve(dict(feat.get("args") or {}), params,
                                    "features.%s.args" % fid)
        # Core validation is shared, but does not execute mesh geometry here.
        checked = _prepare(op, resolved["args"], [None] * len(inputs), factor * 0.001,
                           "features.%s" % fid)
        if checked.get("caps") is False:
            raise AttaError("brep_unsupported", "open caps are not a solid STEP feature",
                            path="features.%s.args.caps" % fid,
                            fix="use caps=true or deliver the open mesh")
        resolved["args"] = _to_mm(op, resolved["args"], factor)
        ordered.append(resolved)
        done.add(fid)

    visit(output)
    built: dict[str, object] = {}
    built_tags: dict[str, dict] = {}
    used: list[str] = []
    selections, settings = [], []
    for feat in ordered:
        op = str(feat.get("op") or "")
        fid = str(feat.get("id") or "")
        if op not in SUPPORTED:
            raise AttaError(
                "brep_unsupported",
                "op %r has no B-rep mapping (mesh-only)" % op,
                path="features.%s" % fid,
                fix="remove it for STEP delivery, or use the mesh backend",
            )
        args = feat.get("args") or {}
        tags_dict = _merged_tags(feat, "features.%s.tags" % fid)
        in_ids = [str(i) for i in (feat.get("inputs") or []) if str(i) in built]
        inputs = [built[i] for i in in_ids]
        in_tags = [built_tags[i] for i in in_ids]
        tag_map = {"tags": {}, "features": {}}
        if op in ("box", "cylinder", "sphere", "cone", "torus"):
            shape = _primitive(b, feat)
            roles = _primitive_roles(op, shape, args)
            tag_map = {"tags": _tag_faces_by_role(b, shape, roles, tags_dict), "features": {}}
        elif op == "revolve":
            shape, tag_map = _revolve(b, args, tags_dict=tags_dict)
        elif op == "extrude":
            shape, tag_map = _extrude(b, args, tags_dict=tags_dict)
        elif op in ("union", "difference", "intersection"):
            if not inputs:
                raise AttaError("bad_feature", "%s needs inputs" % op,
                                path="features.%s.inputs" % fid,
                                fix="list two or more earlier feature ids")
            shape = inputs[0]
            cur_tags = in_tags[0]
            for other, other_tags in zip(inputs[1:], in_tags[1:]):
                prev = shape
                shape = (shape + other if op == "union"
                         else shape - other if op == "difference"
                         else shape & other)
                cur_tags = _propagate_history(b, [(prev, cur_tags), (other, other_tags)], shape)
            tag_map = cur_tags
        elif op in CAD_OPS:
            settings.append(_mesh_settings(args))
            shape, selection = _finish_edges(inputs[0], op, args, face_tags=in_tags[0])
            selections.append({"feature": fid, **selection})
            tag_map = _propagate_history(b, [(inputs[0], in_tags[0])], shape)
        elif op == "transform":
            if not inputs:
                raise AttaError("bad_feature", "transform needs one input",
                                path="features.%s.inputs" % fid,
                                fix="list the feature to transform")
            shape = _transform(b, inputs[0], args)
            pt_fn, dir_fn, sc = _transform_fns(args)
            tag_map = _propagate_transform(b, inputs[0], shape, in_tags[0], pt_fn, dir_fn, sc)
        elif op == "mirror":
            if not inputs:
                raise AttaError("bad_feature", "mirror needs one input",
                                path="features.%s.inputs" % fid,
                                fix="list the feature to mirror")
            plane = str(args.get("plane", "x")).lower()
            about = {"x": b.Plane.YZ, "y": b.Plane.XZ, "z": b.Plane.XY}.get(plane)
            if about is None:
                raise AttaError("bad_arg", "mirror.plane must be x/y/z",
                                path="args.plane", fix="use plane='x'")
            shape = b.mirror(inputs[0], about=about)
            idx = ("x", "y", "z").index(plane)
            def _reflect(v, ax=idx):
                lst = list(v)
                lst[ax] = -lst[ax]
                return tuple(lst)
            tag_map = _propagate_transform(b, inputs[0], shape, in_tags[0], _reflect, _reflect, 1.0)
        elif op in ("pattern_linear", "pattern_polar"):
            if not inputs:
                raise AttaError("bad_feature", "%s needs one input" % op,
                                path="features.%s.inputs" % fid,
                                fix="list the feature to pattern")
            shape, tag_map = _pattern(b, inputs[0], feat, in_tags=in_tags[0])
        else:  # pragma: no cover - SUPPORTED guards this
            raise AttaError("brep_unsupported", "op %r" % op, path="features.%s" % fid)
        _valid_solid(shape, "features.%s" % fid)
        tag_map.setdefault("features", {})[fid] = list(shape.faces())
        built[fid] = shape
        built_tags[fid] = tag_map
        used.append(op)

    output = str((model or {}).get("output") or feats[-1].get("id"))
    if output not in built:
        raise AttaError("bad_output", "graph.output %r is not a feature id" % output,
                        path="model.output", fix="use one of: %s" % ", ".join(built))
    part = built[output]
    part_tags = built_tags[output]
    part._atta_face_tags = part_tags
    import hashlib
    import json
    source_hash = hashlib.sha256(json.dumps({"features": ordered, "output": output, "units": "mm"},
                                           sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {"ok": True, "backend": "brep", "engine": "build123d/OCCT", "source_sha256": source_hash,
            "shape": part, "volume_mm3": float(part.volume),
            "ops": used, "output": output, "selections": selections,
            "face_tags": {k: len(v) for k, v in (part_tags.get("tags") or {}).items()},
            "mesh_settings": list(map(min, zip(*settings))) if settings else [0.05, 12.0, 100_000]}


def tessellate(shape, linear_deflection=0.05, angular_deflection_deg=12.0,
               max_triangles=100_000):
    """Return a bounded watertight mesh, or an actionable AttaError."""
    try:
        return _tessellate(shape, linear_deflection, angular_deflection_deg, max_triangles)
    except AttaError:
        raise
    except Exception as exc:
        raise AttaError("brep_tessellation", "OCCT tessellation failed: %s" % exc,
                        path="tessellation", fix="adjust deflection or simplify the solid; no weld fallback") from exc


def _tessellate(shape, linear_deflection=0.05, angular_deflection_deg=12.0,
                max_triangles=100_000):
    """Tessellate mm OCCT solids to a metre Mesh without spatial welding.

    Face nodes are joined only by their shared TopoDS edge polygon ordinals
    and TopoDS endpoint identities. Each solid has its own identity namespace,
    so distinct touching solids never become a non-manifold mesh by accident.
    """
    b = _require()
    import numpy as np
    from .mesh import Mesh, concat
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    linear, angle, budget = _mesh_settings({"linear_deflection": linear_deflection,
        "angular_deflection_deg": angular_deflection_deg, "max_triangles": max_triangles})
    _valid_solid(shape, "tessellation")
    diag = float(shape.bounding_box().diagonal)
    # Reject unreasonable precision before entering the native mesher. Native
    # OCCT has no triangle-count cancellation API; the output count is checked
    # before allocating Python/numpy mesh arrays, not claimed as an RSS limit.
    if linear < max(1e-7, diag * 1e-5):
        raise AttaError("brep_mesh_budget", "linear deflection is too small for this shape",
                        path="args.linear_deflection",
                        fix="use >= max(1e-7 mm, bounding-box diagonal / 100000)")
    cast_edge = getattr(TopoDS, "Edge_s", None) or TopoDS.Edge
    parts, triangle_count, node_count, native_status = [], 0, 0, []
    for solid in shape.solids():
        mesher = BRepMesh_IncrementalMesh(solid.wrapped, linear, False, math.radians(angle), False)
        status = int(mesher.GetStatusFlags())
        native_status.append(status)
        # ReMesh(0x08) is refined edge discretization; Reused(0x80) is a
        # valid cached triangulation (common in patterned solids). All other
        # and future unknown status bits fail closed.
        if not mesher.IsDone() or status & ~0x88:
            raise AttaError("brep_tessellation", "OCCT could not mesh all faces cleanly",
                            path="tessellation", fix="adjust deflection or simplify the solid")
        data, offset = [], 0
        for face in solid.faces():
            loc = TopLoc_Location()
            poly = BRep_Tool.Triangulation_s(face.wrapped, loc)
            if poly is None or poly.NbTriangles() == 0:
                raise AttaError("brep_tessellation", "OCCT returned an unmeshed face",
                                path="tessellation", fix="simplify the solid")
            triangle_count += poly.NbTriangles()
            node_count += poly.NbNodes()
            if triangle_count > budget or node_count > budget * 3:
                raise AttaError("brep_mesh_budget", "OCCT triangulation exceeds max_triangles",
                                path="args.max_triangles",
                                fix="increase linear/angular deflection or the bounded triangle budget")
            data.append((face, poly, loc, offset))
            offset += poly.NbNodes()
        parent = list(range(offset))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def join(i, j):
            a, c = root(i), root(j)
            if a != c:
                parent[max(a, c)] = min(a, c)
        edge_nodes, vertex_nodes = {}, {}
        verts, faces = [], []
        for face, poly, loc, off in data:
            transform = loc.Transformation()
            for i in range(1, poly.NbNodes() + 1):
                point = poly.Node(i).Transformed(transform)
                verts.append((point.X() * 0.001, point.Y() * 0.001, point.Z() * 0.001))
            reverse = face.wrapped.Orientation() == TopAbs_REVERSED
            for i in range(1, poly.NbTriangles() + 1):
                tri = poly.Triangle(i)
                ids = [tri.Value(j) + off - 1 for j in (1, 2, 3)]
                faces.append(ids[::-1] if reverse else ids)
            explorer = TopExp_Explorer(face.wrapped, TopAbs_EDGE)
            while explorer.More():
                edge = cast_edge(explorer.Current())
                polygon = BRep_Tool.PolygonOnTriangulation_s(edge, poly, loc)
                if polygon is None:
                    raise AttaError("brep_tessellation", "missing shared-edge polygon",
                                    path="tessellation", fix="use a supported solid; no weld fallback")
                ids = [polygon.Node(i) + off - 1 for i in range(1, polygon.NbNodes() + 1)]
                forward = cast_edge(edge.Oriented(TopAbs_FORWARD))
                edge_key = b.Edge(forward)
                parameters = [polygon.Parameter(i) for i in range(1, polygon.NbNodes() + 1)] if polygon.HasParameters() else None
                previous = edge_nodes.get(edge_key)
                if previous is not None:
                    previous_ids, previous_parameters = previous
                    if (len(ids) != len(previous_ids) or parameters != previous_parameters):
                        raise AttaError("brep_tessellation", "inconsistent shared-edge discretization",
                                        path="tessellation", fix="retry with a coarser deflection")
                    for a, c in zip(ids, previous_ids):
                        join(a, c)
                else:
                    edge_nodes[edge_key] = (ids, parameters)
                first = TopExp.FirstVertex_s(forward)
                last = TopExp.LastVertex_s(forward)
                for vertex, node in ((first, ids[0]), (last, ids[-1])):
                    if vertex.IsNull():
                        raise AttaError("brep_tessellation", "edge has no topological endpoint",
                                        path="tessellation", fix="use a finite closed solid")
                    key = b.Vertex(vertex)
                    if key in vertex_nodes:
                        join(node, vertex_nodes[key])
                    else:
                        vertex_nodes[key] = node
                if BRep_Tool.Degenerated_s(edge):
                    for node in ids:
                        join(ids[0], node)
                explorer.Next()
        roots = np.asarray([root(i) for i in range(offset)], dtype=np.int64)
        unique, inverse = np.unique(roots, return_inverse=True)
        triangles = inverse[np.asarray(faces, dtype=np.int64)]
        keep = ((triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
                & (triangles[:, 2] != triangles[:, 0]))
        part = Mesh(np.asarray(verts)[unique], triangles[keep])
        topology = part.topology()
        if not topology["closed"] or topology["non_manifold_edges"]:
            raise AttaError("brep_tessellation", "topological stitching did not produce a watertight mesh",
                            path="tessellation", fix="simplify the model; no spatial weld fallback")
        parts.append(part)
    mesh = concat(parts)
    mesh.meta["tessellation"] = {"method": "OCCT shared-edge/vertex identity; per-solid namespace",
        "linear_deflection_mm": linear, "angular_deflection_deg": angle,
        "max_triangles": budget, "raw_triangles": triangle_count, "triangles": len(mesh.faces),
        "spatial_weld": False, "exact": False, "strict_surface_error_bound": False,
        "native_allocation_hard_limit": False, "occt_status_flags": native_status}
    return mesh


def mesh_graph(features, params=None, units="mm", output=None):
    """Evaluator exact branch; rebuild once, then mesh the same STEP geometry."""
    report = build_graph({"kind": "graph", "features": features, "params": params or {},
                          "units": units, "output": output})
    mesh = tessellate(report["shape"], *report["mesh_settings"])
    measured_volume = mesh.volume() * 1e9
    mesh.meta["tessellation"].update({"mesh_volume_mm3": measured_volume,
        "volume_relative_error": abs(measured_volume - report["volume_mm3"]) / report["volume_mm3"]})
    mesh.meta.update({"backend": "brep", "engine": report["engine"],
                      "analytic_volume_mm3": report["volume_mm3"],
                      "selection_receipts": report["selections"],
                      "brep_ops": report["ops"], "brep_output": report["output"],
                      "source_sha256": report["source_sha256"]})
    return mesh


def export_step(spec: Mapping, path: str) -> dict:
    """Build a graph and write ``path`` as STEP (millimetres)."""
    b = _require()
    report = build_graph(spec)
    try:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        b.export_step(report["shape"], path, unit=b.Unit.MM)
        size = os.path.getsize(path) if os.path.exists(path) else 0
    except OSError as exc:
        raise AttaError("brep_export", "STEP output could not be written: %s" % exc,
                        path="path", fix="use a writable file path whose parent is a directory",
                        details={"path": os.fspath(path), "errno": exc.errno}) from exc
    if size <= 0:
        raise AttaError("brep_export", "STEP export produced no bytes",
                        path="path", fix="check write permissions")
    return {"ok": True, "path": path, "bytes": size, "backend": "brep",
            "engine": report["engine"], "volume_mm3": report["volume_mm3"],
            "ops": report["ops"], "source_sha256": report["source_sha256"],
            "selection_receipts": report["selections"]}
