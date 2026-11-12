# -*- coding: utf-8 -*-
"""Form-first ops: ``skeleton`` and ``silhouette``.

These are the two "build the whole before the parts" entry points.  Both are
registered as feature-graph ops, so they compose with the existing 25 ops and
work inside a Spec v3 ``graph`` model without touching the engine.

* ``skeleton`` — joints + bones, grown into a solid.  Pose and proportion live
  in the skeleton; radii only control how much material wraps it.
* ``silhouette`` — three orthographic outlines intersected into a visual hull.
  The direct answer to "I know the shape, not the parameters".

Both take millimetres at the spec boundary, like every other op.
"""

from __future__ import annotations

import numpy as np

from .. import form as _F
from ..errors import AttaError

__all__ = []


def _skeleton_op(*, joints, bones=None, segments=32, blend=0.0):
    mesh, info = _F.skeleton_solid(joints, bones or [], segments=int(segments),
                                   blend=float(blend))
    mesh.meta.update({"form": "skeleton", **info})
    return mesh


def _skeleton_prepare(args, scale, path):
    """Validate a skeleton at the spec boundary (mm in, metres out)."""
    raw_joints = args.get("joints")
    if not isinstance(raw_joints, (list, tuple)) or not raw_joints:
        raise AttaError("bad_arg", "skeleton.joints must be a non-empty array",
                        path=path + ".args.joints",
                        fix='pass [{"id":"hip","at_mm":[0,0,0],"radius_mm":6}]')
    joints = []
    seen = set()
    for i, row in enumerate(raw_joints):
        if not isinstance(row, dict):
            raise AttaError("bad_arg", "joint must be an object",
                            path=path + ".args.joints[%d]" % i,
                            fix='use {"id":..., "at_mm":[...], "radius_mm":...}')
        jid = row.get("id")
        if not isinstance(jid, str) or not jid:
            raise AttaError("bad_arg", "joint needs a non-empty id",
                            path=path + ".args.joints[%d].id" % i,
                            fix="give each joint a unique id")
        if jid in seen:
            raise AttaError("duplicate_id", "joint id %r is used twice" % jid,
                            path=path + ".args.joints[%d].id" % i,
                            fix="rename one of them")
        seen.add(jid)
        at = np.asarray(row.get("at_mm"), dtype=np.float64).reshape(-1)
        if at.shape != (3,) or not np.isfinite(at).all():
            raise AttaError("bad_arg", "joint.at_mm must be 3 finite numbers",
                            path=path + ".args.joints[%d].at_mm" % i,
                            fix="pass [x, y, z] in mm")
        radius = row.get("radius_mm", 0.0)
        radius = float(radius) if radius is not None else 0.0
        if not np.isfinite(radius) or radius < 0.0:
            raise AttaError("bad_arg", "joint.radius_mm must be >= 0",
                            path=path + ".args.joints[%d].radius_mm" % i,
                            fix="use radius_mm >= 0")
        joints.append({"id": jid, "at": (at * scale).tolist(),
                       "radius": radius * scale})

    raw_bones = args.get("bones") or []
    if not isinstance(raw_bones, (list, tuple)):
        raise AttaError("bad_arg", "skeleton.bones must be an array",
                        path=path + ".args.bones", fix="pass [] or a list of bones")
    bones = []
    for i, row in enumerate(raw_bones):
        if not isinstance(row, dict):
            raise AttaError("bad_arg", "bone must be an object",
                            path=path + ".args.bones[%d]" % i,
                            fix='use {"a":"hip","b":"knee"}')
        out = {"a": row.get("a"), "b": row.get("b")}
        for key in ("radius_a_mm", "radius_b_mm"):
            if row.get(key) is not None:
                val = float(row[key])
                if not np.isfinite(val) or val < 0.0:
                    raise AttaError("bad_arg", "bone.%s must be >= 0" % key,
                                    path=path + ".args.bones[%d].%s" % (i, key),
                                    fix="use a non-negative radius in mm")
                out[key.replace("_mm", "")] = val * scale
        bones.append(out)

    segments = int(round(float(args.get("segments", 32))))
    if segments < 8:
        raise AttaError("bad_arg", "skeleton.segments must be >= 8",
                        path=path + ".args.segments", fix="use segments=32")
    blend = float(args.get("blend_mm", 0.0)) * scale
    if blend < 0.0:
        raise AttaError("bad_arg", "skeleton.blend_mm must be >= 0",
                        path=path + ".args.blend_mm",
                        fix="use blend_mm: 0, or a few mm for an organic joint")
    return {"joints": joints, "bones": bones, "segments": segments,
            "blend": blend}


def _silhouette_op(*, views, height=None):
    mesh, info = _F.silhouette_solid(
        views, height=(float(height) if height is not None else None))
    mesh.meta.update({"form": "silhouette", **info})
    return mesh


def _silhouette_prepare(args, scale, path):
    raw = args.get("views")
    if not isinstance(raw, dict) or not raw:
        raise AttaError("bad_arg", "silhouette.views must be a non-empty object",
                        path=path + ".args.views",
                        fix='use {"front":{"outline_mm":[[x,z],...]}, ...}')
    views = {}
    for name, row in raw.items():
        key = str(name).lower()
        if not isinstance(row, dict):
            raise AttaError("bad_arg", "view %r must be an object" % key,
                            path=path + ".args.views.%s" % key,
                            fix='use {"outline_mm":[[u,v],...]}')
        loop = np.asarray(row.get("outline_mm"), dtype=np.float64)
        if loop.ndim != 2 or loop.shape[1] != 2 or len(loop) < 3:
            raise AttaError("bad_arg", "view %r outline_mm must be (n>=3, 2)" % key,
                            path=path + ".args.views.%s.outline_mm" % key,
                            fix="pass a closed 2D outline in mm")
        if not np.isfinite(loop).all():
            raise AttaError("bad_arg", "view %r outline contains NaN or Inf" % key,
                            path=path + ".args.views.%s.outline_mm" % key,
                            fix="remove non-finite coordinates")
        out = {"outline": (loop * scale).tolist()}
        depth = row.get("depth_mm")
        if depth is not None:
            d = np.asarray(depth, dtype=np.float64).reshape(-1)
            if d.shape != (2,) or not np.isfinite(d).all() or d[1] <= d[0]:
                raise AttaError("bad_arg", "view %r depth_mm must be [min,max]" % key,
                                path=path + ".args.views.%s.depth_mm" % key,
                                fix="use [0, 40] in mm")
            out["depth"] = (d * scale).tolist()
        views[key] = out
    height = args.get("height_mm")
    height = float(height) * scale if height is not None else None
    return {"views": views, "height": height}


def register():
    """Register both ops; called once at import."""
    from ..features import register_op

    register_op(
        "skeleton",
        {"required": ("joints",), "arity": 0,
         "optional": ("bones", "segments", "blend_mm"),
         "doc": "Grow a solid around a skeleton: joints=[{id,at_mm,radius_mm}], "
                "bones=[{a,b,radius_a_mm,radius_b_mm}]. Pose and proportion come "
                "from the skeleton, volume from the radii; blend_mm rounds the "
                "joint. mm.",
         "args": {"joints": "anything", "bones": "anything",
                  "segments": ("int", 8), "blend_mm": "length"}},
        _skeleton_op,
    )
    register_op(
        "silhouette",
        {"required": ("views",), "arity": 0,
         "optional": ("height_mm",),
         "doc": "Visual hull from orthographic outlines: views={front:{outline_mm},"
                "top:{...},right:{...}} with optional depth_mm per view. mm.",
         "args": {"views": "anything", "height_mm": "length"}},
        _silhouette_op,
    )
    _skeleton_op.prepare = _skeleton_prepare
    _silhouette_op.prepare = _silhouette_prepare


try:
    register()
except Exception:  # pragma: no cover - a family must never break the engine
    pass
