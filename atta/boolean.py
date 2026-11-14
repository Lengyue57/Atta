# -*- coding: utf-8 -*-
"""CSG booleans and symmetry/pattern helpers.  Units: metres.

Booleans go through trimesh's ``manifold`` engine (manifold3d).  Results are
welded and required to remain closed; a failure raises ``AttaError`` with a
concrete fix instead of returning a broken mesh.
"""

from __future__ import annotations

import json
import math

import numpy as np

from .errors import AttaError
from .mesh import Mesh, concat, oriented, topology, weld

__all__ = ["boolean", "union", "difference", "intersection", "mirror",
           "pattern_linear", "pattern_polar"]

_OPS = ("union", "difference", "intersection")
_PROVENANCE_LIMIT = 128
_PROVENANCE_BYTES = 262144
_QUALITY_KEYS = ("nurbs", "field", "sampling", "extraction", "decimation",
                 "field_decimation", "exact", "surface_error_bound")


def _quality_provenance(meshes):
    """Flatten source quality records, never claiming bounds for the result.

    Deduplication prevents repeated CSG on the same sources from growing a
    binary metadata tree. Hard limits reject rather than drop a source warning.
    JSON round-tripping also detaches the result from mutable input metadata.
    """
    sources, issues = {}, {}
    total_bytes = visited = 0

    def budget():
        raise AttaError("geometry_provenance_budget", "geometry quality provenance exceeds its budget",
                        path="boolean", fix="split the operation into smaller independently verified models")

    def add(table, value):
        nonlocal total_bytes
        try:
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False)
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise AttaError("bad_arg", "geometry quality metadata must be finite JSON",
                            path="boolean", fix="provide serializable source quality metadata") from exc
        if encoded not in table:
            total_bytes += len(encoded.encode("utf-8"))
            if len(table) >= _PROVENANCE_LIMIT or total_bytes > _PROVENANCE_BYTES:
                budget()
            table[encoded] = json.loads(encoded)

    pending = [m.meta for m in meshes]
    seen = set()
    while pending:
        meta = pending.pop()
        if not isinstance(meta, dict) or id(meta) in seen:
            continue
        seen.add(id(meta))
        visited += 1
        if visited > _PROVENANCE_LIMIT * 4:
            budget()
        provenance = meta.get("geometry_provenance") or {}
        if not isinstance(provenance, dict) or not isinstance(meta.get("nurbs", {}), dict):
            raise AttaError("bad_arg", "source quality records must be objects",
                            path="boolean", fix="provide structured geometry quality metadata")
        inherited = provenance.get("sources", [])
        rows = meta.get("issues") or []
        parts = meta.get("parts") or []
        if any(not isinstance(items, list) or len(items) > _PROVENANCE_LIMIT
               for items in (inherited, rows, parts)):
            budget()
        if any(not isinstance(row, dict) for row in inherited + rows + parts):
            raise AttaError("bad_arg", "source quality records must be objects",
                            path="boolean", fix="provide structured geometry quality metadata")
        for source in inherited:
            add(sources, source)
        source = {key: meta[key] for key in _QUALITY_KEYS if key in meta}
        if source:
            source.update({key: meta[key] for key in ("kind", "op", "part_id") if key in meta})
            add(sources, source)
        for issue in rows:
            if isinstance(issue, dict):
                add(issues, issue)
        nurbs = meta.get("nurbs") or {}
        if nurbs.get("target_met") is False and not any(
                issue.get("code") == "nurbs_tolerance_unmet" for issue in rows
                if isinstance(issue, dict)):
            add(issues, {"code": "nurbs_tolerance_unmet", "severity": "warn",
                         "message": "a source surface approximation target was not met",
                         "fix": "increase source tessellation budgets or loosen its tolerance",
                         "details": {key: nurbs.get(key) for key in
                                     ("sampled_error", "tolerance", "surface_error_bound")}})
        pending.extend(parts)
    result = {}
    if sources:
        result["geometry_provenance"] = {"method": "source_quality/v1",
                                         "sources": list(sources.values())}
    if issues:
        result["issues"] = list(issues.values())
    return result


def _to_trimesh(mesh):
    import trimesh
    return trimesh.Trimesh(vertices=np.asarray(mesh.verts, dtype=np.float64),
                           faces=np.asarray(mesh.faces, dtype=np.int64),
                           process=False)


def _from_trimesh(obj):
    import trimesh
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if len(g.faces)]
        if not geoms:
            raise AttaError("boolean_empty", "boolean produced an empty scene",
                            path="boolean", fix="make the two solids overlap")
        obj = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    if obj is None or len(obj.faces) == 0:
        raise AttaError("boolean_empty", "boolean produced an empty mesh", path="boolean",
                        fix="make the two solids overlap")
    return Mesh(np.asarray(obj.vertices, dtype=np.float64),
                np.asarray(obj.faces, dtype=np.int64), {"kind": "boolean"})


def _to_manifold(mesh):
    import manifold3d
    return manifold3d.Manifold(manifold3d.Mesh(
        vert_properties=np.asarray(mesh.verts, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))


def _from_manifold(obj):
    mm = obj.to_mesh()
    verts = np.asarray(mm.vert_properties, dtype=np.float64)
    faces = np.asarray(mm.tri_verts, dtype=np.int64)
    if len(faces) == 0:
        raise AttaError("boolean_empty", "boolean produced an empty mesh", path="boolean",
                        fix="make the two solids overlap")
    return Mesh(verts[:, :3], faces, {"kind": "boolean"})


def _run(a, b, op):
    """Run the boolean through manifold3d directly.

    manifold3d is tried first, not trimesh: trimesh's ``manifold`` engine is a
    wrapper around the same library, and importing trimesh costs ~0.5 s cold —
    paid on every plugin call, since each call starts a fresh interpreter.
    """
    try:
        ma, mb = _to_manifold(a), _to_manifold(b)
        out = {"union": ma + mb, "difference": ma - mb,
               "intersection": ma ^ mb}[op]
        return _from_manifold(out), "manifold3d"
    except AttaError:
        raise
    except Exception:  # noqa: BLE001 - fall back to trimesh
        pass
    try:
        import trimesh
        out = getattr(trimesh.boolean, op)([_to_trimesh(a), _to_trimesh(b)],
                                           engine="manifold")
        return _from_trimesh(out), "trimesh"
    except AttaError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AttaError("boolean_failed", "boolean %s failed: %s" % (op, exc),
                        path="boolean.%s" % op,
                        fix="check that both inputs are watertight and actually overlap; "
                            "reduce self-intersections or shrink the tool") from exc


def boolean(a, b, op="union") -> Mesh:
    """CSG of two closed solids: ``op`` is union, difference or intersection.

    **Do not weld the output.**  When two solids touch along an edge or a face,
    manifold3d emits two coincident but topologically *distinct* vertices — one
    per solid.  Welding them at 1e-9 snaps the pair together and collapses a
    proper 2-manifold into a 4-faces-per-edge non-manifold edge, which then
    fails the ``closed`` test.  The geometry was correct; the post-processing
    broke it.

    A tangent union legitimately produces **two** connected components (two
    solids sharing a boundary), so ``n_components > 1`` is accepted here; the
    verify gate reports the count and the caller decides whether that is
    intended.
    """
    name = str(op).lower()
    if name not in _OPS:
        raise AttaError("bad_arg", "boolean op must be union, difference or intersection",
                        path="boolean.op", fix="use op='union'")
    for label, mesh in (("a", a), ("b", b)):
        if not isinstance(mesh, Mesh):
            raise AttaError("bad_arg", "%s must be a Mesh" % label, path="boolean.%s" % label,
                            fix="pass a Mesh from primitives/surfaces")
        if not topology(mesh)["closed"]:
            raise AttaError("boolean_input_open", "boolean input %s is not closed" % label,
                            path="boolean.%s" % label,
                            fix="pass a watertight mesh; close caps or thicken the surface")

    quality = _quality_provenance((a, b))
    out, backend = _run(a, b, name)
    out.meta.update(quality)
    out = oriented(out)
    topo = topology(out)
    if not topo["closed"]:
        # Fallback only: an exact weld (tol 0 is a no-op, so use a tiny one) can
        # rescue inputs that carried coincident vertices through manifold3d.
        # It must never be the default, because welding *first* is what breaks
        # tangency.
        retry = oriented(weld(out, 1e-12))
        if topology(retry)["closed"]:
            retry.meta["boolean"] = {"backend": backend, "welded_retry": True}
            return retry
        raise AttaError("boolean_not_closed", "boolean %s result is not closed" % name,
                        path="boolean.%s" % name,
                        fix="nudge the tool by 1e-4 m, lower the overlap, or rebuild the "
                            "inputs with more segments")
    out.meta["boolean"] = {"backend": backend,
                           "components": int(topo["n_components"])}
    return out


def union(a, b) -> Mesh:
    return boolean(a, b, "union")


def difference(a, b) -> Mesh:
    return boolean(a, b, "difference")


def intersection(a, b) -> Mesh:
    return boolean(a, b, "intersection")


def mirror(mesh, plane="x") -> Mesh:
    """Mirror across a principal plane (``x``/``y``/``z`` or ``yz``/``xz``/``xy``)."""
    if not isinstance(mesh, Mesh):
        raise AttaError("bad_arg", "mesh must be a Mesh", path="mirror", fix="pass a Mesh")
    p = str(plane).lower()
    axes = {"x": 0, "y": 1, "z": 2, "yz": 0, "xz": 1, "xy": 2}
    if p not in axes:
        raise AttaError("bad_arg", "plane must be x, y, z, yz, xz or xy", path="mirror.plane",
                        fix="use plane='x'")
    v = mesh.verts.copy()
    v[:, axes[p]] *= -1.0
    faces = mesh.faces[:, ::-1].copy()
    return oriented(weld(Mesh(v, faces, {**mesh.meta, "kind": "mirror"}), 1e-9))


def _axis_index(axis):
    a = str(axis).lower()
    if a not in ("x", "y", "z"):
        raise AttaError("bad_arg", "axis must be x, y or z", path="pattern.axis",
                        fix="use axis='x'")
    return a


def pattern_linear(mesh, count, spacing, axis="x") -> Mesh:
    """``count`` copies translated by ``spacing`` metres along ``axis``."""
    if not isinstance(mesh, Mesh):
        raise AttaError("bad_arg", "mesh must be a Mesh", path="pattern", fix="pass a Mesh")
    n = int(count)
    if n < 1:
        raise AttaError("bad_arg", "count must be >= 1", path="pattern.count",
                        fix="use count >= 1")
    d = float(spacing)
    if not math.isfinite(d):
        raise AttaError("bad_arg", "spacing must be finite", path="pattern.spacing",
                        fix="pass a finite spacing in metres")
    a = _axis_index(axis)
    t = [0.0, 0.0, 0.0]
    t["xyz".index(a)] = d
    parts = [mesh if i == 0 else mesh.transform(translate=np.asarray(t) * i)
             for i in range(n)]
    out = concat(parts)
    out.meta["kind"] = "pattern_linear"
    return out


def pattern_polar(mesh, count, axis="z", angle_deg=360.0, center=(0, 0, 0)) -> Mesh:
    """``count`` rotated copies; a full 360-degree sweep spaces them evenly."""
    if not isinstance(mesh, Mesh):
        raise AttaError("bad_arg", "mesh must be a Mesh", path="pattern", fix="pass a Mesh")
    n = int(count)
    if n < 1:
        raise AttaError("bad_arg", "count must be >= 1", path="pattern.count",
                        fix="use count >= 1")
    a = _axis_index(axis)
    total = float(angle_deg)
    if not math.isfinite(total):
        raise AttaError("bad_arg", "angle_deg must be finite", path="pattern.angle_deg",
                        fix="pass a finite angle in degrees")
    step = total / n if abs(abs(total) - 360.0) <= 1e-9 else (
        total / (n - 1) if n > 1 else 0.0)
    parts = []
    for i in range(n):
        rot = [0.0, 0.0, 0.0]
        rot["xyz".index(a)] = step * i
        parts.append(mesh if i == 0 else mesh.transform(rotate_deg=rot, pivot=center))
    out = concat(parts)
    out.meta["kind"] = "pattern_polar"
    return out
