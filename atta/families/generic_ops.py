# -*- coding: utf-8 -*-
"""Generic feature-graph ops: hull, minkowski, polyhedron, heightmap, deform, morph, projection_extrude.

These ops widen Atta's generic feature-graph vocabulary so unseen shapes can
be built directly from Spec v3 DAGs without touching ``atta/features.py`` or
``atta/engine.py``.

Conventions
-----------
* Public graph arguments are in **millimetres** (or model units converted by
  ``scale`` in each op's ``prepare``); angles are in **degrees**.
* Kernel inputs and outputs are :class:`~atta.mesh.Mesh` in **metres**.
* Every op returns a closed, outward-oriented manifold mesh or raises
  :class:`~atta.errors.AttaError` with ``code``, ``path`` and ``fix``.
* Hard triangle, vertex and sample budgets fail closed before allocating huge
  arrays.  All execution is deterministic.
"""

from __future__ import annotations

import math

import numpy as np

from ..errors import AttaError
from ..features import register_op
from ..mesh import Mesh, oriented, topology, weld

__all__ = [
    "convex_hull",
    "minkowski_sum",
    "build_polyhedron",
    "build_heightmap",
    "deform_mesh",
    "morph_meshes",
    "projection_extrude",
]

_MAX_VERTS_CAP = 1_000_000
_MAX_TRIS_CAP = 1_500_000
_MAX_SAMPLES_CAP = 1_200_000
_DEFAULT_MAX_VERTS = 250_000
_DEFAULT_MAX_TRIS = 500_000
_DEFAULT_MAX_SAMPLES = 500_000
_DEFAULT_MAX_FACE_PRODUCT = 12_000
_DEFAULT_MAX_PAIRWISE_POINTS = 500_000
_MIN_VOL_M3 = 1e-18


# ---------------------------------------------------------------- validation helpers

def _num(val, name: str, path: str) -> float:
    if isinstance(val, (bool, np.bool_)):
        raise AttaError("bad_arg", "%s must be a number, not boolean" % name,
                        path=path, fix="pass a numeric value")
    try:
        x = float(val)
    except (TypeError, ValueError) as exc:
        raise AttaError("bad_arg", "%s must be a finite number" % name,
                        path=path, fix="pass a numeric value") from exc
    if not math.isfinite(x):
        raise AttaError("bad_arg", "%s must be finite" % name,
                        path=path, fix="remove NaN or Inf")
    return x


def _pos_num(val, name: str, path: str) -> float:
    x = _num(val, name, path)
    if x <= 0.0:
        raise AttaError("bad_arg", "%s must be > 0" % name,
                        path=path, fix="pass a positive number")
    return x


def _int_bound(val, name: str, path: str, lo: int, hi: int) -> int:
    x = _num(val, name, path)
    if abs(x - round(x)) > 1e-9:
        raise AttaError("bad_arg", "%s must be an integer" % name,
                        path=path, fix="pass a whole number")
    k = int(round(x))
    if k < lo or k > hi:
        raise AttaError("bad_arg", "%s must be in [%d, %d]" % (name, lo, hi),
                        path=path, fix="use %s between %d and %d" % (name, lo, hi))
    return k


def _flag(val) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


def _axis(val, name: str, path: str) -> str:
    a = str(val or "z").lower()
    if a not in ("x", "y", "z"):
        raise AttaError("bad_arg", "%s must be 'x', 'y' or 'z'" % name,
                        path=path, fix="use 'x', 'y' or 'z'")
    return a


def _check_input_mesh(mesh, label: str, path: str, *, require_closed: bool = True) -> dict:
    if not isinstance(mesh, Mesh):
        raise AttaError("bad_arg", "%s must be a Mesh" % label,
                        path=path, fix="wire a solid feature into inputs")
    if len(mesh.verts) == 0 or len(mesh.faces) == 0:
        raise AttaError("bad_arg", "%s mesh is empty" % label,
                        path=path, fix="pass non-empty geometry")
    if not np.isfinite(mesh.verts).all():
        raise AttaError("bad_arg", "%s mesh contains NaN or Inf vertices" % label,
                        path=path, fix="ensure all input vertices are finite")
    topo = topology(mesh)
    if require_closed and (not topo["closed"] or topo["non_manifold_edges"] > 0):
        raise AttaError("open_input",
                        "%s must be a closed manifold (boundary=%d, non_manifold=%d)"
                        % (label, topo["boundary_edges"], topo["non_manifold_edges"]),
                        path=path, fix="close or repair %s before passing it in" % label)
    return topo


def _assert_closed_manifold(mesh: Mesh, op: str, path: str) -> dict:
    topo = topology(mesh)
    if topo["empty"] or not topo["closed"] or topo["non_manifold_edges"] > 0:
        raise AttaError(
            "%s_not_closed" % op,
            "%s failed to produce a closed manifold (boundary=%d, non_manifold=%d)"
            % (op, topo["boundary_edges"], topo["non_manifold_edges"]),
            path=path,
            fix="check input geometry or increase resolution/subdivision",
        )
    if topo["degenerate_faces"] > 0:
        raise AttaError(
            "%s_degenerate" % op,
            "%s produced %d degenerate triangle(s)" % (op, topo["degenerate_faces"]),
            path=path,
            fix="check input scale and remove collinear or duplicate vertices",
        )
    return topo


# ---------------------------------------------------------------- 1. hull

def _hull_from_points(pts: np.ndarray, *, kind: str = "hull",
                      max_vertices: int = _DEFAULT_MAX_VERTS,
                      max_triangles: int = _DEFAULT_MAX_TRIS,
                      path: str = "inputs") -> Mesh:
    from scipy.spatial import ConvexHull, QhullError

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise AttaError("bad_arg", "points must have shape (N, 3)", path=path,
                        fix="pass 3D vertex coordinates")
    if not np.isfinite(pts).all():
        raise AttaError("bad_arg", "hull points contain NaN or Inf", path=path,
                        fix="remove non-finite coordinates")
    if len(pts) > max_vertices:
        raise AttaError("budget_exceeded",
                        "hull input has %d vertices, exceeding max_vertices=%d"
                        % (len(pts), max_vertices),
                        path="max_vertices",
                        fix="simplify inputs or raise max_vertices up to %d" % _MAX_VERTS_CAP,
                        details={"vertices": int(len(pts)), "max_vertices": int(max_vertices)})

    uniq = np.unique(pts, axis=0)
    if len(uniq) < 4:
        raise AttaError("hull_degenerate",
                        "convex hull requires at least 4 distinct 3D points",
                        path=path, fix="pass solids with non-zero 3D volume")
    try:
        hull = ConvexHull(uniq)
    except (QhullError, ValueError) as exc:
        raise AttaError("hull_degenerate",
                        "convex hull is degenerate (coplanar or collinear points): %s" % exc,
                        path=path, fix="ensure input geometry spans 3 non-coplanar dimensions") from exc

    if float(hull.volume) <= _MIN_VOL_M3:
        raise AttaError("hull_degenerate",
                        "convex hull has zero volume",
                        path=path, fix="ensure input geometry is 3D, not flat")

    simplices = np.asarray(hull.simplices, dtype=np.int64)
    if len(simplices) > max_triangles:
        raise AttaError("budget_exceeded",
                        "hull has %d triangles, exceeding max_triangles=%d"
                        % (len(simplices), max_triangles),
                        path="max_triangles",
                        fix="simplify inputs or raise max_triangles up to %d" % _MAX_TRIS_CAP,
                        details={"triangles": int(len(simplices)), "max_triangles": int(max_triangles)})

    used = np.unique(simplices)
    remap = np.full(len(hull.points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    verts = np.asarray(hull.points, dtype=np.float64)[used]
    faces = remap[simplices]

    # Orient outward relative to the interior centroid of the convex polytope.
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    center = verts.mean(axis=0)
    flip = np.einsum("ij,ij->i", nrm, tri.mean(axis=1) - center) < 0.0
    faces[flip] = faces[flip][:, ::-1]

    out = oriented(Mesh(verts, faces, {"kind": kind, "op": kind, "exact": True}))
    topo = topology(out)
    if not topo["closed"] or topo["non_manifold_edges"] > 0 or topo["degenerate_faces"] > 0:
        out = oriented(weld(out, 1e-12))
    _assert_closed_manifold(out, kind, path)
    return out


def convex_hull(meshes, *, max_vertices: int = _DEFAULT_MAX_VERTS,
                max_triangles: int = _DEFAULT_MAX_TRIS) -> Mesh:
    """Compute the oriented closed convex hull of one or more meshes (metres)."""
    if not meshes:
        raise AttaError("bad_arg", "hull needs at least one input mesh",
                        path="inputs", fix='pass inputs=["<id>", ...]')
    total_v = 0
    clouds = []
    for i, m in enumerate(meshes):
        _check_input_mesh(m, "inputs[%d]" % i, "inputs[%d]" % i, require_closed=False)
        total_v += len(m.verts)
        if total_v > max_vertices:
            raise AttaError("budget_exceeded",
                            "hull inputs have %d vertices, exceeding max_vertices=%d"
                            % (total_v, max_vertices),
                            path="max_vertices",
                            fix="reduce input mesh resolution or raise max_vertices",
                            details={"vertices": total_v, "max_vertices": int(max_vertices)})
        clouds.append(m.verts)
    pts = np.vstack(clouds)
    out = _hull_from_points(pts, kind="hull", max_vertices=max_vertices,
                            max_triangles=max_triangles, path="inputs")
    out.meta["hull"] = {"n_inputs": len(meshes), "exact": True}
    return out


def _hull_op(*, meshes, max_vertices=_DEFAULT_MAX_VERTS,
             max_triangles=_DEFAULT_MAX_TRIS) -> Mesh:
    return convex_hull(meshes, max_vertices=max_vertices, max_triangles=max_triangles)


def _hull_prepare(args, scale, path):
    mv = _int_bound(args.get("max_vertices", _DEFAULT_MAX_VERTS),
                    "max_vertices", path + ".args.max_vertices", 4, _MAX_VERTS_CAP)
    mt = _int_bound(args.get("max_triangles", _DEFAULT_MAX_TRIS),
                    "max_triangles", path + ".args.max_triangles", 4, _MAX_TRIS_CAP)
    return {"max_vertices": mv, "max_triangles": mt}


# ---------------------------------------------------------------- 2. minkowski

def _convex_hull_verts_if_convex(mesh: Mesh, topo: dict):
    """Return hull vertices if ``mesh`` is a single convex solid, else None."""
    from scipy.spatial import ConvexHull, QhullError

    if topo["n_components"] != 1 or topo["genus"] != 0:
        return None
    verts = np.asarray(mesh.verts, dtype=np.float64)
    try:
        hull = ConvexHull(np.unique(verts, axis=0))
    except (QhullError, ValueError):
        return None
    eq = np.asarray(hull.equations, dtype=np.float64)
    resid = verts @ eq[:, :3].T + eq[:, 3]
    scale = max(float(mesh.bbox()["diag"]), 1e-9)
    if bool((resid.max(axis=1) < -1e-6 * scale).any()):
        return None
    vol_mesh = float(mesh.volume())
    vol_hull = float(hull.volume)
    if vol_hull <= _MIN_VOL_M3 or abs(vol_mesh - vol_hull) > 1e-4 * vol_hull:
        return None
    return np.asarray(hull.points[hull.vertices], dtype=np.float64)


def _minkowski_sdf_pair(a: Mesh, b: Mesh, *, resolution: int,
                        max_samples: int, max_triangles: int) -> Mesh:
    """Documented voxel-convolution + EDT SDF approximation of A (+) B (exact:false)."""
    from scipy.ndimage import distance_transform_edt
    from scipy.signal import fftconvolve
    from .. import sdf, sdfkit
    from .offset_ops import _inside_grid

    lo_a, hi_a = a.verts.min(axis=0), a.verts.max(axis=0)
    lo_b, hi_b = b.verts.min(axis=0), b.verts.max(axis=0)
    lo = lo_a + lo_b
    hi = hi_a + hi_b
    span = float((hi - lo).max())
    if span <= 1e-12:
        raise AttaError("minkowski_degenerate", "minkowski operands have zero extent",
                        path="inputs", fix="pass non-degenerate 3D solids")

    spacing = span / float(resolution)
    # Estimate padded A + B grid sizes and coarsen if needed to fit max_samples.
    for _ in range(64):
        pad = 2.5 * spacing
        na = np.ceil((hi_a + pad - (lo_a - pad)) / spacing).astype(int) + 1
        nb = np.maximum(1, np.ceil((hi_b + 0.5 * spacing - (lo_b - 0.5 * spacing)) / spacing).astype(int) + 1)
        out_shape = na + nb - 1 + 4  # +4 for boundary False padding around conv
        est = int(np.prod(na)) + int(np.prod(nb)) + int(np.prod(out_shape))
        if est <= max_samples:
            break
        spacing *= max(1.02, (est / float(max_samples)) ** (1.0 / 3.0))
    else:
        raise AttaError("budget_exceeded",
                        "minkowski SDF grid exceeds max_samples=%d" % max_samples,
                        path="max_samples",
                        fix="reduce resolution or raise max_samples up to %d" % _MAX_SAMPLES_CAP)

    # Also enforce sdf.fit_spacing compatibility on final output domain.
    spacing = max(spacing, sdf.fit_spacing(lo, hi, spacing, max_samples=max_samples))
    pad = 2.5 * spacing
    lo_a_p = lo_a - pad
    na = np.ceil((hi_a + pad - lo_a_p) / spacing).astype(int) + 1
    lo_b_p = lo_b - 0.5 * spacing
    nb = np.maximum(1, np.ceil((hi_b + 0.5 * spacing - lo_b_p) / spacing).astype(int) + 1)

    ins_a = _inside_grid(na, lo_a_p, spacing, a, axis=2)
    if not ins_a.any():
        raise AttaError("minkowski_empty",
                        "operand A is thinner than the SDF grid spacing (%.3g mm)" % (spacing * 1e3),
                        path="resolution", fix="increase resolution or use exact mode on simpler meshes")
    ins_b = _inside_grid(nb, lo_b_p, spacing, b, axis=2)
    if not ins_b.any():
        # Sub-voxel operand B: evaluate at its centroid voxel so small tools still shift/dilate.
        ins_b[nb[0] // 2, nb[1] // 2, nb[2] // 2] = True

    conv = fftconvolve(ins_a.astype(np.float64), ins_b.astype(np.float64), mode="full")
    mask = conv > 0.5
    mp = np.pad(mask, 2, mode="constant", constant_values=False)
    org = lo_a_p + lo_b_p - 2.0 * spacing
    phi = (distance_transform_edt(~mp) - distance_transform_edt(mp)) * spacing
    out = sdfkit.extract(phi, org, spacing, max_triangles=max_triangles)
    out = oriented(out)
    _assert_closed_manifold(out, "minkowski", "inputs")
    out.meta.update({
        "kind": "minkowski",
        "op": "minkowski",
        "exact": False,
        "minkowski": {
            "exact": False,
            "method": "sdf_voxel_convolution",
            "resolution": int(resolution),
            "spacing_mm": float(spacing * 1e3),
        },
    })
    return out


def _minkowski_pair(a: Mesh, b: Mesh, *, method: str, resolution: int,
                    max_vertices: int, max_triangles: int,
                    max_samples: int, max_face_product: int) -> Mesh:
    topo_a = _check_input_mesh(a, "a", "inputs[0]")
    topo_b = _check_input_mesh(b, "b", "inputs[1]")

    if method != "sdf":
        va = _convex_hull_verts_if_convex(a, topo_a)
        vb = _convex_hull_verts_if_convex(b, topo_b)
        if va is not None and vb is not None:
            pair_budget = min(max_vertices, _DEFAULT_MAX_PAIRWISE_POINTS)
            prod = len(va) * len(vb)
            if prod > pair_budget:
                if method == "auto":
                    return _minkowski_sdf_pair(a, b, resolution=resolution,
                                               max_samples=max_samples,
                                               max_triangles=max_triangles)
                raise AttaError(
                    "budget_exceeded",
                    "pairwise vertex sum has %d points, exceeding budget %d" % (prod, pair_budget),
                    path="max_vertices",
                    fix="simplify inputs, raise max_vertices, or use method='sdf'",
                    details={"pairwise_points": prod, "budget": pair_budget},
                )
            pts = (va[:, None, :] + vb[None, :, :]).reshape(-1, 3)
            out = _hull_from_points(pts, kind="minkowski", max_vertices=pair_budget,
                                    max_triangles=max_triangles, path="inputs")
            out.meta["exact"] = True
            out.meta["minkowski"] = {"exact": True, "method": "convex_hull_pairwise_sum",
                                     "pairwise_points": int(prod)}
            return out

        face_prod = len(a.faces) * len(b.faces)
        if face_prod <= max_face_product:
            from ..boolean import _from_manifold, _to_manifold
            try:
                m = _to_manifold(a).minkowski_sum(_to_manifold(b))
                out = _from_manifold(m)
                if topology(out)["degenerate_faces"] > 0:
                    tol = max(1e-9, float(out.bbox()["diag"]) * 1e-7)
                    out = _from_manifold(m.simplify(tol))
                out = oriented(out)
                if len(out.faces) > max_triangles:
                    raise AttaError(
                        "budget_exceeded",
                        "minkowski output has %d triangles, exceeding max_triangles=%d"
                        % (len(out.faces), max_triangles),
                        path="max_triangles",
                        fix="simplify inputs or raise max_triangles",
                    )
                _assert_closed_manifold(out, "minkowski", "inputs")
                out.meta["kind"] = "minkowski"
                out.meta["op"] = "minkowski"
                out.meta["exact"] = True
                out.meta["minkowski"] = {"exact": True, "method": "manifold_minkowski_sum",
                                         "face_product": int(face_prod)}
                return out
            except AttaError:
                raise
            except Exception as exc:
                if method == "exact":
                    raise AttaError("minkowski_failed",
                                    "exact minkowski_sum failed: %s" % exc,
                                    path="inputs", fix="use method='sdf' or simplify inputs") from exc

        if method == "exact":
            raise AttaError(
                "budget_exceeded",
                "non-convex minkowski face product %d exceeds max_face_product=%d"
                % (face_prod, max_face_product),
                path="max_face_product",
                fix="use method='sdf' for bounded approximation, or simplify inputs",
                details={"face_product": face_prod, "max_face_product": int(max_face_product)},
            )

    return _minkowski_sdf_pair(a, b, resolution=resolution,
                               max_samples=max_samples, max_triangles=max_triangles)


def minkowski_sum(meshes, *, method: str = "auto", resolution: int = 48,
                  max_vertices: int = _DEFAULT_MAX_PAIRWISE_POINTS,
                  max_triangles: int = _DEFAULT_MAX_TRIS,
                  max_samples: int = _DEFAULT_MAX_SAMPLES,
                  max_face_product: int = _DEFAULT_MAX_FACE_PRODUCT) -> Mesh:
    """Compute the Minkowski sum across two or more closed meshes (metres)."""
    if len(meshes) < 2:
        raise AttaError("bad_arg", "minkowski requires at least two input meshes",
                        path="inputs", fix='pass inputs=["<a>", "<b>"]')
    acc = meshes[0]
    exact_all = True
    methods_used = []
    for nxt in meshes[1:]:
        acc = _minkowski_pair(
            acc, nxt, method=method, resolution=resolution,
            max_vertices=max_vertices, max_triangles=max_triangles,
            max_samples=max_samples, max_face_product=max_face_product,
        )
        exact_all = exact_all and bool(acc.meta.get("exact", False))
        methods_used.append(acc.meta["minkowski"]["method"])
    acc.meta["exact"] = exact_all
    acc.meta["minkowski"]["exact"] = exact_all
    if len(meshes) > 2:
        acc.meta["minkowski"]["chain_methods"] = methods_used
    return acc


def _minkowski_op(*, meshes, method="auto", resolution=48,
                  max_vertices=_DEFAULT_MAX_PAIRWISE_POINTS,
                  max_triangles=_DEFAULT_MAX_TRIS,
                  max_samples=_DEFAULT_MAX_SAMPLES,
                  max_face_product=_DEFAULT_MAX_FACE_PRODUCT) -> Mesh:
    return minkowski_sum(
        meshes, method=method, resolution=resolution, max_vertices=max_vertices,
        max_triangles=max_triangles, max_samples=max_samples,
        max_face_product=max_face_product,
    )


def _minkowski_prepare(args, scale, path):
    method = str(args.get("method") or "auto").lower()
    if method not in ("auto", "exact", "sdf"):
        raise AttaError("bad_arg", "method must be auto, exact or sdf",
                        path=path + ".args.method", fix="use method='auto'")
    return {
        "method": method,
        "resolution": _int_bound(args.get("resolution", 48), "resolution",
                                 path + ".args.resolution", 8, 128),
        "max_vertices": _int_bound(args.get("max_vertices", _DEFAULT_MAX_PAIRWISE_POINTS),
                                   "max_vertices", path + ".args.max_vertices", 4, _MAX_VERTS_CAP),
        "max_triangles": _int_bound(args.get("max_triangles", _DEFAULT_MAX_TRIS),
                                    "max_triangles", path + ".args.max_triangles", 4, _MAX_TRIS_CAP),
        "max_samples": _int_bound(args.get("max_samples", _DEFAULT_MAX_SAMPLES),
                                  "max_samples", path + ".args.max_samples", 343, _MAX_SAMPLES_CAP),
        "max_face_product": _int_bound(args.get("max_face_product", _DEFAULT_MAX_FACE_PRODUCT),
                                       "max_face_product", path + ".args.max_face_product",
                                       1, 200_000),
    }


# ---------------------------------------------------------------- 3. polyhedron

def _triangulate_polyhedron_faces(pts: np.ndarray, raw_faces, max_triangles: int) -> np.ndarray:
    from ..surfaces import _earclip, _plane_coords, _polygon_normal

    if not isinstance(raw_faces, (list, tuple)) or len(raw_faces) < 4:
        raise AttaError("bad_arg", "faces must be a list of at least 4 faces",
                        path="faces", fix="pass [[i0,i1,i2], ...] covering a closed solid")
    n_pts = len(pts)
    tris = []
    for fi, face in enumerate(raw_faces):
        if not isinstance(face, (list, tuple)) or len(face) < 3:
            raise AttaError("bad_arg", "face %d must have >= 3 vertex indices" % fi,
                            path="faces[%d]" % fi, fix="pass at least 3 vertex indices")
        idx = []
        for vi in face:
            if isinstance(vi, (bool, np.bool_)) or not isinstance(vi, (int, np.integer)):
                if isinstance(vi, float) and vi.is_integer():
                    vi = int(vi)
                else:
                    raise AttaError("bad_arg", "face %d index %r must be an integer" % (fi, vi),
                                    path="faces[%d]" % fi, fix="use integer indices into points")
            k = int(vi)
            if k < 0 or k >= n_pts:
                raise AttaError("bad_arg",
                                "face %d index %d out of range [0, %d)" % (fi, k, n_pts),
                                path="faces[%d]" % fi, fix="use indices in 0..%d" % (n_pts - 1))
            idx.append(k)
        if len(set(idx)) < 3:
            raise AttaError("polyhedron_degenerate",
                            "face %d repeats vertices and has fewer than 3 distinct indices" % fi,
                            path="faces[%d]" % fi, fix="remove repeated vertex indices")
        if len(idx) == 3:
            tris.append((idx[0], idx[1], idx[2]))
        else:
            if len(tris) + (len(idx) - 2) > max_triangles:
                raise AttaError("budget_exceeded",
                                "polyhedron exceeds max_triangles=%d" % max_triangles,
                                path="max_triangles", fix="simplify faces or raise max_triangles")
            poly3 = pts[np.asarray(idx, dtype=np.int64)]
            try:
                normal = _polygon_normal(poly3)
                local = _earclip(_plane_coords(poly3, normal))
            except AttaError as exc:
                raise AttaError("polyhedron_bad_face",
                                "face %d could not be triangulated: %s" % (fi, exc.message),
                                path="faces[%d]" % fi,
                                fix="ensure the face polygon is planar and non-self-intersecting") from exc
            for a, b, c in local:
                tris.append((idx[a], idx[b], idx[c]))
        if len(tris) > max_triangles:
            raise AttaError("budget_exceeded",
                            "polyhedron exceeds max_triangles=%d" % max_triangles,
                            path="max_triangles", fix="simplify faces or raise max_triangles")
    return np.asarray(tris, dtype=np.int64)


def _orient_components_outward(mesh: Mesh) -> Mesh:
    """Orient every connected component consistently and outward."""
    from ..repair import orient_consistently

    out = orient_consistently(mesh)
    verts = np.asarray(out.verts, dtype=np.float64)
    faces = np.asarray(out.faces, dtype=np.int64).copy()
    n_f = len(faces)
    if n_f == 0:
        return out

    # Union-find faces by shared vertices to orient each component independently.
    parent = np.arange(len(verts), dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in faces:
        ra, rb, rc = find(int(a)), find(int(b)), find(int(c))
        if ra != rb:
            parent[rb] = ra
            rb = ra
        if ra != rc:
            parent[rc] = ra

    roots = np.array([find(int(f[0])) for f in faces], dtype=np.int64)
    for root in np.unique(roots):
        mask = roots == root
        sub = faces[mask]
        origin = verts[sub[0, 0]]
        a = verts[sub[:, 0]] - origin
        b = verts[sub[:, 1]] - origin
        c = verts[sub[:, 2]] - origin
        signed6 = float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))))
        if signed6 < 0.0:
            faces[mask] = sub[:, ::-1]
    return Mesh(verts, faces, dict(out.meta))


def build_polyhedron(points, faces, *, convexity: int = 1, fix_winding: bool = True,
                     weld_tol: float = 0.0, max_vertices: int = _DEFAULT_MAX_VERTS,
                     max_triangles: int = _DEFAULT_MAX_TRIS) -> Mesh:
    """Build and validate a closed oriented Mesh from OpenSCAD-style points + faces (metres)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise AttaError("bad_arg", "points must be (n>=4, 3)", path="points",
                        fix="pass at least 4 [x, y, z] coordinates")
    if not np.isfinite(pts).all():
        raise AttaError("bad_arg", "points contain NaN or Inf", path="points",
                        fix="remove non-finite coordinates")
    if len(pts) > max_vertices:
        raise AttaError("budget_exceeded",
                        "polyhedron has %d points, exceeding max_vertices=%d"
                        % (len(pts), max_vertices),
                        path="max_vertices", fix="reduce points or raise max_vertices")

    tris = _triangulate_polyhedron_faces(pts, faces, max_triangles)
    mesh = Mesh(pts, tris, {"kind": "polyhedron", "op": "polyhedron", "convexity": int(convexity)})
    if weld_tol > 0.0:
        mesh = weld(mesh, weld_tol)

    topo = topology(mesh)
    if topo["degenerate_faces"] > 0:
        raise AttaError("polyhedron_degenerate",
                        "polyhedron has %d degenerate (zero-area) triangle(s)"
                        % topo["degenerate_faces"],
                        path="faces", fix="remove duplicate vertices or collinear face loops")
    if topo["boundary_edges"] > 0:
        raise AttaError("polyhedron_open",
                        "polyhedron is not closed (%d boundary edge(s))" % topo["boundary_edges"],
                        path="faces", fix="close all faces so every edge has 2 incident faces")
    if topo["non_manifold_edges"] > 0:
        raise AttaError("polyhedron_non_manifold",
                        "polyhedron has %d non-manifold edge(s)" % topo["non_manifold_edges"],
                        path="faces", fix="ensure every edge is shared by exactly 2 faces")

    from ..mesh import _signed_volume_and_centroid
    signed_vol, _ = _signed_volume_and_centroid(mesh)
    fixed_winding = False
    if topo["inconsistent_winding_edges"] > 0 or signed_vol < 0.0:
        if not fix_winding:
            code = ("polyhedron_bad_winding" if topo["inconsistent_winding_edges"] > 0
                    else "polyhedron_inside_out")
            raise AttaError(code,
                            "polyhedron has inconsistent or inward face winding "
                            "(inconsistent_edges=%d, signed_volume=%.4g)"
                            % (topo["inconsistent_winding_edges"], signed_vol),
                            path="faces",
                            fix="orient all faces CCW outward or pass fix_winding=true")
        mesh = _orient_components_outward(mesh)
        fixed_winding = True
        topo2 = topology(mesh)
        if topo2["inconsistent_winding_edges"] > 0:
            raise AttaError("polyhedron_non_orientable",
                            "polyhedron surface is non-orientable (%d inconsistent edge(s) remain)"
                            % topo2["inconsistent_winding_edges"],
                            path="faces",
                            fix="check face connectivity; Möbius-like surfaces cannot enclose a solid")

    if float(mesh.volume()) <= _MIN_VOL_M3:
        raise AttaError("polyhedron_degenerate", "polyhedron encloses zero volume",
                        path="points", fix="ensure points are not coplanar")
    _assert_closed_manifold(mesh, "polyhedron", "faces")
    mesh.meta["polyhedron"] = {"fixed_winding": fixed_winding,
                               "n_input_faces": len(faces),
                               "n_tris": int(len(mesh.faces))}
    return mesh


def _polyhedron_op(*, points, faces, convexity=1, fix_winding=True,
                   weld_tol=0.0, max_vertices=_DEFAULT_MAX_VERTS,
                   max_triangles=_DEFAULT_MAX_TRIS) -> Mesh:
    return build_polyhedron(
        points, faces, convexity=convexity, fix_winding=fix_winding,
        weld_tol=weld_tol, max_vertices=max_vertices, max_triangles=max_triangles,
    )


def _polyhedron_prepare(args, scale, path):
    try:
        pts = np.asarray(args.get("points"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AttaError("bad_arg", "points must be a 2D array of [x,y,z]",
                        path=path + ".args.points", fix="pass [[x,y,z], ...]") from exc
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise AttaError("bad_arg", "points must be (n>=4, 3)",
                        path=path + ".args.points", fix="pass at least 4 [x,y,z] points in mm")
    if not np.isfinite(pts).all():
        raise AttaError("bad_arg", "points contain NaN or Inf",
                        path=path + ".args.points", fix="remove non-finite coordinates")
    wt = _num(args.get("weld_tol", 0.0), "weld_tol", path + ".args.weld_tol")
    if wt < 0.0:
        raise AttaError("bad_arg", "weld_tol must be >= 0",
                        path=path + ".args.weld_tol", fix="use weld_tol >= 0")
    raw_faces = args.get("faces")
    if not isinstance(raw_faces, (list, tuple)) or len(raw_faces) < 4:
        raise AttaError("bad_arg", "faces must be a list of at least 4 faces",
                        path=path + ".args.faces",
                        fix="pass [[i0,i1,i2], ...] covering a closed solid")
    return {
        "points": pts * float(scale),
        "faces": raw_faces,
        "convexity": _int_bound(args.get("convexity", 1), "convexity",
                                path + ".args.convexity", 1, 1000),
        "fix_winding": _flag(args.get("fix_winding", True)),
        "weld_tol": wt * float(scale),
        "max_vertices": _int_bound(args.get("max_vertices", _DEFAULT_MAX_VERTS),
                                   "max_vertices", path + ".args.max_vertices", 4, _MAX_VERTS_CAP),
        "max_triangles": _int_bound(args.get("max_triangles", _DEFAULT_MAX_TRIS),
                                    "max_triangles", path + ".args.max_triangles", 4, _MAX_TRIS_CAP),
    }


# ---------------------------------------------------------------- 4. heightmap

def build_heightmap(heights, *, dx: float, dy: float, base: float = 1e-3,
                    center: bool = True, invert: bool = False,
                    max_vertices: int = _DEFAULT_MAX_VERTS,
                    max_triangles: int = _DEFAULT_MAX_TRIS) -> Mesh:
    """Convert a 2D height grid (metres) into a closed solid with a flat base at z=-base."""
    h = np.asarray(heights, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] < 2 or h.shape[1] < 2:
        raise AttaError("bad_arg", "heights must be a 2D matrix of shape (rows>=2, cols>=2)",
                        path="heights", fix="pass a 2D grid [[z00, z01], [z10, z11]] in mm")
    if not np.isfinite(h).all():
        raise AttaError("bad_arg", "heights contain NaN or Inf", path="heights",
                        fix="remove non-finite height entries")
    if not (math.isfinite(dx) and math.isfinite(dy) and dx > 0.0 and dy > 0.0):
        raise AttaError("bad_arg", "cell spacing dx, dy must be > 0", path="cell_size",
                        fix="pass positive cell_size or size")
    if not math.isfinite(base) or base < 0.0:
        raise AttaError("bad_arg", "base must be finite and >= 0", path="base",
                        fix="use base >= 0 mm")

    rows, cols = h.shape
    cr, cc = rows - 1, cols - 1
    # 4 top tris/cell (with bilinear center vertex) + 2*(2*cr + 2*cc) side tris + 2*(cr+cc) bottom tris
    n_perim = 2 * (cr + cc)
    n_verts = rows * cols + cr * cc + n_perim + 1
    n_tris = 4 * cr * cc + 3 * n_perim
    if n_verts > max_vertices:
        raise AttaError("budget_exceeded",
                        "heightmap requires %d vertices, exceeding max_vertices=%d"
                        % (n_verts, max_vertices),
                        path="max_vertices", fix="downsample heights grid or raise max_vertices")
    if n_tris > max_triangles:
        raise AttaError("budget_exceeded",
                        "heightmap requires %d triangles, exceeding max_triangles=%d"
                        % (n_tris, max_triangles),
                        path="max_triangles", fix="downsample heights grid or raise max_triangles")

    z_top = -h if invert else h.copy()
    z_bot = -float(base)
    if float(z_top.min()) <= z_bot + 1e-12:
        raise AttaError("heightmap_degenerate",
                        "minimum height %.4g mm is <= base plane %.4g mm (zero/negative thickness)"
                        % (float(z_top.min()) * 1e3, z_bot * 1e3),
                        path="base",
                        fix="increase base above %.4g mm so all heights sit strictly above the base"
                            % max(0.0, -float(z_top.min()) * 1e3 + 1.0))

    lx, ly = cc * dx, cr * dy
    xs = np.linspace(0.0, lx, cols) - (0.5 * lx if center else 0.0)
    ys = np.linspace(0.0, ly, rows) - (0.5 * ly if center else 0.0)
    # heights[r, c] maps r -> y, c -> x
    gx, gy = np.meshgrid(xs, ys)
    grid_verts = np.column_stack([gx.ravel(), gy.ravel(), z_top.ravel()])

    # Cell centers (bilinear average gives symmetric triangulation and exact bilinear volume).
    cx = 0.5 * (xs[:-1] + xs[1:])
    cy = 0.5 * (ys[:-1] + ys[1:])
    cgx, cgy = np.meshgrid(cx, cy)
    cz = 0.25 * (z_top[:-1, :-1] + z_top[:-1, 1:] + z_top[1:, :-1] + z_top[1:, 1:])
    center_verts = np.column_stack([cgx.ravel(), cgy.ravel(), cz.ravel()])
    c_off = rows * cols

    # Top triangles: CCW when viewed from +Z.
    rr, cc_idx = np.meshgrid(np.arange(cr, dtype=np.int64),
                             np.arange(cc, dtype=np.int64), indexing="ij")
    v00 = (rr * cols + cc_idx).ravel()
    v10 = (rr * cols + (cc_idx + 1)).ravel()
    v11 = ((rr + 1) * cols + (cc_idx + 1)).ravel()
    v01 = ((rr + 1) * cols + cc_idx).ravel()
    vc = c_off + np.arange(cr * cc, dtype=np.int64)
    top_tris = np.vstack([
        np.column_stack([v00, v10, vc]),
        np.column_stack([v10, v11, vc]),
        np.column_stack([v11, v01, vc]),
        np.column_stack([v01, v00, vc]),
    ])

    # Perimeter ring in CCW order around +Z: bottom row -> right col -> top row rev -> left col rev.
    perim_top_idx = np.concatenate([
        np.arange(cols, dtype=np.int64),
        np.arange(1, rows, dtype=np.int64) * cols + (cols - 1),
        (rows - 1) * cols + np.arange(cols - 2, -1, -1, dtype=np.int64),
        np.arange(rows - 2, 0, -1, dtype=np.int64) * cols,
    ])
    b_off = c_off + cr * cc
    bot_verts = grid_verts[perim_top_idx].copy()
    bot_verts[:, 2] = z_bot
    bot_center = np.array([[0.0 if center else 0.5 * lx,
                            0.0 if center else 0.5 * ly,
                            z_bot]], dtype=np.float64)
    bc_idx = b_off + n_perim

    p_i = np.arange(n_perim, dtype=np.int64)
    p_j = (p_i + 1) % n_perim
    ti, tj = perim_top_idx[p_i], perim_top_idx[p_j]
    bi, bj = b_off + p_i, b_off + p_j
    side_tris = np.vstack([
        np.column_stack([ti, bi, bj]),
        np.column_stack([ti, bj, tj]),
    ])
    bot_tris = np.column_stack([bj, bi, np.full(n_perim, bc_idx, dtype=np.int64)])

    all_verts = np.vstack([grid_verts, center_verts, bot_verts, bot_center])
    all_faces = np.vstack([top_tris, side_tris, bot_tris])
    mesh = oriented(Mesh(all_verts, all_faces, {
        "kind": "heightmap", "op": "heightmap",
        "heightmap": {"rows": int(rows), "cols": int(cols),
                      "dx_mm": float(dx * 1e3), "dy_mm": float(dy * 1e3),
                      "base_mm": float(base * 1e3)},
    }))
    _assert_closed_manifold(mesh, "heightmap", "heights")
    return mesh


def _heightmap_op(*, heights, dx, dy, base=1e-3, center=True, invert=False,
                  max_vertices=_DEFAULT_MAX_VERTS, max_triangles=_DEFAULT_MAX_TRIS) -> Mesh:
    return build_heightmap(
        heights, dx=dx, dy=dy, base=base, center=center, invert=invert,
        max_vertices=max_vertices, max_triangles=max_triangles,
    )


def _heightmap_prepare(args, scale, path):
    try:
        h = np.asarray(args.get("heights"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AttaError("bad_arg", "heights must be a 2D numeric array",
                        path=path + ".args.heights",
                        fix="pass [[z00,z01],[z10,z11]] in mm") from exc
    if h.ndim != 2 or h.shape[0] < 2 or h.shape[1] < 2:
        raise AttaError("bad_arg", "heights must have shape (rows>=2, cols>=2)",
                        path=path + ".args.heights",
                        fix="pass a 2D grid with at least 2 rows and 2 columns")
    if not np.isfinite(h).all():
        raise AttaError("bad_arg", "heights contain NaN or Inf",
                        path=path + ".args.heights", fix="remove non-finite values")
    if "cell_size" in args and "size" in args and args["cell_size"] is not None and args["size"] is not None:
        raise AttaError("bad_arg", "specify either cell_size or size, not both",
                        path=path + ".args", fix="remove one of cell_size or size")
    rows, cols = h.shape
    if args.get("size") is not None:
        sz = np.asarray(args["size"], dtype=np.float64).reshape(-1)
        if sz.shape != (2,) or not np.isfinite(sz).all() or np.any(sz <= 0.0):
            raise AttaError("bad_arg", "size must be [Lx, Ly] with positive numbers",
                            path=path + ".args.size", fix="pass [Lx, Ly] in mm")
        dx = float(sz[0]) * float(scale) / float(cols - 1)
        dy = float(sz[1]) * float(scale) / float(rows - 1)
    else:
        raw_cs = args.get("cell_size", 1.0)
        if isinstance(raw_cs, (list, tuple, np.ndarray)):
            arr = np.asarray(raw_cs, dtype=np.float64).reshape(-1)
            if arr.shape != (2,) or not np.isfinite(arr).all() or np.any(arr <= 0.0):
                raise AttaError("bad_arg", "cell_size must be > 0 or [dx, dy] > 0",
                                path=path + ".args.cell_size", fix="pass positive spacing in mm")
            dx, dy = float(arr[0]) * float(scale), float(arr[1]) * float(scale)
        else:
            cs = _pos_num(raw_cs, "cell_size", path + ".args.cell_size") * float(scale)
            dx = dy = cs
    base = _num(args.get("base", 1.0), "base", path + ".args.base") * float(scale)
    if base < 0.0:
        raise AttaError("bad_arg", "base must be >= 0", path=path + ".args.base",
                        fix="pass base >= 0 mm")
    return {
        "heights": h * float(scale),
        "dx": dx,
        "dy": dy,
        "base": base,
        "center": _flag(args.get("center", True)),
        "invert": _flag(args.get("invert", False)),
        "max_vertices": _int_bound(args.get("max_vertices", _DEFAULT_MAX_VERTS),
                                   "max_vertices", path + ".args.max_vertices", 4, _MAX_VERTS_CAP),
        "max_triangles": _int_bound(args.get("max_triangles", _DEFAULT_MAX_TRIS),
                                    "max_triangles", path + ".args.max_triangles", 4, _MAX_TRIS_CAP),
    }


# ---------------------------------------------------------------- 5. deform

def _unique_undirected_edges(faces: np.ndarray) -> np.ndarray:
    he = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(he, axis=1), axis=0)


def _subdivide_to_max_edge(mesh: Mesh, max_edge: float, *,
                           max_vertices: int, max_triangles: int,
                           max_rounds: int = 16) -> Mesh:
    """Iteratively split edges longer than ``max_edge`` with strict pre-allocation budgets."""
    if len(mesh.faces) > max_triangles or len(mesh.verts) > max_vertices:
        raise AttaError("budget_exceeded",
                        "deform input exceeds budget (verts=%d/%d, tris=%d/%d)"
                        % (len(mesh.verts), max_vertices, len(mesh.faces), max_triangles),
                        path="max_triangles", fix="simplify input or raise budgets")

    cur = mesh.copy()
    for _ in range(max_rounds):
        v = np.asarray(cur.verts, dtype=np.float64)
        f = np.asarray(cur.faces, dtype=np.int64)
        edges = _unique_undirected_edges(f)
        if len(edges) == 0:
            break
        lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
        long_mask = lengths > max_edge
        if not long_mask.any():
            return cur
        long_edges = edges[long_mask]
        n_new_verts = len(v) + len(long_edges)
        if n_new_verts > max_vertices:
            raise AttaError(
                "budget_exceeded",
                "auto-subdivision would create %d vertices, exceeding max_vertices=%d"
                % (n_new_verts, max_vertices),
                path="max_edge",
                fix="increase max_edge or raise max_vertices up to %d" % _MAX_VERTS_CAP,
                details={"needed_vertices": n_new_verts, "max_vertices": int(max_vertices)},
            )

        mid = {}
        extra = []
        for a, b in long_edges:
            key = (int(a), int(b))
            mid[key] = len(v) + len(extra)
            extra.append((v[a] + v[b]) * 0.5)

        # Exact predicted triangle count before allocating face list.
        new_tris_est = 0
        tri_splits = []
        for row in f:
            a, b, c = int(row[0]), int(row[1]), int(row[2])
            ab = mid.get((min(a, b), max(a, b)))
            bc = mid.get((min(b, c), max(b, c)))
            ca = mid.get((min(c, a), max(c, a)))
            cnt = (ab is not None) + (bc is not None) + (ca is not None)
            new_tris_est += cnt + 1
            tri_splits.append((a, b, c, ab, bc, ca))
        if new_tris_est > max_triangles:
            raise AttaError(
                "budget_exceeded",
                "auto-subdivision would create %d triangles, exceeding max_triangles=%d"
                % (new_tris_est, max_triangles),
                path="max_edge",
                fix="increase max_edge or raise max_triangles up to %d" % _MAX_TRIS_CAP,
                details={"needed_triangles": new_tris_est, "max_triangles": int(max_triangles)},
            )

        new_faces = []
        for a, b, c, ab, bc, ca in tri_splits:
            if ab is not None and bc is not None and ca is not None:
                new_faces.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
            elif ab is not None and bc is not None:
                new_faces.extend([(a, ab, bc), (a, bc, c), (ab, b, bc)])
            elif bc is not None and ca is not None:
                new_faces.extend([(b, bc, ca), (b, ca, a), (bc, c, ca)])
            elif ca is not None and ab is not None:
                new_faces.extend([(c, ca, ab), (c, ab, b), (ca, a, ab)])
            elif ab is not None:
                new_faces.extend([(a, ab, c), (ab, b, c)])
            elif bc is not None:
                new_faces.extend([(b, bc, a), (bc, c, a)])
            elif ca is not None:
                new_faces.extend([(c, ca, b), (ca, a, b)])
            else:
                new_faces.append((a, b, c))
        cur = Mesh(np.vstack([v, np.asarray(extra, dtype=np.float64)]),
                   np.asarray(new_faces, dtype=np.int64), dict(cur.meta))
    return cur


def _bernstein_basis(n: int, t: np.ndarray) -> np.ndarray:
    """Evaluate Bernstein polynomials B_{i, n}(t) for i=0..n -> shape (len(t), n+1)."""
    t = np.clip(t, 0.0, 1.0)
    one_m = 1.0 - t
    cols = []
    for i in range(n + 1):
        coeff = float(math.comb(n, i))
        cols.append(coeff * (t ** i) * (one_m ** (n - i)))
    return np.column_stack(cols)


def _apply_ffd_lattice(v: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                       shape: tuple[int, int, int], offsets: np.ndarray) -> np.ndarray:
    nx, ny, nz = shape
    span = np.maximum(hi - lo, 1e-12)
    u = (v[:, 0] - lo[0]) / span[0]
    w_y = (v[:, 1] - lo[1]) / span[1]
    w_z = (v[:, 2] - lo[2]) / span[2]
    bx = _bernstein_basis(nx - 1, u)      # (N, nx)
    by = _bernstein_basis(ny - 1, w_y)    # (N, ny)
    bz = _bernstein_basis(nz - 1, w_z)    # (N, nz)
    disp = np.einsum("ni,nj,nk,ijkd->nd", bx, by, bz, offsets)
    return v + disp


def deform_mesh(mesh: Mesh, *, mode: str = "twist", axis: str = "z",
                bend_axis: str | None = None, angle_deg: float = 0.0,
                start_scale: float = 1.0, end_scale: float = 1.0,
                lattice: dict | None = None, max_edge: float | None = None,
                max_vertices: int = _DEFAULT_MAX_VERTS,
                max_triangles: int = _DEFAULT_MAX_TRIS) -> Mesh:
    """Apply smooth axis or FFD deformation to a closed mesh with auto-subdivision (metres)."""
    _check_input_mesh(mesh, "mesh", "inputs[0]")
    mode = str(mode or "twist").lower()
    if mode not in ("twist", "bend", "taper", "lattice"):
        raise AttaError("bad_arg", "deform mode must be twist, bend, taper or lattice",
                        path="mode", fix="use twist, bend, taper or lattice")
    ax = _axis(axis, "axis", "axis")
    ax_i = "xyz".index(ax)

    bb = mesh.bbox()
    diag = float(bb["diag"])
    if diag <= 1e-12:
        raise AttaError("deform_degenerate", "input mesh has zero bounding box",
                        path="inputs[0]", fix="pass a non-degenerate 3D solid")

    target_edge = (float(max_edge) if max_edge is not None
                   else max(diag / 28.0, 1e-6))
    if not math.isfinite(target_edge) or target_edge <= 0.0:
        raise AttaError("bad_arg", "max_edge must be positive", path="max_edge",
                        fix="pass max_edge > 0 mm")

    sub = _subdivide_to_max_edge(mesh, target_edge,
                                 max_vertices=max_vertices, max_triangles=max_triangles)
    v = np.asarray(sub.verts, dtype=np.float64).copy()
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    center = 0.5 * (lo + hi)
    span = float(hi[ax_i] - lo[ax_i])
    if span <= 1e-12:
        raise AttaError("deform_degenerate", "mesh has zero extent along %s axis" % ax,
                        path="axis", fix="choose an axis along which the mesh has positive length")

    u_i, v_i = [i for i in range(3) if i != ax_i]
    if mode == "twist":
        ang = math.radians(float(angle_deg))
        if abs(ang) > 1e-12:
            t = (v[:, ax_i] - lo[ax_i]) / span
            a = ang * (t - 0.5)
            ca, sa = np.cos(a), np.sin(a)
            du = v[:, u_i] - center[u_i]
            dv = v[:, v_i] - center[v_i]
            v[:, u_i] = center[u_i] + ca * du - sa * dv
            v[:, v_i] = center[v_i] + sa * du + ca * dv
    elif mode == "bend":
        bax = _axis(bend_axis, "bend_axis", "bend_axis") if bend_axis is not None else "xyz"[u_i]
        b_i = "xyz".index(bax)
        if b_i == ax_i:
            raise AttaError("bad_arg", "bend_axis (%s) must differ from axis (%s)" % (bax, ax),
                            path="bend_axis", fix="choose a perpendicular bend_axis")
        ang = math.radians(float(angle_deg))
        if abs(ang) > 1e-12:
            R = span / ang
            s = v[:, ax_i] - center[ax_i]
            db = v[:, b_i] - center[b_i]
            r_ratio = 1.0 - db / R
            if float(r_ratio.min()) <= 1e-6:
                raise AttaError(
                    "deform_self_intersection",
                    "bend angle %.3g deg folds the inner radius through itself" % float(angle_deg),
                    path="angle_deg",
                    fix="reduce |angle_deg| so bend radius |R| > half the thickness along %s" % bax,
                )
            theta = s / R
            r = R - db
            v[:, ax_i] = center[ax_i] + r * np.sin(theta)
            v[:, b_i] = center[b_i] + R - r * np.cos(theta)
    elif mode == "taper":
        s0, s1 = float(start_scale), float(end_scale)
        if not (math.isfinite(s0) and math.isfinite(s1)) or s0 <= 0.0 or s1 <= 0.0:
            raise AttaError("bad_arg", "start_scale and end_scale must be > 0",
                            path="start_scale", fix="use positive scale factors, e.g. 1.0 and 0.5")
        t = (v[:, ax_i] - lo[ax_i]) / span
        s = s0 + (s1 - s0) * t
        v[:, u_i] = center[u_i] + (v[:, u_i] - center[u_i]) * s
        v[:, v_i] = center[v_i] + (v[:, v_i] - center[v_i]) * s
    elif mode == "lattice":
        if not isinstance(lattice, dict):
            raise AttaError("bad_arg", "deform mode='lattice' requires lattice={shape, offsets}",
                            path="lattice", fix='pass {"shape":[2,2,2],"offsets":[[...],...]}')
        shape = lattice["shape"]
        offsets = lattice["offsets"]
        v = _apply_ffd_lattice(v, lo, hi, shape, offsets)

    if not np.isfinite(v).all():
        raise AttaError("deform_failed", "deformed vertices became non-finite",
                        path="mode", fix="reduce deformation magnitude")
    out = oriented(Mesh(v, sub.faces.copy(), dict(sub.meta)))
    _assert_closed_manifold(out, "deform", "mode")
    out.meta.update({
        "kind": "deform", "op": "deform",
        "deform": {"mode": mode, "axis": ax, "max_edge_mm": float(target_edge * 1e3),
                   "subdivided_tris": int(len(out.faces))},
    })
    return out


def _deform_op(*, mesh, mode, axis="z", bend_axis=None, angle_deg=0.0,
               start_scale=1.0, end_scale=1.0, lattice=None, max_edge=None,
               max_vertices=_DEFAULT_MAX_VERTS, max_triangles=_DEFAULT_MAX_TRIS) -> Mesh:
    return deform_mesh(
        mesh, mode=mode, axis=axis, bend_axis=bend_axis, angle_deg=angle_deg,
        start_scale=start_scale, end_scale=end_scale, lattice=lattice,
        max_edge=max_edge, max_vertices=max_vertices, max_triangles=max_triangles,
    )


def _deform_prepare(args, scale, path):
    mode = str(args.get("mode") or "").lower()
    if mode not in ("twist", "bend", "taper", "lattice"):
        raise AttaError("bad_arg", "mode must be twist, bend, taper or lattice",
                        path=path + ".args.mode", fix="use twist, bend, taper or lattice")
    ax = _axis(args.get("axis", "z"), "axis", path + ".args.axis")
    bax = (_axis(args["bend_axis"], "bend_axis", path + ".args.bend_axis")
           if args.get("bend_axis") is not None else None)
    angle_deg = _num(args.get("angle_deg", 0.0), "angle_deg", path + ".args.angle_deg")
    s0 = _pos_num(args.get("start_scale", 1.0), "start_scale", path + ".args.start_scale")
    s1 = _pos_num(args.get("end_scale", 1.0), "end_scale", path + ".args.end_scale")
    max_edge = (
        _pos_num(args["max_edge"], "max_edge", path + ".args.max_edge") * float(scale)
        if args.get("max_edge") is not None else None
    )
    lat_out = None
    if mode == "lattice":
        raw_lat = args.get("lattice")
        if not isinstance(raw_lat, dict):
            raise AttaError("bad_arg", "lattice must be an object with shape and offsets",
                            path=path + ".args.lattice",
                            fix='pass {"shape":[2,2,2],"offsets":[[0,0,0],...]}')
        raw_shape = raw_lat.get("shape")
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 3:
            raise AttaError("bad_arg", "lattice.shape must be [nx, ny, nz]",
                            path=path + ".args.lattice.shape", fix="use [2, 2, 2]")
        shape = tuple(_int_bound(raw_shape[k], "shape[%d]" % k,
                                 path + ".args.lattice.shape", 2, 8)
                      for k in range(3))
        if "offsets" not in raw_lat:
            raise AttaError("bad_arg", "lattice.offsets is required",
                            path=path + ".args.lattice", fix="add offsets in mm")
        try:
            offs = np.asarray(raw_lat["offsets"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise AttaError("bad_arg", "lattice.offsets must be numeric",
                            path=path + ".args.lattice.offsets",
                            fix="pass (nx*ny*nz, 3) or (nx, ny, nz, 3) offsets in mm") from exc
        if offs.ndim == 2 and offs.shape == (math.prod(shape), 3):
            offs = offs.reshape(*shape, 3)
        elif offs.shape != (*shape, 3):
            raise AttaError("bad_arg",
                            "lattice.offsets shape %r does not match %r" % (offs.shape, (*shape, 3)),
                            path=path + ".args.lattice.offsets",
                            fix="pass %d [dx,dy,dz] offsets in mm" % math.prod(shape))
        if not np.isfinite(offs).all():
            raise AttaError("bad_arg", "lattice.offsets contain NaN or Inf",
                            path=path + ".args.lattice.offsets", fix="remove non-finite offsets")
        lat_out = {"shape": shape, "offsets": offs * float(scale)}
    return {
        "mode": mode, "axis": ax, "bend_axis": bax, "angle_deg": angle_deg,
        "start_scale": s0, "end_scale": s1, "lattice": lat_out, "max_edge": max_edge,
        "max_vertices": _int_bound(args.get("max_vertices", _DEFAULT_MAX_VERTS),
                                   "max_vertices", path + ".args.max_vertices", 4, _MAX_VERTS_CAP),
        "max_triangles": _int_bound(args.get("max_triangles", _DEFAULT_MAX_TRIS),
                                    "max_triangles", path + ".args.max_triangles", 4, _MAX_TRIS_CAP),
    }


# ---------------------------------------------------------------- 6. morph

def _signed_distance_grid(shape, org: np.ndarray, spacing: float, mesh: Mesh) -> np.ndarray:
    """Sample exact signed distance to a closed triangle mesh on a regular grid."""
    from scipy.spatial import cKDTree
    from .offset_ops import _inside_grid, _point_tri_dist, _tri_arrays

    _, _, tri = _tri_arrays(mesh)
    nx, ny, nz = (int(x) for x in shape)
    xs = org[0] + spacing * np.arange(nx)
    ys = org[1] + spacing * np.arange(ny)
    zs = org[2] + spacing * np.arange(nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    cen = tri.mean(axis=1)
    k = int(min(24, len(tri)))
    _, idx = cKDTree(cen).query(pts, k=k, workers=1)
    if k == 1:
        idx = idx[:, None]
    dist = np.empty(len(pts), dtype=np.float64)
    for s in range(0, len(pts), 2048):
        p = pts[s:s + 2048]
        ii = idx[s:s + 2048]
        dd = _point_tri_dist(p[:, None, :], tri[ii, 0], tri[ii, 1], tri[ii, 2])
        dist[s:s + 2048] = dd.min(axis=1)

    inside = _inside_grid(shape, org, spacing, mesh, axis=2).ravel()
    return np.where(inside, -dist, dist).reshape(shape)


def morph_meshes(a: Mesh, b: Mesh, *, t: float = 0.5, resolution: int = 40,
                 align_centers: bool = False,
                 max_samples: int = _DEFAULT_MAX_SAMPLES,
                 max_triangles: int = _DEFAULT_MAX_TRIS) -> Mesh:
    """Interpolate between two closed meshes via linear SDF blend at t in [0, 1] (metres)."""
    _check_input_mesh(a, "a", "inputs[0]")
    _check_input_mesh(b, "b", "inputs[1]")
    tt = _num(t, "t", "t")
    if not (0.0 <= tt <= 1.0):
        raise AttaError("bad_arg", "morph t must be in [0, 1], got %.4g" % tt,
                        path="t", fix="pass t between 0.0 and 1.0")

    b_work = b
    if align_centers:
        ca = np.asarray(a.bbox()["center"], dtype=np.float64)
        cb = np.asarray(b.bbox()["center"], dtype=np.float64)
        b_work = b.transform(translate=tuple(ca - cb))

    if tt == 0.0:
        out = a.copy()
        out.meta.update({"kind": "morph", "op": "morph",
                         "morph": {"t": 0.0, "method": "endpoint_a", "exact": True}})
        return out
    if tt == 1.0:
        out = b_work.copy()
        out.meta.update({"kind": "morph", "op": "morph",
                         "morph": {"t": 1.0, "method": "endpoint_b", "exact": True}})
        return out

    from .. import sdf, sdfkit

    lo = np.minimum(a.verts.min(axis=0), b_work.verts.min(axis=0))
    hi = np.maximum(a.verts.max(axis=0), b_work.verts.max(axis=0))
    span = float((hi - lo).max())
    if span <= 1e-12:
        raise AttaError("morph_degenerate", "morph inputs have zero bounding box",
                        path="inputs", fix="pass non-degenerate 3D solids")

    req_spacing = span / float(resolution)
    spacing = sdf.fit_spacing(lo, hi, req_spacing, max_samples=max_samples)
    X, Y, Z, org = sdfkit.grid(lo, hi, spacing, max_samples=max_samples)
    shape = X.shape
    del X, Y, Z

    phi_a = _signed_distance_grid(shape, org, spacing, a)
    phi_b = _signed_distance_grid(shape, org, spacing, b_work)
    phi = (1.0 - tt) * phi_a + tt * phi_b
    if float(phi.min()) >= 0.0:
        raise AttaError("morph_empty",
                        "interpolated SDF has no interior at t=%.3g (shapes are too far apart)" % tt,
                        path="t",
                        fix="pass align_centers=true or move inputs closer so they overlap")

    out = sdfkit.extract(phi, org, spacing, max_triangles=max_triangles)
    out = oriented(out)
    _assert_closed_manifold(out, "morph", "inputs")
    out.meta.update({
        "kind": "morph", "op": "morph", "exact": False,
        "morph": {
            "t": float(tt),
            "method": "sdf_linear_blend",
            "resolution": int(resolution),
            "spacing_mm": float(spacing * 1e3),
            "coarsened": bool(spacing > req_spacing),
            "align_centers": bool(align_centers),
        },
    })
    return out


def _morph_op(*, meshes, t=0.5, resolution=40, align_centers=False,
              max_samples=_DEFAULT_MAX_SAMPLES, max_triangles=_DEFAULT_MAX_TRIS) -> Mesh:
    if len(meshes) != 2:
        raise AttaError("bad_arg", "morph requires exactly 2 input meshes, got %d" % len(meshes),
                        path="inputs", fix='pass inputs=["<a>", "<b>"]')
    return morph_meshes(
        meshes[0], meshes[1], t=t, resolution=resolution,
        align_centers=align_centers, max_samples=max_samples,
        max_triangles=max_triangles,
    )


def _morph_prepare(args, scale, path):
    t = _num(args.get("t", 0.5), "t", path + ".args.t")
    if not (0.0 <= t <= 1.0):
        raise AttaError("bad_arg", "t must be in [0, 1]", path=path + ".args.t",
                        fix="use t between 0.0 and 1.0")
    return {
        "t": t,
        "resolution": _int_bound(args.get("resolution", 40), "resolution",
                                 path + ".args.resolution", 8, 128),
        "align_centers": _flag(args.get("align_centers", False)),
        "max_samples": _int_bound(args.get("max_samples", _DEFAULT_MAX_SAMPLES),
                                  "max_samples", path + ".args.max_samples", 343, _MAX_SAMPLES_CAP),
        "max_triangles": _int_bound(args.get("max_triangles", _DEFAULT_MAX_TRIS),
                                    "max_triangles", path + ".args.max_triangles", 4, _MAX_TRIS_CAP),
    }


# ---------------------------------------------------------------- 7. projection_extrude

def projection_extrude(mesh: Mesh, *, thickness: float, cut: bool = False,
                       cut_offset: float = 0.0, axis: str = "z",
                       center: bool = False,
                       max_triangles: int = _DEFAULT_MAX_TRIS,
                       max_input_triangles: int = _DEFAULT_MAX_TRIS) -> Mesh:
    """Project (or slice when ``cut=True``) a closed mesh onto 2D and re-extrude by ``thickness`` (metres)."""
    _check_input_mesh(mesh, "mesh", "inputs[0]")
    if len(mesh.faces) > max_input_triangles:
        raise AttaError("budget_exceeded",
                        "projection_extrude input has %d triangles, exceeding max_input_triangles=%d"
                        % (len(mesh.faces), max_input_triangles),
                        path="max_input_triangles", fix="simplify input or raise max_input_triangles")
    t = float(thickness)
    if not math.isfinite(t) or t <= 0.0:
        raise AttaError("bad_arg", "thickness must be > 0", path="thickness",
                        fix="pass a positive extrusion thickness in mm")
    ax = _axis(axis, "axis", "axis")
    c_off = float(cut_offset)
    if not math.isfinite(c_off):
        raise AttaError("bad_arg", "cut_offset must be finite", path="cut_offset",
                        fix="pass a finite cut plane offset in mm")

    # Rotate input so the projection axis aligns with +Z, then rotate result back.
    to_z = {"z": (0.0, 0.0, 0.0), "x": (0.0, -90.0, 0.0), "y": (90.0, 0.0, 0.0)}[ax]
    from_z = {"z": (0.0, 0.0, 0.0), "x": (0.0, 90.0, 0.0), "y": (-90.0, 0.0, 0.0)}[ax]
    work = mesh if ax == "z" else mesh.transform(rotate_deg=to_z)

    from ..boolean import _from_manifold, _to_manifold
    try:
        man = _to_manifold(work)
        cs = man.slice(c_off) if cut else man.project()
    except Exception as exc:
        raise AttaError("projection_failed", "2D projection failed: %s" % exc,
                        path="inputs[0]", fix="ensure input mesh is a clean manifold solid") from exc

    polys = cs.to_polygons()
    if cs.is_empty() or not polys or float(cs.area()) <= 1e-14:
        raise AttaError("projection_empty",
                        "projection(cut=%s, cut_offset=%.4g mm) produced an empty 2D outline"
                        % (bool(cut), c_off * 1e3),
                        path="cut_offset" if cut else "inputs[0]",
                        fix=("adjust cut_offset so the slice plane intersects the solid"
                             if cut else "ensure input solid has positive projected area"))

    ext = cs.extrude(t)
    out = _from_manifold(ext)
    if center:
        out = out.transform(translate=(0.0, 0.0, -0.5 * t))
    if ax != "z":
        out = out.transform(rotate_deg=from_z)
    out = oriented(out)

    if len(out.faces) > max_triangles:
        raise AttaError("budget_exceeded",
                        "projection_extrude produced %d triangles, exceeding max_triangles=%d"
                        % (len(out.faces), max_triangles),
                        path="max_triangles", fix="simplify input mesh or raise max_triangles")

    _assert_closed_manifold(out, "projection_extrude", "inputs[0]")
    out.meta.update({
        "kind": "projection_extrude", "op": "projection_extrude",
        "projection_extrude": {
            "cut": bool(cut), "cut_offset_mm": float(c_off * 1e3),
            "thickness_mm": float(t * 1e3), "axis": ax,
            "n_polygons": len(polys), "area_mm2": float(cs.area() * 1e6),
        },
    })
    return out


def _projection_extrude_op(*, mesh, thickness, cut=False, cut_offset=0.0,
                           axis="z", center=False,
                           max_triangles=_DEFAULT_MAX_TRIS,
                           max_input_triangles=_DEFAULT_MAX_TRIS) -> Mesh:
    return projection_extrude(
        mesh, thickness=thickness, cut=cut, cut_offset=cut_offset, axis=axis,
        center=center, max_triangles=max_triangles,
        max_input_triangles=max_input_triangles,
    )


def _projection_extrude_prepare(args, scale, path):
    t = _pos_num(args.get("thickness"), "thickness", path + ".args.thickness") * float(scale)
    c_off = _num(args.get("cut_offset", 0.0), "cut_offset", path + ".args.cut_offset") * float(scale)
    return {
        "thickness": t,
        "cut": _flag(args.get("cut", False)),
        "cut_offset": c_off,
        "axis": _axis(args.get("axis", "z"), "axis", path + ".args.axis"),
        "center": _flag(args.get("center", False)),
        "max_triangles": _int_bound(args.get("max_triangles", _DEFAULT_MAX_TRIS),
                                    "max_triangles", path + ".args.max_triangles", 4, _MAX_TRIS_CAP),
        "max_input_triangles": _int_bound(args.get("max_input_triangles", _DEFAULT_MAX_TRIS),
                                          "max_input_triangles", path + ".args.max_input_triangles",
                                          4, _MAX_TRIS_CAP),
    }


# ---------------------------------------------------------------- registration

def register():
    """Register the seven generic feature-graph ops."""
    _hull_op.prepare = _hull_prepare
    register_op("hull", {
        "required": (),
        "optional": ("max_vertices", "max_triangles"),
        "min_inputs": 1,
        "doc": "Convex hull of one or more input meshes (oriented closed manifold).",
        "args": {"max_vertices": ("int", 4), "max_triangles": ("int", 4)},
    }, _hull_op)

    _minkowski_op.prepare = _minkowski_prepare
    register_op("minkowski", {
        "required": (),
        "optional": ("method", "resolution", "max_vertices", "max_triangles",
                     "max_samples", "max_face_product"),
        "min_inputs": 2,
        "doc": "Minkowski sum A (+) B: exact pairwise-vertex hull for convex+convex, "
               "exact manifold sum for small non-convex meshes, or bounded SDF convolution "
               "(meta.exact=false) when method='sdf' or face product exceeds budget.",
        "args": {"method": "enum:auto|exact|sdf", "resolution": ("int", 8),
                 "max_vertices": ("int", 4), "max_triangles": ("int", 4),
                 "max_samples": ("int", 343), "max_face_product": ("int", 1)},
    }, _minkowski_op)

    _polyhedron_op.prepare = _polyhedron_prepare
    register_op("polyhedron", {
        "required": ("points", "faces"),
        "optional": ("convexity", "fix_winding", "weld_tol", "max_vertices", "max_triangles"),
        "arity": 0,
        "doc": "OpenSCAD-style polyhedron from points (mm) and face index loops; validates "
               "closure and manifoldness and fixes inconsistent/inward winding when fix_winding=true.",
        "args": {"points": "mat3", "faces": "anything", "convexity": ("int", 1),
                 "fix_winding": "bool", "weld_tol": "length",
                 "max_vertices": ("int", 4), "max_triangles": ("int", 4)},
    }, _polyhedron_op)

    _heightmap_op.prepare = _heightmap_prepare
    register_op("heightmap", {
        "required": ("heights",),
        "optional": ("cell_size", "size", "base", "center", "invert",
                     "max_vertices", "max_triangles"),
        "arity": 0,
        "doc": "OpenSCAD surface()-equivalent: 2D grid of heights (mm) turned into a "
               "closed watertight solid over a flat base at z=-base.",
        "args": {"heights": "anything", "cell_size": "anything", "size": "anything",
                 "base": "length", "center": "bool", "invert": "bool",
                 "max_vertices": ("int", 4), "max_triangles": ("int", 4)},
    }, _heightmap_op)

    _deform_op.prepare = _deform_prepare
    register_op("deform", {
        "required": ("mode",),
        "optional": ("axis", "bend_axis", "angle_deg", "start_scale", "end_scale",
                     "lattice", "max_edge", "max_vertices", "max_triangles"),
        "arity": 1,
        "doc": "Smooth mesh deformation (twist, bend, taper, or Bernstein FFD lattice) "
               "with bounded auto-subdivision to max_edge (mm).",
        "args": {"mode": "enum:twist|bend|taper|lattice", "axis": "enum:x|y|z",
                 "bend_axis": "enum:x|y|z", "angle_deg": "number",
                 "start_scale": "number", "end_scale": "number",
                 "lattice": "anything", "max_edge": "length",
                 "max_vertices": ("int", 4), "max_triangles": ("int", 4)},
    }, _deform_op)

    _morph_op.prepare = _morph_prepare
    register_op("morph", {
        "required": (),
        "optional": ("t", "resolution", "align_centers", "max_samples", "max_triangles"),
        "min_inputs": 2,
        "doc": "Interpolate two closed meshes via linear SDF blend at t in [0, 1].",
        "args": {"t": "number", "resolution": ("int", 8), "align_centers": "bool",
                 "max_samples": ("int", 343), "max_triangles": ("int", 4)},
    }, _morph_op)

    _projection_extrude_op.prepare = _projection_extrude_prepare
    register_op("projection_extrude", {
        "required": ("thickness",),
        "optional": ("cut", "cut_offset", "axis", "center",
                     "max_triangles", "max_input_triangles"),
        "arity": 1,
        "doc": "OpenSCAD projection(cut=true/false) followed by linear extrusion by thickness (mm).",
        "args": {"thickness": "length", "cut": "bool", "cut_offset": "length",
                 "axis": "enum:x|y|z", "center": "bool",
                 "max_triangles": ("int", 4), "max_input_triangles": ("int", 4)},
    }, _projection_extrude_op)


register()
