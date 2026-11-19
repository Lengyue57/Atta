# -*- coding: utf-8 -*-
"""Metrology: turn a triangle mesh into millimetre facts.

The mesh lives in metres; every public length here is millimetres.  Volume,
area, centroid and bbox are exact for the given triangulation.  Wall
thickness, dihedral and curvature are sampled or derived diagnostics, so all
results are marked ``exact: False``.
"""

from __future__ import annotations

import numpy as np

from . import mesh as M

EPS = 1e-12
_DET_EPS = 1e-14
_RAY_RAYS = 32
#: Triangles per ray-block.  Measured on a 25 482-triangle mug (wall_thickness,
#: best of 3): 4096 -> 856 ms, 512 -> 763 ms (~1.1x), with min/p05/p95 and the
#: hit count bit-identical.  512 keeps the (rays x triangles) working set in
#: cache.  Do not shrink it further without measuring: per-block overhead starts
#: to dominate, and this number is not a guess.
_RAY_TRIS = 512


# ---------------------------------------------------------------- helpers

def _tri_arrays(v: np.ndarray, f: np.ndarray):
    return v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]


def _face_normals(v: np.ndarray, f: np.ndarray):
    a, b, c = _tri_arrays(v, f)
    ab, ac = b - a, c - a
    scale = np.maximum(np.max(np.abs(ab), axis=1), np.max(np.abs(ac), axis=1))
    safe = np.where(scale > 0, scale, 1.0)
    n = np.cross(ab / safe[:, None], ac / safe[:, None])
    ln = np.linalg.norm(n, axis=1)
    out = np.zeros_like(n)
    ok = np.isfinite(ln) & (ln > np.finfo(float).tiny)
    out[ok] = n[ok] / ln[ok, None]
    return out, 0.5 * ln * scale**2


def _signed_volume(v: np.ndarray, f: np.ndarray) -> float:
    if len(f) == 0:
        return 0.0
    # Translate the tetrahedral fan near the mesh before integrating. Without
    # this, a small part at a large world coordinate loses all volume digits.
    origin = v[f[0, 0]]
    a, b, c = _tri_arrays(v, f)
    a, b, c = a - origin, b - origin, c - origin
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _edge_table(v: np.ndarray, f: np.ndarray):
    """Undirected edges with face ownership, vectorised.

    Returns ``(reps, counts, sorted_order, starts, face_ids)`` where
    ``sorted_order`` indexes the 3M directed-edge array sorted by edge key.
    """
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    face_ids = np.tile(np.arange(len(f), dtype=np.int64), 3)
    base = max(1, int(len(v)))
    key = np.minimum(e[:, 0], e[:, 1]).astype(np.int64) * base + \
        np.maximum(e[:, 0], e[:, 1]).astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    uniq, starts, counts = np.unique(ks, return_index=True, return_counts=True)
    reps = e[order[starts]]
    return reps, counts, order, starts, face_ids


# ---------------------------------------------------------------- rays

def _ray_block(o, d, a, e1, e2, t_min):
    """Moller-Trumbore for a block of rays ``(R,3)`` and tris ``(T,3)``."""
    h = np.cross(d[:, None, :], e2[None, :, :])
    det = np.einsum("tj,rtj->rt", e1, h)
    ok = np.abs(det) > _DET_EPS
    inv = np.zeros_like(det)
    np.divide(1.0, det, out=inv, where=ok)
    s = o[:, None, :] - a[None, :, :]
    u = np.einsum("rtj,rtj->rt", s, h) * inv
    q = np.cross(s, e1[None, :, :])
    vv = np.einsum("rtj,rj->rt", q, d) * inv
    t = np.einsum("tj,rtj->rt", e2, q) * inv
    hit = ok & (u >= -1e-9) & (vv >= -1e-9) & (u + vv <= 1.0 + 1e-9) & (t > t_min)
    return t, hit


def _ray_reduce(origins, dirs, a, e1, e2, mode="min", t_min=1e-7,
                normals=None, min_dot=None):
    """Reduce ray/triangle hits per ray: ``min`` distance or ``count``.

    ``normals`` + ``min_dot`` filter hits whose surface faces away from the
    ray (a grazing hit is not a wall thickness).  For an inward wall ray the
    opposite surface normal is roughly parallel to the ray, so ``min_dot=0.5``
    keeps real walls and rejects bevel/lip slivers.
    """
    origins = np.ascontiguousarray(origins, dtype=np.float64)
    dirs = np.ascontiguousarray(dirs, dtype=np.float64)
    n = len(origins)
    out = (np.full(n, np.inf) if mode == "min"
           else np.zeros(n, dtype=np.int64))
    if n == 0 or len(a) == 0:
        return out
    for s in range(0, n, _RAY_RAYS):
        o = origins[s:s + _RAY_RAYS]
        d = dirs[s:s + _RAY_RAYS]
        for k in range(0, len(a), _RAY_TRIS):
            t, hit = _ray_block(o, d, a[k:k + _RAY_TRIS],
                                e1[k:k + _RAY_TRIS], e2[k:k + _RAY_TRIS], t_min)
            if mode == "min":
                if normals is not None and min_dot is not None:
                    align = np.einsum("tj,rj->rt", normals[k:k + _RAY_TRIS],
                                      d)
                    hit = hit & (align > float(min_dot))
                tt = np.where(hit, t, np.inf)
                out[s:s + len(o)] = np.minimum(out[s:s + len(o)],
                                               tt.min(axis=1))
            else:
                out[s:s + len(o)] += hit.sum(axis=1)
    return out


def _tri_edges(v, f):
    a, b, c = _tri_arrays(v, f)
    return a, b - a, c - a


# ---------------------------------------------------------------- measures

def _display(value, decimals):
    """Keep historical rounding at product scale without erasing tiny geometry."""
    value = float(value)
    if 0 < abs(value) < 10.0 ** (2 - decimals):
        return float(format(value, ".12g"))
    return round(value, decimals)


def bbox_mm(mesh: M.Mesh) -> dict:
    b = M.bbox(mesh)
    out = {k: [_display(float(x) * 1000.0, 4) for x in b[k]]
           for k in ("min", "max", "size", "center")}
    out["diag_mm"] = _display(float(b["diag"]) * 1000.0, 4)
    return out


def volume_mm3(mesh: M.Mesh) -> float:
    return _display(float(M.volume(mesh)) * 1e9, 6)


def area_mm2(mesh: M.Mesh) -> float:
    return _display(float(M.area(mesh)) * 1e6, 6)


def centroid_mm(mesh: M.Mesh) -> list[float]:
    return [_display(float(x) * 1000.0, 4) for x in M.centroid(mesh)]


def mass_properties(mesh: M.Mesh, density_kg_m3: float | None = None) -> dict:
    vol_m3 = float(M.volume(mesh))
    out = {
        "volume_mm3": _display(vol_m3 * 1e9, 6),
        "centroid_mm": centroid_mm(mesh),
        "density_kg_m3": (None if density_kg_m3 is None
                          else float(density_kg_m3)),
        "mass_kg": None,
        "exact": False,
    }
    if density_kg_m3 is not None:
        out["mass_kg"] = float(vol_m3 * float(density_kg_m3))
    return out


# ---------------------------------------------------------------- wall

def concave_edge_count(mesh: M.Mesh) -> int:
    """Number of reflex (concave) manifold edges.

    A convex solid -- cube, sphere, cylinder, cone, plate -- has none: every
    two-face edge bends away from the material.  A part with a cavity, a bore,
    or a pocket necessarily has at least one.  This is the honest discriminator
    between "this part has walls" and "this part is a lump of material".
    """
    v = np.asarray(mesh.verts, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if len(f) == 0:
        return 0
    reps, counts, order, starts, face_ids = _edge_table(v, f)
    two = np.where(counts == 2)[0]
    if len(two) == 0:
        return 0
    f0 = face_ids[order[starts[two]]]
    f1 = face_ids[order[starts[two] + 1]]
    rep = reps[two]
    f1v = f[f1]
    # The third vertex of f1 (the one not on the shared edge).
    mask = (f1v != rep[:, 0:1]) & (f1v != rep[:, 1:2])
    other = v[f1v[np.arange(len(f1v)), mask.argmax(axis=1)]]
    fn, _ = _face_normals(v, f)
    a, _, _ = _tri_arrays(v, f)
    # Positive => the neighbour lies above f0's plane => reflex/concave edge.
    d = np.einsum("ij,ij->i", fn[f0], other - a[f0])
    diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    tol = max(1e-12, diag * 1e-7)
    return int((d > tol).sum())


def wall_thickness(mesh: M.Mesh, max_samples: int = 600,
                   seed: int = 0) -> dict:
    """Wall thickness by inward face-normal rays.

    Face centres are sampled (area is not weighted); the nearest forward hit
    gives a local thickness.  This is a sampled lower bound, never a proof.

    A ray from an inward-facing surface only measures a *wall* if the part is
    actually hollow there.  On a solid part the ray simply crosses the body and
    reports the chord -- a 20 mm cube would "measure" a 20 mm wall, which is a
    plausible-looking lie.  So when the mesh has no reflex edge at all (a
    convex solid) the measurement is reported as not applicable: ``min_mm`` and
    its percentiles are ``None`` and ``applicable`` is ``False``.
    """
    v = np.asarray(mesh.verts, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    empty = {"min_mm": None, "p05_mm": None, "p50_mm": None, "p95_mm": None,
             "n_samples": 0, "n_hits": 0, "coverage": 0.0, "exact": False,
             "applicable": False, "note": "empty mesh"}
    if len(f) == 0:
        return empty
    if concave_edge_count(mesh) == 0:
        out = dict(empty)
        out.update({
            "n_samples": int(min(max(1, max_samples), len(f))),
            "note": "solid part: no opposing surface, wall thickness is not "
                    "applicable (a convex solid has no wall to measure)",
        })
        return out
    fn, _ = _face_normals(v, f)
    a, b, c = _tri_arrays(v, f)
    centroids = (a + b + c) / 3.0
    signed = _signed_volume(v, f)
    inward = fn if signed < 0.0 else -fn
    k = max(1, min(int(max_samples), len(f)))
    if k < len(f):
        rng = np.random.default_rng(int(seed))
        idx = np.unique(rng.integers(0, len(f), size=k))
    else:
        idx = np.arange(len(f), dtype=np.int64)
    diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    off = max(1e-7, diag * 1e-6)
    origins = centroids[idx] + inward[idx] * off
    dirs = inward[idx]
    a3, e1, e2 = _tri_edges(v, f)
    t = _ray_reduce(origins, dirs, a3, e1, e2, mode="min", t_min=off * 0.5,
                    normals=fn, min_dot=0.8)
    good = np.isfinite(t) & (t > 0.0)
    d = t[good] * 1000.0
    if len(d) == 0:
        out = dict(empty)
        out.update({"n_samples": int(len(idx)), "n_hits": 0,
                    "applicable": False,
                    "note": "no inward hit; open or zero-thickness shell"})
        return out
    return {
        "min_mm": round(float(d.min()), 4),
        "p05_mm": round(float(np.percentile(d, 5)), 4),
        "p50_mm": round(float(np.percentile(d, 50)), 4),
        "p95_mm": round(float(np.percentile(d, 95)), 4),
        "n_samples": int(len(idx)),
        "n_hits": int(len(d)),
        "coverage": round(float(len(d)) / max(1, len(idx)), 4),
        "exact": False,
        "applicable": True,
        "method": "ray_face_center_inward",
    }


# ---------------------------------------------------------------- angles

def dihedral(mesh: M.Mesh, thresholds=(30.0, 60.0)) -> dict:
    """Dihedral angles across two-face edges; 0 means coplanar."""
    v = np.asarray(mesh.verts, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    th = [float(x) for x in (thresholds or (30.0, 60.0))] or [30.0, 60.0]
    empty = {"p50_deg": 0.0, "p95_deg": 0.0, "max_deg": 0.0, "n_edges": 0,
             "count_gt_30": 0, "count_gt_60": 0,
             "thresholds_deg": th,
             "counts_above": [{"threshold_deg": t, "count": 0} for t in th],
             "exact": False}
    if len(f) == 0:
        return empty
    _, counts, order, starts, face_ids = _edge_table(v, f)
    two = np.where(counts == 2)[0]
    if len(two) == 0:
        return empty
    f0 = face_ids[order[starts[two]]]
    f1 = face_ids[order[starts[two] + 1]]
    fn, _ = _face_normals(v, f)
    dots = np.clip(np.einsum("ij,ij->i", fn[f0], fn[f1]), -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    return {
        "p50_deg": round(float(np.percentile(ang, 50)), 4),
        "p95_deg": round(float(np.percentile(ang, 95)), 4),
        "max_deg": round(float(ang.max()), 4),
        "n_edges": int(len(ang)),
        "count_gt_30": int((ang > th[0]).sum()),
        "count_gt_60": int((ang > th[-1]).sum()),
        "thresholds_deg": th,
        "counts_above": [{"threshold_deg": t, "count": int((ang > t).sum())}
                         for t in th],
        "exact": False,
    }


def vertex_normals(mesh: M.Mesh) -> np.ndarray:
    v = np.asarray(mesh.verts, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if len(f) == 0:
        return np.zeros_like(v)
    fn, area = _face_normals(v, f)
    # A common normalising area preserves weighting at all physical scales.
    area_scale = float(area.max())
    weighted = fn * (area / area_scale if area_scale > 0 else area)[:, None]
    out = np.zeros_like(v)
    for c in range(3):
        out[:, c] = np.bincount(f.ravel(), weights=np.repeat(weighted[:, c], 3),
                                minlength=len(v))
    ln = np.linalg.norm(out, axis=1)
    ok = ln > EPS
    out[ok] /= ln[ok, None]
    out[~ok] = 0.0
    return out


def curvature(mesh: M.Mesh) -> dict:
    """Curvature proxy: angle between vertex normals across shared edges."""
    v = np.asarray(mesh.verts, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    empty = {"p50_deg": 0.0, "p95_deg": 0.0, "max_deg": 0.0, "n_edges": 0,
             "exact": False}
    if len(f) == 0:
        return empty
    reps, _, _, _, _ = _edge_table(v, f)
    vn = vertex_normals(mesh)
    n0 = np.linalg.norm(vn[reps[:, 0]], axis=1)
    n1 = np.linalg.norm(vn[reps[:, 1]], axis=1)
    good = (n0 > 0.5) & (n1 > 0.5)
    if not good.any():
        return empty
    dots = np.clip(np.einsum("ij,ij->i", vn[reps[good, 0]], vn[reps[good, 1]]),
                   -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    return {
        "p50_deg": round(float(np.percentile(ang, 50)), 4),
        "p95_deg": round(float(np.percentile(ang, 95)), 4),
        "max_deg": round(float(ang.max()), 4),
        "n_edges": int(len(ang)),
        "exact": False,
        "method": "vertex_normal_variation",
    }


# ---------------------------------------------------------------- enclosure & self-intersection

def dangling_edge_length_mm(mesh: M.Mesh) -> float:
    """Total length of boundary (single-face / open) edges in millimetres."""
    v = np.asarray(mesh.verts, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if len(f) == 0:
        return 0.0
    reps, counts, _, _, _ = _edge_table(v, f)
    open_edges = reps[counts == 1]
    if len(open_edges) == 0:
        return 0.0
    lengths = np.linalg.norm(v[open_edges[:, 1]] - v[open_edges[:, 0]], axis=1)
    return _display(float(lengths.sum()) * 1000.0, 6)


def flux_enclosure_error(mesh: M.Mesh) -> float:
    """Dimensionless Gauss-divergence surface enclosure residual.

    Combines the vector surface closure residual
    ``||sum(A_i * n_i)|| / sum(A_i)`` (nonzero for open shells and partially
    flipped meshes) with the signed-volume orientation residual (2.0 for
    globally inverted closed meshes).  Closed outward manifolds yield ~0.0.
    """
    v = np.asarray(mesh.verts, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if len(f) == 0:
        return 0.0
    a, b, c = _tri_arrays(v, f)
    cross = 0.5 * np.cross(b - a, c - a)
    areas = np.linalg.norm(cross, axis=1)
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= EPS:
        return 0.0
    vec_residual = float(np.linalg.norm(cross.sum(axis=0))) / total_area
    signed_vol = _signed_volume(v, f)
    abs_vol = abs(signed_vol)
    orient_residual = (
        abs(signed_vol - abs_vol) / max(abs_vol, EPS)
        if signed_vol < 0.0 and abs_vol > EPS * (total_area ** 1.5)
        else 0.0
    )
    return float(max(vec_residual, orient_residual))


def _morton_3d(pts_norm: np.ndarray) -> np.ndarray:
    """Quantise normalised [0, 1]^3 coordinates into 30-bit Morton codes."""
    q = np.clip((pts_norm * 1023.0).astype(np.uint32), 0, 1023)

    def _expand(x: np.ndarray) -> np.ndarray:
        x = (x | (x << 16)) & np.uint32(0x030000FF)
        x = (x | (x << 8)) & np.uint32(0x0300F00F)
        x = (x | (x << 4)) & np.uint32(0x030C30C3)
        x = (x | (x << 2)) & np.uint32(0x09249249)
        return x

    return _expand(q[:, 0]) | (_expand(q[:, 1]) << 1) | (_expand(q[:, 2]) << 2)


def _coplanar_sat_vec(A0: np.ndarray, A1: np.ndarray, A2: np.ndarray,
                      B0: np.ndarray, B1: np.ndarray, B2: np.ndarray,
                      nA: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Vectorised 2D Separating Axis Theorem for positive-area coplanar overlap."""
    m = len(A0)
    if m == 0:
        return np.zeros(0, dtype=bool)
    least = np.argmin(np.abs(nA), axis=1)
    ref = np.zeros_like(nA)
    ref[np.arange(m), least] = 1.0
    u = np.cross(nA, ref)
    un = np.linalg.norm(u, axis=1, keepdims=True)
    u = np.divide(u, np.where(un > 0, un, 1.0))
    w = np.cross(nA, u)

    pa_3d = np.stack([A0, A1, A2], axis=1)
    pb_3d = np.stack([B0, B1, B2], axis=1)
    pa = np.stack([np.einsum("mij,mj->mi", pa_3d, u),
                   np.einsum("mij,mj->mi", pa_3d, w)], axis=2)
    pb = np.stack([np.einsum("mij,mj->mi", pb_3d, u),
                   np.einsum("mij,mj->mi", pb_3d, w)], axis=2)

    def _ensure_ccw(p: np.ndarray):
        area2 = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
                 (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
        flip = area2 < 0.0
        if flip.any():
            p = p.copy()
            p[flip] = p[flip][:, [0, 2, 1], :]
        return p, np.abs(area2) > 1e-24

    pa, ok_a = _ensure_ccw(pa)
    pb, ok_b = _ensure_ccw(pb)
    eps2 = np.maximum(1e-20, 1e-8 * (scale ** 2))

    def _separated(t_edge: np.ndarray, t_other: np.ndarray) -> np.ndarray:
        sep_any = np.zeros(m, dtype=bool)
        for e in range(3):
            p0 = t_edge[:, e, :]
            p1 = t_edge[:, (e + 1) % 3, :]
            edge = p1 - p0
            rel = t_other - p0[:, None, :]
            cr = edge[:, None, 0] * rel[:, :, 1] - edge[:, None, 1] * rel[:, :, 0]
            sep_any |= (cr <= eps2[:, None]).all(axis=1)
        return sep_any

    sep = _separated(pa, pb) | _separated(pb, pa)
    return ok_a & ok_b & (~sep)


def _tri_tri_intersect_batch(v: np.ndarray, f: np.ndarray,
                             ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Vectorised Moller/SAT triangle-triangle intersection for candidate pairs."""
    if len(ca) == 0:
        return np.zeros(0, dtype=bool)
    A0, A1, A2 = v[f[ca, 0]], v[f[ca, 1]], v[f[ca, 2]]
    B0, B1, B2 = v[f[cb, 0]], v[f[cb, 1]], v[f[cb, 2]]
    nA = np.cross(A1 - A0, A2 - A0)
    nB = np.cross(B1 - B0, B2 - B0)
    lenA = np.linalg.norm(nA, axis=1)
    lenB = np.linalg.norm(nB, axis=1)
    hit = np.zeros(len(ca), dtype=bool)
    valid = (lenA > 1e-24) & (lenB > 1e-24)
    if not valid.any():
        return hit

    idx = np.where(valid)[0]
    A0, A1, A2 = A0[idx], A1[idx], A2[idx]
    B0, B1, B2 = B0[idx], B1[idx], B2[idx]
    nA = nA[idx] / lenA[idx, None]
    nB = nB[idx] / lenB[idx, None]
    scale = np.maximum(np.sqrt(lenA[idx]), np.sqrt(lenB[idx]))
    peps = np.maximum(1e-14, 1e-9 * scale)

    db = np.stack([((B0 - A0) * nA).sum(1),
                   ((B1 - A0) * nA).sum(1),
                   ((B2 - A0) * nA).sum(1)], axis=1)
    db[np.abs(db) <= peps[:, None]] = 0.0
    one_side_b = (((db >= 0).all(1) & ((db > 0).sum(1) >= 2)) |
                  ((db <= 0).all(1) & ((db < 0).sum(1) >= 2)))

    da = np.stack([((A0 - B0) * nB).sum(1),
                   ((A1 - B0) * nB).sum(1),
                   ((A2 - B0) * nB).sum(1)], axis=1)
    da[np.abs(da) <= peps[:, None]] = 0.0
    one_side_a = (((da >= 0).all(1) & ((da > 0).sum(1) >= 2)) |
                  ((da <= 0).all(1) & ((da < 0).sum(1) >= 2)))

    keep = ~(one_side_a | one_side_b)
    if not keep.any():
        return hit

    idx = idx[keep]
    A0, A1, A2 = A0[keep], A1[keep], A2[keep]
    B0, B1, B2 = B0[keep], B1[keep], B2[keep]
    nA, nB = nA[keep], nB[keep]
    da, db = da[keep], db[keep]
    peps, scale = peps[keep], scale[keep]

    cross_n = np.cross(nA, nB)
    cross_len = np.linalg.norm(cross_n, axis=1)
    coplanar = (cross_len <= 1e-6) & (np.abs(db).max(1) <= peps)
    noncop = cross_len > 1e-6

    if noncop.any():
        i_nc = np.where(noncop)[0]
        D = cross_n[i_nc] / cross_len[i_nc, None]

        def _interval_on_line(P0, P1, P2, d):
            P = np.stack([P0, P1, P2], axis=1)
            proj = np.einsum("mij,mj->mi", P, D)
            lo = np.full(len(P0), np.inf)
            hi = np.full(len(P0), -np.inf)
            for k in range(3):
                on = d[:, k] == 0.0
                lo = np.where(on, np.minimum(lo, proj[:, k]), lo)
                hi = np.where(on, np.maximum(hi, proj[:, k]), hi)
            for i0, i1 in ((0, 1), (1, 2), (2, 0)):
                cr = (d[:, i0] * d[:, i1]) < 0.0
                if cr.any():
                    den = d[cr, i0] - d[cr, i1]
                    t = d[cr, i0] / den
                    pt = P[cr, i0] + t[:, None] * (P[cr, i1] - P[cr, i0])
                    pv = (pt * D[cr]).sum(1)
                    lo[cr] = np.minimum(lo[cr], pv)
                    hi[cr] = np.maximum(hi[cr], pv)
            return lo, hi

        ta_lo, ta_hi = _interval_on_line(A0[i_nc], A1[i_nc], A2[i_nc], da[i_nc])
        tb_lo, tb_hi = _interval_on_line(B0[i_nc], B1[i_nc], B2[i_nc], db[i_nc])
        coord_max = np.maximum(
            np.maximum(np.abs(A0[i_nc]).max(1), np.abs(A1[i_nc]).max(1)),
            np.maximum(np.abs(B0[i_nc]).max(1), np.abs(B1[i_nc]).max(1)),
        )
        leps = np.maximum(1e-14, np.maximum(1e-8 * scale[i_nc], 5e-7 * coord_max))
        hit[idx[i_nc]] = np.minimum(ta_hi, tb_hi) > np.maximum(ta_lo, tb_lo) + leps

    if coplanar.any():
        i_cp = np.where(coplanar)[0]
        hit[idx[i_cp]] = _coplanar_sat_vec(
            A0[i_cp], A1[i_cp], A2[i_cp],
            B0[i_cp], B1[i_cp], B2[i_cp],
            nA[i_cp], scale[i_cp],
        )
    return hit


def self_intersection(mesh: M.Mesh, max_pairs: int = 200000) -> dict:
    """Detect non-adjacent triangle-triangle self-intersections via Morton BVH."""
    v = np.asarray(mesh.verts, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    n = len(f)
    max_pairs = max(0, int(max_pairs))
    if n < 2 or len(v) == 0 or not np.isfinite(v).all():
        return {
            "intersecting_pairs": 0,
            "checked_pairs": 0,
            "self_intersection_ratio": 0.0,
            "truncated": False,
            "exact": True,
        }

    tris = v[f]
    diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    eps = max(1e-14, diag * 1e-9)
    tmin = tris.min(axis=1) - eps
    tmax = tris.max(axis=1) + eps
    cent = (tmin + tmax) * 0.5
    cmin = cent.min(axis=0)
    span = np.maximum(cent.max(axis=0) - cmin, 1e-12)
    idx = np.argsort(_morton_3d((cent - cmin) / span), kind="stable")
    tmin, tmax, f_ord = tmin[idx], tmax[idx], f[idx]

    levels_min = [tmin]
    levels_max = [tmax]
    cur_min, cur_max = tmin, tmax
    while len(cur_min) > 1:
        mc = len(cur_min)
        pairs = mc // 2
        nxt_min = np.minimum(cur_min[:2 * pairs:2], cur_min[1:2 * pairs:2])
        nxt_max = np.maximum(cur_max[:2 * pairs:2], cur_max[1:2 * pairs:2])
        if mc % 2 == 1:
            nxt_min = np.vstack([nxt_min, cur_min[-1:]])
            nxt_max = np.vstack([nxt_max, cur_max[-1:]])
        levels_min.append(nxt_min)
        levels_max.append(nxt_max)
        cur_min, cur_max = nxt_min, nxt_max

    pa = np.array([0], dtype=np.int64)
    pb = np.array([0], dtype=np.int64)
    truncated = False
    cap = max(2_000_000, max_pairs * 16)
    for lvl in range(len(levels_min) - 1, 0, -1):
        n_child = len(levels_min[lvl - 1])
        cmin, cmax = levels_min[lvl - 1], levels_max[lvl - 1]
        same = pa == pb
        ps = pa[same]
        c0, c1 = 2 * ps, 2 * ps + 1
        has1 = c1 < n_child
        same_a = np.concatenate([c0, c1[has1], c0[has1]])
        same_b = np.concatenate([c0, c1[has1], c1[has1]])
        pd_a, pd_b = pa[~same], pb[~same]
        a0, a1 = 2 * pd_a, 2 * pd_a + 1
        b0, b1 = 2 * pd_b, 2 * pd_b + 1
        ha1, hb1 = a1 < n_child, b1 < n_child
        ha1b1 = ha1 & hb1
        diff_a = np.concatenate([a0, a0[hb1], a1[ha1], a1[ha1b1]])
        diff_b = np.concatenate([b0, b1[hb1], b0[ha1], b1[ha1b1]])
        pa_all = np.concatenate([same_a, diff_a])
        pb_all = np.concatenate([same_b, diff_b])
        ok = (
            (cmax[pa_all, 0] >= cmin[pb_all, 0]) & (cmax[pb_all, 0] >= cmin[pa_all, 0]) &
            (cmax[pa_all, 1] >= cmin[pb_all, 1]) & (cmax[pb_all, 1] >= cmin[pa_all, 1]) &
            (cmax[pa_all, 2] >= cmin[pb_all, 2]) & (cmax[pb_all, 2] >= cmin[pa_all, 2])
        )
        pa, pb = pa_all[ok], pb_all[ok]
        if len(pa) > cap:
            pa, pb = pa[:cap], pb[:cap]
            truncated = True

    diff = pa < pb
    pa, pb = pa[diff], pb[diff]
    fa, fb = f_ord[pa], f_ord[pb]
    a0, a1, a2 = fa[:, 0], fa[:, 1], fa[:, 2]
    b0, b1, b2 = fb[:, 0], fb[:, 1], fb[:, 2]
    adj = (
        (a0 == b0) | (a0 == b1) | (a0 == b2) |
        (a1 == b0) | (a1 == b1) | (a1 == b2) |
        (a2 == b0) | (a2 == b1) | (a2 == b2)
    )
    ca, cb = idx[pa[~adj]], idx[pb[~adj]]
    if len(ca) > max_pairs:
        ca, cb = ca[:max_pairs], cb[:max_pairs]
        truncated = True

    hits = _tri_tri_intersect_batch(v, f, ca, cb)
    n_inter = int(hits.sum())
    if n_inter > 0:
        bad_tris = np.unique(np.concatenate([ca[hits], cb[hits]]))
        sir = round(float(len(bad_tris)) / float(n), 8)
    else:
        sir = 0.0
    return {
        "intersecting_pairs": n_inter,
        "checked_pairs": int(len(ca)),
        "self_intersection_ratio": sir,
        "truncated": bool(truncated),
        "exact": not bool(truncated),
    }


# ---------------------------------------------------------------- aggregate

_dihedral_fn = dihedral
_curvature_fn = curvature


def measure(mesh: M.Mesh, *, wall: bool = False, dihedral: bool = True,
            curvature: bool = False,
            self_intersect: bool = True,
            density_kg_m3: float | None = None) -> dict:
    out = {
        "bbox_mm": bbox_mm(mesh),
        "volume_mm3": volume_mm3(mesh),
        "area_mm2": area_mm2(mesh),
        "centroid_mm": centroid_mm(mesh),
        "topology": M.topology(mesh),
        "dangling_edge_length_mm": dangling_edge_length_mm(mesh),
        "flux_enclosure_error": flux_enclosure_error(mesh),
        "units": "mm",
        "exact": False,
    }
    if self_intersect:
        out["self_intersection"] = self_intersection(mesh)
    if density_kg_m3 is not None:
        out["mass"] = mass_properties(mesh, density_kg_m3)
    if wall:
        out["wall"] = wall_thickness(mesh)
    if dihedral:
        out["dihedral"] = _dihedral_fn(mesh)
    if curvature:
        out["curvature"] = _curvature_fn(mesh)
    return out


__all__ = ["bbox_mm", "volume_mm3", "area_mm2", "centroid_mm",
           "mass_properties", "wall_thickness", "concave_edge_count",
           "dihedral", "curvature", "vertex_normals",
           "self_intersection", "dangling_edge_length_mm",
           "flux_enclosure_error", "measure"]
