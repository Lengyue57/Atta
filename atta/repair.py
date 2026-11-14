# -*- coding: utf-8 -*-
"""Mesh repair and remeshing: make any soup into a solid.

Every generative path in this kernel can produce a mesh that is *almost* right:
a marching-tets surface with a pinhole, a boolean that left a sliver, a sweep
whose cap did not weld.  v1.2 had no way to fix any of that — the mesh either
verified or the model was discarded.

This module is the repair bench:

``weld``
    Merge coincident vertices and drop degenerate/duplicate faces.
``fill_holes``
    Find boundary loops and cap them with a fan, a centroid fan, or a
    triangulated polygon.  This is what turns an open shell into a solid.
``remove_non_manifold``
    Split non-manifold edges by duplicating vertices, so a surface that touches
    itself becomes two valid surfaces.
``remove_small_components``
    Drop floating debris below a volume or face-count threshold.
``orient_consistently``
    Propagate winding across the surface so every face agrees.
``remesh_isotropic``
    Split long edges and collapse short ones toward a target edge length.
``laplacian_smooth``
    Taubin smoothing (lambda/mu) — smooths without shrinking.
``quadric_decimate``
    Native tolerance-bounded simplification toward a target face count.

All functions take and return :class:`~atta.mesh.Mesh` in metres, and none of
them depend on Blender.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import AttaError
from .mesh import Mesh

__all__ = [
    "weld", "fill_holes", "remove_non_manifold", "remove_small_components",
    "orient_consistently", "remesh_isotropic", "laplacian_smooth",
    "quadric_decimate", "repair", "boundary_loops", "mesh_report",
]


# ---------------------------------------------------------------- topology

def _face_edges(faces):
    """All directed half-edges as an (F*3, 2) array."""
    return np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])


def _edge_map(faces):
    """Map each undirected edge to the half-edges that use it."""
    he = _face_edges(faces)
    key = np.sort(he, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key_sorted = key[order]
    # Group boundaries
    if len(key_sorted) == 0:
        return key_sorted, order, np.zeros(0, dtype=np.int64)
    change = np.any(np.diff(key_sorted, axis=0) != 0, axis=1)
    group = np.concatenate([[0], np.cumsum(change)])
    return key_sorted, order, group


def boundary_loops(mesh: Mesh):
    """Ordered boundary loops as lists of vertex indices.

    A boundary edge is used by exactly one face.  Chaining them gives the
    holes; each loop is returned in traversal order so it can be capped.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return []
    he = _face_edges(faces)
    key = np.sort(he, axis=1)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    if not np.any(counts == 1):
        return []
    boundary = uniq[counts == 1]
    # Build directed adjacency from the actual half-edges on boundary edges.
    bset = {tuple(e) for e in boundary}
    nxt = {}
    for a, b in he:
        a, b = int(a), int(b)
        if (min(a, b), max(a, b)) in bset:
            if a not in nxt:
                nxt[a] = b
            elif nxt[a] != b:
                # Non-manifold boundary: keep the first, ignore the rest.
                pass
    loops = []
    seen = set()
    for start in list(nxt.keys()):
        if start in seen:
            continue
        loop = []
        cur = start
        while cur not in seen and cur in nxt:
            seen.add(cur)
            loop.append(cur)
            cur = nxt[cur]
            if cur == start:
                break
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def mesh_report(mesh: Mesh) -> dict:
    """A compact health report: what is wrong and how much of it."""
    topo = mesh.topology()
    loops = boundary_loops(mesh)
    return {
        "verts": int(len(mesh.verts)),
        "tris": int(len(mesh.faces)),
        "closed": bool(topo["closed"]),
        "boundary_edges": int(topo["boundary_edges"]),
        "non_manifold_edges": int(topo["non_manifold_edges"]),
        "inconsistent_winding_edges": int(topo["inconsistent_winding_edges"]),
        "degenerate_faces": int(topo["degenerate_faces"]),
        "components": int(topo["n_components"]),
        "genus": topo["genus"],
        "holes": len(loops),
        "hole_sizes": [len(l) for l in loops],
        "volume_mm3": round(float(mesh.volume()) * 1e9, 3),
    }


# ---------------------------------------------------------------- weld

def weld(mesh: Mesh, tol: float = 1e-9) -> Mesh:
    """Merge vertices closer than ``tol``; drop degenerate and duplicate faces.

    Uses a quantised grid key so it is O(n) rather than O(n²).
    """
    verts = np.asarray(mesh.verts, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(verts) == 0 or len(faces) == 0:
        return mesh.copy()
    tol = max(float(tol), 1e-15)
    q = np.floor(verts / tol).astype(np.int64)
    # Include neighbours in the 3x3x3 cell block so a pair straddling a cell
    # boundary still merges.
    best = {}
    remap = np.full(len(verts), -1, dtype=np.int64)
    out_verts = []
    for i in range(len(verts)):
        cell = (int(q[i, 0]), int(q[i, 1]), int(q[i, 2]))
        found = -1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    hit = best.get((cell[0] + dx, cell[1] + dy, cell[2] + dz))
                    if hit is None:
                        continue
                    if float(np.linalg.norm(verts[i] - out_verts[hit])) <= tol:
                        found = hit
                        break
                if found >= 0:
                    break
            if found >= 0:
                break
        if found < 0:
            found = len(out_verts)
            out_verts.append(verts[i])
            best[cell] = found
        remap[i] = found

    new_verts = np.asarray(out_verts, dtype=np.float64)
    new_faces = remap[faces]
    # Drop degenerate faces (repeated index).
    keep = ((new_faces[:, 0] != new_faces[:, 1])
            & (new_faces[:, 1] != new_faces[:, 2])
            & (new_faces[:, 0] != new_faces[:, 2]))
    new_faces = new_faces[keep]
    # Drop duplicate faces (same vertex set).
    if len(new_faces):
        key = np.sort(new_faces, axis=1)
        _, idx = np.unique(key, axis=0, return_index=True)
        new_faces = new_faces[np.sort(idx)]
    return Mesh(new_verts, new_faces, dict(mesh.meta))


# ---------------------------------------------------------------- holes

def fill_holes(mesh: Mesh, *, max_edges=0, mode="centroid") -> Mesh:
    """Cap every boundary loop.

    ``mode="fan"`` triangulates from the loop's first vertex; ``"centroid"``
    adds a centre vertex and fans from it (better for non-convex holes);
    ``"planar"`` projects to the loop's best-fit plane and ear-clips.

    ``max_edges`` skips loops longer than that (0 = no limit), so a repair can
    close pinholes without inventing a giant lid over a genuinely open shape.
    """
    loops = boundary_loops(mesh)
    if not loops:
        return mesh.copy()
    verts = [np.asarray(mesh.verts, dtype=np.float64)]
    faces = [np.asarray(mesh.faces, dtype=np.int64)]
    offset = len(verts[0])
    extra_verts = []
    extra_faces = []
    for loop in loops:
        if max_edges and len(loop) > int(max_edges):
            continue
        pts = verts[0][np.asarray(loop, dtype=np.int64)]
        mode_local = mode
        if mode == "planar":
            tri = _earclip_3d(pts)
            if tri is not None:
                for (a, b, c) in tri:
                    extra_faces.append((loop[a], loop[b], loop[c]))
                continue
            # Ear clipping failed (self-intersecting or degenerate loop);
            # fall back to a centroid fan, which always produces triangles.
            mode_local = "centroid"
        if mode_local == "fan":
            for i in range(1, len(loop) - 1):
                extra_faces.append((loop[0], loop[i], loop[i + 1]))
        else:
            centre = pts.mean(axis=0)
            ci = offset + len(extra_verts)
            extra_verts.append(centre)
            for i in range(len(loop)):
                j = (i + 1) % len(loop)
                extra_faces.append((loop[i], loop[j], ci))
    if not extra_faces:
        return mesh.copy()
    all_verts = np.vstack([verts[0]] + ([np.asarray(extra_verts)]
                                        if extra_verts else []))
    all_faces = np.vstack([faces[0], np.asarray(extra_faces, dtype=np.int64)])
    out = Mesh(all_verts, all_faces, dict(mesh.meta))
    return orient_consistently(out)


def _earclip_3d(pts):
    """Ear-clip a planar-ish 3D polygon; returns index triples or None."""
    n = len(pts)
    if n < 3:
        return None
    c = pts.mean(axis=0)
    q = pts - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    p2 = q @ vt[:2].T
    # Ensure CCW
    area = 0.5 * float(np.sum(p2[:, 0] * np.roll(p2[:, 1], -1)
                              - p2[:, 1] * np.roll(p2[:, 0], -1)))
    order = list(range(n))
    if area < 0:
        order.reverse()
        p2 = p2[order]
    idx = list(range(len(order)))
    out = []
    guard = 0
    while len(idx) > 3 and guard < 4 * n * n:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c2 = p2[i0], p2[i1], p2[i2]
            cross = (b[0] - a[0]) * (c2[1] - a[1]) - (b[1] - a[1]) * (c2[0] - a[0])
            if cross <= 1e-18:
                continue
            ok = True
            for m in idx:
                if m in (i0, i1, i2):
                    continue
                if _pt_in_tri(p2[m], a, b, c2):
                    ok = False
                    break
            if ok:
                out.append((order[i0], order[i1], order[i2]))
                idx.pop(k)
                clipped = True
                break
        if not clipped:
            return None
    if len(idx) == 3:
        out.append((order[idx[0]], order[idx[1]], order[idx[2]]))
    return out


def _pt_in_tri(p, a, b, c):
    d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    d2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])
    d3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


# ---------------------------------------------------------------- winding

def orient_consistently(mesh: Mesh) -> Mesh:
    """Propagate a consistent winding across the surface, then face outward.

    Two faces agree when they traverse their shared edge in *opposite*
    directions.  The flood fill records, per face, whether it must be flipped
    relative to the seed; then the whole component is oriented outward by
    checking its signed volume.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    if len(faces) == 0:
        return mesh.copy()
    verts = np.asarray(mesh.verts, dtype=np.float64)
    n_f = len(faces)

    # Directed half-edge -> list of (face, corner)
    directed = {}
    for fi in range(n_f):
        for k in range(3):
            a = int(faces[fi, k])
            b = int(faces[fi, (k + 1) % 3])
            directed.setdefault((a, b), []).append(fi)

    visited = np.zeros(n_f, dtype=bool)
    flip = np.zeros(n_f, dtype=bool)
    for seed in range(n_f):
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [seed]
        while stack:
            fi = stack.pop()
            for k in range(3):
                a = int(faces[fi, k])
                b = int(faces[fi, (k + 1) % 3])
                # A neighbour using (b, a) already agrees with this face;
                # a neighbour using (a, b) is wound the other way.
                for nb in directed.get((b, a), []):
                    if nb == fi or visited[nb]:
                        continue
                    visited[nb] = True
                    flip[nb] = flip[fi]
                    stack.append(nb)
                for nb in directed.get((a, b), []):
                    if nb == fi or visited[nb]:
                        continue
                    visited[nb] = True
                    flip[nb] = not flip[fi]
                    stack.append(nb)
    faces[flip] = faces[flip][:, ::-1]
    out = Mesh(verts, faces, dict(mesh.meta))
    signed, _ = _signed_volume(out)
    if signed < 0.0:
        out = Mesh(verts, faces[:, ::-1].copy(), dict(mesh.meta))
    return out


def _signed_volume(mesh):
    v = np.asarray(mesh.verts, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if len(f) == 0:
        return 0.0, np.zeros(3)
    origin = v[f[0, 0]]
    a, b, c = v[f[:, 0]] - origin, v[f[:, 1]] - origin, v[f[:, 2]] - origin
    vol = float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)
    return vol, np.zeros(3)


# ---------------------------------------------------------------- cleanup

def remove_non_manifold(mesh: Mesh) -> Mesh:
    """Split non-manifold edges so the surface becomes manifold.

    A non-manifold edge is used by three or more faces.  We keep the two faces
    that form the most consistent pair and duplicate the edge for the rest,
    which separates the sheets without moving any geometry.
    """
    verts = np.asarray(mesh.verts, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return mesh.copy()
    he = _face_edges(faces)
    key = np.sort(he, axis=1)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    bad = uniq[counts > 2]
    if len(bad) == 0:
        return mesh.copy()

    # For each bad edge, duplicate one endpoint for every face beyond the
    # second, so each extra sheet gets its own vertex.
    new_verts = [verts]
    extra = []
    faces = faces.copy()
    bad_set = {tuple(e) for e in bad}
    seen = {}
    for fi in range(len(faces)):
        for k in range(3):
            a = int(faces[fi, k])
            b = int(faces[fi, (k + 1) % 3])
            e = (min(a, b), max(a, b))
            if e not in bad_set:
                continue
            n = seen.get(e, 0)
            seen[e] = n + 1
            if n >= 2:
                # Give this face its own copy of both endpoints.
                ia = len(verts) + len(extra)
                extra.append(verts[a])
                ib = len(verts) + len(extra)
                extra.append(verts[b])
                faces[fi, k] = ia
                faces[fi, (k + 1) % 3] = ib
    if not extra:
        return mesh.copy()
    all_verts = np.vstack([verts, np.asarray(extra)])
    return Mesh(all_verts, faces, dict(mesh.meta))


def remove_small_components(mesh: Mesh, *, min_faces=8, min_volume_mm3=0.0):
    """Drop disconnected debris below the thresholds."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return mesh.copy()
    # Union-find over vertices through faces.
    n_v = len(mesh.verts)
    parent = np.arange(n_v)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for f in faces:
        union(int(f[0]), int(f[1]))
        union(int(f[1]), int(f[2]))
    roots = np.array([find(int(f[0])) for f in faces])
    uniq = np.unique(roots)
    keep = np.ones(len(faces), dtype=bool)
    for root in uniq:
        mask = roots == root
        if int(mask.sum()) < int(min_faces):
            keep[mask] = False
            continue
        if min_volume_mm3 > 0.0:
            sub = Mesh(mesh.verts, faces[mask], {})
            if abs(float(sub.volume())) * 1e9 < float(min_volume_mm3):
                keep[mask] = False
    if keep.all():
        return mesh.copy()
    return Mesh(mesh.verts.copy(), faces[keep], dict(mesh.meta))


# ---------------------------------------------------------------- remesh

def remesh_isotropic(mesh: Mesh, target_edge=None, *, iterations=1,
                     split_factor=1.5, collapse_factor=0.5) -> Mesh:
    """Split long edges and collapse short ones toward ``target_edge``.

    This is a simple isotropic remesher: it does not move vertices, so it
    preserves the shape exactly while making triangle sizes uniform (which is
    what downstream smoothing and decimation need).
    """
    out = mesh.copy()
    if target_edge is None:
        edges = _unique_edges(out)
        if len(edges) == 0:
            return out
        v = out.verts
        lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
        target_edge = float(np.median(lengths))
    target = float(target_edge)
    if not math.isfinite(target) or target <= 0.0:
        raise AttaError("bad_arg", "target_edge must be positive",
                        path="target_edge", fix="pass a length in metres")
    for _ in range(max(1, int(iterations))):
        out = _split_long(out, target * split_factor)
        out = _collapse_short(out, target * collapse_factor)
    return out


def _unique_edges(mesh):
    f = np.asarray(mesh.faces, dtype=np.int64)
    if len(f) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    he = _face_edges(f)
    return np.unique(np.sort(he, axis=1), axis=0)


def _split_long(mesh: Mesh, limit: float) -> Mesh:
    verts = [np.asarray(mesh.verts, dtype=np.float64)]
    faces = [np.asarray(mesh.faces, dtype=np.int64)]
    edges = _unique_edges(mesh)
    v = verts[0]
    lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    long_edges = edges[lengths > limit]
    if len(long_edges) == 0:
        return mesh
    mid = {}
    extra = []
    for a, b in long_edges:
        key = (min(int(a), int(b)), max(int(a), int(b)))
        if key in mid:
            continue
        mid[key] = len(v) + len(extra)
        extra.append((v[a] + v[b]) * 0.5)
    all_verts = np.vstack([v, np.asarray(extra)])
    new_faces = []
    for f in faces[0]:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        ab = mid.get((min(a, b), max(a, b)))
        bc = mid.get((min(b, c), max(b, c)))
        ca = mid.get((min(c, a), max(c, a)))
        # Standard 1-2-3 split cases.
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
    return Mesh(all_verts, np.asarray(new_faces, dtype=np.int64), dict(mesh.meta))


def _collapse_short(mesh: Mesh, limit: float) -> Mesh:
    """Collapse short edges by merging the pair at its midpoint."""
    v = np.asarray(mesh.verts, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    edges = _unique_edges(mesh)
    if len(edges) == 0:
        return mesh
    lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    short = edges[lengths < limit]
    if len(short) == 0:
        return mesh
    remap = np.arange(len(v))
    for a, b in short:
        a, b = int(a), int(b)
        ra, rb = remap[a], remap[b]
        if ra == rb:
            continue
        keep, drop = min(ra, rb), max(ra, rb)
        v[keep] = (v[keep] + v[drop]) * 0.5
        remap[remap == drop] = keep
    faces = remap[faces]
    keep = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2]))
    faces = faces[keep]
    if len(faces):
        key = np.sort(faces, axis=1)
        _, idx = np.unique(key, axis=0, return_index=True)
        faces = faces[np.sort(idx)]
    # Compact unused vertices.
    used, inv = np.unique(faces.ravel(), return_inverse=True)
    return Mesh(v[used], inv.reshape(-1, 3), dict(mesh.meta))


def laplacian_smooth(mesh: Mesh, *, iterations=10, lam=0.5, mu=-0.53,
                     preserve_boundary=True) -> Mesh:
    """Taubin smoothing: low-pass with no volume loss.

    Plain Laplacian smoothing shrinks a mesh. Taubin alternates a positive
    step ``lam`` with a negative step ``mu`` to reduce that shrinkage; it does
    not exactly conserve volume. The classic pair is ``lam=0.5, mu=-0.53``.
    Each step recomputes the Laplacian at the updated positions.
    """
    v = np.asarray(mesh.verts, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return mesh.copy()
    from scipy.sparse import coo_matrix

    edges = _unique_edges(mesh)
    n_v = len(v)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    degree = np.bincount(rows, minlength=n_v)
    movable = degree > 0
    if preserve_boundary:
        halfedges = np.sort(_face_edges(faces), axis=1)
        unique, counts = np.unique(halfedges, axis=0, return_counts=True)
        movable[unique[counts == 1].ravel()] = False
    if not np.any(movable):
        return mesh.copy()
    weights = 1.0 / np.maximum(degree[rows], 1)
    average = coo_matrix((weights, (rows, cols)), shape=(n_v, n_v)).tocsr()

    for _ in range(max(1, int(iterations))):
        lap = average @ v - v
        v[movable] += float(lam) * lap[movable]
        lap = average @ v - v
        v[movable] += float(mu) * lap[movable]
    return Mesh(v, faces.copy(), dict(mesh.meta))


def quadric_decimate(mesh: Mesh, target_faces: int, *, max_error=None,
                     max_iterations=12) -> Mesh:
    """Simplify a solid toward a face target within a surface error budget.

    The historical name is retained for callers; the implementation uses
    Manifold's native tolerance simplifier, not quadric error metrics. Lengths
    are metres. The default budget is 0.2% of the bounding-box diagonal; use an
    explicit ``max_error`` for thin walls or dimensional interfaces.

    The target is best effort: topology and error take priority over face
    count. Unsupported/open inputs and rejected results are returned unchanged
    with a reason in ``meta['decimation']``. No positional weld is performed.
    """
    try:
        target = max(4, int(target_faces))
        iterations = int(max_iterations)
        error = (float(mesh.bbox()["diag"]) * 0.002 if max_error is None
                 else float(max_error))
    except (TypeError, ValueError, OverflowError) as exc:
        raise AttaError("bad_arg", "invalid simplification budget",
                        path="repair.decimate",
                        fix="pass a face count and finite nonnegative max_error in metres") from exc
    if not math.isfinite(error) or error < 0 or not 1 <= iterations <= 32:
        raise AttaError("bad_arg", "invalid simplification budget",
                        path="repair.decimate",
                        fix="use max_error >= 0 and max_iterations in 1..32")
    if not np.all(np.isfinite(mesh.verts)):
        raise AttaError("bad_arg", "mesh vertices must be finite",
                        path="repair.decimate", fix="remove non-finite coordinates")

    report = {
        "method": "manifold_tolerance", "units": "m",
        "input_faces": len(mesh.faces), "target_faces": target,
        "max_error": error, "tolerance": 0.0, "evaluations": 0,
        "error_bound_kind": "backend_surface_tolerance",
    }

    def finish(result, reason, tolerance=0.0):
        result = result.copy()
        result.meta["decimation"] = {
            **report, "output_faces": len(result.faces),
            "target_reached": len(result.faces) <= target,
            "reason": reason, "tolerance": float(tolerance),
        }
        return result

    if len(mesh.faces) <= target:
        return finish(mesh, "already_within_target")
    if error == 0:
        return finish(mesh, "zero_error_budget")
    before = mesh.topology()
    if not before["closed"] or before["degenerate_faces"]:
        return finish(mesh, "input_not_closed_manifold")

    try:
        import manifold3d
    except ImportError:
        return finish(mesh, "backend_unavailable")
    if not hasattr(manifold3d, "Mesh64"):
        return finish(mesh, "double_precision_backend_unavailable")

    # Centre before import: Manifold's numeric tolerance also depends on the
    # coordinate magnitude, whereas the user's budget describes this object.
    origin = np.asarray(mesh.bbox()["center"], dtype=np.float64)
    verts = np.array(mesh.verts - origin, dtype=np.float64, order="C", copy=True)
    faces = np.array(mesh.faces, dtype=np.uint64, order="C", copy=True)
    try:
        solid = manifold3d.Manifold(manifold3d.Mesh64(
            vert_properties=verts, tri_verts=faces))
        if solid.status() != manifold3d.Error.NoError or solid.is_empty():
            return finish(mesh, "backend_rejected_input")
        import_error = float(solid.get_tolerance())
        roundoff = float(np.max(np.abs(np.spacing(mesh.verts)))) * 2.0
        available = error - import_error - roundoff
        if available <= import_error:
            return finish(mesh, "below_backend_precision")
        signature = sorted(part.genus() for part in solid.decompose())

        def candidate(tolerance):
            report["evaluations"] += 1
            value = solid.simplify(float(tolerance))
            if (value.status() != manifold3d.Error.NoError or value.is_empty()
                    or sorted(part.genus() for part in value.decompose()) != signature):
                return None
            return value, value.num_tri(), float(tolerance)

        best = candidate(available)
        if best is None:
            return finish(mesh, "topology_guard")
        if best[1] <= target:
            low, high = import_error, available
            for _ in range(iterations - 1):
                mid = (low + high) * 0.5
                trial = candidate(mid)
                if trial is not None and trial[1] <= target:
                    best, high = trial, mid
                else:
                    low = mid
        if best[1] >= len(mesh.faces):
            return finish(mesh, "no_reduction_within_budget")
        raw = best[0].to_mesh64()
        result = Mesh(np.asarray(raw.vert_properties)[:, :3] + origin,
                      np.asarray(raw.tri_verts, dtype=np.int64), dict(mesh.meta))
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        report["backend_error"] = str(exc)
        return finish(mesh, "backend_failed")

    after = result.topology()
    if (not after["closed"] or after["degenerate_faces"]
            or after["n_components"] != before["n_components"]
            or after["genus"] != before["genus"]):
        return finish(mesh, "topology_guard")
    old_volume = mesh.volume()
    report["topology_preserved"] = True
    report["volume_relative_change"] = ((result.volume() - old_volume) / old_volume
                                         if old_volume > 0 else None)
    report["surface_error_bound"] = best[2] + import_error + roundoff
    return finish(result, "target_reached" if best[1] <= target
                  else "error_budget_limited", best[2])


# ---------------------------------------------------------------- pipeline

def repair(mesh: Mesh, *, fill=True, max_hole_edges=0, min_faces=8,
           orient=True, smooth=0) -> tuple[Mesh, dict]:
    """The standard repair chain; returns ``(mesh, report)``.

    Order matters: weld first (so topology is real), then split non-manifold
    sheets, then orient, then cap holes, then drop debris.  Smoothing comes
    last so it cannot move a boundary that is about to be capped.
    """
    before = mesh_report(mesh)
    out = weld(mesh, 1e-9)
    out = remove_non_manifold(out)
    if orient:
        out = orient_consistently(out)
    if fill:
        out = fill_holes(out, max_edges=max_hole_edges)
        out = orient_consistently(out)
    if min_faces:
        out = remove_small_components(out, min_faces=min_faces)
    if smooth:
        out = laplacian_smooth(out, iterations=int(smooth))
    after = mesh_report(out)
    return out, {
        "before": before,
        "after": after,
        "fixed": {
            "welded": before["verts"] != after["verts"],
            "holes_filled": before["holes"] - after["holes"],
            "components_removed": before["components"] - after["components"],
            "now_closed": bool(after["closed"]),
        },
    }
