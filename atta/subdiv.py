# -*- coding: utf-8 -*-
"""Subdivision surfaces: turn a coarse control cage into a smooth limit surface.

The v1.2 kernel could only build what its primitives describe: a box is a box,
a revolve is a surface of revolution.  Organic form had to be approximated with
dense lofts — slow, and impossible to control.

Subdivision changes the unit of design.  You give a **coarse control cage** (a
handful of quads an agent can reason about) and the scheme produces the smooth
limit surface.  This is how film and games have modelled organic form for
thirty years, and it is the largest single jump in geometric freedom available
to a triangle kernel.

Schemes
-------
``catmull_clark``
    Quad-based, the industry default.  Face / edge / vertex points, boundary
    rules, semi-sharp creases.  A cube becomes a rounded blob; crease the
    twelve edges and it stays a rounded box.
``loop``
    Triangle-based.  Cheaper, keeps triangles, good for shells.
``simple``
    Linear midpoint split: no smoothing, just more triangles.

All pure numpy, all on the existing :class:`~atta.mesh.Mesh`.
"""

from __future__ import annotations

import numpy as np

from .errors import AttaError
from .mesh import Mesh

__all__ = ["catmull_clark", "loop", "subdivide", "SCHEMES", "crease_from_angle"]


# ---------------------------------------------------------------- topology

def _edge_arrays(faces):
    """Unique undirected edges and the per-face edge ids.

    Returns ``(edges, face_edges)`` where ``edges`` is ``(E, 2)`` sorted pairs
    and ``face_edges`` is ``(F, n)`` with the edge id of face corner ``k``
    (between ``f[k]`` and ``f[k+1]``).
    """
    n = faces.shape[1]
    # (F, n) start and end vertex of each corner edge
    starts = faces
    ends = np.roll(faces, -1, axis=1)
    pairs = np.stack([starts, ends], axis=2)          # (F, n, 2)
    flat = pairs.reshape(-1, 2)
    key = np.sort(flat, axis=1)
    edges, inverse = np.unique(key, axis=0, return_inverse=True)
    return edges, inverse.reshape(faces.shape[0], n)


def _edge_adjacency(edges, face_edges, n_faces):
    """For each edge: first face, second face (-1 if boundary), use count."""
    flat = face_edges.ravel()
    face_of = np.repeat(np.arange(n_faces), face_edges.shape[1])
    order = np.argsort(flat, kind="stable")
    e_sorted = flat[order]
    f_sorted = face_of[order]
    counts = np.bincount(e_sorted, minlength=len(edges))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    first = np.full(len(edges), -1, dtype=np.int64)
    second = np.full(len(edges), -1, dtype=np.int64)
    has = counts > 0
    first[has] = f_sorted[starts[has]]
    two = counts >= 2
    if two.any():
        second[two] = f_sorted[starts[two] + 1]
    return first, second, counts


def _opposite_vertex(faces, edge, face):
    """The vertex of ``face`` that is neither endpoint of ``edge``."""
    a, b = int(edge[0]), int(edge[1])
    for v in faces[face]:
        v = int(v)
        if v != a and v != b:
            return v
    return -1


# ---------------------------------------------------------------- catmull-clark

def catmull_clark(mesh: Mesh, levels: int = 1, *, crease_angle_deg=None,
                  creases=None) -> Mesh:
    """Catmull-Clark subdivision.

    ``crease_angle_deg`` derives creases automatically from the control cage:
    every edge whose two faces meet at more than this angle is held sharp.
    That is the practical knob — pass ``crease_angle_deg=30`` and a box keeps
    its edges while its corners round off.

    ``creases`` adds explicit ``(vertex_a, vertex_b)`` index pairs on top.
    """
    levels = int(levels)
    if levels < 0:
        raise AttaError("bad_arg", "levels must be >= 0", path="levels",
                        fix="use levels: 2")
    if levels == 0:
        return mesh.copy()
    if len(mesh.faces) == 0:
        raise AttaError("bad_arg", "cannot subdivide an empty mesh", path="mesh",
                        fix="build geometry first")

    out = mesh
    for _ in range(levels):
        sharp = set()
        if crease_angle_deg is not None:
            sharp |= set(crease_from_angle(out, crease_angle_deg))
        for a, b in (creases or []):
            a, b = int(a), int(b)
            if a != b:
                sharp.add((min(a, b), max(a, b)))
        out = _cc_once(out, sharp)
    return out


def _cc_once(mesh: Mesh, sharp_edges: set) -> Mesh:
    verts = np.asarray(mesh.verts, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_v, n_f = len(verts), len(faces)
    n_corner = faces.shape[1]

    face_pts = verts[faces].mean(axis=1)                     # (F, 3)
    edges, face_edges = _edge_arrays(faces)                  # (E,2), (F,n)
    e_first, e_second, e_counts = _edge_adjacency(edges, face_edges, n_f)
    boundary = e_counts < 2

    # --- edge points
    edge_pts = np.empty((len(edges), 3), dtype=np.float64)
    inter = ~boundary
    if inter.any():
        ei = np.nonzero(inter)[0]
        e = edges[ei]
        edge_pts[ei] = (verts[e[:, 0]] + verts[e[:, 1]]
                        + face_pts[e_first[ei]] + face_pts[e_second[ei]]) * 0.25
    if boundary.any():
        e = edges[boundary]
        edge_pts[boundary] = (verts[e[:, 0]] + verts[e[:, 1]]) * 0.5

    # --- vertex adjacency
    vert_edges = [[] for _ in range(n_v)]
    vert_faces = [[] for _ in range(n_v)]
    for ei, (a, b) in enumerate(edges):
        vert_edges[int(a)].append(ei)
        vert_edges[int(b)].append(ei)
    for fi, f in enumerate(faces):
        for v in f:
            vert_faces[int(v)].append(fi)

    boundary_verts = set()
    for ei in np.nonzero(boundary)[0]:
        boundary_verts.add(int(edges[ei, 0]))
        boundary_verts.add(int(edges[ei, 1]))

    sharp_verts = set()
    for a, b in sharp_edges:
        sharp_verts.add(a)
        sharp_verts.add(b)

    # --- new vertex positions
    new_verts = np.empty((n_v + len(edges) + n_f, 3), dtype=np.float64)
    new_verts[:n_v] = verts
    new_verts[n_v:n_v + len(edges)] = edge_pts
    new_verts[n_v + len(edges):] = face_pts

    for v in range(n_v):
        inc_e = vert_edges[v]
        if not inc_e:
            continue
        # Boundary rule: 3/4 P + 1/8 (two boundary neighbours).
        if v in boundary_verts:
            bnd = [ei for ei in inc_e if boundary[ei]]
            if len(bnd) >= 2:
                others = []
                for ei in bnd[:2]:
                    a, b = int(edges[ei, 0]), int(edges[ei, 1])
                    others.append(b if a == v else a)
                new_verts[v] = (verts[v] * 6.0
                                + verts[others].sum(axis=0)) / 8.0
                continue
        # Crease rule: a vertex on exactly two sharp edges follows the crease
        # curve (1/8, 6/8, 1/8) instead of the smooth surface.
        if v in sharp_verts:
            sh = [ei for ei in inc_e
                  if (min(int(edges[ei, 0]), int(edges[ei, 1])),
                      max(int(edges[ei, 0]), int(edges[ei, 1]))) in sharp_edges]
            if len(sh) == 2:
                others = []
                for ei in sh:
                    a, b = int(edges[ei, 0]), int(edges[ei, 1])
                    others.append(b if a == v else a)
                new_verts[v] = (verts[v] * 6.0
                                + verts[others].sum(axis=0)) / 8.0
                continue
        n = len(vert_faces[v])
        if n == 0:
            continue
        F = face_pts[vert_faces[v]].mean(axis=0)
        mids = np.array([(verts[edges[ei, 0]] + verts[edges[ei, 1]]) * 0.5
                         for ei in inc_e])
        R = mids.mean(axis=0)
        new_verts[v] = (F + 2.0 * R + (n - 3.0) * verts[v]) / float(n)

    # --- faces: every n-gon becomes n quads, each quad two triangles
    quads = []
    for fi in range(n_f):
        f = faces[fi]
        fp = n_v + len(edges) + fi
        for k in range(n_corner):
            v0 = int(f[k])
            e_next = int(face_edges[fi, k])
            e_prev = int(face_edges[fi, (k - 1) % n_corner])
            quads.append((v0, n_v + e_next, fp, n_v + e_prev))
    q = np.asarray(quads, dtype=np.int64)
    tris = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    return Mesh(new_verts, tris, dict(mesh.meta))


# ---------------------------------------------------------------- loop

def loop(mesh: Mesh, levels: int = 1) -> Mesh:
    """Loop subdivision for triangle meshes (3/8-3/8-1/8 stencil)."""
    levels = int(levels)
    if levels < 0:
        raise AttaError("bad_arg", "levels must be >= 0", path="levels",
                        fix="use levels: 2")
    out = mesh.copy()
    for _ in range(levels):
        out = _loop_once(out)
    return out


def _loop_once(mesh: Mesh) -> Mesh:
    verts = np.asarray(mesh.verts, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_v = len(verts)
    edges, face_edges = _edge_arrays(faces)
    e_first, e_second, e_counts = _edge_adjacency(edges, face_edges, len(faces))
    boundary = e_counts < 2

    edge_pts = np.empty((len(edges), 3), dtype=np.float64)
    inter = ~boundary
    if inter.any():
        ei = np.nonzero(inter)[0]
        e = edges[ei]
        opp = np.zeros((len(ei), 3), dtype=np.float64)
        for k, j in enumerate(ei):
            a = _opposite_vertex(faces, edges[j], int(e_first[j]))
            b = _opposite_vertex(faces, edges[j], int(e_second[j]))
            if a >= 0 and b >= 0:
                opp[k] = verts[a] + verts[b]
            elif a >= 0:
                opp[k] = verts[a] * 2.0
            elif b >= 0:
                opp[k] = verts[b] * 2.0
        edge_pts[ei] = (verts[e[:, 0]] + verts[e[:, 1]]) * 0.375 + opp * 0.125
    if boundary.any():
        e = edges[boundary]
        edge_pts[boundary] = (verts[e[:, 0]] + verts[e[:, 1]]) * 0.5

    vert_edges = [[] for _ in range(n_v)]
    for ei, (a, b) in enumerate(edges):
        vert_edges[int(a)].append(ei)
        vert_edges[int(b)].append(ei)
    bnd_verts = set()
    for ei in np.nonzero(boundary)[0]:
        bnd_verts.add(int(edges[ei, 0]))
        bnd_verts.add(int(edges[ei, 1]))

    new_verts = np.empty((n_v + len(edges), 3), dtype=np.float64)
    new_verts[:n_v] = verts
    new_verts[n_v:] = edge_pts
    for v in range(n_v):
        inc = vert_edges[v]
        if not inc:
            continue
        if v in bnd_verts:
            mids = []
            for ei in inc:
                if boundary[ei]:
                    a, b = int(edges[ei, 0]), int(edges[ei, 1])
                    mids.append(b if a == v else a)
            if len(mids) >= 2:
                new_verts[v] = (verts[v] * 6.0
                                + verts[mids[:2]].sum(axis=0)) / 8.0
            continue
        neigh = []
        for ei in inc:
            a, b = int(edges[ei, 0]), int(edges[ei, 1])
            neigh.append(b if a == v else a)
        n = len(neigh)
        beta = (1.0 / n) * (5.0 / 8.0
                            - (3.0 / 8.0 + 0.25 * np.cos(2.0 * np.pi / n)) ** 2)
        new_verts[v] = ((1.0 - n * beta) * verts[v]
                        + beta * verts[neigh].sum(axis=0))

    tris = []
    for fi in range(len(faces)):
        a, b, c = (int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2]))
        ab = n_v + int(face_edges[fi, 0])
        bc = n_v + int(face_edges[fi, 1])
        ca = n_v + int(face_edges[fi, 2])
        tris.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
    return Mesh(new_verts, np.asarray(tris, dtype=np.int64), dict(mesh.meta))


# ---------------------------------------------------------------- simple

def _simple(mesh: Mesh, levels: int = 1) -> Mesh:
    """Linear midpoint split: no smoothing, just four times the triangles."""
    out = mesh.copy()
    for _ in range(int(levels)):
        verts = np.asarray(out.verts, dtype=np.float64)
        faces = np.asarray(out.faces, dtype=np.int64)
        n_v = len(verts)
        edges, face_edges = _edge_arrays(faces)
        mids = (verts[edges[:, 0]] + verts[edges[:, 1]]) * 0.5
        new_verts = np.vstack([verts, mids])
        tris = []
        for fi in range(len(faces)):
            a, b, c = (int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2]))
            ab = n_v + int(face_edges[fi, 0])
            bc = n_v + int(face_edges[fi, 1])
            ca = n_v + int(face_edges[fi, 2])
            tris.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        out = Mesh(new_verts, np.asarray(tris, dtype=np.int64), dict(out.meta))
    return out


# ---------------------------------------------------------------- dispatch

def subdivide(mesh: Mesh, scheme: str = "catmull_clark", levels: int = 1,
              **kwargs) -> Mesh:
    """Subdivide with the named scheme."""
    name = str(scheme or "catmull_clark").lower()
    if name in ("catmull_clark", "catmull-clark", "cc", "quad"):
        return catmull_clark(mesh, levels=levels, **kwargs)
    if name in ("loop", "tri", "triangle"):
        return loop(mesh, levels=levels)
    if name in ("simple", "linear", "midpoint"):
        return _simple(mesh, levels=levels)
    raise AttaError("bad_arg", "unknown subdivision scheme %r" % scheme,
                    path="scheme", fix="use catmull_clark, loop or simple")


def crease_from_angle(mesh: Mesh, angle_deg: float = 40.0):
    """Edges whose adjacent faces meet at more than ``angle_deg``.

    The bridge from a hard-surface cage to a subdivided result: crease the
    sharp edges and a box keeps its edges while its corners round off.
    """
    verts = np.asarray(mesh.verts, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return []
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln > 0.0, ln, 1.0)

    edges, face_edges = _edge_arrays(faces)
    f1, f2, counts = _edge_adjacency(edges, face_edges, len(faces))
    cos_lim = float(np.cos(np.deg2rad(float(angle_deg))))
    out = []
    for ei in range(len(edges)):
        if counts[ei] < 2:
            out.append((int(edges[ei, 0]), int(edges[ei, 1])))
            continue
        if float(n[f1[ei]] @ n[f2[ei]]) < cos_lim:
            out.append((int(edges[ei, 0]), int(edges[ei, 1])))
    return out
