# -*- coding: utf-8 -*-
"""Bounded regular-grid sampling and exact-edge marching tetrahedra.

Coordinates, spacing and field distances use metres.  Extraction preserves the
translation-invariant tetrahedra and interpolation policy of the original
lattice kernel; only sign-changing cells enter its per-tetrahedron work arrays.
"""
from __future__ import annotations

import math

import numpy as np

from .errors import AttaError
from .mesh import Mesh, oriented

DEFAULT_MAX_SAMPLES = 2_500_000
DEFAULT_CHUNK_CELLS = 16_384
DEFAULT_MAX_TRIANGLES = 2_000_000

_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                     [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float64)
_TETS = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
                  [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]], dtype=np.int64)
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_CASES = {
    1: ((0, 1, 2),), 2: ((0, 3, 4),), 4: ((1, 3, 5),), 8: ((2, 4, 5),),
    3: ((1, 2, 4), (1, 4, 3)), 5: ((0, 3, 5), (0, 5, 2)),
    6: ((0, 1, 5), (0, 5, 4)), 7: ((2, 4, 5),),
    9: ((0, 1, 5), (0, 5, 4)), 10: ((0, 2, 5), (0, 5, 3)),
    11: ((1, 3, 5),), 12: ((1, 3, 4), (1, 4, 2)),
    13: ((0, 3, 4),), 14: ((0, 1, 2),),
}


def grid_shape(lo, hi, spacing):
    """Return the actual padded shape without allocating a sample grid."""
    try:
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        spacing = float(spacing)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(lo).all() \
                or not np.isfinite(hi).all() or np.any(hi < lo) \
                or not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("bounds or spacing are invalid")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return tuple(int(math.ceil((hi[i] - lo[i] + 5.0 * spacing) / spacing)) + 1
                         for i in range(3))
    except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        raise AttaError("bad_arg", "grid bounds and spacing must be finite, ordered and representable",
                        path="spacing", fix="use positive spacing and local finite lo <= hi") from exc


def fit_spacing(lo, hi, spacing, max_samples=DEFAULT_MAX_SAMPLES):
    """Coarsen until the *rounded, padded* shape is within the sample budget.

    The padding is five voxels independent of physical extent; samples do not
    scale exactly as spacing**-3.  Recompute after every step, including ceil.
    """
    try:
        if isinstance(max_samples, (bool, np.bool_)) or int(max_samples) != max_samples:
            raise ValueError("sample budget must be an integer")
        max_samples = int(max_samples)
        spacing = float(spacing)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AttaError("bad_arg", "sample budget and spacing must be representable numbers",
                        path="max_samples", fix="use an integer budget and positive finite spacing") from exc
    if max_samples < 343:
        raise AttaError("bad_arg", "sample budget cannot contain the padded grid",
                        path="max_samples", fix="use max_samples >= 343")
    spacing = float(spacing)
    for _ in range(128):
        shape = grid_shape(lo, hi, spacing)
        samples = math.prod(shape)
        if samples <= max_samples:
            return spacing
        spacing *= max(1.01, (samples / max_samples) ** (1.0 / 3.0))
    raise AttaError("sdf_too_dense", "grid cannot fit the sample budget",
                    path="spacing", fix="increase max_samples or reduce the bounds")


def _cell_triangles(f8, cell, base_node, delta, origin, spacing, n_tot):
    """Yield the legacy triangle edge keys and positions for active cells."""
    base = origin + cell * spacing
    for tet in _TETS:
        fv = f8[:, tet]
        inside = fv < 0.0
        cid = (inside.astype(np.int64) * np.array([1, 2, 4, 8])).sum(axis=1)
        if not np.any((cid > 0) & (cid < 15)):
            continue
        nodes4 = base_node[:, None] + delta[tet][None, :]
        p = base[:, None, :] + _CORNERS[tet][None, :, :] * spacing
        e = np.empty((len(cell), 6, 3))
        keys = np.empty((len(cell), 6), dtype=np.int64)
        for k, (a, b) in enumerate(_TET_EDGES):
            fa, fb = fv[:, a], fv[:, b]
            den = fa - fb
            safe = np.where(np.abs(den) > 0.0, den, 1.0)
            t = np.clip(fa / safe, 1e-3, 1.0 - 1e-3)
            e[:, k] = p[:, a] + t[:, None] * (p[:, b] - p[:, a])
            na, nb = nodes4[:, a], nodes4[:, b]
            keys[:, k] = np.minimum(na, nb) * n_tot + np.maximum(na, nb)
        cnt_in = inside.sum(axis=1)[:, None]
        cin = np.where(inside[:, :, None], p, 0.0).sum(axis=1) / np.maximum(cnt_in, 1)
        cout = np.where(~inside[:, :, None], p, 0.0).sum(axis=1) / np.maximum(4 - cnt_in, 1)
        direction = cout - cin
        for case, tris in _CASES.items():
            mask = cid == case
            if not mask.any():
                continue
            for e0, e1, e2 in tris:
                q0, q1, q2 = e[mask, e0], e[mask, e1], e[mask, e2]
                flip = np.einsum("ij,ij->i", np.cross(q1 - q0, q2 - q0), direction[mask]) < 0.0
                yield (np.stack([keys[mask, e0],
                                 np.where(flip, keys[mask, e2], keys[mask, e1]),
                                 np.where(flip, keys[mask, e1], keys[mask, e2])], axis=1),
                       np.stack([q0, np.where(flip[:, None], q2, q1),
                                 np.where(flip[:, None], q1, q2)], axis=1))


def marching_tets(field, origin, spacing, *, chunk_cells=DEFAULT_CHUNK_CELLS,
                  max_triangles=DEFAULT_MAX_TRIANGLES):
    """Extract negative-inside samples with bounded per-cell working memory.

    Blocks own disjoint cells, but all use original grid node IDs.  Shared edges
    are joined by integer keys, including across blocks, without spatial welds.
    The output budget is checked before retaining each triangle batch.
    """
    try:
        f = np.asarray(field, dtype=np.float64)
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        spacing = np.broadcast_to(np.asarray(spacing, dtype=np.float64).reshape(-1), (3,)).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise AttaError("bad_arg", "field, origin or spacing has invalid numeric shape",
                        path="field", fix="use a numeric 3D field, three origin coordinates and positive spacing") from exc
    if f.ndim != 3 or min(f.shape) < 2:
        raise AttaError("bad_arg", "field needs at least 2 samples per axis",
                        path="field", fix="pass a 3D field with >= 2 samples per axis")
    if not np.isfinite(origin).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise AttaError("bad_arg", "origin and spacing must be finite; spacing positive",
                        path="spacing", fix="pass metre-based coordinates and positive spacing")
    try:
        if any(isinstance(v, (bool, np.bool_)) or int(v) != v
               for v in (chunk_cells, max_triangles)):
            raise ValueError("extraction budgets must be integers")
        chunk_cells, max_triangles = int(chunk_cells), int(max_triangles)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AttaError("bad_arg", "extraction budgets must be finite integers",
                        path="field", fix="use positive integer chunk_cells and max_triangles") from exc
    if chunk_cells < 1 or max_triangles < 1:
        raise AttaError("bad_arg", "extraction budgets must be positive",
                        path="field", fix="use positive chunk_cells and max_triangles")
    nx, ny, nz = f.shape
    cx, cy, cz = nx - 1, ny - 1, nz - 1
    n_tot = math.prod(f.shape)
    if n_tot > math.isqrt(np.iinfo(np.int64).max):
        raise AttaError("sdf_too_dense", "grid exceeds the integer edge-key range",
                        path="field", fix="reduce the sample count")
    delta = ((_CORNERS[:, 0].astype(np.int64) * ny
              + _CORNERS[:, 1].astype(np.int64)) * nz
             + _CORNERS[:, 2].astype(np.int64))
    all_keys, all_positions, all_faces = [], [], []
    retained_vertices = triangle_count = active_cells = 0
    flat = f.ravel()
    # Check finite values in bounded slices, even when a sample is unused.
    for start in range(0, len(flat), chunk_cells):
        if not np.isfinite(flat[start:start + chunk_cells]).all():
            raise AttaError("bad_arg", "field contains NaN or infinity",
                            path="field", fix="sample a finite scalar field")
    for start in range(0, cx * cy * cz, chunk_cells):
        ids = np.arange(start, min(start + chunk_cells, cx * cy * cz), dtype=np.int64)
        cell = np.column_stack([ids // (cy * cz), (ids // cz) % cy, ids % cz])
        base_node = (cell[:, 0] * ny + cell[:, 1]) * nz + cell[:, 2]
        f8 = flat[base_node[:, None] + delta]
        negative = f8 < 0.0
        active = negative.any(axis=1) & ~negative.all(axis=1)
        if not active.any():
            continue
        cell, base_node, f8 = cell[active], base_node[active], f8[active]
        active_cells += len(cell)
        chunk_keys, chunk_positions = [], []
        for keys, positions in _cell_triangles(f8, cell, base_node, delta,
                                                origin, spacing, n_tot):
            triangle_count += len(keys)
            if triangle_count > max_triangles:
                raise AttaError("sdf_too_dense", "extracted surface exceeds %d triangles" % max_triangles,
                                path="field", fix="increase spacing or reduce field detail",
                                details={"max_triangles": max_triangles,
                                         "triangles_at_least": triangle_count})
            chunk_keys.append(keys)
            chunk_positions.append(positions)
        if not chunk_keys:
            continue
        keys = np.concatenate(chunk_keys).ravel()
        positions = np.concatenate(chunk_positions).reshape(-1, 3)
        unique, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        all_keys.append(unique)
        all_positions.append(positions[first])
        all_faces.append(inverse.reshape(-1, 3) + retained_vertices)
        retained_vertices += len(unique)
    if not all_keys:
        raise AttaError("lattice_empty", "the field has no negative region",
                        path="field", fix="increase thickness or cell_size")
    keys = np.concatenate(all_keys)
    positions = np.concatenate(all_positions)
    faces = np.concatenate(all_faces)
    del all_keys, all_positions, all_faces
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    verts = positions[first]
    faces = inverse[faces]
    del keys, positions, inverse
    meta = {"kind": "marching_tets", "extraction": {
        "method": "active_chunked_tets", "samples": n_tot,
        "cells": cx * cy * cz, "active_cells": active_cells,
        "chunk_cells": chunk_cells, "max_triangles": max_triangles,
        "triangles": triangle_count,
    }}
    return oriented(Mesh(verts, faces, meta))
