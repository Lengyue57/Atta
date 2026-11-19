# -*- coding: utf-8 -*-
"""Signed-distance modelling: smooth blends, displacement, noise, warps.

The v1.2 implicit path had one operator: ``max`` (a hard union) and one field
family (TPMS).  That is enough for a lattice and useless for a creature: hard
unions leave visible creases where two forms meet, and there is no way to add
surface interest.

This module is the missing vocabulary.  Everything is a numpy function over a
sampled field, so it composes freely:

Operators
    ``union`` / ``subtract`` / ``intersect`` (hard), ``smooth_union`` /
    ``smooth_subtract`` / ``smooth_intersect`` (polynomial and exponential
    blends), ``chamfer_union`` (a 45-degree crease instead of a fillet),
    ``stair_union`` (a stepped seam), ``round_union``.

Primitives
    sphere, box, cylinder, cone, torus, capsule, plane, ellipsoid, and the
    ``mesh_sdf`` bridge that turns any existing triangle mesh into a field.

Modifiers
    ``displace`` (along the normal by a noise field), ``twist``, ``bend``,
    ``taper``, ``repeat`` (infinite/repeated domain), ``elongate``,
    ``shell``, ``onion``.

Noise
    Value noise, fBm, ridged multifractal and Worley/cellular, all in numpy
    with no external dependency, seeded and deterministic.

The field convention matches :mod:`atta.families.lattice`: negative is inside,
positive is outside, units are metres.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    # operators
    "union", "subtract", "intersect",
    "smooth_union", "smooth_subtract", "smooth_intersect",
    "chamfer_union", "stair_union", "round_union",
    # primitives
    "sphere", "box", "cylinder", "cone", "torus", "capsule", "plane",
    "ellipsoid", "mesh_sdf",
    # modifiers
    "displace", "twist", "bend", "taper", "repeat", "elongate",
    "shell", "onion", "offset",
    # noise
    "value_noise", "fbm", "ridged", "worley",
    # TPMS analytic fields, metric sheet distance, topology & extraction guards
    "TPMS_SPECS", "tpms_unit_cell_genus", "tpms_eval", "tpms_block_genus",
    "tpms_sheet_distance", "tpms_sheet_volume_estimate",
    "feature_spacing", "prune_boundary_slivers", "cap_boundary",
    # grid
    "grid", "sample", "extract",
]


# ---------------------------------------------------------------- operators

def union(a, b):
    return np.minimum(a, b)


def subtract(a, b):
    """``a`` minus ``b``."""
    return np.maximum(a, -b)


def intersect(a, b):
    return np.maximum(a, b)


def smooth_union(a, b, k):
    """Polynomial smooth minimum: rounds the crease where two forms meet.

    ``k`` is the blend radius in metres.  This is the single most useful
    operator for organic form: two overlapping spheres become one mass instead
    of two lumps with a seam.
    """
    if k <= 0.0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def smooth_subtract(a, b, k):
    """``a`` minus ``b`` with a rounded lip."""
    return -smooth_union(-a, b, k)


def smooth_intersect(a, b, k):
    return -smooth_union(-a, -b, k)


def chamfer_union(a, b, k):
    """Union with a flat 45-degree chamfer instead of a fillet."""
    if k <= 0.0:
        return np.minimum(a, b)
    return np.minimum(np.minimum(a, b), (a - k + b) * 0.7071067811865476)


def stair_union(a, b, k, steps=4):
    """Union with a stepped seam — the "machined" look."""
    if k <= 0.0 or steps < 1:
        return np.minimum(a, b)
    d = np.clip((b - a + k) / (2.0 * k), 0.0, 1.0)
    q = np.floor(d * float(steps)) / float(steps)
    return a + (b - a) * q - k * q * (1.0 - q)


def round_union(a, b, k):
    """Union that rounds outward (adds material at the seam)."""
    return smooth_union(a, b, k) - k * 0.25


# ---------------------------------------------------------------- primitives

def sphere(X, Y, Z, center=(0.0, 0.0, 0.0), radius=1.0):
    cx, cy, cz = center
    return np.hypot(np.hypot(X - cx, Y - cy), Z - cz) - float(radius)


def box(X, Y, Z, size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0), round_r=0.0):
    """Exact box SDF; ``round_r`` rounds the edges (a Minkowski sum with a ball)."""
    hx, hy, hz = (0.5 * float(s) for s in size)
    cx, cy, cz = center
    qx = np.abs(X - cx) - hx + round_r
    qy = np.abs(Y - cy) - hy + round_r
    qz = np.abs(Z - cz) - hz + round_r
    outside = np.hypot(np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)),
                       np.maximum(qz, 0.0))
    inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0.0)
    return outside + inside - round_r


def cylinder(X, Y, Z, radius=1.0, height=1.0, center=(0.0, 0.0, 0.0),
             axis="z", round_r=0.0):
    cx, cy, cz = center
    dx, dy, dz = X - cx, Y - cy, Z - cz
    if axis == "x":
        u, v, w = dy, dz, dx
    elif axis == "y":
        u, v, w = dx, dz, dy
    else:
        u, v, w = dx, dy, dz
    radial = np.hypot(u, v) - float(radius) + round_r
    axial = np.abs(w) - 0.5 * float(height) + round_r
    outside = np.hypot(np.maximum(radial, 0.0), np.maximum(axial, 0.0))
    inside = np.minimum(np.maximum(radial, axial), 0.0)
    return outside + inside - round_r


def cone(X, Y, Z, r_bottom=1.0, r_top=0.0, height=1.0,
         center=(0.0, 0.0, 0.0)):
    """Capped cone/frustum, axis +Z, centred on ``center``."""
    cx, cy, cz = center
    dx, dy, dz = X - cx, Y - cy, Z - cz
    h = float(height)
    z = np.clip(dz + 0.5 * h, 0.0, h)
    r_at = r_bottom + (r_top - r_bottom) * (z / h if h else 0.0)
    radial = np.sqrt(dx * dx + dy * dy) - r_at
    axial = np.abs(dz) - 0.5 * h
    return np.maximum(radial, axial)


def torus(X, Y, Z, major=1.0, minor=0.2, center=(0.0, 0.0, 0.0), axis="z"):
    cx, cy, cz = center
    dx, dy, dz = X - cx, Y - cy, Z - cz
    if axis == "z":
        radial = np.hypot(dx, dy)
        return np.hypot(radial - float(major), dz) - float(minor)
    if axis == "x":
        radial = np.hypot(dy, dz)
        return np.hypot(radial - float(major), dx) - float(minor)
    radial = np.hypot(dx, dz)
    return np.hypot(radial - float(major), dy) - float(minor)


def capsule(X, Y, Z, a=(0.0, 0.0, -1.0), b=(0.0, 0.0, 1.0), radius=0.2):
    """Exact capsule: the Minkowski sum of a segment and a ball."""
    ax, ay, az = a
    bx, by, bz = b
    pax, pay, paz = X - ax, Y - ay, Z - az
    ex, ey, ez = bx - ax, by - ay, bz - az
    den = ex * ex + ey * ey + ez * ez
    if den <= 1e-18:
        return np.sqrt(pax * pax + pay * pay + paz * paz) - float(radius)
    t = np.clip((pax * ex + pay * ey + paz * ez) / den, 0.0, 1.0)
    dx = pax - t * ex
    dy = pay - t * ey
    dz = paz - t * ez
    return np.sqrt(dx * dx + dy * dy + dz * dz) - float(radius)


def plane(X, Y, Z, normal=(0.0, 0.0, 1.0), offset=0.0):
    n = np.asarray(normal, dtype=np.float64)
    ln = float(np.linalg.norm(n))
    if ln <= 1e-15:
        return np.full(X.shape, np.inf)
    n = n / ln
    return X * n[0] + Y * n[1] + Z * n[2] - float(offset)


def ellipsoid(X, Y, Z, radii=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)):
    """Approximate ellipsoid SDF (the standard scaled-sphere bound)."""
    rx, ry, rz = (max(float(r), 1e-12) for r in radii)
    cx, cy, cz = center
    qx = (X - cx) / rx
    qy = (Y - cy) / ry
    qz = (Z - cz) / rz
    k0 = np.sqrt(qx * qx + qy * qy + qz * qz)
    k1 = np.sqrt((qx / rx) ** 2 + (qy / ry) ** 2 + (qz / rz) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)


def mesh_sdf(X, Y, Z, verts, faces, *, sign="winding"):
    """Signed distance to a triangle mesh, sampled on a grid.

    Uses point-to-triangle distance for magnitude and a ray-parity test for
    sign.  This is the bridge that lets an existing recipe become a field term
    — e.g. smooth-union a mug into a creature's hand.
    """
    from . import metrology

    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    dist = _point_tri_distance(pts, v, f)
    inside = _ray_parity(pts, v, f) if sign == "winding" else np.zeros(len(pts), bool)
    phi = np.where(inside, -dist, dist).reshape(X.shape)
    return phi


def _point_tri_distance(pts, verts, faces):
    """Unsigned distance from points to a triangle soup (vectorised per face)."""
    best = np.full(len(pts), np.inf)
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    for i in range(len(faces)):
        best = np.minimum(best, _pt_tri(pts, a[i], b[i], c[i]))
    return best


def _pt_tri(p, a, b, c):
    """Point-to-triangle distance (Ericson, Real-Time Collision Detection)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ap @ ab
    d2 = ap @ ac
    bp = p - b
    d3 = bp @ ab
    d4 = bp @ ac
    cp = p - c
    d5 = cp @ ab
    d6 = cp @ ac
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    out = np.empty(len(p), dtype=np.float64)
    # Regions
    m = (d1 <= 0) & (d2 <= 0)
    out[m] = np.linalg.norm(ap[m], axis=1)
    m2 = (d3 >= 0) & (d4 <= d3)
    out[m2] = np.linalg.norm(bp[m2], axis=1)
    m3 = (d6 >= 0) & (d5 <= d6)
    out[m3] = np.linalg.norm(cp[m3], axis=1)
    m4 = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = np.where(np.abs(d1 - d3) > 1e-18, d1 / (d1 - d3), 0.0)
    out[m4] = np.linalg.norm(p[m4] - (a + t[m4, None] * ab), axis=1)
    m5 = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = np.where(np.abs(d2 - d6) > 1e-18, d2 / (d2 - d6), 0.0)
    out[m5] = np.linalg.norm(p[m5] - (a + t[m5, None] * ac), axis=1)
    m6 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = np.where(np.abs((d4 - d3) + (d5 - d6)) > 1e-18,
                 (d4 - d3) / ((d4 - d3) + (d5 - d6)), 0.0)
    out[m6] = np.linalg.norm(p[m6] - (b + t[m6, None] * (c - b)), axis=1)
    rest = ~(m | m2 | m3 | m4 | m5 | m6)
    if rest.any():
        den = np.where(np.abs(denom) > 1e-18, denom, 1.0)
        v = vb[rest] / den[rest]
        w = vc[rest] / den[rest]
        proj = a + v[:, None] * ab + w[:, None] * ac
        out[rest] = np.linalg.norm(p[rest] - proj, axis=1)
    return out


def _ray_parity(pts, verts, faces, direction=(0.0, 0.0, 1.0)):
    """Inside test by counting ray-triangle hits (Möller-Trumbore)."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    inside = np.zeros(len(pts), dtype=bool)
    a = verts[faces[:, 0]]
    e1 = verts[faces[:, 1]] - a
    e2 = verts[faces[:, 2]] - a
    for i in range(len(faces)):
        h = np.cross(d, e2[i])
        det = float(e1[i] @ h)
        if abs(det) < 1e-15:
            continue
        inv = 1.0 / det
        s = pts - a[i]
        u = inv * (s @ h)
        ok_u = (u >= 0.0) & (u <= 1.0)
        q = np.cross(s, e1[i])
        v = inv * (q @ d)
        ok_v = (v >= 0.0) & (u + v <= 1.0)
        t = inv * (q @ e2[i])
        hit = ok_u & ok_v & (t > 1e-12)
        inside ^= hit
    return inside


# ---------------------------------------------------------------- modifiers

def displace(phi, X, Y, Z, amount, frequency=1.0, seed=0, octaves=1,
             kind="fbm"):
    """Push the surface along its normal by a noise field.

    ``amount`` is the peak displacement in metres; ``frequency`` is cycles per
    metre.  This is how a smooth blob gets skin, bark or rock character without
    changing its silhouette much.
    """
    n = _noise_field(X, Y, Z, frequency=frequency, seed=seed, octaves=octaves,
                     kind=kind)
    return phi - float(amount) * n


def twist(phi_fn, X, Y, Z, angle_deg=0.0, axis="z", height=None):
    """Twist the field about ``axis``.  ``phi_fn`` is called with warped coords."""
    if angle_deg == 0.0:
        return phi_fn(X, Y, Z)
    ang = np.deg2rad(float(angle_deg))
    if axis == "z":
        t = (Z - Z.min()) / max(float(Z.max() - Z.min()), 1e-12)
        a = ang * (t - 0.5)
        ca, sa = np.cos(a), np.sin(a)
        return phi_fn(ca * X - sa * Y, sa * X + ca * Y, Z)
    if axis == "x":
        t = (X - X.min()) / max(float(X.max() - X.min()), 1e-12)
        a = ang * (t - 0.5)
        ca, sa = np.cos(a), np.sin(a)
        return phi_fn(X, ca * Y - sa * Z, sa * Y + ca * Z)
    t = (Y - Y.min()) / max(float(Y.max() - Y.min()), 1e-12)
    a = ang * (t - 0.5)
    ca, sa = np.cos(a), np.sin(a)
    return phi_fn(ca * X + sa * Z, Y, -sa * X + ca * Z)


def bend(phi_fn, X, Y, Z, angle_deg=0.0, axis="x"):
    """Bend the field about ``axis`` (a circular arc deformation)."""
    if angle_deg == 0.0:
        return phi_fn(X, Y, Z)
    ang = np.deg2rad(float(angle_deg))
    if axis == "x":
        span = max(float(X.max() - X.min()), 1e-12)
        radius = span / max(abs(ang), 1e-9)
        cx = 0.5 * float(X.max() + X.min())
        u = X - cx
        theta = u / radius
        r = radius + Z
        return phi_fn(cx + r * np.sin(theta), Y, r * np.cos(theta) - radius)
    span = max(float(Y.max() - Y.min()), 1e-12)
    radius = span / max(abs(ang), 1e-9)
    cy = 0.5 * float(Y.max() + Y.min())
    u = Y - cy
    theta = u / radius
    r = radius + Z
    return phi_fn(X, cy + r * np.sin(theta), r * np.cos(theta) - radius)


def taper(phi_fn, X, Y, Z, start_scale=1.0, end_scale=1.0, axis="z"):
    """Scale the cross-section linearly along ``axis``."""
    if axis == "z":
        lo, hi = float(Z.min()), float(Z.max())
        t = (Z - lo) / max(hi - lo, 1e-12)
        s = start_scale + (end_scale - start_scale) * t
        s = np.maximum(s, 1e-6)
        return phi_fn(X / s, Y / s, Z) * np.minimum(s, 1.0)
    if axis == "x":
        lo, hi = float(X.min()), float(X.max())
        t = (X - lo) / max(hi - lo, 1e-12)
        s = np.maximum(start_scale + (end_scale - start_scale) * t, 1e-6)
        return phi_fn(X, Y / s, Z / s) * np.minimum(s, 1.0)
    lo, hi = float(Y.min()), float(Y.max())
    t = (Y - lo) / max(hi - lo, 1e-12)
    s = np.maximum(start_scale + (end_scale - start_scale) * t, 1e-6)
    return phi_fn(X / s, Y, Z / s) * np.minimum(s, 1.0)


def repeat(phi_fn, X, Y, Z, period=(0.0, 0.0, 0.0), limit=None):
    """Repeat the domain: infinite arrays of the same form."""
    px, py, pz = (float(p) for p in period)
    wx = _wrap(X, px)
    wy = _wrap(Y, py)
    wz = _wrap(Z, pz)
    phi = phi_fn(wx, wy, wz)
    if limit is not None:
        lx, ly, lz = (float(v) for v in limit)
        box_d = np.maximum(np.abs(X) - lx, np.maximum(np.abs(Y) - ly,
                                                     np.abs(Z) - lz))
        phi = np.maximum(phi, box_d)
    return phi


def _wrap(v, period):
    if period <= 0.0:
        return v
    return (v + 0.5 * period) % period - 0.5 * period


def elongate(phi_fn, X, Y, Z, amount=(0.0, 0.0, 0.0)):
    """Stretch the form by moving the domain inward along each axis."""
    ax, ay, az = (float(a) for a in amount)
    return phi_fn(X - np.clip(X, -0.5 * ax, 0.5 * ax),
                  Y - np.clip(Y, -0.5 * ay, 0.5 * ay),
                  Z - np.clip(Z, -0.5 * az, 0.5 * az))


def shell(phi, thickness):
    """Hollow a solid: keep a wall of ``thickness`` metres."""
    return np.maximum(phi, -(phi + float(thickness)))


def onion(phi, thickness):
    """A closed shell of the given thickness (no openings)."""
    return np.abs(phi) - 0.5 * float(thickness)


def offset(phi, distance):
    """Grow (positive) or shrink (negative) the solid."""
    return phi - float(distance)


# ---------------------------------------------------------------- noise

def _hash3(ix, iy, iz, seed):
    """Deterministic integer hash -> float in [0, 1).

    All arithmetic stays in int64 with explicit wrapping, so it is stable
    across platforms and does not emit numpy overflow warnings.
    """
    with np.errstate(over="ignore"):
        h = (ix.astype(np.int64) * np.int64(374761393)
             + iy.astype(np.int64) * np.int64(668265263)
             + iz.astype(np.int64) * np.int64(2147483647)
             + np.int64(seed) * np.int64(1442695041))
        h = (h ^ (h >> np.int64(13))) * np.int64(1274126177)
        h = h ^ (h >> np.int64(16))
    return (h & np.int64(0x7FFFFFFF)).astype(np.float64) / float(0x7FFFFFFF)


def value_noise(X, Y, Z, frequency=1.0, seed=0):
    """Trilinear value noise in [0, 1], no external dependency."""
    x = X * frequency
    y = Y * frequency
    z = Z * frequency
    ix, iy, iz = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - ix, y - iy, z - iz
    # Smoothstep for C1 continuity
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    uz = fz * fz * (3.0 - 2.0 * fz)
    ix = ix.astype(np.int64)
    iy = iy.astype(np.int64)
    iz = iz.astype(np.int64)

    def corner(dx, dy, dz):
        return _hash3(ix + dx, iy + dy, iz + dz, seed)

    c000, c100 = corner(0, 0, 0), corner(1, 0, 0)
    c010, c110 = corner(0, 1, 0), corner(1, 1, 0)
    c001, c101 = corner(0, 0, 1), corner(1, 0, 1)
    c011, c111 = corner(0, 1, 1), corner(1, 1, 1)
    x00 = c000 + ux * (c100 - c000)
    x10 = c010 + ux * (c110 - c010)
    x01 = c001 + ux * (c101 - c001)
    x11 = c011 + ux * (c111 - c011)
    y0 = x00 + uy * (x10 - x00)
    y1 = x01 + uy * (x11 - x01)
    return y0 + uz * (y1 - y0)


def fbm(X, Y, Z, frequency=1.0, seed=0, octaves=4, lacunarity=2.0,
        gain=0.5):
    """Fractal Brownian motion: layered value noise in [0, 1]."""
    total = np.zeros(X.shape, dtype=np.float64)
    amp = 1.0
    norm = 0.0
    f = float(frequency)
    for o in range(max(1, int(octaves))):
        total += amp * value_noise(X, Y, Z, frequency=f, seed=seed + o * 101)
        norm += amp
        amp *= float(gain)
        f *= float(lacunarity)
    return total / max(norm, 1e-12)


def ridged(X, Y, Z, frequency=1.0, seed=0, octaves=4):
    """Ridged multifractal: sharp crests, good for rock and bark."""
    total = np.zeros(X.shape, dtype=np.float64)
    amp = 1.0
    norm = 0.0
    f = float(frequency)
    for o in range(max(1, int(octaves))):
        n = value_noise(X, Y, Z, frequency=f, seed=seed + o * 137)
        total += amp * (1.0 - np.abs(2.0 * n - 1.0))
        norm += amp
        amp *= 0.5
        f *= 2.0
    return total / max(norm, 1e-12)


def worley(X, Y, Z, frequency=1.0, seed=0):
    """Cellular (Worley) noise: distance to the nearest feature point in [0,1]."""
    x = X * frequency
    y = Y * frequency
    z = Z * frequency
    ix, iy, iz = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), \
        np.floor(z).astype(np.int64)
    best = np.full(X.shape, np.inf)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cx, cy, cz = ix + dx, iy + dy, iz + dz
                fx = _hash3(cx, cy, cz, seed)
                fy = _hash3(cx, cy, cz, seed + 7919)
                fz = _hash3(cx, cy, cz, seed + 104729)
                px = cx + fx
                py = cy + fy
                pz = cz + fz
                d = np.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
                best = np.minimum(best, d)
    return np.clip(best, 0.0, 1.0)


def _noise_field(X, Y, Z, frequency=1.0, seed=0, octaves=1, kind="fbm"):
    if kind == "value":
        return value_noise(X, Y, Z, frequency=frequency, seed=seed)
    if kind == "ridged":
        return ridged(X, Y, Z, frequency=frequency, seed=seed, octaves=octaves)
    if kind == "worley":
        return worley(X, Y, Z, frequency=frequency, seed=seed)
    return fbm(X, Y, Z, frequency=frequency, seed=seed, octaves=octaves)


# ---------------------------------------------------------------- TPMS & topology

#: Analytic crystallographic and differential-geometry constants for the four
#: standard triply-periodic minimal surfaces (TPMS) in their conventional cubic
#: translational cell ``[0, P]^3``.
#:
#: References
#: ----------
#: * H. A. Schwarz (1890), *Gesammelte Mathematische Abhandlungen*, Springer
#:   (Schwarz P and Schwarz D minimal surfaces, 1865).
#: * A. H. Schoen (1970), *"Infinite Periodic Minimal Surfaces without
#:   Self-Intersections"*, NASA Technical Note TN D-5541 (Gyroid discovery;
#:   quotient genus and dimensionless cell surface areas of P, D, G, C(P)).
#: * W. H. Meeks III (1990), *"The Theory of Triply Periodic Minimal Surfaces"*,
#:   Indiana Univ. Math. J. 39(3):877-936 (genus >= 3 theorem on T^3).
#: * K. Grosse-Brauckmann (1997), *"Gyroids of Constant Mean Curvature"*,
#:   Experimental Mathematics 6(1):33-50 (BCC primitive cell g=3 vs conventional
#:   cubic cell g=5; cubic-cell area A/a^2 = 3.09144).
#: * H. Weyl (1939), *"On the Volume of Tubes"*, Amer. J. Math. 61(2):461-472
#:   (exact tubular volume V(t) = A*t + (pi*chi/6)*t^3 for H = 0 surfaces).
#: * O. Al-Ketan & R. K. Abu Al-Rub (2019), *"Multifunctional Mechanical
#:   Metamaterials Based on Triply Periodic Minimal Surface Lattices"*,
#:   Adv. Eng. Mater. 21(10):1900524 (sheet-based vs skeletal TPMS solids).
TPMS_SPECS: dict[str, dict] = {
    "schwarz_p": {
        "space_group": "Pm-3m",
        "lattice": "SC",
        "primitive_genus": 3,
        "primitive_cells_per_cube": 1,
        "conventional_cubic_genus": 3,
        "euler_per_cube": -4,
        # Schoen (1970) minimal P: 2.3451; trigonometric nodal cos x+cos y+cos z=0: 2.3451
        "area_coeff_cube": 2.3451,
        "offset_area_quad": 0.145,
        # Lower-corner phase in periods -> u,v,w in [pi, pi + 2*pi*n]: cos = -1 on all 6 faces
        "canonical_phase_cells": (0.5, 0.5, 0.5),
        "safe_offset_max": 0.85,
    },
    "gyroid": {
        "space_group": "Ia-3d",
        "lattice": "BCC",
        "primitive_genus": 3,
        "primitive_cells_per_cube": 2,
        "conventional_cubic_genus": 5,
        "euler_per_cube": -8,
        # Schoen (1970) / Grosse-Brauckmann (1997): 3.09144 minimal; 3.09181 nodal
        "area_coeff_cube": 3.0918,
        "offset_area_quad": 0.215,
        # Quarter-cell x shift keeps all 8 box corners off the gyroid inversion centres
        "canonical_phase_cells": (-0.25, 0.0, 0.0),
        "safe_offset_max": 0.35,
    },
    "diamond": {
        "space_group": "Fd-3m",
        "lattice": "FCC",
        "primitive_genus": 3,
        "primitive_cells_per_cube": 4,
        "conventional_cubic_genus": 9,
        "euler_per_cube": -16,
        # Schoen (1970) Schwarz D in conventional cubic [0,a]^3 (4 FCC cells): 3.8378
        "area_coeff_cube": 3.8378,
        "offset_area_quad": 0.33,
        # Phase (pi/2, pi/2, pi/2) -> F(corner)=+1, smooth diagonal cuts on all 6 faces
        "canonical_phase_cells": (0.25, 0.25, 0.25),
        "safe_offset_max": 0.55,
    },
    "neovius": {
        "space_group": "Pm-3m",
        "lattice": "SC",
        "primitive_genus": 9,
        "primitive_cells_per_cube": 1,
        "conventional_cubic_genus": 9,
        "euler_per_cube": -16,
        # Schoen (1970) C(P) / Neovius in [0,a]^3: 3.5105
        "area_coeff_cube": 3.5105,
        "offset_area_quad": 0.04,
        # Phase (pi, pi, pi) -> F(corner)=-13, 2 roots per period on each box edge
        "canonical_phase_cells": (0.5, 0.5, 0.5),
        "safe_offset_max": 0.80,
    },
}

#: Exact 14-neighbour structuring element matching the Kuhn 6-tetrahedron
#: decomposition in :mod:`atta.sdf` (edges along the six cube directions,
#: the three positive face diagonals and the (1,1,1) body diagonal).
_KUHN_14_STRUCT = np.zeros((3, 3, 3), dtype=bool)
for _dx, _dy, _dz in (
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, -1, -1),
    (1, 1, 1), (-1, -1, -1),
):
    _KUHN_14_STRUCT[1 + _dx, 1 + _dy, 1 + _dz] = True


def _check_finite_num(value, path, *, positive=False):
    from .errors import AttaError

    if isinstance(value, (bool, np.bool_)):
        raise AttaError("bad_arg", "%s must be a number, not a bool" % path,
                        path=path, fix="pass a finite numeric value")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AttaError("bad_arg", "%s must be a finite number" % path,
                        path=path, fix="pass a finite numeric value") from exc
    if not math.isfinite(out) or (positive and out <= 0.0):
        raise AttaError("bad_arg", "%s must be %s" % (path, "positive and finite" if positive else "finite"),
                        path=path, fix="pass a valid finite number")
    return out


def _check_vec3(value, path, *, positive=False):
    from .errors import AttaError

    if not isinstance(value, (list, tuple, np.ndarray)):
        raise AttaError("bad_arg", "%s must be a 3-element vector" % path,
                        path=path, fix="pass a 3-element numeric list or tuple")
    arr = np.asarray(value, dtype=object).reshape(-1)
    if arr.shape != (3,):
        raise AttaError("bad_arg", "%s must have 3 elements" % path,
                        path=path, fix="pass a 3-element numeric list or tuple")
    return tuple(_check_finite_num(arr[i], "%s[%d]" % (path, i), positive=positive) for i in range(3))


def tpms_unit_cell_genus(kind: str) -> dict:
    """Return crystallographic unit-cell genus and Euler constants for a TPMS.

    Distinguishes the *primitive* translational cell genus (3 for P, D, G; 9
    for Neovius) from the *conventional cubic* ``[0, P]^3`` quotient genus
    (3 for Schwarz P, 5 for Gyroid with 2 BCC cells/cube, 9 for Diamond with
    4 FCC cells/cube, 9 for Neovius).
    """
    from .errors import AttaError

    key = str(kind)
    if key not in TPMS_SPECS:
        raise AttaError("bad_arg", "unknown TPMS kind %r" % (kind,),
                        path="field", fix="use one of: " + ", ".join(sorted(TPMS_SPECS)))
    return dict(TPMS_SPECS[key])


def tpms_eval(kind: str, X, Y, Z, period, *, offset=0.0, phase=(0.0, 0.0, 0.0)):
    """Evaluate a nodal TPMS scalar field and its exact analytic spatial gradient.

    ``period`` is the cubic unit-cell period in metres; ``phase`` is a 3-tuple
    in *periods* (cells), so ``u = (2*pi/period)*X + 2*pi*phase[0]``.
    Returns ``(F - offset, dF/dx, dF/dy, dF/dz)`` with gradient in ``m^-1``.
    """
    from .errors import AttaError

    key = str(kind)
    if key not in TPMS_SPECS:
        raise AttaError("bad_arg", "unknown TPMS kind %r" % (kind,),
                        path="field", fix="use one of: " + ", ".join(sorted(TPMS_SPECS)))
    period = _check_finite_num(period, "period", positive=True)
    offset = _check_finite_num(offset, "offset")
    ph = _check_vec3(phase, "phase")
    k = 2.0 * math.pi / period
    u = k * X + (2.0 * math.pi * ph[0])
    v = k * Y + (2.0 * math.pi * ph[1])
    w = k * Z + (2.0 * math.pi * ph[2])
    sx, cx = np.sin(u), np.cos(u)
    sy, cy = np.sin(v), np.cos(v)
    sz, cz = np.sin(w), np.cos(w)

    if key == "gyroid":
        F = sx * cy + sy * cz + sz * cx
        gx = k * (cx * cy - sz * sx)
        gy = k * (-sx * sy + cy * cz)
        gz = k * (-sy * sz + cz * cx)
    elif key == "schwarz_p":
        F = cx + cy + cz
        gx = -k * sx
        gy = -k * sy
        gz = -k * sz
    elif key == "diamond":
        F = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
        gx = k * (cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz)
        gy = k * (sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz)
        gz = k * (sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz)
    else:  # neovius
        F = 3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz
        gx = -k * sx * (3.0 + 4.0 * cy * cz)
        gy = -k * sy * (3.0 + 4.0 * cx * cz)
        gz = -k * sz * (3.0 + 4.0 * cx * cy)
    return F - offset, gx, gy, gz


def _edge_root_count(kind: str, axis: int, phi_rad, offset: float) -> int | None:
    """Count transverse zeros of ``F(edge) - offset = 0`` over one ``2*pi`` period.

    Along any axis-aligned line, all four standard nodal TPMS reduce to a pure
    first-harmonic sinusoid ``A*sin(t) + B*cos(t) + C = 0``:
    * if ``|C| == sqrt(A^2 + B^2)`` (tangential grazing) or the corner itself
      lies on the zero level set (``|B + C| < 1e-7``), returns ``None``
      (non-transverse cut; no exact genus claim).
    * otherwise returns ``2`` when ``|C| < sqrt(A^2 + B^2)`` and ``0`` when
      ``|C| > sqrt(A^2 + B^2)``.
    """
    s0, c0 = math.sin(phi_rad[0]), math.cos(phi_rad[0])
    s1, c1 = math.sin(phi_rad[1]), math.cos(phi_rad[1])
    s2, c2 = math.sin(phi_rad[2]), math.cos(phi_rad[2])
    if kind == "gyroid":
        if axis == 0:
            A, B, C = c1, s2, s1 * c2 - offset
        elif axis == 1:
            A, B, C = c2, s0, s2 * c0 - offset
        else:
            A, B, C = c0, s1, s0 * c1 - offset
    elif kind == "schwarz_p":
        if axis == 0:
            A, B, C = 0.0, 1.0, c1 + c2 - offset
        elif axis == 1:
            A, B, C = 0.0, 1.0, c0 + c2 - offset
        else:
            A, B, C = 0.0, 1.0, c0 + c1 - offset
    elif kind == "diamond":
        if axis == 0:
            A, B, C = s1 * s2 + c1 * c2, s1 * c2 + c1 * s2, -offset
        elif axis == 1:
            A, B, C = s0 * s2 + c0 * c2, s0 * c2 + c0 * s2, -offset
        else:
            A, B, C = s0 * s1 + c0 * c1, s0 * c1 + c0 * s1, -offset
    else:  # neovius
        if axis == 0:
            A, B, C = 0.0, 3.0 + 4.0 * c1 * c2, 3.0 * (c1 + c2) - offset
        elif axis == 1:
            A, B, C = 0.0, 3.0 + 4.0 * c0 * c2, 3.0 * (c0 + c2) - offset
        else:
            A, B, C = 0.0, 3.0 + 4.0 * c0 * c1, 3.0 * (c0 + c1) - offset
    # Corner value at t = phi_rad[axis]
    t0 = phi_rad[axis]
    corner_val = A * math.sin(t0) + B * math.cos(t0) + C
    if abs(corner_val) < 1e-6:
        return None
    amp = math.hypot(A, B)
    if abs(abs(C) - amp) < 1e-6:
        return None
    return 2 if abs(C) < amp else 0


def tpms_block_genus(kind: str, k: int, l: int, m: int, *,
                     phase=None, offset=0.0, mode="sheet") -> int | None:
    """Analytic genus of a ``k x l x m`` unit-cell TPMS block clipped by a box.

    Theorem (compactly-supported Euler characteristic / Mayer-Vietoris)
    -------------------------------------------------------------------
    Let ``M`` be a smooth periodic minimal surface in ``T^3 = R^3/(P*Z)^3``
    with conventional cubic quotient genus ``g_T3`` (``chi(M_T3) = 2 - 2*g_T3``)
    and let ``B`` be a ``k x l x m``-period axis-aligned box whose faces and
    edges intersect ``M`` transversely and whose corners avoid ``M``.
    Because ``M`` meets every face 2-torus ``T_ij`` in closed loops
    (``chi(S^1) = 0``), all quadratic face-area terms vanish identically,
    leaving only the bulk term and the edge intersection counts
    ``(n_x, n_y, n_z)`` per period:

        chi(M_B) = k*l*m * chi(M_T3) + n_x*k + n_y*l + n_z*m.

    Thickening ``M_B`` into a connected sheet solid ``S = M_B x [-t/2, t/2]``
    gives ``chi(dS) = 2*chi(S) = 2*chi(M_B)`` (Lefschetz duality), hence

        g(dS) = 1 - chi(M_B) = (2*g_T3 - 2)*k*l*m - n_x*k - n_y*l - n_z*m + 1.

    Canonical sheet formulas (lower-corner phase = ``TPMS_SPECS[kind]``):
    * ``schwarz_p`` (``g_T3 = 3``): ``4*k*l*m + 1``
    * ``gyroid``    (``g_T3 = 5``): ``8*k*l*m - 2*k - 2*l + 1``
    * ``diamond``   (``g_T3 = 9``): ``16*k*l*m - 2*k - 2*l - 2*m + 1``
    * ``neovius``   (``g_T3 = 9``): ``16*k*l*m - 2*k - 2*l - 2*m + 1``

    For ``mode="network"`` on ``schwarz_p``, the single labyrinth solid retracts
    to a ``k x l x m`` simple-cubic grid graph of cycle rank
    ``2*k*l*m - k*l - k*m - l*m + 1``.

    Returns ``None`` when ``offset`` exceeds the connected-sheet pinch regime
    or when the chosen ``phase`` places a box corner/edge tangency on ``M``.
    """
    from .errors import AttaError

    key = str(kind)
    if key not in TPMS_SPECS:
        raise AttaError("bad_arg", "unknown TPMS kind %r" % (kind,),
                        path="field", fix="use one of: " + ", ".join(sorted(TPMS_SPECS)))
    dims = []
    for name, val in (("k", k), ("l", l), ("m", m)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, np.integer)):
            raise AttaError("bad_arg", "%s must be a positive integer" % name,
                            path=name, fix="pass a positive integer cell count")
        iv = int(val)
        if iv < 1:
            raise AttaError("bad_arg", "%s must be >= 1" % name,
                            path=name, fix="pass a positive integer cell count")
        dims.append(iv)
    kx, ky, kz = dims
    offset = _check_finite_num(offset, "offset")
    spec = TPMS_SPECS[key]

    if mode == "network":
        if key != "schwarz_p":
            return None
        if abs(offset) > spec["safe_offset_max"]:
            return None
        return 2 * kx * ky * kz - kx * ky - kx * kz - ky * kz + 1
    if mode != "sheet":
        raise AttaError("bad_arg", "mode must be 'sheet' or 'network'",
                        path="mode", fix="pass mode='sheet' or mode='network'")

    if abs(offset) > spec["safe_offset_max"]:
        return None
    ph_cells = spec["canonical_phase_cells"] if phase is None else _check_vec3(phase, "phase")
    phi_rad = tuple(2.0 * math.pi * c for c in ph_cells)
    counts = [_edge_root_count(key, ax, phi_rad, offset) for ax in range(3)]
    if any(c is None for c in counts):
        return None
    nx, ny, nz = counts
    bulk = 2 * spec["conventional_cubic_genus"] - 2  # == -euler_per_cube
    return bulk * kx * ky * kz - nx * kx - ny * ky - nz * kz + 1


def tpms_sheet_distance(kind: str, X, Y, Z, period, thickness, *,
                        offset=0.0, phase=(0.0, 0.0, 0.0),
                        band_margin=None, newton_steps=3):
    """Metric signed distance (metres) to a thickened TPMS sheet of ``thickness`` m.

    Normalising ``|F - offset|`` by a global median ``|grad F|`` leaves a
    30-45 % thickness variation between necks and saddles (and volume drift).
    Instead, this computes the first-order Sampson distance ``|F| / |grad F|``
    with the *exact analytic* gradient everywhere, then refines every sample in
    the narrow band around the sheet with ``newton_steps`` damped Newton
    closest-point projections onto ``F(p) - offset = 0``:

        p_{j+1} = p_j - clamp(G(p_j) * grad G(p_j) / |grad G(p_j)|^2, 0.22/k).

    Inside the narrow band (which covers every active cell corner of
    ``marching_tets``), ``||x - p||_2 - 0.5*thickness`` is the true Euclidean
    signed distance in metres to the sheet wall, independent of field scale.
    """
    from .errors import AttaError

    period = _check_finite_num(period, "period", positive=True)
    thickness = _check_finite_num(thickness, "thickness", positive=True)
    offset = _check_finite_num(offset, "offset")
    if thickness >= 0.5 * period:
        raise AttaError("bad_arg", "TPMS sheet thickness must be < 0.5 * period",
                        path="thickness", fix="use thickness <= 0.16 * period")
    if isinstance(newton_steps, (bool, np.bool_)) or int(newton_steps) != newton_steps or int(newton_steps) < 0:
        raise AttaError("bad_arg", "newton_steps must be a non-negative integer",
                        path="newton_steps", fix="pass newton_steps=3")
    newton_steps = int(newton_steps)
    ph = _check_vec3(phase, "phase")

    k = 2.0 * math.pi / period
    F, gx, gy, gz = tpms_eval(kind, X, Y, Z, period, offset=offset, phase=ph)
    g_floor = 0.05 * k
    gmag = np.sqrt(gx * gx + gy * gy + gz * gz + g_floor * g_floor)
    d0 = np.abs(F) / gmag
    half_t = 0.5 * thickness
    if band_margin is None:
        margin = max(half_t, 0.15 * period)
    else:
        margin = _check_finite_num(band_margin, "band_margin", positive=True)
    if newton_steps == 0:
        return d0 - half_t

    # Narrow-band mask: generous enough to include every corner of every cell
    # that can cross d == half_t, even where |grad F| varies by 1.8x.
    band_limit = 1.6 * (half_t + margin)
    mask = d0 <= band_limit
    d_out = np.maximum(d0, half_t + 0.5 * margin)
    if not bool(mask.any()):
        return d_out - half_t

    bx = X[mask].copy()
    by = Y[mask].copy()
    bz = Z[mask].copy()
    px, py, pz = bx.copy(), by.copy(), bz.copy()
    cap = 0.22 / k
    g2_floor = (0.20 * k) ** 2
    for _ in range(newton_steps):
        G, jx, jy, jz = tpms_eval(kind, px, py, pz, period, offset=offset, phase=ph)
        g2 = jx * jx + jy * jy + jz * jz
        inv = G / np.maximum(g2, g2_floor)
        dx, dy, dz = inv * jx, inv * jy, inv * jz
        step = np.sqrt(dx * dx + dy * dy + dz * dz)
        scale = np.where(step > cap, cap / np.maximum(step, 1e-30), 1.0)
        px -= dx * scale
        py -= dy * scale
        pz -= dz * scale
    dist_band = np.sqrt((bx - px) ** 2 + (by - py) ** 2 + (bz - pz) ** 2)
    d_out[mask] = dist_band
    return d_out - half_t


def tpms_sheet_volume_estimate(kind: str, size, period, thickness, *, offset=0.0) -> float:
    """Analytic Weyl (1939) tubular volume estimate (m^3) for a TPMS sheet block.

    For a minimal surface (mean curvature ``H = 0``), Weyl's tube formula gives
    ``V(t) = A * t + (pi * chi / 6) * t^3`` per cubic unit cell, scaled by the
    number of cubic cells ``prod(size) / period^3``.
    """
    from .errors import AttaError

    spec = tpms_unit_cell_genus(kind)
    sz = _check_vec3(size, "size", positive=True)
    period = _check_finite_num(period, "period", positive=True)
    thickness = _check_finite_num(thickness, "thickness", positive=True)
    offset = _check_finite_num(offset, "offset")
    n_cells = float(sz[0] * sz[1] * sz[2]) / (period ** 3)
    area_factor = spec["area_coeff_cube"] * max(0.0, 1.0 - spec["offset_area_quad"] * offset * offset)
    v_cell = area_factor * (period ** 2) * thickness + (math.pi * spec["euler_per_cube"] / 6.0) * (thickness ** 3)
    return max(0.0, n_cells * v_cell)


def feature_spacing(size, min_feature, *, min_voxels=3.0, requested_spacing=None,
                    max_samples=2_500_000) -> tuple[float, tuple[int, int, int]]:
    """Derive voxel spacing from ``min_feature`` and validate against ``max_samples``.

    Always resolves ``min_feature`` with at least ``min_voxels`` samples (or
    finer if ``requested_spacing`` is smaller) and fails closed with
    ``AttaError(code="sdf_too_dense")`` when the resulting padded grid exceeds
    ``max_samples`` rather than silently under-sampling thin walls into pinholes.
    """
    from .errors import AttaError
    from .sdf import grid_shape

    sz = np.array(_check_vec3(size, "size", positive=True), dtype=np.float64)
    min_feature = _check_finite_num(min_feature, "min_feature", positive=True)
    min_voxels = _check_finite_num(min_voxels, "min_voxels", positive=True)
    if isinstance(max_samples, (bool, np.bool_)) or int(max_samples) != max_samples or int(max_samples) < 343:
        raise AttaError("bad_arg", "max_samples must be an integer >= 343",
                        path="max_samples", fix="pass an integer budget >= 343")
    sp = min_feature / min_voxels
    if requested_spacing is not None:
        req = _check_finite_num(requested_spacing, "spacing", positive=True)
        sp = min(sp, req)
    half = 0.5 * sz
    shape = grid_shape(-half, half, sp)
    samples = math.prod(shape)
    if samples > int(max_samples):
        raise AttaError(
            "sdf_too_dense",
            "grid needs %d samples at spacing %.4f mm (budget %d)" % (
                samples, sp * 1e3, int(max_samples)),
            path="resolution",
            fix="use a thicker wall, a larger period, a smaller domain or higher max_samples",
            details={"samples": samples, "max_samples": int(max_samples),
                     "spacing_mm": sp * 1e3, "shape": list(shape)},
        )
    return float(sp), shape


def prune_boundary_slivers(phi, clip_sdf, spacing, *, max_fraction=0.05):
    """Suppress detached boundary-grazing crumbs chipped off a clipped TPMS field.

    Clipping a connected periodic sheet with a non-commensurate box or sphere
    can graze lobes whose bridge to the main body lies outside the clip region,
    leaving tiny loose slivers on the cut surface.  Using the exact 14-neighbour
    Kuhn-tetrahedron adjacency of :func:`atta.sdf.marching_tets`, any secondary
    component of ``phi < 0`` that both touches the clip boundary
    (``clip_sdf > -2.5 * spacing``) and contains fewer than ``max_fraction`` of
    the largest component's voxels is pushed strictly positive before meshing.

    Large disconnected bodies or interior droplets (e.g. from excessive field
    offset) are never touched.
    """
    from scipy import ndimage

    inside = phi < 0.0
    if not bool(inside.any()):
        return phi, 0
    labels, n_comp = ndimage.label(inside, structure=_KUHN_14_STRUCT)
    if n_comp <= 1:
        return phi, 0
    counts = np.bincount(labels.ravel())[1:]
    main_idx = int(np.argmax(counts))
    main_count = int(counts[main_idx])
    limit = max(1, int(math.ceil(float(max_fraction) * main_count)))
    near_clip = clip_sdf > -2.5 * float(spacing)
    drop_ids = []
    for cid in range(1, n_comp + 1):
        if cid == main_idx + 1:
            continue
        if int(counts[cid - 1]) < limit and bool(np.any(near_clip[labels == cid])):
            drop_ids.append(cid)
    if not drop_ids:
        return phi, 0
    out = np.array(phi, dtype=np.float64, copy=True)
    drop_mask = np.isin(labels, drop_ids)
    out[drop_mask] = np.maximum(np.abs(out[drop_mask]), 1e-3 * float(spacing))
    return out, len(drop_ids)


def cap_boundary(phi, spacing):
    """Ensure the outer 1-voxel shell of ``phi`` is strictly positive so MT closes."""
    sp = _check_finite_num(spacing, "spacing", positive=True)
    floor = 1e-3 * sp
    out = np.array(phi, dtype=np.float64, copy=True)
    for sl in (
        (0, slice(None), slice(None)), (-1, slice(None), slice(None)),
        (slice(None), 0, slice(None)), (slice(None), -1, slice(None)),
        (slice(None), slice(None), 0), (slice(None), slice(None), -1),
    ):
        out[sl] = np.maximum(out[sl], floor)
    return out


# ---------------------------------------------------------------- grid

def grid(lo, hi, spacing, *, max_samples=2_500_000):
    """Sample a padded grid after checking its size, without changing spacing."""
    from .sdf import grid_shape
    from .errors import AttaError

    shape = grid_shape(lo, hi, spacing)
    if math.prod(shape) > int(max_samples):
        raise AttaError("sdf_too_dense", "grid exceeds %d samples" % max_samples,
                        path="spacing", fix="increase spacing before sampling",
                        details={"samples": math.prod(shape), "max_samples": int(max_samples)})
    lo = np.asarray(lo, dtype=np.float64)
    spacing = float(spacing)
    org = lo - 2.5 * spacing
    axes = [org[i] + np.arange(shape[i]) * spacing for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return X, Y, Z, org


def sample(fn, lo, hi, spacing):
    """Sample a field function on a grid; returns ``(phi, org, spacing)``."""
    X, Y, Z, org = grid(lo, hi, spacing)
    return fn(X, Y, Z), org, float(spacing)


def extract(phi, org, spacing, *, cap_open_boundary=False, **budgets):
    """Bounded marching-tets isosurface of ``phi < 0``; returns a Mesh.

    Pass ``cap_open_boundary=True`` to close any negative samples that reach
    the outer border of the sampled box.
    """
    from .sdf import marching_tets
    if cap_open_boundary:
        phi = cap_boundary(phi, spacing)
    return marching_tets(phi, org, spacing, **budgets)
