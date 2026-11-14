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


def extract(phi, org, spacing, **budgets):
    """Bounded marching-tets isosurface of ``phi < 0``; returns a Mesh."""
    from .sdf import marching_tets
    return marching_tets(phi, org, spacing, **budgets)
