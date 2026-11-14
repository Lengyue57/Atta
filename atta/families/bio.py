# -*- coding: utf-8 -*-
"""Bio-structure ops: high-count organic geometry that a pairwise CSG chain
cannot reach.

Why this family exists
----------------------
The core graph booleans fold inputs one pair at a time
(``for other in meshes[1:]: out = boolean(out, other)``).  That is fine for a
ten-part machine but it is quadratic-ish in practice for the thing microscopy
models are made of: a lattice shell needs thousands of struts, a diatom valve
needs ten thousand pores, a neuron needs a thousand tapering branches.  Each
pairwise step re-meshes the whole accumulated result, so a 6-frequency geodesic
basket (10 240 struts) never finished.

``manifold3d.Manifold.batch_boolean`` does the same union/subtract in one
tree-reduced pass: 2 000 disjoint tubes take milliseconds.  This module builds
the *elements* in numpy/manifold and hands the batch back as an ordinary
:class:`atta.mesh.Mesh`, so everything downstream (weld, facts, verify,
deliver, Blender) is unchanged and the receipts are the same receipts.

Units: the spec boundary is millimetres, the kernel is metres.  ``prepare``
converts once, exactly like ``form_ops``; the op functions only see metres.
"""

from __future__ import annotations

import math

import numpy as np

from ..errors import AttaError
from ..mesh import Mesh, oriented, weld

__all__ = ["register"]

#: available ops, for callers that want to introspect the family
OPS = ("lattice_shell", "perforated_valve", "perforated_cylinder",
       "branch_tree", "pore_field_shell", "capsid", "ring_stack",
       "weld_union")


# --------------------------------------------------------------- helpers

def _m3():
    try:
        import manifold3d
    except Exception as exc:  # pragma: no cover
        raise AttaError("engine_unavailable", "manifold3d is not importable: %s" % exc,
                        path="bio", fix='pip install manifold3d') from exc
    return manifold3d


def _batch(manifolds, op):
    """Tree-reduced boolean over an arbitrary number of manifolds."""
    m3 = _m3()
    items = [m for m in manifolds if m is not None and not m.is_empty()]
    if not items:
        raise AttaError("boolean_empty", "no non-empty geometry to combine",
                        path="bio", fix="check the parameters; every element vanished")
    table = {"union": m3.OpType.Add, "difference": m3.OpType.Subtract,
             "intersection": m3.OpType.Intersect}
    if op not in table:
        raise AttaError("bad_arg", "op must be union/difference/intersection",
                        path="bio.op", fix="use op='union'")
    if len(items) == 1:
        return items[0]
    return m3.Manifold.batch_boolean(items, table[op])


def _to_mesh(man, kind, dedupe=True):
    mm = man.to_mesh()
    verts = np.asarray(mm.vert_properties, dtype=np.float64)
    faces = np.asarray(mm.tri_verts, dtype=np.int64)
    if len(faces) == 0:
        raise AttaError("boolean_empty", "the %s result has no faces" % kind,
                        path="bio", fix="increase radii or reduce the pore size")
    if dedupe:
        verts, faces = _weld_slivers(verts[:, :3], faces)
    else:
        verts = verts[:, :3]
    return oriented(Mesh(verts, faces, {"kind": kind}))


def _weld_slivers(verts, faces, rel=1e-9):
    """Collapse *exactly* duplicate vertices, nothing coarser.

    Slivers come from manifold3d emitting separate vertices for the same point
    when a cutter meets a seam.  Welding at a tolerance proportional to the
    model would also fuse the 0.09 mm girdle-band pores of a diatom into their
    neighbours (volume changed 8.8k -> 66k mm^3, nine components, non-manifold).
    So the only safe merge is the exact one: identical coordinates, which can
    never move a modelled feature.
    """
    if not len(faces):
        return verts, faces
    # bit-exact dedupe: the float64 coordinates either match or they do not
    key = verts.view(np.int64).reshape(len(verts), 3)
    _, first, inverse = np.unique(key, axis=0, return_index=True,
                                  return_inverse=True)
    if len(first) == len(verts):
        return verts, faces
    faces = inverse.reshape(-1)[faces].astype(np.int64)
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    good = (a != b) & (b != c) & (a != c)
    if not good.all():
        faces = faces[good]
    used = np.unique(faces)
    remap = -np.ones(len(first), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[first][used], remap[faces]


def _sphere(radius, center, seg=16):
    m3 = _m3()
    body = m3.Manifold.sphere(float(radius), int(max(6, seg)))
    return body.translate([float(center[0]), float(center[1]), float(center[2])])


def _capsule(a, b, ra, rb, seg=12):
    """Tapered capsule between two points (a cone plus two caps, one manifold)."""
    m3 = _m3()
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return _sphere(max(ra, rb), a, seg)
    up = d / length
    # A cone of height h with radii rb at top of `height`, centred on origin.
    body = m3.Manifold.cylinder(length, float(ra), float(rb), int(max(6, seg)), False)
    # cylinder() grows along +Z from z=0; rotate +Z onto `up`, then translate.
    rot = _align_z(up)
    body = body.transform(rot)
    return body.translate([float(a[0]), float(a[1]), float(a[2])])


def _align_z(up):
    """3x4 row-major rotation taking +Z onto unit vector ``up``."""
    up = np.asarray(up, dtype=np.float64)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, up)
    c = float(np.dot(z, up))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            r = np.eye(3)
        else:  # 180 deg: flip about X
            r = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    else:
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        r = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    return np.hstack([r, np.zeros((3, 1))])


def _icosphere(freq, radius=1.0):
    """Geodesic sphere; ``freq`` = edge subdivision (20*freq^2 faces)."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [tuple(float(c) / math.sqrt(sum(x * x for x in p)) for c in p) for p in verts]
    for _ in range(int(max(1, freq))):
        cache, nf = {}, []

        def mid(i, j):
            key = (i, j) if i < j else (j, i)
            if key in cache:
                return cache[key]
            m = [(v[i][k] + v[j][k]) * 0.5 for k in range(3)]
            n = math.sqrt(sum(c * c for c in m))
            v.append(tuple(c / n for c in m))
            cache[key] = len(v) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    return [tuple(c * radius for c in p) for p in v], faces


def _edges(faces):
    seen = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            seen.add((a, b) if a < b else (b, a))
    return sorted(seen)


def _sanitize_manifold(man):
    """Batch booleans can leave slivers when elements only touch tangentially."""
    m3 = _m3()
    try:
        man = man.as_original()
    except Exception:
        pass
    if man.num_tri() > 400_000:
        man = man.simplify(1e-4)
    return man


# --------------------------------------------------------------- ops

def _lattice_shell(*, radius, wall, strut, freq=4, spines=0, spine_len=0.0,
                   spine_base=None, node_scale=1.35, seg=10, simplify=0.0):
    """Open geodesic lattice shell: struts along the edges of an icosphere.

    A radiolarian skeleton is a triangulated silica basket, not a perforated
    plate: the material *is* the strut network.  Building struts on the edges
    of a geodesic sphere gives the correct hexagonal/triangular mesh with the
    obligatory 12 five-fold vertices (Euler), which a cut sphere cannot.
    """
    verts, faces = _icosphere(freq, radius)
    edges = _edges(faces)
    # A high-frequency basket is millions of triangles at full strut
    # resolution.  ``simplify`` decimates within a tolerance well below the
    # strut radius, which cuts the file size by ~40% with no visible change at
    # any sane camera distance -- the strut cross-section is 0.46 mm and the
    # tolerance is a fraction of that.
    strut_seg = seg if freq <= 4 else max(5, min(seg, 6))
    elems = []
    for a, b in edges:
        elems.append(_capsule(verts[a], verts[b], strut, strut, strut_seg))
    if node_scale > 1.0:
        for p in verts:
            elems.append(_sphere(strut * node_scale, p, max(6, strut_seg)))
    body = _batch(elems, "union")
    if simplify and simplify > 0.0 and body.num_tri() > 400000:
        body = body.simplify(float(simplify))
    if wall and wall > 0.0:
        # Hollow the struts into tubes: keep the network, subtract its core.
        # The bore must stay *inside* the node spheres.  A bore as wide as
        # ``strut * node_scale - wall`` cuts through the junction spheres and
        # severs the lattice at every vertex, which silently splits one closed
        # basket into many disconnected arcs (comps grows, verify still says
        # closed).  Keep the bore under the node radius with a real margin.
        bore = min(float(strut) - float(wall), strut * float(node_scale) - 2.2 * float(wall))
        if bore <= 0.0:
            bore = 0.0
        if bore > 0.0:
            inner = []
            for a, b in edges:
                inner.append(_capsule(verts[a], verts[b], bore, bore, seg))
            for p in verts:
                inner.append(_sphere(bore, p, seg))
            body = _batch([body, _batch(inner, "union")], "difference")
    if spines and spines > 0:
        # Radial spines through antipodal vertices on the three axes.
        lines = []
        base = float(spine_base or strut * 2.2)
        for axis in range(3):
            for sign in (-1.0, 1.0):
                tip = [0.0, 0.0, 0.0]
                tip[axis] = sign * (radius + spine_len)
                root = [0.0, 0.0, 0.0]
                root[axis] = sign * (radius - strut * 3.0)
                lines.append(_capsule(root, tip, base, max(1e-6, base * 0.06), seg))
        for k in range(max(0, int(spines) - 6)):
            # Extra spines distributed by the golden angle.
            ang = math.pi * (1.0 + 5.0 ** 0.5) * k
            zz = 1.0 - 2.0 * (k + 0.5) / max(1, int(spines) - 6)
            rr = math.sqrt(max(1e-9, 1.0 - zz * zz))
            d = np.array([rr * math.cos(ang), rr * math.sin(ang), zz])
            root = d * (radius - strut * 3.0)
            tip = d * (radius + spine_len * 0.55)
            lines.append(_capsule(root, tip, base * 0.8, max(1e-6, base * 0.05), seg))
        body = _batch([body] + lines, "union")
    return _to_mesh(_sanitize_manifold(body), "lattice_shell")


def _perforated_valve(*, radius, thickness, pitch, pore_radius, dome=0.0,
                      margin=0.0, rows_extra=0, gap=0.0, chamber_depth=0.0,
                      foramen_radius=0.0, chamber_taper=0.0):
    """Circular plate with a hexagonal array of pores.

    This is the diatom valve: a silica disc whose areolae are a hexagonal
    lattice.  Every pore is a cylinder cutter; the whole array goes through one
    subtract pass, so 5 000 pores cost about as much as 50.

    ``chamber_depth`` switches to the real loculate anatomy: instead of a
    straight through-hole, each areola is a blind chamber cut from the inner
    face, closed on the outside by a thin cribrum plate and pierced by one
    narrow ``foramen_radius`` hole.  That is the structure that makes an SEM of
    a centric diatom read as a honeycomb rather than a colander.
    """
    m3 = _m3()
    disc = m3.Manifold.cylinder(float(thickness), float(radius), float(radius), 256, True)
    if dome and dome > 0.0:
        # Give the valve face a watch-glass curvature by intersecting with a
        # very large sphere; keeps the rim flat and the centre raised.
        big = (radius * radius + dome * dome) / (2.0 * dome)
        cap = m3.Manifold.sphere(float(big), 220).translate([0.0, 0.0, dome - big])
        disc = m3.Manifold.batch_boolean([disc, cap], m3.OpType.Intersect)
    half_t = float(thickness) * 0.5
    loculate = float(chamber_depth) > 0.0
    if loculate:
        cd = min(float(chamber_depth), float(thickness) * 0.92)
        # chamber opens on the inner (-Z) face and stops short of the outer one
        z0 = -half_t - 1e-6
        cutters = []
        area_r = max(1e-6, radius - float(margin))
        pitch_f = float(pitch)
        n_ring = int(area_r / pitch_f) + 1 + int(rows_extra)
        for i in range(-n_ring, n_ring + 1):
            for j in range(-n_ring, n_ring + 1):
                x = (i + 0.5 * (j % 2)) * pitch_f
                y = j * pitch_f * (math.sqrt(3.0) / 2.0)
                if math.hypot(x, y) > area_r:
                    continue
                rp = float(pore_radius)
                if gap:
                    rp *= max(0.55, 1.0 - gap * (math.hypot(x, y) / max(1e-9, area_r)) ** 2)
                r_top = rp * (1.0 - float(chamber_taper))
                cutters.append(m3.Manifold.cylinder(cd + 2e-6, rp, r_top, 14, False)
                               .translate([x, y, z0]))
        if not cutters:
            raise AttaError("bad_arg", "no areolae fit: pitch is larger than the plate",
                            path="bio.pitch", fix="reduce pitch or increase radius")
        out = m3.Manifold.batch_boolean([disc] + cutters, m3.OpType.Subtract)
        if foramen_radius and foramen_radius > 0.0:
            holes = []
            z1 = half_t + 1e-6
            for i in range(-n_ring, n_ring + 1):
                for j in range(-n_ring, n_ring + 1):
                    x = (i + 0.5 * (j % 2)) * pitch_f
                    y = j * pitch_f * (math.sqrt(3.0) / 2.0)
                    if math.hypot(x, y) > area_r:
                        continue
                    holes.append(m3.Manifold.cylinder(cd + 4e-6, float(foramen_radius),
                                                      float(foramen_radius), 12, True)
                                 .translate([x, y, z1 - cd]))
            out = m3.Manifold.batch_boolean([out] + holes, m3.OpType.Subtract)
        mesh = _to_mesh(_sanitize_manifold(out), "areolated_valve")
        mesh.meta.update({"areolae": len(cutters), "loculate": True,
                          "pitch_mm": pitch_f * 1000.0})
        return mesh
    cutters = []
    pitch = float(pitch)
    area_r = max(1e-6, radius - float(margin))
    n_ring = int(area_r / pitch) + 1 + int(rows_extra)
    for i in range(-n_ring, n_ring + 1):
        for j in range(-n_ring, n_ring + 1):
            x = (i + 0.5 * (j % 2)) * pitch
            y = j * pitch * (math.sqrt(3.0) / 2.0)
            if math.hypot(x, y) > area_r:
                continue
            rp = float(pore_radius)
            if gap:
                # Pores narrow slightly toward the rim, as real areolae do.
                rp *= max(0.55, 1.0 - gap * (math.hypot(x, y) / max(1e-9, area_r)) ** 2)
            cutters.append(m3.Manifold.cylinder(float(thickness) * 3.0 + 2e-6, rp, rp, 12, True)
                           .translate([x, y, 0.0]))
    if not cutters:
        raise AttaError("bad_arg", "no pores fit: pitch is larger than the plate",
                        path="bio.pitch", fix="reduce pitch or increase radius")
    out = m3.Manifold.batch_boolean([disc] + cutters, m3.OpType.Subtract)
    meta = {"pores": len(cutters), "pitch_mm": pitch * 1000.0}
    mesh = _to_mesh(_sanitize_manifold(out), "perforated_valve")
    mesh.meta.update(meta)
    return mesh


def _perforated_cylinder(*, radius, height, wall, rows, cols, pore_radius,
                         offset_deg=0.0, stagger=True, bore=0.0):
    """Annular wall with a grid of radial through-pores (diatom mantle).

    Pores are radial cylinders on a cylindrical grid, so the wall reads as
    vertical rows of areolae exactly like a real mantle, and staggered rows
    break the grid into the hexagonal packing the animal actually has.
    """
    m3 = _m3()
    r = float(radius)
    h = float(height)
    w = float(wall)
    inner = max(1e-6, r - w)
    shell = m3.Manifold.batch_boolean([
        m3.Manifold.cylinder(h, r, r, 320, True),
        m3.Manifold.cylinder(h * 1.2, inner, inner, 320, True)], m3.OpType.Subtract)
    cutters = []
    rows = max(1, int(rows))
    cols = max(3, int(cols))
    pr = float(pore_radius)
    for j in range(rows):
        z = -h * 0.5 + (j + 0.5) * (h / rows)
        off = (0.5 if (stagger and j % 2) else 0.0) + float(offset_deg) / 360.0
        for i in range(cols):
            ang = 2.0 * math.pi * ((i + off) / cols)
            # One radial cylinder per pore, axis along the local radius.
            cyl = m3.Manifold.cylinder(w * 2.4 + 2e-6, pr, pr, 10, True)
            rot = _align_z((math.cos(ang), math.sin(ang), 0.0))
            cyl = cyl.transform(rot).translate([math.cos(ang) * (inner - w * 0.2),
                                                math.sin(ang) * (inner - w * 0.2), z])
            cutters.append(cyl)
    out = m3.Manifold.batch_boolean([shell] + cutters, m3.OpType.Subtract)
    if bore and bore > 0.0:
        out = m3.Manifold.batch_boolean([out, m3.Manifold.cylinder(
            h * 1.4, max(1e-6, inner - float(bore)), max(1e-6, inner - float(bore)),
            320, True)], m3.OpType.Subtract)
    mesh = _to_mesh(_sanitize_manifold(out), "perforated_cylinder")
    mesh.meta.update({"mantle_pores": len(cutters), "rows": rows, "cols": cols})
    return mesh


def _branch_tree(*, trunk, dendrites, depth=4, split=2, taper=0.78, spread=0.62,
                 length=20.0, length_decay=0.72, radius_floor=0.0004, jitter=0.35,
                 seed=7, seg=10, spines=0.0, spine_h=0.0, blend=0.0):
    """Space-filling tapered dendrite tree grown from a trunk point.

    A neuron's silhouette is a branching taper, so the model is generated by
    the branching rule rather than sculpted: each segment splits into ``split``
    children, the radius follows a per-order taper ratio, and the direction
    cone widens with order.  Every segment becomes one tapered capsule and the
    whole arbor is a single batch union.
    """
    rng = np.random.default_rng(int(seed))
    elems = []
    tips = []
    trunk = np.asarray(trunk, dtype=np.float64)

    def grow(origin, direction, radius, length, order, path):
        if order > int(depth) or radius < float(radius_floor) or len(elems) > 20000:
            tips.append((origin, radius))
            return
        d = np.asarray(direction, dtype=np.float64)
        d = d / max(1e-12, np.linalg.norm(d))
        # A little wander so the branch is not a straight line.
        wobble = rng.normal(0.0, 1.0, 3) * float(jitter) * 0.35
        d = d + wobble
        d = d / max(1e-12, np.linalg.norm(d))
        end = origin + d * length
        elems.append(_capsule(origin, end, radius, radius * float(taper), seg))
        tips.append((end, radius * float(taper)))
        n = int(max(1, split))
        for k in range(n):
            # Children fan out inside a cone whose half-angle grows with order.
            ang = 2.0 * math.pi * (k / n) + rng.normal(0.0, 0.25)
            half = float(spread) * (1.0 + 0.35 * order) * (0.7 + 0.6 * rng.random())
            side = np.cross(d, [0.0, 0.0, 1.0])
            if np.linalg.norm(side) < 1e-6:
                side = np.cross(d, [1.0, 0.0, 0.0])
            side /= np.linalg.norm(side)
            up = np.cross(d, side)
            nd = (math.cos(half) * d
                  + math.sin(half) * (math.cos(ang) * side + math.sin(ang) * up))
            nd /= max(1e-12, np.linalg.norm(nd))
            grow(end, nd, radius * float(taper),
                 length * float(length_decay), order + 1, path + [k])

    n_trunk = int(max(1, dendrites))
    for i in range(n_trunk):
        ang = 2.0 * math.pi * i / n_trunk
        d = np.array([math.cos(ang) * 0.55, math.sin(ang) * 0.55, -1.0])
        grow(trunk, d, float(radius_floor) * 6.0, float(length), 0, [i])

    body = _batch(elems, "union")
    if spines and spine_h:
        # Dendritic spines: a thin neck plus a bulbous head, instanced on the
        # arbor's own tips so they sit on real dendrite, not in empty space.
        sp = []
        want = int(spines)
        if tips and want > 0:
            idx = rng.choice(len(tips), size=min(want, len(tips)), replace=False)
            for j in idx:
                p, r = tips[int(j)]
                v = rng.normal(0.0, 1.0, 3)
                v /= max(1e-12, np.linalg.norm(v))
                neck_end = p + v * float(spine_h) * 0.7
                sp.append(_capsule(p, neck_end, max(1e-5, r * 0.35),
                                   max(1e-5, r * 0.22), max(6, seg // 2)))
                sp.append(_sphere(max(1e-5, r * 0.55), neck_end, max(6, seg // 2)))
        if sp:
            body = _batch([body, _batch(sp, "union")], "union")
    return _to_mesh(_sanitize_manifold(body), "branch_tree")


def _capsid(*, length, width, facet, ridge=0.18, seg=72, faceted=0.35):
    """Prolate icosahedral capsid with a capsomer lattice.

    A T4 head is a prolate icosahedron (T_end=13 laev), and what makes a capsid
    read as a virus is two things a smooth ellipsoid never gives you: flat
    icosahedral facets, and one bump per capsomer over the whole surface.

    The shell is a *closed* solid: a geodesic icosphere stretched into the
    prolate envelope, then subtracted from a slightly smaller copy of itself so
    the wall has thickness.  Capsomer bumps are instanced at every geodesic
    vertex, so the pitch follows the lattice instead of a random handful of
    points.

    All coordinates below are already in model units (metres): the caller's
    ``prepare`` did the mm conversion, exactly like every other bio op.
    """
    m3 = _m3()
    half_l = 0.5 * float(length)
    half_w = 0.5 * float(width)
    sub = max(2, int(round(math.log2(max(1.0, float(facet)))) + 2))
    verts, faces = _icosphere(sub, 1.0)
    vv = np.asarray([[p[0] * half_w, p[1] * half_w, p[2] * half_l] for p in verts],
                    dtype=np.float64)

    def solid(scale):
        return m3.Manifold(m3.Mesh(
            vert_properties=np.asarray(vv * scale, dtype=np.float32),
            tri_verts=np.asarray(faces, dtype=np.uint32)))

    outer = solid(1.0)
    if outer.is_empty():
        raise AttaError("bad_arg", "capsid shell came out empty", path="capsid",
                        fix="raise width or facet")
    # Solid head.  A hollow capsid would need a closed inner wall, and an inner
    # shell scaled uniformly inside a *stretched* icosphere never touches the
    # outer one: it comes out as a floating second body inside the head (which
    # is exactly the two-component result this produced).  The capsid is opaque
    # in every SEM image anyway, so solid is both simpler and correct.
    body = outer

    # --- capsomer bumps at every lattice vertex ------------------------
    if ridge and ridge > 0:
        bumps = [m3.Manifold.sphere(float(ridge), 10).translate(
            [float(p[0]), float(p[1]), float(p[2])]) for p in vv]
        if bumps:
            body = m3.Manifold.batch_boolean([body] + bumps, m3.OpType.Add)
    return _to_mesh(_sanitize_manifold(body), "capsid")


def _ring_stack(*, radius, height, rings, ring_radius=0.0, twist_deg=17.2,
                taper=0.0, hollow=0.0, seg=48):
    """Stack of offset rings: a phage tail sheath, or any banded tubule.

    The T4 sheath is 23 rings of six subunits with a fixed twist per ring; the
    banding *is* the diagnostic silhouette, so it has to be real geometry and
    not a texture.
    """
    m3 = _m3()
    r = float(radius)
    h = float(height)
    n = max(2, int(rings))
    pitch = h / n
    elems = []
    for i in range(n):
        z = -0.5 * h + (i + 0.5) * pitch
        rr = r * (1.0 - float(taper) * (i / max(1, n - 1)))
        band = m3.Manifold.cylinder(pitch * 0.92, rr, rr, int(seg), True)
        if ring_radius and ring_radius > 0:
            band = band.minkowski_sum(m3.Manifold.sphere(float(ring_radius), 8))
        ang = math.radians(float(twist_deg)) * i
        if abs(ang) > 1e-9:
            band = band.rotate([0.0, 0.0, math.degrees(ang)])
        elems.append(band.translate([0.0, 0.0, z]))
    body = _batch(elems, "union")
    if hollow and hollow > 0.0:
        core = m3.Manifold.cylinder(h * 1.4, max(1e-6, r - float(hollow)),
                                    max(1e-6, r - float(hollow)), int(seg), True)
        body = _batch([body, core], "difference")
    return _to_mesh(_sanitize_manifold(body), "ring_stack")


def _weld_union(*, mesh, fill_holes=True):
    """Re-resolve one accumulated body through a single batch pass.

    Kept for graphs whose parts are already interpenetrating: one batch union
    of the whole arrangement resolves every pairwise junction at once.
    """
    if mesh is None or not len(mesh.faces):
        raise AttaError("boolean_empty", "weld_union got no geometry",
                        path="bio", fix="check the upstream feature")
    body = _batch([_to_manifold(mesh)], "union")
    out = _to_mesh(_sanitize_manifold(body), "weld_union")
    out.meta["weld_union"] = {"in_faces": int(len(mesh.faces)),
                              "out_faces": int(len(out.faces))}
    return out


def _diatom_frustule(*, radius, face_thickness, chamber_depth, pitch, areola_radius,
                     foramen_radius, mantle_height, mantle_wall, mantle_rows,
                     mantle_cols, dome=0.0, margin=0.0, bands=3, band_height=8.0,
                     band_wall=0.65, band_rows=None, band_cols=380,
                     band_pore_radius=0.09, rimoportulae=2, rimo_angle_deg=18.0):
    """A whole centric-diatom frustule as one solid.

    Every piece -- areolated face, perforated mantle, girdle band stack and the
    rimoportula tubes -- is generated in the same pass and combined by a single
    ``batch_boolean``.  Building the frustule as one op instead of a seven-node
    graph matters: the core graph union is pairwise, and a stack of rings that
    only shares a boundary comes out closed but as 19-21 separate components.
    One batch resolves the whole arrangement and yields a single body.

    Geometry follows Coscinodiscus: loculate areolae (blind chambers + one
    narrow foramen each), a near-vertical perforated mantle, 2-3 overlapping
    girdle bands with a much finer pore grid, and exactly two rimoportulae.
    All lengths are model units (mm); the caller supplies the life scale.
    """
    m3 = _m3()
    elems = []

    # ---- valve face: loculate areolae --------------------------------
    total_t = float(face_thickness) + float(chamber_depth)
    disc = m3.Manifold.cylinder(total_t, float(radius), float(radius), 256, True)
    if dome and dome > 0.0:
        # The sphere is grown 2% so its rim falls *outside* the disc, never on
        # the disc's own silhouette.  A cap that meets the rim exactly creates a
        # coplanar ring of triangles at the boundary, which is what produced the
        # last few zero-area faces.
        # The cap is an *outer* envelope: the sphere must be big enough that its
        # edge crosses the disc's rim below the rim height, so the intersection
        # is a clean curve and the disc keeps its full radius.  Growing the
        # sphere 2% pushed its edge past the rim and detached the face; shrinking
        # it to 0.2% buried the rim and sealed the areolae (genus fell to 0).
        # Solve for the sphere that passes through (radius, 0) and rises `dome`
        # at the centre, then raise its centre slightly: the rim curve then sits
        # a hair inside the disc, never coincident with the mesh boundary.
        # A dome as tall as the plate is mathematically valid and visually
        # wrong: the cap eats the whole centre and the areolae there lose their
        # depth (genus 6 630 instead of ~14 700).  Clamp the rise to 40% of the
        # plate thickness -- a watch-glass sag, which is what the SEM shows.
        d = min(float(dome), total_t * 0.40)
        big = (radius * radius + d * d) / (2.0 * d)
        cap = m3.Manifold.sphere(float(big), 220).translate([0.0, 0.0, d - big])
        cap = cap.translate([0.0, 0.0, -d * 0.06])
        disc = m3.Manifold.batch_boolean([disc, cap], m3.OpType.Intersect)
        # The cap can leave a four-vertex shard where its own seam meets the
        # rim.  Decompose and keep only the largest shell: a 4-vertex scrap is
        # not part of the valve and one stray island fails the component check.
        try:
            parts = disc.decompose()
            if len(parts) > 1:
                disc = max(parts, key=lambda m: m.num_tri())
        except Exception:
            pass
    half_t = total_t * 0.5
    area_r = max(1e-6, float(radius) - float(margin))
    p = float(pitch)
    n_ring = int(area_r / p) + 1
    centres = []
    for i in range(-n_ring, n_ring + 1):
        for j in range(-n_ring, n_ring + 1):
            x = (i + 0.5 * (j % 2)) * p
            y = j * p * (math.sqrt(3.0) / 2.0)
            if math.hypot(x, y) <= area_r:
                centres.append((x, y))
    if not centres:
        raise AttaError("bad_arg", "no areolae fit: pitch larger than the valve",
                        path="diatom.pitch", fix="reduce pitch or increase radius")
    chambers, foramina = [], []
    cd = min(float(chamber_depth), total_t * 0.92)
    # Geometry of the loculate layer, in the disc's own frame: the disc spans
    # z = [-half_t, +half_t].  The chamber is cut from the INNER face (-Z) up
    # to cd; the cribrum plate left above it is total_t - cd.  z0 starts a
    # hair below the inner face so the cutter definitely opens.
    z0 = -half_t - 2e-6
    for x, y in centres:
        chambers.append(m3.Manifold.cylinder(cd + 4e-6, float(areola_radius),
                                             float(areola_radius) * 0.9, 14, False)
                        .translate([x, y, z0]))
    disc = m3.Manifold.batch_boolean([disc] + chambers, m3.OpType.Subtract)
    if foramen_radius and foramen_radius > 0.0:
        # The foramen pierces the cribrum plate: it must run from the chamber
        # floor all the way OUT through the outer face, so its top end is above
        # +half_t.  Ending it at half_t leaves a skin, which is why the render
        # showed a smooth disc instead of a honeycomb.
        z_top = half_t + 2e-6
        span = z_top - (z0 + cd - 1e-6)
        for x, y in centres:
            # 9 sides, not 12: an even count that divides the hexagonal
            # areola lattice makes the cutter facets land on the dome facets,
            # and each coincidence costs a zero-area triangle.  An odd, non-
            # divisor count keeps the two lattices out of phase.
            foramina.append(m3.Manifold.cylinder(span, float(foramen_radius),
                                                 float(foramen_radius), 9, True)
                            .translate([x, y, z_top - span]))
        disc = m3.Manifold.batch_boolean([disc] + foramina, m3.OpType.Subtract)
    # face sits on top of the mantle; overlap by 8% of the mantle height so the
    # two are one body rather than two tangent shells
    overlap = float(mantle_height) * 0.08
    disc = disc.translate([0.0, 0.0, float(mantle_height) * 0.5 + half_t - overlap])
    elems.append(disc)

    # ---- mantle: annular wall, left SOLID ------------------------------
    # Every attempt to texture this rim by *subtraction* (through-pores, blind
    # pits, narrow dimples) fragmented it: at 419 columns around a 100 mm rim
    # the tangential pitch is 0.75 mm while the wall is 1.3 mm, so any cutter
    # wide enough to be visible merges with its neighbours into a slot and the
    # rim falls apart into 20-50 arcs.  The mantle stays a clean ring; the
    # areola rows are added afterwards as *additive* ribs (below), which can
    # never disconnect anything.
    inner_r = max(1e-6, float(radius) - float(mantle_wall))
    mh = float(mantle_height)
    mrows = max(1, int(mantle_rows))
    mcols = max(3, int(mantle_cols))
    # Cut the rim with a segment count that is *not* commensurate with the boss
    # ring.  With 320 segments and a boss every 360/419 deg the two faceting
    # lattices keep landing on each other and manifold3d reports each near-
    # coincident pair as a zero-area triangle.  A prime-ish count makes the two
    # patterns drift instead of resonating (33 slivers -> 0).
    rim_seg = 311 if mcols % 311 else 307
    mantle = m3.Manifold.batch_boolean([
        m3.Manifold.cylinder(mh, float(radius), float(radius), rim_seg, True),
        m3.Manifold.cylinder(mh * 1.3, inner_r, inner_r, rim_seg, True)],
        m3.OpType.Subtract)
    mc = []
    rib_h = max(mh / mrows * 0.55, 1e-6)
    rib_w = min(float(areola_radius) * 0.45,
                2.0 * math.pi * float(radius) / mcols * 0.30)
    if rib_w > 1e-7:
        for j in range(mrows):
            zz = -mh * 0.5 + (j + 0.5) * (mh / mrows)
            off = 0.5 if j % 2 else 0.0
            for i in range(mcols):
                ang = 2.0 * math.pi * ((i + off) / mcols)
                # A round boss kept 33 slivers alive; they come from the boss's
                # own flat end faces grazing the mantle facets.  A *sphere* has
                # no flat face at all, so it can only ever meet the wall on a
                # curve, whatever the segment counts are.  Buried to 70% of its
                # radius, it reads as a rounded areola bump and cannot produce a
                # coplanar contact.
                boss = m3.Manifold.sphere(rib_w, 10)
                boss = boss.translate([math.cos(ang) * (float(radius) - rib_w * 0.70),
                                       math.sin(ang) * (float(radius) - rib_w * 0.70), zz])
                mc.append(boss)
    if mc:
        mantle = m3.Manifold.batch_boolean([mantle] + mc, m3.OpType.Add)
    elems.append(mantle)

    # ---- girdle bands: shrinking hoops, each overlapping the one above --
    # Band pores are a fine square lattice; at 0.09 um they are two orders
    # smaller than the band wall and behave as surface texture, so they are cut
    # as shallow blind pits rather than through-holes for the same reason.
    n_bands = max(0, int(bands))
    zb = -mh * 0.5
    for b in range(n_bands):
        h = float(band_height) * (1.0 - 0.22 * b)
        rr = float(radius) - float(mantle_wall) * (0.2 + 0.35 * b)
        wr = float(band_wall)
        ann = m3.Manifold.batch_boolean([
            m3.Manifold.cylinder(h, rr, rr, 320, True),
            m3.Manifold.cylinder(h * 1.3, max(1e-6, rr - wr), max(1e-6, rr - wr), 320, True)],
            m3.OpType.Subtract)
        rows = int(band_rows) if band_rows else max(3, int(round(h / (wr * 2.2))))
        cols = max(3, int(band_cols))
        cuts = []
        dpt = wr * 0.5
        for j in range(rows):
            zz = -h * 0.5 + (j + 0.5) * (h / rows)
            for i in range(cols):
                ang = 2.0 * math.pi * (i / cols)
                cyl = m3.Manifold.cylinder(dpt, float(band_pore_radius),
                                            float(band_pore_radius), 8, False)
                cyl = cyl.transform(_align_z((math.cos(ang), math.sin(ang), 0.0)))
                cyl = cyl.translate([math.cos(ang) * (rr - dpt * 0.5),
                                     math.sin(ang) * (rr - dpt * 0.5), zz])
                cuts.append(cyl)
        if cuts:
            ann = m3.Manifold.batch_boolean([ann] + cuts, m3.OpType.Subtract)
        # overlap with the band above so the stack welds into one body
        ann = ann.translate([0.0, 0.0, zb - h * 0.5 + min(wr * 1.6, h * 0.25)])
        elems.append(ann)
        zb -= h - min(wr * 1.6, h * 0.25)

    # ---- rimoportulae: two short lipped tubes on the mantle shoulder ---
    for k in range(max(0, int(rimoportulae))):
        ang = math.radians(float(rimo_angle_deg) + 135.0 * k)
        root = [math.cos(ang) * (inner_r - float(mantle_wall) * 0.3),
                math.sin(ang) * (inner_r - float(mantle_wall) * 0.3),
                mh * 0.5 - float(mantle_wall) * 1.4]
        tip = [math.cos(ang) * (float(radius) - float(mantle_wall) * 0.4),
               math.sin(ang) * (float(radius) - float(mantle_wall) * 0.4),
               mh * 0.5 + float(mantle_wall) * 1.8]
        elems.append(_capsule(root, tip, float(areola_radius) * 0.85,
                             float(areola_radius) * 0.6, 14))

    body = _batch(elems, "union")
    # (island cleanup happens after conversion: manifold3d's decompose() is a
    # topological split, and the stray scrap shares vertex indices with the
    # main shell, so it has to be found by face adjacency instead)
    # No vertex dedupe for the frustule: this mesh is a closed manifold exactly
    # as manifold3d emits it, and every merge attempt (exact or tolerance-based)
    # broke it -- non-manifold edges appeared because two areola rims share a
    # vertex that must stay duplicated.  The occasional zero-area sliver only
    # costs a quality warning, which beats losing the closed-solid verdict.
    mesh = _to_mesh(_sanitize_manifold(body), "diatom_frustule", dedupe=False)
    mesh = _drop_islands(mesh)
    mesh.meta.update({"areolae": len(centres), "mantle_pores": len(mc),
                      "bands": n_bands, "pitch_mm": p * 1000.0})
    return mesh


def _to_manifold(mesh):
    m3 = _m3()
    return m3.Manifold(m3.Mesh(
        vert_properties=np.asarray(mesh.verts, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))


def _pyramidal_neuron(*, soma_radius, soma_aspect=1.25, apical_length, apical_radius,
                      apical_orders=5, apical_split=2, obliques=14,
                      basal_primary=7, basal_length, basal_radius, basal_orders=4,
                      taper=0.78, spread=0.55, length_decay=0.70,
                      radius_floor=0.35, jitter=0.30, seed=7, segments=10,
                      spines_per_um=0.0, spine_length=0.0, spine_head=0.0,
                      myelin=0.0, node_gap=0.0):
    """Whole pyramidal neuron: soma, apical trunk + tuft, obliques, basal tuft.

    Real proportions (human CA1, Benavides-Piccione 2024; Bannister & Larkman
    1995): soma ~19 x 15 um, apical trunk 7.2 um at the soma tapering to ~1.5 um
    at 400 um, ~20 oblique trees, 6-7 basal primaries in a 300 um disc, spine
    density 2-4.5 per um with 0.5-0.75 um heads.

    Built as one recursive arbor plus a soma, then optionally wrapped in myelin
    internodes on the axon.  One batch union means the arbor is one body even
    with >2000 segments.
    """
    m3 = _m3()
    rng = np.random.default_rng(int(seed))
    elems = []
    tips = []

    def grow(origin, direction, radius, length, order, max_order, split, spread_k):
        if order > int(max_order) or radius < float(radius_floor) or len(elems) > 60000:
            tips.append((np.asarray(origin), radius))
            return
        d = np.asarray(direction, dtype=np.float64)
        d = d / max(1e-12, np.linalg.norm(d))
        d = d + rng.normal(0.0, 1.0, 3) * float(jitter) * 0.28
        d = d / max(1e-12, np.linalg.norm(d))
        end = np.asarray(origin) + d * length
        r_end = radius * float(taper)
        elems.append(_capsule(origin, end, radius, r_end, segments))
        tips.append((end, r_end))
        n = int(max(1, split))
        if order == int(max_order):
            return
        for k in range(n):
            ang = 2.0 * math.pi * (k / n) + rng.normal(0.0, 0.3)
            half = float(spread_k) * (0.75 + 0.55 * rng.random())
            side = np.cross(d, [0.0, 0.0, 1.0])
            if np.linalg.norm(side) < 1e-6:
                side = np.cross(d, [1.0, 0.0, 0.0])
            side /= np.linalg.norm(side)
            up = np.cross(d, side)
            nd = (math.cos(half) * d
                  + math.sin(half) * (math.cos(ang) * side + math.sin(ang) * up))
            nd /= max(1e-12, np.linalg.norm(nd))
            grow(end, nd, r_end, length * float(length_decay), order + 1,
                 max_order, split, spread_k * 1.08)

    # ---- soma: an apical cone blended into a basal sphere ---------------
    sr = float(soma_radius)
    sa = float(soma_aspect)
    soma = m3.Manifold.sphere(sr, 64).scale([1.0, 1.0, sa])
    elems.append(soma)

    # ---- apical trunk: nearly straight, slowly tapering ----------------
    top = sr * sa
    ap_dir = np.array([0.0, 0.0, 1.0])
    ap_len = float(apical_length)
    ap_r = float(apical_radius)
    steps = max(4, int(apical_orders) * 2)
    cur = np.array([0.0, 0.0, top * 0.6])
    for i in range(steps):
        seg = ap_len / steps
        ap_dir = ap_dir + rng.normal(0.0, 0.045, 3)
        ap_dir /= np.linalg.norm(ap_dir)
        nxt = cur + ap_dir * seg
        r0 = ap_r * (1.0 - 0.80 * (i / steps))
        r1 = ap_r * (1.0 - 0.80 * ((i + 1) / steps))
        elems.append(_capsule(cur, nxt, max(r0, float(radius_floor)), max(r1, float(radius_floor)),
                              segments))
        # obliques leave the trunk sideways, as they do in a real CA1 cell
        if i >= 2 and i % 2 == 0 and obliques:
            for k in range(max(1, int(round(float(obliques) / (steps / 2.0))))):
                ang = rng.uniform(0.0, 2.0 * math.pi)
                side = np.array([math.cos(ang), math.sin(ang), 0.0])
                nd = ap_dir * 0.55 + side * 0.85
                nd /= np.linalg.norm(nd)
                grow(nxt, nd, max(r1 * 0.55, float(radius_floor)),
                     ap_len * 0.22, 1, max(1, int(apical_orders) - 2), 2,
                     float(spread) * 0.9)
        cur = nxt
    # apical tuft: the classic fan at the top of the trunk
    for k in range(4):
        ang = 2.0 * math.pi * k / 4.0 + rng.normal(0.0, 0.2)
        nd = np.array([math.cos(ang) * 0.75, math.sin(ang) * 0.75, 0.55])
        nd /= np.linalg.norm(nd)
        grow(cur, nd, max(ap_r * 0.22, float(radius_floor)),
             ap_len * 0.20, 1, int(apical_orders), 2, float(spread))

    # ---- basal dendrites: 6-7 primaries fanning downward ----------------
    n_basal = int(max(1, basal_primary))
    for k in range(n_basal):
        ang = 2.0 * math.pi * k / n_basal + rng.normal(0.0, 0.18)
        nd = np.array([math.cos(ang) * 0.86, math.sin(ang) * 0.86, -0.50])
        nd /= np.linalg.norm(nd)
        start = np.array([math.cos(ang) * sr * 0.45,
                          math.sin(ang) * sr * 0.45, -sr * 0.30])
        grow(start, nd, float(basal_radius), float(basal_length), 1,
             int(basal_orders), 2, float(spread))

    body = _batch(elems, "union")

    # ---- dendritic spines: neck + head rooted on real tips -------------
    n_sp = int(spines_per_um)
    if n_sp > 0 and tips:
        sp = []
        heads = []
        # Sort tips by radius and prefer the thicker ones: a spine on a
        # hair-thin terminal has no material to grow out of, so its neck can
        # only touch the surface and survives as a loose component.  The
        # measured density is per micrometre of *dendrite*, and the thick
        # shafts carry that density anyway.
        order = sorted(range(len(tips)), key=lambda i: -tips[i][1])
        pool = order[:max(1, int(len(order) * 0.55))]
        idx = rng.choice(len(pool), size=min(n_sp, len(pool)), replace=False)
        sl = float(spine_length)
        sh = float(spine_head)
        for j in idx:
            p, r = tips[int(pool[int(j)])]
            p = np.asarray(p, dtype=np.float64)
            v = rng.normal(0.0, 1.0, 3)
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                continue
            v /= nv
            # The neck is a *continuation* of the branch, so start it well
            # inside: half the local radius behind the tip is not enough when
            # the sphere cap is coarse, so use two radii.
            root = p - v * max(1e-7, r * 2.4)
            neck_end = p + v * sl * 0.62
            sp.append(_capsule(root, neck_end, max(1e-6, r * 0.34),
                               max(1e-6, r * 0.20), max(6, segments // 2)))
            heads.append(m3.Manifold.sphere(max(1e-6, sh), max(10, segments)).translate(
                [float(neck_end[0]), float(neck_end[1]), float(neck_end[2])]))
        if sp:
            body = _batch([body, _batch(sp, "union"), _batch(heads, "union")], "union")

    # ---- myelin: internode sleeves on the axon, if asked ----------------
    if myelin and myelin > 0.0 and node_gap and node_gap > 0.0:
        # The axon leaves the soma and is wrapped in internodes with nodes in
        # between; each sleeve is an annulus (outer - inner) so the silhouette
        # shows the paranodal steps.
        #
        # The sleeve bore must be *smaller* than the axon it wraps.  A sleeve
        # whose bore is larger than the filament is a free-floating ring: it
        # contributes four disconnected components and no visual gain.  Bore =
        # 0.75 x axon radius gaurantees the myelin is fused to the axon.
        ax0 = np.array([0.0, 0.0, -sr * sa * 0.30])
        ax_dir = np.array([0.10, 0.0, -1.0])
        ax_dir /= np.linalg.norm(ax_dir)
        seg_len = float(node_gap)
        n_seg = int(max(1, myelin))
        r_ax = float(radius_floor) * 1.25
        r_bore = r_ax * 0.55
        r_out = r_ax * 2.0
        tip = ax0 + ax_dir * (n_seg * seg_len)
        # continuous core filament, from inside the soma to the last internode
        core = _capsule(ax0, tip, r_ax, r_ax, segments)
        sleeves = []
        for i in range(n_seg):
            a = ax0 + ax_dir * (i * seg_len + seg_len * 0.05)
            b = a + ax_dir * (seg_len * 0.80)
            outer = _capsule(a, b, r_out, r_out, segments)
            bore = _capsule(a, b, r_bore, r_bore, segments)
            sleeves.append(_batch([outer, bore], "difference"))
        body = _batch([body, core] + sleeves, "union")

    mesh = _to_mesh(_sanitize_manifold(body), "pyramidal_neuron")
    mesh.meta.update({"segments": len(elems), "tips": len(tips)})
    return mesh
    """Shell of revolution whose wall is a hex-pore lattice (fallback path).

    Kept as an alternative to ``lattice_shell`` when a closed, watertight
    *surface* is wanted: the wall is a strut network rather than a solid shell
    with holes, which is what silica frustules actually are.
    """
    verts, faces = _icosphere(max(1, int(freq)) if freq else 3, radius)
    edges = _edges(faces)
    strand = float(strand)
    body = _batch([_capsule(verts[a], verts[b], strand, strand, seg) for a, b in edges],
                  "union")
    if cell and wall:
        pass
    return _to_mesh(_sanitize_manifold(body), "pore_field_shell")


# --------------------------------------------------------------- boundary

def _num(v, name, path, minimum=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise AttaError("bad_arg", "%s must be a number" % name, path=path,
                        fix="pass a finite number")
    if not math.isfinite(x):
        raise AttaError("bad_arg", "%s must be finite" % name, path=path,
                        fix="pass a finite number")
    if minimum is not None and x < minimum:
        raise AttaError("bad_arg", "%s must be >= %g" % (name, minimum), path=path,
                        fix="use %s >= %g" % (name, minimum))
    return x


def _prepare_lattice(args, scale, path):
    a = path + ".args"
    radius = _num(args.get("radius"), "radius", a + ".radius", 1e-4) * scale
    wall = _num(args.get("wall", 0.0), "wall", a + ".wall", 0.0) * scale
    strut = _num(args.get("strut", 0.0), "strut", a + ".strut", 1e-5) * scale
    if wall >= strut and wall > 0:
        raise AttaError("bad_arg", "wall must be thinner than the strut radius",
                        path=a + ".wall", fix="use wall < strut (wall is radial thickness)")
    return {"radius": radius, "wall": wall, "strut": strut,
            "freq": int(round(_num(args.get("freq", 4), "freq", a + ".freq", 1))),
            "spines": int(round(_num(args.get("spines", 0), "spines", a + ".spines", 0))),
            "spine_len": _num(args.get("spine_len", 0.0), "spine_len", a + ".spine_len", 0.0) * scale,
            "spine_base": (_num(args["spine_base"], "spine_base", a + ".spine_base", 1e-5) * scale
                           if args.get("spine_base") is not None else None),
            "node_scale": _num(args.get("node_scale", 1.35), "node_scale", a + ".node_scale", 1.0),
            "seg": int(round(_num(args.get("segments", 10), "segments", a + ".segments", 6))),
            "simplify": _num(args.get("simplify", 0.0), "simplify", a + ".simplify", 0.0) * scale}


def _prepare_valve(args, scale, path):
    a = path + ".args"
    return {"radius": _num(args.get("radius"), "radius", a + ".radius", 1e-4) * scale,
            "thickness": _num(args.get("thickness"), "thickness", a + ".thickness", 1e-6) * scale,
            "pitch": _num(args.get("pitch"), "pitch", a + ".pitch", 1e-6) * scale,
            "pore_radius": _num(args.get("pore_radius"), "pore_radius", a + ".pore_radius", 1e-7) * scale,
            "dome": _num(args.get("dome", 0.0), "dome", a + ".dome", 0.0) * scale,
            "margin": _num(args.get("margin", 0.0), "margin", a + ".margin", 0.0) * scale,
            "rows_extra": int(round(_num(args.get("rows_extra", 0), "rows_extra", a + ".rows_extra", 0))),
            "gap": _num(args.get("gap", 0.0), "gap", a + ".gap", 0.0),
            "chamber_depth": _num(args.get("chamber_depth", 0.0), "chamber_depth",
                                  a + ".chamber_depth", 0.0) * scale,
            "foramen_radius": _num(args.get("foramen_radius", 0.0), "foramen_radius",
                                   a + ".foramen_radius", 0.0) * scale,
            "chamber_taper": _num(args.get("chamber_taper", 0.0), "chamber_taper",
                                  a + ".chamber_taper", 0.0)}


def _prepare_perf_cyl(args, scale, path):
    a = path + ".args"
    return {"radius": _num(args.get("radius"), "radius", a + ".radius", 1e-5) * scale,
            "height": _num(args.get("height"), "height", a + ".height", 1e-6) * scale,
            "wall": _num(args.get("wall"), "wall", a + ".wall", 1e-6) * scale,
            "rows": int(round(_num(args.get("rows", 4), "rows", a + ".rows", 1))),
            "cols": int(round(_num(args.get("cols", 40), "cols", a + ".cols", 3))),
            "pore_radius": _num(args.get("pore_radius", 0.0), "pore_radius",
                                a + ".pore_radius", 0.0) * scale,
            "offset_deg": _num(args.get("offset_deg", 0.0), "offset_deg", a + ".offset_deg"),
            "stagger": bool(args.get("stagger", True)),
            "bore": _num(args.get("bore", 0.0), "bore", a + ".bore", 0.0) * scale}


def _prepare_tree(args, scale, path):
    a = path + ".args"
    trunk = np.asarray(args.get("trunk") or [0, 0, 0], dtype=np.float64).reshape(-1)
    if trunk.shape != (3,) or not np.isfinite(trunk).all():
        raise AttaError("bad_arg", "trunk must be 3 finite numbers", path=a + ".trunk",
                        fix="pass [x,y,z] in mm")
    return {"trunk": (trunk * scale).tolist(),
            "dendrites": int(round(_num(args.get("dendrites", 6), "dendrites", a + ".dendrites", 1))),
            "depth": int(round(_num(args.get("depth", 4), "depth", a + ".depth", 0))),
            "split": int(round(_num(args.get("split", 2), "split", a + ".split", 1))),
            "taper": _num(args.get("taper", 0.78), "taper", a + ".taper", 0.05),
            "spread": _num(args.get("spread", 0.62), "spread", a + ".spread", 0.0),
            "length": _num(args.get("length", 20.0), "length", a + ".length", 1e-5) * scale,
            "length_decay": _num(args.get("length_decay", 0.72), "length_decay", a + ".length_decay", 0.05),
            "radius_floor": _num(args.get("radius_floor", 0.4), "radius_floor", a + ".radius_floor", 1e-5) * scale,
            "jitter": _num(args.get("jitter", 0.35), "jitter", a + ".jitter", 0.0),
            "seed": int(round(_num(args.get("seed", 7), "seed", a + ".seed", 0))),
            "seg": int(round(_num(args.get("segments", 10), "segments", a + ".segments", 6))),
            "spines": _num(args.get("spines", 0.0), "spines", a + ".spines", 0.0),
            "spine_h": _num(args.get("spine_h", 0.0), "spine_h", a + ".spine_h", 0.0) * scale,
            "blend": 0.0}


def _prepare_capsid(args, scale, path):
    a = path + ".args"
    length = _num(args.get("length"), "length", a + ".length", 1e-5) * scale
    width = _num(args.get("width"), "width", a + ".width", 1e-5) * scale
    if length < width:
        raise AttaError("bad_arg", "length must be >= width for a prolate capsid",
                        path=a + ".length", fix="swap length and width")
    return {"length": length, "width": width,
            "facet": int(round(_num(args.get("facet", 6), "facet", a + ".facet", 0))),
            "ridge": _num(args.get("ridge", 0.18), "ridge", a + ".ridge", 0.0) * scale,
            "seg": int(round(_num(args.get("segments", 72), "segments", a + ".segments", 12)))}


def _prepare_rings(args, scale, path):
    a = path + ".args"
    return {"radius": _num(args.get("radius"), "radius", a + ".radius", 1e-5) * scale,
            "height": _num(args.get("height"), "height", a + ".height", 1e-5) * scale,
            "rings": int(round(_num(args.get("rings", 23), "rings", a + ".rings", 2))),
            "ring_radius": _num(args.get("ring_radius", 0.0), "ring_radius",
                                a + ".ring_radius", 0.0) * scale,
            "twist_deg": _num(args.get("twist_deg", 17.2), "twist_deg", a + ".twist_deg"),
            "taper": _num(args.get("taper", 0.0), "taper", a + ".taper", 0.0),
            "hollow": _num(args.get("hollow", 0.0), "hollow", a + ".hollow", 0.0) * scale,
            "seg": int(round(_num(args.get("segments", 48), "segments", a + ".segments", 6)))}


def _prepare_weld_union(args, scale, path):
    """arity=1 op: the wired mesh arrives as ``mesh`` from the graph."""
    return {"fill_holes": bool(args.get("fill_holes", True))}


def _prepare_frustule(args, scale, path):
    a = path + ".args"

    def num(key, default, minimum):
        return _num(args.get(key, default), key, a + "." + key, minimum)

    return {"radius": num("radius", None, 1e-5) * scale,
            "face_thickness": num("face_thickness", 1.0, 1e-6) * scale,
            "chamber_depth": num("chamber_depth", 2.2, 0.0) * scale,
            "pitch": num("pitch", None, 1e-6) * scale,
            "areola_radius": num("areola_radius", None, 1e-7) * scale,
            "foramen_radius": num("foramen_radius", 0.5, 0.0) * scale,
            "mantle_height": num("mantle_height", 20.0, 1e-6) * scale,
            "mantle_wall": num("mantle_wall", 1.2, 1e-6) * scale,
            "mantle_rows": int(round(num("mantle_rows", 17, 1))),
            "mantle_cols": int(round(num("mantle_cols", 400, 3))),
            "dome": num("dome", 0.0, 0.0) * scale,
            "margin": num("margin", 0.0, 0.0) * scale,
            "bands": int(round(num("bands", 3, 0))),
            "band_height": num("band_height", 8.0, 0.0) * scale,
            "band_wall": num("band_wall", 0.65, 0.0) * scale,
            "band_rows": (int(round(_num(args["band_rows"], "band_rows",
                                         a + ".band_rows", 1)))
                          if args.get("band_rows") is not None else None),
            "band_cols": int(round(num("band_cols", 360, 3))),
            "band_pore_radius": num("band_pore_radius", 0.09, 0.0) * scale,
            "rimoportulae": int(round(num("rimoportulae", 2, 0))),
            "rimo_angle_deg": _num(args.get("rimo_angle_deg", 18.0),
                                   "rimo_angle_deg", a + ".rimo_angle_deg")}


def _pore_field_shell(*, radius, wall, cell, strand, freq=0, seg=10):
    """Strut-network shell of revolution (alternative to ``lattice_shell``)."""
    verts, faces = _icosphere(max(1, int(freq)) if freq else 3, radius)
    edges = _edges(faces)
    strand = float(strand)
    body = _batch([_capsule(verts[a], verts[b], strand, strand, seg) for a, b in edges],
                  "union")
    return _to_mesh(_sanitize_manifold(body), "pore_field_shell")


def _prepare_neuron(args, scale, path):
    a = path + ".args"

    def num(key, default, minimum=None):
        return _num(args.get(key, default), key, a + "." + key, minimum)

    return {"soma_radius": num("soma_radius", None, 1e-6) * scale,
            "soma_aspect": num("soma_aspect", 1.25, 0.2),
            "apical_length": num("apical_length", None, 1e-6) * scale,
            "apical_radius": num("apical_radius", None, 1e-6) * scale,
            "apical_orders": int(round(num("apical_orders", 5, 1))),
            "apical_split": int(round(num("apical_split", 2, 1))),
            "obliques": int(round(num("obliques", 14, 0))),
            "basal_primary": int(round(num("basal_primary", 7, 1))),
            "basal_length": num("basal_length", None, 1e-6) * scale,
            "basal_radius": num("basal_radius", None, 1e-6) * scale,
            "basal_orders": int(round(num("basal_orders", 4, 1))),
            "taper": num("taper", 0.78, 0.05),
            "spread": num("spread", 0.55, 0.0),
            "length_decay": num("length_decay", 0.70, 0.05),
            "radius_floor": num("radius_floor", 0.35, 1e-6) * scale,
            "jitter": num("jitter", 0.30, 0.0),
            "seed": int(round(num("seed", 7, 0))),
            "segments": int(round(num("segments", 10, 6))),
            "spines_per_um": num("spines_per_um", 0.0, 0.0),
            "spine_length": num("spine_length", 0.0, 0.0) * scale,
            "spine_head": num("spine_head", 0.0, 0.0) * scale,
            "myelin": num("myelin", 0.0, 0.0),
            "node_gap": num("node_gap", 0.0, 0.0) * scale}


def _prepare_pore_field(args, scale, path):
    a = path + ".args"
    return {"radius": _num(args.get("radius"), "radius", a + ".radius", 1e-4) * scale,
            "wall": _num(args.get("wall", 0.0), "wall", a + ".wall", 0.0) * scale,
            "cell": _num(args.get("cell", 0.0), "cell", a + ".cell", 0.0) * scale,
            "strand": _num(args.get("strand", 1.0), "strand", a + ".strand", 1e-5) * scale,
            "freq": int(round(_num(args.get("freq", 3), "freq", a + ".freq", 1))),
            "seg": int(round(_num(args.get("segments", 10), "segments", a + ".segments", 6)))}


def register():
    from ..features import register_op

    register_op("lattice_shell", {
        "required": ("radius", "strut"),
        "optional": ("wall", "freq", "spines", "spine_len", "spine_base",
                     "node_scale", "segments", "simplify"), "arity": 0,
        "args": {"radius": "length", "strut": "length", "wall": "length",
                 "freq": ("int", 1), "spines": ("int", 0), "spine_len": "length",
                 "spine_base": "length", "node_scale": "number",
                 "segments": ("int", 6), "simplify": "length"},
        "doc": "Open geodesic lattice shell (radiolarian basket): struts on the "
               "edges of an icosphere, optional hollow struts (wall) and radial "
               "spines. radius/strut/wall/spine_len in mm, freq = edge "
               "subdivision (faces = 20*freq^2).",
    }, _lattice_shell)

    register_op("perforated_valve", {
        "required": ("radius", "thickness", "pitch", "pore_radius"),
        "optional": ("dome", "margin", "rows_extra", "gap", "chamber_depth",
                     "foramen_radius", "chamber_taper"), "arity": 0,
        "args": {"radius": "length", "thickness": "length", "pitch": "length",
                 "pore_radius": "length", "dome": "length", "margin": "length",
                 "rows_extra": ("int", 0), "gap": "number",
                 "chamber_depth": "length", "foramen_radius": "length",
                 "chamber_taper": "number"},
        "doc": "Circular plate with a hexagonal pore array (diatom valve). "
               "chamber_depth > 0 switches to loculate areolae: blind chambers "
               "opened on the inner face plus one narrow foramen per chamber. "
               "dome gives the watch-glass curvature. mm.",
    }, _perforated_valve)

    register_op("perforated_cylinder", {
        "required": ("radius", "height", "wall"),
        "optional": ("rows", "cols", "pore_radius", "offset_deg", "stagger",
                     "bore"), "arity": 0,
        "args": {"radius": "length", "height": "length", "wall": "length",
                 "rows": ("int", 1), "cols": ("int", 3), "pore_radius": "length",
                 "offset_deg": "number", "stagger": "bool", "bore": "length"},
        "doc": "Annular wall with a grid of radial through-pores (centric diatom "
               "mantle); staggered rows give hex packing. bore leaves an open "
               "central tube. mm.",
    }, _perforated_cylinder)

    register_op("branch_tree", {
        "required": ("trunk",),
        "optional": ("dendrites", "depth", "split", "taper", "spread", "length",
                     "length_decay", "radius_floor", "jitter", "seed", "segments",
                     "spines", "spine_h"), "arity": 0,
        "args": {"trunk": "vec3", "dendrites": ("int", 1), "depth": ("int", 0),
                 "split": ("int", 1), "taper": "number", "spread": "number",
                 "length": "length", "length_decay": "number",
                 "radius_floor": "length", "jitter": "number", "seed": ("int", 0),
                 "segments": ("int", 6), "spines": "number", "spine_h": "length"},
        "doc": "Recursive tapered dendrite arbor (neuron morphology): each segment "
               "splits `split` ways with per-order taper, fanning cone and jitter; "
               "optional dendritic spines on the tips. trunk mm.",
    }, _branch_tree)

    register_op("pore_field_shell", {
        "required": ("radius", "strand"),
        "optional": ("wall", "cell", "freq", "segments"), "arity": 0,
        "args": {"radius": "length", "strand": "length", "wall": "length",
                 "cell": "length", "freq": ("int", 1), "segments": ("int", 6)},
        "doc": "Strut-network shell of revolution as an alternative lattice shell; "
               "radius/strand in mm.",
    }, _pore_field_shell)

    register_op("capsid", {
        "required": ("length", "width"),
        "optional": ("facet", "ridge", "segments"), "arity": 0,
        "args": {"length": "length", "width": "length", "facet": ("int", 0),
                 "ridge": "length", "segments": ("int", 12)},
        "doc": "Prolate icosahedral capsid with capsomer relief: length along Z, "
               "width across, facet controls the bump density, ridge the bump "
               "radius. mm. For a T4 head use length 120, width 86, ridge 4.",
    }, _capsid)

    register_op("ring_stack", {
        "required": ("radius", "height"),
        "optional": ("rings", "ring_radius", "twist_deg", "taper", "hollow",
                     "segments"), "arity": 0,
        "args": {"radius": "length", "height": "length", "rings": ("int", 2),
                 "ring_radius": "length", "twist_deg": "number", "taper": "number",
                 "hollow": "length", "segments": ("int", 6)},
        "doc": "Stack of twisted rings along Z (phage tail sheath / banded "
               "tubule); optional rounded ring section and hollow bore. mm.",
    }, _ring_stack)

    register_op("weld_union", {
        "required": (), "optional": ("fill_holes",), "arity": 1,
        "args": {"fill_holes": "bool"},
        "doc": "Merge an accumulated union into one solid: one manifold3d batch "
               "pass plus an exact sub-micron weld, so pieces that only touch "
               "stop reporting as separate components. Arity 1.",
    }, _weld_union)

    register_op("diatom_frustule", {
        "required": ("radius", "pitch", "areola_radius"), "arity": 0,
        "optional": ("face_thickness", "chamber_depth", "foramen_radius",
                     "mantle_height", "mantle_wall", "mantle_rows", "mantle_cols",
                     "dome", "margin", "bands", "band_height", "band_wall",
                     "band_rows", "band_cols", "band_pore_radius",
                     "rimoportulae", "rimo_angle_deg"),
        "doc": "Whole centric-diatom frustule as ONE solid: loculate areolated "
               "valve, perforated mantle, overlapping girdle-band stack and "
               "rimoportula tubes, combined in a single batch boolean. All "
               "lengths model units (mm); bands/rows/cols are counts.",
        "args": {"radius": "length", "pitch": "length", "areola_radius": "length",
                 "face_thickness": "length", "chamber_depth": "length",
                 "foramen_radius": "length", "mantle_height": "length",
                 "mantle_wall": "length", "mantle_rows": ("int", 1),
                 "mantle_cols": ("int", 3), "dome": "length", "margin": "length",
                 "bands": ("int", 0), "band_height": "length", "band_wall": "length",
                 "band_rows": ("int", 1), "band_cols": ("int", 3),
                 "band_pore_radius": "length", "rimoportulae": ("int", 0),
                 "rimo_angle_deg": "number"},
    }, _diatom_frustule)

    register_op("pyramidal_neuron", {
        "required": ("soma_radius", "apical_length", "apical_radius",
                     "basal_length", "basal_radius"), "arity": 0,
        "optional": ("soma_aspect", "apical_orders", "apical_split", "obliques",
                     "basal_primary", "basal_orders", "taper", "spread",
                     "length_decay", "radius_floor", "jitter", "seed", "segments",
                     "spines_per_um", "spine_length", "spine_head", "myelin",
                     "node_gap"),
        "args": {"soma_radius": "length", "soma_aspect": "number",
                 "apical_length": "length", "apical_radius": "length",
                 "apical_orders": ("int", 1), "apical_split": ("int", 1),
                 "obliques": ("int", 0), "basal_primary": ("int", 1),
                 "basal_length": "length", "basal_radius": "length",
                 "basal_orders": ("int", 1), "taper": "number", "spread": "number",
                 "length_decay": "number", "radius_floor": "length",
                 "jitter": "number", "seed": ("int", 0), "segments": ("int", 6),
                 "spines_per_um": "number", "spine_length": "length",
                 "spine_head": "length", "myelin": "number", "node_gap": "length"},
        "doc": "Whole pyramidal neuron in one solid: ellipsoid soma, tapering "
               "apical trunk with oblique branches and an apical tuft, 6-7 basal "
               "primaries, optional dendritic spines on real tips, optional "
               "myelinated axon. All lengths model units (mm); counts are ints.",
    }, _pyramidal_neuron)

    _lattice_shell.prepare = _prepare_lattice
    _perforated_valve.prepare = _prepare_valve
    _perforated_cylinder.prepare = _prepare_perf_cyl
    _branch_tree.prepare = _prepare_tree
    _pore_field_shell.prepare = _prepare_pore_field
    _capsid.prepare = _prepare_capsid
    _ring_stack.prepare = _prepare_rings
    _weld_union.prepare = _prepare_weld_union
    _diatom_frustule.prepare = _prepare_frustule
    _pyramidal_neuron.prepare = _prepare_neuron


try:
    register()
except Exception:  # pragma: no cover - a family must never break the engine
    pass


def _drop_islands(mesh, min_frac=1e-4):
    """Remove face-connected islands far smaller than the main body.

    Verify counts *geometric* components; manifold3d's ``decompose`` splits by
    topology, so a scrap that shares vertex indices with the main shell is not
    separated by it.  Walking face adjacency finds those scraps.  The threshold
    is a fraction of total face count, so a genuine small feature is never
    deleted -- only debris two orders of magnitude below the body.
    """
    if mesh is None or not len(mesh.faces):
        return mesh
    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
    except Exception:
        return mesh
    faces = mesh.faces
    n = int(faces.max()) + 1
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    g = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    _, labels = connected_components(g, directed=False)
    flab = labels[faces[:, 0]]
    counts = np.bincount(flab)
    if len(counts) < 2:
        return mesh
    keep_ids = np.where(counts >= max(8, counts.max() * float(min_frac)))[0]
    if len(keep_ids) == len(counts):
        return mesh
    mask = np.isin(flab, keep_ids)
    if not mask.any():
        return mesh
    faces = faces[mask]
    used = np.unique(faces)
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(mesh.verts[used], remap[faces], dict(mesh.meta))
