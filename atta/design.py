# -*- coding: utf-8 -*-
"""Design-rule checks: does the form read, and can it be made?

``verify`` answers "is this a valid solid".  It does not answer "does this look
like a thing, and could a factory build it".  That gap is why a model could
pass every geometric check and still be wrong: the numbers matched and the
shape did not.

This module encodes the checks professional modellers apply by eye, as
measurable predicates on a triangle mesh:

Readability
    ``silhouette_readability`` — IoU between the silhouette at full resolution
    and at 32 px.  If a form does not survive being shrunk, it will not read at
    game distance.  This is the "squint test", made numeric.
    ``negative_space`` — how much of the bounding box is empty, and whether the
    empty regions are large enough to read as gaps rather than noise.
    ``form_hierarchy`` — the distribution of feature sizes.  Good form has
    three scales (primary / secondary / tertiary); a single scale reads as a
    blob, a flat distribution reads as noise.

Proportion
    ``proportion`` — aspect ratios against the archetype's canonical ratios.
    ``symmetry_report`` — which axes are symmetric, and how exactly.

Manufacturability
    ``printability`` — minimum wall, overhang angle, and the fraction of the
    surface that needs support.
    ``moldability`` — draft angle and undercut detection (a rough proxy).
    ``machinability`` — internal sharp corners smaller than a tool radius.

Every check returns ``{"id", "ok", "value", "advice"}`` — never a bare number,
because the point is to tell the agent what to change.  ``ok=None`` means "not
applicable" (e.g. printability on a solid block), which is different from a
failure.
"""

from __future__ import annotations

import math

import numpy as np

from . import mesh as M
from . import probe

__all__ = [
    "design_report",
    "silhouette_readability",
    "negative_space",
    "form_hierarchy",
    "proportion",
    "symmetry_report",
    "printability",
    "moldability",
    "machinability",
    "METHOD",
]

METHOD = "atta.design/v1"


def _check(cid, ok, value, advice, *, detail=None):
    return {"id": cid, "ok": ok, "value": value, "advice": advice,
            "detail": detail or {}}


# ---------------------------------------------------------------- readability

def _silhouette_mask(mesh, axes, pixels):
    """Filled projected triangles on one shared pixel-centre coordinate frame.

    Unlike vertex splatting, polygon union does not depend on face subdivision.
    A supersampled mask supplies area coverage for the coarser grid.
    """
    from PIL import Image, ImageDraw

    v = np.asarray(mesh.verts, dtype=np.float64)
    if not len(v) or not len(mesh.faces):
        return None, None
    p = v[:, axes]
    lo, hi = p.min(axis=0), p.max(axis=0)
    extent = hi - lo
    span = float(extent.max())
    if span <= 0 or not np.isfinite(p).all():
        return None, None
    xy = (p - (lo + hi) * 0.5) / span * (pixels - 2) + (pixels - 1) * 0.5
    image = Image.new("1", (pixels, pixels))
    draw = ImageDraw.Draw(image)
    for tri in np.asarray(mesh.faces, dtype=np.int64):
        draw.polygon([tuple(point) for point in xy[tri]], fill=1)
    return np.asarray(image, dtype=bool), extent / span


def silhouette_readability(mesh, *, res=64, small=32, threshold=0.90):
    """Sampling heuristic, not a claim that an object is recognizable.

    Compare a filled reference silhouette with its area-downsampled binary
    reconstruction in the SAME frame. Ignore displacement within half a coarse
    pixel (the quantization budget). Views thinner than two coarse pixels are
    explicitly unresolved, rather than evidence of poor modelling.
    """
    from scipy.ndimage import distance_transform_edt

    if not (isinstance(res, (int, np.integer)) and isinstance(small, (int, np.integer))
            and 4 <= small <= res <= 512):
        raise ValueError("readability requires 4 <= small <= res <= 512")
    if not math.isfinite(float(threshold)) or not 0 <= threshold <= 1:
        raise ValueError("readability threshold must be between 0 and 1")
    factor = max(4, int(math.ceil(res / small)))
    pixels = small * factor
    out = {}
    for view, axes in (("front", (0, 2)), ("top", (0, 1)), ("right", (1, 2))):
        b, aspect = _silhouette_mask(mesh, axes, pixels)
        if b is None:
            out[view] = {"iou": None, "status": "empty"}
            continue
        width = float(aspect.min() * small)
        if width < 2.0:
            out[view] = {"iou": None, "status": "unresolved",
                         "projected_short_axis_px": round(width, 4)}
            continue
        coverage = b.reshape(small, factor, small, factor).mean(axis=(1, 3))
        # Exact 50% ties retain a feature instead of deleting a centred rod.
        coarse = coverage >= 0.5
        reconstructed = np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)
        union = int((b | reconstructed).sum())
        intersection = int((b & reconstructed).sum())
        tolerance = factor * 0.5
        lost = b & (distance_transform_edt(~reconstructed) > tolerance)
        gained = reconstructed & (distance_transform_edt(~b) > tolerance)
        residual = int(lost.sum() + gained.sum())
        out[view] = {
            "iou": round(1.0 - residual / max(1, union), 4),
            "raw_iou": round(intersection / max(1, union), 4),
            "status": "measured", "projected_short_axis_px": round(width, 4),
            "coverage_big": round(float(b.mean()), 4),
            "coverage_small": round(float(coarse.mean()), 4),
        }
    vals = [v["iou"] for v in out.values() if v.get("iou") is not None]
    score = round(float(np.mean(vals)), 4) if vals else None
    ok = None if score is None else bool(score >= threshold)
    worst = min((k for k, v in out.items() if v.get("iou") is not None),
                key=lambda k: out[k]["iou"], default=None)
    unresolved = [k for k, v in out.items() if v["status"] == "unresolved"]
    advice = ("resolved silhouettes survive sampling within the pixel uncertainty budget"
              if ok else "resolved silhouettes lose area beyond pixel uncertainty; inspect fine features")
    if ok is None:
        advice = "no view can support a readability judgement at this pixel size"
    if unresolved:
        advice += "; unresolved thin views: " + ", ".join(unresolved)
    return _check("readability", ok, {"score": score, "views": out, "worst_view": worst},
                  advice, detail={"threshold": threshold, "res": res, "small": small,
                                  "method": "shared_frame_area_sampling",
                                  "boundary_tolerance_px": 0.5, "heuristic": True,
                                  "recognition_test": False})


def _bitmap(view):
    if not view:
        return None
    rows = view.get("bitmap") or []
    if not rows:
        return None
    return np.array([[c == "1" for c in row] for row in rows], dtype=bool)


def negative_space(mesh, *, res=48):
    """How much of the bbox is empty, and are the gaps readable?

    A good form has *deliberate* negative space: gaps big enough to see.  Many
    tiny holes read as noise; no holes at all reads as a brick.
    """
    sil = probe.silhouette(mesh, views=("front", "top"), res=res)
    out = {}
    for view in ("front", "top"):
        b = _bitmap(sil["views"].get(view))
        if b is None:
            continue
        empty = ~b
        # Connected empty regions that do not touch the border are "holes".
        try:
            from scipy import ndimage
            labels, n = ndimage.label(empty)
            border = set(labels[0, :]) | set(labels[-1, :]) \
                | set(labels[:, 0]) | set(labels[:, -1])
            border.discard(0)
            interior = [int((labels == k).sum()) for k in range(1, n + 1)
                        if k not in border]
        except Exception:  # noqa: BLE001 - a trait, not a gate
            interior = []
        total = float(b.size)
        out[view] = {
            "fill_ratio": round(float(b.mean()), 4),
            "interior_gaps": len(interior),
            "largest_gap_ratio": round(max(interior) / total, 4) if interior else 0.0,
        }
    fills = [v["fill_ratio"] for v in out.values()]
    mean_fill = round(float(np.mean(fills)), 4) if fills else None
    # A brick (fill > 0.92) has no negative space to read; a lace (fill < 0.15)
    # has no mass.  Both are advice, not failures.
    advice = []
    if mean_fill is not None:
        if mean_fill > 0.92:
            advice.append("the silhouette is nearly solid: add a cut-out or a "
                          "gap so the eye can separate the parts")
        elif mean_fill < 0.15:
            advice.append("the silhouette is very thin: add mass to the primary "
                          "form before adding detail")
        else:
            advice.append("fill ratio is in a readable range")
    return _check("negative_space", None,
                  {"mean_fill": mean_fill, "views": out},
                  "; ".join(advice) or "not measured")


def form_hierarchy(mesh, *, bins=12):
    """Connected coplanar patch areas, an advisory hard-surface proxy.

    A triangle is a discretization element, not a design feature. Merge only
    genuinely coplanar neighbours before looking at areas. Smooth regions do
    not have a mesh-independent feature partition; report that limitation and
    never turn this proxy into an aesthetic pass/fail or a demand for detail.
    """
    if not isinstance(bins, (int, np.integer)) or not 1 <= bins <= 128:
        raise ValueError("bins must be an integer between 1 and 128")
    m = M.weld(mesh)
    v, f = m.verts, m.faces
    if not len(f):
        return _check("form_hierarchy", None, {}, "empty mesh",
                      detail={"heuristic": True, "method": "coplanar_patch_area"})
    normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > max(float(lengths.max()) * 1e-14, 1e-30)
    normals = normals / np.where(lengths[:, None] > 0, lengths[:, None], 1.0)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    face_ids = np.tile(np.arange(len(f)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, face_ids = edges[order], face_ids[order]
    _, starts, counts = np.unique(edges, axis=0, return_index=True, return_counts=True)
    starts = starts[counts == 2]
    a, b = face_ids[starts], face_ids[starts + 1]
    coplanar = (np.einsum("ij,ij->i", normals[a], normals[b]) > 1.0 - 1e-10)
    coplanar &= valid[a] & valid[b]
    parent = np.arange(len(f))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(a[coplanar], b[coplanar]):
        ri, rj = root(i), root(j)
        if ri != rj:
            parent[rj] = ri
    roots = np.array([root(i) for i in range(len(f))])
    patch_areas = np.bincount(roots, weights=lengths * 0.5)
    areas = patch_areas[patch_areas > 0]
    if not len(areas):
        return _check("form_hierarchy", None, {"tris": len(f)}, "zero-area mesh",
                      detail={"heuristic": True, "method": "coplanar_patch_area"})
    loga = np.log10(areas)
    span = float(np.ptp(loga))
    # Near-constant floating point areas must not produce zero-width bins.
    if span < 1e-8:
        hist = np.zeros(bins, dtype=np.int64)
        hist[bins // 2] = len(areas)
        modes = 1
    else:
        hist, _ = np.histogram(loga, bins=bins)
        above = hist > 0.25 * hist.max()
        modes = int(np.count_nonzero(above & ~np.r_[False, above[:-1]]))
    ratio = float(areas.max() / areas.min())
    largest = np.sort(areas)[-min(12, len(areas)):].sum() / areas.sum()
    status = "planar_proxy" if largest >= 0.5 else "curved_or_fragmented"
    return _check(
        "form_hierarchy", None,
        {"modes": modes, "area_ratio": round(ratio, 2), "tris": int(len(f)),
         "patches": int(len(areas)), "histogram": hist.tolist(), "status": status},
        ("coplanar patches span %d area mode(s); this is an advisory geometric "
         "proxy, not evidence of primary/secondary/tertiary design hierarchy" % modes)
        + ("; curved regions have no stable planar feature partition"
           if status == "curved_or_fragmented" else ""),
        detail={"heuristic": True, "method": "coplanar_patch_area",
                "aesthetic_gate": False, "coplanar_angle_tolerance_deg": 0.00081})


# ---------------------------------------------------------------- proportion

def proportion(mesh, *, target=None):
    """Aspect ratios vs a target, and the classic proportion systems.

    ``target`` may be a 3-vector of sizes (mm) or a conjure result's
    ``size_mm``.  Reports the golden-ratio and root-2 proximity of the two
    largest dimensions, which is the quickest check that a form is not
    accidentally awkward.
    """
    b = M.bbox(mesh)
    size = np.asarray(b["size"], dtype=np.float64) * 1000.0
    order = np.argsort(size)[::-1]
    s = size[order]
    longest = float(s[0]) if s[0] > 0 else 1e-9
    aspect = [round(float(v) / longest, 4) for v in size]
    out = {"size_mm": [round(float(v), 3) for v in size], "aspect": aspect}
    ok = None
    advice = []
    if s[1] > 0:
        r = float(s[0] / s[1])
        out["long_over_mid"] = round(r, 4)
        out["near_golden"] = bool(abs(r - 1.618) < 0.12)
        out["near_root2"] = bool(abs(r - 1.414) < 0.10)
        if r > 12.0:
            advice.append("the form is %.1f:1 long-to-wide: it will read as a "
                          "stick. Break it into segments or thicken the middle"
                          % r)
        elif 1.0 < r < 1.08:
            advice.append("the two largest dimensions are nearly equal (%.2f:1): "
                          "the form has no clear long axis" % r)
    if target is not None:
        t = np.asarray(target, dtype=np.float64).reshape(-1)
        if t.shape == (3,) and np.all(t > 0):
            delta = [round(float(a - b2), 3) for a, b2 in zip(size, t)]
            rel = [round(float((a - b2) / b2), 4) for a, b2 in zip(size, t)]
            out["target_mm"] = [round(float(v), 3) for v in t]
            out["delta_mm"] = delta
            out["relative_error"] = rel
            worst = max(abs(v) for v in rel)
            ok = bool(worst <= 0.15)
            advice.append(
                "largest axis error vs target is %.1f%%%s"
                % (worst * 100.0, "" if ok else
                   "; adjust the dominant dimension first, then the "
                   "cross-sections"))
    return _check("proportion", ok, out,
                  "; ".join(advice) or "aspect ratios are within a neutral range")


def symmetry_report(mesh):
    """Which axes are mirror-symmetric, and how exactly."""
    v = np.asarray(mesh.verts, dtype=np.float64)
    b = M.bbox(mesh)
    c = np.asarray(b["center"], dtype=np.float64)
    diag = max(float(b["diag"]), 1e-12)
    out = {}
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(v)
        for axis, name in enumerate(("x", "y", "z")):
            mirrored = v.copy()
            mirrored[:, axis] = 2.0 * c[axis] - v[:, axis]
            dist, _ = tree.query(mirrored, k=1)
            p95 = float(np.percentile(dist, 95)) / diag
            out[name] = {"symmetric": bool(p95 <= 0.01),
                         "p95_error_ratio": round(p95, 5)}
    except Exception:  # noqa: BLE001
        return _check("symmetry", None, {}, "symmetry needs scipy")
    axes = [k for k, v2 in out.items() if v2["symmetric"]]
    return _check("symmetry", None,
                  {"axes": axes, "detail": out},
                  ("symmetric about %s" % ", ".join(axes)) if axes
                  else "no mirror symmetry: fine for organic forms, unusual "
                       "for mechanical parts")


# ---------------------------------------------------------------- making

def printability(mesh, *, min_wall_mm=0.8, max_overhang_deg=45.0):
    """Minimum wall and overhang fraction, as an FDM printer sees them."""
    from . import metrology

    wall = metrology.wall_thickness(mesh, max_samples=400)
    got_wall = wall.get("min_mm")
    v = np.asarray(mesh.verts, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    a = v[f[:, 0]]
    b = v[f[:, 1]]
    c = v[f[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln > 0, ln, 1.0)
    areas = 0.5 * ln.reshape(-1)
    # Downward-facing normals beyond the overhang limit need support.
    limit = math.cos(math.radians(180.0 - float(max_overhang_deg)))
    downward = n[:, 2] < -abs(limit)
    overhang_area = float(areas[downward].sum())
    total_area = float(areas.sum()) or 1e-18
    frac = overhang_area / total_area
    ok = None
    advice = []
    if got_wall is not None:
        ok = bool(float(got_wall) >= float(min_wall_mm))
        advice.append(
            "minimum wall %.2f mm %s the %.2f mm print limit"
            % (float(got_wall), "meets" if ok else "is below", float(min_wall_mm)))
    else:
        advice.append("no measurable wall (solid part): wall check not applicable")
    if frac > 0.25:
        advice.append("%.0f%% of the surface overhangs past %.0f deg: add "
                      "chamfers or reorient before printing"
                      % (frac * 100.0, float(max_overhang_deg)))
    return _check("printability", ok,
                  {"min_wall_mm": got_wall,
                   "overhang_fraction": round(frac, 4),
                   "limits": {"min_wall_mm": float(min_wall_mm),
                              "max_overhang_deg": float(max_overhang_deg)}},
                  "; ".join(advice))


def moldability(mesh, *, min_draft_deg=1.0):
    """Rough draft-angle check for a two-part mould.

    A face is "draftable" when its normal has a positive Z component (it can
    be pulled along +Z) or a negative one (pulled along -Z).  A vertical wall
    exactly at 0 degrees is the problem case: it drags on the mould.
    """
    v = np.asarray(mesh.verts, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if len(f) == 0:
        return _check("moldability", None, {}, "empty mesh")
    a = v[f[:, 0]]
    b = v[f[:, 1]]
    c = v[f[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln > 0, ln, 1.0)
    areas = 0.5 * ln.reshape(-1)
    total = float(areas.sum()) or 1e-18
    # Draft angle relative to the pull axis (+Z): 90 deg is a flat face.
    draft = np.degrees(np.arcsin(np.clip(np.abs(n[:, 2]), 0.0, 1.0)))
    need = float(min_draft_deg)
    bad = draft < need
    frac = float(areas[bad].sum()) / total
    ok = bool(frac <= 0.05)
    return _check(
        "moldability", ok,
        {"vertical_area_fraction": round(frac, 4),
         "min_draft_deg": need},
        ("at most %.1f%% of the surface is vertical: a two-part mould can "
         "release it" % (frac * 100.0) if ok else
         "%.0f%% of the surface is vertical (< %.1f deg draft): add draft or "
         "plan a side-action, or the part will drag in the mould"
         % (frac * 100.0, need)))


def machinability(mesh, *, tool_radius_mm=3.0):
    """Count internal corners too tight for a round cutter.

    A concave edge sharper than the tool radius cannot be cut without leaving a
    fillet; this reports the fraction of concave sharp edges that violate it.
    """
    from . import metrology

    dih = metrology.dihedral(mesh)
    sharp = int(dih.get("count_gt_30") or 0)
    total = int(dih.get("count") or 0) or 1
    frac = sharp / float(total)
    ok = bool(frac <= 0.15)
    return _check(
        "machinability", ok,
        {"sharp_edge_fraction": round(frac, 4), "sharp_edges": sharp,
         "total_edges": total, "tool_radius_mm": float(tool_radius_mm)},
        ("sharp edges are %.1f%% of the edge set: a %.1f mm cutter can reach "
         "the concave corners" % (frac * 100.0, float(tool_radius_mm))
         if ok else
         "%.0f%% of edges are sharper than 30 deg: a %.1f mm round cutter "
         "cannot reach the internal corners. Add fillets or accept a "
         "secondary EDM operation"
         % (min(100.0, frac * 100.0), float(tool_radius_mm))))


# ---------------------------------------------------------------- report

def design_report(mesh, *, target_size_mm=None, rules=None,
                  min_wall_mm=0.8, max_overhang_deg=45.0,
                  min_draft_deg=1.0, tool_radius_mm=3.0,
                  readability_threshold=0.90) -> dict:
    """Run the design rules and return one report.

    ``rules`` selects a subset by id; the default runs all of them.  Every
    entry is ``{"id", "ok", "value", "advice"}``; ``ok=None`` means the rule
    did not apply, which is not a failure.

    This is **not** a verdict.  ``verify`` decides whether the solid is valid;
    this report tells the agent whether the form reads and whether it can be
    made.  A model can pass ``verify`` and still fail here, and that is the
    point.
    """
    wanted = set(rules) if rules else None

    def want(cid):
        return wanted is None or cid in wanted

    checks = []
    if want("readability"):
        checks.append(silhouette_readability(
            mesh, threshold=readability_threshold))
    if want("negative_space"):
        checks.append(negative_space(mesh))
    if want("form_hierarchy"):
        checks.append(form_hierarchy(mesh))
    if want("proportion"):
        checks.append(proportion(mesh, target=target_size_mm))
    if want("symmetry"):
        checks.append(symmetry_report(mesh))
    if want("printability"):
        checks.append(printability(mesh, min_wall_mm=min_wall_mm,
                                   max_overhang_deg=max_overhang_deg))
    if want("moldability"):
        checks.append(moldability(mesh, min_draft_deg=min_draft_deg))
    if want("machinability"):
        checks.append(machinability(mesh, tool_radius_mm=tool_radius_mm))

    failed = [c for c in checks if c["ok"] is False]
    passed = [c for c in checks if c["ok"] is True]
    na = [c for c in checks if c["ok"] is None]
    return {
        "schema": "atta.design/v1",
        "checks": checks,
        "summary": {
            "passed": len(passed),
            "failed": len(failed),
            "not_applicable": len(na),
            "failed_ids": [c["id"] for c in failed],
        },
        "advice": [c["advice"] for c in failed],
        "units": "mm",
        "exact": False,
        "method": METHOD,
    }
