# -*- coding: utf-8 -*-
"""2D engineering drawing projection and DXF / SVG sheet export.

Uses OCCT hidden-line removal (``HLRBRep_Algo`` + ``HLRBRep_HLRToShape`` +
``HLRAlgo_Projector``) when an analytic B-rep ``shape`` is provided
(``backend="occt_hlr"``), and a deterministic pure-numpy crease + silhouette
edge projection with 2D-barycentric depth occlusion when ``shape is None``
(``backend="mesh_hlr"``).

All projected coordinates are in millimetres, using the same view basis as
:mod:`atta.probe` so orthographic views align in third-angle projection.
"""

from __future__ import annotations

import math
import os
from xml.sax.saxutils import escape as _xml_escape

import numpy as np

from . import mesh as M
from . import probe
from .errors import AttaError

__all__ = ["project_drawing", "export_dxf", "export_svg"]

DEFAULT_VIEWS = ("front", "top", "right", "iso")
_UNIT_SCALE_FROM_MM = {"mm": 1.0, "cm": 0.1, "m": 0.001, "in": 1.0 / 25.4}


def _normalize_views(views) -> tuple[str, ...]:
    if views is None:
        return DEFAULT_VIEWS
    if isinstance(views, str):
        items = tuple(x.strip().lower() for x in views.split(",") if x.strip())
    else:
        items = tuple(str(x).strip().lower() for x in views if str(x).strip())
    if not items:
        raise AttaError("bad_arg", "views must contain at least one view name",
                        path="views", fix="use views=('front','top','right','iso')")
    for v in items:
        if v not in probe._VIEW_DIRS:
            raise AttaError("bad_arg", "unknown drawing view %r" % v, path="views",
                            fix="valid views: " + ", ".join(sorted(probe._VIEW_DIRS)))
    return items


def _bbox_mm(mesh: M.Mesh | None, shape) -> dict:
    if mesh is not None:
        b = M.bbox(mesh)
        return {
            "min": [round(float(x) * 1000.0, 6) for x in b["min"]],
            "max": [round(float(x) * 1000.0, 6) for x in b["max"]],
            "size": [round(float(x) * 1000.0, 6) for x in b["size"]],
            "center": [round(float(x) * 1000.0, 6) for x in b["center"]],
        }
    bb = shape.bounding_box()
    lo = [round(float(x), 6) for x in tuple(bb.min)]
    hi = [round(float(x), 6) for x in tuple(bb.max)]
    size = [round(hi[i] - lo[i], 6) for i in range(3)]
    center = [round(0.5 * (lo[i] + hi[i]), 6) for i in range(3)]
    return {"min": lo, "max": hi, "size": size, "center": center}


def _curve_signature(item: dict) -> tuple:
    kind = item["kind"]
    pts = item["points"]
    if kind == "circle":
        cx, cy = item["center"]
        return ("circle", round(cx, 4), round(cy, 4), round(item["radius"], 4))
    if kind == "arc":
        cx, cy = item["center"]
        return ("arc", round(cx, 4), round(cy, 4), round(item["radius"], 4),
                round(item["start_deg"] % 360.0, 3), round(item["end_deg"] % 360.0, 3))
    p0 = (round(pts[0][0], 4), round(pts[0][1], 4))
    p1 = (round(pts[-1][0], 4), round(pts[-1][1], 4))
    pm = (round(pts[len(pts) // 2][0], 4), round(pts[len(pts) // 2][1], 4))
    return (kind, min(p0, p1), max(p0, p1), pm, len(pts))


def _sort_key(item: dict):
    pts = item["points"]
    p0 = (round(pts[0][0], 6), round(pts[0][1], 6))
    p1 = (round(pts[-1][0], 6), round(pts[-1][1], 6))
    ends = min((p0, p1), (p1, p0))
    return (item["kind"], ends, len(pts))


def _view_bounds(visible: list[dict], hidden: list[dict]) -> dict:
    pts = [pt for item in (visible + hidden) for pt in item["points"]]
    if not pts:
        return {"min": [0.0, 0.0], "max": [0.0, 0.0], "size": [0.0, 0.0], "center": [0.0, 0.0]}
    arr = np.asarray(pts, dtype=np.float64)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    sz = hi - lo
    c = 0.5 * (lo + hi)
    return {
        "min": [round(float(lo[0]), 6), round(float(lo[1]), 6)],
        "max": [round(float(hi[0]), 6), round(float(hi[1]), 6)],
        "size": [round(float(sz[0]), 6), round(float(sz[1]), 6)],
        "center": [round(float(c[0]), 6), round(float(c[1]), 6)],
    }


def _extract_occt_edges(compound, samples_per_curve: int = 24) -> list[dict]:
    if compound is None or compound.IsNull():
        return []
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepLib import BRepLib
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    cast_edge = getattr(TopoDS, "Edge_s", None) or TopoDS.Edge
    out = []
    exp = TopExp_Explorer(compound, TopAbs_EDGE)
    while exp.More():
        edge = cast_edge(exp.Current())
        exp.Next()
        BRepLib.BuildCurve3d_s(edge)
        c = BRepAdaptor_Curve(edge)
        u0, u1 = float(c.FirstParameter()), float(c.LastParameter())
        if not (math.isfinite(u0) and math.isfinite(u1)) or abs(u1 - u0) <= 1e-12:
            continue
        ctype = c.GetType().name
        if ctype == "GeomAbs_Line":
            p0 = c.Value(u0)
            p1 = c.Value(u1)
            x0, y0 = round(float(p0.X()), 6), round(float(p0.Y()), 6)
            x1, y1 = round(float(p1.X()), 6), round(float(p1.Y()), 6)
            if math.hypot(x1 - x0, y1 - y0) <= 1e-6:
                continue
            out.append({"kind": "line", "points": [[x0, y0], [x1, y1]]})
        elif ctype == "GeomAbs_Circle":
            circ = c.Circle()
            cx = round(float(circ.Location().X()), 6)
            cy = round(float(circ.Location().Y()), 6)
            r = round(float(circ.Radius()), 6)
            if r <= 1e-6:
                continue
            span = abs(u1 - u0)
            is_full = abs(span - 2.0 * math.pi) <= 1e-5
            n = max(24, samples_per_curve * 2) if is_full else max(12, int(round(samples_per_curve * span / math.pi)))
            us = np.linspace(u0, u1, n + 1)
            pts = [[round(float(c.Value(float(u)).X()), 6), round(float(c.Value(float(u)).Y()), 6)] for u in us]
            if is_full:
                out.append({"kind": "circle", "center": [cx, cy], "radius": r, "points": pts})
            else:
                p0 = c.Value(u0)
                p1 = c.Value(u1)
                a0 = math.degrees(math.atan2(float(p0.Y()) - cy, float(p0.X()) - cx))
                a1 = math.degrees(math.atan2(float(p1.Y()) - cy, float(p1.X()) - cx))
                ccw = float(circ.Axis().Direction().Z()) >= 0.0
                start_deg = round(a0 if ccw else a1, 6)
                end_deg = round(a1 if ccw else a0, 6)
                out.append({
                    "kind": "arc",
                    "center": [cx, cy],
                    "radius": r,
                    "start_deg": start_deg,
                    "end_deg": end_deg,
                    "points": pts,
                })
        else:
            n = max(16, samples_per_curve)
            us = np.linspace(u0, u1, n + 1)
            pts = [[round(float(c.Value(float(u)).X()), 6), round(float(c.Value(float(u)).Y()), 6)] for u in us]
            out.append({"kind": "polyline", "points": pts})
    return out


def _project_occt(shape, views: tuple[str, ...], include_hidden: bool) -> dict:
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.HLRAlgo import HLRAlgo_Projector
    from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape

    raw = shape.wrapped if hasattr(shape, "wrapped") else shape
    views_out = {}
    for view in views:
        d = np.asarray(probe._VIEW_DIRS[view], dtype=np.float64)
        d /= np.linalg.norm(d)
        right, _up = probe._basis(view)
        ax2 = gp_Ax2(
            gp_Pnt(0.0, 0.0, 0.0),
            gp_Dir(float(-d[0]), float(-d[1]), float(-d[2])),
            gp_Dir(float(right[0]), float(right[1]), float(right[2])),
        )
        hlr = HLRBRep_Algo()
        hlr.Add(raw)
        hlr.Projector(HLRAlgo_Projector(ax2))
        hlr.Update()
        hlr.Hide()
        ts = HLRBRep_HLRToShape(hlr)

        vis_raw = []
        for comp in (ts.VCompound(), ts.OutLineVCompound(), ts.Rg1LineVCompound()):
            vis_raw.extend(_extract_occt_edges(comp))
        vis, seen_vis = [], set()
        for item in vis_raw:
            sig = _curve_signature(item)
            if sig not in seen_vis:
                seen_vis.add(sig)
                vis.append(item)
        vis.sort(key=_sort_key)

        hid = []
        if include_hidden:
            hid_raw = []
            for comp in (ts.HCompound(), ts.OutLineHCompound(), ts.Rg1LineHCompound()):
                hid_raw.extend(_extract_occt_edges(comp))
            seen_hid = set()
            for item in hid_raw:
                sig = _curve_signature(item)
                if sig in seen_vis or sig in seen_hid:
                    continue
                seen_hid.add(sig)
                hid.append(item)
            hid.sort(key=_sort_key)

        views_out[view] = {
            "view": view,
            "visible": vis,
            "hidden": hid,
            "bounds_mm": _view_bounds(vis, hid),
        }
    return views_out


def _mesh_topology_edges(verts_mm: np.ndarray, faces: np.ndarray, crease_deg: float = 20.0):
    if len(faces) == 0:
        return np.zeros((0, 3)), [], []
    a = verts_mm[faces[:, 0]]
    b = verts_mm[faces[:, 1]]
    c = verts_mm[faces[:, 2]]
    raw_n = np.cross(b - a, c - a)
    norms = np.linalg.norm(raw_n, axis=1)
    safe = np.where(norms > 1e-12, norms, 1.0)
    normals = raw_n / safe[:, None]
    normals[norms <= 1e-12] = 0.0

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, tri in enumerate(faces):
        for u, v in ((int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))):
            key = (u, v) if u < v else (v, u)
            edge_faces.setdefault(key, []).append(fi)

    cos_thresh = math.cos(math.radians(crease_deg))
    creases = []
    smooth = []
    for (u, v), adj in edge_faces.items():
        if len(adj) != 2:
            creases.append((u, v, adj))
        else:
            dot = float(np.dot(normals[adj[0]], normals[adj[1]]))
            if dot <= cos_thresh:
                creases.append((u, v, adj))
            else:
                smooth.append((u, v, adj))
    return normals, creases, smooth


def _occluded_mask(pts_uvz: np.ndarray, tri_uvz: np.ndarray, tol: float) -> np.ndarray:
    """Return boolean mask of which 2D+depth points lie strictly behind a front triangle."""
    n_pts = len(pts_uvz)
    if n_pts == 0 or len(tri_uvz) == 0:
        return np.zeros(n_pts, dtype=bool)
    p0 = tri_uvz[:, 0, :]  # (T, 3)
    p1 = tri_uvz[:, 1, :]
    p2 = tri_uvz[:, 2, :]
    tri_umin = np.minimum(np.minimum(p0[:, 0], p1[:, 0]), p2[:, 0])
    tri_umax = np.maximum(np.maximum(p0[:, 0], p1[:, 0]), p2[:, 0])
    tri_vmin = np.minimum(np.minimum(p0[:, 1], p1[:, 1]), p2[:, 1])
    tri_vmax = np.maximum(np.maximum(p0[:, 1], p1[:, 1]), p2[:, 1])
    tri_zmax = np.maximum(np.maximum(p0[:, 2], p1[:, 2]), p2[:, 2])

    v0 = p1[:, :2] - p0[:, :2]
    v1 = p2[:, :2] - p0[:, :2]
    denom = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    valid_tri = np.abs(denom) > 1e-10

    occ = np.zeros(n_pts, dtype=bool)
    chunk = 256
    for start in range(0, n_pts, chunk):
        pts = pts_uvz[start:start + chunk]
        pu = pts[:, 0][:, None]
        pv = pts[:, 1][:, None]
        pz = pts[:, 2][:, None]
        cand = (
            valid_tri[None, :]
            & (pu >= tri_umin[None, :] - 1e-7)
            & (pu <= tri_umax[None, :] + 1e-7)
            & (pv >= tri_vmin[None, :] - 1e-7)
            & (pv <= tri_vmax[None, :] + 1e-7)
            & (tri_zmax[None, :] > pz + tol)
        )
        pi, ti = np.where(cand)
        if len(pi) == 0:
            continue
        dp = pts[pi, :2] - p0[ti, :2]
        d = denom[ti]
        w1 = (dp[:, 0] * v1[ti, 1] - dp[:, 1] * v1[ti, 0]) / d
        w2 = (v0[ti, 0] * dp[:, 1] - v0[ti, 1] * dp[:, 0]) / d
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not inside.any():
            continue
        pi = pi[inside]
        ti = ti[inside]
        z_tri = w0[inside] * p0[ti, 2] + w1[inside] * p1[ti, 2] + w2[inside] * p2[ti, 2]
        behind = z_tri > pts[pi, 2] + tol
        if behind.any():
            occ[start + pi[behind]] = True
    return occ


def _project_mesh(mesh: M.Mesh, views: tuple[str, ...], include_hidden: bool) -> dict:
    verts_mm = np.asarray(mesh.verts, dtype=np.float64) * 1000.0
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals, creases, smooth = _mesh_topology_edges(verts_mm, faces)
    diag = float(np.linalg.norm(verts_mm.max(axis=0) - verts_mm.min(axis=0))) if len(verts_mm) else 1.0
    tol = max(1e-3, 1e-4 * diag)

    views_out = {}
    for view in views:
        d = np.asarray(probe._VIEW_DIRS[view], dtype=np.float64)
        d /= np.linalg.norm(d)
        right, up = probe._basis(view)
        view_dir = -d

        u_v = verts_mm @ right
        v_v = verts_mm @ up
        z_v = verts_mm @ view_dir
        uvz = np.column_stack([u_v, v_v, z_v])

        facing = normals @ view_dir
        front_tris = uvz[faces[facing > 1e-7]]

        cand_edges = list(creases)
        for u, v, adj in smooth:
            f0, f1 = adj[0], adj[1]
            s0, s1 = facing[f0] > 1e-9, facing[f1] > 1e-9
            if s0 != s1 and max(abs(facing[f0]), abs(facing[f1])) > 1e-7:
                cand_edges.append((u, v, adj))

        if not cand_edges:
            views_out[view] = {"view": view, "visible": [], "hidden": [], "bounds_mm": _view_bounds([], [])}
            continue

        n_sub = 4
        ts = (np.arange(n_sub, dtype=np.float64) + 0.5) / float(n_sub)
        u_idx = np.array([e[0] for e in cand_edges], dtype=np.int64)
        v_idx = np.array([e[1] for e in cand_edges], dtype=np.int64)
        pa = uvz[u_idx]  # (E, 3)
        pb = uvz[v_idx]  # (E, 3)
        sample_pts = (1.0 - ts[None, :, None]) * pa[:, None, :] + ts[None, :, None] * pb[:, None, :]
        occ = _occluded_mask(sample_pts.reshape(-1, 3), front_tris, tol).reshape(len(cand_edges), n_sub)

        vis_raw, hid_raw = [], []
        for ei in range(len(cand_edges)):
            row = occ[ei]
            a_uv = pa[ei, :2]
            b_uv = pb[ei, :2]
            if math.hypot(float(b_uv[0] - a_uv[0]), float(b_uv[1] - a_uv[1])) <= 1e-6:
                continue
            # Merge contiguous visible/hidden sub-segments along the edge.
            k = 0
            while k < n_sub:
                is_hid = bool(row[k])
                k_end = k + 1
                while k_end < n_sub and bool(row[k_end]) == is_hid:
                    k_end += 1
                t0 = k / float(n_sub)
                t1 = k_end / float(n_sub)
                q0 = (1.0 - t0) * a_uv + t0 * b_uv
                q1 = (1.0 - t1) * a_uv + t1 * b_uv
                seg = {
                    "kind": "line",
                    "points": [
                        [round(float(q0[0]), 6), round(float(q0[1]), 6)],
                        [round(float(q1[0]), 6), round(float(q1[1]), 6)],
                    ],
                }
                if is_hid:
                    hid_raw.append(seg)
                else:
                    vis_raw.append(seg)
                k = k_end

        vis, seen_vis = [], set()
        for item in vis_raw:
            sig = _curve_signature(item)
            if sig not in seen_vis:
                seen_vis.add(sig)
                vis.append(item)
        vis.sort(key=_sort_key)

        hid = []
        if include_hidden:
            seen_hid = set()
            for item in hid_raw:
                sig = _curve_signature(item)
                if sig in seen_vis or sig in seen_hid:
                    continue
                seen_hid.add(sig)
                hid.append(item)
            hid.sort(key=_sort_key)

        views_out[view] = {
            "view": view,
            "visible": vis,
            "hidden": hid,
            "bounds_mm": _view_bounds(vis, hid),
        }
    return views_out


def project_drawing(mesh: M.Mesh | None, shape=None,
                    views=("front", "top", "right", "iso"),
                    include_hidden: bool = True) -> dict:
    """Project 3D geometry into 2D engineering views (millimetres)."""
    if mesh is None and shape is None:
        raise AttaError("bad_arg", "project_drawing requires a mesh or B-rep shape",
                        path="mesh", fix="pass a Mesh or OCCT shape")
    vnames = _normalize_views(views)
    if shape is not None:
        backend = "occt_hlr"
        views_out = _project_occt(shape, vnames, bool(include_hidden))
    else:
        backend = "mesh_hlr"
        views_out = _project_mesh(mesh, vnames, bool(include_hidden))
    return {
        "ok": True,
        "backend": backend,
        "units": "mm",
        "bbox_mm": _bbox_mm(mesh, shape),
        "views": views_out,
    }


def _layout_views(drawing: dict) -> tuple[dict[str, tuple[float, float]], tuple[float, float, float, float]]:
    """Compute third-angle (or grid) 2D offsets ``{view: (dx, dy)}`` in mm."""
    views_dict = drawing["views"]
    names = list(views_dict.keys())
    sizes = [views_dict[v]["bounds_mm"]["size"] for v in names]
    max_dim = max([s[0] for s in sizes] + [s[1] for s in sizes] + [10.0])
    gap = max(20.0, 0.35 * max_dim)

    def bnd(name):
        return views_dict[name]["bounds_mm"]

    offsets: dict[str, tuple[float, float]] = {}
    if set(names) <= {"front", "top", "right", "iso"} and "front" in views_dict:
        # Third-angle orthographic layout: front bottom-left, top above front,
        # right to the right of front, iso top-right.
        fb = bnd("front")
        offsets["front"] = (-fb["min"][0], -fb["min"][1])
        front_w, front_h = fb["size"]
        if "top" in views_dict:
            tb = bnd("top")
            offsets["top"] = (-tb["min"][0], front_h + gap - tb["min"][1])
        if "right" in views_dict:
            rb = bnd("right")
            offsets["right"] = (front_w + gap - rb["min"][0], -rb["min"][1])
        if "iso" in views_dict:
            ib = bnd("iso")
            col_x = front_w + gap
            row_y = front_h + gap
            offsets["iso"] = (col_x - ib["min"][0], row_y - ib["min"][1])
    else:
        cols = max(1, int(math.ceil(math.sqrt(len(names)))))
        col_w = [0.0] * cols
        rows = (len(names) + cols - 1) // cols
        row_h = [0.0] * rows
        for idx, name in enumerate(names):
            r, c = divmod(idx, cols)
            w, h = bnd(name)["size"]
            col_w[c] = max(col_w[c], w)
            row_h[r] = max(row_h[r], h)
        x_starts = [0.0]
        for c in range(cols - 1):
            x_starts.append(x_starts[-1] + col_w[c] + gap)
        y_starts = [0.0]
        for r in range(rows - 1):
            y_starts.append(y_starts[-1] + row_h[r] + gap)
        for idx, name in enumerate(names):
            r, c = divmod(idx, cols)
            b = bnd(name)
            offsets[name] = (x_starts[c] - b["min"][0], y_starts[rows - 1 - r] - b["min"][1])

    all_x, all_y = [], []
    for name, (dx, dy) in offsets.items():
        b = bnd(name)
        all_x.extend([b["min"][0] + dx, b["max"][0] + dx])
        all_y.extend([b["min"][1] + dy, b["max"][1] + dy])
    sheet = (min(all_x), min(all_y), max(all_x), max(all_y))
    return offsets, sheet


def export_dxf(path: str, mesh: M.Mesh | None, shape=None,
               views=("front", "top", "right", "iso"),
               title: str = "", units: str = "mm") -> dict:
    """Write a layered 2D engineering drawing DXF via ``ezdxf``."""
    import ezdxf

    units_norm = str(units or "mm").lower()
    if units_norm not in _UNIT_SCALE_FROM_MM:
        raise AttaError("bad_arg", "units must be mm/cm/m/in", path="units", fix="use units='mm'")
    u_scale = _UNIT_SCALE_FROM_MM[units_norm]
    ez_units = {"mm": ezdxf.units.MM, "cm": ezdxf.units.CM, "m": ezdxf.units.M, "in": ezdxf.units.IN}[units_norm]

    drawing = project_drawing(mesh, shape=shape, views=views, include_hidden=True)
    offsets, (smin_x, smin_y, smax_x, smax_y) = _layout_views(drawing)
    span = max(smax_x - smin_x, smax_y - smin_y, 20.0)

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ez_units
    doc.layers.add("VISIBLE", color=7, linetype="CONTINUOUS", lineweight=35)
    doc.layers.add("HIDDEN", color=8, linetype="DASHED", lineweight=18)
    doc.layers.add("CENTER", color=1, linetype="CENTER", lineweight=18)
    doc.layers.add("ANNOTATION", color=3, linetype="CONTINUOUS", lineweight=25)
    msp = doc.modelspace()

    def draw_items(items, layer: str, dx: float, dy: float, add_centers: bool):
        for item in items:
            kind = item["kind"]
            if kind == "circle":
                cx = (item["center"][0] + dx) * u_scale
                cy = (item["center"][1] + dy) * u_scale
                r = item["radius"] * u_scale
                msp.add_circle((cx, cy), r, dxfattribs={"layer": layer})
                if add_centers:
                    ext = r * 1.25
                    msp.add_line((cx - ext, cy), (cx + ext, cy), dxfattribs={"layer": "CENTER"})
                    msp.add_line((cx, cy - ext), (cx, cy + ext), dxfattribs={"layer": "CENTER"})
            elif kind == "arc":
                cx = (item["center"][0] + dx) * u_scale
                cy = (item["center"][1] + dy) * u_scale
                r = item["radius"] * u_scale
                msp.add_arc((cx, cy), r, item["start_deg"], item["end_deg"], dxfattribs={"layer": layer})
                if add_centers and r > 0:
                    ext = r * 1.15
                    msp.add_line((cx - ext * 0.35, cy), (cx + ext * 0.35, cy), dxfattribs={"layer": "CENTER"})
                    msp.add_line((cx, cy - ext * 0.35), (cx, cy + ext * 0.35), dxfattribs={"layer": "CENTER"})
            elif kind == "line":
                p0, p1 = item["points"][0], item["points"][-1]
                msp.add_line(
                    ((p0[0] + dx) * u_scale, (p0[1] + dy) * u_scale),
                    ((p1[0] + dx) * u_scale, (p1[1] + dy) * u_scale),
                    dxfattribs={"layer": layer},
                )
            else:
                pts = [((p[0] + dx) * u_scale, (p[1] + dy) * u_scale) for p in item["points"]]
                if len(pts) >= 2:
                    msp.add_lwpolyline(pts, dxfattribs={"layer": layer})

    text_h = max(2.5, span * 0.025) * u_scale
    for vname, vdata in drawing["views"].items():
        dx, dy = offsets[vname]
        draw_items(vdata["hidden"], "HIDDEN", dx, dy, add_centers=False)
        draw_items(vdata["visible"], "VISIBLE", dx, dy, add_centers=True)
        b = vdata["bounds_mm"]
        lx = (0.5 * (b["min"][0] + b["max"][0]) + dx) * u_scale
        ly = (b["min"][1] + dy - max(6.0, span * 0.05)) * u_scale
        msp.add_text(
            vname.upper(),
            height=text_h,
            dxfattribs={"layer": "ANNOTATION", "insert": (lx, ly)},
        )

    # Overall dimensions on front (or first) view + title block.
    dims = drawing["bbox_mm"]["size"]
    dim_anchor_view = "front" if "front" in drawing["views"] else next(iter(drawing["views"]))
    ab = drawing["views"][dim_anchor_view]["bounds_mm"]
    adx, ady = offsets[dim_anchor_view]
    x0 = (ab["min"][0] + adx) * u_scale
    x1 = (ab["max"][0] + adx) * u_scale
    y0 = (ab["min"][1] + ady) * u_scale
    y1 = (ab["max"][1] + ady) * u_scale
    dim_off = max(8.0, span * 0.08) * u_scale
    if abs(x1 - x0) > 1e-6:
        msp.add_linear_dim(
            base=(0.5 * (x0 + x1), y1 + dim_off * 0.6),
            p1=(x0, y1),
            p2=(x1, y1),
            override={"dimtxt": text_h * 0.8},
            dxfattribs={"layer": "ANNOTATION"},
        ).render()
    if abs(y1 - y0) > 1e-6:
        msp.add_linear_dim(
            base=(x0 - dim_off * 0.8, 0.5 * (y0 + y1)),
            p1=(x0, y0),
            p2=(x0, y1),
            angle=90,
            override={"dimtxt": text_h * 0.8},
            dxfattribs={"layer": "ANNOTATION"},
        ).render()

    margin = max(15.0, span * 0.14)
    bx0 = (smin_x - margin) * u_scale
    by0 = (smin_y - margin * 1.5) * u_scale
    bx1 = (smax_x + margin) * u_scale
    by1 = (smax_y + margin) * u_scale
    msp.add_lwpolyline([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)], close=True,
                       dxfattribs={"layer": "ANNOTATION"})
    label_title = title.strip() or "Atta Engineering Drawing"
    dim_str = "OVERALL: %.2f x %.2f x %.2f mm (%s)" % (dims[0], dims[1], dims[2], drawing["backend"])
    msp.add_text(
        label_title,
        height=text_h * 1.1,
        dxfattribs={"layer": "ANNOTATION", "insert": (bx0 + 3.0 * u_scale, by0 + text_h * 1.9)},
    )
    msp.add_text(
        dim_str,
        height=text_h * 0.85,
        dxfattribs={"layer": "ANNOTATION", "insert": (bx0 + 3.0 * u_scale, by0 + text_h * 0.6)},
    )

    try:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        doc.saveas(path)
        size = os.path.getsize(path) if os.path.exists(path) else 0
    except OSError as exc:
        raise AttaError("drawing_export", "DXF output could not be written: %s" % exc,
                        path="path", fix="use a writable file path whose parent is a directory") from exc
    if size <= 0:
        raise AttaError("drawing_export", "DXF export produced no bytes",
                        path="path", fix="check write permissions")

    return {
        "format": "dxf",
        "ok": True,
        "path": str(path),
        "bytes": int(size),
        "backend": drawing["backend"],
        "views": list(drawing["views"].keys()),
        "entities": int(len(msp)),
        "units": units_norm,
    }


def export_svg(path: str, mesh: M.Mesh | None, shape=None,
               views=("front", "top", "right", "iso"),
               title: str = "", dims_mm: list | None = None) -> dict:
    """Write a clean multi-view engineering drawing SVG with title block."""
    drawing = project_drawing(mesh, shape=shape, views=views, include_hidden=True)
    offsets, (smin_x, smin_y, smax_x, smax_y) = _layout_views(drawing)
    dims = [round(float(x), 3) for x in (dims_mm if dims_mm is not None else drawing["bbox_mm"]["size"])]

    content_w = max(smax_x - smin_x, 10.0)
    content_h = max(smax_y - smin_y, 10.0)
    svg_w, svg_h = 1000.0, 700.0
    pad_x, pad_top, pad_bot = 70.0, 55.0, 110.0
    scale = min((svg_w - 2.0 * pad_x) / content_w, (svg_h - pad_top - pad_bot) / content_h)
    ox = pad_x + 0.5 * ((svg_w - 2.0 * pad_x) - content_w * scale) - smin_x * scale
    # SVG +Y is down, CAD +Y is up.
    oy = pad_top + 0.5 * ((svg_h - pad_top - pad_bot) - content_h * scale) + smax_y * scale

    def tx(x_mm: float) -> float:
        return ox + x_mm * scale

    def ty(y_mm: float) -> float:
        return oy - y_mm * scale

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" viewBox="0 0 %.0f %.0f">' % (
            svg_w, svg_h, svg_w, svg_h),
        '  <rect x="0" y="0" width="%.0f" height="%.0f" fill="#ffffff"/>' % (svg_w, svg_h),
        '  <rect x="16" y="16" width="%.0f" height="%.0f" fill="none" stroke="#1f2937" stroke-width="1.5"/>' % (
            svg_w - 32, svg_h - 32),
    ]

    for vname, vdata in drawing["views"].items():
        dx, dy = offsets[vname]
        lines.append('  <g id="view-%s">' % _xml_escape(vname))
        # Hidden edges
        for item in vdata["hidden"]:
            pts = " ".join("%.2f,%.2f" % (tx(p[0] + dx), ty(p[1] + dy)) for p in item["points"])
            lines.append('    <polyline points="%s" fill="none" stroke="#9ca3af" stroke-width="0.9" stroke-dasharray="5,3"/>' % pts)
        # Centerlines for visible circles/arcs
        for item in vdata["visible"]:
            if item["kind"] in ("circle", "arc"):
                cx = tx(item["center"][0] + dx)
                cy = ty(item["center"][1] + dy)
                r_px = item["radius"] * scale
                ext = r_px * (1.25 if item["kind"] == "circle" else 0.4)
                lines.append('    <line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#dc2626" stroke-width="0.7" stroke-dasharray="8,3,2,3"/>' % (
                    cx - ext, cy, cx + ext, cy))
                lines.append('    <line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#dc2626" stroke-width="0.7" stroke-dasharray="8,3,2,3"/>' % (
                    cx, cy - ext, cx, cy + ext))
        # Visible edges
        for item in vdata["visible"]:
            if item["kind"] == "circle":
                cx = tx(item["center"][0] + dx)
                cy = ty(item["center"][1] + dy)
                r_px = item["radius"] * scale
                lines.append('    <circle cx="%.2f" cy="%.2f" r="%.2f" fill="none" stroke="#111827" stroke-width="1.5"/>' % (
                    cx, cy, r_px))
            else:
                pts = " ".join("%.2f,%.2f" % (tx(p[0] + dx), ty(p[1] + dy)) for p in item["points"])
                lines.append('    <polyline points="%s" fill="none" stroke="#111827" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' % pts)

        b = vdata["bounds_mm"]
        lx = tx(0.5 * (b["min"][0] + b["max"][0]) + dx)
        ly = ty(b["min"][1] + dy) + 18.0
        lines.append('    <text x="%.2f" y="%.2f" text-anchor="middle" font-family="monospace" font-size="12" font-weight="bold" fill="#374151">%s</text>' % (
            lx, ly, _xml_escape(vname.upper())))
        lines.append('  </g>')

    # Overall bbox callout on front (or first) view
    anchor = "front" if "front" in drawing["views"] else next(iter(drawing["views"]))
    ab = drawing["views"][anchor]["bounds_mm"]
    adx, ady = offsets[anchor]
    ax0, ax1 = tx(ab["min"][0] + adx), tx(ab["max"][0] + adx)
    ay0, ay1 = ty(ab["max"][1] + ady), ty(ab["min"][1] + ady)
    if abs(ax1 - ax0) > 4.0:
        dim_y = ay0 - 14.0
        w_mm = round(ab["size"][0], 2)
        lines.append('  <line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#2563eb" stroke-width="1.0"/>' % (ax0, dim_y, ax1, dim_y))
        lines.append('  <text x="%.2f" y="%.2f" text-anchor="middle" font-family="monospace" font-size="11" fill="#1d4ed8">%.2f mm</text>' % (
            0.5 * (ax0 + ax1), dim_y - 4.0, w_mm))
    if abs(ay1 - ay0) > 4.0:
        dim_x = ax0 - 16.0
        h_mm = round(ab["size"][1], 2)
        lines.append('  <line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#2563eb" stroke-width="1.0"/>' % (dim_x, ay0, dim_x, ay1))
        lines.append('  <text x="%.2f" y="%.2f" text-anchor="middle" font-family="monospace" font-size="11" fill="#1d4ed8" transform="rotate(-90 %.2f %.2f)">%.2f mm</text>' % (
            dim_x - 4.0, 0.5 * (ay0 + ay1), dim_x - 4.0, 0.5 * (ay0 + ay1), h_mm))

    # Title block
    tb_x, tb_y, tb_w, tb_h = svg_w - 396.0, svg_h - 86.0, 380.0, 70.0
    label_title = title.strip() or "Atta Engineering Drawing"
    dim_text = "%.2f x %.2f x %.2f mm" % (dims[0], dims[1], dims[2]) if len(dims) >= 3 else str(dims)
    lines.extend([
        '  <g id="title-block">',
        '    <rect x="%.0f" y="%.0f" width="%.0f" height="%.0f" fill="#f9fafb" stroke="#1f2937" stroke-width="1.2"/>' % (tb_x, tb_y, tb_w, tb_h),
        '    <text x="%.0f" y="%.0f" font-family="sans-serif" font-size="14" font-weight="bold" fill="#111827">%s</text>' % (
            tb_x + 12, tb_y + 24, _xml_escape(label_title)),
        '    <text x="%.0f" y="%.0f" font-family="monospace" font-size="11" fill="#374151">BBOX: %s</text>' % (
            tb_x + 12, tb_y + 44, _xml_escape(dim_text)),
        '    <text x="%.0f" y="%.0f" font-family="monospace" font-size="10" fill="#6b7280">UNITS: mm | BACKEND: %s</text>' % (
            tb_x + 12, tb_y + 60, _xml_escape(drawing["backend"])),
        '  </g>',
        '</svg>',
    ])

    payload = "\n".join(lines) + "\n"
    try:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(payload)
        size = os.path.getsize(path) if os.path.exists(path) else 0
    except OSError as exc:
        raise AttaError("drawing_export", "SVG output could not be written: %s" % exc,
                        path="path", fix="use a writable file path whose parent is a directory") from exc

    return {
        "format": "svg_drawing",
        "ok": True,
        "path": str(path),
        "bytes": int(size),
        "backend": drawing["backend"],
        "views": list(drawing["views"].keys()),
        "dims_mm": dims,
    }
