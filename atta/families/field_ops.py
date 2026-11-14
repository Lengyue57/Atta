# -*- coding: utf-8 -*-
"""Nested implicit field composition; public lengths are model units (mm)."""
from __future__ import annotations

import numpy as np

from .. import fieldtree
from ..errors import AttaError
from ..features import register_op


def _prepare(args, scale, path):
    tree = fieldtree.compile_tree(args.get("tree"), scale=scale, path=path + ".args.tree")
    try:
        bounds = np.asarray(args.get("bounds"), dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise AttaError("bad_arg", "bounds must be [min_xyz,max_xyz]", path=path + ".args.bounds",
                        fix="provide finite ordered coordinates in model units")
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise AttaError("bad_arg", "bounds must be finite and strictly ordered",
                        path=path + ".args.bounds", fix="provide [[xmin,ymin,zmin],[xmax,ymax,zmax]]")
    out = {"field": tree, "bounds": bounds * scale,
           "spacing": fieldtree.number(args.get("spacing"), path + ".args.spacing", positive=True) * scale}
    for key in ("max_samples", "max_evaluations", "max_triangles", "chunk_points", "budget_policy"):
        if key in args:
            out[key] = args[key]
    if "min_feature" in args:
        out["min_feature"] = fieldtree.number(args["min_feature"], path + ".args.min_feature", positive=True) * scale
    return out


def _build(**kwargs):
    return fieldtree.mesh_field(**kwargs)


_build.prepare = _prepare
register_op("field_tree", {
    "required": ("tree", "bounds", "spacing"), "arity": 0,
    "optional": ("max_samples", "max_evaluations", "max_triangles", "chunk_points", "budget_policy", "min_feature"),
    "args": {"tree": "anything", "bounds": "anything", "spacing": "length",
             "min_feature": "length", "max_samples": ("int", 343),
             "max_evaluations": ("int", 343), "max_triangles": ("int", 4),
             "chunk_points": ("int", 1), "budget_policy": "enum:error|coarsen"},
    "doc": "Nested bounded scalar fields: primitives/CSG/smooth CSG/TPMS/transform/shell/noise/twist. "
           "Explicit bounds + spacing (mm); hard sampling/work/output limits. No exact distance claim.",
}, _build)
