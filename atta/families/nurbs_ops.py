# -*- coding: utf-8 -*-
"""Rational spline graph op. Unit conversion occurs ONLY in prepare."""
from __future__ import annotations

import numpy as np

from ..errors import AttaError
from ..features import register_op
from ..nurbs import NURBSSurface, tessellate, _array, _positive

__all__ = []


def _nurbs_op(*, surface, **options):
    return tessellate(surface, **options)


def _prepare(args, scale, path):
    try:
        # Validate bounded rectangular numeric input before making scaled copies.
        control = _array(args["control_points"], "control_points")
        factor = _positive(scale, "unit_scale")
        with np.errstate(over="ignore", invalid="ignore"):
            control *= factor
        surface = NURBSSurface(control, **{k: args[k] for k in
                               ("weights", "degree_u", "degree_v", "knots_u", "knots_v")
                               if k in args})
        tolerance = _positive(args.get("tolerance", 0.05), "tolerance") * factor
        # Kernel preflights all tessellation options again for direct callers.
        options = {k: args[k] for k in ("u_samples", "v_samples", "max_vertices",
                   "max_faces", "max_evaluations", "max_refinements", "fail_on_tolerance", "flip") if k in args}
        return {"surface": surface, "tolerance": tolerance, **options}
    except AttaError as exc:
        raise AttaError(exc.code, exc.message, path=path + ".args." + (exc.path or "control_points"),
                        fix=exc.fix, details=exc.details, severity=exc.severity) from exc


_nurbs_op.prepare = _prepare
register_op("nurbs_surface", {
    "required": ("control_points",), "arity": 0,
    "optional": ("weights", "degree_u", "degree_v", "knots_u", "knots_v",
                 "tolerance", "u_samples", "v_samples", "max_vertices", "max_faces",
                 "max_evaluations", "max_refinements", "fail_on_tolerance", "flip"),
    "doc": "Open rational tensor-product B-spline patch: control_points[u][v][3] in model units, "
           "positive weights, degree 1..8, optional knot vectors. Bounded conforming grid; "
           "tolerance is sampled triangle-correspondence error, NOT a global bound. "
           "Default tolerance=0.05 model units; fail_on_tolerance=true. Use thicken for a solid.",
    "args": {"control_points": "anything", "weights": "anything", "knots_u": "anything", "knots_v": "anything",
             "degree_u": ("int", 1), "degree_v": ("int", 1), "tolerance": "length",
             "u_samples": ("int", 2), "v_samples": ("int", 2), "max_vertices": ("int", 4),
             "max_faces": ("int", 2), "max_evaluations": ("int", 1), "max_refinements": ("int", 0),
             "fail_on_tolerance": "bool", "flip": "bool"},
}, _nurbs_op)
