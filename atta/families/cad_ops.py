# -*- coding: utf-8 -*-
"""Exact edge finishing. Registration is cheap; OCCT is imported only on demand."""
from ..errors import AttaError
from ..features import register_op


def _exact_only(**kwargs):
    raise AttaError("brep_provenance_required",
                    "fillet/chamfer require the complete analytic source graph",
                    path="inputs", fix="evaluate the graph; do not pass a triangle mesh directly")


for _op, _size in (("fillet", "radius"), ("chamfer", "length")):
    register_op(_op, {
        "required": (_size, "edges"),
        "optional": ("linear_deflection", "angular_deflection_deg", "max_triangles"),
        "arity": 1,
        "args": {_size: "length", "edges": "anything",
                 "linear_deflection": "length", "angular_deflection_deg": "number",
                 "max_triangles": ("int", 4)},
        "lengths": (_size, "linear_deflection"),
        "ints": ("max_triangles",),
        "doc": ("Exact OCCT %s on a supported closed BREP graph. edges='all' or a "
                "geometric selector (parallel, kind, at, expect_count); no numeric indices. "
                "Mesh and STEP share the same analytic operation. Sizes need not be feasible "
                "on every shape; unsupported ancestry and kernel failures are errors." % _op),
    }, _exact_only)
