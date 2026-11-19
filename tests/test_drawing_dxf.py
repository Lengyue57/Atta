# -*- coding: utf-8 -*-
"""Tests for 2D engineering drawing projection, DXF, and SVG export (P0-3)."""

import xml.etree.ElementTree as ET

import ezdxf
import pytest

from atta import brep, drawing, features
from atta.errors import AttaError


def _occt():
    return pytest.importorskip("build123d")


def _bracket_spec():
    return {
        "schema": "atta.spec/v3",
        "name": "bracket",
        "model": {
            "kind": "graph",
            "units": "mm",
            "features": [
                {"id": "plate", "op": "box", "args": {"size": [30, 20, 10]}},
                {"id": "hole", "op": "cylinder", "args": {"radius": 4, "height": 20}},
                {"id": "out", "op": "difference", "inputs": ["plate", "hole"]},
            ],
            "output": "out",
        },
    }


def test_project_drawing_occt_hlr_exact_circles_and_hidden_lines():
    _occt()
    spec = _bracket_spec()
    mesh = features.evaluate(spec["model"]["features"])
    shape = brep.build_graph(spec)["shape"]

    proj = drawing.project_drawing(mesh, shape=shape, views=("front", "top", "right", "iso"))
    assert proj["ok"] is True
    assert proj["backend"] == "occt_hlr"
    assert proj["units"] == "mm"
    assert proj["bbox_mm"]["size"] == pytest.approx([30.0, 20.0, 10.0], abs=1e-4)
    assert set(proj["views"]) == {"front", "top", "right", "iso"}

    # Top view must contain the exact analytic circle for the 4 mm radius hole.
    top_vis = proj["views"]["top"]["visible"]
    circles = [item for item in top_vis if item["kind"] == "circle"]
    assert len(circles) == 1
    assert circles[0]["radius"] == pytest.approx(4.0, abs=1e-5)
    assert circles[0]["center"] == pytest.approx([0.0, 0.0], abs=1e-5)

    # Front view must have hidden edges for the internal through-hole walls.
    front_hid = proj["views"]["front"]["hidden"]
    assert len(front_hid) >= 2
    xs = sorted({round(pt[0], 3) for item in front_hid for pt in item["points"]})
    assert -4.0 in xs and 4.0 in xs

    # Determinism & include_hidden=False
    proj_no_hid = drawing.project_drawing(mesh, shape=shape, views=("front",), include_hidden=False)
    assert proj_no_hid["views"]["front"]["hidden"] == []
    proj_again = drawing.project_drawing(mesh, shape=shape, views=("front", "top", "right", "iso"))
    assert proj == proj_again


def test_project_drawing_occt_hlr_filleted_part():
    _occt()
    spec = {
        "schema": "atta.spec/v3",
        "name": "filleted_box",
        "model": {
            "kind": "graph",
            "units": "mm",
            "features": [
                {"id": "box", "op": "box", "args": {"size": [30, 20, 10]}},
                {"id": "out", "op": "fillet", "inputs": ["box"], "args": {"radius": 1.5, "edges": "all"}},
            ],
            "output": "out",
        },
    }
    mesh = features.evaluate(spec["model"]["features"])
    shape = brep.build_graph(spec)["shape"]
    proj = drawing.project_drawing(mesh, shape=shape)
    assert proj["backend"] == "occt_hlr"
    for vname in ("front", "top", "right", "iso"):
        assert len(proj["views"][vname]["visible"]) >= 4


def test_project_drawing_mesh_hlr_fallback():
    spec = _bracket_spec()
    mesh = features.evaluate(spec["model"]["features"])
    proj = drawing.project_drawing(mesh, shape=None, views=("front", "top", "right", "iso"))
    assert proj["ok"] is True
    assert proj["backend"] == "mesh_hlr"
    assert proj["bbox_mm"]["size"] == pytest.approx([30.0, 20.0, 10.0], abs=0.2)
    for vname in ("front", "top", "right", "iso"):
        assert len(proj["views"][vname]["visible"]) >= 4
    # Through-hole edges are occluded in front view -> classified as hidden.
    assert len(proj["views"]["front"]["hidden"]) >= 2

    proj_no_hid = drawing.project_drawing(mesh, shape=None, views=("front",), include_hidden=False)
    assert proj_no_hid["views"]["front"]["hidden"] == []


def test_project_drawing_rejects_invalid_inputs():
    with pytest.raises(AttaError) as exc:
        drawing.project_drawing(None, shape=None)
    assert exc.value.code == "bad_arg"

    mesh = features.evaluate([{"id": "b", "op": "box", "args": {"size": [10, 10, 10]}}])
    with pytest.raises(AttaError) as exc2:
        drawing.project_drawing(mesh, views=("oblique",))
    assert exc2.value.code == "bad_arg"


@pytest.mark.parametrize("use_occt", [True, False])
def test_export_dxf_layers_entities_and_units(tmp_path, use_occt):
    spec = _bracket_spec()
    mesh = features.evaluate(spec["model"]["features"])
    shape = None
    if use_occt:
        _occt()
        shape = brep.build_graph(spec)["shape"]

    out_path = tmp_path / "bracket.dxf"
    receipt = drawing.export_dxf(str(out_path), mesh, shape=shape, title="Bracket Rev A", units="mm")
    assert receipt["ok"] is True
    assert receipt["format"] == "dxf"
    assert receipt["backend"] == ("occt_hlr" if use_occt else "mesh_hlr")
    assert receipt["bytes"] > 0
    assert receipt["entities"] > 10
    assert receipt["views"] == ["front", "top", "right", "iso"]

    doc = ezdxf.readfile(str(out_path))
    layer_names = {layer.dxf.name for layer in doc.layers}
    assert {"VISIBLE", "HIDDEN", "CENTER", "ANNOTATION"} <= layer_names
    assert doc.units == ezdxf.units.MM

    msp = doc.modelspace()
    by_layer = {}
    for ent in msp:
        by_layer.setdefault(ent.dxf.layer, []).append(ent)
    assert len(by_layer.get("VISIBLE", [])) >= 8
    assert len(by_layer.get("HIDDEN", [])) >= 2
    assert len(by_layer.get("ANNOTATION", [])) >= 4
    texts = [e.dxf.text for e in by_layer["ANNOTATION"] if e.dxftype() == "TEXT"]
    assert any("Bracket Rev A" in t for t in texts)
    assert any("FRONT" in t for t in texts)
    assert any(e.dxftype() == "DIMENSION" for e in by_layer["ANNOTATION"])
    if use_occt:
        assert any(e.dxftype() == "CIRCLE" for e in by_layer["VISIBLE"])
        assert len(by_layer.get("CENTER", [])) >= 2

    # Verify cm unit scaling works cleanly
    cm_path = tmp_path / "bracket_cm.dxf"
    cm_receipt = drawing.export_dxf(str(cm_path), mesh, shape=shape, units="cm")
    assert cm_receipt["units"] == "cm"
    cm_doc = ezdxf.readfile(str(cm_path))
    assert cm_doc.units == ezdxf.units.CM


@pytest.mark.parametrize("use_occt", [True, False])
def test_export_svg_engineering_sheet(tmp_path, use_occt):
    spec = _bracket_spec()
    mesh = features.evaluate(spec["model"]["features"])
    shape = None
    if use_occt:
        _occt()
        shape = brep.build_graph(spec)["shape"]

    svg_path = tmp_path / "bracket.svg"
    receipt = drawing.export_svg(str(svg_path), mesh, shape=shape, title="Bracket Sheet", dims_mm=[30, 20, 10])
    assert receipt["ok"] is True
    assert receipt["format"] == "svg_drawing"
    assert receipt["backend"] == ("occt_hlr" if use_occt else "mesh_hlr")
    assert receipt["bytes"] > 200
    assert receipt["dims_mm"] == [30.0, 20.0, 10.0]

    text = svg_path.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    assert root.tag.endswith("svg")
    assert "Bracket Sheet" in text
    assert "30.00 x 20.00 x 10.00 mm" in text
    for view_name in ("FRONT", "TOP", "RIGHT", "ISO"):
        assert view_name in text
