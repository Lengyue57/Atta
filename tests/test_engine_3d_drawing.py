# -*- coding: utf-8 -*-
"""Engine & tool envelope tests for graph/assembly patching, mesh_agreement, 2D drawings, and tool warnings."""

from __future__ import annotations

import hashlib
import importlib
import math
import sys
import types
from pathlib import Path

import pytest

from atta import engine, spec as specmod
from atta.agent import tools
from atta.errors import AttaError
from atta.mesh import Mesh
from atta.version import VERSION


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTA_HOME", str(tmp_path))
    return tmp_path


def _plate_graph_spec():
    return {
        "schema": specmod.SCHEMA,
        "name": "plate-hole",
        "model": {
            "kind": "graph",
            "units": "mm",
            "segments": 64,
            "output": "cut",
            "features": [
                {"id": "plate", "op": "box",
                 "args": {"size": [40, 20, 10], "center": [20, 10, 5]}},
                {"id": "hole", "op": "cylinder",
                 "args": {"radius": 3, "height": 20, "center": [20, 10, 5]}},
                {"id": "cut", "op": "difference",
                 "args": {}, "inputs": ["plate", "hole"]},
            ],
        },
        "quality": "draft",
        "material": "steel",
    }


def _assembly_spec():
    return {
        "schema": specmod.SCHEMA,
        "name": "block-asm",
        "model": {
            "kind": "assembly",
            "parts": [
                {
                    "id": "base",
                    "fixed": True,
                    "graph": {
                        "output": "b",
                        "features": [
                            {"id": "b", "op": "box",
                             "args": {"size": [30, 30, 10], "center": [0, 0, 5]}},
                        ],
                    },
                },
                {
                    "id": "peg",
                    "graph": {
                        "output": "p",
                        "features": [
                            {"id": "p", "op": "cylinder",
                             "args": {"radius": 4, "height": 20, "center": [0, 0, 20]}},
                        ],
                    },
                },
            ],
            "mates": [],
        },
        "quality": "draft",
        "material": "steel",
    }


# --------------------------------------------------- engine.spec + engine.model

def test_engine_spec_patch_and_preflight(home):
    s = _plate_graph_spec()
    pf = engine.spec(s, op="preflight")
    assert pf["op"] == "preflight"
    assert pf["ok"] is True
    assert pf["checks"]["patchability"] is True

    patched = engine.spec(s, op="patch", patches={
        "features": {"hole": {"args": {"radius": 5}}},
    })
    assert patched["op"] == "patch"
    assert patched["ok"] is True
    assert patched["delta"]["modified_features"]["hole"]["after"]["args"]["radius"] == 5


def test_engine_model_graph_and_assembly_patches(home):
    base = engine.model(spec=_plate_graph_spec())
    base_vol = base["facts"]["volume_mm3"]

    patched = engine.model(
        spec=_plate_graph_spec(),
        patches={"features": {"hole": {"args": {"radius": 6}}}},
    )
    assert patched["status"] == "ok"
    assert patched["applied_patches"]
    assert patched["patch_delta"]["changed"] is True
    # Larger hole reduces solid volume
    assert patched["facts"]["volume_mm3"] < base_vol - 500.0
    expected_vol = 40 * 20 * 10 - math.pi * (6 ** 2) * 10
    assert patched["facts"]["volume_mm3"] == pytest.approx(expected_vol, rel=0.02)

    with pytest.raises(AttaError) as exc:
        engine.model(spec=_plate_graph_spec(),
                     patches={"features": {"nope": {"args": {"radius": 2}}}})
    assert exc.value.code == "patch_failed"

    asm_res = engine.model(
        spec=_assembly_spec(),
        patches={"parts": {"peg": {"graph": {"features": {"p": {"args": {"radius": 6}}}}}}},
    )
    assert asm_res["model_id"]
    assert asm_res["applied_patches"]
    assert asm_res["patch_delta"]["changed"] is True


# --------------------------------------------------- engine.verify mesh_agreement

def test_engine_verify_resolves_mesh_agreement_target(home, monkeypatch):
    ref = engine.model(spec={
        "schema": specmod.SCHEMA,
        "name": "ref_box",
        "model": {"kind": "graph", "features": [
            {"id": "b", "op": "box", "args": {"size": [20, 20, 20]}},
        ]},
    })
    target = engine.model(spec={
        "schema": specmod.SCHEMA,
        "name": "tgt_box",
        "model": {"kind": "graph", "features": [
            {"id": "b", "op": "box", "args": {"size": [20, 20, 20]}},
        ]},
    })

    import inspect
    seen = {}
    real_verify = engine.verifymod.verify
    supports_ref = "reference_mesh" in inspect.signature(real_verify).parameters

    def spy_verify(mesh, *, constraints=None, expected_genus=None,
                   level="standard", render_receipt=None, parts=None,
                   assumptions=None, reference_mesh=None):
        seen["reference_mesh"] = reference_mesh
        seen["constraints"] = constraints
        if supports_ref:
            return real_verify(
                mesh, constraints=constraints, expected_genus=expected_genus,
                level=level, render_receipt=render_receipt, parts=parts,
                assumptions=assumptions, reference_mesh=reference_mesh,
            )
        # Before verify-metrics merges, strip mesh_agreement when calling legacy verify
        cons_legacy = {k: v for k, v in (constraints or {}).items()
                       if k != "mesh_agreement"}
        return real_verify(
            mesh, constraints=cons_legacy, expected_genus=expected_genus,
            level=level, render_receipt=render_receipt, parts=parts,
            assumptions=assumptions,
        )

    monkeypatch.setattr(engine.verifymod, "verify", spy_verify)

    out = engine.verify(
        target["model_id"],
        constraints={
            "mesh_agreement": {
                "target_model_id": ref["model_id"],
                "chamfer_max_mm": 0.5,
                "hausdorff_p95_max_mm": 1.0,
                "volumetric_iou_min": 0.9,
            },
        },
    )
    assert out["verdict"] == "pass"
    assert isinstance(seen["reference_mesh"], Mesh)
    assert seen["reference_mesh"].volume() * 1e9 == pytest.approx(8000.0, rel=1e-4)


# --------------------------------------------------- engine.deliver drawings

def _install_fake_drawing(monkeypatch, calls):
    mod = types.ModuleType("atta.drawing")

    def export_dxf(path, mesh, *, shape=None, views=("front", "top", "right", "iso"),
                   title="", units="mm"):
        calls.append(("dxf", shape is not None, title))
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
        p.write_bytes(payload)
        return {
            "format": "dxf",
            "ok": True,
            "path": str(p),
            "bytes": len(payload),
            "backend": "occt_hlr" if shape is not None else "mesh_hlr",
            "views": list(views),
            "entities": 4,
        }

    def export_svg(path, mesh, *, shape=None, views=("front", "top", "right", "iso"),
                   title="", dims_mm=None):
        calls.append(("svg_drawing", shape is not None, title, dims_mm))
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = b"<svg xmlns='http://www.w3.org/2000/svg'><rect width='10' height='10'/></svg>"
        p.write_bytes(payload)
        return {
            "format": "svg_drawing",
            "ok": True,
            "path": str(p),
            "bytes": len(payload),
            "backend": "occt_hlr" if shape is not None else "mesh_hlr",
            "views": list(views),
        }

    mod.export_dxf = export_dxf
    mod.export_svg = export_svg
    monkeypatch.setitem(sys.modules, "atta.drawing", mod)
    return mod


def test_engine_deliver_dxf_and_svg_drawing_graph_and_mesh_fallback(home, monkeypatch):
    assert "dxf" in engine.EXPORT_FORMATS
    assert "svg_drawing" in engine.EXPORT_FORMATS
    cat = engine.catalog()
    assert "dxf" in cat["formats"]
    assert "svg_drawing" in cat["formats"]

    calls = []
    try:
        importlib.import_module("atta.drawing")
        has_real_drawing = True
    except ModuleNotFoundError:
        has_real_drawing = False
        _install_fake_drawing(monkeypatch, calls)

    # 1. Graph model (tries OCCT B-rep HLR if build123d is available)
    graph_res = engine.model(spec=_plate_graph_spec())
    deliv = engine.deliver(
        graph_res["model_id"],
        formats=["dxf", "svg_drawing"],
        outdir=str(home / "out_graph"),
    )
    assert deliv["status"] == "ok", deliv
    by_fmt = {f["format"]: f for f in deliv["files"]}
    assert set(by_fmt) == {"dxf", "svg_drawing"}
    for fmt, row in by_fmt.items():
        assert row["ok"] is True
        p = Path(row["path"])
        assert p.is_file() and p.stat().st_size > 0
        assert row["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
        assert row["backend"] in ("occt_hlr", "mesh_hlr")

    # 2. Recipe / mesh-only fallback (shape=None -> mesh_hlr)
    mug = engine.model("product.mug", quality="draft")
    deliv2 = engine.deliver(
        mug["model_id"],
        formats=["dxf", "svg_drawing"],
        outdir=str(home / "out_mug"),
    )
    assert deliv2["status"] == "ok", deliv2
    by_fmt2 = {f["format"]: f for f in deliv2["files"]}
    assert by_fmt2["dxf"]["backend"] == "mesh_hlr"
    assert by_fmt2["svg_drawing"]["backend"] == "mesh_hlr"
    if not has_real_drawing:
        # Second call had shape=None
        assert calls[-1][0] == "svg_drawing" and calls[-1][1] is False
        assert calls[-1][3] is not None and len(calls[-1][3]) == 3


# --------------------------------------------------- agent tools warnings & schemas

def test_tool_envelope_warns_on_dropped_unknown_args_and_stays_clean_otherwise(home):
    # Clean call: exact envelope keys preserved
    clean_res = tools.call_tool("atta_spec", {
        "op": "preflight",
        "spec": _plate_graph_spec(),
    })
    assert clean_res["ok"] is True
    assert set(clean_res) == {"ok", "tool", "data", "error", "ms"}
    assert clean_res["data"]["op"] == "preflight"

    # Unknown argument: surfaced in warnings[] instead of silently dropped
    warn_res = tools.call_tool("atta_spec", {
        "op": "patch",
        "spec": _plate_graph_spec(),
        "patches": {"features": {"hole": {"args": {"radius": 7}}}},
        "hallucinated_param": 123,
    })
    assert warn_res["ok"] is True
    assert warn_res["data"]["op"] == "patch"
    assert warn_res["data"]["spec"]["model"]["features"][1]["args"]["radius"] == 7
    assert "warnings" in warn_res
    assert warn_res["warnings"][0]["code"] == "ignored_argument"
    assert warn_res["warnings"][0]["argument"] == "hallucinated_param"
    assert "hallucinated_param" in warn_res["warnings"][0]["message"]


def test_tool_schemas_expose_patch_preflight_drawings_and_version_tags():
    spec_schema = tools.TOOL_BY_NAME["atta_spec"]["inputSchema"]["properties"]
    assert set(spec_schema["op"]["enum"]) >= {
        "validate", "propose", "compile", "schema", "patch", "preflight",
    }
    assert "patches" in spec_schema

    deliver_tool = tools.TOOL_BY_NAME["atta_deliver"]
    assert "dxf" in deliver_tool["inputSchema"]["properties"]["formats"]["description"]
    assert "svg_drawing" in deliver_tool["inputSchema"]["properties"]["formats"]["description"]

    prefix = "[Atta v%s] " % VERSION
    for tool in tools.TOOLS:
        assert tool["description"].startswith(prefix), tool["name"]
