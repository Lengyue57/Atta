# -*- coding: utf-8 -*-
"""Tests for Text2CAD L0-L3 prompt abstraction tiers, provenance honesty, and zero-tautology bench."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atta import conjure  # noqa: E402
from scripts import run_bench  # noqa: E402

BENCH_PATH = ROOT / "corpus" / "bench" / "generalization_v3_prompts.json"
TIERS = ("L0_abstract", "L1_beginner", "L2_intermediate", "L3_expert")


def test_v3_prompts_bench_structure_and_stratification():
    raw = BENCH_PATH.read_text(encoding="utf-8")
    assert "Text2CAD" in raw or "Text2CAD-style" in raw
    assert "SadilKhan" not in raw
    assert "cad-recode" not in raw
    assert "CC BY-NC" not in raw

    doc = json.loads(raw)
    assert doc["license"] == "MIT"
    assert tuple(doc["tiers"]) == TIERS
    cases = doc["cases"]
    assert len(cases) >= 40

    counts = Counter(c["tier"] for c in cases)
    assert set(counts.keys()) == set(TIERS)
    assert len(set(counts.values())) == 1, f"uneven tier stratification: {counts}"
    assert all(v >= 10 for v in counts.values())

    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        assert isinstance(c["prompt"], str) and len(c["prompt"].strip()) >= 5
        assert isinstance(c["predicates"], dict) and len(c["predicates"]) >= 3


def test_conjure_from_prompt_provenance_honesty_across_tiers():
    # L0: shape intent only -> invented
    l0 = conjure.conjure_from_prompt("a coffee mug", seed=0)
    assert l0["archetype"]["key"] == "mug"
    assert l0["provenance"]["size_mm"] == "invented"
    assert l0["provenance_detail"]["size_mm"]["measured"] is False

    # L1: style + material + parts, no numbers -> invented
    l1 = conjure.conjure_from_prompt(
        "an industrial steel pipe flange with central bore and bolt ring", seed=1)
    assert l1["archetype"]["key"] == "flange"
    assert l1["style"] == "industrial"
    assert l1["prompt_parsed"]["material"] == "steel"
    assert l1["provenance"]["size_mm"] == "invented"
    assert l1["provenance_detail"]["size_mm"]["measured"] is False

    # L2: rough / partial dimensions -> derived, NEVER measured
    l2_rough = conjure.conjure_from_prompt(
        "a steel mounting flange around 160 mm diameter with 8 bolt holes", seed=0)
    assert l2_rough["archetype"]["key"] == "flange"
    assert l2_rough["provenance"]["size_mm"] == "derived"
    assert l2_rough["provenance_detail"]["size_mm"]["measured"] is False
    assert l2_rough["ratios"]["bolt_count"] == 8

    l2_partial = conjure.conjure_from_prompt(
        "a ceramic coffee mug roughly 105 mm tall with a side handle", seed=2)
    assert l2_partial["provenance"]["size_mm"] == "derived"
    assert l2_partial["provenance_detail"]["size_mm"]["measured"] is False

    # L3: explicit 3D dimensions in mm + wall thickness -> measured
    l3 = conjure.conjure_from_prompt(
        "a ceramic coffee mug, 115 x 85 x 95 mm, wall thickness 4 mm, stable upright",
        seed=0,
    )
    assert l3["archetype"]["key"] == "mug"
    assert l3["size_mm"] == [115.0, 85.0, 95.0]
    assert l3["provenance"]["size_mm"] == "measured"
    assert l3["provenance_detail"]["size_mm"]["measured"] is True
    assert l3["provenance"]["ratios.wall"] == "measured"


def test_prompt_eval_zero_tautology_catches_geometry_and_provenance_failures():
    # Adversarial check: verify _eval_prompt_predicates fails honestly when
    # measured geometry or provenance violates expectations.
    pred = {
        "closed": True,
        "n_components": 1,
        "genus_exact": 7,
        "min_wall_mm": 3.0,
        "bbox_check": {"mode": "sorted_xyz", "target_mm": [160, 160, 20], "tol_pct": 5.0},
        "stability_required": True,
        "readability_min": 0.85,
        "expected_provenance_size": "invented",
    }
    bad_plan = {"provenance": {"size_mm": "measured"}}  # L0/L1 lying as measured
    res = run_bench._eval_prompt_predicates(
        pred,
        topo={"closed": False, "n_components": 2, "genus": 0},
        bbox_size=[100.0, 100.0, 50.0],
        wall={"min_mm": None},
        design_checks={
            "stability": {"ok": False, "value": {"clearance_to_hull_mm": -5.0}},
            "readability": {"value": {"score": 0.60}},
        },
        plan=bad_plan,
        spec_material="ceramic",
    )
    for key in ("closed", "n_components", "genus_exact", "min_wall_mm",
                "bbox_check", "stability", "readability", "provenance_honesty"):
        assert res[key]["ok"] is False, f"expected {key} to fail"


def test_run_prompt_case_one_per_tier_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTA_HOME", str(tmp_path / "atta-home"))
    doc = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    by_tier = {}
    for c in doc["cases"]:
        # Pick fast deterministic geometric cases (non-sdf_blend) from each tier
        if c["tier"] not in by_tier and c["id"] in {
            "v3_l0_02", "v3_l1_07", "v3_l2_01", "v3_l3_01"
        }:
            by_tier[c["tier"]] = c
    assert set(by_tier.keys()) == set(TIERS)

    mini_path = tmp_path / "mini_prompts.json"
    out_path = tmp_path / "mini_out.json"
    mini_path.write_text(json.dumps({
        "bench": "mini_v3",
        "tiers": list(TIERS),
        "cases": [by_tier[t] for t in TIERS],
    }), encoding="utf-8")

    report = run_bench.run_prompts_bench(
        str(mini_path), out_path=str(out_path), quiet=True)
    assert out_path.exists()
    s = report["summary"]
    assert s["cases"] == 4
    assert s["case_pass"] == 1.0
    for t in TIERS:
        assert s["by_tier"][t]["case_pass"] == 1.0
    for r in report["cases"]:
        assert r["propose_ok"] is True
        assert r["build_success"] is True
        assert r["verify_pass"] is True
        assert r["predicate_pass"] is True
