#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute corpus/coverage_report.json and corpus/license_report.json from corpus/normalized/*.jsonl.

Matches the schema and definitions of ``atta.coverage_report/v1`` and
``atta.license_report/v1``:
- accepted: ``quality.license_ok is True`` and ``quality.license_tier in ("A", "B")``
- substantive: accepted AND ``len(parameters) >= 1``
- reference_only: accepted AND ``len(parameters) == 0``
- families: distinct ``shape.family`` across substantive records

Usage::

    python3 scripts/coverage_report.py
    python3 scripts/coverage_report.py --corpus-dir corpus/normalized --coverage-out corpus/coverage_report.json --license-out corpus/license_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

BASIS_TEXT = (
    "coverage is measured on SUBSTANTIVE records (accepted AND ≥1 parameter). "
    "reference_only accepted records are reported separately and never counted toward a target."
)


def is_accepted(rec: dict[str, Any]) -> bool:
    """Return True if record passes Tier A/B license policy."""
    q = rec.get("quality") or {}
    return bool(q.get("license_ok")) and q.get("license_tier") in ("A", "B")


def is_substantive(rec: dict[str, Any]) -> bool:
    """Return True if record is accepted and defines at least one parameter."""
    return is_accepted(rec) and len(rec.get("parameters") or []) >= 1


def classify_source_kind(rec: dict[str, Any]) -> str:
    """Map a record's source URL and kind to a license_report source_kind bucket."""
    src = rec.get("source") or {}
    url = (src.get("url") or "").lower()
    if "github.com/" in url:
        return "github"
    if "printables.com/" in url:
        return "printables"
    if "wikibooks.org/" in url:
        return "wikibooks"
    if "deep-geometry.github.io/" in url:
        return "deep-geometry.github.io"
    if "docs.blender.org/" in url:
        return "blender_docs"
    if "readthedocs.io/" in url or "readthedocs.org/" in url:
        return "readthedocs"
    if rec.get("kind") == "dataset":
        return "dataset_site"
    if "gitlab.com/" in url:
        return "gitlab"
    if "thingiverse.com/" in url:
        return "thingiverse"
    return "other_site"


def load_records(corpus_dir: Path) -> list[dict[str, Any]]:
    """Load all JSONL records in deterministic sorted filename order."""
    records: list[dict[str, Any]] = []
    for path in sorted(corpus_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return records


def build_coverage_report(
    records: list[dict[str, Any]],
    matrix: dict[str, Any],
    generated: str,
) -> dict[str, Any]:
    """Compute atta.coverage_report/v1 dictionary from records and coverage matrix."""
    accepted_recs = [r for r in records if is_accepted(r)]
    substantive_recs = [r for r in accepted_recs if len(r.get("parameters") or []) >= 1]
    ref_only_recs = [r for r in accepted_recs if len(r.get("parameters") or []) == 0]

    all_sub_families = {
        (r.get("shape") or {}).get("family")
        for r in substantive_recs
        if (r.get("shape") or {}).get("family")
    }

    by_cat_acc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cat_sub: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cat_ref: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rec in accepted_recs:
        cat = (rec.get("shape") or {}).get("category", "other")
        by_cat_acc[cat].append(rec)
        if len(rec.get("parameters") or []) >= 1:
            by_cat_sub[cat].append(rec)
        else:
            by_cat_ref[cat].append(rec)

    categories_out: list[dict[str, Any]] = []
    met_cats: list[str] = []
    short_cats: list[str] = []
    short_detail: dict[str, dict[str, int]] = {}

    for cat_cfg in matrix.get("categories") or []:
        cid = cat_cfg["id"]
        target_ex = int(cat_cfg.get("target_examples", 0))
        target_fam = int(cat_cfg.get("target_families", 0))
        acc_count = len(by_cat_acc[cid])
        sub_list = by_cat_sub[cid]
        sub_count = len(sub_list)
        ref_count = len(by_cat_ref[cid])

        fam_counts = Counter(
            (r.get("shape") or {}).get("family")
            for r in sub_list
            if (r.get("shape") or {}).get("family")
        )
        fam_detail = dict(fam_counts.most_common())
        fam_count = len(fam_detail)

        gap_ex = max(0, target_ex - sub_count)
        gap_fam = max(0, target_fam - fam_count)
        ex_ok = sub_count >= target_ex
        fam_ok = fam_count >= target_fam
        is_met = ex_ok and fam_ok

        categories_out.append(
            {
                "category": cid,
                "priority": cat_cfg.get("priority", ""),
                "target_examples": target_ex,
                "target_families": target_fam,
                "accepted": acc_count,
                "substantive": sub_count,
                "reference_only": ref_count,
                "families": fam_count,
                "families_detail": fam_detail,
                "gap_examples": gap_ex,
                "gap_families": gap_fam,
                "examples_ok": ex_ok,
                "families_ok": fam_ok,
                "met": is_met,
            }
        )
        if is_met:
            met_cats.append(cid)
        else:
            short_cats.append(cid)
            short_detail[cid] = {
                "substantive": sub_count,
                "target": target_ex,
                "deficit": gap_ex,
            }

    return {
        "schema": "atta.coverage_report/v1",
        "generated": generated,
        "basis": BASIS_TEXT,
        "totals": {
            "records": len(records),
            "accepted": len(accepted_recs),
            "substantive": len(substantive_recs),
            "reference_only": len(ref_only_recs),
            "families": len(all_sub_families),
        },
        "targets": matrix.get("targets") or {},
        "categories": categories_out,
        "met": met_cats,
        "short": short_cats,
        "short_detail": short_detail,
    }


def build_license_report(
    records: list[dict[str, Any]],
    generated: str,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute atta.license_report/v1 dictionary from records."""
    accepted_recs = [r for r in records if is_accepted(r)]
    substantive_recs = [r for r in accepted_recs if len(r.get("parameters") or []) >= 1]
    ref_only_recs = [r for r in accepted_recs if len(r.get("parameters") or []) == 0]

    tier_counts: Counter[str] = Counter()
    tier_acc_counts: Counter[str] = Counter()
    spdx_counts: Counter[str] = Counter()
    source_kind_counts: Counter[str] = Counter()
    tier_by_sk: dict[str, Counter[str]] = defaultdict(Counter)

    tier_c_with_code: list[str] = []
    tier_b_with_code: list[str] = []
    tier_c_accepted: list[str] = []
    tier_c_register: list[dict[str, Any]] = []

    for rec in records:
        q = rec.get("quality") or {}
        src = rec.get("source") or {}
        tier = q.get("license_tier", "unknown")
        spdx = src.get("license_spdx", "NOASSERTION")
        sk = classify_source_kind(rec)
        has_code = bool(rec.get("code"))

        tier_counts[tier] += 1
        if is_accepted(rec):
            tier_acc_counts[tier] += 1
        spdx_counts[spdx] += 1
        source_kind_counts[sk] += 1
        tier_by_sk[sk][tier] += 1

        if tier == "C":
            if has_code:
                tier_c_with_code.append(rec["id"])
            if q.get("license_ok"):
                tier_c_accepted.append(rec["id"])
            tier_c_register.append(
                {
                    "id": rec["id"],
                    "family": (rec.get("shape") or {}).get("family", ""),
                    "spdx": spdx,
                    "code_present": has_code,
                }
            )
        elif tier == "B" and has_code:
            tier_b_with_code.append(rec["id"])

    tier_c_register.sort(key=lambda item: item["id"])
    verdict = (
        "PASS"
        if not (tier_c_with_code or tier_b_with_code or tier_c_accepted)
        else "FAIL"
    )

    out: dict[str, Any] = {
        "schema": "atta.license_report/v1",
        "generated": generated,
        "total_records": len(records),
        "accepted": len(accepted_recs),
        "substantive": len(substantive_recs),
        "reference_only": len(ref_only_recs),
        "by_tier": dict(tier_counts.most_common()),
        "by_tier_accepted_only": dict(tier_acc_counts.most_common()),
        "by_spdx": dict(spdx_counts.most_common()),
        "by_source_kind": dict(source_kind_counts.most_common()),
        "tier_by_source_kind": {
            sk: dict(counts.most_common()) for sk, counts in tier_by_sk.items()
        },
        "policy_checks": {
            "tier_c_with_code": tier_c_with_code,
            "tier_b_with_code": tier_b_with_code,
            "tier_c_accepted": tier_c_accepted,
            "verdict": verdict,
        },
        "tier_c_register": tier_c_register,
    }

    if existing:
        if "repo_verdicts" in existing:
            out["repo_verdicts"] = existing["repo_verdicts"]
        if "unverified_claims" in existing:
            uc = dict(existing["unverified_claims"])
            uc["count"] = source_kind_counts.get("printables", 0)
            out["unverified_claims"] = uc
        if "audit_findings" in existing:
            out["audit_findings"] = existing["audit_findings"]

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        default=str(ROOT / "corpus" / "normalized"),
        help="Directory containing normalized *.jsonl files",
    )
    parser.add_argument(
        "--matrix",
        default=str(ROOT / "docs" / "coverage_matrix.json"),
        help="Path to coverage_matrix.json",
    )
    parser.add_argument(
        "--coverage-out",
        default=str(ROOT / "corpus" / "coverage_report.json"),
        help="Output path for coverage_report.json",
    )
    parser.add_argument(
        "--license-out",
        default=str(ROOT / "corpus" / "license_report.json"),
        help="Output path for license_report.json",
    )
    parser.add_argument(
        "--existing-license-report",
        default=str(ROOT / "corpus" / "license_report.json"),
        help="Existing license_report.json to preserve historical audit blocks from",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional ISO-8601 UTC timestamp to embed as 'generated'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing output files",
    )
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus_dir)
    matrix_path = Path(args.matrix)
    cov_out = Path(args.coverage_out)
    lic_out = Path(args.license_out)
    existing_lic_path = Path(args.existing_license_report)

    records = load_records(corpus_dir)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    existing_lic = None
    if lic_out.exists():
        existing_lic = json.loads(lic_out.read_text(encoding="utf-8"))
    elif existing_lic_path.exists():
        existing_lic = json.loads(existing_lic_path.read_text(encoding="utf-8"))

    generated = args.timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cov_report = build_coverage_report(records, matrix, generated)
    lic_report = build_license_report(records, generated, existing=existing_lic)

    if not args.dry_run:
        cov_out.write_text(
            json.dumps(cov_report, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        lic_out.write_text(
            json.dumps(lic_report, indent=1, ensure_ascii=False), encoding="utf-8"
        )

    print(
        f"records={cov_report['totals']['records']} "
        f"accepted={cov_report['totals']['accepted']} "
        f"substantive={cov_report['totals']['substantive']} "
        f"reference_only={cov_report['totals']['reference_only']} "
        f"families={cov_report['totals']['families']} "
        f"met={len(cov_report['met'])}/{len(cov_report['categories'])} "
        f"policy={lic_report['policy_checks']['verdict']}"
    )
    if cov_report["short"]:
        print("short:", cov_report["short_detail"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
