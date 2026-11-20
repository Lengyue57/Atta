# -*- coding: utf-8 -*-
"""Verify package manifests, plugin metadata, docs, and skills match live counts."""

from __future__ import annotations

import json
import re
from pathlib import Path

from atta import engine, version
from atta.agent import tools

REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _pyproject_version() -> str:
    text = _read("pyproject.toml")
    try:
        import tomllib
        return tomllib.loads(text)["project"]["version"]
    except ImportError:
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        assert m, "version not found in pyproject.toml"
        return m.group(1)


def _semver_from_pep440(py_ver: str) -> str:
    # "1.2.3b1" -> "1.2.3-beta.1"
    m = re.fullmatch(r"(\d+\.\d+\.\d+)b(\d+)", py_ver)
    if m:
        return f"{m.group(1)}-beta.{m.group(2)}"
    return py_ver


def test_package_and_plugin_versions_match():
    py_ver = version.VERSION
    assert py_ver == "1.2.3b1"
    js_ver = _semver_from_pep440(py_ver)
    assert js_ver == "1.2.3-beta.1"

    assert _pyproject_version() == py_ver

    root_pkg = json.loads(_read("package.json"))
    plugin_pkg = json.loads(_read("plugin/package.json"))
    zcode_pkg = json.loads(_read("plugins/atta/.zcode-plugin/plugin.json"))
    market = json.loads(_read("plugins/marketplace.json"))
    ag_pkg = json.loads(_read("plugins/antigravity/plugin.json"))

    assert root_pkg["version"] == js_ver
    assert plugin_pkg["version"] == js_ver
    assert zcode_pkg["version"] == js_ver
    assert market["plugins"][0]["version"] == js_ver
    assert ag_pkg["version"] == js_ver

    index_js = _read("plugin/lib/index.js")
    m = re.search(r"const VERSION = '([^']+)'", index_js)
    assert m and m.group(1) == js_ver


def test_live_engine_counts():
    cat = engine.catalog()
    assert len(tools.TOOLS) == 12
    assert len(cat["recipes"]) == 46
    assert len(cat["materials"]) == 53
    assert len(cat["features"]) >= 56


def test_readmes_and_release_notes_match_version_and_counts():
    py_ver = version.VERSION
    js_ver = _semver_from_pep440(py_ver)

    readme = _read("README.md")
    readme_zh = _read("README.zh-CN.md")
    release = _read("docs/RELEASE-1.2.3beta.md")

    for text in (readme, readme_zh, release):
        assert py_ver in text
        assert js_ver in text
        assert "1.2.3beta" in text
        assert "12" in text
        assert "46" in text
        assert "53" in text
        assert "56" in text
        assert "1.2.2b3" not in text
        assert "1.2.2-beta.3" not in text
        assert "23 recipes" not in text
        assert "23 个 recipe" not in text


def test_skills_and_architecture_docs_have_no_stale_counts():
    skill_zh = _read("plugin/skills/atta/SKILL.md")
    skill_en = _read("plugins/atta/skills/atta-modeling/SKILL.md")
    spec_ref = _read("plugin/skills/atta/references/spec-v3.md")
    trouble_ref = _read("plugin/skills/atta/references/troubleshooting.md")
    arch = _read("docs/ARCHITECTURE.md")
    roadmap = _read("docs/ROADMAP.md")

    assert "1.2.3beta" in skill_zh
    assert "53 种" in skill_zh
    assert "46 个 recipe" in skill_zh
    assert "47 种" not in skill_zh
    assert "1.2.2beta" not in skill_zh

    assert "1.2.3beta" in skill_en
    assert "46 recipes" in skill_en
    assert "53 materials" in skill_en
    assert "56+ ops" in skill_en
    assert "1.2.2beta" not in skill_en

    assert "56+" in spec_ref
    assert "25 个" not in spec_ref
    assert "十二" in trouble_ref
    assert "十动词" not in trouble_ref

    assert "12 tool contracts" in arch
    assert "56+ ops" in arch
    assert "46 curated" in arch
    assert "10 tool contracts" not in arch
    assert "25 graph ops" not in arch
    assert "Graph ops are 25" not in arch

    assert "12 verbs" in roadmap
    assert "12-tool script" in roadmap
    assert "10 verbs" not in roadmap
    assert "10-tool script" not in roadmap
