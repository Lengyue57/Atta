# -*- coding: utf-8 -*-
"""Installer contract: absolute paths, ZCode-plugin, and Antigravity install."""

from __future__ import annotations

import json
from pathlib import Path

from atta.agent import install


def test_mcp_entry_uses_absolute_paths():
    entry = install.mcp_entry()
    assert entry["command"].startswith("/")
    assert entry["cwd"].startswith("/")
    assert entry["env"]["ATTA_ROOT"].startswith("/")
    assert entry["args"] == ["-m", "atta.agent.mcp"]


def test_zcode_plugin_dry_run_reports_real_paths():
    out = install.install_zcode_plugin(dry_run=True)
    assert out["ok"] is True
    assert out["dry_run"] is True
    assert out["marketplace_exists"] is True
    assert out["marketplace"].endswith("plugins/marketplace.json")
    server = out["config"]["mcpServers"]["atta"]
    assert server["command"].startswith("/")
    assert server["cwd"].startswith("/")
    assert server["args"] == ["-m", "atta.agent.mcp"]


def test_claude_and_cursor_snippets_are_absolute():
    for fn in (install.claude_config, install.cursor_config):
        cfg = fn()
        text = str(cfg)
        assert "atta.agent.mcp" in text


def test_antigravity_dry_run_does_not_write(tmp_path):
    target = tmp_path / "antigravity-dry"
    out = install.install_antigravity(target_dir=target, dry_run=True)
    assert out["ok"] is True
    assert out["host"] == "antigravity"
    assert out["dry_run"] is True
    assert out["written"] is False
    assert out["created"] is False
    assert not target.exists()

    assert out["plugin"]["name"] == "atta"
    assert out["plugin"]["version"] == "1.2.3-beta.1"
    server = out["config"]["mcpServers"]["atta"]
    assert server["command"].startswith("/")
    assert server["args"] == ["-m", "atta.agent.mcp"]
    assert server["env"]["ATTA_ROOT"].startswith("/")
    assert server["env"]["PYTHONPATH"] == server["env"]["ATTA_ROOT"]
    assert server["env"]["PYTHONUNBUFFERED"] == "1"


def test_antigravity_default_without_create_does_not_write(tmp_path):
    target = tmp_path / "antigravity-no-create"
    out = install.install_antigravity(target_dir=target, create=False, dry_run=False)
    assert out["ok"] is True
    assert out["written"] is False
    assert not target.exists()


def test_antigravity_install_writes_layout_and_preserves_workspace(tmp_path):
    target = tmp_path / "atta-plugin"
    # Pre-existing runtime workspace data and a stale skill file
    ws_file = target / "_atta" / "models.json"
    ws_file.parent.mkdir(parents=True)
    ws_file.write_text('{"keep": true}\n', encoding="utf-8")
    stale_skill = target / "skills" / "atta" / "references" / "stale.md"
    stale_skill.parent.mkdir(parents=True)
    stale_skill.write_text("stale\n", encoding="utf-8")

    out = install.install_antigravity(target_dir=target, create=True)
    assert out["ok"] is True
    assert out["written"] is True
    assert out["created"] is True

    plugin_json = json.loads((target / "plugin.json").read_text(encoding="utf-8"))
    assert plugin_json["name"] == "atta"
    assert plugin_json["version"] == "1.2.3-beta.1"
    assert plugin_json["logo"] == "assets/logo.svg"

    mcp_cfg = json.loads((target / "mcp_config.json").read_text(encoding="utf-8"))
    server = mcp_cfg["mcpServers"]["atta"]
    assert Path(server["command"]).is_absolute()
    assert Path(server["env"]["ATTA_ROOT"]).is_absolute()
    assert server["env"]["PYTHONPATH"] == server["env"]["ATTA_ROOT"]
    assert server["env"]["PYTHONUNBUFFERED"] == "1"

    assert (target / "assets" / "logo.svg").is_file()
    assert (target / "skills" / "atta" / "SKILL.md").is_file()
    for ref in ("assembly.md", "quality.md", "solving.md", "spec-v3.md", "troubleshooting.md"):
        assert (target / "skills" / "atta" / "references" / ref).is_file()

    # Stale skill file pruned, _atta workspace preserved untouched
    assert not stale_skill.exists()
    assert ws_file.read_text(encoding="utf-8") == '{"keep": true}\n'


def test_antigravity_cli_subcommand(tmp_path, capsys):
    target = tmp_path / "cli-atta"
    rc = install.main(["antigravity", "--target", str(target), "--print"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert not target.exists()

    rc = install.main(["antigravity", "--target", str(target), "--create"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["written"] is True
    assert (target / "mcp_config.json").is_file()
