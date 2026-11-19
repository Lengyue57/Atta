# Atta 1.2.3beta — Antigravity Plugin

This directory holds the Antigravity plugin manifest template (`plugin.json`).
The installer materializes a full Antigravity plugin folder (by default
`~/.gemini/config/plugins/atta/`) containing:

- `plugin.json` — copied from `plugins/antigravity/plugin.json`
- `mcp_config.json` — stdio MCP entry with absolute `python`, `ATTA_ROOT`, `PYTHONPATH`
- `assets/logo.svg` — copied from `docs/assets/mark.svg`
- `skills/atta/` — synced from `plugin/skills/atta/` (`SKILL.md` and `references/*.md`)

## Install / Update

```bash
python3 -m atta.agent.install antigravity --print   # dry-run preview
python3 -m atta.agent.install antigravity --create  # write/update ~/.gemini/config/plugins/atta
```

Existing workspace state under `_atta/` inside the target plugin directory is
never modified or removed when updating.
