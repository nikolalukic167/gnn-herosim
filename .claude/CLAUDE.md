@../AGENTS.md

## Claude Code specifics

- Hooks: `.claude/hooks/post-tool-call.py` (tool-call provenance) and
  `.claude/hooks/record-hygiene.py` (advisory research-record check after edits to the
  watched record files), wired in `.claude/settings.local.json`.
- Skills: `.claude/skills/`. Agents: `.claude/agents/`.
- Codex-specific agent/skill ports and cloud-agent access notes live in `.codex/`; see
  `.codex/README.md`. Not needed for Claude Code sessions.
