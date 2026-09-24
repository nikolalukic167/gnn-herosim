# Repository Codex support

Adapted from the reusable contents of `.claude`; original files remain unchanged.

- `agents/*.toml`: eight project agents, with native `name`, `description`, and
  `developer_instructions`. Models and permissions inherit the session. Claude
  model aliases, tool allowlists, and effort metadata were not copied.
- `skills/*/SKILL.md`: three adapted skills. `.agents/skills/<name>` links here
  for repository discovery. The strict review remains explicit-only through
  `agents/openai.yaml` rather than Claude frontmatter.
- `scripts/check-record-hygiene.sh`: explicit replacement for the record-hygiene
  hook. Run after research-record edits and before finishing. It returns the real
  test exit status; it is not installed as an automatic tool hook.

Excluded: `settings.local.json` and `hooks/post-tool-call.py` (the Claude tool-event
provenance integration). No permissions, network services, notifications, model
selection, or global Codex configuration were imported. Root `AGENTS.md` is the
canonical instruction entry point for every agent, including Claude Code:
`.claude/CLAUDE.md` is now a thin `@../AGENTS.md` import plus a short Claude-only
appendix, so there is no second copy of the research narrative to keep in sync.

Migration corrections: lineage closure rewrites AGENTS.md's own standing-answer
block directly (the same file Claude and Codex both read now, not a routing-only
copy); incomplete gates are not declared falsified; training uses experiment
configs; dataset resume must verify full placement sweeps; cluster submission
honors authorization already given in the task.

Agent definitions are available to hosts supporting project custom agents. If a
host cannot select a custom agent, read its `developer_instructions` for the same
workflow without assuming that it was spawned. Reload/restart Codex if newly
installed definitions are not visible; installation does not demonstrate that
this already-running task has reloaded them.

Formats follow the official [custom-agent documentation](https://learn.chatgpt.com/docs/agent-configuration/subagents)
and [skill discovery documentation](https://learn.chatgpt.com/docs/build-skills).

## Cloud agent access: Mitrix via Tailscale

Cloud Agents can reach Mitrix over Tailscale userspace networking + SOCKS SSH.

Required Cursor secrets (injected at agent start):
- `TS_AUTH_KEY` — Tailscale **auth** key (`tskey-auth-...`), reusable preferred
- `MITRIX_SSH_KEY` — private SSH key contents
- `MITRIX_SSH_USER` — SSH user (e.g. `root`)
- `MITRIX_SSH_HOST` — Mitrix Tailscale hostname or MagicDNS name (e.g. `mitrix`)

Steps:

```bash
# 1. Install Tailscale if needed
curl -fsSL https://tailscale.com/install.sh | sh

# 2. Start userspace networking (keep this process alive: tmux/nohup)
sudo mkdir -p /var/run/tailscale /var/lib/tailscale
sudo tailscaled --tun=userspace-networking \
  --outbound-http-proxy-listen=localhost:1054 \
  --socks5-server=localhost:1055 \
  --statedir=/var/lib/tailscale &
# wait until localhost:1055 is listening

# 3. Join tailnet
sudo tailscale up --authkey="$TS_AUTH_KEY" --hostname=cursor-cloud-agent

# 4. Write key (Cursor often flattens PEM newlines to spaces — rebuild if needed)
umask 077

python3 - <<'PY'
import os, textwrap, pathlib
sk = os.environ["MITRIX_SSH_KEY"].strip()
begin, end = "-----BEGIN OPENSSH PRIVATE KEY-----", "-----END OPENSSH PRIVATE KEY-----"
if "\n" not in sk and sk.startswith(begin) and sk.endswith(end):
    b64 = "".join(sk[len(begin):-len(end)].split())
    sk = begin + "\n" + "\n".join(textwrap.wrap(b64, 70)) + "\n" + end + "\n"
pathlib.Path("/tmp/mitrix_key").write_text(sk)
os.chmod("/tmp/mitrix_key", 0o600)
PY

# 5. SSH via Tailscale SOCKS (OpenSSH needs ProxyCommand; ALL_PROXY alone is not enough)
sudo apt-get install -y netcat-openbsd >/dev/null

ssh -i /tmp/mitrix_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  -o ProxyCommand='nc -x 127.0.0.1:1055 -X 5 %h %p' \
  "$MITRIX_SSH_USER@$MITRIX_SSH_HOST" 'hostname; whoami; uptime'
```

If that prints hostname / root / uptime, you're connected.

Notes:
- Secrets are injected when the agent starts; adding them mid-run requires restarting the agent.
- Keep `tailscaled` running in the background (tmux/`nohup`); a plain `&` in a one-shot shell may exit with the shell.
- If `tailscale up` reports `invalid key: API key ... not valid`, regenerate a Tailscale **auth** key (not an API key) and update `TS_AUTH_KEY`.
