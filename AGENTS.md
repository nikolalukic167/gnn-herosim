# Agent notes

## Mitrix-first execution policy for Cloud Agents

When Mitrix secrets are present, default to running user task commands on Mitrix over SSH.
Do not run build/test/dev commands on the Cloud Agent VM unless the task explicitly requires local-only work.

Use the Cloud Agent VM only for:
- Tailscale bootstrap
- SSH key materialization
- File transfer/sync steps needed to support remote execution

Required Cursor secrets (injected at agent start):
- `TS_AUTH_KEY` - Tailscale auth key (`tskey-auth-...`)
- `MITRIX_SSH_KEY` - private SSH key contents
- `MITRIX_SSH_USER` - SSH user (example: `root`)
- `MITRIX_SSH_HOST` - Mitrix Tailscale hostname or MagicDNS name (example: `mitrix`)

### 1) One-time local bootstrap (per Cloud Agent VM)

```bash
# install dependencies if needed
curl -fsSL https://tailscale.com/install.sh | sh
sudo apt-get install -y netcat-openbsd >/dev/null

# keep tailscaled alive in tmux
SESSION_NAME="tailscale-mitrix"
tmux -f /exec-daemon/tmux.portal.conf has-session -t "=$SESSION_NAME" 2>/dev/null || \
  tmux -f /exec-daemon/tmux.portal.conf new-session -d -s "$SESSION_NAME" -c "$PWD" -- "${SHELL:-bash}" -l
tmux -f /exec-daemon/tmux.portal.conf send-keys -t "$SESSION_NAME:0.0" \
  'pgrep -x tailscaled >/dev/null || (sudo mkdir -p /var/run/tailscale /var/lib/tailscale && sudo tailscaled --tun=userspace-networking --outbound-http-proxy-listen=localhost:1054 --socks5-server=localhost:1055 --statedir=/var/lib/tailscale)' C-m

# join tailnet
sudo tailscale up --authkey="$TS_AUTH_KEY" --hostname=cursor-cloud-agent

# materialize SSH key safely
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
```

### 2) Define a persistent SSH helper and use it for all work

```bash
mitrix_ssh() {
  ssh -i /tmp/mitrix_key \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ControlMaster=auto \
    -o ControlPersist=10m \
    -o ControlPath=/tmp/mitrix-ctrl-%C \
    -o ProxyCommand='nc -x 127.0.0.1:1055 -X 5 %h %p' \
    "$MITRIX_SSH_USER@$MITRIX_SSH_HOST" "$@"
}
```

### 3) Keep a remote tmux session on Mitrix and run commands there

```bash
# create/reuse remote session
mitrix_ssh "tmux has-session -t cursor-agent 2>/dev/null || tmux new-session -d -s cursor-agent"

# run commands remotely (examples)
mitrix_ssh "hostname; whoami; uptime"
mitrix_ssh "cd /path/on/mitrix && ls"
mitrix_ssh "cd /path/on/mitrix && pipenv run python3 -m pytest"

# interactive attach when needed
mitrix_ssh "tmux attach -t cursor-agent"
```

Operational rules for agents:
- Execute user task commands via `mitrix_ssh` by default.
- Keep `tailscaled` running in tmux for the full task duration.
- Reconnect and continue if SSH or Tailscale drops.
- If `tailscale up` reports `invalid key: API key ... not valid`, replace `TS_AUTH_KEY` with a Tailscale auth key (not an API key).