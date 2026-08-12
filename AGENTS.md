# Agent notes

## Mitrix via Tailscale

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

If that prints hostname / root / uptime, you’re connected.

Notes:
- Secrets are injected when the agent starts; adding them mid-run requires restarting the agent.
- Keep `tailscaled` running in the background (tmux/`nohup`); a plain `&` in a one-shot shell may exit with the shell.
- If `tailscale up` reports `invalid key: API key ... not valid`, regenerate a Tailscale **auth** key (not an API key) and update `TS_AUTH_KEY`.