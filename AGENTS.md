# Agent notes

## Mitrix via Tailscale

Cloud Agents can reach Mitrix over Tailscale userspace networking + SOCKS SSH.

Required Cursor secrets (injected at agent start):

- `TS_AUTH_KEY` — Tailscale **auth** key (`tskey-auth-...`), reusable preferred
- `MITRIX_SSH_KEY` — private SSH key contents
- `MITRIX_SSH_USER` — SSH user (e.g. `root`)
- `MITRIX_SSH_HOST` — Mitrix Tailscale hostname or MagicDNS name

Steps:

```bash
# 1. Install Tailscale if needed
curl -fsSL https://tailscale.com/install.sh | sh

# 2. Start userspace networking (no systemd in many Cloud Agent pods)
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

# 5. SSH via Tailscale SOCKS
export ALL_PROXY=socks5h://localhost:1055/
ssh -i /tmp/mitrix_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  "$MITRIX_SSH_USER@$MITRIX_SSH_HOST" 'hostname; whoami; uptime'
```

If that prints hostname / root / uptime, you’re connected.

Notes:

- Secrets are injected when the agent starts; adding them mid-run requires restarting the agent.
- Prefer `socks5h://` so DNS resolves through the SOCKS proxy on the tailnet.
- If `tailscale up` reports `invalid key: API key ... not valid`, regenerate a Tailscale auth key (not an API key) and update `TS_AUTH_KEY`.
- Keep `tailscaled` running in the background (tmux/`nohup`); a plain `&` in a one-shot shell may exit with the shell.
