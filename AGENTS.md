# Agent notes

## Mitrix via Tailscale

Cloud Agents can reach Mitrix over Tailscale userspace networking + SOCKS SSH.

Required Cursor secrets (injected at agent start):

- `TS_AUTH_KEY` — Tailscale auth key
- `MITRIX_SSH_KEY` — private SSH key contents
- `MITRIX_SSH_USER` — SSH user (e.g. `root`)
- `MITRIX_SSH_HOST` — Mitrix Tailscale hostname or IP

Steps:

```bash
# 1. Install Tailscale if needed
curl -fsSL https://tailscale.com/install.sh | sh

# 2. Start userspace networking
sudo tailscaled --tun=userspace-networking \
  --outbound-http-proxy-listen=localhost:1054 \
  --socks5-server=localhost:1055 &

# 3. Join tailnet
sudo tailscale up --authkey="$TS_AUTH_KEY" --hostname=cursor-cloud-agent

# 4. Write key
umask 077
printenv MITRIX_SSH_KEY > /tmp/mitrix_key
chmod 600 /tmp/mitrix_key

# 5. SSH via Tailscale SOCKS
export ALL_PROXY=socks5h://localhost:1055/
ssh -i /tmp/mitrix_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  "$MITRIX_SSH_USER@$MITRIX_SSH_HOST" 'hostname; whoami; uptime'
```

If that prints hostname / root / uptime, you’re connected.

Notes:

- Secrets are injected when the agent starts; adding them mid-run requires restarting the agent.
- Prefer `socks5h://` so DNS resolves through the SOCKS proxy on the tailnet.
