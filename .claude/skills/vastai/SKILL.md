---
name: vastai
description: Manage Vast.ai GPU instances for ML/AI workloads. Use when the user asks to rent a GPU, run training/inference remotely, search for GPU instances, connect to a remote Jupyter kernel, start/stop/destroy Vast.ai instances, check GPU spending, or says "start me the instance". Triggers on vast.ai, vastai, "rent a GPU", "remote GPU", "GPU instance", "cloud GPU", "start the instance".
---

# Vast.ai GPU Instance Management

Budget: <$10/month. Always run the spending report at session start and end.

## CLI Commands Summary (required)

After completing ANY action from this skill, always end by printing a "Commands I ran for you" block listing the exact CLI commands executed, so the user can learn and replicate. Format:

```
**Commands I ran for you:**
\`\`\`bash
command1
command2
...
\`\`\`
```

Use the actual values (IDs, hosts, ports), not placeholders.

## Session Start / End

Run `python3 scripts/vastai_spending.py` (in this skill's directory) to show current credit, monthly spend, and active instances. Do this at the **start** and **end** of every Vast.ai session. Warn if spend exceeds $8.

## Gotchas & Lessons Learned

- **`vastai login` doesn't exist.** Auth is `vastai tfa login --method-type totp --code <CODE>`. Omitting `--method-type totp` gives "2FA method not found".
- **`success: False` on create doesn't mean failure.** The response may still contain `new_contract` with a valid ID. Always check `vastai show instances` after creation.
- **Instances can get stuck in `created`/`stopped`.** If `vastai start instance` returns "Required resources are currently unavailable" and the status doesn't move to `loading` within ~30s, destroy and pick a different offer. Don't wait forever.
- **SSH host/port changes every rental.** Always parse `ssh_host` and `ssh_port` from `vastai show instances --raw` and update `~/.ssh/config` BEFORE telling user to connect.
- **SCP needs host key flags** to avoid interactive prompts:
  ```bash
  scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P <PORT> <FILE> root@<SSH_HOST>:/workspace/
  ```
- **Cursor terminal cwd error.** After connecting via Remote-SSH, Cursor tries to use the local Mac home path as the terminal cwd on the remote Linux machine. Fix: set the remote machine-level VS Code setting during startup:
  ```bash
  ssh vastai-gpu "mkdir -p /root/.vscode-server/data/Machine && echo '{\"terminal.integrated.cwd\": \"/workspace/ARENA\"}' > /root/.vscode-server/data/Machine/settings.json"
  ```
- **torchvision/torch version mismatch.** The base PyTorch image may ship with mismatched `torchvision` (e.g. `cu124` vs `cu130`), causing `RuntimeError: operator torchvision::nms does not exist` when importing from `transformers`. Fix: `pip install torchvision --upgrade` after instance starts. The startup script handles this automatically.
- **Jupyter direct port is NOT `direct_port_start`.** That's the SSH port. Jupyter runs on container port `8080` — get the mapped host port from `ports["8080/tcp"][0]["HostPort"]` in the instance JSON.
- **Loading takes 1-3 minutes typically.** Poll every 15s. Status progression: `created` → `loading` → `running`. May briefly show empty string during transitions — keep polling.

## Authentication

```bash
# Check if auth works
vastai show user --raw

# If 401 / 2FA error, authenticate with TOTP:
vastai tfa login --method-type totp --code <6-digit-code>
```

Ask user for their authenticator code if 2FA is needed.

## Search for Instances

User prefers: RTX 4090, 1 GPU, high reliability, sorted by price, sufficient disk (>40GB).

```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.95 inet_down>200' -o 'dph' --limit 10
```

When presenting results, highlight: price ($/hr), disk space, DLP score, location, and max rental days. Flag instances with <20GB disk as potentially too small for ML workloads.

## Create an Instance

```bash
vastai create instance <OFFER_ID> \
  --image pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel \
  --disk 50 \
  --onstart-cmd 'pip install jupyterlab transformers einops jaxtyping transformer_lens circuitsvis ipykernel plotly scipy'
```

After creation, start it and poll status:

```bash
vastai start instance <ID>
vastai show instances
```

## "Start me the instance" — Full Startup Flow

When user says "start me the instance" or similar, **ask which mode they want**:

- **`jupyter`** (default) — Direct HTTPS JupyterLab URL. No SSH needed. Works on restricted networks (public WiFi, firewalls). Git via JupyterLab terminal.
- **`ssh`** — SSH tunnel to JupyterLab. Requires SSH access. Enables git agent forwarding for push/pull.

```bash
bash scripts/vastai_start.sh jupyter   # direct HTTPS URL (default)
bash scripts/vastai_start.sh ssh       # SSH tunnel mode
```

Both modes handle the full flow:
1. Checks spending and auth
2. Finds existing instance or searches/creates a new one (auto-retries up to 3 offers if stuck)
3. Prints pricing ($/hr, $/day, $/mo) when instance starts
4. Polls until running (up to 4 min)

**Jupyter mode** then: prints the direct `https://IP:PORT/?token=TOKEN` URL — done.

**SSH mode** then additionally:
5. Updates `~/.ssh/config` with correct host/port and `ForwardAgent yes`
6. Loads SSH key into agent, waits for SSH
7. Installs Python packages, clones/pulls ARENA repo
8. Sets git identity, verifies GitHub auth
9. Starts JupyterLab, opens SSH tunnel to `localhost:8888`

## Connect: Two Modes

### Jupyter mode (default) — Direct HTTPS URL

Instance is created with `--jupyter --jupyter-lab --direct` flags. Vast.ai exposes JupyterLab on a direct HTTPS port. No SSH needed.

- URL format: `https://<PUBLIC_IP>:<DIRECT_PORT>/?token=<JUPYTER_TOKEN>`
- Browser may warn about self-signed certificate — click through
- Git: use the JupyterLab terminal (no agent forwarding — set up credentials manually if needed)
- Best for: restricted networks, public WiFi, firewalls that block SSH

### SSH mode — Tunnel-based JupyterLab

Instance is created with SSH runtype. JupyterLab is started manually and tunneled via SSH.

```bash
# Start JupyterLab on remote
ssh -A vastai-gpu "nohup jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --ServerApp.token='' --ServerApp.password='' --notebook-dir=/workspace/ARENA > /tmp/jupyter.log 2>&1 &"

# Open SSH tunnel locally
ssh -A -L 8888:localhost:8888 -N vastai-gpu &

# Open http://localhost:8888
```

- Git: agent forwarding works, push/pull seamlessly
- Stop tunnel: `pkill -f "ssh.*-L 8888:localhost:8888.*vastai-gpu"`
- Best for: unrestricted networks, need git push/pull

## Git Setup on Remote

Run this automatically during the startup flow, after the instance is running and SSH config is updated:

1. Ensure local SSH agent has keys loaded:
   ```bash
   ssh-add -l || ssh-add ~/.ssh/id_rsa
   ```
2. Clone the repo on the remote (if not already present) and set SSH remote URL:
   ```bash
   ssh -A vastai-gpu "cd /workspace && git clone <REPO_HTTPS_URL> <REPO_NAME> 2>/dev/null; cd <REPO_NAME> && git remote set-url origin <REPO_SSH_URL>"
   ```
3. Verify auth works:
   ```bash
   ssh -A vastai-gpu "ssh -o StrictHostKeyChecking=no -T git@github.com 2>&1 || true"
   ```

Get the repo URL from the local working directory's git remote (`git remote get-url origin`). The user can then `git pull` and `git push` from the Cursor terminal on the remote.

### Updating SSH config automatically

When starting an instance, always update `~/.ssh/config` with the current host/port for the `vastai-gpu` entry. The SSH host and port can change between rentals. Parse them from `vastai show instances --raw` (fields: `ssh_host`, `ssh_port`).

## Check Connection

When user asks to check connection, verify all three layers:

```bash
# 1. Instance running?
vastai show instances

# 2. SSH working?
ssh -A -o ConnectTimeout=5 vastai-gpu "echo ok"

# 3. JupyterLab process alive?
ssh -A vastai-gpu "pgrep -f jupyter-lab && echo 'JupyterLab running'"

# 4. Tunnel alive locally?
curl -s -o /dev/null -w "%{http_code}" http://localhost:8888/
```

If SSH is refused but instance shows running, reboot the instance:
```bash
vastai reboot instance <ID>
```
Then wait for SSH, restart JupyterLab, and reopen the tunnel.

## Stop vs Destroy

- **Stop** — preserves data, still incurs small storage cost:
  ```bash
  vastai stop instance <ID>
  ```
- **Destroy** — deletes everything, stops all billing:
  ```bash
  vastai destroy instance <ID> --yes
  ```

Always confirm with user before destroying. Remind them to download any work first.

## End Session Checklist

1. Run spending report (`scripts/vastai_spending.py`)
2. Ask: keep instance running, stop, or destroy?
3. If destroying, offer to download files first:
   ```bash
   scp -P <PORT> root@<SSH_HOST>:/workspace/<FILE> .
   ```
4. Execute the chosen action
5. Confirm no instances are left running unexpectedly: `vastai show instances`
