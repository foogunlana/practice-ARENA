#!/usr/bin/env bash
set -euo pipefail

# === Config ===
GPU="RTX_4090"
IMAGE="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
DISK=50
ONSTART='pip install jupyterlab transformers einops jaxtyping transformer_lens circuitsvis ipykernel plotly scipy ipywidgets widgetsnbextension torchvision --upgrade'
REPO_SSH="git@github.com:foogunlana/practice-ARENA.git"
REPO_NAME="ARENA"
HOME_DIR="/Users/folusoogunlana"
SSH_KEY="$HOME_DIR/.ssh/id_rsa"
SSH_CONFIG="$HOME_DIR/.ssh/config"
BUDGET=10.00
MAX_POLL=16       # 16 x 15s = 4 min
POLL_INTERVAL=15
MAX_RETRIES=3     # try up to 3 different offers

# === Launch mode: "jupyter" (direct HTTPS URL) or "ssh" (SSH tunnel) ===
MODE="${1:-jupyter}"

# === Colors ===
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()   { echo -e "${RED}[ERROR]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }

info "Launch mode: $MODE"

# === 1. Spending Report ===
info "Checking Vast.ai spending..."
CREDIT=$(vastai show user --raw 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('credit',0))" 2>/dev/null || echo "?")
echo "  Credit balance: \$$CREDIT"
if python3 -c "exit(0 if float('$CREDIT') < 2 else 1)" 2>/dev/null; then
  warn "Low credit balance! Consider topping up."
fi

# === 2. Auth Check ===
if ! vastai show user --raw &>/dev/null; then
  err "Vast.ai auth failed. Run: vastai tfa login --method-type totp --code <YOUR_CODE>"
  exit 1
fi
ok "Authenticated with Vast.ai"

# === 3. Check for existing instances ===
INSTANCES=$(vastai show instances --raw 2>/dev/null || echo "[]")
EXISTING_ID=$(echo "$INSTANCES" | python3 -c "
import sys,json
d = json.load(sys.stdin)
for i in d:
    print(i['id'])
    break
else:
    print('')
" 2>/dev/null)

if [ -n "$EXISTING_ID" ]; then
  EXISTING_STATUS=$(echo "$INSTANCES" | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(d[0].get('actual_status','unknown'))
" 2>/dev/null)

  if [ "$EXISTING_STATUS" = "running" ]; then
    ok "Instance $EXISTING_ID already running"
    INSTANCE_ID="$EXISTING_ID"
  else
    info "Found existing instance $EXISTING_ID (status: $EXISTING_STATUS). Starting..."
    vastai start instance "$EXISTING_ID" 2>&1 || true
    INSTANCE_ID="$EXISTING_ID"
  fi
else
  # === 4. Search and create ===
  info "No existing instances. Searching for $GPU offers..."

  # Add direct_port_count filter for jupyter mode
  SEARCH_FILTER="gpu_name=${GPU} num_gpus=1 reliability>0.95 inet_down>200"
  if [ "$MODE" = "jupyter" ]; then
    SEARCH_FILTER="$SEARCH_FILTER direct_port_count>1"
  fi

  for ATTEMPT in $(seq 1 $MAX_RETRIES); do
    OFFERS=$(vastai search offers "$SEARCH_FILTER" -o 'dph' --limit 10 --raw 2>/dev/null)
    # Pick the cheapest offer with >40GB disk, skipping previously failed ones
    OFFER_ID=$(echo "$OFFERS" | python3 -c "
import sys,json
skip = set('${SKIP_OFFERS:-}'.split(',')) - {''}
d = json.load(sys.stdin)
for o in d:
    if o.get('disk_space',0) > 40 and str(o['id']) not in skip:
        print(o['id'])
        break
" 2>/dev/null)

    if [ -z "$OFFER_ID" ]; then
      err "No suitable offers found"
      exit 1
    fi

    info "Attempt $ATTEMPT: Creating instance from offer $OFFER_ID..."
    if [ "$MODE" = "jupyter" ]; then
      CREATE_OUT=$(vastai create instance "$OFFER_ID" \
        --image "$IMAGE" \
        --disk "$DISK" \
        --jupyter --jupyter-lab --jupyter-dir /workspace \
        --direct \
        --onstart-cmd "$ONSTART" 2>&1)
    else
      CREATE_OUT=$(vastai create instance "$OFFER_ID" \
        --image "$IMAGE" \
        --disk "$DISK" \
        --onstart-cmd "$ONSTART" 2>&1)
    fi
    echo "  $CREATE_OUT"

    INSTANCE_ID=$(echo "$CREATE_OUT" | python3 -c "
import sys,ast
for line in sys.stdin:
    if 'new_contract' in line:
        d = ast.literal_eval(line.split('. ',1)[-1]) if '. ' in line else ast.literal_eval(line)
        print(d.get('new_contract',''))
        break
" 2>/dev/null || echo "")

    if [ -z "$INSTANCE_ID" ]; then
      # Try parsing from show instances
      sleep 3
      INSTANCE_ID=$(vastai show instances --raw 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['id'] if d else '')" 2>/dev/null)
    fi

    if [ -z "$INSTANCE_ID" ]; then
      warn "Failed to get instance ID. Retrying..."
      SKIP_OFFERS="${SKIP_OFFERS:-},$OFFER_ID"
      continue
    fi

    info "Instance created: $INSTANCE_ID. Starting..."
    vastai start instance "$INSTANCE_ID" 2>&1 || true

    # Quick check: does it move to loading within 30s?
    STUCK=true
    for j in $(seq 1 3); do
      sleep 10
      STATUS=$(vastai show instances --raw 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
inst=[x for x in d if x['id']==$INSTANCE_ID]
print(inst[0]['actual_status'] if inst else 'gone')
" 2>/dev/null)
      if [ "$STATUS" = "loading" ] || [ "$STATUS" = "running" ]; then
        STUCK=false
        break
      fi
    done

    if [ "$STUCK" = true ] && [ "$ATTEMPT" -lt "$MAX_RETRIES" ]; then
      warn "Instance $INSTANCE_ID stuck (status: $STATUS). Destroying and trying next offer..."
      vastai destroy instance "$INSTANCE_ID" --yes 2>&1 || true
      SKIP_OFFERS="${SKIP_OFFERS:-},$OFFER_ID"
      sleep 2
      continue
    fi
    break
  done
fi

# === 5. Poll until running ===
info "Waiting for instance $INSTANCE_ID to be ready..."
for i in $(seq 1 $MAX_POLL); do
  STATUS=$(vastai show instances --raw 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
inst=[x for x in d if x['id']==$INSTANCE_ID]
print(inst[0]['actual_status'] if inst else 'gone')
" 2>/dev/null)
  echo "  [$i/$MAX_POLL] Status: $STATUS"
  if [ "$STATUS" = "running" ]; then
    ok "Instance is running!"
    # Print pricing info
    vastai show instances --raw 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
inst=[x for x in d if x['id']==$INSTANCE_ID][0]
dph=inst.get('dph_total',0)
gpu=inst.get('gpu_name','?')
loc=inst.get('geolocation','?')
print(f'  GPU: {gpu} | \${dph:.4f}/hr (\${dph*24:.2f}/day, \${dph*24*30:.2f}/mo) | {loc}')
"
    break
  fi
  if [ "$STATUS" = "gone" ]; then
    err "Instance disappeared"
    exit 1
  fi
  if [ "$i" -eq "$MAX_POLL" ]; then
    err "Timed out waiting for instance to start"
    exit 1
  fi
  sleep $POLL_INTERVAL
done

# === 6. Get instance details ===
INST_DATA=$(vastai show instances --raw 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
inst=[x for x in d if x['id']==$INSTANCE_ID][0]
# Jupyter runs on port 8080 inside the container — get the mapped host port
ports = inst.get('ports', {})
jupyter_port = ports.get('8080/tcp', [{}])[0].get('HostPort', str(inst.get('direct_port_start','')))
print(inst['ssh_host'], inst['ssh_port'], inst.get('public_ipaddr',''), jupyter_port, inst.get('jupyter_token',''), inst.get('dph_total',0), inst.get('image_runtype','ssh'))
")
read SSH_HOST SSH_PORT PUBLIC_IP JUPYTER_PORT JUPYTER_TOKEN DPH IMAGE_RUNTYPE <<< "$INST_DATA"

# === Jupyter mode: direct HTTPS URL (no SSH needed) ===
if [ "$MODE" = "jupyter" ]; then
  JUPYTER_URL="https://${PUBLIC_IP}:${JUPYTER_PORT}/?token=${JUPYTER_TOKEN}"
  ok "JupyterLab URL: $JUPYTER_URL"

  echo ""
  echo -e "${GREEN}========================================${NC}"
  echo -e "${GREEN} Instance ready!${NC}"
  echo -e "${GREEN}========================================${NC}"
  echo "  Instance ID:  $INSTANCE_ID"
  echo "  GPU:          $GPU"
  echo "  Cost:         \$$DPH/hr"
  echo "  Mode:         Direct JupyterLab (no SSH)"
  echo ""
  echo "  Open in browser:"
  echo "  $JUPYTER_URL"
  echo ""
  echo "  Note: Browser may warn about self-signed certificate — click through it."
  echo "  Git: use the JupyterLab terminal for git pull/push."
  echo ""
  exit 0
fi

# === SSH mode: tunnel-based JupyterLab ===
ok "SSH: $SSH_HOST:$SSH_PORT"

info "Updating ~/.ssh/config..."
mkdir -p "$(dirname "$SSH_CONFIG")"
python3 -c "
import re, os
config_path = '$SSH_CONFIG'
new_block = '''Host vastai-gpu
    HostName $SSH_HOST
    Port $SSH_PORT
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ForwardAgent yes
'''
if os.path.exists(config_path):
    with open(config_path) as f:
        content = f.read()
    content = re.sub(r'Host vastai-gpu\n(\s+\S.*\n)*', '', content).strip()
    content = new_block + '\n\n' + content if content else new_block
else:
    content = new_block
with open(config_path, 'w') as f:
    f.write(content.strip() + '\n')
"
ok "SSH config updated"

# === 7. Load SSH key into agent ===
if ! ssh-add -l &>/dev/null; then
  info "Loading SSH key into agent..."
  ssh-add "$SSH_KEY" 2>/dev/null || warn "Could not add SSH key. Git push may not work."
fi

# === 8. Wait for SSH to be ready ===
info "Waiting for SSH to accept connections..."
for i in $(seq 1 12); do
  if ssh -A -o ConnectTimeout=5 vastai-gpu "echo ok" &>/dev/null; then
    ok "SSH connected"
    break
  fi
  if [ "$i" -eq 12 ]; then
    err "SSH not responding after 60s. Try: vastai reboot instance $INSTANCE_ID"
    exit 1
  fi
  sleep 5
done

# === 9. Install Python packages ===
info "Installing Python packages on remote..."
ssh -A vastai-gpu "pip install ipykernel torchvision plotly scipy ipywidgets widgetsnbextension --upgrade 2>&1 | tail -1"
ok "Packages installed"

# === 10. Clone/pull repo ===
info "Setting up $REPO_NAME repo on remote..."
ssh -A vastai-gpu "
  cd /workspace
  if [ -d '$REPO_NAME/.git' ]; then
    cd $REPO_NAME && git remote set-url origin $REPO_SSH && git pull 2>&1
  else
    git clone $REPO_SSH $REPO_NAME 2>&1
    cd $REPO_NAME && git remote set-url origin $REPO_SSH
  fi
"
ok "Repo ready at /workspace/$REPO_NAME"

# === 11. Set git identity ===
info "Setting git identity on remote..."
LOCAL_EMAIL=$(git config --global user.email)
LOCAL_NAME=$(git config --global user.name)
ssh -A vastai-gpu "git config --global user.email '$LOCAL_EMAIL' && git config --global user.name '$LOCAL_NAME'"
ok "Git identity set ($LOCAL_NAME <$LOCAL_EMAIL>)"

# === 12. Verify git auth ===
info "Verifying GitHub auth via agent forwarding..."
ssh -A vastai-gpu "ssh -o StrictHostKeyChecking=no -T git@github.com 2>&1 || true" | head -1

# === 13. Start JupyterLab ===
info "Starting JupyterLab on remote..."
ssh -A vastai-gpu "pkill -f 'jupyter-lab' 2>/dev/null; nohup jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --ServerApp.token='' --ServerApp.password='' --notebook-dir=/workspace/$REPO_NAME > /tmp/jupyter.log 2>&1 &"
sleep 2

# === 14. Open SSH tunnel ===
info "Opening SSH tunnel (localhost:8888 → remote:8888)..."
pkill -f "ssh.*-L 8888:localhost:8888.*vastai-gpu" 2>/dev/null || true
ssh -A -L 8888:localhost:8888 -N vastai-gpu &
TUNNEL_PID=$!
sleep 2

if curl -s -o /dev/null -w "%{http_code}" http://localhost:8888/ 2>/dev/null | grep -q "200\|302"; then
  ok "JupyterLab ready at http://localhost:8888"
else
  warn "Tunnel started (PID $TUNNEL_PID) but JupyterLab may still be loading. Try http://localhost:8888 in a moment."
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Instance ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "  Instance ID:  $INSTANCE_ID"
echo "  GPU:          $GPU"
echo "  SSH:          ssh -A vastai-gpu"
echo "  Cost:         \$$DPH/hr"
echo "  Repo:         /workspace/$REPO_NAME"
echo "  Tunnel PID:   $TUNNEL_PID"
echo ""
echo "  JupyterLab:   http://localhost:8888"
echo "  To stop tunnel later: kill $TUNNEL_PID"
echo ""
