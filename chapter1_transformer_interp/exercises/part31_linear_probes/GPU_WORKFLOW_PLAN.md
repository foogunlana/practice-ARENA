---
author: claude
created: 2026-05-31
purpose: |
  Plan for an extract-on-GPU, analyze-on-laptop workflow for ARENA Ch 1.3 (Linear Probes),
  so the daily golden-slot work runs locally on an 8GB Apple Silicon Mac and the GPU is
  only needed for occasional activation extraction.
context: |
  During the May monthly retro, Bo flagged that GPU setup eats ~half of each morning
  session (vastai SSH/port-22 blocked at the library, RunPod/Colab cold-start stalls).
  Inspection of 1.3.1_Linear_Probes_exercises.ipynb showed it loads Llama-3.1-8B,
  Llama-2-13B and Llama-3.3-70B — too big for the 8GB Mac — but probe training itself
  runs on CPU. This plan splits the GPU-bound and CPU-bound halves.
references:
  - "1.3.1_Linear_Probes_exercises.ipynb"
  - conversation context (May 2026 monthly retro)
---

# Ch 1.3 GPU Workflow Plan — Extract on GPU, Analyze on Laptop

## The problem

GPU setup is stealing ~1 hour of a 2-hour golden slot, almost every session:

- **vastai** — library blocks port 22 (SSH), so it needs a VPN to reach Jupyter. Fresh pod each time → reinstall everything.
- **RunPod** — HTTPS Jupyter works from the library, but pods stall / images take forever to pull.
- **Colab** — usually best, but drops sessions / stalls.

Switching providers a fourth time won't fix it — the failure mode they share is **cold-start variance** on ephemeral GPU.

## The key insight

The `1.3.1_Linear_Probes` notebook needs a GPU **only to extract activations** from large models:

- `Llama-3.1-8B-Instruct` (~16GB bf16)
- `Llama-2-13b-hf` (~26GB)
- `Llama-3.3-70B-Instruct` (~70GB 8-bit) — optional/advanced

But the **probe training and analysis run on CPU** — the notebook itself uses
`LRProbe.from_data(..., device="cpu")` and stores activations as `float32 on CPU`.

So the work splits cleanly:

| Step | Where | How often | Cost |
|------|-------|-----------|------|
| 1. Run model → cache hidden states to disk | GPU pod | Once per model | Minutes of GPU time (~cents) |
| 2. Download cached activations (MBs, not GBs) | — | Once per model | Trivial |
| 3. Train probes, sweep layers, analyse, plot | **Local Mac, CPU** | **Daily** | Free, instant |

The GPU stops being a daily dependency. You touch a pod **twice a chapter**, not every morning.

## Workflow

### One-time per model (GPU pod)

1. Spin up a persistent pod — **A100-40GB or A6000-48GB** (~$1–1.5/hr) handles 8B and 13B comfortably.
   - **Stop, don't destroy** between uses → env + disk persist → fast restart, no re-pull.
   - Reach it via **HTTPS Jupyter** (works from the library, no SSH, no VPN).
2. Run an **extraction script** (see below) that:
   - loads the model with `output_hidden_states=True`,
   - runs the probe datasets (cities, neg_cities, sp_en_trans, etc.) through it,
   - saves per-layer activations `[n_statements, d_model]` to `.pt`/`.npy` files.
3. Download the `.pt` files (megabytes).
4. **Stop the pod.**

### Daily (local, golden slot)

- Load the cached activations from disk.
- Do all of 1.3: train `LRProbe` / `MMProbe`, layer sweeps, direction analysis, honesty/deception
  probing, plots. All CPU, all local, instant start.

### 70B section

- Optional. One 8-bit run on an 80GB card (A100-80GB / H100), or skip for now — it won't gate
  your understanding of probing.

## To build

- [ ] `extract_activations.py` — small script: model name + dataset → cached activation tensors on disk.
      Parameterise by model and layer set. Save to `./activations/<model>/<dataset>.pt`.
- [ ] Refactor the notebook so the analysis cells load from `./activations/...` instead of holding a
      live model in memory.
- [ ] Persistent RunPod template (A100-40GB / A6000) + network volume, saved so restart is ~2 min.
- [ ] `.gitignore` the `activations/` dir (or keep small ones; large ones stay out of git).

## Why this is the right fix

It's the "fix the structure, not the willpower" move that unlocked the 4am slot in April — but
applied to compute. Instead of paying the cold-start tax every morning, you pay it once per model
and bank the rest of the chapter as friction-free local work.
