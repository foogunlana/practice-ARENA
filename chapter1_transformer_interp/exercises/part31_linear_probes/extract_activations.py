#!/usr/bin/env python3
"""
extract_activations.py — GPU-side half of the Ch 1.3 (Linear Probes) workflow.

Runs a large model ONCE on a GPU pod, caches per-layer last-token hidden states for
the Geometry-of-Truth datasets to disk, then you download the (small) .pt files and do
all probe training / layer sweeps / analysis locally on CPU.

See GPU_WORKFLOW_PLAN.md for the why. Conventions match the notebook's
`extract_activations` (cell 19): last non-padding token, layers indexed directly into
`outputs.hidden_states` (index 0 = embeddings, index l = output of block l-1), stored as
float32 on CPU.

Usage (on the GPU pod):

    python extract_activations.py --model meta-llama/Llama-2-13b-hf

    python extract_activations.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --datasets cities neg_cities sp_en_trans larger_than \
        --layers all \
        --out-dir ./activations

Output: ./activations/<model_slug>/<dataset>.pt  (one file per dataset)
Each file is a dict you load locally with torch.load(..., map_location="cpu").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer


# Datasets used across sections 1-3 of the notebook. neg_cities is needed for the
# paired cities+neg_cities training in the causal-intervention section.
DEFAULT_DATASETS = ["cities", "neg_cities", "sp_en_trans", "larger_than"]


def parse_layers(spec: str, num_layers: int) -> list[int]:
    """Parse --layers into a list of indices into outputs.hidden_states (0..num_layers).

    Accepts: "all", a range "a-b" (inclusive), or a comma/space list "0,7,14".
    """
    spec = spec.strip().lower()
    n_hidden = num_layers + 1  # hidden_states has num_layers + 1 entries
    if spec == "all":
        return list(range(n_hidden))
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    parts = [p for p in spec.replace(",", " ").split() if p]
    return [int(p) for p in parts]


@t.no_grad()
def extract_activations(
    statements: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: list[int],
    batch_size: int = 25,
) -> dict[int, t.Tensor]:
    """Extract last-token hidden states for `layers`, returning {layer: [n_statements, d_model]} on CPU float32.

    Mirrors the notebook's convention: the activation is taken at the last *non-padding*
    token (attention_mask.sum - 1, valid because padding_side="right"), and `layers`
    index directly into `outputs.hidden_states`.
    """
    per_layer: dict[int, list[t.Tensor]] = {l: [] for l in layers}

    for b in range(0, len(statements), batch_size):
        batch = statements[b : b + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=False, return_tensors="pt"
        ).to(model.device)

        outputs = model(input_ids=inputs["input_ids"], output_hidden_states=True)

        inds = inputs["attention_mask"].sum(-1) - 1  # [batch] last real token index
        rows = t.arange(inds.shape[0], device=model.device)

        for l in layers:
            # hidden_states[l]: [batch, seq, d_model] -> [batch, d_model] at last real token
            sel = outputs.hidden_states[l][rows, inds]
            per_layer[l].append(sel.cpu().float())

    return {l: t.cat(chunks) for l, chunks in per_layer.items()}


def load_dataset_csv(datasets_dir: Path, name: str) -> tuple[list[str], t.Tensor]:
    """Load a Geometry-of-Truth CSV: returns (statements, labels[float32])."""
    df = pd.read_csv(datasets_dir / f"{name}.csv")
    assert "statement" in df.columns and "label" in df.columns, (
        f"{name}.csv must have 'statement' and 'label' columns; got {list(df.columns)}"
    )
    statements = df["statement"].astype(str).tolist()
    labels = t.tensor(df["label"].values, dtype=t.float32)
    return statements, labels


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id, e.g. meta-llama/Llama-2-13b-hf")
    p.add_argument(
        "--datasets-dir",
        default=None,
        help="Dir with <name>.csv files. Default: ./geometry-of-truth/datasets relative to this script's exercises dir.",
    )
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Dataset names (no .csv). Use 'all' to take every CSV in the dir.")
    p.add_argument("--layers", default="all", help="'all', a range 'a-b' (inclusive), or a list '0,7,14'. Indexes into outputs.hidden_states.")
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--out-dir", default="./activations")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--load-in-8bit", action="store_true", help="8-bit quantization (needs bitsandbytes) — for 70B on an 80GB card.")
    p.add_argument("--max-statements", type=int, default=None, help="Debug: cap statements per dataset.")
    args = p.parse_args()

    # Resolve datasets dir (default mirrors the notebook's GOT_DATASETS).
    if args.datasets_dir:
        datasets_dir = Path(args.datasets_dir)
    else:
        exercises_dir = Path(__file__).resolve().parents[1]
        datasets_dir = exercises_dir / "geometry-of-truth" / "datasets"
    assert datasets_dir.exists(), f"Datasets dir not found: {datasets_dir} (clone geometry-of-truth there, or pass --datasets-dir)"

    # Expand 'all'.
    if args.datasets == ["all"]:
        dataset_names = sorted(p.stem for p in datasets_dir.glob("*.csv"))
    else:
        dataset_names = args.datasets
    assert dataset_names, "No datasets selected."

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("⚠️  No CUDA detected — this is the GPU-side script and will be very slow on CPU.")

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required: last real token = attention_mask.sum - 1

    model_kwargs: dict = {"device_map": "auto"}
    if args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["dtype"] = getattr(t, args.dtype)

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    num_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    layers = parse_layers(args.layers, num_layers)
    bad = [l for l in layers if not (0 <= l <= num_layers)]
    assert not bad, f"Layers {bad} out of range 0..{num_layers}"
    print(f"Model: {num_layers} blocks, d_model={d_model}. Extracting hidden_states layers: {layers}")

    model_slug = args.model.split("/")[-1]
    out_dir = Path(args.out_dir) / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": args.model,
        "num_layers": num_layers,
        "d_model": d_model,
        "layers": layers,
        "layer_index_convention": "index into outputs.hidden_states; 0=embeddings, l=output of block l-1 (matches notebook cell 19)",
        "datasets": {},
    }

    for name in dataset_names:
        statements, labels = load_dataset_csv(datasets_dir, name)
        if args.max_statements:
            statements, labels = statements[: args.max_statements], labels[: args.max_statements]
        print(f"  [{name}] {len(statements)} statements ...", flush=True)

        acts = extract_activations(statements, model, tokenizer, layers, batch_size=args.batch_size)

        payload = {
            "activations": acts,            # {layer: tensor[n_statements, d_model]} float32 cpu
            "labels": labels,               # tensor[n_statements] float32 (1=true, 0=false)
            "statements": statements,       # list[str]
            "model": args.model,
            "layers": layers,
            "d_model": d_model,
            "num_layers": num_layers,
            "dataset": name,
        }
        out_path = out_dir / f"{name}.pt"
        t.save(payload, out_path)
        size = out_path.stat().st_size
        manifest["datasets"][name] = {"n_statements": len(statements), "file": out_path.name, "bytes": size}
        print(f"    saved {out_path}  ({human_bytes(size)})")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    total = sum(d["bytes"] for d in manifest["datasets"].values())
    print(f"\nDone. {len(dataset_names)} datasets -> {out_dir}  (total {human_bytes(total)})")
    print(f"Manifest: {manifest_path}")
    print("Download the whole folder, then load locally with torch.load(path, map_location='cpu').")


if __name__ == "__main__":
    main()
