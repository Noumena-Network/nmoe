# nmoe.data - Data Pipeline

Complete data preprocessing, model inference, and HYDRA quality filtering for nmoe.

## Model Setup - GPT-OSS (20B / 120B)

### Download Checkpoints

```bash
# Activate environment
source /workspace/nmoe/.venv/bin/activate

# Install dependencies
uv pip install huggingface-hub xxhash safetensors

# Download gpt-oss-20b (39GB) - for HYDRA grading & K2 rephrasing
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='openai/gpt-oss-20b',
    local_dir='/data/checkpoints/gpt-oss-20b',
    local_dir_use_symlinks=False
)
"

# Download gpt-oss-120b (240GB) - for HYDRA oracle labeling
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='openai/gpt-oss-120b',
    local_dir='/data/checkpoints/gpt-oss-120b',
    local_dir_use_symlinks=False
)
"
```

### Set PYTHONPATH

The inference code requires both `nmoe` and `triton_kernels` on the Python path:

```bash
export PYTHONPATH=/workspace/nmoe:/workspace/nmoe/triton/python/triton_kernels
```

**Why?** `triton_kernels` is a custom package inside the triton submodule containing MoE kernels (matmul_ogs, topk, swiglu, mxfp quantization) required for inference.

### Test Model Loading & Generation

```python
import torch
import tiktoken
from nmoe.data.model import Transformer, BatchedGenerator

# Test gpt-oss-20b (20B params, 32 experts, 24 layers)
device = torch.device('cuda')
model_20b = Transformer.from_checkpoint('/data/checkpoints/gpt-oss-20b', device=device)
print(f"Loaded: {model_20b.config.num_hidden_layers} layers, {model_20b.config.num_experts} experts")

# Test generation
enc = tiktoken.get_encoding('o200k_base')
gen = BatchedGenerator('/data/checkpoints/gpt-oss-20b', max_seq_len=512, max_batch=1)

prompt = 'The future of artificial intelligence is'
tokens = enc.encode(prompt)
seq_id = gen.add(tokens, max_tokens=50)

generated = []
while not gen.idle:
    result = gen.step()
    if not result: break
    logits, seq_ids = result
    next_token = logits[0].argmax().item()
    generated.append(next_token)
    gen.update(seq_ids[0], next_token, finished=len(generated)>=50)

print(enc.decode(tokens + generated))
```

### Model Architecture

- **gpt-oss-20b**: 24 layers, 2880 hidden, 64 heads (8 KV), 32 experts (4 active), 128 sliding window
- **gpt-oss-120b**: 36 layers, larger hidden, 128 experts (4 active)
- **Quantization**: MXFP4 for MLP expert weights, BF16 for attention
- **Features**: Streaming attention with sinks, RoPE with YARN scaling, SwiGLU with limiting

### Checkpoint Format

OpenAI's gpt-oss checkpoints use HuggingFace format:
- Config: `config.json` (architecture, rope_scaling, quantization_config)
- Weights: `model-*.safetensors` (sharded safetensors files)
- Naming: `model.layers.{i}.{module}.{param}` (e.g., `model.layers.0.self_attn.q_proj.weight`)

The checkpoint loader in `model.py` automatically handles:
- Config field mapping (HuggingFace → ModelConfig)
- Weight name translation (model.layers.* → blocks.*)
- MXFP4 decompression (blocks + scales → BF16)
- QKV concatenation (separate q/k/v projections → combined qkv)

---

## Data Preprocessing Pipeline

Complete data preprocessing and loading pipeline for nmoe training.

## Quick Start

### 1. Smoke Test (100M tokens, ~2 min)

```bash
python -m nmoe.data.cli prep-mixture \
  --config configs/moonlet.toml \
  --flow dev \
  --stage pretrain \
  --max-tokens 100M \
  --workers 8
```

### 2. Research Run (1B tokens, ~15 min)

```bash
python -m nmoe.data.cli prep-mixture \
  --config configs/moonlet.toml \
  --flow research \
  --stage pretrain \
  --max-tokens 1B \
  --workers 8
```

**Token targets by flow:**

| Flow | Tokens | Use Case |
|------|--------|----------|
| `dev` | 100M | Smoke test (~7 min training) |
| `research` | 1B | Architecture research (~1.2 hr) |
| `ablation` | 6B | A vs B comparison (~5-7 hr) |

**Notes:**
- `--max-tokens` stops after the specified total (e.g., `100M`, `1B`, `1.5T`)
- Streams directly from HuggingFace (no local download required)
- HuggingFace caches parquet files, so re-runs don't re-download
- Tokenizes with `o200k_harmony` (vocab 201k, EOS 199999)
- Do NOT use `--parallel` for large datasets (buffers everything in memory)

### Preprocess a Single HuggingFace Dataset

```bash
python -m nmoe.data.cli prep \
  --source huggingface \
  --dataset HuggingFaceFW/fineweb-edu \
  --split train \
  --output /data/fineweb_edu/train \
  --name fineweb_edu \
  --num-shards 64 \
  --parallel
```

### Verify Dataset Integrity

```bash
python -m nmoe.data.cli verify /data/fineweb_edu/train/manifest.json
python -m nmoe.data.cli info /data/fineweb_edu/train/manifest.json
```

## Architecture

```
                     MIXTURE TOML (single source of truth)
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   HuggingFace          HuggingFace          HuggingFace
   Dataset A            Dataset B            Dataset C
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                    prep-mixture CLI
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   /data/source_a/      /data/source_b/      /data/source_c/
   ├── train/           ├── train/           ├── train/
   │   └── *.npy        │   └── *.npy        │   └── *.npy
   └── valid/           └── valid/           └── valid/
       └── *.npy            └── *.npy            └── *.npy
                              │
                              ▼
                    DeterministicLoader
                    (SWRR mixing, exact resume)
```

## Mixture TOML with HuggingFace Sources

Define data sources directly in the mixture TOML:

```toml
# configs/mixtures/my_mixture.toml

[mixtures.pretrain_mix]
total_tokens_b = 100.0
sample_temperature = 1.0

[[mixtures.pretrain_mix.sources]]
id = "fineweb"
tokens_b = 80.0
percent = 80.0
hf_dataset = "HuggingFaceFW/fineweb-edu"
hf_subset = "sample-10BT"           # optional
hf_split = "train"                   # default: "train"
hf_valid_split = "train[:1%]"        # optional, for validation
text_field = "text"                  # default: "text"

[[mixtures.pretrain_mix.sources]]
id = "code"
tokens_b = 20.0
percent = 20.0
hf_dataset = "bigcode/the-stack-v2"
hf_subset = "python"
hf_split = "train"
```

## Data Layout Convention

```
/data/
├── {source_id}/
│   ├── train/
│   │   ├── manifest.json
│   │   └── shard_NNNN/
│   │       ├── {source_id}-v1-shard-NNNNNN.npy
│   │       └── {source_id}-v1-shard-NNNNNN.idx
│   └── valid/
│       ├── manifest.json
│       └── shard_NNNN/
│           └── ...
```

Path resolution: `{data_root}/{source_id}/{split}/**/*.npy`

## Pipeline Stages

```
HuggingFace Dataset (streaming)
    ↓
[Normalize] → NFC normalization, strip whitespace, clean control chars
    ↓
[Tokenize] → tiktoken with parallel workers (o200k_harmony)
    ↓
[Shard] → deterministic doc→shard assignment via MD5 hash
    ↓
[Pack] → concatenate tokens with EOS, write .npy + .idx
    ↓
[Manifest] → inventory with SHA256 checksums
```

## File Formats

### Shard Files (.npy)

- NumPy array of uint32 token IDs
- Documents concatenated with EOS token (199999 for o200k_harmony)
- Typical size: ~2GB per shard (500M tokens)

### Index Files (.idx)

Document boundary indices for random access within shards.

```
Header (32 bytes):
  magic:      b"NMOEIDX\x00"     (8 bytes)
  version:    uint64             (8 bytes) = 1
  num_docs:   uint64             (8 bytes)
  reserved:   uint64             (8 bytes) = 0

Body:
  doc_boundaries: uint64[num_docs * 2]  # (start_idx, end_idx) pairs
```

### Manifest (.manifest.json)

```json
{
  "dataset": "fineweb_edu",
  "version": "v1",
  "tokenizer": "o200k_harmony",
  "vocab_size": 201088,
  "eos_token_id": 199999,
  "dtype": "uint32",
  "total_tokens": 1000000000,
  "total_documents": 5000000,
  "num_shards": 64,
  "shards": [...]
}
```

**HYDRA Quality Grading Runbook**
- Purpose: Distill a 120B oracle’s multi‑dimensional grades (0–4) into fast heads over a frozen 20B backbone, then grade/train at scale.
- Teacher: gpt‑oss‑120b (strict JSON via Harmony final channel only).
- Student: gpt‑oss‑20b with Probe (L18 pooled) and MTP‑Judge (depth=1, shared head, teacher‑forced during training).

**Probe + Judge Design**
- Probe (L18)
  - Inputs: pooled hidden from layer 18 of the 20B backbone (no logits path).
  - Head: single linear projection → 5 scores in [0,4] for helpfulness, correctness, coherence, complexity, density.
  - Role: fast triage. After calibration (learned per‑dim weights and τ), the aggregated score is compared against thresholds:
    - Below `τ_drop` → drop immediately.
    - Otherwise pass to Judge; in some flows, ≥ `τ_keep` may keep without Judge.
  - Training: supervised regression on oracle labels; default MSE (kept simple for speed).

- Judge (MTP‑Judge, L24)
  - Inputs: layer‑24 sequence states. Pre‑processing for stability: `nan_to_num` (map NaN/±Inf to finite), per‑token `LayerNorm`, then FP32.
  - Head: a shallow MTP‑style block:
    - prepend a learned “quality” token → project to `mid_dim` → 1 tiny Transformer encoder layer → shared linear head → 5 scores.
    - depth=1 teacher‑forcing during training: embed teacher scores at the quality token and re‑predict (second pass shares head).
  - Objective: regress oracle’s 5 scores (0–4). Loss: SmoothL1 (beta=0.5) on clamped predictions/targets (clamp only for loss).
  - Training hygiene: warmup (default 500 steps), AdamW with small weight decay, gradient‑norm clipping (1.0), AMP BF16 for the backbone feature extraction, and `no_logits=True` to skip LM‑head compute.
  - Inference: single forward pass (no teacher input) to produce 5 scores; aggregate via calibrated weights → decision {drop, band, keep}.

### HYDRA End‑to‑End (Automated via scripts/)

**TODO(oss):** The `scripts/` automation wrappers referenced below are not included in this public release yet. Use the manual commands further down in this README.

Use these one‑liners for a fully automated, resume‑safe run. Edit envs to taste; sensible defaults are baked in.

Prereqs
- Pod or local shell: set `POD` if using Kubernetes (optional), or run locally with `LOCAL=1` on scripts.
- Python env: activate your virtualenv/Conda; ensure this repo is importable (`pip install -e .` or `export PYTHONPATH=$PWD:$PYTHONPATH`).
- Data root: `DATA_ROOT` points to your prepared shard tree (e.g., `/data/your_corpus`).
- Model checkpoints: define `CKPT_120B` and `CKPT_20B` paths for the oracle and student models, respectively.

Stage 0 — Shared environment (example; customize paths)
```bash
export POD=<your_pod_name>                 # optional; omit if running locally
export DATA_ROOT=<path_to_prepared_shards> # e.g., /data/corpus_v1
export SLICES_ID=<slices_folder_name>      # e.g., slices_corpus_v1
export ORACLE_RUN_ID=<run_id>              # e.g., oracle_120b_v1_corpus_v1
export CKPT_120B=<path_to_120b_ckpt>      # e.g., /models/gpt-oss-120b/original/original
export CKPT_20B=<path_to_20b_ckpt>        # e.g., /models/gpt-oss-20b/original/original
```

The HYDRA pipeline has five stages. See the detailed inline examples below for each stage, or create wrapper scripts for your environment.

**Stage 1** — Build docid slices (idempotent)
**Stage 2** — Oracle labeling on 8× GPUs (streaming + resume)
**Stage 3** — Merge + Calibrate (fits weights and τ on val)
**Stage 4** — Train HYDRA heads over frozen 20B
**Stage 5** — Grade any corpus with HYDRA

Resume & Idempotency
- Slices skip rebuild when `docids_{train,val,test}.jsonl` already exist.
- Oracle workers append+flush per record; safe to restart any shard. `summary.json` marks completion; the launcher skips completed shards.
- Merge/Calibrate overwrites merged files and calibration JSON; safe to re‑run.
- Head training overwrites weights in `OUT_DIR` (version by changing the path if needed).
- Grading overwrites `quality_scores.jsonl` in `OUT_DIR` (version via path).

Logs & Monitoring
- Oracle logs: `/tmp/oracle_{train|val}_${ORACLE_RUN_ID}_GPU.log`
- Status: use `nvidia-smi` and `ps aux | grep nmoe` to check GPU usage and shard state
- GPUs: `kubectl exec "$POD" -- nvidia-smi` (or run `nvidia-smi` locally)

Troubleshooting (quick)
- Port busy, orphan procs: kill labeler processes and clear stray GPU procs (`nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9`).
- Missing Harmony encoding: ensure `openai_harmony` (and tokenizer deps) are installed in your Python env.
- Low parse rate on some web shards is expected; math/wiki sources typically score higher.
- Calibration with very small val sets yields near‑uniform weights—re‑run after more val shards complete.

Prerequisites
- Pod: `nmoe-debug-<id>`
- Venv: `/workspace/nmoe/.venv`
- Env: `export PYTHONPATH=/workspace/nmoe:/workspace/nmoe/triton/python/triton_kernels`
- Data root: `/data/olmo3_dev` (or `/data/olmo3_research`)
- Checkpoints:
  - 120B: `/data/models/gpt-oss-120b/original/original`
  - 20B:  `/data/models/gpt-oss-20b/original/original`

1) Build DocID Slices (train/val/test) from shards
- Run inside the pod (pure Python; no nmoe imports):
```
PY=/workspace/nmoe/.venv/bin/python
DATA_ROOT=/data/olmo3_dev
OUT=/data/labels/slices_olmo3_dev
kubectl exec -it $POD -- bash -lc "\
$PY - << 'PY'
import os, json, hashlib, struct
from pathlib import Path
import numpy as np
ROOT=Path("/data/olmo3_dev"); OUT=Path("/data/labels/slices_olmo3_dev"); OUT.mkdir(parents=True, exist_ok=True)
MAGIC=b"NMOEIDX\x00"; HDR=32
def read_idx(p):
  with open(p,'rb') as f:
    h=f.read(HDR); m,v,n,_=struct.unpack('<8sQQQ',h); 
    if m!=MAGIC or v!=1: return None
    b=f.read(); a=np.frombuffer(b,dtype=np.uint64); 
    return a.reshape(-1,2) if a.size%2==0 else None
def split(doc_iter,base):
  tr=(base/"docids_train.jsonl").open('w'); va=(base/"docids_val.jsonl").open('w'); te=(base/"docids_test.jsonl").open('w')
  nt=nv=ne=0
  for src,rel,b in doc_iter:
    for s,e in b.tolist():
      did=f"{src}//{rel}#s={s}:e={e}"; h=int(hashlib.md5(did.encode()).hexdigest()[:8],16)%100
      row=json.dumps({'doc_id':did,'source':src},ensure_ascii=False)+'\n'
      if h<80: tr.write(row); nt+=1
      elif h<90: va.write(row); nv+=1
      else: te.write(row); ne+=1
  tr.close(); va.close(); te.close(); print({'train':nt,'val':nv,'test':ne})
docs=[]
for srcdir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
  src=srcdir.name; tdir=srcdir/'train'
  if not tdir.exists(): continue
  for npy in sorted(tdir.rglob('*.npy')):
    idx=npy.with_suffix('.idx')
    if not idx.exists(): continue
    rel=str(npy.relative_to(ROOT)); b=read_idx(idx)
    if b is None: continue
    docs.append((src,rel,b))
split(docs, Path('/data/labels/slices_olmo3_dev'))
PY" 
```
- Split into 8 shards per split (still inside pod):
```
kubectl exec -it $POD -- bash -lc "\
PY=/workspace/nmoe/.venv/bin/python; $PY - << 'PY'
from pathlib import Path
base=Path('/data/labels/slices_olmo3_dev')
for split in ('train','val'):
  src=base/f'docids_{split}.jsonl'
  outs=[(base/f'{split}_shard_{i:02d}').open('w',encoding='utf-8') for i in range(8)]
  with src.open('r',encoding='utf-8') as f:
    for j,line in enumerate(f): outs[j%8].write(line)
  for o in outs: o.close()
  for i in range(8): p=base/f'{split}_shard_{i:02d}'; print(p, sum(1 for _ in p.open('r',encoding='utf-8')))
PY" 
```

2) Launch Oracle Labelers (8× GPUs)
- Final‑only Harmony JSON, right‑trim prompts, robust stop; output per shard:
```
kubectl exec -it $POD -- bash -lc "\
export PYTHONPATH=/workspace/nmoe:/workspace/nmoe/triton/python/triton_kernels;
VENV=/workspace/nmoe/.venv/bin/python; CKPT=/data/checkpoints/gpt-oss-20b;
OUTT=/data/labels/oracle_120b_v1_olmo3_dev/train; OUTV=/data/labels/oracle_120b_v1_olmo3_dev/val; 
mkdir -p $OUTT $OUTV; 
for i in $(seq 0 7); do (
  export CUDA_VISIBLE_DEVICES=$i; 
  $VENV -m nmoe.data.hydra oracle-label \
    --input-docids /data/labels/slices_olmo3_dev/train_shard_0$(printf '%d' $i) \
    --data-root /data/olmo3_dev \
    --checkpoint $CKPT \
    --out $OUTT/shard_$i \
    --max-batch 32 --max-new 2048 --max-ctx 4096 \
    > /tmp/oracle_train_olmo3_$i.log 2>&1 && \
  $VENV -m nmoe.data.hydra oracle-label \
    --input-docids /data/labels/slices_olmo3_dev/val_shard_0$(printf '%d' $i) \
    --data-root /data/olmo3_dev \
    --checkpoint $CKPT \
    --out $OUTV/shard_$i \
    --max-batch 32 --max-new 2048 --max-ctx 4096 \
    > /tmp/oracle_val_olmo3_$i.log 2>&1
) & echo launched $i; done; sleep 2; pgrep -af 'python -m nmoe.data.hydra oracle-label' || true"
```
- Monitor:
  - `tail -f /tmp/oracle_train_olmo3_*.log /tmp/oracle_val_olmo3_*.log`
  - `ls /data/labels/oracle_120b_v1_olmo3_dev/*/shard_*/*json* | xargs -r wc -l`

3) Merge Labels + Calibrate Aggregation Weights and Thresholds
```
kubectl exec -it $POD -- bash -lc "\
mkdir -p /data/labels/oracle_120b_v1_olmo3_dev/merged; \
find /data/labels/oracle_120b_v1_olmo3_dev/train -name scores.jsonl -exec cat {} + > /data/labels/oracle_120b_v1_olmo3_dev/merged/labels_train.jsonl; \
find /data/labels/oracle_120b_v1_olmo3_dev/val   -name scores.jsonl -exec cat {} + > /data/labels/oracle_120b_v1_olmo3_dev/merged/labels_val.jsonl; \
export PYTHONPATH=/workspace/nmoe:/workspace/nmoe/triton/python/triton_kernels; \
/workspace/nmoe/.venv/bin/python -m nmoe.data.hydra calibrate \
  --labels /data/labels/oracle_120b_v1_olmo3_dev/merged/labels_val.jsonl \
  --out /data/calibration/hydra_20b_v1 \
  --target-keep 0.50"
```
- Writes `/data/calibration/hydra_20b_v1/calibration_summary.json` with `weights`, `tau_drop`, `tau_keep`.

4) Train HYDRA Heads (20B backbone frozen)
```
# Probe (GPU0)
kubectl exec -it $POD -- bash -lc "\
CUDA_VISIBLE_DEVICES=0 /workspace/nmoe/.venv/bin/python -m nmoe.data.train probe \
  --labels /data/labels/oracle_120b_v1_olmo3_dev/merged/labels_train.jsonl \
  --checkpoint /data/models/gpt-oss-20b/original/original \
  --data-root /data/olmo3_dev \
  --out /checkpoints/hydra_20b_v1 \
  --batch-size 256 --epochs 1 --lr 1e-3"

# Judge (GPU1)
kubectl exec -it $POD -- bash -lc "\
CUDA_VISIBLE_DEVICES=1 /workspace/nmoe/.venv/bin/python -m nmoe.data.train judge \
  --labels /data/labels/oracle_120b_v1_olmo3_dev/merged/labels_train.jsonl \
  --checkpoint /data/models/gpt-oss-20b/original/original \
  --data-root /data/olmo3_dev \
  --out /checkpoints/hydra_20b_v1 \
  --batch-size 128 --epochs 1 --lr 5e-4"
```
- Artifacts: `/checkpoints/hydra_20b_v1/{hydra_probe.pt, hydra_judge.pt}`

5) Grade a Corpus with HYDRA (8× optional)
```
kubectl exec -it $POD -- bash -lc "\
export PYTHONPATH=/workspace/nmoe:/workspace/nmoe/triton/python/triton_kernels; \
/workspace/nmoe/.venv/bin/python -m nmoe.data.hydra grade \
  --input-docids /data/labels/slices_olmo3_dev/docids_test.jsonl \
  --data-root /data/olmo3_dev \
  --checkpoint /data/models/gpt-oss-20b/original/original \
  --heads-dir /checkpoints/hydra_20b_v1 \
  --calibration /data/calibration/hydra_20b_v1 \
  --out /data/quality/pretrain_hydra_20b_v1/test \
  --max-ctx 4096 --max-batch 64"
```
- Outputs: `quality_scores.jsonl` (+ `summary.json`) with `scores`, `aggregated`, and `decision` ∈ {drop, band, keep}.

Troubleshooting
- `ModuleNotFoundError: triton_kernels`: ensure `PYTHONPATH` includes `/workspace/nmoe/triton/python/triton_kernels`.
- `FileNotFoundError` for shard path: pass the correct `--data-root` (e.g., `/data/olmo3_dev`) matching the `doc_id` rel_path prefix.
- Slow/empty logs: check GPU assignment and `nvidia-smi`; tail `/tmp/oracle_*` logs; verify output dirs contain `scores.jsonl`.

Notes
- Scoring uses Harmony StreamableParser final‑only parsing; prompts are right‑trimmed to keep the assistant‑start suffix and use a robust stop condition; any lingering sequences are finalized via EOS.
- Calibration fits aggregation weights on validation labels and suggests `tau_drop`/`tau_keep`; adjust target keep‑rate as needed.

## CLI Reference

### `prep-mixture` - Preprocess from Mixture (Recommended)

```bash
python -m nmoe.data.cli prep-mixture \
  --config configs/moonlet.toml \    # Reads mixture/flow profiles + data_root
  --flow dev \                       # Override flow (dev|research|ablation|proxy|full_train)
  --stage pretrain \                 # Stage to process (pretrain|mid|long)
  --splits train,valid \             # Comma-separated splits
  --tokenizer o200k_harmony \        # Tokenizer (default)
  --vocab-size 201088 \              # Vocab size (default for o200k_harmony)
  --eos-token-id 199999 \            # EOS id (default for o200k_harmony)
  --num-shards 64 \                  # Shards per source
  --workers 8 \                      # Parallel workers
  --parallel \                       # Use multiprocessing
  --force \                          # Reprocess even if output exists
  --continue-on-error \              # Keep going if a source fails
  --max-tokens 100M \                # Optional early stop total
  --dry-run                          # Show what would be done
```

### `prep` - Preprocess Single Dataset

```bash
python -m nmoe.data.cli prep \
  --source huggingface \
  --dataset HuggingFaceFW/fineweb-edu \
  --split train \
  --subset sample-10BT \
  --text-field text \
  --output /data/fineweb_edu/train \
  --name fineweb_edu \
  --tokenizer o200k_harmony \
  --num-shards 64 \
  --parallel
```

### `verify` - Verify Manifest

```bash
python -m nmoe.data.cli verify /path/to/manifest.json [--checksums]
```

### `info` - Show Manifest Info

```bash
python -m nmoe.data.cli info /path/to/manifest.json
```

### `regenerate-index` - Recover Lost Index

```bash
python -m nmoe.data.cli regenerate-index /path/to/shard.npy --eos-token-id 199999
```

### `inspect` - Data Quality Inspection

Inspect preprocessed shards for quality issues (double EOS tokens, token distribution, etc.).

```bash
# Inspect with manifest (reads EOS token from manifest)
python -m nmoe.data.cli inspect --manifest /data/fineweb_edu/manifest.json

# Inspect without manifest (manual EOS specification)
python -m nmoe.data.cli inspect --data-dir /data/fineweb_edu --eos-token-id 199999

# Inspect all shards with verbose output
python -m nmoe.data.cli inspect --manifest /data/fineweb_edu/manifest.json --all --verbose

# Include token distribution statistics
python -m nmoe.data.cli inspect --data-dir /data/fineweb_edu --eos-token-id 199999 --stats

# Inspect first 3 shards only
python -m nmoe.data.cli inspect --data-dir /data/fineweb_edu --eos-token-id 199999 --num-shards 3

# Sample random tokens for manual inspection
python -m nmoe.data.cli inspect --manifest /data/fineweb_edu/manifest.json --sample 100
```

**Options:**
- `--manifest PATH` - Path to manifest.json (reads EOS token automatically)
- `--data-dir PATH` - Data directory (if no manifest)
- `--eos-token-id INT` - EOS token ID (default: 199999, required if no manifest)
- `--num-shards N` - Number of shards to inspect (default: 1)
- `--all` - Inspect all shards
- `--verbose, -v` - Show detailed per-issue output
- `--sample N` - Sample N random tokens per shard
- `--stats` - Compute token distribution statistics

**Output:**
- Token counts and EOS statistics per shard
- Double EOS detection (preprocessing bug indicator)
- Optional: token distribution, vocab coverage
- Exit code 0 if no issues, 1 if issues found

**Common Issues Detected:**
- **Double EOS tokens**: Indicates both dataprep and ShardedWriter were appending EOS (fixed in nmoe/dataprep.py)
- **Low vocab coverage**: May indicate tokenizer mismatch or corrupted data
- **Unexpected EOS counts**: Document boundary issues

## Python API

### Build Loaders for Training

```python
from nmoe.data import build_loader

# Training loader
train_loader, train_plan = build_loader(
    cfg, rank=0, world_size=8, split="train"
)

# Validation loader (use different flow_mode with lower scale)
valid_loader, valid_plan = build_loader(
    cfg, rank=0, world_size=8, split="valid"
)

# Training loop
for step in range(max_steps):
    inputs, targets = train_loader.next()
    loss = train_step(model, inputs, targets)

    if step % eval_every == 0:
        val_loss = evaluate(model, valid_loader)
```

### Checkpoint Resume

```python
# Save
state = train_loader.state_dict()
torch.save({"loader": state, ...}, "checkpoint.pt")

# Resume
ckpt = torch.load("checkpoint.pt")
train_loader.load_state_dict(ckpt["loader"])
# Continues from exact same position
```

### Low-Level Preprocessing

```python
from nmoe.data import (
    HuggingFaceSource,
    PrepConfig,
    ParallelPrepPipeline,
)

source = HuggingFaceSource(
    dataset="HuggingFaceFW/fineweb-edu",
    split="train",
    streaming=True,
)

config = PrepConfig(
    output_dir="/data/fineweb_edu/train",
    dataset_name="fineweb_edu",
    tokenizer="o200k_harmony",
    num_shards=64,
    num_workers=8,
)

pipeline = ParallelPrepPipeline(source, config)
manifest = pipeline.run()
```

### Low-Level Shard Access

```python
from nmoe.data import IndexReader
import numpy as np

# Read index
idx = IndexReader("/data/fineweb_edu/train/shard_0000/fineweb_edu-v1-shard-000000.idx")
print(f"Documents: {idx.num_docs}")

# Get document boundaries
start, end = idx.get_document(42)

# Load tokens
tokens = np.load("/data/fineweb_edu/train/shard_0000/fineweb_edu-v1-shard-000000.npy", mmap_mode="r")
doc_tokens = tokens[start:end]
```

## Batch Inference Engine

For synthetic data generation, rephrasing, and data grading pipelines.

### Throughput (single B200 GPU)

| Model | Batch Size | Throughput | Notes |
|-------|------------|------------|-------|
| gpt-oss-20b | 1 | 17 tok/s | Baseline |
| gpt-oss-20b | 64 | 220 tok/s | 13x speedup |
| gpt-oss-20b | 128 | 502 tok/s | 29x speedup |
| gpt-oss-120b | 1 | 11 tok/s | Baseline |
| gpt-oss-120b | 64 | 83 tok/s | 7x speedup |
| gpt-oss-120b | 256 | 173 tok/s | 15x speedup |

### Usage

```python
from nmoe.data import BatchedGenerator

gen = BatchedGenerator(
    checkpoint="/data/models/gpt-oss-20b/original/original",
    max_batch=128,
)

# Batch inference
prompts = [tokenizer.encode(p) for p in texts]
results = gen.generate_batch(prompts, max_tokens=256, temperature=0.0)
```

### Multi-GPU (Data Parallelism)

For maximum throughput, run one process per GPU:

```bash
# 8 GPUs, each processing 1/8 of data
CUDA_VISIBLE_DEVICES=0 python process.py --shard 0/8 &
CUDA_VISIBLE_DEVICES=1 python process.py --shard 1/8 &
...
CUDA_VISIBLE_DEVICES=7 python process.py --shard 7/8 &
wait
```

Theoretical throughput with 8x B200 GPUs:
- **20B model**: 8 × 502 = **4,016 tok/s** (~347M tokens/day)
- **120B model**: 8 × 173 = **1,384 tok/s** (~120M tokens/day)

## Module Structure

```
nmoe/data/
├── __init__.py     # Public API exports
├── cli.py          # Command-line interface (prep, prep-mixture, verify, info, inspect)
├── loader.py       # DeterministicLoader, build_loader()
├── mixture.py      # MixturePlan, HFSource, resolve_plan(), populate_paths()
├── dataset.py      # NumpyFSLDataset, Cursor
├── index.py        # IndexReader, IndexWriter
├── sources.py      # HuggingFaceSource, JSONLSource, ArrowSource
├── sinks.py        # ShardWriter, ShardedWriter, ManifestInfo
├── prep.py         # PrepPipeline, ParallelPrepPipeline
├── inspect.py      # inspect_shards() - data quality validation
├── model.py        # Transformer, BatchedGenerator (inference engine)
└── transforms.py   # normalize_text(), tokenize() [o200k_harmony default]
```

## Defaults

| Setting | Default |
|---------|---------|
| Tokenizer | `o200k_harmony` |
| Vocab size | 201,088 |
| EOS token ID | 199,999 |
| Text field | `text` |
| Train split | `train` |
| Shards per source | 64 |

## Design Principles

1. **HuggingFace-first**: Mixture TOML defines HF sources, single source of truth
2. **Convention over configuration**: `{source_id}/{split}/` path layout
3. **Streaming-first**: Never load entire datasets into memory
4. **Deterministic**: Same input → same output (MD5-based sharding)
5. **Exact resume**: Checkpoint includes global sequence position
6. **Recoverable**: Index files can be regenerated from shards
7. **Verifiable**: SHA256 checksums in manifest

## References

- [OLMo-core data pipeline](https://github.com/allenai/OLMo-core)
- [metaseq IndexedDataset](https://github.com/facebookresearch/metaseq)
- [Fewer Truncations Improve Language Modeling](https://arxiv.org/abs/2404.10830)
