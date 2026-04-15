# WIP
# --------------------------
# WIP
# --------------------------
# WIP
# --------------------------
# WIP
# --------------------------
# WIP

# Megatron-LM Qwen2.5-0.5B German Pretraining

Complete pipeline guide for pretraining Qwen2.5-0.5B on German data using Megatron-LM on the julia2 HPC cluster. Covers container setup, data preparation, checkpoint conversion, training, and export — including fixes for all known issues.

**Environment:** NVIDIA L40S GPUs · julia2 cluster (jn101/jn012) · NGC PyTorch 24.02+ · Megatron-LM Core 0.8.0+

---

## Architecture Overview

```
Tensor Parallel (TP=1):
  [Embedding] → 24 × [Attention (GQA 14/2) + SwiGLU (4864)] → [Output]

Data Parallel (DP=2):
  Replicated across both GPUs

Vocab:   151936 → padded to 151680 (divisible by 128)
Batch:   micro=8 × grad_accum=16 × 2 GPUs = global 256
```

---

## Prerequisites

| Item | Value |
|------|-------|
| SLURM account | `s472389` |
| Dataset | `data/tokenized_train1deu_text_document` (~10B tokens) |
| Tokenizer files | `vocab.json` + `merges.txt` (Qwen2.5-0.5B) |
| Base directory | `/data/42-julia-hpc-rz-wuenlp/s472389/caidas/megatron_gpt2` |

---

## Step 1 — SLURM Job Script

Save as `megatron_train.slurm` and submit with `sbatch megatron_train.slurm`.

```bash
#!/bin/bash
#SBATCH --job-name=qwen25_fast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2                  # 2× L40S
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

module purge && module load cuda/12.1.1 python/3.11.8

cd /data/42-julia-hpc-rz-wuenlp/s472389/caidas/megatron_gpt2

torchrun --nproc_per_node=2 --node_rank=0 \
  train_qwen25.py \
  --num-layers 24 \
  --hidden-size 896 \
  --ffn-hidden-size 4864 \            # Important: NOT 3328
  --num-attention-heads 14 \
  --num-query-groups 2 \              # GQA
  --seq-length 1024 \
  --vocab-file vocab.json \
  --merge-file merges.txt \
  --vocab-size 151936 \               # Padded to 151680 at runtime
  --make-vocab-size-divisible-by 128 \
  --tokenizer-type GPT2BPETokenizer \
  --data-path data/tokenized_train1deu_text_document \
  --data-cache-path data/cache \
  --split 98,1,1 \
  --micro-batch-size 8 \
  --global-batch-size 256 \
  --train-iters 30000 \
  --save checkpoints/qwen25_fast_%j \
  --save-interval 500 \
  --load checkpoints/qwen25_fast_%j \
  --bf16 \
  --lr 0.0006 \
  --lr-decay-style cosine \
  --lr-warmup-iters 500 \             # Overrides any checkpoint value
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --log-interval 100 \
  --tensorboard-dir tensorboard \
  --distributed-backend nccl
```

---

## Step 2 — Container Setup

```bash
# Create a writable build directory (avoids read-only filesystem errors)
mkdir -p megatron_kernels/build
export MEGATRON_FUSED_KERNELS_BUILD_DIR=$PWD/megatron_kernels/build

# Pull container and install dependencies
podman pull nvcr.io/nvidia/pytorch:24.02-py3
pip install onelogger wandb tiktoken==0.6.0
pip install nvidia-modelopt --no-deps    # Fixes peft version conflict

# Verify tokenizer
python -c "
from megatron.training.tokenizer.tokenizer import build_tokenizer
t = build_tokenizer({'type':'GPT2BPETokenizer','vocab_file':'vocab.json','merge_file':'merges.txt'})
print('Vocab size:', t.vocab_size)  # Expected: 151936
"
```

---

## Step 3 — Data Preparation

Tokenization only needs to be run once. Megatron will automatically build and cache index files on first use.

```bash
# Tokenize raw text
python scripts/tokenize_dataset.py \
  --input raw_german_text \
  --output data/tokenized_train1deu_text_document \
  --tokenizer Qwen/Qwen2.5-0.5B

# Index files are cached at --data-cache-path data/cache
```

---

## Step 4 — Checkpoint Conversion (Modalities/HF → Megatron)

Use this if starting from a pretrained Modalities or HuggingFace checkpoint rather than training from scratch.

Save as `conversion.sh`:

```bash
#!/bin/bash

# Stage 1: Convert Modalities checkpoint to HuggingFace format
python convert_to_hf.py \
  --input checkpoints/modalities_exp2.bin \
  --output checkpoints/qwen25_bridge \
  --model Qwen/Qwen2.5-0.5B \
  --tp-size 1 \
  --pp-size 1

# Stage 2: Reshape to exact Megatron dimensions
python convert_to_hf_2gpu.py \
  --load checkpoints/qwen25_bridge \
  --save checkpoints/qwen25_megatron_ready \
  --ffn-hidden 4864 \         # Must match --ffn-hidden-size in SLURM script
  --untie-embeddings           # Required for correct weight handling
```

Then use `--load checkpoints/qwen25_megatron_ready` in your SLURM script.

---

## Step 5 — Monitoring

### Job status

```bash
squeue -u s472389
scontrol show job 2470843
```

### Live log tailing

```bash
tail -f logs/qwen25_fast_2470843.out | grep 'iteration\|loss\|saving'
```

### TensorBoard

```bash
# On the cluster
tensorboard --logdir tensorboard --port 6006

# Tunnel from your local machine
ssh -L 6006:localhost:6006 julia2
```

### ClusterCockpit (Würzburg HPC)

Open [clustercockpit.julia.uni-wuerzburg.de](https://clustercockpit.julia.uni-wuerzburg.de) and look up job 2470843.

Healthy indicators: `acc_util > 80%`, `cpu_user ≈ 100%`

### Target metrics

```
iteration 27100 | lm loss 2.208   (target: < 2.16 at 30k)
elapsed ms:     8293              (stable range: 8–10s per iteration)
validation PPL: 8.73
```

---

## Step 6 — Export and Evaluation

```bash
# Run evaluation on the final checkpoint
torchrun --nproc_per_node=2 train_qwen25.py \
  --load checkpoints/qwen25_fast_*/iter_030000 \
  --test-iters 256 \
  --no-train

# Export to HuggingFace format
python convert_to_hf.py \
  --input checkpoints/qwen25_fast_*/iter_030000 \
  --output hf_models/qwen25_german
```

---

## Troubleshooting

| Error | Log symptom | Fix |
|-------|-------------|-----|
| Unknown argument | `unrecognized arguments --log-loss-scale` | Remove the `--log-loss-scale` flag |
| Kernel build fails | `build directory failed` | Set `MEGATRON_FUSED_KERNELS_BUILD_DIR=$PWD/build` |
| Vocab size mismatch | `151665 → 151680` | Use `--vocab-size 151936 --make-vocab-size-divisible-by 128` |
| Wrong FFN dimension | Shape `[3328, 896]` in logs | Set `--ffn-hidden-size 4864` |
| LR warmup too long | `128000 iters` loaded from checkpoint | Add `--lr-warmup-iters 500` to override |
| peft version warning | `peft 0.13.2 vs 0.17.0` | Run `pip install nvidia-modelopt --no-deps` |
| Slow data init | `176s` startup delay | Add `--data-cache-path data/cache` |

---

## Expected Runtime

Full 30,000-iteration run completes in approximately **48 hours** on 2× L40S GPUs, reaching a validation loss below 2.16.
