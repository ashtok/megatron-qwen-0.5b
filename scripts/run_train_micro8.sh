#!/bin/bash
#SBATCH --job-name=qwen25_meg_micro8
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_micro8_%j.out
#SBATCH --error=logs/train_micro8_%j.err

set -euo pipefail

cd /data/42-julia-hpc-rz-wuenlp/s472389/caidas/megatron_gpt2

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apptainer exec --nv ../nemo_sandbox \
  torchrun --standalone --nproc-per-node=2 \
  scripts/train_qwen25.py \
    --transformer-impl local \
    --no-persist-layer-norm \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 896 \
    --num-attention-heads 14 \
    --group-query-attention \
    --num-query-groups 2 \
    --ffn-hidden-size 4864 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 8 \
    --global-batch-size 256 \
    --train-iters 30000 \
    --lr 6e-4 \
    --min-lr 6e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 500 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --bf16 \
    --disable-bias-linear \
    --swiglu \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --vocab-size 151936 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen2.5-0.5B \
    --data-path data/tokenized/train1_deu_text_document \
    --split 98,1,1 \
    --save checkpoints/qwen25_megatron_micro8 \
    --load checkpoints/qwen25_megatron_micro8 \
    --save-interval 500 \
    --eval-interval 99999 \
    --eval-iters 1 \
    --log-interval 100 \
    --no-check-for-nan-in-loss-and-grad \
    --rerun-mode disabled \
    --wandb-project qwen25_pretrain_german \
    --wandb-entity ashtok897-university-of-wuerzburg \
    --wandb-exp-name megatron-2gpu-micro8 \
    --wandb-save-dir /data/42-julia-hpc-rz-wuenlp/s472389/caidas/megatron_gpt2/wandb_storage
