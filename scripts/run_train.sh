#!/bin/bash
#SBATCH --job-name=qwen25_megatron_cmp
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

cd /data/42-julia-hpc-rz-wuenlp/s472389/caidas/megatron_gpt2

apptainer exec --nv ../nemo_sandbox \
  torchrun --standalone --nproc-per-node=1 \
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
    --micro-batch-size 4 \
    --global-batch-size 128 \
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
    --save checkpoints/qwen25_megatron_cmp \
    --load checkpoints/qwen25_megatron_cmp \
    --save-interval 1000 \
    --eval-interval 500 \
    --eval-iters 20 \
    --log-interval 100
