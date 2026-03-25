#!/bin/bash
set -euo pipefail

MEGATRON_LM=/opt/megatron-lm
DATA_SRC=/data/42-julia-hpc-rz-wuenlp/s472389/modalities_test/data/merged

echo "=== Source data check ==="
ls -lh "$DATA_SRC/train1_deu.jsonl"

echo "=== Sample JSONL ==="
head -3 "$DATA_SRC/train1_deu.jsonl"

mkdir -p data/tokenized

echo "Tokenizing train1_deu (90% train / 10% valid split handled at training time)..."
python3 $MEGATRON_LM/tools/preprocess_data.py \
  --input "$DATA_SRC/train1_deu.jsonl" \
  --output-prefix data/tokenized/train1_deu \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model Qwen/Qwen2.5-0.5B \
  --workers 4 \
  --append-eod

echo "✓ SUCCESS!"
ls -lh data/tokenized/
