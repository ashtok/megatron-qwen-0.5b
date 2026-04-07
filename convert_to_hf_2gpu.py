import os
import json
import glob
import argparse
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

# ── Config ────────────────────────────────────────────────────────────────────
HF_MODEL_ID  = "Qwen/Qwen2.5-0.5B"
CKPT_BASE    = "/data/42-julia-hpc-rz-wuenlp/s472389/caidas/megatron_gpt2/checkpoints/qwen25-2gpu-bs256-seq1024-lr6e4"
HF_OUT_BASE  = "/data/42-julia-hpc-rz-wuenlp/s472389/caidas/megatron_gpt2/hf_models/qwen25_2gpu_bs256"

NUM_LAYERS   = 24
NUM_HEADS    = 14
NUM_KV_HEADS = 2
HIDDEN       = 896
HEAD_DIM     = HIDDEN // NUM_HEADS  # 64
FFN_HIDDEN   = 4864
VOCAB_SIZE   = 151936
# ─────────────────────────────────────────────────────────────────────────────


def find_base_model():
    candidates = glob.glob(
        "/data/42-julia-hpc-rz-wuenlp/s472389/.cache/huggingface/hub/"
        "models--Qwen--Qwen2.5-0.5B/snapshots/*/config.json"
    )
    if not candidates:
        candidates = glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/config.json"
            )
        )
    if candidates:
        return os.path.dirname(candidates[0])
    print("  WARNING: local HF cache not found, falling back to hub (needs internet)")
    return HF_MODEL_ID


def load_checkpoint(iter_dir):
    print(f"  Reading from {iter_dir}")
    reader   = FileSystemReader(iter_dir)
    metadata = reader.read_metadata()
    state = {}
    for key, md in metadata.state_dict_metadata.items():
        if key.startswith("optimizer"):
            continue
        if hasattr(md, "size"):
            state[key] = torch.empty(md.size, dtype=md.properties.dtype)
    dcp.load(state, storage_reader=reader)
    print(f"  Loaded {len(state)} tensors")
    return state


def pad_vocab(t, target):
    if t.shape[0] == target:
        return t
    pad = torch.zeros(target - t.shape[0], t.shape[1], dtype=t.dtype)
    print(f"  Padding vocab {t.shape[0]} → {target}")
    return torch.cat([t, pad], dim=0)


def split_qkv_gqa(qkv, num_heads, num_kv_heads, head_dim):
    """
    Megatron GQA interleaved layout:
    For each KV group g: [Q_heads * head_dim, K * head_dim, V * head_dim]
    """
    n     = num_heads // num_kv_heads  # Q heads per KV group
    chunk = (n + 2) * head_dim
    q_parts, k_parts, v_parts = [], [], []
    for g in range(num_kv_heads):
        block = qkv[g * chunk : (g + 1) * chunk]
        q_parts.append(block[:n * head_dim])
        k_parts.append(block[n * head_dim : (n + 1) * head_dim])
        v_parts.append(block[(n + 1) * head_dim :])
    return torch.cat(q_parts), torch.cat(k_parts), torch.cat(v_parts)


def remap(state):
    hf = {}

    # Print all keys for debugging
    print("  Checkpoint keys:")
    for k in sorted(state.keys()):
        print(f"    {k}  {list(state[k].shape)}")

    hf["model.embed_tokens.weight"] = pad_vocab(
        state["embedding.word_embeddings.weight"], VOCAB_SIZE)
    hf["lm_head.weight"] = pad_vocab(
        state["output_layer.weight"], VOCAB_SIZE)
    hf["model.norm.weight"] = state["decoder.final_layernorm.weight"]

    ln_in  = state["decoder.layers.input_layernorm.weight"]
    ln_pre = state["decoder.layers.pre_mlp_layernorm.weight"]
    qkv_w  = state["decoder.layers.self_attention.linear_qkv.weight"]
    o_w    = state["decoder.layers.self_attention.linear_proj.weight"]
    fc1_w  = state["decoder.layers.mlp.linear_fc1.weight"]
    fc2_w  = state["decoder.layers.mlp.linear_fc2.weight"]

    q_size  = NUM_HEADS    * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM

    for i in range(NUM_LAYERS):
        hf[f"model.layers.{i}.input_layernorm.weight"]          = ln_in[i]
        hf[f"model.layers.{i}.post_attention_layernorm.weight"] = ln_pre[i]
        hf[f"model.layers.{i}.self_attn.o_proj.weight"]         = o_w[i]
        hf[f"model.layers.{i}.mlp.down_proj.weight"]            = fc2_w[i]

        q, k, v = split_qkv_gqa(qkv_w[i], NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)
        hf[f"model.layers.{i}.self_attn.q_proj.weight"] = q
        hf[f"model.layers.{i}.self_attn.k_proj.weight"] = k
        hf[f"model.layers.{i}.self_attn.v_proj.weight"] = v

        # Zero biases — trained with --disable-bias-linear
        hf[f"model.layers.{i}.self_attn.q_proj.bias"] = torch.zeros(q_size)
        hf[f"model.layers.{i}.self_attn.k_proj.bias"] = torch.zeros(kv_size)
        hf[f"model.layers.{i}.self_attn.v_proj.bias"] = torch.zeros(kv_size)

        gate, up = fc1_w[i].chunk(2, dim=0)
        hf[f"model.layers.{i}.mlp.gate_proj.weight"] = gate
        hf[f"model.layers.{i}.mlp.up_proj.weight"]   = up

    return hf


def convert_single(iteration, ckpt_base, out_base, base_model):
    iter_dir = os.path.join(ckpt_base, f"iter_{iteration:07d}")
    out_dir  = os.path.join(out_base, f"iter{iteration:07d}")

    if not os.path.isdir(iter_dir):
        print(f"  ERROR: checkpoint not found at {iter_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  Iteration  : {iteration}")
    print(f"  Checkpoint : {iter_dir}")
    print(f"  Output     : {out_dir}")
    print(f"{'='*60}")

    print("\nStep 1 — Loading checkpoint …")
    state = load_checkpoint(iter_dir)

    print("\nStep 2 — Remapping weights …")
    hf_state = remap(state)
    print(f"  Mapped {len(hf_state)} HF tensors ✓")

    print("\nStep 3 — Building HuggingFace model …")
    config = Qwen2Config.from_pretrained(base_model)
    config.intermediate_size       = FFN_HIDDEN
    config.tie_word_embeddings     = False
    config.num_attention_heads     = NUM_HEADS
    config.num_key_value_heads     = NUM_KV_HEADS
    config.hidden_size             = HIDDEN
    config.num_hidden_layers       = NUM_LAYERS
    config.vocab_size              = VOCAB_SIZE

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model     = AutoModelForCausalLM.from_config(config)

    result = model.load_state_dict(hf_state, strict=False)
    if result.missing_keys:
        print(f"  Missing   : {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Unexpected: {result.unexpected_keys}")
    if not result.missing_keys and not result.unexpected_keys:
        print("  All weights loaded ✓")

    model = model.to(torch.bfloat16)

    print(f"\nStep 4 — Saving to {out_dir} …")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "conversion_meta.json"), "w") as f:
        json.dump({
            "source": iter_dir,
            "iteration": iteration,
            "base_model": HF_MODEL_ID,
            "ffn_hidden_size": FFN_HIDDEN,
            "tie_word_embeddings": False,
            "note": "QKV biases zeroed; untied embeddings; GQA split"
        }, f, indent=2)

    print(f"\nDone ✓  →  {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Megatron qwen25-2gpu checkpoint to HF format")
    parser.add_argument(
        "--iters", type=int, nargs="+",
        default=[30000],
        help="Iteration(s) to convert, e.g. --iters 30000 or --iters 10000 20000 30000"
    )
    parser.add_argument("--all", action="store_true", help="Convert all available iterations")
    parser.add_argument("--ckpt-base", default=CKPT_BASE)
    parser.add_argument("--out-base",  default=HF_OUT_BASE)
    args = parser.parse_args()

    base_model = find_base_model()
    print(f"Base model : {base_model}")

    if args.all:
        dirs = sorted(glob.glob(os.path.join(args.ckpt_base, "iter_*")))
        iterations = [int(os.path.basename(d).replace("iter_", "")) for d in dirs]
        print(f"Found {len(iterations)} checkpoints")
    else:
        iterations = args.iters

    for it in iterations:
        convert_single(it, args.ckpt_base, args.out_base, base_model)


if __name__ == "__main__":
    main()
