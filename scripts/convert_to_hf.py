import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

# ── Config (matches run_train_micro8.sh exactly) ──────────────────────────────
HF_BASE_MODEL = "Qwen/Qwen2.5-0.5B"
CKPT_DIR      = "checkpoints/qwen25_megatron_micro8"
HF_OUT_DIR    = "hf_models/qwen25_megatron_micro8"
TARGET_ITER   = 30000   # set to None to auto-pick latest

NUM_LAYERS   = 24
NUM_HEADS    = 14
NUM_KV_HEADS = 2
HIDDEN       = 896
FFN          = 4864
HEAD_DIM     = HIDDEN // NUM_HEADS   # 64
# ─────────────────────────────────────────────────────────────────────────────


def get_latest_iter(ckpt_dir):
    iters = [
        int(n[5:]) for n in os.listdir(ckpt_dir)
        if n.startswith("iter_") and n[5:].isdigit()
    ]
    return max(iters)


def load_megatron_state(iter_dir):
    """
    Megatron torch_dist format stores shards under:
      iter_dir/model/        (common layout)
    or flat .pt files directly in iter_dir.
    Walk everything and merge.
    """
    state = {}
    loaded_files = 0

    search_root = os.path.join(iter_dir, "model")
    if not os.path.isdir(search_root):
        search_root = iter_dir

    for root, _, files in os.walk(search_root):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            # torch_dist uses .distcp shards; also handle .pt / .bin
            if not any(fname.endswith(ext) for ext in (".pt", ".bin", ".distcp")):
                continue
            try:
                shard = torch.load(fpath, map_location="cpu", weights_only=False)
                if isinstance(shard, dict):
                    src = shard.get("model", shard)
                    if isinstance(src, dict):
                        state.update(src)
                        loaded_files += 1
            except Exception as e:
                print(f"  Skipping {fname}: {e}")

    print(f"  Loaded {loaded_files} shard file(s), {len(state)} tensors total")
    return state


def find(state, key):
    """Try bare key, 'module.' prefix, and 'module.decoder.' prefix."""
    for pfx in ("", "module.", "module.decoder."):
        v = state.get(pfx + key)
        if v is not None:
            return v
    return None


def remap(state):
    hf = {}

    # ── show first 15 keys to confirm prefix style ──
    print("  Sample keys from checkpoint:")
    for k in list(state.keys())[:15]:
        print(f"    {k}")

    # Embeddings
    hf["model.embed_tokens.weight"] = find(state, "embedding.word_embeddings.weight")
    hf["lm_head.weight"]            = find(state, "output_layer.weight")
    hf["model.norm.weight"]         = find(state, "decoder.final_layernorm.weight")

    q_size  = NUM_HEADS    * HEAD_DIM   # 14 * 64 = 896
    kv_size = NUM_KV_HEADS * HEAD_DIM   #  2 * 64 = 128

    for i in range(NUM_LAYERS):
        p = f"decoder.layers.{i}"

        # ── LayerNorms ──────────────────────────────────────────────────────
        hf[f"model.layers.{i}.input_layernorm.weight"] = \
            find(state, f"{p}.input_layernorm.weight")
        hf[f"model.layers.{i}.post_attention_layernorm.weight"] = \
            find(state, f"{p}.pre_mlp_layernorm.weight")

        # ── Attention ───────────────────────────────────────────────────────
        # Megatron fuses Q+K+V into one ColumnParallelLinear:
        #   shape = [(num_heads + 2*num_kv_heads) * head_dim, hidden]
        #         = [(14 + 4) * 64, 896] = [1152, 896]
        qkv = find(state, f"{p}.self_attention.linear_qkv.weight")
        if qkv is not None:
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=0)
            hf[f"model.layers.{i}.self_attn.q_proj.weight"] = q
            hf[f"model.layers.{i}.self_attn.k_proj.weight"] = k
            hf[f"model.layers.{i}.self_attn.v_proj.weight"] = v
        else:
            print(f"  WARNING: QKV missing at layer {i}")

        hf[f"model.layers.{i}.self_attn.o_proj.weight"] = \
            find(state, f"{p}.self_attention.linear_proj.weight")

        # ── MLP (SwiGLU) ────────────────────────────────────────────────────
        # Megatron fuses gate+up into linear_fc1:
        #   shape = [2 * ffn_hidden, hidden] = [9728, 896]
        gate_up = find(state, f"{p}.mlp.linear_fc1.weight")
        if gate_up is not None:
            gate, up = gate_up.chunk(2, dim=0)
            hf[f"model.layers.{i}.mlp.gate_proj.weight"] = gate
            hf[f"model.layers.{i}.mlp.up_proj.weight"]   = up
        else:
            print(f"  WARNING: gate_up missing at layer {i}")

        hf[f"model.layers.{i}.mlp.down_proj.weight"] = \
            find(state, f"{p}.mlp.linear_fc2.weight")

    return hf


def main():
    iteration = TARGET_ITER if TARGET_ITER else get_latest_iter(CKPT_DIR)
    iter_dir  = os.path.join(CKPT_DIR, f"iter_{iteration:07d}")
    out_dir   = os.path.join(HF_OUT_DIR, f"iter{iteration:07d}")

    print(f"{'='*60}")
    print(f"Checkpoint : {iter_dir}")
    print(f"Output     : {out_dir}")
    print(f"{'='*60}\n")

    if not os.path.isdir(iter_dir):
        raise FileNotFoundError(f"Not found: {iter_dir}")

    print("Step 1 — Loading Megatron checkpoint …")
    state = load_megatron_state(iter_dir)

    if not state:
        print("\nDirectory contents:")
        for f in os.listdir(iter_dir):
            print(f"  {f}")
        raise RuntimeError("No tensors loaded — see directory listing above")

    print("\nStep 2 — Remapping weights …")
    hf_state = remap(state)

    missing = [k for k, v in hf_state.items() if v is None]
    if missing:
        print(f"\nWARNING — {len(missing)} unmapped keys:")
        for k in missing:
            print(f"  {k}")
    else:
        print("  All weights mapped successfully ✓")

    print("\nStep 3 — Building HuggingFace model …")
    config    = Qwen2Config.from_pretrained(HF_BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL)
    model     = AutoModelForCausalLM.from_config(config)

    result = model.load_state_dict(hf_state, strict=False)
    if result.missing_keys:
        print(f"  HF missing keys  : {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Unexpected keys  : {result.unexpected_keys}")

    model = model.to(torch.bfloat16)

    print(f"\nStep 4 — Saving to {out_dir} …")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "source_checkpoint": iter_dir,
        "iteration": iteration,
        "base_model": HF_BASE_MODEL,
        "num_layers": NUM_LAYERS,
        "hidden_size": HIDDEN,
        "num_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "ffn_hidden_size": FFN,
        "head_dim": HEAD_DIM,
    }
    with open(os.path.join(out_dir, "conversion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone ✓  →  {out_dir}")


if __name__ == "__main__":
    main()
