import sys
sys.path.insert(0, "/opt/megatron-lm")

from functools import partial
import torch

from megatron.training import get_args, pretrain
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules, TransformerLayer
from megatron.core.transformer.attention import SelfAttentionSubmodules, SelfAttention
from megatron.core.transformer.mlp import MLPSubmodules, MLP
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.torch_layer_norm import WrappedTorchNorm
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.training.global_vars import get_tokenizer
from megatron.training.utils import average_losses_across_data_parallel_group


def get_local_rms_spec():
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=WrappedTorchNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=None,
                    k_layernorm=None,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=WrappedTorchNorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    config = core_transformer_config_from_args(args)
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_local_rms_spec(),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
    )
    return model


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    data = next(data_iterator)
    tokens      = data["tokens"].cuda().long()
    labels      = data["labels"].cuda().long()
    loss_mask   = data["loss_mask"].cuda().float()
    position_ids = data["position_ids"].cuda().long()
    attention_mask = data["attention_mask"].cuda() if "attention_mask" in data else None
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, GPTDataset
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    args = get_args()
    tokenizer = get_tokenizer()
    # _HuggingFaceTokenizer doesn't implement eos; patch it
    try:
        _ = tokenizer.eos
    except NotImplementedError:
        type(tokenizer).eos = property(lambda self: self.tokenizer.eos_token_id)
    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=args.seq_length,
        blend=[[args.data_path[0]], [1.0]],
        split=args.split,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
    )
    datasets = BlendedMegatronDatasetBuilder(
        GPTDataset, train_val_test_num_samples, lambda: True, config
    ).build()
    return datasets


if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )
