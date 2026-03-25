# scripts/train_qwen25.py
import sys
sys.path.insert(0, "/opt/megatron-lm")

from megatron.training import get_args, pretrain
from megatron.training.tokenizer import build_tokenizer
from megatron.core.models.gpt import GPTModel
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.spec_utils import import_module
from megatron.core import mpu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
import torch


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    config = core_transformer_config_from_args(args)
    transformer_layer_spec = get_gpt_layer_local_spec(
        num_experts=None, moe_grouped_gemm=False
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
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


def forward_step(data_iterator, model):
    from megatron.training import get_tokenizer
    from megatron.core.pipeline_parallel import get_forward_backward_func
    args = get_args()

    data = next(data_iterator)
    tokens = data["text"].long().cuda()
    labels = tokens[:, 1:].contiguous()
    input_ids = tokens[:, :-1].contiguous()
    attention_mask = None  # causal by default

    output = model(input_ids, None, attention_mask)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output.view(-1, output.size(-1)), labels.view(-1))
    return loss, {"lm loss": loss.detach()}


def train_valid_test_datasets_provider(train_val_test_num_samples):
    from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset, GPTDataset
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    args = get_args()

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=args.seq_length,
        blend=[[args.data_path[0]], [1.0]],
        split=args.split,
        num_workers=4,
    )

    datasets = BlendedMegatronDatasetBuilder(
        GPTDataset, train_val_test_num_samples, lambda: True, config
    ).build()
    return datasets


if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
    )
