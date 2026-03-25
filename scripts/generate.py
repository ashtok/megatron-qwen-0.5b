import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTProvider126M
from megatron.bridge.training.config import (
    ConfigContainer, TokenizerConfig, CheckpointConfig
)

# Load checkpoint and generate
bridge = AutoBridge.from_megatron_checkpoint(
    "./checkpoints/gpt2_de_scratch/iter_0010000",
    model_provider=GPTProvider126M(),
)
# Then run generation
