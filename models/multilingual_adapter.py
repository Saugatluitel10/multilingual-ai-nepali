"""
Adapter module for efficient fine-tuning of multilingual transformer models.
Supports insertion after attention blocks for XLM-Roberta, mBERT, or LLaMA-3.1 variants.
"""

import torch
import torch.nn as nn

class Adapter(nn.Module):
    """
    Lightweight adapter layer for parameter-efficient fine-tuning.
    """
    def __init__(self, hidden_size, reduction_factor=16, non_linearity='relu'):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, hidden_size // reduction_factor)
        if non_linearity == 'relu':
            self.activation = nn.ReLU()
        elif non_linearity == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported non-linearity: {non_linearity}")
        self.up_proj = nn.Linear(hidden_size // reduction_factor, hidden_size)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return x + residual


def add_adapters_to_transformer(model, reduction_factor=16, non_linearity='relu', after_attention=True):
    """
    Insert adapters into each layer of a HuggingFace transformer model.
    Args:
        model: Pretrained transformer model (e.g., XLMRobertaModel, BertModel)
        reduction_factor: Adapter bottleneck size
        non_linearity: Activation function
        after_attention: If True, insert after attention block; else after feed-forward
    Returns:
        The model with adapters inserted.
    """
    for i, layer in enumerate(model.encoder.layer):
        hidden_size = layer.output.dense.out_features
        adapter = Adapter(hidden_size, reduction_factor, non_linearity)
        if after_attention:
            # Insert after self-attention output
            orig_fn = layer.attention.output.forward
            def new_forward(self, hidden_states, input_tensor):
                out = orig_fn(hidden_states, input_tensor)
                return adapter(out)
            layer.attention.output.forward = new_forward.__get__(layer.attention.output, type(layer.attention.output))
        else:
            # Insert after intermediate/feed-forward output
            orig_fn = layer.output.forward
            def new_forward(self, hidden_states, input_tensor):
                out = orig_fn(hidden_states, input_tensor)
                return adapter(out)
            layer.output.forward = new_forward.__get__(layer.output, type(layer.output))
    return model
