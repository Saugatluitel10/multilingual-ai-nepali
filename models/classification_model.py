"""
Classification heads for sentiment and misinformation detection on top of multilingual transformers.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from .multilingual_adapter import add_adapters_to_transformer

class ClassificationHead(nn.Module):
    """
    Simple classification head for sequence classification tasks.
    """
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(hidden_size, num_labels)
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class MultilingualClassificationModel(nn.Module):
    """
    Multilingual transformer with adapter and classification head.
    Supports XLM-Roberta, mBERT, LLaMA-3.1 variants.
    """
    def __init__(self, model_name, num_labels, adapter_config=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if adapter_config:
            self.encoder = add_adapters_to_transformer(
                self.encoder,
                reduction_factor=adapter_config.get('reduction_factor', 16),
                non_linearity=adapter_config.get('non_linearity', 'relu'),
                after_attention=adapter_config.get('after_attention', True)
            )
        hidden_size = self.encoder.config.hidden_size
        self.classifier = ClassificationHead(hidden_size, num_labels)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token (or first token)
        return self.classifier(pooled)
