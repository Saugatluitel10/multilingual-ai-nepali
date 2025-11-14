"""
Encoder-decoder model for Nepali↔English translation using multilingual transformers.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

class MultilingualTranslationModel(nn.Module):
    """
    Encoder-decoder translation model for Nepali↔English.
    Wraps a multilingual seq2seq model (e.g., mBART, mT5).
    """
    def __init__(self, model_name='facebook/mbart-large-50-many-to-many-mmt'):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
