# 🚀 Model Training Guide

This guide covers how to train sentiment and misinformation models using both standard Hugging Face transformers and parameter-efficient adapters.

## 1. Standard Hugging Face Model Training

Train with the full model (all parameters):

```bash
python3 src/training/train_sentiment.py --config configs/sentiment_config.yaml --model_type hf
```

- Uses `transformers.AutoModelForSequenceClassification`.
- Fine-tunes all layers of the base model (e.g., XLM-RoBERTa, mBERT).

## 2. Adapter-Based Training (Efficient Fine-Tuning)

Train with lightweight adapter layers for parameter efficiency:

```bash
python3 src/training/train_sentiment.py --config configs/sentiment_config.yaml --model_type adapter
```

- Uses the custom `MultilingualClassificationModel` from `models/classification_model.py`.
- Only adapter layers are updated during training; base model weights remain mostly frozen.
- Adapter config is read from the YAML file:

```yaml
model:
  name: "xlm-roberta-base"
  num_labels: 3
  adapter_config:
    type: "pfeiffer"
    reduction_factor: 16
    non_linearity: "relu"
```

- **type**: Adapter architecture ("pfeiffer" is common for NLP)
- **reduction_factor**: Bottleneck size (smaller = more efficient)
- **non_linearity**: Activation function ("relu" or "gelu")

## 3. Model Selection

Supported base models:
- XLM-RoBERTa (recommended for multilingual)
- mBERT
- LLaMA-3.1 multilingual variant (if available)

## 4. Multitask Fine-Tuning with Synthetic Data

Fine-tune on both real and synthetic (augmented) data with a multitask objective:
- **Masked Language Modeling (MLM)** + **Classification**
- Logs metrics per language subset
- Supports experiment tracking with Weights & Biases

### Example CLI Usage
```bash
python3 src/training/fine_tune_llm.py \
  --config configs/sentiment_config.yaml \
  --model_type adapter \
  --mlm_weight 1.0 \
  --clf_weight 1.0 \
  --experiment multilingual_mlm_clf
```
- Add a `synthetic_path` entry under `data:` in your config to include augmented data.

### Example Config Block
```yaml
data:
  train_path: "data/processed/sentiment_train.csv"
  synthetic_path: "data/synthetic/sentiment_aug.csv"
  val_path: "data/processed/sentiment_val.csv"
  max_length: 128
```

- All other model and training options are as above.

## 5. Example Config (`configs/sentiment_config.yaml`)

```yaml
model:
  name: "xlm-roberta-base"
  num_labels: 3
  adapter_config:
    type: "pfeiffer"
    reduction_factor: 16
    non_linearity: "relu"
```

## 5. Advanced: Customizing Adapter Placement

- By default, adapters are inserted after each attention block.
- To change placement, set `after_attention: false` in `adapter_config` (places after feed-forward block).

## 6. Training Arguments

All other training arguments (epochs, batch size, etc.) are configured in the YAML file.

## 7. Output

- Models and tokenizers are saved to the directory specified in `output.model_dir`.
- Training logs and metrics are saved to `output.log_dir` and TensorBoard.

## 8. Troubleshooting

- Ensure your config YAML matches the expected structure.
- For adapter training, only the adapter layers are updated; this is much faster and more memory-efficient for low-resource settings.
- For more details on adapters, see the [AdapterHub documentation](https://adapterhub.ml/).

---

For translation model training, see `models/translation_model.py` and adapt the script similarly.
