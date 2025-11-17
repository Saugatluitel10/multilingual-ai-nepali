"""
Multitask fine-tuning script for LLMs on real + synthetic data.
Supports masked language modeling (MLM) and classification multitask objectives.
Logs performance by language subset and supports experiment tracking.
"""

import os
import sys
import argparse
import yaml
import torch
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, f1_score, recall_score
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.preprocess import TextPreprocessor
from data.augmentation import MultilingualAugmenter
from models.classification_model import MultilingualClassificationModel

# ---- Utility Functions ----
def compute_metrics_multitask(pred, val_df=None, tokenizer=None):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    metrics = {'accuracy': acc, 'f1': f1, 'recall': recall}
    # Per-language logging
    if val_df is not None:
        for lang in val_df['language'].unique():
            mask = val_df['language'] == lang
            if mask.sum() > 0:
                lang_labels = labels[mask]
                lang_preds = preds[mask]
                metrics[f'f1_{lang}'] = f1_score(lang_labels, lang_preds, average='weighted')
                metrics[f'acc_{lang}'] = accuracy_score(lang_labels, lang_preds)
    # BLEU (for translation tasks, placeholder)
    if 'references' in pred and 'translations' in pred and tokenizer is not None:
        references = [[tokenizer.tokenize(ref)] for ref in pred.references]
        translations = [tokenizer.tokenize(hyp) for hyp in pred.translations]
        bleu = corpus_bleu(references, translations)
        metrics['bleu'] = bleu
    return metrics

# ---- Data Loading ----
def load_and_prepare_data(config, preprocessor):
    # Load real and synthetic (augmented) data
    train_real = pd.read_csv(config['data']['train_path'])
    train_synth = pd.read_csv(config['data']['synthetic_path']) if 'synthetic_path' in config['data'] else None
    val_df = pd.read_csv(config['data']['val_path'])
    train_real = preprocessor.preprocess_dataset(train_real)
    if train_synth is not None:
        train_synth = preprocessor.preprocess_dataset(train_synth)
        train_df = pd.concat([train_real, train_synth], ignore_index=True)
    else:
        train_df = train_real
    val_df = preprocessor.preprocess_dataset(val_df)
    return train_df, val_df

# ---- Multitask Dataset Preparation ----
def prepare_multitask_dataset(df, tokenizer, max_length=128):
    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    dataset = Dataset.from_pandas(df[['processed_text', 'label']].rename(columns={'processed_text': 'text'}))
    tokenized = dataset.map(tokenize_fn, batched=True)
    return tokenized

# ---- Main Training Loop ----
def main():
    parser = argparse.ArgumentParser(description="Multitask fine-tuning: MLM + Classification on real + synthetic data.")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config.')
    parser.add_argument('--model_type', type=str, default='adapter', choices=['hf', 'adapter'], help='Model type.')
    parser.add_argument('--mlm_weight', type=float, default=1.0, help='Weight for MLM loss.')
    parser.add_argument('--clf_weight', type=float, default=1.0, help='Weight for classification loss.')
    parser.add_argument('--experiment', type=str, default='multitask_run', help='Experiment name for tracking.')
    args = parser.parse_args()

    # Config and logging
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    wandb.init(project="multilingual-ai-nepali", name=args.experiment, config=config)

    preprocessor = TextPreprocessor()
    train_df, val_df = load_and_prepare_data(config, preprocessor)

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    num_labels = config['model']['num_labels']
    adapter_config = config['model'].get('adapter_config', None)

    # Prepare datasets
    train_dataset = prepare_multitask_dataset(train_df, tokenizer, config['data']['max_length'])
    val_dataset = prepare_multitask_dataset(val_df, tokenizer, config['data']['max_length'])

    # Model
    if args.model_type == 'adapter':
        model = MultilingualClassificationModel(
            model_name=config['model']['name'],
            num_labels=num_labels,
            adapter_config=adapter_config
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model']['name'],
            num_labels=num_labels
        )
    # MLM (masked language modeling) head
    mlm_model = AutoModelForMaskedLM.from_pretrained(config['model']['name'])
    mlm_model.resize_token_embeddings(len(tokenizer))

    # ---- Custom Trainer for Multitask ----
    class MultitaskTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Classification loss
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            clf_loss = torch.nn.functional.cross_entropy(outputs.logits, inputs['labels'])
            # MLM loss (random mask)
            mlm_inputs = inputs['input_ids'].clone()
            mask_token_id = tokenizer.mask_token_id
            rand = torch.rand(mlm_inputs.shape)
            mask_arr = (rand < 0.15) & (mlm_inputs != tokenizer.pad_token_id)
            selection = []
            for i in range(mlm_inputs.shape[0]):
                selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
            for i in range(mlm_inputs.shape[0]):
                mlm_inputs[i, selection[i]] = mask_token_id
            mlm_labels = inputs['input_ids'].clone()
            mlm_labels[~mask_arr] = -100
            mlm_outputs = mlm_model(input_ids=mlm_inputs, attention_mask=inputs['attention_mask'], labels=mlm_labels)
            mlm_loss = mlm_outputs.loss
            # Weighted multitask loss
            loss = args.clf_weight * clf_loss + args.mlm_weight * mlm_loss
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=config['output']['model_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        evaluation_strategy="steps",
        eval_steps=config['output']['eval_steps'],
        save_steps=config['output']['save_steps'],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=config['output']['log_dir'],
        logging_steps=100,
        report_to="wandb"
    )

    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_multitask,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model(config['output']['model_dir'])
    tokenizer.save_pretrained(config['output']['model_dir'])
    wandb.finish()

if __name__ == "__main__":
    main()
