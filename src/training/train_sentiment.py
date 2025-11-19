"""
Training script for sentiment analysis model
"""

import os
import yaml
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from training.custom_trainer import CustomTrainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.preprocess import TextPreprocessor
from data.augmentation import MultilingualAugmenter

# Import custom model modules
from models.classification_model import MultilingualClassificationModel

def load_adapter_config(model_config: dict) -> dict:
    """Extract adapter config from model config dict."""
    adapter_keys = ['adapter_config', 'reduction_factor', 'non_linearity', 'after_attention']
    if 'adapter_config' in model_config:
        return model_config['adapter_config']
    # Support legacy flat configs
    return {k: model_config[k] for k in adapter_keys if k in model_config}

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess data"""
    df = pd.read_csv(data_path)
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_dataset(df)

def prepare_dataset(df: pd.DataFrame, tokenizer, max_length: int = 128):
    """Prepare dataset for training"""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    
    dataset = Dataset.from_pandas(df[['processed_text', 'label']].rename(
        columns={'processed_text': 'text'}
    ))
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(config_path: str, model_type: str = 'hf', advanced_logging: bool = False):
    """Main training function"""
    # Load configuration
    config = load_config(config_path)
    
    print("Loading data...")
    train_df = load_data(config['data']['train_path'])
    val_df = load_data(config['data']['val_path'])
    
    # Data augmentation if enabled
    if config['data'].get('augmentation', False):
        print("Applying data augmentation...")
        augmenter = MultilingualAugmenter()
        aug_factor = config['data'].get('augmentation_factor', 2)
        
        aug_texts, aug_labels = augmenter.augment_dataset(
            train_df['processed_text'].tolist(),
            train_df['label'].tolist(),
            augmentation_factor=aug_factor
        )
        
        train_df = pd.DataFrame({
            'processed_text': aug_texts,
            'label': aug_labels
        })
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Load tokenizer and model
    print(f"Loading model: {config['model']['name']} ({model_type})")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if model_type == 'adapter':
        adapter_config = load_adapter_config(config['model'])
        model = MultilingualClassificationModel(
            model_name=config['model']['name'],
            num_labels=config['model']['num_labels'],
            adapter_config=adapter_config
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model']['name'],
            num_labels=config['model']['num_labels']
        )
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_df, tokenizer, config['data']['max_length'])
    val_dataset = prepare_dataset(val_df, tokenizer, config['data']['max_length'])
    
    # Prepare checkpoint directory
    checkpoint_dir = os.path.join(config['output']['model_dir'], 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
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
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=config['output']['log_dir'],
        logging_steps=100,
        report_to="tensorboard",
        fp16=True,  # Mixed precision
        gradient_checkpointing=True
    )
    
    # Initialize trainer
    trainer_cls = CustomTrainer if advanced_logging else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Auto-generate evaluation summary
    import json
    from datetime import datetime
    summary = {
        'model_name': config['model']['name'],
        'adapter': model_type == 'adapter',
        'config': config,
        'best_metric': trainer.state.best_metric,
        'best_model_checkpoint': trainer.state.best_model_checkpoint,
        'metrics': trainer.state.log_history,
        'timestamp': datetime.now().isoformat()
    }
    md = f"""# Evaluation Summary\n\n- **Model:** {config['model']['name']}\n- **Adapter:** {model_type == 'adapter'}\n- **Best Metric:** {trainer.state.best_metric}\n- **Best Checkpoint:** {trainer.state.best_model_checkpoint}\n- **Timestamp:** {summary['timestamp']}\n\n## Metrics\n\n"""
    for log in trainer.state.log_history:
        if 'eval_loss' in log:
            md += f"Step {log.get('step','?')}: " + ", ".join([f"{k}: {v:.4f}" for k, v in log.items() if isinstance(v, float)]) + "\n"
    with open(os.path.join(checkpoint_dir, 'eval_summary.md'), 'w') as f:
        f.write(md)
    with open(os.path.join(checkpoint_dir, 'eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAuto-generated evaluation summary at {checkpoint_dir}/eval_summary.md and .json")
    print("Training completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train sentiment analysis model with or without adapters.")
    parser.add_argument('--config', type=str, default='configs/sentiment_config.yaml',
                       help='Path to config file')
    parser.add_argument('--model_type', type=str, default='hf', choices=['hf', 'adapter'],
                       help='Model type: "hf" (HuggingFace) or "adapter" (with adapters)')
    parser.add_argument('--advanced_logging', action='store_true', help='Enable advanced GPU/memory/logging (CustomTrainer)')
    args = parser.parse_args()
    
    train_model(args.config, model_type=args.model_type, advanced_logging=args.advanced_logging)
