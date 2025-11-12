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
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.preprocess import TextPreprocessor
from data.augmentation import MultilingualAugmenter

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

def train_model(config_path: str):
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
    print(f"Loading model: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['name'],
        num_labels=config['model']['num_labels']
    )
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_df, tokenizer, config['data']['max_length'])
    val_dataset = prepare_dataset(val_df, tokenizer, config['data']['max_length'])
    
    # Training arguments
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
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=config['output']['log_dir'],
        logging_steps=100,
        report_to="tensorboard"
    )
    
    # Initialize trainer
    trainer = Trainer(
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
    trainer.save_model(config['output']['model_dir'])
    tokenizer.save_pretrained(config['output']['model_dir'])
    
    print("Training completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/sentiment_config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    train_model(args.config)
