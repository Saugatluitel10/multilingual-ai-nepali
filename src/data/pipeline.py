"""
Complete data pipeline integrating ingestion, preprocessing, augmentation, and tokenization
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from ingestion import DataIngestion, load_sample_datasets
from preprocess import TextPreprocessor, load_and_preprocess
from augmentation import MultilingualAugmenter
from tokenizer_utils import MultilingualTokenizer, create_dataset_tokenizer
from validation import DataValidator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Complete data pipeline for multilingual NLP
    """
    
    def __init__(self, config_path: str = "windsurf.json"):
        """
        Initialize data pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.ingestion = DataIngestion(
            data_dir=self.config.get('data', {}).get('output', {}).get('processed_dir', 'data/raw')
        )
        self.preprocessor = TextPreprocessor()
        self.augmenter = MultilingualAugmenter(use_gpu=False)
        self.validator = DataValidator(config_path)
        
        # Configuration
        self.preprocessing_enabled = self.config.get('data', {}).get('ingestion', {}).get('preprocessing', {}).get('enabled', True)
        self.augmentation_enabled = self.config.get('data', {}).get('ingestion', {}).get('augmentation', {}).get('enabled', True)
        self.validation_enabled = self.config.get('data', {}).get('validation', {}).get('enabled', True)
        
        logger.info("✅ Data pipeline initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from windsurf.json"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"⚠️  Config file not found: {self.config_path}, using defaults")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing config file: {e}")
            return {}
    
    def run_full_pipeline(self, 
                         input_path: str,
                         output_dir: str = "data/processed",
                         task_type: str = "sentiment",
                         tokenizer_name: str = "xlm-roberta-base") -> Tuple[Dict, pd.DataFrame]:
        """
        Run the complete data pipeline
        
        Args:
            input_path: Path to input data file
            output_dir: Directory to save processed data
            task_type: Type of task (sentiment, misinfo, etc.)
            tokenizer_name: Name of tokenizer to use
            
        Returns:
            Tuple of (tokenized_data, processed_dataframe)
        """
        logger.info("🚀 Starting full data pipeline...")
        logger.info(f"   Input: {input_path}")
        logger.info(f"   Output: {output_dir}")
        logger.info(f"   Task: {task_type}")
        
        # Step 1: Data Ingestion
        logger.info("\n📥 Step 1: Data Ingestion")
        df = self._ingest_data(input_path)
        logger.info(f"   Loaded {len(df)} samples")
        
        # Step 2: Data Validation (pre-processing)
        if self.validation_enabled:
            logger.info("\n🔍 Step 2: Data Validation")
            is_valid, errors = self.validator.validate_dataframe(df, "input_data")
            if not is_valid:
                logger.warning(f"   Validation found {len(errors)} issues")
        
        # Step 3: Data Preprocessing
        if self.preprocessing_enabled:
            logger.info("\n🧹 Step 3: Data Preprocessing")
            df = self._preprocess_data(df)
            logger.info(f"   Preprocessed {len(df)} samples")
        
        # Step 4: Data Augmentation
        if self.augmentation_enabled:
            logger.info("\n🔄 Step 4: Data Augmentation")
            df = self._augment_data(df)
            logger.info(f"   Augmented to {len(df)} samples")
        
        # Step 5: Tokenization
        logger.info("\n📝 Step 5: Tokenization")
        tokenized_data = self._tokenize_data(df, tokenizer_name)
        logger.info(f"   Tokenized {len(df)} samples")
        
        # Step 6: Save processed data
        logger.info("\n💾 Step 6: Saving processed data")
        output_path = self._save_processed_data(df, output_dir, task_type)
        logger.info(f"   Saved to {output_path}")
        
        logger.info("\n✅ Pipeline completed successfully!")
        
        return tokenized_data, df
    
    def _ingest_data(self, input_path: str) -> pd.DataFrame:
        """Ingest data from file"""
        file_path = Path(input_path)
        
        if file_path.suffix == '.csv':
            return self.ingestion.load_csv(str(file_path))
        elif file_path.suffix == '.json':
            return self.ingestion.load_json(str(file_path))
        elif file_path.suffix == '.jsonl':
            return self.ingestion.load_jsonl(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data"""
        # Get preprocessing steps from config
        steps = self.config.get('data', {}).get('ingestion', {}).get('preprocessing', {}).get('steps', [])
        
        # Apply preprocessing
        df = self.preprocessor.preprocess_dataset(df, text_column='text')
        
        return df
    
    def _augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Augment data"""
        # Get augmentation config
        aug_config = self.config.get('data', {}).get('ingestion', {}).get('augmentation', {})
        methods = aug_config.get('methods', ['swap', 'delete', 'insert'])
        aug_factor = aug_config.get('augmentation_factor', 2)
        
        # Apply augmentation
        if 'label' in df.columns:
            texts = df['processed_text'].tolist()
            labels = df['label'].tolist()
            
            aug_texts, aug_labels = self.augmenter.augment_dataset(
                texts, labels, augmentation_factor=aug_factor
            )
            
            df_augmented = pd.DataFrame({
                'text': aug_texts,
                'processed_text': aug_texts,
                'label': aug_labels
            })
            
            return df_augmented
        
        return df
    
    def _tokenize_data(self, df: pd.DataFrame, tokenizer_name: str) -> Dict:
        """Tokenize data"""
        tokenizer = MultilingualTokenizer(tokenizer_name)
        
        texts = df['processed_text'].tolist()
        labels = df['label'].tolist() if 'label' in df.columns else [0] * len(texts)
        
        tokenized_data, _ = create_dataset_tokenizer(
            texts, labels, tokenizer, max_length=128
        )
        
        return tokenized_data
    
    def _save_processed_data(self, df: pd.DataFrame, output_dir: str, task_type: str) -> str:
        """Save processed data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{task_type}_processed.csv"
        full_path = output_path / filename
        
        df.to_csv(full_path, index=False, encoding='utf-8')
        
        return str(full_path)
    
    def create_sample_pipeline(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Create and run pipeline with sample data
        
        Returns:
            Tuple of (tokenized_data, processed_dataframe)
        """
        logger.info("🎯 Creating sample data pipeline...")
        
        # Load sample datasets
        samples = load_sample_datasets()
        sentiment_df = samples['sentiment']
        
        # Save sample data
        sample_path = "data/raw/sample_sentiment.csv"
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        sentiment_df.to_csv(sample_path, index=False)
        
        # Run pipeline
        return self.run_full_pipeline(
            input_path=sample_path,
            output_dir="data/processed",
            task_type="sentiment",
            tokenizer_name="xlm-roberta-base"
        )
    
    def get_pipeline_stats(self) -> Dict:
        """
        Get statistics about the pipeline
        
        Returns:
            Dictionary with pipeline statistics
        """
        return {
            'preprocessing_enabled': self.preprocessing_enabled,
            'augmentation_enabled': self.augmentation_enabled,
            'validation_enabled': self.validation_enabled,
            'config_path': str(self.config_path),
            'components': {
                'ingestion': type(self.ingestion).__name__,
                'preprocessor': type(self.preprocessor).__name__,
                'augmenter': type(self.augmenter).__name__,
                'validator': type(self.validator).__name__
            }
        }


def main():
    """Main function to demonstrate pipeline"""
    print("=" * 60)
    print("🚀 Multilingual AI Data Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Show pipeline stats
    stats = pipeline.get_pipeline_stats()
    print("\n📊 Pipeline Configuration:")
    print(json.dumps(stats, indent=2))
    
    # Run sample pipeline
    print("\n" + "=" * 60)
    print("Running sample data pipeline...")
    print("=" * 60 + "\n")
    
    try:
        tokenized_data, processed_df = pipeline.create_sample_pipeline()
        
        print("\n" + "=" * 60)
        print("✅ Pipeline completed successfully!")
        print("=" * 60)
        
        print(f"\n📊 Results:")
        print(f"   Processed samples: {len(processed_df)}")
        print(f"   Tokenized shape: {tokenized_data['input_ids'].shape}")
        print(f"\n   Sample processed text:")
        for i, text in enumerate(processed_df['processed_text'].head(3), 1):
            print(f"   {i}. {text}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
