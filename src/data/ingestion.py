"""
Data ingestion module for loading multilingual datasets
Supports CSV/JSON files and API-based data collection
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
import requests
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Data ingestion class for loading and collecting multilingual data
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data ingestion
        
        Args:
            data_dir: Directory to store raw data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_csv(self, file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding (default: utf-8)
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"✅ Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading CSV: {e}")
            raise
    
    def load_json(self, file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load data from JSON file
        
        Args:
            file_path: Path to JSON file
            encoding: File encoding (default: utf-8)
            
        Returns:
            DataFrame with loaded data
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            # Handle both list of dicts and dict with records
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(data)
                
            logger.info(f"✅ Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading JSON: {e}")
            raise
    
    def load_jsonl(self, file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load data from JSONL (JSON Lines) file
        
        Args:
            file_path: Path to JSONL file
            encoding: File encoding (default: utf-8)
            
        Returns:
            DataFrame with loaded data
        """
        try:
            data = []
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            df = pd.DataFrame(data)
            logger.info(f"✅ Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading JSONL: {e}")
            raise
    
    def load_huggingface_dataset(self, dataset_name: str, split: str = 'train', 
                                 subset: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset from Hugging Face Hub
        
        Args:
            dataset_name: Name of the dataset on HF Hub
            split: Dataset split (train/validation/test)
            subset: Optional subset/configuration name
            
        Returns:
            DataFrame with loaded data
        """
        try:
            from datasets import load_dataset
            
            logger.info(f"📥 Loading {dataset_name} from Hugging Face...")
            
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            df = pd.DataFrame(dataset)
            logger.info(f"✅ Loaded {len(df)} rows from {dataset_name}")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading HF dataset: {e}")
            raise
    
    def scrape_nepali_news(self, url: str, max_articles: int = 100) -> pd.DataFrame:
        """
        Scrape Nepali news articles from a given URL
        
        Args:
            url: Base URL of the news portal
            max_articles: Maximum number of articles to scrape
            
        Returns:
            DataFrame with scraped articles
        """
        # Placeholder implementation - customize based on specific news portal
        logger.warning("⚠️  News scraping requires site-specific implementation")
        logger.info("Please implement scraping logic based on target website structure")
        
        # Example structure
        articles = []
        # TODO: Implement actual scraping logic
        # Use BeautifulSoup or Scrapy for web scraping
        
        return pd.DataFrame(articles)
    
    def fetch_twitter_data(self, api_key: Optional[str] = None, 
                          query: str = "", max_results: int = 100) -> pd.DataFrame:
        """
        Fetch data from Twitter API (requires Twitter Academic API access)
        
        Args:
            api_key: Twitter API key (or load from environment)
            query: Search query for tweets
            max_results: Maximum number of tweets to fetch
            
        Returns:
            DataFrame with tweet data
        """
        api_key = api_key or os.getenv('TWITTER_API_KEY')
        
        if not api_key:
            logger.warning("⚠️  Twitter API key not found")
            logger.info("Set TWITTER_API_KEY in .env file to use Twitter data collection")
            return pd.DataFrame()
        
        # Placeholder for Twitter API integration
        logger.info(f"🐦 Fetching tweets with query: {query}")
        
        # TODO: Implement Twitter API v2 integration
        # Use tweepy or requests to fetch data
        
        tweets = []
        return pd.DataFrame(tweets)
    
    def combine_datasets(self, datasets: List[pd.DataFrame], 
                        remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Combine multiple datasets into one
        
        Args:
            datasets: List of DataFrames to combine
            remove_duplicates: Whether to remove duplicate rows
            
        Returns:
            Combined DataFrame
        """
        if not datasets:
            logger.warning("⚠️  No datasets to combine")
            return pd.DataFrame()
        
        combined = pd.concat(datasets, ignore_index=True)
        
        if remove_duplicates:
            original_len = len(combined)
            combined = combined.drop_duplicates()
            removed = original_len - len(combined)
            if removed > 0:
                logger.info(f"🗑️  Removed {removed} duplicate rows")
        
        logger.info(f"✅ Combined dataset: {len(combined)} rows")
        return combined
    
    def save_dataset(self, df: pd.DataFrame, filename: str, 
                    format: str = 'csv') -> str:
        """
        Save dataset to file
        
        Args:
            df: DataFrame to save
            filename: Output filename
            format: Output format (csv, json, jsonl)
            
        Returns:
            Path to saved file
        """
        output_path = self.data_dir / filename
        
        try:
            if format == 'csv':
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif format == 'json':
                df.to_json(output_path, orient='records', force_ascii=False, indent=2)
            elif format == 'jsonl':
                df.to_json(output_path, orient='records', lines=True, force_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"💾 Saved dataset to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"❌ Error saving dataset: {e}")
            raise
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict:
        """
        Get information about the dataset
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        
        return info


def load_sample_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load sample datasets for testing
    
    Returns:
        Dictionary of sample datasets
    """
    # Sample Nepali-English code-mixed data
    sentiment_data = {
        'text': [
            'यो movie राम्रो थियो but ending disappointing थियो',
            'I love Nepali खाना especially momo',
            'Weather आज बहुत अच्छा छ',
            'This place is beautiful तर crowded छ',
            'मलाई यो गीत मन पर्दैन, too loud'
        ],
        'label': [1, 2, 2, 1, 0],  # 0: negative, 1: neutral, 2: positive
        'language': ['mixed', 'mixed', 'mixed', 'mixed', 'mixed']
    }
    
    misinfo_data = {
        'text': [
            'Breaking: नेपालमा नयाँ खोज भयो',
            'Scientists confirm this amazing fact',
            'यो खबर पूर्णतया सत्य हो',
        ],
        'label': [0, 1, 0],  # 0: false, 1: true
        'source': ['social_media', 'news', 'social_media']
    }
    
    return {
        'sentiment': pd.DataFrame(sentiment_data),
        'misinfo': pd.DataFrame(misinfo_data)
    }


if __name__ == "__main__":
    # Example usage
    ingestion = DataIngestion()
    
    # Load sample datasets
    samples = load_sample_datasets()
    
    print("📊 Sample Sentiment Data:")
    print(samples['sentiment'])
    print(f"\n📊 Dataset Info:")
    print(json.dumps(ingestion.get_dataset_info(samples['sentiment']), indent=2))
    
    # Save sample data
    ingestion.save_dataset(samples['sentiment'], 'sample_sentiment.csv')
    ingestion.save_dataset(samples['misinfo'], 'sample_misinfo.json')
    
    print("\n✅ Data ingestion module ready!")
