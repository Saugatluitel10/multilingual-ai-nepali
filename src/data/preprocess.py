"""
Data preprocessing utilities for Nepali-English code-mixed text
"""

import re
import pandas as pd
from typing import List, Tuple
import unicodedata

class TextPreprocessor:
    """Preprocessor for multilingual and code-mixed text"""
    
    def __init__(self):
        self.nepali_unicode_range = r'[\u0900-\u097F]'
        
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        return unicodedata.normalize('NFKC', text)
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_mentions(self, text: str) -> str:
        """Remove @mentions"""
        return re.sub(r'@\w+', '', text)
    
    def remove_hashtags(self, text: str, keep_text: bool = True) -> str:
        """Remove hashtags, optionally keeping the text"""
        if keep_text:
            return re.sub(r'#', '', text)
        return re.sub(r'#\w+', '', text)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace"""
        return ' '.join(text.split())
    
    def detect_language_mix(self, text: str) -> str:
        """Detect if text is Nepali, English, or mixed"""
        has_nepali = bool(re.search(self.nepali_unicode_range, text))
        has_english = bool(re.search(r'[a-zA-Z]', text))
        
        if has_nepali and has_english:
            return "mixed"
        elif has_nepali:
            return "nepali"
        elif has_english:
            return "english"
        else:
            return "unknown"
    
    def preprocess(self, text: str, remove_urls: bool = True, 
                   remove_mentions: bool = True, 
                   remove_hashtags: bool = False) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            remove_urls: Whether to remove URLs
            remove_mentions: Whether to remove @mentions
            remove_hashtags: Whether to remove hashtags
            
        Returns:
            Preprocessed text
        """
        # Normalize unicode
        text = self.normalize_unicode(text)
        
        # Remove unwanted elements
        if remove_urls:
            text = self.remove_urls(text)
        if remove_mentions:
            text = self.remove_mentions(text)
        if remove_hashtags:
            text = self.remove_hashtags(text)
        
        # Clean whitespace
        text = self.remove_extra_whitespace(text)
        
        return text.strip()
    
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Preprocess entire dataset
        
        Args:
            df: Input dataframe
            text_column: Name of the text column
            
        Returns:
            Preprocessed dataframe with additional metadata
        """
        df = df.copy()
        
        # Preprocess text
        df['processed_text'] = df[text_column].apply(self.preprocess)
        
        # Add language detection
        df['language'] = df['processed_text'].apply(self.detect_language_mix)
        
        # Add text length
        df['text_length'] = df['processed_text'].apply(len)
        
        return df

def load_and_preprocess(file_path: str, text_column: str = 'text') -> pd.DataFrame:
    """
    Load and preprocess data from file
    
    Args:
        file_path: Path to data file (CSV)
        text_column: Name of text column
        
    Returns:
        Preprocessed dataframe
    """
    df = pd.read_csv(file_path)
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_dataset(df, text_column)

if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    # Test with code-mixed text
    sample_text = "यो movie राम्रो थियो but the ending was disappointing #nepalicinema"
    processed = preprocessor.preprocess(sample_text)
    language = preprocessor.detect_language_mix(processed)
    
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")
    print(f"Language: {language}")
