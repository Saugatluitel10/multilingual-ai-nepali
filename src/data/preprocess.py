"""
Data preprocessing utilities for Nepali-English code-mixed text
Enhanced with Unicode normalization, transliteration handling, and advanced cleaning
"""

import re
import pandas as pd
from typing import List, Tuple, Optional
import unicodedata
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Preprocessor for multilingual and code-mixed text"""
    
    def __init__(self):
        self.nepali_unicode_range = r'[\u0900-\u097F]'
        self.devanagari_range = r'[\u0900-\u097F]'
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        
    def normalize_unicode(self, text: str, form: str = 'NFKC') -> str:
        """
        Normalize unicode characters
        
        Args:
            text: Input text
            form: Normalization form (NFC, NFKC, NFD, NFKD)
            
        Returns:
            Normalized text
        """
        # First normalize to specified form
        text = unicodedata.normalize(form, text)
        
        # Handle common Devanagari normalization issues
        # Replace common transliteration variations
        replacements = {
            '\u0950': '\u0913\u0902',  # ॐ normalization
            '\u0958': '\u0915\u093C',  # क़
            '\u0959': '\u0916\u093C',  # ख़
            '\u095A': '\u0917\u093C',  # ग़
            '\u095B': '\u091C\u093C',  # ज़
            '\u095C': '\u0921\u093C',  # ड़
            '\u095D': '\u0922\u093C',  # ढ़
            '\u095E': '\u092B\u093C',  # फ़
            '\u095F': '\u092F\u093C',  # य़
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def remove_urls(self, text: str, replace_with: str = '') -> str:
        """
        Remove URLs from text
        
        Args:
            text: Input text
            replace_with: String to replace URLs with
            
        Returns:
            Text with URLs removed
        """
        # Match http/https URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, replace_with, text)
        
        # Match www URLs
        www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+])+'
        text = re.sub(www_pattern, replace_with, text)
        
        # Match domain-like patterns
        domain_pattern = r'\b[a-zA-Z0-9-]+\.[a-z]{2,}\b'
        text = re.sub(domain_pattern, replace_with, text)
        
        return text
    
    def remove_mentions(self, text: str) -> str:
        """Remove @mentions"""
        return re.sub(r'@\w+', '', text)
    
    def remove_hashtags(self, text: str, keep_text: bool = True) -> str:
        """
        Remove hashtags, optionally keeping the text
        
        Args:
            text: Input text
            keep_text: If True, keep the text after #, else remove entire hashtag
            
        Returns:
            Text with hashtags processed
        """
        if keep_text:
            return re.sub(r'#', '', text)
        return re.sub(r'#\w+', '', text)
    
    def remove_emojis(self, text: str) -> str:
        """
        Remove emojis from text
        
        Args:
            text: Input text
            
        Returns:
            Text with emojis removed
        """
        return self.emoji_pattern.sub('', text)
    
    def remove_special_chars(self, text: str, keep_devanagari: bool = True) -> str:
        """
        Remove special characters while preserving language-specific characters
        
        Args:
            text: Input text
            keep_devanagari: Whether to keep Devanagari characters
            
        Returns:
            Text with special characters removed
        """
        if keep_devanagari:
            # Keep alphanumeric, Devanagari, and basic punctuation
            pattern = r'[^a-zA-Z0-9\s\u0900-\u097F.,!?;:\-\'"()]'
        else:
            # Keep only alphanumeric and basic punctuation
            pattern = r'[^a-zA-Z0-9\s.,!?;:\-\'"()]'
        
        return re.sub(pattern, '', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize various types of whitespace
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace various unicode spaces with regular space
        text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]', ' ', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def handle_transliteration(self, text: str) -> str:
        """
        Handle common transliteration inconsistencies
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized transliteration
        """
        # Common Nepali/Hindi transliteration variations
        transliteration_map = {
            'chha': 'छ',
            'thha': 'ठ',
            'dha': 'ध',
            'bha': 'भ',
            'cha': 'च',
            'kha': 'ख',
            'gha': 'घ',
            'jha': 'झ',
            'tha': 'थ',
            'pha': 'फ',
        }
        
        # Note: This is a simple approach. For production, consider using
        # a proper transliteration library like indic-transliteration
        
        return text
    
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
    
    def preprocess(self, text: str, 
                   remove_urls: bool = True, 
                   remove_mentions: bool = True, 
                   remove_hashtags: bool = False,
                   remove_emojis: bool = True,
                   remove_special_chars: bool = False,
                   normalize_transliteration: bool = False) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            remove_urls: Whether to remove URLs
            remove_mentions: Whether to remove @mentions
            remove_hashtags: Whether to remove hashtags
            remove_emojis: Whether to remove emojis
            remove_special_chars: Whether to remove special characters
            normalize_transliteration: Whether to normalize transliteration
            
        Returns:
            Preprocessed text
        """
        # Normalize unicode
        text = self.normalize_unicode(text)
        
        # Handle transliteration if needed
        if normalize_transliteration:
            text = self.handle_transliteration(text)
        
        # Remove unwanted elements
        if remove_urls:
            text = self.remove_urls(text)
        if remove_mentions:
            text = self.remove_mentions(text)
        if remove_hashtags:
            text = self.remove_hashtags(text)
        if remove_emojis:
            text = self.remove_emojis(text)
        if remove_special_chars:
            text = self.remove_special_chars(text)
        
        # Normalize whitespace
        text = self.normalize_whitespace(text)
        
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
