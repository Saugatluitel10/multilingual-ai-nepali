"""
Tokenizer utilities for multilingual models
Supports XLM-RoBERTa, IndicBERT, mT5, and other multilingual tokenizers
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import torch
from transformers import (
    AutoTokenizer,
    XLMRobertaTokenizer,
    MT5Tokenizer,
    PreTrainedTokenizer
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultilingualTokenizer:
    """
    Wrapper for multilingual tokenizers with support for code-mixed text
    """
    
    # Supported model types and their tokenizers
    SUPPORTED_MODELS = {
        'xlm-roberta-base': 'xlm-roberta-base',
        'xlm-roberta-large': 'xlm-roberta-large',
        'indicbert': 'ai4bharat/indic-bert',
        'muril': 'google/muril-base-cased',
        'mt5-small': 'google/mt5-small',
        'mt5-base': 'google/mt5-base',
        'mbert': 'bert-base-multilingual-cased',
    }
    
    def __init__(self, model_name: str = 'xlm-roberta-base', 
                 cache_dir: Optional[str] = None):
        """
        Initialize tokenizer
        
        Args:
            model_name: Name of the model or tokenizer
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.getenv('MODEL_CACHE_DIR', './models/cache')
        
        # Get the actual model path
        self.model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        
        logger.info(f"✅ Loaded tokenizer: {self.model_name}")
        logger.info(f"   Vocab size: {len(self.tokenizer)}")
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load the appropriate tokenizer
        
        Returns:
            Loaded tokenizer
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                cache_dir=self.cache_dir
            )
            return tokenizer
        except Exception as e:
            logger.error(f"❌ Error loading tokenizer: {e}")
            raise
    
    def tokenize(self, text: Union[str, List[str]], 
                 max_length: int = 128,
                 padding: Union[bool, str] = True,
                 truncation: bool = True,
                 return_tensors: Optional[str] = None) -> Dict:
        """
        Tokenize text
        
        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: Return format ('pt' for PyTorch, 'tf' for TensorFlow)
            
        Returns:
            Dictionary with tokenized outputs
        """
        try:
            encoded = self.tokenizer(
                text,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors
            )
            return encoded
        except Exception as e:
            logger.error(f"❌ Tokenization error: {e}")
            raise
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(self, texts: List[str], 
                     max_length: int = 128,
                     padding: bool = True,
                     truncation: bool = True) -> Dict:
        """
        Batch encode multiple texts
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            Dictionary with encoded outputs
        """
        return self.tokenize(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors='pt'
        )
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.tokenizer)
    
    def get_special_tokens(self) -> Dict[str, Union[str, int]]:
        """
        Get special tokens information
        
        Returns:
            Dictionary with special tokens
        """
        return {
            'pad_token': self.tokenizer.pad_token,
            'pad_token_id': self.tokenizer.pad_token_id,
            'unk_token': self.tokenizer.unk_token,
            'unk_token_id': self.tokenizer.unk_token_id,
            'sep_token': self.tokenizer.sep_token,
            'sep_token_id': self.tokenizer.sep_token_id,
            'cls_token': self.tokenizer.cls_token,
            'cls_token_id': self.tokenizer.cls_token_id,
            'mask_token': self.tokenizer.mask_token if hasattr(self.tokenizer, 'mask_token') else None,
            'mask_token_id': self.tokenizer.mask_token_id if hasattr(self.tokenizer, 'mask_token_id') else None,
        }
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text tokenization
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with analysis results
        """
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text)
        
        return {
            'original_text': text,
            'tokens': tokens,
            'token_ids': token_ids,
            'num_tokens': len(tokens),
            'num_token_ids': len(token_ids),
            'decoded_text': self.decode(token_ids)
        }
    
    def save_tokenizer(self, save_path: str):
        """
        Save tokenizer to disk
        
        Args:
            save_path: Path to save tokenizer
        """
        try:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"💾 Saved tokenizer to {save_path}")
        except Exception as e:
            logger.error(f"❌ Error saving tokenizer: {e}")
            raise


def compare_tokenizers(text: str, model_names: List[str] = None) -> Dict:
    """
    Compare tokenization across different models
    
    Args:
        text: Input text to tokenize
        model_names: List of model names to compare
        
    Returns:
        Dictionary with comparison results
    """
    if model_names is None:
        model_names = ['xlm-roberta-base', 'indicbert', 'mbert']
    
    results = {}
    
    for model_name in model_names:
        try:
            tokenizer = MultilingualTokenizer(model_name)
            analysis = tokenizer.analyze_text(text)
            results[model_name] = {
                'num_tokens': analysis['num_tokens'],
                'tokens': analysis['tokens'][:10],  # First 10 tokens
                'vocab_size': tokenizer.get_vocab_size()
            }
        except Exception as e:
            logger.warning(f"⚠️  Could not load {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


def create_dataset_tokenizer(texts: List[str], 
                            labels: List[int],
                            tokenizer: MultilingualTokenizer,
                            max_length: int = 128) -> Tuple[Dict, List[int]]:
    """
    Create tokenized dataset for training
    
    Args:
        texts: List of input texts
        labels: List of labels
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (encoded_data, labels)
    """
    logger.info(f"📝 Tokenizing {len(texts)} texts...")
    
    encoded = tokenizer.batch_encode(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True
    )
    
    # Add labels
    encoded['labels'] = torch.tensor(labels)
    
    logger.info(f"✅ Tokenization complete!")
    logger.info(f"   Input IDs shape: {encoded['input_ids'].shape}")
    logger.info(f"   Attention mask shape: {encoded['attention_mask'].shape}")
    
    return encoded, labels


def get_tokenizer_for_model(model_name: str, **kwargs) -> MultilingualTokenizer:
    """
    Factory function to get tokenizer for a specific model
    
    Args:
        model_name: Name of the model
        **kwargs: Additional arguments for tokenizer
        
    Returns:
        MultilingualTokenizer instance
    """
    return MultilingualTokenizer(model_name, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("🔧 Tokenizer Utilities Demo\n")
    
    # Sample Nepali-English code-mixed text
    sample_texts = [
        "यो movie राम्रो थियो but ending disappointing थियो",
        "I love Nepali खाना especially momo",
        "Weather आज बहुत अच्छा छ"
    ]
    
    # Test XLM-RoBERTa tokenizer
    print("📝 Testing XLM-RoBERTa tokenizer:")
    tokenizer = MultilingualTokenizer('xlm-roberta-base')
    
    for text in sample_texts:
        analysis = tokenizer.analyze_text(text)
        print(f"\nText: {text}")
        print(f"Tokens: {analysis['tokens'][:10]}...")
        print(f"Num tokens: {analysis['num_tokens']}")
    
    # Compare tokenizers
    print("\n\n📊 Comparing tokenizers:")
    comparison = compare_tokenizers(sample_texts[0])
    for model, result in comparison.items():
        print(f"\n{model}:")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Vocab size: {result['vocab_size']}")
            print(f"  Num tokens: {result['num_tokens']}")
    
    print("\n✅ Tokenizer utilities ready!")
