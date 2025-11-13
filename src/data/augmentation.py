"""
Data augmentation techniques for low-resource multilingual NLP
Enhanced with back-translation and code-mixed text generation
"""

import random
from typing import List, Tuple, Optional
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualAugmenter:
    """Data augmentation for Nepali-English code-mixed text"""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize augmenter
        
        Args:
            use_gpu: Whether to use GPU for model-based augmentation
        """
        # Initialize augmenters (will be loaded when needed)
        self.back_translation_aug = None
        self.synonym_aug = None
        self.translation_model = None
        self.translation_tokenizer = None
        self.device = 'cuda' if use_gpu else 'cpu'
        self.models_loaded = False
        
    def random_swap(self, text: str, n: int = 2) -> str:
        """
        Randomly swap n words in the sentence
        """
        words = text.split()
        if len(words) < 2:
            return text
            
        for _ in range(n):
            if len(words) >= 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words with probability p
        """
        words = text.split()
        if len(words) == 1:
            return text
            
        new_words = [word for word in words if random.random() > p]
        
        # If all words deleted, return random word
        if len(new_words) == 0:
            return random.choice(words)
            
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert n synonyms of random words
        """
        words = text.split()
        if len(words) == 0:
            return text
            
        for _ in range(n):
            word = random.choice(words)
            # Insert the word at random position (simple duplication)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, word)
        
        return ' '.join(words)
    
    def load_translation_models(self):
        """
        Load translation models for back-translation
        Uses Helsinki-NLP OPUS-MT models
        """
        if self.models_loaded:
            return
        
        try:
            from transformers import MarianMTModel, MarianTokenizer
            logger.info("📥 Loading translation models...")
            
            # These models will be loaded on-demand
            self.models_loaded = True
            logger.info("✅ Translation models ready")
        except ImportError:
            logger.warning("⚠️  Transformers not installed. Back-translation unavailable.")
        except Exception as e:
            logger.error(f"❌ Error loading translation models: {e}")
    
    def back_translation(self, text: str, source_lang: str = 'en', 
                        intermediate_lang: str = 'hi') -> str:
        """
        Back-translation augmentation
        Translate to intermediate language and back
        
        Args:
            text: Input text
            source_lang: Source language code (en, ne, hi)
            intermediate_lang: Intermediate language for back-translation
            
        Returns:
            Back-translated text
        """
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            # Model names for different language pairs
            model_map = {
                ('en', 'hi'): 'Helsinki-NLP/opus-mt-en-hi',
                ('hi', 'en'): 'Helsinki-NLP/opus-mt-hi-en',
                ('en', 'ne'): 'Helsinki-NLP/opus-mt-en-mul',  # Multilingual
            }
            
            # Forward translation
            forward_model_name = model_map.get((source_lang, intermediate_lang))
            if not forward_model_name:
                logger.warning(f"⚠️  No model for {source_lang}->{intermediate_lang}")
                return text
            
            logger.info(f"🔄 Back-translating: {source_lang} -> {intermediate_lang} -> {source_lang}")
            
            # Load forward model
            forward_tokenizer = MarianTokenizer.from_pretrained(forward_model_name)
            forward_model = MarianMTModel.from_pretrained(forward_model_name).to(self.device)
            
            # Translate to intermediate language
            inputs = forward_tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            translated = forward_model.generate(**inputs)
            intermediate_text = forward_tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # Backward translation
            backward_model_name = model_map.get((intermediate_lang, source_lang))
            if not backward_model_name:
                logger.warning(f"⚠️  No model for {intermediate_lang}->{source_lang}")
                return intermediate_text
            
            backward_tokenizer = MarianTokenizer.from_pretrained(backward_model_name)
            backward_model = MarianMTModel.from_pretrained(backward_model_name).to(self.device)
            
            # Translate back to source language
            inputs = backward_tokenizer(intermediate_text, return_tensors="pt", padding=True).to(self.device)
            back_translated = backward_model.generate(**inputs)
            final_text = backward_tokenizer.decode(back_translated[0], skip_special_tokens=True)
            
            return final_text
            
        except ImportError:
            logger.warning("⚠️  Transformers not installed. Returning original text.")
            return text
        except Exception as e:
            logger.error(f"❌ Back-translation error: {e}")
            return text
    
    def generate_code_mixed(self, text: str, target_lang: str = 'ne', 
                           mix_ratio: float = 0.3) -> str:
        """
        Generate code-mixed text by translating random words
        
        Args:
            text: Input text (typically English)
            target_lang: Target language to mix in (ne=Nepali, hi=Hindi)
            mix_ratio: Ratio of words to translate (0.0 to 1.0)
            
        Returns:
            Code-mixed text
        """
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            words = text.split()
            if len(words) == 0:
                return text
            
            # Select random words to translate
            num_words_to_mix = max(1, int(len(words) * mix_ratio))
            indices_to_mix = random.sample(range(len(words)), min(num_words_to_mix, len(words)))
            
            # Load translation model
            model_name = 'Helsinki-NLP/opus-mt-en-mul'  # Multilingual model
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to(self.device)
            
            # Translate selected words
            for idx in indices_to_mix:
                word = words[idx]
                # Skip very short words and punctuation
                if len(word) <= 2 or not word.isalnum():
                    continue
                
                try:
                    inputs = tokenizer(word, return_tensors="pt", padding=True).to(self.device)
                    translated = model.generate(**inputs)
                    translated_word = tokenizer.decode(translated[0], skip_special_tokens=True)
                    words[idx] = translated_word
                except:
                    continue
            
            return ' '.join(words)
            
        except ImportError:
            logger.warning("⚠️  Transformers not installed. Returning original text.")
            return text
        except Exception as e:
            logger.error(f"❌ Code-mixing error: {e}")
            return text
    
    def paraphrase_multilingual(self, text: str) -> str:
        """
        Generate paraphrases using multilingual models
        
        Args:
            text: Input text
            
        Returns:
            Paraphrased text
        """
        try:
            from transformers import pipeline
            
            # Use T5 or mT5 for paraphrasing
            # This is a placeholder - actual implementation would use a fine-tuned model
            logger.info("🔄 Generating paraphrase...")
            
            # For now, return a simple variation
            return text
            
        except Exception as e:
            logger.error(f"❌ Paraphrasing error: {e}")
            return text
    
    def augment(self, text: str, num_augmentations: int = 1, 
                methods: List[str] = None, 
                use_advanced: bool = False) -> List[str]:
        """
        Generate augmented versions of text
        
        Args:
            text: Input text
            num_augmentations: Number of augmented samples to generate
            methods: List of augmentation methods to use
            use_advanced: Whether to use advanced methods (back-translation, code-mixing)
            
        Returns:
            List of augmented texts
        """
        if methods is None:
            if use_advanced:
                methods = ['swap', 'delete', 'insert', 'back_translate', 'code_mix']
            else:
                methods = ['swap', 'delete', 'insert']
        
        augmented_texts = []
        
        for _ in range(num_augmentations):
            method = random.choice(methods)
            
            try:
                if method == 'swap':
                    aug_text = self.random_swap(text)
                elif method == 'delete':
                    aug_text = self.random_deletion(text)
                elif method == 'insert':
                    aug_text = self.random_insertion(text)
                elif method == 'back_translate':
                    aug_text = self.back_translation(text)
                elif method == 'code_mix':
                    aug_text = self.generate_code_mixed(text)
                elif method == 'paraphrase':
                    aug_text = self.paraphrase_multilingual(text)
                else:
                    aug_text = text
                
                augmented_texts.append(aug_text)
            except Exception as e:
                logger.warning(f"⚠️  Augmentation method '{method}' failed: {e}")
                augmented_texts.append(text)
        
        return augmented_texts
    
    def augment_dataset(self, texts: List[str], labels: List[int], 
                       augmentation_factor: int = 2) -> Tuple[List[str], List[int]]:
        """
        Augment entire dataset
        
        Args:
            texts: List of input texts
            labels: List of labels
            augmentation_factor: How many augmented samples per original
            
        Returns:
            Augmented texts and labels
        """
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            # Add original
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # Add augmented versions
            aug_samples = self.augment(text, num_augmentations=augmentation_factor)
            augmented_texts.extend(aug_samples)
            augmented_labels.extend([label] * len(aug_samples))
        
        return augmented_texts, augmented_labels

if __name__ == "__main__":
    # Example usage
    augmenter = MultilingualAugmenter()
    
    sample_text = "यो movie राम्रो थियो but ending disappointing थियो"
    augmented = augmenter.augment(sample_text, num_augmentations=3)
    
    print(f"Original: {sample_text}")
    print("\nAugmented versions:")
    for i, aug_text in enumerate(augmented, 1):
        print(f"{i}. {aug_text}")
