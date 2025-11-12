"""
Data augmentation techniques for low-resource multilingual NLP
"""

import random
from typing import List, Tuple
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

class MultilingualAugmenter:
    """Data augmentation for Nepali-English code-mixed text"""
    
    def __init__(self):
        # Initialize augmenters (will be loaded when needed)
        self.back_translation_aug = None
        self.synonym_aug = None
        
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
    
    def back_translation(self, text: str, source_lang: str = 'en', 
                        intermediate_lang: str = 'hi') -> str:
        """
        Back-translation augmentation
        Translate to intermediate language and back
        
        Note: Requires translation models
        """
        # TODO: Implement with actual translation models
        # For now, return original text
        return text
    
    def augment(self, text: str, num_augmentations: int = 1, 
                methods: List[str] = None) -> List[str]:
        """
        Generate augmented versions of text
        
        Args:
            text: Input text
            num_augmentations: Number of augmented samples to generate
            methods: List of augmentation methods to use
            
        Returns:
            List of augmented texts
        """
        if methods is None:
            methods = ['swap', 'delete', 'insert']
        
        augmented_texts = []
        
        for _ in range(num_augmentations):
            method = random.choice(methods)
            
            if method == 'swap':
                aug_text = self.random_swap(text)
            elif method == 'delete':
                aug_text = self.random_deletion(text)
            elif method == 'insert':
                aug_text = self.random_insertion(text)
            elif method == 'back_translate':
                aug_text = self.back_translation(text)
            else:
                aug_text = text
            
            augmented_texts.append(aug_text)
        
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
