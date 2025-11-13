# 📊 Data Pipeline Documentation

This document describes the complete data ingestion and preparation pipeline for the Multilingual AI Nepali project.

## 🎯 Overview

The data pipeline handles the complete workflow from raw data ingestion to tokenized datasets ready for model training. It includes:

1. **Data Ingestion** - Loading data from various sources
2. **Data Validation** - Ensuring data quality
3. **Data Preprocessing** - Cleaning and normalizing text
4. **Data Augmentation** - Generating synthetic samples
5. **Tokenization** - Converting text to model inputs

## 🏗️ Architecture

```
Raw Data → Ingestion → Validation → Preprocessing → Augmentation → Tokenization → Training Data
```

## 📁 Module Overview

### 1. `src/data/ingestion.py`

**Purpose**: Load multilingual datasets from various sources

**Features**:
- Load CSV, JSON, JSONL files
- Load datasets from Hugging Face Hub
- Scrape Nepali news portals (placeholder)
- Fetch Twitter data via API (placeholder)
- Combine multiple datasets
- Remove duplicates

**Example Usage**:
```python
from src.data.ingestion import DataIngestion

ingestion = DataIngestion(data_dir="data/raw")

# Load CSV file
df = ingestion.load_csv("data/raw/sentiment_data.csv")

# Load from Hugging Face
df = ingestion.load_huggingface_dataset("dataset_name", split="train")

# Combine datasets
combined = ingestion.combine_datasets([df1, df2, df3])

# Save dataset
ingestion.save_dataset(combined, "combined_data.csv")
```

### 2. `src/data/preprocess.py`

**Purpose**: Clean and normalize Nepali-English code-mixed text

**Features**:
- **Unicode Normalization**: Handle Devanagari script variations
- **URL Removal**: Remove HTTP/HTTPS and domain URLs
- **Emoji Removal**: Clean emoji characters
- **Special Character Handling**: Remove or preserve language-specific characters
- **Transliteration Normalization**: Handle common transliteration inconsistencies
- **Whitespace Normalization**: Clean various unicode spaces
- **Language Detection**: Identify Nepali, English, or mixed text

**Example Usage**:
```python
from src.data.preprocess import TextPreprocessor

preprocessor = TextPreprocessor()

# Preprocess single text
text = "यो movie राम्रो थियो but ending disappointing थियो 😊"
processed = preprocessor.preprocess(
    text,
    remove_urls=True,
    remove_emojis=True,
    normalize_transliteration=True
)

# Preprocess entire dataset
df_processed = preprocessor.preprocess_dataset(df, text_column='text')

# Detect language
language = preprocessor.detect_language_mix(text)  # Returns: 'mixed'
```

### 3. `src/data/augmentation.py`

**Purpose**: Generate synthetic training data for low-resource scenarios

**Features**:
- **Random Swap**: Swap word positions
- **Random Deletion**: Delete words randomly
- **Random Insertion**: Insert duplicate words
- **Back-Translation**: Translate to intermediate language and back (English ↔ Hindi ↔ English)
- **Code-Mixed Generation**: Generate code-mixed text by translating random words
- **Multilingual Paraphrasing**: Generate paraphrases (placeholder)

**Example Usage**:
```python
from src.data.augmentation import MultilingualAugmenter

augmenter = MultilingualAugmenter(use_gpu=False)

# Simple augmentation
text = "यो movie राम्रो थियो"
augmented = augmenter.augment(text, num_augmentations=3)

# Advanced augmentation with back-translation
augmented = augmenter.augment(
    text, 
    num_augmentations=2,
    use_advanced=True
)

# Generate code-mixed text
code_mixed = augmenter.generate_code_mixed(
    "This is a good movie",
    target_lang='ne',
    mix_ratio=0.3
)

# Augment entire dataset
texts = ["text1", "text2", "text3"]
labels = [0, 1, 2]
aug_texts, aug_labels = augmenter.augment_dataset(
    texts, labels, augmentation_factor=2
)
```

### 4. `src/data/tokenizer_utils.py`

**Purpose**: Tokenize text using multilingual models

**Supported Models**:
- XLM-RoBERTa (base, large)
- IndicBERT
- MuRIL
- mT5 (small, base)
- mBERT

**Example Usage**:
```python
from src.data.tokenizer_utils import MultilingualTokenizer

# Initialize tokenizer
tokenizer = MultilingualTokenizer('xlm-roberta-base')

# Tokenize single text
encoded = tokenizer.tokenize(
    "यो movie राम्रो थियो",
    max_length=128,
    padding=True,
    truncation=True,
    return_tensors='pt'
)

# Batch tokenization
texts = ["text1", "text2", "text3"]
batch_encoded = tokenizer.batch_encode(texts, max_length=128)

# Analyze tokenization
analysis = tokenizer.analyze_text("यो movie राम्रो थियो")
print(f"Tokens: {analysis['tokens']}")
print(f"Num tokens: {analysis['num_tokens']}")

# Compare tokenizers
from src.data.tokenizer_utils import compare_tokenizers
comparison = compare_tokenizers(
    "यो movie राम्रो थियो",
    model_names=['xlm-roberta-base', 'indicbert', 'mbert']
)
```

### 5. `src/data/validation.py`

**Purpose**: Validate data quality based on Windsurf configuration

**Features**:
- Column existence checks
- Null value detection
- Value range validation
- Text length validation
- Data quality metrics
- Configurable validation rules

**Example Usage**:
```python
from src.data.validation import DataValidator

validator = DataValidator(config_path="windsurf.json")

# Validate DataFrame
is_valid, errors = validator.validate_dataframe(df, "my_dataset")

# Validate file
is_valid, errors = validator.validate_file("data/raw/sentiment.csv")

# Get validation summary
summary = validator.get_validation_summary(df)

# Validate entire directory
from src.data.validation import validate_dataset_directory
results = validate_dataset_directory("data/raw", validator)
```

### 6. `src/data/pipeline.py`

**Purpose**: Integrate all components into a complete pipeline

**Example Usage**:
```python
from src.data.pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline(config_path="windsurf.json")

# Run full pipeline
tokenized_data, processed_df = pipeline.run_full_pipeline(
    input_path="data/raw/sentiment_data.csv",
    output_dir="data/processed",
    task_type="sentiment",
    tokenizer_name="xlm-roberta-base"
)

# Run with sample data
tokenized_data, processed_df = pipeline.create_sample_pipeline()

# Get pipeline statistics
stats = pipeline.get_pipeline_stats()
```

## ⚙️ Configuration (windsurf.json)

The pipeline is configured via `windsurf.json`:

```json
{
  "data": {
    "validation": {
      "enabled": true,
      "rules": [...]
    },
    "ingestion": {
      "preprocessing": {
        "enabled": true,
        "steps": ["normalize_unicode", "remove_urls", ...]
      },
      "augmentation": {
        "enabled": true,
        "methods": ["swap", "delete", "insert"],
        "augmentation_factor": 2
      }
    }
  }
}
```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline

```bash
# Run the integrated pipeline
python src/data/pipeline.py
```

### Option 2: Step-by-Step

```python
# 1. Load data
from src.data.ingestion import DataIngestion
ingestion = DataIngestion()
df = ingestion.load_csv("data/raw/sentiment.csv")

# 2. Validate
from src.data.validation import DataValidator
validator = DataValidator()
is_valid, errors = validator.validate_dataframe(df)

# 3. Preprocess
from src.data.preprocess import TextPreprocessor
preprocessor = TextPreprocessor()
df = preprocessor.preprocess_dataset(df)

# 4. Augment
from src.data.augmentation import MultilingualAugmenter
augmenter = MultilingualAugmenter()
texts, labels = augmenter.augment_dataset(
    df['processed_text'].tolist(),
    df['label'].tolist()
)

# 5. Tokenize
from src.data.tokenizer_utils import MultilingualTokenizer
tokenizer = MultilingualTokenizer('xlm-roberta-base')
encoded = tokenizer.batch_encode(texts)
```

## 📊 Data Format

### Input Format

CSV/JSON files should have at minimum:
- `text`: Raw text content
- `label`: Classification label (for supervised tasks)

Example:
```csv
text,label
"यो movie राम्रो थियो but ending disappointing थियो",1
"I love Nepali खाना especially momo",2
```

### Output Format

Processed data includes:
- `text`: Original text
- `processed_text`: Cleaned text
- `label`: Label
- `language`: Detected language (nepali/english/mixed)
- `text_length`: Length of processed text

## 🔧 Advanced Features

### Back-Translation

```python
augmenter = MultilingualAugmenter()
back_translated = augmenter.back_translation(
    "This is a good movie",
    source_lang='en',
    intermediate_lang='hi'
)
```

### Code-Mixed Text Generation

```python
code_mixed = augmenter.generate_code_mixed(
    "This movie was really good",
    target_lang='ne',
    mix_ratio=0.4  # 40% of words translated
)
```

### Custom Validation Rules

Edit `windsurf.json` to add custom validation rules:

```json
{
  "data": {
    "validation": {
      "rules": [
        {
          "name": "custom_rule",
          "type": "value_range",
          "column": "label",
          "min_value": 0,
          "max_value": 2
        }
      ]
    }
  }
}
```

## 🐛 Troubleshooting

### Issue: Transformers models not downloading

**Solution**: Check Hugging Face token in `.env` file

### Issue: Back-translation is slow

**Solution**: 
- Use GPU: `augmenter = MultilingualAugmenter(use_gpu=True)`
- Use simpler augmentation methods for faster processing

### Issue: Unicode errors

**Solution**: Ensure files are UTF-8 encoded

## 📚 Additional Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [XLM-RoBERTa Paper](https://arxiv.org/abs/1911.02116)
- [IndicBERT](https://huggingface.co/ai4bharat/indic-bert)
- [Data Augmentation for NLP](https://arxiv.org/abs/1901.11196)

## 🤝 Contributing

To add new data sources or augmentation methods:

1. Add implementation to respective module
2. Update `windsurf.json` configuration
3. Add tests in `tests/` directory
4. Update this documentation

---

**Last Updated**: 2025-11-13
