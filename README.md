# Multilingual AI for Low-Resource Languages

## 🎯 Project Overview

This project aims to build robust NLP systems that effectively handle Nepali-English code-mixed text for sentiment analysis and misinformation detection. By leveraging fine-tuned multilingual language models and adapter-based architectures with synthetic data augmentation, we overcome data scarcity challenges in low-resource languages.

## 🚀 Running the Application

### Quick Start (Local)
Use the provided script to start both the API and Frontend:
```bash
./start.sh
```
- **API**: http://localhost:8000
- **Frontend**: http://localhost:8501

### Manual Start
1. **Start API**:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
2. **Start Frontend**:
   ```bash
   streamlit run frontend/demo_app.py
   ```

## ☁️ Deployment

### Procfile
A `Procfile` is included for deployment on platforms like Heroku or Render. It currently starts the API.

### Streamlit Cloud
To deploy the frontend on Streamlit Cloud:
1. Push this repository to GitHub.
2. Connect your repository to Streamlit Cloud.
3. Set the `API_URL` and `API_KEY` in the Streamlit Cloud secrets.

## 🚀 Key Features

- **Multilingual Support**: Handles Nepali, English, and code-mixed text
- **Sentiment Analysis**: Classify sentiment in code-mixed social media content
- **Misinformation Detection**: Identify and flag potential misinformation
- **Translation API**: Translate between Nepali and English
- **Adapter-based Architecture**: Efficient fine-tuning with parameter-efficient methods
- **Synthetic Data Augmentation**: Generate training data to overcome scarcity

## 🏗️ Architecture

- **Base Models**: mBERT, XLM-RoBERTa, or similar multilingual transformers
- **Adapters**: Parameter-efficient fine-tuning layers
- **API**: FastAPI-based REST API for inference
- **Data Pipeline**: Preprocessing and augmentation pipeline

## 📁 Project Structure

```
multilingual-ai-nepali/
├── data/                   # Dataset storage
│   ├── raw/               # Raw data files
│   ├── processed/         # Preprocessed data
│   └── synthetic/         # Augmented data
├── models/                # Trained models and checkpoints
├── src/                   # Source code
│   ├── data/             # Data processing scripts
│   ├── models/           # Model architectures
│   ├── training/         # Training scripts
│   └── api/              # API implementation
├── notebooks/            # Jupyter notebooks for experiments
├── tests/                # Unit tests
├── configs/              # Configuration files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🏋️‍♂️ Model Training & Adapters

- See [`TRAINING.md`](./TRAINING.md) for detailed instructions on model training.
- Supports both standard Hugging Face training and efficient adapter-based fine-tuning.
- Example CLI usage:

```bash
# Standard training
python3 src/training/train_sentiment.py --config configs/sentiment_config.yaml --model_type hf

# Adapter-based training
python3 src/training/train_sentiment.py --config configs/sentiment_config.yaml --model_type adapter
```

- For interactive experimentation, see `notebooks/adapter_training_example.ipynb`.

### 🧪 Multitask Fine-Tuning with Synthetic Data

- Fine-tune on both real and synthetic (augmented) data with a multitask objective:
  - **Masked Language Modeling (MLM)** + **Classification**
  - Logs metrics per language subset
  - Supports experiment tracking with Weights & Biases

**Example CLI:**
```bash
python3 src/training/fine_tune_llm.py \
  --config configs/sentiment_config.yaml \
  --model_type adapter \
  --mlm_weight 1.0 \
  --clf_weight 1.0 \
  --experiment multilingual_mlm_clf
```
- Add a `synthetic_path` entry under `data:` in your config to include augmented data.

**Example config block:**
```yaml
data:
  train_path: "data/processed/sentiment_train.csv"
  synthetic_path: "data/synthetic/sentiment_aug.csv"
  val_path: "data/processed/sentiment_val.csv"
  max_length: 128
```

## 🛠️ Installation

```bash
# Clone the repository
```

## 🌐 API Deployment & Usage

### Run the API server (local)
```bash
uvicorn deploy.api_server:app --host 0.0.0.0 --port 8000
```

### Example requests (with API key)
```bash
# Sentiment
curl -X POST http://localhost:8000/predict_sentiment \
  -H 'x-api-key: supersecretkey' \
  -H 'Content-Type: application/json' \
  -d '{"text": "यो राम्रो छ"}'

# Misinformation
curl -X POST http://localhost:8000/detect_misinformation \
  -H 'x-api-key: supersecretkey' \
  -H 'Content-Type: application/json' \
  -d '{"text": "This is false information."}'

# Translation (auto-detects language)
curl -X POST http://localhost:8000/translate_text \
  -H 'x-api-key: supersecretkey' \
  -H 'Content-Type: application/json' \
  -d '{"text": "यो राम्रो छ", "tgt_lang": "en"}'
```

- All endpoints require the `x-api-key` header (see `.env` or code for the key).
- Rate limit: 10 requests per minute per IP.

### API Documentation
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Docker deployment (optional)
```bash
docker build -t multilingual-api .
docker run -p 8000:8000 multilingual-api
```
git clone https://github.com/yourusername/multilingual-ai-nepali.git
cd multilingual-ai-nepali

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

The project uses:
- Nepali-English code-mixed social media data
- Sentiment-labeled datasets
- Misinformation-labeled news articles
- Synthetic augmented data

## 🎓 Model Training

```bash
# Train sentiment analysis model
python src/training/train_sentiment.py --config configs/sentiment_config.yaml

# Train misinformation detection model
python src/training/train_misinfo.py --config configs/misinfo_config.yaml
```

## 🌐 API Usage

```bash
# Start the API server
python src/api/main.py

# Example API call
curl -X POST "http://localhost:8000/predict/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "यो movie राम्रो थियो but ending disappointing थियो"}'
```

## 📈 Results

Results and model performance metrics will be documented here as the project progresses.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License

## 👥 Authors

- Your Name

## 🙏 Acknowledgments

- Multilingual NLP research community
- Open-source transformer libraries (Hugging Face)
- Low-resource language research initiatives
