# 🚀 Environment Setup Guide

This guide will help you set up the development environment for the Multilingual AI Nepali project using Windsurf.

## Prerequisites

- **Conda** (Miniconda or Anaconda)
  - Download from: https://docs.conda.io/en/latest/miniconda.html
- **Git** (for version control)
- **Hugging Face Account** (for model access)
  - Sign up at: https://huggingface.co/join

## Quick Start

### Option 1: Automated Setup (Recommended)

Run the automated setup script:

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

This will:
- Create the conda environment with all dependencies
- Set up directory structure
- Create `.env` file from template

### Option 2: Manual Setup

#### Step 1: Create Conda Environment

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate multilingual-ai-nepali
```

#### Step 2: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Hugging Face token
nano .env  # or use your preferred editor
```

Get your Hugging Face token from: https://huggingface.co/settings/tokens

#### Step 3: Login to Hugging Face (Optional)

For uploading models and accessing gated models:

```bash
huggingface-cli login
```

#### Step 4: Verify Installation

```bash
# Test imports
python -c "import torch; import transformers; print('✅ All imports successful!')"

# Check Hugging Face configuration
python src/utils/hf_config.py
```

## Environment Details

### Core Dependencies

- **Python**: 3.10
- **PyTorch**: 2.0.0+
- **Transformers**: 4.35.0+ (Hugging Face)
- **Datasets**: 2.14.0+ (Hugging Face)
- **FastAPI**: 0.104.0+ (API framework)
- **Streamlit**: 1.28.0+ (UI framework)
- **Sentencepiece**: 0.1.99+ (Tokenization)

### Optional Dependencies

- **Weights & Biases**: For experiment tracking
- **TensorBoard**: For training visualization
- **CUDA**: For GPU acceleration (if available)

## GPU Support

If you have a GPU, modify `environment.yml`:

1. Remove the line: `- cpuonly`
2. Add CUDA toolkit:
   ```yaml
   - pytorch-cuda=11.8  # or your CUDA version
   ```

Then recreate the environment:

```bash
conda env remove -n multilingual-ai-nepali
conda env create -f environment.yml
```

## Hugging Face Configuration

### Getting Your Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "multilingual-ai-nepali")
4. Select "read" permissions (or "write" if you plan to upload models)
5. Copy the token

### Adding Token to .env

Edit your `.env` file:

```bash
HUGGINGFACE_TOKEN=hf_your_actual_token_here
```

### Verify Token

```bash
python -c "from src.utils.hf_config import check_hf_token; check_hf_token()"
```

## Project Structure

After setup, your directory structure should look like:

```
multilingual-ai-nepali/
├── data/
│   ├── raw/              # Raw datasets
│   ├── processed/        # Preprocessed data
│   └── synthetic/        # Augmented data
├── models/
│   ├── sentiment/        # Sentiment models
│   ├── misinfo/          # Misinformation models
│   └── cache/            # HF model cache
├── logs/                 # Training logs
├── src/                  # Source code
├── tests/                # Unit tests
├── environment.yml       # Conda environment
├── .env                  # Environment variables (not in git)
└── setup_environment.sh  # Setup script
```

## Common Issues

### Issue: Conda command not found

**Solution**: Install Miniconda or Anaconda, then restart your terminal.

### Issue: Token not working

**Solution**: 
1. Check that token is correctly copied (no extra spaces)
2. Verify token has correct permissions on Hugging Face
3. Try logging in manually: `huggingface-cli login`

### Issue: Import errors

**Solution**:
```bash
# Ensure environment is activated
conda activate multilingual-ai-nepali

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: CUDA/GPU not detected

**Solution**:
```bash
# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If False, reinstall PyTorch with CUDA support
```

## Next Steps

After successful setup:

1. **Explore the codebase**: Check out `src/` directory
2. **Run data preprocessing**: `python src/data/preprocess.py`
3. **Start the API**: `python src/api/main.py`
4. **Train a model**: `python src/training/train_sentiment.py --config configs/sentiment_config.yaml`
5. **Run tests**: `pytest tests/`

## Additional Resources

- **Hugging Face Documentation**: https://huggingface.co/docs
- **Transformers Library**: https://huggingface.co/docs/transformers
- **FastAPI Documentation**: https://fastapi.tiangolo.com
- **Streamlit Documentation**: https://docs.streamlit.io

## Support

For issues or questions:
- Open an issue on GitHub: https://github.com/Saugatluitel10/multilingual-ai-nepali/issues
- Check existing documentation in the `docs/` folder

---

**Happy Coding! 🎉**
