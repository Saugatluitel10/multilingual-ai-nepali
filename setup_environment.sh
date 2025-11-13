#!/bin/bash

# Multilingual AI Nepali - Environment Setup Script
# This script sets up the conda environment and configures Hugging Face

echo "🚀 Setting up Multilingual AI Nepali Environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment from environment.yml
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate the environment
echo "✅ Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate multilingual-ai-nepali"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your Hugging Face token"
    echo "   Get your token from: https://huggingface.co/settings/tokens"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed data/synthetic
mkdir -p models/sentiment models/misinfo models/cache
mkdir -p logs/sentiment logs/misinfo
mkdir -p tests

echo ""
echo "✨ Setup complete! Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   conda activate multilingual-ai-nepali"
echo ""
echo "2. Configure your Hugging Face token:"
echo "   - Edit the .env file"
echo "   - Add your token from https://huggingface.co/settings/tokens"
echo ""
echo "3. Login to Hugging Face (optional, for model uploads):"
echo "   huggingface-cli login"
echo ""
echo "4. Test the installation:"
echo "   python -c 'import torch; import transformers; print(\"✅ All imports successful!\")'"
echo ""
echo "Happy coding! 🎉"
