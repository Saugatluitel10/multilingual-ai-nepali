 """
Hugging Face configuration and authentication utilities
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, whoami
from huggingface_hub.utils import HfHubHTTPError

# Load environment variables
load_dotenv()


def get_hf_token() -> Optional[str]:
    """
    Get Hugging Face token from environment variables
    
    Returns:
        Token string or None if not found
    """
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token or token == 'your_huggingface_token_here':
        return None
    return token


def check_hf_token() -> bool:
    """
    Check if Hugging Face token is valid
    
    Returns:
        True if token is valid, False otherwise
    """
    token = get_hf_token()
    if not token:
        print("❌ No Hugging Face token found in .env file")
        print("   Please add your token to the .env file")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        return False
    
    try:
        # Try to get user info to validate token
        user_info = whoami(token=token)
        print(f"✅ Hugging Face token is valid!")
        print(f"   Logged in as: {user_info['name']}")
        return True
    except HfHubHTTPError as e:
        print(f"❌ Invalid Hugging Face token: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking token: {e}")
        return False


def setup_huggingface(auto_login: bool = True) -> bool:
    """
    Setup Hugging Face authentication
    
    Args:
        auto_login: Whether to automatically login with token
        
    Returns:
        True if setup successful, False otherwise
    """
    token = get_hf_token()
    
    if not token:
        print("\n⚠️  Hugging Face token not configured")
        print("\nTo use Hugging Face models and datasets:")
        print("1. Get your token from: https://huggingface.co/settings/tokens")
        print("2. Add it to your .env file: HUGGINGFACE_TOKEN=your_token_here")
        print("3. Or run: huggingface-cli login")
        return False
    
    if auto_login:
        try:
            login(token=token, add_to_git_credential=True)
            print("✅ Successfully logged in to Hugging Face!")
            return True
        except Exception as e:
            print(f"❌ Failed to login to Hugging Face: {e}")
            return False
    
    return True


def get_model_cache_dir() -> Path:
    """
    Get the model cache directory from environment or use default
    
    Returns:
        Path to model cache directory
    """
    cache_dir = os.getenv('MODEL_CACHE_DIR', './models/cache')
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def list_available_models(filter_str: Optional[str] = None) -> list:
    """
    List available models from Hugging Face
    
    Args:
        filter_str: Optional filter string for model names
        
    Returns:
        List of model names
    """
    try:
        api = HfApi()
        models = api.list_models(
            filter=filter_str,
            sort="downloads",
            direction=-1,
            limit=10
        )
        return [model.modelId for model in models]
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []


if __name__ == "__main__":
    print("🔧 Hugging Face Configuration Check\n")
    
    # Check token
    token_valid = check_hf_token()
    
    if token_valid:
        print("\n📦 Model cache directory:", get_model_cache_dir())
        
        print("\n🤖 Top multilingual models:")
        models = list_available_models("xlm")
        for i, model in enumerate(models[:5], 1):
            print(f"   {i}. {model}")


