import os
from dotenv import load_dotenv
from pathlib import Path

def load_environment():
    """Load environment variables from .env file."""
    # Find .env file (look in current dir and parent dirs)
    env_path = Path('.env')
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / '.env'
    
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from {env_path}")
    else:
        print("⚠️ No .env file found")

def get_api_key(service_name: str, required: bool = True) -> str:
    """Get API key for a service."""
    key_mapping = {
        'anthropic': 'ANTHROPIC_API_KEY',
        'openai': 'OPENAI_API_KEY', 
        'huggingface': 'HUGGINGFACE_TOKEN',
        'wandb': 'WANDB_API_KEY'
    }
    
    env_var = key_mapping.get(service_name.lower())
    if not env_var:
        raise ValueError(f"Unknown service: {service_name}")
    
    api_key = os.getenv(env_var)
    
    if required and not api_key:
        raise ValueError(f"Missing required API key: {env_var}")
    
    return api_key

# Load environment variables when module is imported
load_environment() 