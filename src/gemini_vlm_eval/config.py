"""
Configuration management for Gemini VLM Evaluation.

Handles API key loading from environment variables with fallback to .env file.
Provides centralized configuration for all scripts.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
_project_root = Path(__file__).parent.parent.parent
_env_file = _project_root / ".env"

if _env_file.exists():
    load_dotenv(_env_file)
else:
    load_dotenv()  # Try to load from current directory or parent directories


def get_api_key() -> str:
    """
    Get Google Gemini API key from environment.
    
    Returns:
        str: The API key
        
    Raises:
        ValueError: If API key is not found in environment
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment.\n"
            "Please set it in one of these ways:\n"
            "  1. Create a .env file in the project root with: GEMINI_API_KEY=your-key\n"
            "  2. Export it: export GEMINI_API_KEY=your-key (Linux/Mac)\n"
            "  3. Set it: $env:GEMINI_API_KEY='your-key' (PowerShell)\n"
            f"\nLooking for .env at: {_env_file}"
        )
    return api_key


# Default model configuration
DEFAULT_MODEL = "gemini-2.5-flash"

# Available models for evaluation
AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini-pro-latest",
]
