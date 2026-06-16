#!/usr/bin/env python3
"""
Template for new evaluation scripts.

This template demonstrates best practices for:
- API key management (using .env file)
- Importing from gemini_vlm_eval package
- Error handling and logging
- Result saving and provenance tracking
"""

import sys
from pathlib import Path
from typing import Optional

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import from package
from gemini_vlm_eval.client import GeminiClient
from gemini_vlm_eval.config import get_api_key, DEFAULT_MODEL
from gemini_vlm_eval.schema import ManifestEntry
from gemini_vlm_eval.video import extract_frames

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    
    # API key is automatically loaded from .env file
    # Verify it's available (will raise ValueError if not found)
    try:
        api_key = get_api_key()
        logger.info("✓ API key loaded successfully from environment")
    except ValueError as e:
        logger.error(f"✗ {e}")
        return 1
    
    # Initialize Gemini client
    client = GeminiClient(model=DEFAULT_MODEL)
    logger.info(f"✓ Initialized Gemini client with model: {DEFAULT_MODEL}")
    
    # Your evaluation logic here
    # ...
    
    logger.info("✓ Evaluation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
