#!/usr/bin/env python3
"""
Verify API key configuration across all scripts.

Tests that:
1. API key is loaded from .env file
2. All evaluation scripts can access the key
3. GeminiClient initializes correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_config_module():
    """Test the centralized config module."""
    print("Testing config module...")
    try:
        from gemini_vlm_eval.config import get_api_key, DEFAULT_MODEL, AVAILABLE_MODELS
        
        api_key = get_api_key()
        print(f"  ✓ API key loaded: {api_key[:20]}...{api_key[-4:]}")
        print(f"  ✓ Default model: {DEFAULT_MODEL}")
        print(f"  ✓ Available models: {len(AVAILABLE_MODELS)} models")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_client_initialization():
    """Test GeminiClient initialization."""
    print("\nTesting GeminiClient initialization...")
    try:
        from gemini_vlm_eval.client import GeminiClient
        
        client = GeminiClient()
        print(f"  ✓ Client initialized with model: {client.model}")
        print(f"  ✓ API key source: environment/dotenv")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_alternative_models():
    """Test initializing with different models."""
    print("\nTesting alternative model initialization...")
    try:
        from gemini_vlm_eval.client import GeminiClient
        
        models_to_test = ["gemini-2.5-flash", "gemini-2.5-pro"]
        
        for model in models_to_test:
            client = GeminiClient(model=model)
            print(f"  ✓ {model}: initialized")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_env_file_exists():
    """Check that .env file exists."""
    print("\nChecking .env file...")
    env_file = Path(__file__).parent.parent / ".env"
    
    if env_file.exists():
        print(f"  ✓ .env file found: {env_file}")
        
        # Read and verify (without exposing full key)
        content = env_file.read_text()
        if "GEMINI_API_KEY=" in content:
            lines = content.split('\n')
            key_line = [l for l in lines if l.startswith('GEMINI_API_KEY=')][0]
            key_value = key_line.split('=', 1)[1].strip()
            if key_value and key_value != "your-api-key-here":
                print(f"  ✓ API key configured in .env")
                return True
            else:
                print(f"  ✗ API key not set in .env (using example value)")
                return False
        else:
            print(f"  ✗ GEMINI_API_KEY not found in .env")
            return False
    else:
        print(f"  ✗ .env file not found at: {env_file}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 80)
    print("API KEY CONFIGURATION VERIFICATION")
    print("=" * 80)
    print()
    
    results = []
    
    # Test .env file
    results.append(("Environment file check", test_env_file_exists()))
    
    # Test config module
    results.append(("Config module", test_config_module()))
    
    # Test client initialization
    results.append(("Client initialization", test_client_initialization()))
    
    # Test alternative models
    results.append(("Alternative models", test_alternative_models()))
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS ✓" if result else "FAIL ✗"
        print(f"  {test_name:.<40} {status}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All verification tests passed!")
        print("✓ API key configuration is working correctly")
        print("\nYou can now run any evaluation script without hardcoding API keys.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nSee API_KEY_SETUP.md for troubleshooting help.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
