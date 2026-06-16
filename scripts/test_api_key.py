#!/usr/bin/env python3
"""
Test Gemini API key and connectivity.
"""

import os
import sys

def test_api_key():
    """Test if Gemini API key is set and working."""
    
    print("=" * 80)
    print("TESTING GEMINI API KEY")
    print("=" * 80)
    print()
    
    # Step 1: Check if API key is set
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ FAIL: GEMINI_API_KEY environment variable is NOT set")
        print()
        print("To set it:")
        print("  PowerShell: $env:GEMINI_API_KEY='your-key-here'")
        print("  Or add to .env file")
        return False
    
    print(f"✓ API key is set")
    print(f"  Length: {len(api_key)} characters")
    print(f"  Prefix: {api_key[:10]}...")
    print()
    
    # Step 2: Test API connectivity
    print("Testing API connectivity...")
    print()
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        
        # Try to list models
        print("  → Fetching available models...")
        models = list(genai.list_models())
        print(f"  ✓ Successfully connected! Found {len(models)} models")
        print()
        
        # Show relevant models
        flash_models = [m.name for m in models if 'flash' in m.name.lower()]
        if flash_models:
            print(f"  Available Flash models:")
            for model_name in flash_models[:5]:
                print(f"    • {model_name}")
        print()
        
        # Step 3: Test actual generation
        print("Testing text generation...")
        model_name = 'gemini-2.5-flash' if any('2.5-flash' in m.name for m in models) else 'gemini-2.0-flash'
        print(f"  Using model: {model_name}")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Reply with exactly: API TEST SUCCESS")
        
        result_text = response.text.strip()
        print(f"  ✓ Generation successful!")
        print(f"  Response: {result_text[:100]}")
        print()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED - API key is working correctly")
        print("=" * 80)
        return True
        
    except ImportError as e:
        print(f"❌ FAIL: Missing required package")
        print(f"  Error: {e}")
        print()
        print("Install with: pip install google-generativeai")
        return False
        
    except Exception as e:
        print(f"❌ FAIL: API test failed")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        print()
        
        if "API key not valid" in str(e) or "INVALID_ARGUMENT" in str(e):
            print("⚠️  This looks like an invalid API key.")
            print("   Check: https://makersuite.google.com/app/apikey")
        elif "quota" in str(e).lower():
            print("⚠️  This looks like a quota/rate limit issue.")
        
        return False

if __name__ == "__main__":
    success = test_api_key()
    sys.exit(0 if success else 1)
