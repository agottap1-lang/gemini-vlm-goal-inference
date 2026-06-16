# API Key Setup Guide

This document explains how to properly configure your Google Gemini API key for the VLM evaluation framework.

## ✅ Recommended Method: Use .env File (Current Setup)

The `.env` file has been configured with your API key and is automatically loaded by all scripts.

**Your current API key:** `YOUR_GEMINI_API_KEY`

### How It Works

1. **Automatic Loading**: The `.env` file in the project root is automatically loaded by the `dotenv` library
2. **Centralized Configuration**: All scripts access the API key through environment variables
3. **Secure**: The `.env` file is excluded from Git via `.gitignore` (never commit API keys!)

### File Locations
- **`.env`** - Your actual API key (already configured ✓)
- **`.env.example`** - Template for other users
- **`src/gemini_vlm_eval/config.py`** - Centralized configuration module
- **`src/gemini_vlm_eval/client.py`** - GeminiClient automatically loads from environment

## 🔧 Alternative Methods

### Method 2: Export Environment Variable

**Linux/Mac (Bash/Zsh):**
```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

### Method 3: Pass Directly to GeminiClient

```python
from gemini_vlm_eval.client import GeminiClient

client = GeminiClient(api_key="YOUR_GEMINI_API_KEY")
```

⚠️ **Not recommended**: Hardcoding API keys in scripts makes them visible in version control and harder to rotate.

## 📝 Best Practices for New Scripts

When writing new evaluation scripts, use this pattern:

```python
#!/usr/bin/env python3
"""Your script description."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# API key is automatically loaded from .env file
from gemini_vlm_eval.client import GeminiClient
from gemini_vlm_eval.config import get_api_key, DEFAULT_MODEL

# Use GeminiClient - it loads API key automatically
client = GeminiClient(model=DEFAULT_MODEL)

# Or verify API key is available:
try:
    api_key = get_api_key()
    print("✓ API key loaded successfully")
except ValueError as e:
    print(f"✗ {e}")
    sys.exit(1)
```

See `scripts/_template.py` for a complete template.

## 🔄 Updating Your API Key

If you need to use a different API key:

1. **Edit `.env` file:**
   ```bash
   # In .env file
   GEMINI_API_KEY=your-new-api-key-here
   ```

2. **Or update environment variable:**
   ```bash
   export GEMINI_API_KEY="your-new-api-key-here"
   ```

No code changes needed! The key is loaded automatically.

## ✅ Verification

Test that your API key is working:

```bash
python scripts/test_api_key.py
```

Or test with a quick inference:

```python
from gemini_vlm_eval.client import GeminiClient

client = GeminiClient()
print("✓ API key loaded and client initialized successfully!")
```

## 🗂️ Updated Scripts

The following scripts have been updated to use the `.env` file instead of hardcoded API keys:

- ✅ `scripts/compare_pro_vs_flash.py`
- ✅ `scripts/multi_model_evaluation.py`
- ✅ `scripts/test_inference_le_l_block.py`

All other scripts in the codebase already use environment variables correctly.

## 🔒 Security Notes

1. **Never commit `.env` files** - Already in `.gitignore` ✓
2. **Rotate keys regularly** - Get new keys at [Google AI Studio](https://aistudio.google.com/app/apikey)
3. **Limit key permissions** - Use API key restrictions in Google Cloud Console
4. **Monitor usage** - Check your API usage in Google Cloud Console

## 🆘 Troubleshooting

**Error: "GEMINI_API_KEY not found in environment"**
- Check that `.env` file exists in project root
- Verify the file contains `GEMINI_API_KEY=your-key` (no quotes, no spaces around `=`)
- Try running: `python -c "from gemini_vlm_eval.config import get_api_key; print(get_api_key())"`

**Error: "Invalid API key"**
- Verify your key at [Google AI Studio](https://aistudio.google.com/app/apikey)
- Check that Gemini API is enabled in your Google Cloud project
- Try the key in a simple test: `python scripts/test_api_key.py`

**Key works in terminal but not in scripts**
- Make sure `.env` file is in the project root (same directory as `pyproject.toml`)
- Check file permissions - `.env` should be readable
- Verify no typos in `GEMINI_API_KEY` (case-sensitive)

---

**Current Status:** ✅ Your API key has been updated and all scripts are configured to use it automatically from the `.env` file.
