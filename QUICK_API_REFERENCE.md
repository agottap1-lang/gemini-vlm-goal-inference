# Quick Reference: API Key Management

## ✅ Current Status

**API Key:** `YOUR_GEMINI_API_KEY` (configured in `.env`)  
**Status:** ✓ Active and working

## 🔧 Common Tasks

### Update API Key

```bash
# Edit .env file
# Change: GEMINI_API_KEY=your-new-key-here

# Verify
python scripts/verify_api_config.py
```

### Test API Key

```bash
python scripts/verify_api_config.py
```

### Write New Script

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# API key loads automatically from .env
from gemini_vlm_eval.client import GeminiClient

client = GeminiClient()
# Ready to use!
```

Or copy `scripts/_template.py`.

### Check Current Key

```bash
# PowerShell
python -c "from gemini_vlm_eval.config import get_api_key; print(get_api_key()[:20] + '...')"
```

### Troubleshooting

**"API key not found":**
- Check `.env` exists in project root
- Verify format: `GEMINI_API_KEY=your-key` (no quotes/spaces)

**"Invalid API key":**
- Get new key: https://aistudio.google.com/app/apikey
- Update `.env` file

## 📁 Important Files

- `.env` - Your API key (DO NOT COMMIT)
- `.env.example` - Template for sharing
- `src/gemini_vlm_eval/config.py` - Config module
- `API_KEY_SETUP.md` - Full documentation

## 🎯 Best Practices

1. ✅ Use `.env` file (current setup)
2. ✅ Never hardcode keys
3. ✅ Never commit `.env`
4. ✅ Use `scripts/_template.py` for new scripts
5. ✅ Run `verify_api_config.py` after updates

---

**Complete Docs:** See `API_KEY_SETUP.md` and `API_KEY_UPDATE_SUMMARY.md`
