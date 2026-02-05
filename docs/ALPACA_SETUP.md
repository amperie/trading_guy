# Alpaca API Setup Guide

## Quick Setup (3 steps)

### 1. Get Your API Credentials

1. Go to [alpaca.markets](https://alpaca.markets) and sign up for a free account
2. Navigate to **Paper Trading** section
3. Generate API keys (they'll start with `PK` for key and `PS` for secret)

### 2. Create `.env` File

Copy the example file and add your credentials:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your credentials
# (or just create .env manually with the content below)
```

Your `.env` file should look like this:

```bash
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=PSxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Important:**
- Use **PAPER** trading credentials (start with PK/PS), NOT live trading keys
- The `.env` file is already in `.gitignore` so it won't be committed to git
- Never share or commit your API keys

### 3. Run Integration Tests

```bash
# The tests will automatically load credentials from .env
pytest tests/integration/test_alpaca_om_integration.py -v -m integration
```

That's it! The environment variables will be loaded automatically.

---

## How It Works

Python's `os.getenv()` automatically reads from environment variables, which includes:
1. System environment variables
2. Variables from `.env` files (when using python-dotenv or similar)
3. Variables set in your shell session

The integration tests use:
```python
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
```

---

## Alternative Methods

### Method 1: Shell Export (Temporary - Current Session Only)

**Windows (Command Prompt):**
```cmd
set ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxx
set ALPACA_SECRET_KEY=PSxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Windows (PowerShell):**
```powershell
$env:ALPACA_API_KEY="PKxxxxxxxxxxxxxxxxxx"
$env:ALPACA_SECRET_KEY="PSxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Mac/Linux (Bash/Zsh):**
```bash
export ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxx
export ALPACA_SECRET_KEY=PSxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Downside:** Variables disappear when you close the terminal.

---

### Method 2: System Environment Variables (Permanent)

**Windows:**
1. Search for "Environment Variables" in Start Menu
2. Click "Edit the system environment variables"
3. Click "Environment Variables" button
4. Under "User variables", click "New"
5. Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`

**Mac/Linux:**
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
export ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxx
export ALPACA_SECRET_KEY=PSxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Then run: `source ~/.bashrc` (or restart terminal)

**Downside:** Variables are visible to all applications.

---

## Recommended: Use `.env` File ✓

The `.env` file method is best because:
- ✅ Project-specific (doesn't affect other projects)
- ✅ Easy to switch between different credentials
- ✅ Already gitignored (secure)
- ✅ Works across all platforms
- ✅ Standard practice in Python projects

---

## Verify Setup

Run this to check if your credentials are loaded:

```bash
python -c "import os; print('API Key:', 'SET' if os.getenv('ALPACA_API_KEY') else 'NOT SET'); print('Secret:', 'SET' if os.getenv('ALPACA_SECRET_KEY') else 'NOT SET')"
```

Should output:
```
API Key: SET
Secret: SET
```

---

## Security Best Practices

1. ✅ **DO** use paper trading credentials for testing
2. ✅ **DO** keep `.env` file in `.gitignore`
3. ✅ **DO** regenerate keys if accidentally exposed
4. ❌ **DON'T** commit API keys to git
5. ❌ **DON'T** share your `.env` file
6. ❌ **DON'T** use live trading keys for automated tests

---

## Troubleshooting

**Problem:** Tests skip with "Alpaca API credentials not found"

**Solution:**
1. Make sure `.env` file exists in project root
2. Check variable names match exactly: `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`
3. No quotes needed in `.env` file: `ALPACA_API_KEY=PKxxxxx` (not `"PKxxxxx"`)
4. Restart your terminal/IDE after creating `.env`

**Problem:** Tests fail with "401 Unauthorized"

**Solution:**
1. Verify you're using **paper** trading credentials (start with PK/PS)
2. Regenerate API keys in Alpaca dashboard
3. Check for trailing spaces in `.env` file
