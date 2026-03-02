#!/bin/bash
# Switch to Python 3.11 venv

echo "Switching to Python 3.11 environment..."

# 1. Deactivate current venv (if any)
deactivate 2>/dev/null

# 2. Backup old venv if exists
if [ -d "venv" ]; then
    echo "Backing up old venv to venv-old..."
    mv venv venv-old
fi

# 3. Rename new venv to standard name 'venv'
if [ -d "venv-py311" ]; then
    echo "Activating new venv..."
    mv venv-py311 venv
else
    echo "Error: venv-py311 not found!"
    exit 1
fi

echo "✅ Success! Python 3.11 environment is now active."
echo "Please run: source venv/bin/activate"
