#!/bin/bash
cd "$(dirname "$0")"

echo "============================================================"
echo "  ION ORCHARD PIPELINE - MAC SETUP"
echo "  Run this once after downloading."
echo "============================================================"
echo ""

# ── Find Python ───────────────────────────────────────────────
PYTHON=""
for candidate in \
    "$HOME/anaconda3/bin/python3" \
    "$HOME/miniconda3/bin/python3" \
    "$HOME/opt/anaconda3/bin/python3" \
    "/opt/homebrew/bin/python3" \
    "/usr/local/bin/python3" \
    "$(which python3 2>/dev/null)"
do
    if [ -f "$candidate" ] 2>/dev/null; then
        VERSION=$("$candidate" -c \
            "import sys; print(sys.version_info >= (3,8))" 2>/dev/null)
        if [ "$VERSION" = "True" ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ Python 3.8+ not found."
    echo ""
    echo "Please install from: https://www.python.org/downloads/"
    echo "Then double-click this file again."
    echo ""
    open "https://www.python.org/downloads/"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✅ Found Python: $PYTHON ($($PYTHON --version))"
echo ""

# ── Create virtual environment ────────────────────────────────
echo "Step 1/3: Creating virtual environment..."

[ -d "venv" ] && rm -rf venv

"$PYTHON" -m venv venv

if [ ! -f "venv/bin/python" ]; then
    echo "❌ Failed to create virtual environment."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "   Done."
echo ""

# ── Install packages ──────────────────────────────────────────
echo "Step 2/3: Installing packages from requirements.txt..."
echo "   This takes 3-5 minutes. Please wait."
echo ""

venv/bin/pip install --upgrade pip --quiet

venv/bin/pip install -r requirements.txt \
    --quiet --no-warn-script-location

if [ $? -ne 0 ]; then
    echo "❌ Package installation failed."
    echo "   Check your internet connection and try again."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "   Done."
echo ""

# ── Verify ────────────────────────────────────────────────────
echo "Step 3/3: Verifying installation..."

venv/bin/python -c "
import streamlit, pandas, statsmodels, sklearn, plotly, openpyxl
print('   streamlit   ', streamlit.__version__, ' OK')
print('   pandas      ', pandas.__version__,    ' OK')
print('   statsmodels ', statsmodels.__version__,' OK')
print('   scikit-learn', sklearn.__version__,    ' OK')
print('   plotly      ', plotly.__version__,     ' OK')
"

if [ $? -ne 0 ]; then
    echo "❌ Verification failed. Please run setup again."
    read -p "Press Enter to exit..."
    exit 1
fi

# ── Set up config ─────────────────────────────────────────────
if [ ! -f "config_ION.xlsx" ] && [ -f "config_template.xlsx" ]; then
    cp config_template.xlsx config_ION.xlsx
    echo ""
    echo "✅ Created config_ION.xlsx — open it and fill in your paths."
fi

# ── Make launchers executable ─────────────────────────────────
chmod +x RUN_APP.command

echo ""
echo "============================================================"
echo "  ✅ SETUP COMPLETE"
echo ""
echo "  From now on just double-click RUN_APP.command"
echo "============================================================"
echo ""
read -p "Press Enter to launch the app now..."
cd app && ../venv/bin/python -m streamlit run app_FINAL.py \
    --server.headless false \
    --browser.gatherUsageStats false \
    --server.port 8501