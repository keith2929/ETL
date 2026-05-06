#!/bin/bash
cd "$(dirname "$0")"

if [ ! -f "venv/bin/python" ]; then
    echo "Setup not complete."
    echo "Please double-click setup_mac.command first."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "============================================================"
echo "  ION ORCHARD LOYALTY PIPELINE"
echo "  Starting — browser opens in ~10 seconds"
echo "  Keep this window open while using the app"
echo "  If browser doesn't open: http://localhost:8501"
echo "============================================================"
echo ""

cd app
../venv/bin/python -m streamlit run app_FINAL.py \
    --server.headless false \
    --browser.gatherUsageStats false \
    --server.port 8501 \
    --theme.base dark

echo ""
read -p "App stopped. Press Enter to exit..."