#!/bin/bash

# Move to the folder containing this script
cd "$(dirname "$0")"

# Find the config file in this folder
CONFIG_FILE=$(ls config_*.xlsx 2>/dev/null | head -n 1)

if [ -z "$CONFIG_FILE" ]; then
    echo "❌ No config file found. Please make sure a config_*.xlsx file is in this folder."
    read -p "Press Enter to close..."
    exit 1
fi

echo "Running pipeline with: $CONFIG_FILE"
echo "Please wait..."

python3 main.py "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Pipeline completed successfully!"
else
    echo ""
    echo "❌ Pipeline failed. Please contact your support team."
fi

read -p "Press Enter to close..."
