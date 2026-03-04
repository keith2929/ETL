#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Capstone Pipeline App..."
pip3 install streamlit pandas openpyxl --quiet
streamlit run app.py --server.headless false
