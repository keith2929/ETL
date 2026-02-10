# Capstone ETL → Power BI Pipeline

## Overview
Python ETL pipeline that ingests mall, brand, and GTO Excel files,
standardises schemas, validates dates, and outputs Power BI–ready datasets.

## Tech Stack
- Python
- Power BI

## Workflow
Raw Excel → ETL (Python) → Clean CSV → Power BI

## How to Run
pip install -r requirements.txt
update config.xlsx
update schema.xlsx if needed
python main.py
