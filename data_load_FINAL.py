"""
data_loader.py
--------------
Loads 3 combined source files, merges on Receipt No,
outputs campaign_all.csv to cleaned data folder.

Source files (raw data folder):
  combined_Mall_Campaign.xlsx   → mall-funded vouchers
  combined_Brand_Campaign.xlsx  → brand-funded vouchers
  combined_Mall_Trans.xlsx      → member transactions (has Amount)

Merge key: Receipt No
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import os
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_file="config_Keith.xlsx") -> dict:
    script_dir  = Path(__file__).resolve().parent
    config_path = script_dir / config_file

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)

    df  = pd.read_excel(config_path, sheet_name='paths')
    cfg = dict(zip(df['Setting'].astype(str).str.strip(), df['Value'].astype(str).str.strip()))
    print(f"📖 Config loaded: {config_file}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Helper: find file by keyword
# ─────────────────────────────────────────────────────────────────────────────
def find_file(folder: str, keyword: str) -> str:
    for f in os.listdir(folder):
        if keyword.lower() in f.lower() and f.endswith('.xlsx'):
            return os.path.join(folder, f)
    return ''


# ─────────────────────────────────────────────────────────────────────────────
# Load & merge the 3 combined files
# ─────────────────────────────────────────────────────────────────────────────
def load_and_merge(raw_folder: str) -> pd.DataFrame:
    """
    1. Stack mall + brand campaign files
    2. Merge with transaction file on receipt_no to get Amount
    Returns campaign_all DataFrame.
    """
    mall_path  = find_file(raw_folder, 'combined_mall_campaign')
    brand_path = find_file(raw_folder, 'combined_brand_campaign')
    trans_path = find_file(raw_folder, 'combined_mall_trans')

    # ── Campaign files ────────────────────────────────────────────────────
    dfs = []

    if mall_path:
        df = pd.read_excel(mall_path).dropna(how='all')
        df['campaign_source'] = 'mall'
        dfs.append(df)
        print(f"✅ Mall Campaign:  {len(df):,} rows")
    else:
        print("⚠️  combined_Mall_Campaign not found")

    if brand_path:
        df = pd.read_excel(brand_path).dropna(how='all')
        df['campaign_source'] = 'brand'
        dfs.append(df)
        print(f"✅ Brand Campaign: {len(df):,} rows")
    else:
        print("⚠️  combined_Brand_Campaign not found")

    if not dfs:
        print("❌ No campaign files found — aborting")
        sys.exit(1)

    campaign = pd.concat(dfs, ignore_index=True)
    campaign.columns = campaign.columns.str.strip()

    # Standardise column names
    campaign = campaign.rename(columns={
        'SrNo.':              'sr_no',
        'Voucher Type Code':  'voucher_code',
        'Voucher Value':      'voucher_value',
        'Redeem Outlet Code': 'outlet_code',
        'Redeem Outlet Name': 'outlet_name',
        'Redeem Date':        'date',
        'Receipt No':         'receipt_no',
    })

    # Parse dates
    campaign['date']       = pd.to_datetime(campaign['date'], errors='coerce', dayfirst=True)
    campaign['month']      = campaign['date'].dt.month_name().str[:3]
    campaign['year']       = campaign['date'].dt.year.astype('Int64')
    campaign['month_year'] = campaign['month'].astype(str) + '-' + campaign['year'].astype(str)

    print(f"   Stacked campaign: {len(campaign):,} rows total")

    # ── Transaction file ──────────────────────────────────────────────────
    if not trans_path:
        print("⚠️  combined_Mall_Trans not found — amount will be missing")
        campaign['amount']           = pd.NA
        campaign['transaction_type'] = pd.NA
        return campaign

    trans = pd.read_excel(trans_path).dropna(how='all')
    trans.columns = trans.columns.str.strip()
    trans = trans.rename(columns={
        'ReceiptNo':    'receipt_no',
        'Trans Date':   'trans_date',
        'Outlet Code':  'outlet_code_trans',
        'Outlet Name':  'outlet_name_trans',
        'Type':         'transaction_type',
        'Amount Spent': 'amount',
        'SrNo.':        'sr_no_trans',
        'TransactRef5': 'transact_ref5',
        'TransactRef6': 'transact_ref6',
    })
    print(f"✅ Mall Trans:     {len(trans):,} rows")

    # Normalise receipt_no
    campaign['receipt_no'] = campaign['receipt_no'].astype(str).str.strip()
    trans['receipt_no']    = trans['receipt_no'].astype(str).str.strip()

    # Left join: keep all campaign rows, bring in amount
    merged = pd.merge(
        campaign,
        trans[['receipt_no', 'amount', 'transaction_type']],
        on='receipt_no', how='left'
    )

    matched = merged['amount'].notna().sum()
    # ── Clean up insignificant voucher values ─────────────────────────────
    # min40: remove voucher_value=7 (only 1 transaction)
    # min300: remove voucher_value=25 (only 2 transactions)
    before = len(merged)
    merged = merged[~(
        (merged['voucher_code'].astype(str).str.lower().str.contains('min40', na=False)) &
        (merged['voucher_value'].astype(str).str.strip() == '7')
    )]
    merged = merged[~(
        (merged['voucher_code'].astype(str).str.lower().str.contains('min300', na=False)) &
        (merged['voucher_value'].astype(str).str.strip() == '25')
    )]
    removed = before - len(merged)
    print(f"   Removed {removed} rows with insignificant voucher values (min40-7, min300-25)")
    print(f"   Merged: {matched:,}/{len(merged):,} rows matched on receipt_no")

    # ── Summary stats ─────────────────────────────────────────────────────
    print(f"\n   📊 Dataset Summary:")
    print(f"   Mall outlets:        {merged[merged['campaign_source']=='mall']['outlet_name'].nunique()}")
    print(f"   Brand outlets:       {merged[merged['campaign_source']=='brand']['outlet_name'].nunique()}")
    print(f"   Mall voucher codes:  {merged[merged['campaign_source']=='mall']['voucher_code'].nunique()}")
    print(f"   Brand voucher codes: {merged[merged['campaign_source']=='brand']['voucher_code'].nunique()}")
    print(f"   Total rows:          {len(merged):,}")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Export helper
# ─────────────────────────────────────────────────────────────────────────────
def export(df: pd.DataFrame, folder: str, name: str):
    os.makedirs(folder, exist_ok=True)
    df.to_excel(os.path.join(folder, f"{name}.xlsx"), index=False)
    df.to_csv(os.path.join(folder,   f"{name}.csv"),  index=False, encoding='utf-8-sig')
    print(f"✅ Saved {name} ({len(df):,} rows) → {folder}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        raw_data     = sys.argv[1]
        cleaned_data = sys.argv[2]
        config_file  = sys.argv[5] if len(sys.argv) > 5 else "config_Keith.xlsx"
        cfg          = load_config(config_file)
    else:
        config_file  = sys.argv[1] if len(sys.argv) == 2 else "config_Keith.xlsx"
        cfg          = load_config(config_file)
        raw_data     = cfg.get('raw_data',     '')
        cleaned_data = cfg.get('cleaned_data', '')

    print("\n" + "="*60)
    print("ETL STARTING")
    print("="*60)
    print(f"📁 Raw data:    {raw_data}")
    print(f"📁 Cleaned out: {cleaned_data}")
    print("="*60 + "\n")

    if not os.path.exists(raw_data):
        print(f"❌ Raw data folder not found: {raw_data}")
        sys.exit(1)

    print("📊 Loading & merging combined files...")
    campaign_all = load_and_merge(raw_data)
    export(campaign_all, cleaned_data, 'campaign_all')

    print("\n" + "="*60)
    print("✅ ETL COMPLETED")
    print(f"   campaign_all: {len(campaign_all):,} rows")
    print("="*60)