"""
data_loader.py
--------------
Loads 3 combined source files, merges on Receipt No,
outputs campaign_all.csv to cleaned data folder.

Also loads the GTO file and computes Member / Non-Member Sales,
saving them as CSVs so regression_FINAL.py can read them directly.

Source files (raw data folder):
  combined_Mall_Campaign.xlsx   → mall-funded vouchers
  combined_Brand_Campaign.xlsx  → brand-funded vouchers
  combined_Mall_Trans.xlsx      → member transactions (has Amount)
  *gto*lease*.xlsx              → GTO monthly sales (header row 7)

Outputs (cleaned data folder):
  campaign_all.xlsx / .csv
  gto_member_sales.csv          → monthly member sales (from campaign amounts)
  gto_nonmember_sales.csv       → monthly non-member sales (GTO − member)

Merge key: Receipt No
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats as sp_stats


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


def safe_float(val):
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Load & merge the 3 combined campaign files
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

    # ── Drop rows where amount is 0 or NaN ─────────────────────────────────
    before_filter = len(merged)
    merged = merged[~((merged['amount'] == 0) | (merged['amount'].isna()))]
    removed = before_filter - len(merged)
    print(f"   Removed {removed} rows with amount = 0 or missing (NaN)")

    # ── Summary stats ─────────────────────────────────────────────────────
    print(f"\n   📊 Dataset Summary:")
    print(f"   Mall outlets:        {merged[merged['campaign_source']=='mall']['outlet_name'].nunique()}")
    print(f"   Brand outlets:       {merged[merged['campaign_source']=='brand']['outlet_name'].nunique()}")
    print(f"   Mall voucher codes:  {merged[merged['campaign_source']=='mall']['voucher_code'].nunique()}")
    print(f"   Brand voucher codes: {merged[merged['campaign_source']=='brand']['voucher_code'].nunique()}")
    print(f"   Total rows:          {len(merged):,}")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# GTO loading  (moved here from regression_FINAL.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_gto(raw_folder: str) -> pd.DataFrame:
    """
    Scans raw_folder for a file matching *gto*lease*.xlsx,
    reads it with header=7, and returns a clean DataFrame with columns:
      shop_name, month_year, gto_amount
    Returns an empty DataFrame if no matching file is found.
    """
    for f in os.listdir(raw_folder):
        if f.startswith('~$'):
            continue
        if 'gto' in f.lower() and 'lease' in f.lower() and f.endswith('.xlsx'):
            path = os.path.join(raw_folder, f)
            df   = pd.read_excel(path, header=7)
            df.columns = df.columns.str.strip()
            df = df.dropna(how='all').reset_index(drop=True)
            df['shop_name']  = df['Shop Name'].astype(str).str.strip().str.lower()
            df['gto_amount'] = pd.to_numeric(df['GTO Amount ($)'], errors='coerce')
            df['month_year'] = pd.to_datetime(
                df['GTO Reporting Month'], errors='coerce'
            ).dt.strftime('%b-%Y')
            print(f"✅ GTO loaded:     {len(df):,} rows from {f}")
            return df[['shop_name', 'month_year', 'gto_amount']].dropna()
    print("⚠️  GTO file not found (expected *gto*lease*.xlsx in raw data folder)")
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Build Member / Non-Member monthly series  (moved from regression_FINAL.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_member_nonmember(campaign: pd.DataFrame, gto: pd.DataFrame) -> dict:
    """
    Member Sales   = campaign amount aggregated by outlet × month
    Non-Member Sales = GTO Amount - Member Sales (outlet × month), clipped at 0
    Returns {'member_monthly': DataFrame, 'nonmember_monthly': DataFrame}
    """
    campaign = campaign.copy()
    campaign['outlet_lower'] = campaign['outlet_name'].astype(str).str.strip().str.lower()
    campaign['amount']       = pd.to_numeric(campaign['amount'], errors='coerce')

    # Member Sales: sum amount by outlet × month
    member = (campaign.groupby(['outlet_lower', 'month_year'])['amount']
                      .sum().reset_index()
                      .rename(columns={'amount': 'member_sales'}))

    # GTO: already shop_name × month_year × gto_amount
    gto_grp = (gto.groupby(['shop_name', 'month_year'])['gto_amount']
                   .sum().reset_index())

    # Merge on outlet_lower == shop_name
    merged = pd.merge(
        gto_grp, member,
        left_on=['shop_name', 'month_year'],
        right_on=['outlet_lower', 'month_year'],
        how='left'
    )
    merged['member_sales']     = merged['member_sales'].fillna(0)
    merged['non_member_sales'] = (merged['gto_amount'] - merged['member_sales']).clip(lower=0)

    def monthly_total(df, col):
        grp = (df.groupby('month_year')[col]
                .sum().reset_index())
        grp['sort_key'] = pd.to_datetime(grp['month_year'], format='%b-%Y', errors='coerce')
        grp = grp[grp['sort_key'].dt.year > 2000]   # drop unparseable dates
        grp = grp.sort_values('sort_key').drop(columns='sort_key')
        return grp

    return {
        'member_monthly':    monthly_total(merged, 'member_sales'),
        'nonmember_monthly': monthly_total(merged, 'non_member_sales'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 24-month forecast helper  (moved from regression_FINAL.py)
# ─────────────────────────────────────────────────────────────────────────────
def forecast_monthly(series_df: pd.DataFrame, value_col: str, n_forecast: int = 24) -> list:
    s = series_df.set_index('month_year')[value_col].astype(float)
    s.index = pd.to_datetime(s.index, format='%b-%Y', errors='coerce')
    s = s.sort_index().asfreq('MS')
    forecasts = []
    last_dt = s.index[-1]
    try:
        if len(s) >= 12:
            mdl = ExponentialSmoothing(s, trend='add', seasonal='add',
                                       seasonal_periods=12,
                                       initialization_method='estimated').fit()
        else:
            mdl = ExponentialSmoothing(s, trend='add', seasonal=None,
                                       initialization_method='estimated').fit()
        pred = mdl.forecast(n_forecast)
        for i, val in enumerate(pred):
            fdt = last_dt + pd.DateOffset(months=i + 1)
            forecasts.append({
                'month_year': fdt.strftime('%b-%Y'),
                'forecast':   safe_float(val),
                'type':       'holt_winters',
            })
    except Exception as e:
        print(f"  WARNING: Holt-Winters failed ({e}), using linear extrapolation")
        s_clean = s.dropna()
        if len(s_clean) >= 2:
            x = np.arange(len(s_clean))
            slope, intercept, *_ = sp_stats.linregress(x, s_clean.values)
            last_dt = s_clean.index[-1]
            for i in range(1, n_forecast + 1):
                fdt = last_dt + pd.DateOffset(months=i)
                forecasts.append({
                    'month_year': fdt.strftime('%b-%Y'),
                    'forecast':   safe_float(intercept + slope * (len(s_clean) + i)),
                    'type':       'linear_extrapolation',
                })
        else:
            for i in range(1, n_forecast + 1):
                fdt = last_dt + pd.DateOffset(months=i)
                forecasts.append({
                    'month_year': fdt.strftime('%b-%Y'),
                    'forecast':   None,
                    'type':       'insufficient_data',
                })
    return forecasts


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

    # ── Step 1: Campaign data ─────────────────────────────────────────────
    print("📊 Loading & merging combined campaign files...")
    campaign_all = load_and_merge(raw_data)
    export(campaign_all, cleaned_data, 'campaign_all')

    # ── Step 2: GTO → Member / Non-Member Sales ───────────────────────────
    print("\n📊 Loading GTO file & computing Member / Non-Member Sales...")
    gto = load_gto(raw_data)

    if not gto.empty:
        mnm = build_member_nonmember(campaign_all, gto)

        member_monthly    = mnm['member_monthly']
        nonmember_monthly = mnm['nonmember_monthly']

        # Attach forecasts as extra columns so regression_FINAL can read one file
        member_fore    = forecast_monthly(member_monthly,    'member_sales')
        nonmember_fore = forecast_monthly(nonmember_monthly, 'non_member_sales')

        # Save actuals
        member_monthly.to_csv(
            os.path.join(cleaned_data, 'gto_member_sales.csv'),
            index=False, encoding='utf-8-sig')
        nonmember_monthly.to_csv(
            os.path.join(cleaned_data, 'gto_nonmember_sales.csv'),
            index=False, encoding='utf-8-sig')

        # Save forecasts
        pd.DataFrame(member_fore).to_csv(
            os.path.join(cleaned_data, 'gto_member_sales_forecast.csv'),
            index=False, encoding='utf-8-sig')
        pd.DataFrame(nonmember_fore).to_csv(
            os.path.join(cleaned_data, 'gto_nonmember_sales_forecast.csv'),
            index=False, encoding='utf-8-sig')

        print(f"✅ Saved gto_member_sales.csv          ({len(member_monthly)} months)")
        print(f"✅ Saved gto_nonmember_sales.csv       ({len(nonmember_monthly)} months)")
        print(f"✅ Saved gto_member_sales_forecast.csv ({len(member_fore)} months forecast)")
        print(f"✅ Saved gto_nonmember_sales_forecast.csv ({len(nonmember_fore)} months forecast)")
    else:
        print("⚠️  GTO not loaded — member/non-member CSVs will not be created.")
        print("    The Time Series tab will show 'No Member/Non-Member data'.")

    print("\n" + "="*60)
    print("✅ ETL COMPLETED")
    print(f"   campaign_all: {len(campaign_all):,} rows")
    print("="*60)