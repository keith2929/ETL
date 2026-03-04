import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MallLoyaltyAnalyzer:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.data = {}
        self.results = {}
        
    def load_data(self):
        """Load all CSV files from the cleaned data folder"""
        files = {
            'campaign':        'campaign_all.csv',
            'transactions':    'mall_member.csv', 
            'monthly_sales':   'gto_monthly_sales.csv',
            'monthly_rent':    'gto_monthly_rent.csv',
            'tenant_turnover': 'gto_tenant_turnover.csv'
        }
        for key, filename in files.items():
            filepath = os.path.join(self.data_folder, filename)
            if os.path.exists(filepath):
                self.data[key] = pd.read_csv(filepath)
                for col in self.data[key].columns:
                    if 'date' in col.lower():
                        self.data[key][col] = pd.to_datetime(self.data[key][col], errors='coerce')
        return self.data


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # If run via main.py: paths passed as args
    # If run directly in Spyder/VS Code: no args, load from config
    if len(sys.argv) >= 3:
        DATA_FOLDER     = sys.argv[1]
        COMBINED_FOLDER = sys.argv[2]
    else:
        import pandas as _pd
        from pathlib import Path as _Path
        _config_file = sys.argv[1] if len(sys.argv) == 2 else "config_Keith.xlsx"
        _script_dir  = _Path(__file__).resolve().parent
        _config_path = _script_dir / _config_file
        _paths_df    = _pd.read_excel(_config_path, sheet_name='paths')
        _config      = dict(zip(_paths_df['Setting'].astype(str).str.strip(), _paths_df['Value']))
        DATA_FOLDER     = str(_config.get('cleaned_data',  '')).strip()
        COMBINED_FOLDER = str(_config.get('combined_data', '')).strip()
        print(f"📖 Loaded config from {_config_file}")

    print("\n" + "="*60)
    print("REGRESSION PROCESS STARTING")
    print("="*60)
    print(f"📁 Cleaned data:  {DATA_FOLDER}")
    print(f"📁 Combined data: {COMBINED_FOLDER}")
    print("="*60 + "\n")

    if not os.path.exists(DATA_FOLDER):
        print(f"❌ ERROR: Cleaned data folder not found at: {DATA_FOLDER}")
        sys.exit(1)

    os.makedirs(COMBINED_FOLDER, exist_ok=True)

    # File paths — all derived from args
    campaign_file        = os.path.join(DATA_FOLDER,     'campaign_all.csv')
    transaction_file     = os.path.join(DATA_FOLDER,     'mall_member_2024_to_2025.csv')
    output_file          = os.path.join(COMBINED_FOLDER, 'combined_campaign_transaction_gto.xlsx')
    output_gto_file      = os.path.join(COMBINED_FOLDER, 'GTO.xlsx')
    output_combined_file = os.path.join(COMBINED_FOLDER, 'combined_campaign_transaction_with_gto.xlsx')

    # Find GTO monthly rent file dynamically (name includes year range)
    gto_files = [f for f in os.listdir(DATA_FOLDER) if 'gto_monthly_rent' in f and f.endswith('.xlsx')]
    if not gto_files:
        print("❌ No gto_monthly_rent file found in cleaned data folder.")
        sys.exit(1)
    gto_file = os.path.join(DATA_FOLDER, gto_files[0])

    # ----------------------------
    # Step 1: Merge campaign + transaction
    # ----------------------------
    print("📊 Step 1: Merging campaign and transaction data...")

    campaign    = pd.read_csv(campaign_file)
    transaction = pd.read_csv(transaction_file)

    transaction = transaction.rename(columns={'ReceiptNo': 'receipt_no'})
    campaign['month_year'] = campaign['month'] + '-' + campaign['year'].astype(str)

    merged_campaign = pd.merge(
        campaign,
        transaction[['receipt_no', 'transaction_type', 'amount', 'points_earned']],
        on='receipt_no',
        how='left'
    )

    columns_to_remove = ['sr_no', 'voucher_code', 'voucher_value', 'transaction_date',
                         'year', 'campaign_type', 'month']
    merged_campaign = merged_campaign.drop(columns=columns_to_remove, errors='ignore')

    merged_campaign.to_excel(output_file, index=False)
    print(f"✅ Saved: {output_file}")

    # ----------------------------
    # Step 2: Process GTO data
    # ----------------------------
    print("\n📊 Step 2: Processing GTO data...")

    gto_data = pd.read_excel(gto_file)

    if 'gto_reporting_month' in gto_data.columns:
        gto_data['month_year'] = pd.to_datetime(
            gto_data['gto_reporting_month'], errors='coerce'
        ).dt.strftime('%b-%Y')
    else:
        print("⚠️ Column 'gto_reporting_month' not found. Please check the column name.")

    gto_data_selected = gto_data[['shop_name', 'gto_amount', 'gto_rent', 'month_year']]
    gto_data_selected.to_excel(output_gto_file, index=False)
    print(f"✅ Saved: {output_gto_file}")

    # ----------------------------
    # Step 3: Merge GTO with combined campaign data
    # ----------------------------
    print("\n📊 Step 3: Merging GTO with combined campaign/transaction data...")

    gto_data          = pd.read_excel(output_gto_file)
    combined_campaign = pd.read_excel(output_file)

    combined_campaign['outlet_name'] = combined_campaign['outlet_name'].str.lower().str.strip()
    gto_data['shop_name']            = gto_data['shop_name'].str.lower().str.strip()

    gto_data_unique = gto_data.drop_duplicates(subset=['shop_name', 'month_year'])

    merged_data = pd.merge(
        combined_campaign,
        gto_data_unique[['shop_name', 'month_year', 'gto_amount', 'gto_rent']],
        left_on=['outlet_name', 'month_year'],
        right_on=['shop_name', 'month_year'],
        how='left'
    )

    aggregated_data = merged_data.groupby(
        ['month_year', 'outlet_name'], as_index=False
    ).agg({'gto_amount': 'sum', 'gto_rent': 'sum'})

    aggregated_data.to_excel(output_combined_file, index=False)
    print(f"✅ Saved: {output_combined_file}")

    print("\n" + "="*60)
    print("✅ REGRESSION COMPLETED — outputs saved to:")
    print(f"   {COMBINED_FOLDER}")
    print("="*60)