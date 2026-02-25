import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set file paths
DATA_FOLDER = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\cleaned data"

class MallLoyaltyAnalyzer:
    def __init__(self):
        self.data = {}
        self.results = {}
        
    def load_data(self):
        """Load all CSV files from the cleaned data folder"""
        files = {
            'campaign': 'campaign_all.csv',
            'transactions': 'mall_member.csv', 
            'monthly_sales': 'gto_monthly_sales.csv',
            'monthly_rent': 'gto_monthly_rent.csv',
            'tenant_turnover': 'gto_tenant_turnover.csv'
        }
        
        for key, filename in files.items():
            filepath = os.path.join(DATA_FOLDER, filename)
            if os.path.exists(filepath):
                self.data[key] = pd.read_csv(filepath)
                # Convert date columns
                for col in self.data[key].columns:
                    if 'date' in col.lower():
                        self.data[key][col] = pd.to_datetime(self.data[key][col], errors='coerce')
        return self.data


# File paths for campaign and transaction data
campaign_file = '/Users/kimbogyeong/Desktop/Capstone/cleaned data/campaign_all.csv'
transaction_file = '/Users/kimbogyeong/Desktop/Capstone/cleaned data/mall_member_2024_to_2025.csv'
output_file = '/Users/kimbogyeong/Desktop/Capstone/combined data/combined_campaign_transaction_gto.xlsx'

# Load the campaign data
campaign = pd.read_csv(campaign_file)

# Load the transaction data (mall_member data)
transaction = pd.read_csv(transaction_file)

# Standardize column names to match (convert ReceiptNo to receipt_no)
transaction = transaction.rename(columns={'ReceiptNo': 'receipt_no'})

# Create 'month_year' column in the campaign data (combining 'month' and 'year')
campaign['month_year'] = campaign['month'] + '-' + campaign['year'].astype(str)

# Merge the two datasets based on 'receipt_no'
merged_campaign = pd.merge(campaign, transaction[['receipt_no', "transaction_type", 'amount', 'points_earned']], on='receipt_no', how='left')

# Drop unwanted columns from merged_campaign (after merging)
columns_to_remove = ['sr_no', 'voucher_code', 'voucher_value', 'transaction_date', 'year', 'campaign_type', 'month']
merged_campaign = merged_campaign.drop(columns=columns_to_remove, errors='ignore')

# Check the result (display the first few rows to verify the merge and column removal)
print(merged_campaign.head())

# Save the merged result to a new Excel file in 'combined data' folder
merged_campaign.to_excel(output_file, index=False)

# Confirm the new file was saved
print(f"✅ New file saved: {output_file}")