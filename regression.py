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


