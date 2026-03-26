import pandas as pd
import numpy as np
import os
import json
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG
# =========================
COMBINED_FOLDER = "outputs"

# Base features to control for size and activity
BASE_FEATURE_CANDIDATES = [
    'txn_count', 'customer_count', 'nla_sqft'
]

# =========================
# BUILD DATASET
# =========================
def build_regression_dataset(gto, campaign):
    df = gto.copy()

    # ── CAMPAIGN DUMMIES ──
    # Identifies the specific campaign source or voucher
    voucher_col = 'voucher_code' if 'voucher_code' in campaign.columns else 'campaign_source'

    if voucher_col in campaign.columns:
        campaign[voucher_col] = campaign[voucher_col].astype(str)
        
        # Create 1/0 dummies for every unique campaign
        camp_agg = (
            campaign
            .assign(active=1)
            .groupby(['month_year', voucher_col])['active']
            .max()
            .unstack(fill_value=0)
        )

        camp_agg.columns = [f"camp_{c}" for c in camp_agg.columns]
        df = df.merge(camp_agg, on='month_year', how='left').fillna(0)

    # ── MONTH DUMMIES (Seasonality Control) ──
    df['month'] = pd.to_datetime(df['month_year'], errors='coerce').dt.month
    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)

    return df

# =========================
# MODEL TRAINING & EXPORT
# =========================
def run_and_save_results(df, target='gto_rent'):
    # Select features: Base controls + Campaign Dummies + Month Dummies
    camp_cols = [c for c in df.columns if c.startswith('camp_')]
    month_cols = [c for c in df.columns if c.startswith('month_')]
    base_cols = [c for c in BASE_FEATURE_CANDIDATES if c in df.columns]
    
    X_cols = base_cols + camp_cols + month_cols
    if not X_cols: return
    
    X = df[X_cols].fillna(0)
    y = df[target].fillna(0)

    # Standardise features so coefficients are comparable
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use Lasso for feature selection (zeroes out ineffective campaigns)
    lasso = LassoCV(cv=5)
    lasso.fit(X_scaled, y)

    # Prepare findings for JSON
    coef_list = []
    for feat, coef in zip(X_cols, lasso.coef_):
        coef_list.append({
            'feature': feat,
            'coef': float(coef),
            'type': 'campaign' if feat.startswith('camp_') else 'control'
        })

    results_json = {
        'linear_regression': {
            'lasso_model': {
                'coef_table': coef_list,
                'r_squared': float(lasso.score(X_scaled, y)),
                'features': X_cols
            }
        }
    }

    if not os.path.exists(COMBINED_FOLDER): os.makedirs(COMBINED_FOLDER)
    with open(os.path.join(COMBINED_FOLDER, 'linear_regression_results.json'), 'w') as f:
        json.dump(results_json, f)

def main(cleaned_folder, combined_folder, mapping_path):
    # Standard main entry to load files from the pipeline folders
    gto_path = os.path.join(cleaned_folder, 'gto.csv')
    camp_path = os.path.join(cleaned_folder, 'campaign_all.csv')
    
    if os.path.exists(gto_path) and os.path.exists(camp_path):
        gto = pd.read_csv(gto_path)
        campaign = pd.read_csv(camp_path)
        df = build_regression_dataset(gto, campaign)
        run_and_save_results(df)
        print("✅ Campaign-based regression complete.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])

