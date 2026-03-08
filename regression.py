import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys
from scipy import stats
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

POINTS_COST_SGD = 0.20  # cost per point issued (SGD)


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
# Helper
# -----------------------------
def find_file(folder, keyword, ext='.xlsx'):
    for f in os.listdir(folder):
        if keyword in f.lower() and f.endswith(ext):
            return os.path.join(folder, f)
    return ''

def month_order(s):
    return pd.to_datetime(s, format='%b-%Y', errors='coerce')


# -----------------------------
# Analysis 1 — Month-on-month trends
# -----------------------------
def analyse_mom_trends(campaign, gto):
    if campaign.empty or 'month_year' not in campaign.columns:
        return pd.DataFrame()
    mom = campaign.groupby('month_year').agg(
        total_points_issued = ('points_earned',  'sum'),
        total_txn_amount    = ('amount',         'sum'),
        total_transactions  = ('receipt_no',     'nunique'),
        campaign_cost_sgd   = ('points_earned',  lambda x: x.sum() * POINTS_COST_SGD),
    ).reset_index()
    if not gto.empty and 'month_year' in gto.columns:
        gto_grp = gto.groupby('month_year').agg(
            total_gto_amount=('gto_amount','sum'),
            total_gto_rent  =('gto_rent',  'sum'),
        ).reset_index()
        mom = pd.merge(mom, gto_grp, on='month_year', how='outer').fillna(0)
    mom['sort_key'] = month_order(mom['month_year'])
    mom = mom.sort_values('sort_key').drop(columns='sort_key')
    for col in ['total_gto_amount','total_points_issued','total_txn_amount']:
        if col in mom.columns:
            mom[f'{col}_mom_pct'] = mom[col].pct_change().mul(100).round(2)
    return mom.fillna(0)


# -----------------------------
# Analysis 2 — Points effectiveness
# -----------------------------
def analyse_points_effectiveness(campaign, gto):
    result = {'available': False}
    if campaign.empty or gto.empty or 'month_year' not in campaign.columns:
        return result
    pts = campaign.groupby('month_year').agg(
        points_issued=('points_earned','sum'),
        txn_amount   =('amount',       'sum'),
    ).reset_index()
    gto_grp = gto.groupby('month_year').agg(gto_amount=('gto_amount','sum')).reset_index()
    merged = pd.merge(pts, gto_grp, on='month_year').dropna()
    if len(merged) < 4:
        return result
    X = sm.add_constant(merged['points_issued'])
    model = sm.OLS(merged['gto_amount'], X).fit()
    corr, pval = stats.pearsonr(merged['points_issued'], merged['gto_amount'])
    direction = 'positive' if corr > 0 else 'negative'
    strength  = 'strong' if abs(corr) > 0.7 else ('moderate' if abs(corr) > 0.4 else 'weak')
    sig       = 'statistically significant' if pval < 0.05 else 'not statistically significant'
    return {
        'available':   True,
        'r_squared':   round(model.rsquared, 4),
        'coefficient': round(model.params.get('points_issued', 0), 4),
        'p_value':     round(model.pvalues.get('points_issued', 1), 4),
        'correlation': round(corr, 4),
        'n_months':    len(merged),
        'insight':     (f"There is a {strength} {direction} correlation (r={corr:.2f}) between "
                        f"points issued and GTO, which is {sig} (p={pval:.3f}). "
                        f"The model explains {model.rsquared*100:.1f}% of variance in GTO."),
        'data':        merged.to_dict(orient='records'),
    }


# -----------------------------
# Analysis 3 — Campaign ROI
# -----------------------------
def analyse_campaign_roi(campaign, gto):
    if campaign.empty or 'month_year' not in campaign.columns:
        return pd.DataFrame()
    grp_cols = ['month_year','outlet_name'] if 'outlet_name' in campaign.columns else ['month_year']
    roi = campaign.groupby(grp_cols).agg(
        points_issued  =('points_earned','sum'),
        txn_revenue    =('amount',       'sum'),
        n_transactions =('receipt_no',   'nunique'),
    ).reset_index()
    roi['campaign_cost_sgd'] = roi['points_issued'] * POINTS_COST_SGD
    if not gto.empty and 'month_year' in gto.columns:
        merge_cols = [c for c in grp_cols if c in gto.columns]
        gto_grp = gto.groupby(merge_cols).agg(gto_revenue=('gto_amount','sum')).reset_index()
        roi = pd.merge(roi, gto_grp, on=merge_cols, how='left').fillna(0)
        roi['roi_vs_gto'] = np.where(roi['campaign_cost_sgd'] > 0,
            (roi['gto_revenue'] / roi['campaign_cost_sgd']).round(2), np.nan)
    roi['roi_vs_txn'] = np.where(roi['campaign_cost_sgd'] > 0,
        (roi['txn_revenue'] / roi['campaign_cost_sgd']).round(2), np.nan)
    roi['sort_key'] = month_order(roi['month_year'])
    return roi.sort_values('sort_key').drop(columns='sort_key').fillna(0)


# -----------------------------
# Analysis 4 — Brand vs mall funded
# -----------------------------
def analyse_brand_vs_mall(campaign):
    if campaign.empty:
        return pd.DataFrame(), pd.DataFrame()
    fund_col = next((c for c in ['campaign_source','campaign_type'] if c in campaign.columns), None)
    if not fund_col:
        return pd.DataFrame(), pd.DataFrame()
    agg = {'points_issued': ('points_earned','sum'), 'txn_revenue': ('amount','sum'),
           'n_transactions': ('receipt_no','nunique'), 'avg_points_per_txn': ('points_earned','mean'),
           'avg_spend_per_txn': ('amount','mean')}
    if 'outlet_name' in campaign.columns:
        agg['n_outlets'] = ('outlet_name','nunique')
    summary = campaign.groupby(fund_col).agg(**agg).reset_index()
    summary['campaign_cost_sgd'] = summary['points_issued'] * POINTS_COST_SGD
    summary['roi_vs_txn'] = np.where(summary['campaign_cost_sgd'] > 0,
        (summary['txn_revenue'] / summary['campaign_cost_sgd']).round(2), np.nan)
    summary = summary.rename(columns={fund_col: 'funding_type'})
    monthly = campaign.groupby([fund_col,'month_year']).agg(
        points_issued =('points_earned','sum'),
        txn_revenue   =('amount',       'sum'),
        n_transactions=('receipt_no',   'nunique'),
    ).reset_index().rename(columns={fund_col: 'funding_type'})
    monthly['campaign_cost_sgd'] = monthly['points_issued'] * POINTS_COST_SGD
    monthly['sort_key'] = month_order(monthly['month_year'])
    monthly = monthly.sort_values('sort_key').drop(columns='sort_key')
    return summary, monthly


# -----------------------------
# Analysis 5 — Tenant turnover
# -----------------------------
def analyse_tenant_turnover(turnover, rent):
    result = {'available': False}
    if turnover.empty:
        return result
    result['available'] = True
    trade_col = next((c for c in ['ion_trade_type_name','jv_sg_trade_type','trade_type'] if c in turnover.columns), None)
    if trade_col:
        by_trade = turnover.groupby(trade_col).agg(
            n_tenants=('shop_name','nunique') if 'shop_name' in turnover.columns else ('lease_no','count'),
        ).reset_index().rename(columns={trade_col:'trade_type'})
        result['by_trade'] = by_trade.sort_values('n_tenants', ascending=False)
    if not rent.empty and 'lease_status' in rent.columns:
        result['lease_status'] = rent['lease_status'].value_counts().reset_index().rename(
            columns={'lease_status':'lease_status','count':'count'})
    if not rent.empty and all(c in rent.columns for c in ['gto_amount','nla_sqft','shop_name']):
        eff = rent.groupby('shop_name').agg(total_gto=('gto_amount','sum'), avg_nla=('nla_sqft','mean')).reset_index()
        eff = eff[eff['avg_nla'] > 0]
        eff['gto_per_sqft'] = (eff['total_gto'] / eff['avg_nla']).round(2)
        result['gto_per_sqft'] = eff.sort_values('gto_per_sqft', ascending=False)
    return result


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

    # ----------------------------
    # Step 4: Business analysis + insights
    # ----------------------------
    print("\n📊 Step 4: Running business analysis...")

    # Load GTO and turnover for analysis
    gto_for_analysis      = pd.read_excel(gto_file)
    turnover_file         = find_file(DATA_FOLDER, 'gto_tenant_turnover', '.xlsx') or                             find_file(DATA_FOLDER, 'gto_tenant_turnover', '.csv')
    turnover_df           = pd.read_excel(turnover_file) if turnover_file and turnover_file.endswith('.xlsx')                             else (pd.read_csv(turnover_file) if turnover_file else pd.DataFrame())

    if 'gto_reporting_month' in gto_for_analysis.columns:
        gto_for_analysis['month_year'] = pd.to_datetime(
            gto_for_analysis['gto_reporting_month'], errors='coerce'
        ).dt.strftime('%b-%Y')

    # Use the merged campaign data (has amount + points_earned from transaction)
    combined_campaign_df = pd.read_excel(output_file)
    # Re-attach month_year if dropped
    if 'month_year' not in combined_campaign_df.columns and 'month_year' in campaign.columns:
        combined_campaign_df['month_year'] = campaign['month_year'].values[:len(combined_campaign_df)]

    mom_trends           = analyse_mom_trends(combined_campaign_df, gto_for_analysis)
    points_eff           = analyse_points_effectiveness(combined_campaign_df, gto_for_analysis)
    roi                  = analyse_campaign_roi(combined_campaign_df, gto_for_analysis)
    brand_summary, brand_monthly = analyse_brand_vs_mall(combined_campaign_df)
    turnover_res         = analyse_tenant_turnover(turnover_df, gto_for_analysis)

    report_path = os.path.join(COMBINED_FOLDER, 'insights_report.xlsx')
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        if not mom_trends.empty:
            mom_trends.to_excel(writer, sheet_name='MoM_Trends', index=False)
        if points_eff.get('available'):
            pd.DataFrame([{k: v for k, v in points_eff.items() if k != 'data'}]).to_excel(
                writer, sheet_name='Points_Effectiveness', index=False)
            pd.DataFrame(points_eff['data']).to_excel(writer, sheet_name='Points_Data', index=False)
        if not roi.empty:
            roi.to_excel(writer, sheet_name='Campaign_ROI', index=False)
        if not brand_summary.empty:
            brand_summary.to_excel(writer, sheet_name='Brand_vs_Mall', index=False)
            brand_monthly.to_excel(writer, sheet_name='Brand_vs_Mall_Monthly', index=False)
        if turnover_res.get('available'):
            for key, sheet in [('by_trade','Turnover_by_Trade'),('lease_status','Lease_Status'),('gto_per_sqft','GTO_per_sqft')]:
                if key in turnover_res and not turnover_res[key].empty:
                    turnover_res[key].to_excel(writer, sheet_name=sheet, index=False)

    print(f"  ✅ Saved: {report_path}")

    # Save summary JSON for Streamlit insights tab
    json.dump({
        'points_effectiveness': {k: v for k, v in points_eff.items() if k != 'data'},
        'brand_mall_summary':   brand_summary.to_dict(orient='records') if not brand_summary.empty else [],
        'roi_summary': {
            'total_cost_sgd': float(roi['campaign_cost_sgd'].sum())  if not roi.empty else None,
            'avg_roi_vs_txn': float(roi['roi_vs_txn'].mean())        if not roi.empty and 'roi_vs_txn' in roi.columns else None,
            'avg_roi_vs_gto': float(roi['roi_vs_gto'].mean())        if not roi.empty and 'roi_vs_gto' in roi.columns else None,
        },
    }, open(os.path.join(COMBINED_FOLDER, 'insights.json'), 'w'), indent=2, default=str)

    print("\n" + "="*60)
    print("✅ REGRESSION COMPLETED — outputs saved to:")
    print(f"   {COMBINED_FOLDER}")
    print("="*60)