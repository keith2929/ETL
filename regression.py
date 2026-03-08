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

def safe_float(val):
    """Convert to float safely, return None if not possible."""
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None

def df_to_records_safe(df):
    """Convert DataFrame to JSON-safe records (NaN/Inf → None)."""
    if df is None or df.empty:
        return []
    return json.loads(df.replace([np.inf, -np.inf], np.nan).to_json(orient='records'))


# -----------------------------
# Analysis 1 — Month-on-month trends
# Keys expected by app.py: mom_trends.redemptions_by_month, mom_trends.gto_by_month
# -----------------------------
def analyse_mom_trends(campaign, gto):
    result = {}

    if not campaign.empty and 'month_year' in campaign.columns:
        mom = campaign.groupby('month_year').agg(
            redemptions   =('receipt_no',    'nunique'),
            points_issued =('points_earned', 'sum'),
            txn_amount    =('amount',        'sum'),
        ).reset_index()
        mom['sort_key'] = month_order(mom['month_year'])
        mom = mom.sort_values('sort_key').drop(columns='sort_key')
        for col in ['redemptions', 'points_issued', 'txn_amount']:
            mom[f'{col}_mom_pct'] = mom[col].pct_change().mul(100).round(2)
        result['redemptions_by_month'] = df_to_records_safe(mom)

    if not gto.empty and 'month_year' in gto.columns:
        gto_grp = gto.groupby('month_year').agg(
            total_gto_amount=('gto_amount', 'sum'),
            total_gto_rent  =('gto_rent',   'sum'),
        ).reset_index()
        gto_grp['sort_key'] = month_order(gto_grp['month_year'])
        gto_grp = gto_grp.sort_values('sort_key').drop(columns='sort_key')
        gto_grp['gto_mom_pct'] = gto_grp['total_gto_amount'].pct_change().mul(100).round(2)
        result['gto_by_month'] = df_to_records_safe(gto_grp)

    return result


# -----------------------------
# Analysis 2 — Campaign ROI
# Keys: campaign_roi.summary, campaign_roi.monthly_roi, campaign_roi.top_outlets_by_roi
# -----------------------------
def analyse_campaign_roi(campaign, gto):
    result = {}

    if campaign.empty or 'month_year' not in campaign.columns:
        return result

    monthly = campaign.groupby('month_year').agg(
        points_issued=('points_earned', 'sum'),
        txn_revenue  =('amount',        'sum'),
        redemptions  =('receipt_no',    'nunique'),
    ).reset_index()
    monthly['campaign_cost']     = monthly['points_issued'] * POINTS_COST_SGD
    monthly['voucher_redeemed']  = monthly['txn_revenue']

    if not gto.empty and 'month_year' in gto.columns:
        gto_monthly = gto.groupby('month_year').agg(gto_revenue=('gto_amount', 'sum')).reset_index()
        monthly = pd.merge(monthly, gto_monthly, on='month_year', how='left').fillna(0)
        monthly['roi_ratio'] = np.where(
            monthly['campaign_cost'] > 0,
            (monthly['gto_revenue'] / monthly['campaign_cost']).round(2), np.nan
        )
    else:
        monthly['gto_revenue'] = 0
        monthly['roi_ratio']   = np.nan

    monthly['sort_key'] = month_order(monthly['month_year'])
    monthly = monthly.sort_values('sort_key').drop(columns='sort_key')
    result['monthly_roi'] = df_to_records_safe(monthly)

    valid_roi = monthly['roi_ratio'].dropna()
    result['summary'] = {
        'avg_roi_ratio':           safe_float(valid_roi.mean()),
        'median_roi_ratio':        safe_float(valid_roi.median()),
        'total_voucher_redeemed':  safe_float(monthly['txn_revenue'].sum()),
        'total_gto_revenue':       safe_float(monthly.get('gto_revenue', pd.Series([0])).sum()),
        'total_campaign_cost':     safe_float(monthly['campaign_cost'].sum()),
    }

    if 'outlet_name' in campaign.columns and not gto.empty and 'shop_name' in gto.columns:
        outlet = campaign.groupby('outlet_name').agg(
            points_issued=('points_earned', 'sum'),
            txn_revenue  =('amount',        'sum'),
            redemptions  =('receipt_no',    'nunique'),
        ).reset_index()
        outlet['campaign_cost'] = outlet['points_issued'] * POINTS_COST_SGD
        gto_outlet = gto.groupby('shop_name').agg(gto_revenue=('gto_amount', 'sum')).reset_index()
        merged_outlet = pd.merge(
            outlet, gto_outlet.rename(columns={'shop_name': 'outlet_name'}),
            on='outlet_name', how='left'
        ).fillna(0)
        merged_outlet['roi_ratio'] = np.where(
            merged_outlet['campaign_cost'] > 0,
            (merged_outlet['gto_revenue'] / merged_outlet['campaign_cost']).round(2), np.nan
        )
        top10 = merged_outlet.dropna(subset=['roi_ratio']).nlargest(10, 'roi_ratio')
        result['top_outlets_by_roi'] = df_to_records_safe(
            top10[['outlet_name', 'roi_ratio', 'redemptions', 'campaign_cost', 'gto_revenue']]
        )

    return result


# -----------------------------
# Analysis 3 — Campaign type (brand vs mall)
# Keys: campaign_type.by_funding_type, campaign_type.gto_by_funding_type, campaign_type.monthly_by_type
# -----------------------------
def analyse_campaign_type(campaign, gto):
    result = {}

    if campaign.empty:
        return result

    fund_col = next((c for c in ['campaign_source', 'campaign_type'] if c in campaign.columns), None)
    if not fund_col:
        return result

    summary = campaign.groupby(fund_col).agg(
        redemptions  =('receipt_no',    'nunique'),
        points_issued=('points_earned', 'sum'),
        txn_revenue  =('amount',        'sum'),
        avg_points   =('points_earned', 'mean'),
        avg_spend    =('amount',        'mean'),
    ).reset_index().rename(columns={fund_col: 'funding_type'})
    summary['campaign_cost'] = summary['points_issued'] * POINTS_COST_SGD
    result['by_funding_type'] = df_to_records_safe(summary)

    if not gto.empty and 'month_year' in campaign.columns and 'month_year' in gto.columns \
            and 'outlet_name' in campaign.columns and 'shop_name' in gto.columns:
        camp_gto = pd.merge(
            campaign,
            gto[['shop_name', 'month_year', 'gto_amount']],
            left_on=['outlet_name', 'month_year'], right_on=['shop_name', 'month_year'],
            how='left'
        ).fillna({'gto_amount': 0})
        gto_by_type = camp_gto.groupby(fund_col).agg(
            total_gto_revenue=('gto_amount',   'sum'),
            total_cost       =('points_earned', lambda x: x.sum() * POINTS_COST_SGD),
        ).reset_index().rename(columns={fund_col: 'funding_type'})
        gto_by_type['roi_ratio'] = np.where(
            gto_by_type['total_cost'] > 0,
            (gto_by_type['total_gto_revenue'] / gto_by_type['total_cost']).round(2), np.nan
        )
        result['gto_by_funding_type'] = df_to_records_safe(gto_by_type)

    if 'month_year' in campaign.columns:
        monthly = campaign.groupby([fund_col, 'month_year']).agg(
            redemptions  =('receipt_no',    'nunique'),
            txn_revenue  =('amount',        'sum'),
            points_issued=('points_earned', 'sum'),
        ).reset_index().rename(columns={fund_col: 'funding_type'})
        monthly['sort_key'] = month_order(monthly['month_year'])
        monthly = monthly.sort_values('sort_key').drop(columns='sort_key')
        result['monthly_by_type'] = df_to_records_safe(monthly)

    return result


# -----------------------------
# Analysis 4 — Loyalty points
# Keys: loyalty_points.correlations, loyalty_points.top_outlets_by_points, loyalty_points.regression
# -----------------------------
def analyse_loyalty_points(campaign, gto):
    result = {}

    if campaign.empty or gto.empty:
        return result
    if 'month_year' not in campaign.columns or 'month_year' not in gto.columns:
        return result

    pts = campaign.groupby('month_year').agg(
        points_issued=('points_earned', 'sum'),
        txn_amount   =('amount',        'sum'),
        redemptions  =('receipt_no',    'nunique'),
    ).reset_index()
    gto_grp = gto.groupby('month_year').agg(gto_amount=('gto_amount', 'sum')).reset_index()
    merged  = pd.merge(pts, gto_grp, on='month_year').dropna()

    if len(merged) >= 4:
        corr_pts,   pval_pts   = stats.pearsonr(merged['points_issued'], merged['gto_amount'])
        corr_spend, pval_spend = stats.pearsonr(merged['txn_amount'],    merged['gto_amount'])
        result['correlations'] = {
            'points_vs_gto_revenue': safe_float(corr_pts),
            'spend_vs_gto_revenue':  safe_float(corr_spend),
            'points_pvalue':         safe_float(pval_pts),
            'spend_pvalue':          safe_float(pval_spend),
        }

        # OLS: GTO revenue ~ points_issued + txn_amount
        X     = sm.add_constant(merged[['points_issued', 'txn_amount']])
        model = sm.OLS(merged['gto_amount'], X).fit()
        sig   = [k for k, p in model.pvalues.items() if p < 0.05 and k != 'const']
        result['regression'] = {
            'r_squared':     safe_float(model.rsquared),
            'adj_r_squared': safe_float(model.rsquared_adj),
            'n_obs':         int(len(merged)),
            'coefs':         {k: safe_float(v) for k, v in model.params.items()},
            'pvalues':       {k: safe_float(v) for k, v in model.pvalues.items()},
            'significant':   sig,
            'insight': (
                f"R² = {model.rsquared:.3f} — the model explains "
                f"{model.rsquared*100:.1f}% of variance in GTO revenue. "
                f"Significant predictors: {', '.join(sig) if sig else 'none at p<0.05'}."
            ),
        }

    name_col = 'final_gto_name' if 'final_gto_name' in campaign.columns else \
               'outlet_name'    if 'outlet_name'    in campaign.columns else None
    if name_col:
        top_pts = campaign.groupby(name_col).agg(
            total_points=('points_earned', 'sum'),
            total_spend =('amount',        'sum'),
            redemptions =('receipt_no',    'nunique'),
        ).reset_index().rename(columns={name_col: 'final_gto_name'})
        result['top_outlets_by_points'] = df_to_records_safe(
            top_pts.nlargest(10, 'total_points')
        )

    return result


# -----------------------------
# Analysis 5 — Tenant turnover
# Keys: tenant_turnover.turnover_by_trade_type, tenant_turnover.gto_stayed_vs_exited
# -----------------------------
def analyse_tenant_turnover(turnover, gto):
    result = {}

    if turnover.empty:
        return result

    trade_col = next(
        (c for c in ['ion_trade_type_name', 'jv_sg_trade_type', 'trade_type'] if c in turnover.columns),
        None
    )
    exit_statuses = ['expired', 'terminated', 'exited']

    if trade_col:
        name_col = 'shop_name' if 'shop_name' in turnover.columns else turnover.columns[0]
        total = turnover.groupby(trade_col)[name_col].nunique().rename('total_tenants').reset_index()
        total.columns = ['ion_trade_type_name', 'total_tenants']

        if 'lease_status' in turnover.columns:
            exited = (
                turnover[turnover['lease_status'].str.lower().isin(exit_statuses)]
                .groupby(trade_col).size().rename('exited_count').reset_index()
            )
            exited.columns = ['ion_trade_type_name', 'exited_count']
            by_trade = total.merge(exited, on='ion_trade_type_name', how='left').fillna(0)
            by_trade['turnover_rate_pct'] = (
                by_trade['exited_count'] / by_trade['total_tenants'] * 100
            ).round(2)
            result['turnover_by_trade_type'] = df_to_records_safe(
                by_trade.sort_values('turnover_rate_pct', ascending=False)
            )
        else:
            result['turnover_by_trade_type'] = df_to_records_safe(
                total.sort_values('total_tenants', ascending=False)
            )

    if not gto.empty and 'shop_name' in gto.columns and \
            'shop_name' in turnover.columns and 'lease_status' in turnover.columns:
        status_map = (
            turnover.drop_duplicates('shop_name')
            .set_index('shop_name')['lease_status'].str.lower()
        )
        gto2 = gto.copy()
        gto2['lease_status'] = gto2['shop_name'].map(status_map)
        gto2['stayed'] = ~gto2['lease_status'].isin(exit_statuses)

        def stats_for(mask):
            sub = gto2[mask]
            return {
                'count':          int(sub['shop_name'].nunique()),
                'avg_gto_amount': safe_float(sub['gto_amount'].mean()) if 'gto_amount' in sub.columns else None,
                'avg_gto_rent':   safe_float(sub['gto_rent'].mean())   if 'gto_rent'   in sub.columns else None,
            }

        result['gto_stayed_vs_exited'] = {
            'stayed': stats_for(gto2['stayed']),
            'exited': stats_for(~gto2['stayed']),
        }

    return result


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        DATA_FOLDER     = sys.argv[1]
        COMBINED_FOLDER = sys.argv[2]
    else:
        _config_file = sys.argv[1] if len(sys.argv) == 2 else "config_Keith.xlsx"
        _script_dir  = Path(__file__).resolve().parent
        _paths_df    = pd.read_excel(_script_dir / _config_file, sheet_name='paths')
        _config      = dict(zip(_paths_df['Setting'].astype(str).str.strip(), _paths_df['Value']))
        DATA_FOLDER     = str(_config.get('cleaned_data',  '')).strip()
        COMBINED_FOLDER = str(_config.get('combined_data', '')).strip()
        print(f"📖 Loaded config from {_config_file}")

    print("\n" + "="*60)
    print("REGRESSION PROCESS STARTING")
    print("="*60 + "\n")

    if not os.path.exists(DATA_FOLDER):
        print(f"❌ ERROR: Cleaned data folder not found at: {DATA_FOLDER}")
        sys.exit(1)

    os.makedirs(COMBINED_FOLDER, exist_ok=True)

    campaign_file   = os.path.join(DATA_FOLDER,     'campaign_all.csv')
    output_file     = os.path.join(COMBINED_FOLDER, 'combined_campaign_transaction_gto.xlsx')
    output_gto_file = os.path.join(COMBINED_FOLDER, 'GTO.xlsx')
    output_comb_file= os.path.join(COMBINED_FOLDER, 'combined_campaign_transaction_with_gto.xlsx')

    # Find transaction file dynamically
    txn_files = [f for f in os.listdir(DATA_FOLDER) if 'mall_member' in f and f.endswith('.csv')]
    txn_file  = os.path.join(DATA_FOLDER, txn_files[0]) if txn_files else ''

    # Find GTO rent file dynamically
    gto_rent_xlsx = [f for f in os.listdir(DATA_FOLDER) if 'gto_monthly_rent' in f and f.endswith('.xlsx')]
    gto_rent_csv  = [f for f in os.listdir(DATA_FOLDER) if 'gto_monthly_rent' in f and f.endswith('.csv')]
    if gto_rent_xlsx:
        gto_file = os.path.join(DATA_FOLDER, gto_rent_xlsx[0])
    elif gto_rent_csv:
        gto_file = os.path.join(DATA_FOLDER, gto_rent_csv[0])
    else:
        print("❌ No gto_monthly_rent file found in cleaned data folder.")
        sys.exit(1)

    # ----------------------------
    # Step 1: Load + merge campaign + transaction
    # ----------------------------
    print("📊 Step 1: Merging campaign and transaction data...")

    if not os.path.exists(campaign_file):
        print(f"❌ campaign_all.csv not found at: {campaign_file}")
        sys.exit(1)

    campaign = pd.read_csv(campaign_file)
    for col in campaign.columns:
        if 'date' in col.lower():
            campaign[col] = pd.to_datetime(campaign[col], errors='coerce')

    if 'month' in campaign.columns and 'year' in campaign.columns:
        campaign['month_year'] = campaign['month'].astype(str) + '-' + campaign['year'].astype(str)

    if txn_file and os.path.exists(txn_file):
        transaction = pd.read_csv(txn_file)
        transaction = transaction.rename(columns={'ReceiptNo': 'receipt_no'})
        txn_cols = ['receipt_no'] + [c for c in ['transaction_type', 'amount', 'points_earned']
                                     if c in transaction.columns]
        merged_campaign = pd.merge(campaign, transaction[txn_cols], on='receipt_no', how='left')
    else:
        print("⚠️  Transaction file not found — using campaign data only.")
        merged_campaign = campaign.copy()

    drop_cols = ['sr_no', 'voucher_code', 'voucher_value', 'transaction_date',
                 'year', 'campaign_type', 'month']
    merged_campaign = merged_campaign.drop(columns=drop_cols, errors='ignore')
    merged_campaign.to_excel(output_file, index=False)
    print(f"✅ Saved: {output_file}")

    # ----------------------------
    # Step 2: Process GTO data
    # ----------------------------
    print("\n📊 Step 2: Processing GTO data...")

    gto_data = pd.read_excel(gto_file) if gto_file.endswith('.xlsx') else pd.read_csv(gto_file)

    if 'gto_reporting_month' in gto_data.columns:
        gto_data['month_year'] = pd.to_datetime(
            gto_data['gto_reporting_month'], errors='coerce'
        ).dt.strftime('%b-%Y')
    elif 'month_year' not in gto_data.columns:
        print("⚠️ 'gto_reporting_month' column not found — month_year will be missing.")

    keep = [c for c in ['shop_name','gto_amount','gto_rent','month_year','lease_status','nla_sqft']
            if c in gto_data.columns]
    gto_data[keep].to_excel(output_gto_file, index=False)
    print(f"✅ Saved: {output_gto_file}")

    # ----------------------------
    # Step 3: Merge GTO with campaign
    # ----------------------------
    print("\n📊 Step 3: Merging GTO with combined campaign/transaction data...")

    gto_data2         = pd.read_excel(output_gto_file)
    combined_campaign = pd.read_excel(output_file)

    if 'outlet_name' in combined_campaign.columns:
        combined_campaign['outlet_name'] = combined_campaign['outlet_name'].str.lower().str.strip()
    if 'shop_name' in gto_data2.columns:
        gto_data2['shop_name'] = gto_data2['shop_name'].str.lower().str.strip()

    gto_uniq = gto_data2.drop_duplicates(
        subset=[c for c in ['shop_name','month_year'] if c in gto_data2.columns]
    )
    if 'outlet_name' in combined_campaign.columns and 'month_year' in combined_campaign.columns \
            and 'shop_name' in gto_uniq.columns and 'month_year' in gto_uniq.columns:
        gto_rev_cols = [c for c in ['gto_amount','gto_rent'] if c in gto_uniq.columns]
        merged = pd.merge(
            combined_campaign,
            gto_uniq[['shop_name','month_year'] + gto_rev_cols],
            left_on=['outlet_name','month_year'], right_on=['shop_name','month_year'], how='left'
        )
        agg_dict = {c: 'sum' for c in gto_rev_cols}
        if agg_dict:
            agg = merged.groupby(['month_year','outlet_name'], as_index=False).agg(agg_dict)
            agg.to_excel(output_comb_file, index=False)
            print(f"✅ Saved: {output_comb_file}")

    # ----------------------------
    # Step 4: Run all analyses
    # ----------------------------
    print("\n📊 Step 4: Running analyses...")

    cdf = pd.read_excel(output_file)
    if 'month_year' not in cdf.columns and 'month_year' in campaign.columns:
        cdf['month_year'] = campaign['month_year'].values[:len(cdf)]

    gto_analysis = pd.read_excel(output_gto_file)

    turnover_path = find_file(DATA_FOLDER, 'gto_tenant_turnover', '.xlsx') or \
                    find_file(DATA_FOLDER, 'gto_tenant_turnover', '.csv')
    turnover_df = (pd.read_excel(turnover_path) if turnover_path and turnover_path.endswith('.xlsx') else
                   pd.read_csv(turnover_path)   if turnover_path else pd.DataFrame())

    mom_results      = analyse_mom_trends(cdf, gto_analysis)
    roi_results      = analyse_campaign_roi(cdf, gto_analysis)
    type_results     = analyse_campaign_type(cdf, gto_analysis)
    loyalty_results  = analyse_loyalty_points(cdf, gto_analysis)
    turnover_results = analyse_tenant_turnover(turnover_df, gto_analysis)

    # ----------------------------
    # Step 5: Save insights_report.xlsx
    # ----------------------------
    print("\n📊 Step 5: Saving insights report...")

    report_path = os.path.join(COMBINED_FOLDER, 'insights_report.xlsx')
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        sheets = [
            ('MoM_Campaign',       mom_results.get('redemptions_by_month')),
            ('MoM_GTO',            mom_results.get('gto_by_month')),
            ('Campaign_ROI',       roi_results.get('monthly_roi')),
            ('Top_ROI_Outlets',    roi_results.get('top_outlets_by_roi')),
            ('Brand_vs_Mall',      type_results.get('by_funding_type')),
            ('Brand_vs_Mall_MoM',  type_results.get('monthly_by_type')),
            ('Top_Points_Outlets', loyalty_results.get('top_outlets_by_points')),
            ('Turnover_by_Trade',  turnover_results.get('turnover_by_trade_type')),
        ]
        for sheet_name, records in sheets:
            if records:
                pd.DataFrame(records).to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"  ✅ Saved: {report_path}")

    # ----------------------------
    # Step 6: Save insights.json — keys MUST match app.py tab_insights expectations
    # ----------------------------
    insights_payload = {
        'mom_trends':      mom_results,       # app reads: .redemptions_by_month, .gto_by_month
        'campaign_roi':    roi_results,       # app reads: .summary, .monthly_roi, .top_outlets_by_roi
        'campaign_type':   type_results,      # app reads: .by_funding_type, .gto_by_funding_type, .monthly_by_type
        'loyalty_points':  loyalty_results,   # app reads: .correlations, .top_outlets_by_points, .regression
        'tenant_turnover': turnover_results,  # app reads: .turnover_by_trade_type, .gto_stayed_vs_exited
    }

    json_path = os.path.join(COMBINED_FOLDER, 'insights.json')
    with open(json_path, 'w') as f:
        json.dump(insights_payload, f, indent=2, default=str)
    print(f"  ✅ Saved: {json_path}")

    print("\n" + "="*60)
    print("✅ REGRESSION COMPLETED — outputs saved to:")
    print(f"   {COMBINED_FOLDER}")
    print("="*60)