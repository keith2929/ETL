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
# Analysis 6 — Time series
# Keys: time_series.gto_trend, time_series.campaign_trend,
#        time_series.forecast, time_series.anomalies, time_series.summary
# -----------------------------
def analyse_time_series(campaign, gto):
    """
    Runs time-series analysis on monthly GTO revenue and campaign redemptions:
      - Trend decomposition (trend / seasonal / residual)
      - 3-month simple moving average
      - Linear trend (slope, direction, strength)
      - 3-month ahead forecast (Holt-Winters exponential smoothing if ≥12 pts,
        else linear extrapolation)
      - Anomaly detection (residuals > 2 std dev flagged)
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = {}
    MIN_OBS = 6   # minimum months needed for any analysis
    MIN_DECOMP = 12  # minimum months for seasonal decomposition

    def build_monthly_series(df, value_col, date_col='month_year'):
        if df.empty or value_col not in df.columns or date_col not in df.columns:
            return pd.Series(dtype=float)
        grp = df.groupby(date_col)[value_col].sum().reset_index()
        grp['_dt'] = pd.to_datetime(grp[date_col], format='%b-%Y', errors='coerce')
        grp = grp.dropna(subset=['_dt']).sort_values('_dt').set_index('_dt')[value_col]
        grp.index = pd.DatetimeIndex(grp.index).to_period('M').to_timestamp()
        return grp

    def linear_trend(series):
        x = np.arange(len(series))
        slope, intercept, r, p, _ = stats.linregress(x, series.values)
        direction = 'upward' if slope > 0 else 'downward'
        strength  = 'strong' if abs(r) > 0.7 else ('moderate' if abs(r) > 0.4 else 'weak')
        return {
            'slope':      safe_float(slope),
            'r_squared':  safe_float(r**2),
            'p_value':    safe_float(p),
            'direction':  direction,
            'strength':   strength,
            'significant': bool(p < 0.05),
        }

    def moving_average(series, window=3):
        ma = series.rolling(window, min_periods=1).mean().round(2)
        return [{'month_year': str(k.strftime('%b-%Y')), 'value': safe_float(v)}
                for k, v in ma.items()]

    def detect_anomalies(series, residuals):
        threshold = 2.0
        std = residuals.std()
        mean = residuals.mean()
        anomalies = []
        for dt, res in residuals.items():
            if abs(res - mean) > threshold * std:
                anomalies.append({
                    'month_year': str(dt.strftime('%b-%Y')),
                    'actual':     safe_float(series.get(dt, None)),
                    'residual':   safe_float(res),
                    'direction':  'above' if res > mean else 'below',
                })
        return anomalies

    def forecast_series(series, steps=3):
        forecasts = []
        try:
            if len(series) >= MIN_DECOMP:
                model = ExponentialSmoothing(
                    series, trend='add', seasonal='add', seasonal_periods=12
                ).fit(optimized=True)
            else:
                model = ExponentialSmoothing(
                    series, trend='add', seasonal=None
                ).fit(optimized=True)
            pred = model.forecast(steps)
            last_dt = series.index[-1]
            for i, val in enumerate(pred):
                fdt = last_dt + pd.DateOffset(months=i+1)
                forecasts.append({
                    'month_year': fdt.strftime('%b-%Y'),
                    'forecast':   safe_float(val),
                    'type':       'forecast',
                })
        except Exception as e:
            # Fallback: linear extrapolation
            trend = linear_trend(series)
            last_val = float(series.iloc[-1])
            slope    = trend['slope'] or 0
            last_dt  = series.index[-1]
            for i in range(1, steps + 1):
                fdt = last_dt + pd.DateOffset(months=i)
                forecasts.append({
                    'month_year': fdt.strftime('%b-%Y'),
                    'forecast':   safe_float(last_val + slope * i),
                    'type':       'linear_extrapolation',
                })
        return forecasts

    def analyse_series(series, label):
        out = {'label': label, 'n_months': int(len(series))}
        if len(series) < MIN_OBS:
            out['error'] = f'Not enough data ({len(series)} months, need {MIN_OBS})'
            return out

        out['trend']          = linear_trend(series)
        out['moving_average'] = moving_average(series)
        out['actual']         = [{'month_year': k.strftime('%b-%Y'), 'value': safe_float(v)}
                                  for k, v in series.items()]

        # Decomposition — only if enough observations
        residuals = pd.Series(dtype=float)
        if len(series) >= MIN_DECOMP:
            try:
                decomp = seasonal_decompose(series, model='additive', period=12, extrapolate_trend='freq')
                out['decomposition'] = {
                    'trend':    [{'month_year': k.strftime('%b-%Y'), 'value': safe_float(v)}
                                  for k, v in decomp.trend.dropna().items()],
                    'seasonal': [{'month_year': k.strftime('%b-%Y'), 'value': safe_float(v)}
                                  for k, v in decomp.seasonal.dropna().items()],
                    'residual': [{'month_year': k.strftime('%b-%Y'), 'value': safe_float(v)}
                                  for k, v in decomp.resid.dropna().items()],
                }
                residuals = decomp.resid.dropna()
            except Exception:
                pass
        else:
            # Use residuals from linear detrending instead
            x = np.arange(len(series))
            slope, intercept, *_ = stats.linregress(x, series.values)
            fitted    = pd.Series(intercept + slope * x, index=series.index)
            residuals = series - fitted

        if not residuals.empty:
            out['anomalies'] = detect_anomalies(series, residuals)

        out['forecast'] = forecast_series(series)
        return out

    # Build series
    gto_series      = build_monthly_series(gto,      'gto_amount')
    campaign_series = build_monthly_series(campaign,  'receipt_no') \
        if 'receipt_no' in campaign.columns else \
        build_monthly_series(campaign, 'amount')

    if not gto_series.empty:
        result['gto_trend'] = analyse_series(gto_series, 'GTO Revenue')

    if not campaign_series.empty:
        result['campaign_trend'] = analyse_series(campaign_series, 'Campaign Activity')

    # Cross-series: lead-lag correlation (does campaign activity lead GTO by 1-2 months?)
    if len(gto_series) >= MIN_OBS and len(campaign_series) >= MIN_OBS:
        combined = pd.DataFrame({'gto': gto_series, 'campaign': campaign_series}).dropna()
        if len(combined) >= MIN_OBS:
            lead_lag = {}
            for lag in range(0, 4):
                shifted = combined['campaign'].shift(lag)
                valid   = pd.concat([shifted, combined['gto']], axis=1).dropna()
                if len(valid) >= 4:
                    c, p = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
                    lead_lag[f'lag_{lag}'] = {
                        'correlation': safe_float(c),
                        'p_value':     safe_float(p),
                        'significant': bool(p < 0.05),
                        'label':       f'Campaign → GTO ({lag} month lag)',
                    }
            result['lead_lag'] = lead_lag

            # Best lag
            best = max(lead_lag.items(),
                       key=lambda x: abs(x[1]['correlation'] or 0))
            result['summary'] = {
                'best_lag':          best[0],
                'best_correlation':  best[1]['correlation'],
                'interpretation': (
                    f"Campaign activity best predicts GTO revenue at a "
                    f"{best[0].replace('lag_', '')}-month lag "
                    f"(r={best[1]['correlation']:.2f}, "
                    f"{'significant' if best[1]['significant'] else 'not significant'})."
                ),
            }

    return result


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        DATA_FOLDER     = sys.argv[1]
        COMBINED_FOLDER = sys.argv[2]
        # shop_mapping passed as optional 3rd arg from main.py
        MAPPING_FILE    = sys.argv[3] if len(sys.argv) > 3 else ''
    else:
        _config_file = sys.argv[1] if len(sys.argv) == 2 else "config_Keith.xlsx"
        _script_dir  = Path(__file__).resolve().parent
        _paths_df    = pd.read_excel(_script_dir / _config_file, sheet_name='paths')
        _config      = dict(zip(_paths_df['Setting'].astype(str).str.strip(), _paths_df['Value']))
        DATA_FOLDER     = str(_config.get('cleaned_data',  '')).strip()
        COMBINED_FOLDER = str(_config.get('combined_data', '')).strip()
        MAPPING_FILE    = str(_config.get('shop_mapping',  '')).strip()
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

    # ----------------------------
    # Filter campaign to matched shops only (exact, fuzzy, confirmed)
    # Excludes unmatched outlets and gto_only rows from all downstream analysis
    # ----------------------------
    MATCHED_METHODS = {'exact', 'fuzzy', 'confirmed'}
    if MAPPING_FILE and os.path.exists(MAPPING_FILE):
        try:
            mapping_df = pd.read_excel(MAPPING_FILE, sheet_name='mapping')
            mapping_df['campaign_name'] = mapping_df['campaign_name'].astype(str).str.strip().str.lower()
            mapping_df['method']        = mapping_df['method'].astype(str).str.strip().str.lower()
            matched_names = set(
                mapping_df.loc[mapping_df['method'].isin(MATCHED_METHODS), 'campaign_name']
            )
            if 'outlet_name' in campaign.columns:
                before = len(campaign)
                campaign = campaign[
                    campaign['outlet_name'].str.strip().str.lower().isin(matched_names)
                ].copy()
                after = len(campaign)
                print(f"✅ Filtered to matched shops only: {after:,} rows kept, "
                      f"{before - after:,} unmatched rows excluded.")
            else:
                print("⚠️  outlet_name column not found in campaign — skipping match filter.")
        except Exception as e:
            print(f"⚠️  Could not load shop mapping for filtering: {e}")
    else:
        print("⚠️  shop_mapping not found — analysis includes all campaign rows.")

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
    ts_results       = analyse_time_series(cdf, gto_analysis)

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
            ('TS_GTO_Forecast',    ts_results.get('gto_trend', {}).get('forecast')),
            ('TS_Campaign_Forecast', ts_results.get('campaign_trend', {}).get('forecast')),
            ('TS_Anomalies_GTO',   ts_results.get('gto_trend', {}).get('anomalies')),
        ]
        for sheet_name, records in sheets:
            if records:
                pd.DataFrame(records).to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"  ✅ Saved: {report_path}")

    # ----------------------------
    # Step 6: Save insights.json — keys MUST match app.py tab_insights expectations
    # ----------------------------
    insights_payload = {
        'mom_trends':      mom_results,
        'campaign_roi':    roi_results,
        'campaign_type':   type_results,
        'loyalty_points':  loyalty_results,
        'tenant_turnover': turnover_results,
        'time_series':     ts_results,       # keys: gto_trend, campaign_trend, lead_lag, summary
    }

    json_path = os.path.join(COMBINED_FOLDER, 'insights.json')
    with open(json_path, 'w') as f:
        json.dump(insights_payload, f, indent=2, default=str)
    print(f"  ✅ Saved: {json_path}")

    print("\n" + "="*60)
    print("✅ REGRESSION COMPLETED — outputs saved to:")
    print(f"   {COMBINED_FOLDER}")
    print("="*60)