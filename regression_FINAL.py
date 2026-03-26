"""
regression.py
-------------
Time Series Analysis — Y = Amount (member spend)
  Dummy var 1: Months (seasonality)

Also computes:
  - Monthly trend (actual + 3-month forecast)
  - Moving average
  - Anomaly detection
  - Lead-lag correlation (campaign activity → amount)

Usage:
  python3 regression.py <cleaned_data> <combined_data> [shop_mapping]
  python3 regression.py       # uses config_Keith.xlsx
"""

import warnings
warnings.filterwarnings('ignore')

import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def safe_float(val):
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except:
        return None

def to_records(df):
    if df is None or df.empty: return []
    return json.loads(df.replace([np.inf, -np.inf], np.nan).to_json(orient='records'))

def month_order(s):
    return pd.to_datetime(s, format='%b-%Y', errors='coerce')


# ─────────────────────────────────────────────────────────────────────────────
# Time Series: Y = Amount, X = Month dummies
# ─────────────────────────────────────────────────────────────────────────────
def analyse_time_series(campaign: pd.DataFrame) -> dict:
    """
    Time series analysis on monthly Amount (member spend).
    
    1. Monthly aggregation of Amount
    2. Month dummies regression (OLS) — which months are significantly higher/lower?
    3. Trend analysis (linear regression on time index)
    4. 3-month moving average
    5. 3-month ahead forecast (Holt-Winters or linear extrapolation)
    6. Anomaly detection
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    result = {}
    MIN_OBS = 6

    if 'amount' not in campaign.columns or 'month_year' not in campaign.columns:
        return {'error': 'amount or month_year column missing'}

    # ── Monthly aggregation ───────────────────────────────────────────────
    camp = campaign.copy()
    camp['amount'] = pd.to_numeric(camp['amount'], errors='coerce')

    monthly = (camp.groupby('month_year')
                   .agg(
                       total_amount  =('amount', 'sum'),
                       avg_amount    =('amount', 'mean'),
                       txn_count     =('receipt_no', 'nunique') if 'receipt_no' in camp.columns
                                      else ('amount', 'count'),
                       redemptions   =('receipt_no', 'nunique') if 'receipt_no' in camp.columns
                                      else ('amount', 'count'),
                   )
                   .reset_index())

    monthly['sort_key'] = month_order(monthly['month_year'])
    monthly = monthly.sort_values('sort_key').reset_index(drop=True)

    if len(monthly) < MIN_OBS:
        return {'error': f"Only {len(monthly)} months of data — need ≥{MIN_OBS}"}

    result['monthly_amount'] = to_records(monthly.drop(columns='sort_key'))

    # ── Month dummies regression (OLS) ────────────────────────────────────
    # Which months are significantly different from the base month?
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler

    monthly['_month_num'] = monthly['sort_key'].dt.month
    month_dummies = pd.get_dummies(monthly['_month_num'], prefix='month', drop_first=True)
    X = sm.add_constant(month_dummies.astype(float))
    y = monthly['total_amount'].astype(float)

    if len(y) >= MIN_OBS and X.shape[1] > 1:
        try:
            ols = sm.OLS(y, X).fit()
            coef_df = pd.DataFrame({
                'month':       ['const'] + [c for c in month_dummies.columns],
                'coef':        ols.params.values,
                'p_value':     ols.pvalues.values,
                'significant': (ols.pvalues.values < 0.05),
            })
            result['month_dummies_regression'] = {
                'coef_table':    to_records(coef_df),
                'r_squared':     safe_float(ols.rsquared),
                'adj_r_squared': safe_float(ols.rsquared_adj),
                'f_pvalue':      safe_float(ols.f_pvalue),
                'n_obs':         int(ols.nobs),
                'insight': (
                    f"Month dummies explain {ols.rsquared*100:.1f}% of variance in monthly spend "
                    f"(Adj-R²={ols.rsquared_adj:.3f}). "
                    + ("Significant months: " +
                       ", ".join(coef_df[coef_df['significant'] & (coef_df['month'] != 'const')]['month'].tolist())
                       if any(coef_df[coef_df['month'] != 'const']['significant'])
                       else "No months significantly different from base at p<0.05.")
                ),
                'base_month': 'January (month=1)',
            }
        except Exception as e:
            result['month_dummies_regression'] = {'error': str(e)}

    # ── MoM % change ──────────────────────────────────────────────────────
    monthly['mom_pct_change'] = monthly['total_amount'].pct_change().mul(100).round(2)
    result['mom_trends'] = to_records(monthly[['month_year','total_amount','avg_amount',
                                                'txn_count','mom_pct_change']].drop(columns=['sort_key'], errors='ignore'))

    # ── Linear trend ──────────────────────────────────────────────────────
    x_idx = np.arange(len(monthly))
    slope, intercept, r, p, _ = stats.linregress(x_idx, monthly['total_amount'].values)
    result['trend'] = {
        'slope':      safe_float(slope),
        'r_squared':  safe_float(r**2),
        'p_value':    safe_float(p),
        'direction':  'upward' if slope > 0 else 'downward',
        'strength':   'strong' if abs(r) > 0.7 else ('moderate' if abs(r) > 0.4 else 'weak'),
        'significant': bool(p < 0.05),
    }

    # ── 3-month moving average ────────────────────────────────────────────
    monthly['ma3'] = monthly['total_amount'].rolling(3, min_periods=1).mean().round(2)
    result['moving_average'] = to_records(monthly[['month_year', 'ma3']])

    # ── Actual data points ────────────────────────────────────────────────
    result['actual'] = [
        {'month_year': r['month_year'], 'value': safe_float(r['total_amount'])}
        for _, r in monthly.iterrows()
    ]

    # ── Forecast (3 months ahead) ─────────────────────────────────────────
    series = monthly.set_index('sort_key')['total_amount'].astype(float)
    series.index = pd.DatetimeIndex(series.index).to_period('M').to_timestamp()
    forecasts = []
    try:
        if len(series) >= 12:
            mdl = ExponentialSmoothing(series, trend='add', seasonal='add',
                                       seasonal_periods=12).fit(optimized=True)
        else:
            mdl = ExponentialSmoothing(series, trend='add', seasonal=None).fit(optimized=True)
        pred    = mdl.forecast(3)
        last_dt = series.index[-1]
        for i, val in enumerate(pred):
            fdt = last_dt + pd.DateOffset(months=i+1)
            forecasts.append({'month_year': fdt.strftime('%b-%Y'),
                               'forecast': safe_float(val), 'type': 'forecast'})
    except:
        # Fallback: linear extrapolation
        last_val = float(series.iloc[-1])
        last_dt  = series.index[-1]
        for i in range(1, 4):
            fdt = last_dt + pd.DateOffset(months=i)
            forecasts.append({'month_year': fdt.strftime('%b-%Y'),
                               'forecast': safe_float(last_val + slope * i),
                               'type': 'linear_extrapolation'})
    result['forecast'] = forecasts

    # ── Anomaly detection ─────────────────────────────────────────────────
    residuals  = monthly['total_amount'] - (intercept + slope * x_idx)
    std_resid  = residuals.std()
    mean_resid = residuals.mean()
    anomalies  = []
    for i, (_, row) in enumerate(monthly.iterrows()):
        res = residuals.iloc[i]
        if abs(res - mean_resid) > 2 * std_resid:
            anomalies.append({
                'month_year': row['month_year'],
                'actual':     safe_float(row['total_amount']),
                'residual':   safe_float(res),
                'direction':  'above' if res > mean_resid else 'below',
            })
    result['anomalies'] = anomalies

    # ── Campaign type breakdown ───────────────────────────────────────────
    if 'campaign_source' in campaign.columns:
        by_source = (camp.groupby(['month_year', 'campaign_source'])['amount']
                         .sum().reset_index())
        by_source['sort_key'] = month_order(by_source['month_year'])
        by_source = by_source.sort_values('sort_key').drop(columns='sort_key')
        result['amount_by_source'] = to_records(by_source)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ROI summary (campaign cost vs GTO rent)
# ─────────────────────────────────────────────────────────────────────────────
def analyse_roi(campaign: pd.DataFrame, gto: pd.DataFrame) -> dict:
    """Simple ROI: total_amount redeemed vs GTO rent per month."""
    result = {}

    if 'month_year' not in campaign.columns or 'month_year' not in gto.columns:
        return result

    camp_monthly = (campaign.groupby('month_year')
                            .agg(total_amount=('amount', 'sum'),
                                 redemptions=('receipt_no', 'nunique')
                                 if 'receipt_no' in campaign.columns
                                 else ('amount', 'count'))
                            .reset_index())

    gto_monthly = (gto.groupby('month_year')['gto_rent']
                      .sum().reset_index()
                      .rename(columns={'gto_rent': 'total_gto_rent'}))

    merged = pd.merge(camp_monthly, gto_monthly, on='month_year', how='outer').fillna(0)
    merged['sort_key'] = month_order(merged['month_year'])
    merged = merged.sort_values('sort_key').drop(columns='sort_key').reset_index(drop=True)

    result['monthly_roi'] = to_records(merged)

    valid = merged[merged['total_amount'] > 0]
    if not valid.empty:
        result['summary'] = {
            'total_amount_redeemed': safe_float(merged['total_amount'].sum()),
            'total_gto_rent':        safe_float(merged['total_gto_rent'].sum()),
            'avg_monthly_amount':    safe_float(merged['total_amount'].mean()),
            'avg_monthly_gto_rent':  safe_float(merged['total_gto_rent'].mean()),
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(cleaned_folder: str, combined_folder: str, mapping_file: str = ''):
    print("\n" + "="*60)
    print("TIME SERIES & ROI ANALYSIS STARTING")
    print("="*60)

    camp_path = os.path.join(cleaned_folder, 'campaign_all.csv')
    if not os.path.exists(camp_path):
        print(f"❌ campaign_all.csv not found: {camp_path}")
        sys.exit(1)

    campaign = pd.read_csv(camp_path)
    campaign['amount'] = pd.to_numeric(campaign.get('amount', pd.Series()), errors='coerce')
    print(f"  Campaign: {len(campaign):,} rows")

    # GTO rent
    gto = pd.DataFrame()
    for f in os.listdir(cleaned_folder):
        if 'gto_monthly_rent' in f.lower() and f.endswith(('.xlsx', '.csv')):
            path = os.path.join(cleaned_folder, f)
            gto  = pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)
            break
    if not gto.empty:
        print(f"  GTO rent: {len(gto):,} rows")

    # Run analyses
    ts_result  = analyse_time_series(campaign)
    roi_result = analyse_roi(campaign, gto) if not gto.empty else {}

    # Save insights.json
    insights = {
        'time_series': ts_result,
        'roi':         roi_result,
    }
    os.makedirs(combined_folder, exist_ok=True)
    json_path = os.path.join(combined_folder, 'insights.json')
    with open(json_path, 'w') as f:
        json.dump(insights, f, indent=2, default=str)
    print(f"\n✅ Saved: {json_path}")

    # Save Excel report
    report_path = os.path.join(combined_folder, 'insights_report.xlsx')
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        sheets = [
            ('Monthly_Amount',    ts_result.get('monthly_amount')),
            ('Month_Regression',  ts_result.get('month_dummies_regression', {}).get('coef_table')),
            ('MoM_Trends',        ts_result.get('mom_trends')),
            ('Forecast',          ts_result.get('forecast')),
            ('Anomalies',         ts_result.get('anomalies')),
            ('Amount_by_Source',  ts_result.get('amount_by_source')),
            ('ROI_Monthly',       roi_result.get('monthly_roi')),
        ]
        for sheet_name, records in sheets:
            if records:
                pd.DataFrame(records).to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"✅ Saved: {report_path}")

    # Print summary
    trend = ts_result.get('trend', {})
    print(f"\n  Trend: {trend.get('direction','—')} ({trend.get('strength','—')}), "
          f"R²={trend.get('r_squared','—')}, p={trend.get('p_value','—')}")
    if 'month_dummies_regression' in ts_result:
        mdr = ts_result['month_dummies_regression']
        if 'insight' in mdr:
            print(f"  Month seasonality: {mdr['insight']}")

    print("\n" + "="*60)
    print("✅ REGRESSION (TIME SERIES) COMPLETED")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else '')
    else:
        config_file = sys.argv[1] if len(sys.argv) == 2 else "config_Keith.xlsx"
        script_dir  = Path(__file__).resolve().parent
        df          = pd.read_excel(script_dir / config_file, sheet_name='paths')
        cfg         = dict(zip(df['Setting'].astype(str).str.strip(), df['Value'].astype(str).str.strip()))
        main(cfg.get('cleaned_data',''), cfg.get('combined_data',''), cfg.get('shop_mapping',''))