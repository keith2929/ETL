"""
linear_regression.py
--------------------
Regression 1 — What drives transaction Amount at outlet level?
  Unit : outlet × month
  Y    = amount (total spend per outlet per month)
  X1   = is_brand (Mall=0, Brand=1)
  X2   = voucher_code dummies (top 30 by frequency)


No GTO data used.

Usage:
  python3 linear_regression.py <cleaned_data> <combined_data> [shop_mapping]
  python3 linear_regression.py        # uses config_Kim.xlsx
"""

import warnings
warnings.filterwarnings('ignore')

import os, sys, json
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MIN_ROWS    = 10
MAX_DUMMIES = 30   # max voucher code dummies


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


# ─────────────────────────────────────────────────────────────────────────────
# OLS engine
# ─────────────────────────────────────────────────────────────────────────────
def run_ols(df: pd.DataFrame, target: str, features: list, label: str = '') -> dict:
    result = {'target': target, 'features': features, 'label': label}

    subset = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < MIN_ROWS:
        result['error'] = f"Too few rows ({len(subset)}) — need at least {MIN_ROWS}"
        return result

    X_raw = subset[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y     = subset[target].astype(float)

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=features, index=X_raw.index)
    X_const  = sm.add_constant(X_scaled)

    model = sm.OLS(y, X_const).fit()

    coef_df = pd.DataFrame({
        'feature':     ['const'] + features,
        'coef':        model.params.values,
        'std_err':     model.bse.values,
        't_stat':      model.tvalues.values,
        'p_value':     model.pvalues.values,
        'ci_lower':    model.conf_int()[0].values,
        'ci_upper':    model.conf_int()[1].values,
        'significant': (model.pvalues.values < 0.05),
    })
    result['coef_table'] = to_records(coef_df)

    result['model_fit'] = {
        'r_squared':     safe_float(model.rsquared),
        'adj_r_squared': safe_float(model.rsquared_adj),
        'f_statistic':   safe_float(model.fvalue),
        'f_pvalue':      safe_float(model.f_pvalue),
        'n_obs':         int(model.nobs),
        'aic':           safe_float(model.aic),
        'bic':           safe_float(model.bic),
        'dw_stat':       safe_float(sm.stats.stattools.durbin_watson(model.resid)),
    }

    # VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    try:
        result['vif'] = to_records(pd.DataFrame({
            'feature': features,
            'vif': [safe_float(variance_inflation_factor(X_const.values, i + 1))
                    for i in range(len(features))]
        }))
    except:
        result['vif'] = []

    # 5-fold CV
    try:
        n_splits = min(5, len(subset))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rmse_list, mae_list, r2_list = [], [], []
        Xn, yn = X_scaled.values, y.values
        for tr, te in kf.split(Xn):
            m    = sm.OLS(yn[tr], sm.add_constant(Xn[tr])).fit()
            pred = m.predict(sm.add_constant(Xn[te]))
            rmse_list.append(mean_squared_error(yn[te], pred, squared=False))
            mae_list.append(mean_absolute_error(yn[te], pred))
            r2_list.append(r2_score(yn[te], pred))
        result['cross_validation'] = {
            'cv_rmse_mean': safe_float(np.mean(rmse_list)),
            'cv_rmse_std':  safe_float(np.std(rmse_list)),
            'cv_mae_mean':  safe_float(np.mean(mae_list)),
            'cv_r2_mean':   safe_float(np.mean(r2_list)),
        }
    except Exception as e:
        result['cross_validation'] = {'error': str(e)}

    # Residuals
    result['residuals'] = to_records(pd.DataFrame({
        'fitted':    model.fittedvalues.values,
        'residual':  model.resid.values,
        'std_resid': (model.resid / (model.resid.std() or 1)).values,
    }))

    # Insight
    sig = coef_df[(coef_df['significant']) & (coef_df['feature'] != 'const')]
    parts = [
        f"{r['feature']} {'↑' if r['coef'] > 0 else '↓'} "
        f"(β={r['coef']:.3f}, p={r['p_value']:.3f})"
        for _, r in sig.iterrows()
    ]
    result['insight'] = (
        f"R²={model.rsquared:.3f}, Adj-R²={model.rsquared_adj:.3f}, "
        f"n={int(model.nobs)}. "
        + ("Significant: " + "; ".join(parts) if parts
           else "No significant predictors at p<0.05.")
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Regression 1
#Y = amount (per receipt)
#X = voucher_code dummies 
 #   is_brand
#n = total receipt no
# ─────────────────────────────────────────────────────────────────────────────
def regression_1(campaign: pd.DataFrame) -> dict:
    print("\n  ── Regression 1: Y = Amount (per receipt, by campaign) ──")

    if 'amount' not in campaign.columns:
        return {'error': 'amount column not found'}
    if 'voucher_code' not in campaign.columns:
        return {'error': 'voucher_code column not found'}

    df = campaign.copy()
    df['amount']       = pd.to_numeric(df['amount'], errors='coerce')
    df['voucher_code'] = df['voucher_code'].astype(str).str.strip()
    df = df.dropna(subset=['amount']).reset_index(drop=True)

    if len(df) < MIN_ROWS:
        return {'error': f"Too few rows ({len(df)})"}

    print(f"    Dataset: {len(df):,} receipt rows")

    # X1: is_brand
    df['is_brand'] = (df['campaign_source'].astype(str).str.lower() == 'brand').astype(int)

    # X2: voucher_code dummies
    # base = (drop_first=True)
    top_codes = df['voucher_code'].value_counts().head(MAX_DUMMIES).index.tolist()
    vdummies  = pd.get_dummies(
        df['voucher_code'].where(df['voucher_code'].isin(top_codes), other='other'),
        prefix='camp', drop_first=True
    )

    df_reg = pd.concat([df[['amount', 'is_brand']].reset_index(drop=True),
                        vdummies.reset_index(drop=True)], axis=1)

    camp_cols = [c for c in df_reg.columns if c.startswith('camp_')]
    features  = ['is_brand'] + camp_cols
    features  = [f for f in features if pd.api.types.is_numeric_dtype(df_reg[f])]

    print(f"    Features: 1 control + {len(camp_cols)} campaign dummies = {len(features)} total")

    result = run_ols(df_reg, 'amount', features,
                     label='Regression 1: Y=Amount (per receipt, by campaign)')
    result['n_rows'] = len(df_reg)
    result['feature_groups'] = {
        'control':  ['is_brand'],
        'campaign': camp_cols,
    }

    # ── Campaign summary table  ─────────────────────────────────────
    if 'receipt_no' in df.columns:
        agg = (df.groupby(['voucher_code', 'is_brand'])
                 .agg(total_amount=('amount',     'sum'),
                      avg_amount  =('amount',     'mean'),
                      n_receipts  =('receipt_no', 'nunique'))
                 .reset_index()
                 .sort_values('total_amount', ascending=False))
    else:
        agg = (df.groupby(['voucher_code', 'is_brand'])
                 .agg(total_amount=('amount', 'sum'),
                      avg_amount  =('amount', 'mean'),
                      n_receipts  =('amount', 'count'))
                 .reset_index()
                 .sort_values('total_amount', ascending=False))

    result['campaign_summary'] = to_records(agg)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def build_summary(reg1: dict) -> list:
    rows = []
    for key, res in [('regression_1_amount', reg1)]:
        if 'error' in res:
            rows.append({'model_key': key, 'error': res.get('error')})
            continue
        fit = res.get('model_fit', {})
        cv  = res.get('cross_validation', {})
        rows.append({
            'model_key':     key,
            'target':        res.get('target', ''),
            'label':         res.get('label', ''),
            'n_obs':         fit.get('n_obs'),
            'r_squared':     fit.get('r_squared'),
            'adj_r_squared': fit.get('adj_r_squared'),
            'f_pvalue':      fit.get('f_pvalue'),
            'cv_r2':         cv.get('cv_r2_mean'),
            'cv_rmse':       cv.get('cv_rmse_mean'),
            'insight':       res.get('insight', ''),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(cleaned_folder: str, combined_folder: str, mapping_file: str = ''):
    print("\n" + "="*60)
    print("LINEAR REGRESSION STARTING")
    print("="*60)

    camp_path = os.path.join(cleaned_folder, 'campaign_all.csv')
    if not os.path.exists(camp_path):
        print(f"❌ campaign_all.csv not found in {cleaned_folder}")
        sys.exit(1)

    campaign = pd.read_csv(camp_path)
    print(f"  Campaign loaded: {len(campaign):,} rows")

    # Run regressions
    reg1 = regression_1(campaign)

    # Output
    output = {
        'linear_regression': {
            'regression_1_amount': reg1,
            'summary':             build_summary(reg1),
        }
    }

    # Print summary
    for name, res in [('Regression 1', reg1)]:
        if 'error' in res:
            print(f"\n  ⚠️  {name}: {res['error']}")
        else:
            fit = res.get('model_fit', {})
            print(f"\n  {name}: R²={fit.get('r_squared','—')}, "
                  f"Adj-R²={fit.get('adj_r_squared','—')}, "
                  f"p={fit.get('f_pvalue','—')}, n={fit.get('n_obs','—')}")
            print(f"  {res.get('insight','')}")

    # Save JSON
    os.makedirs(combined_folder, exist_ok=True)
    json_path = os.path.join(combined_folder, 'linear_regression_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✅ Saved: {json_path}")

    # Save summary Excel
    summary_df = pd.DataFrame(output['linear_regression']['summary'])
    if not summary_df.empty:
        xlsx_path = os.path.join(combined_folder, 'linear_regression_summary.xlsx')
        summary_df.to_excel(xlsx_path, index=False)
        print(f"✅ Saved: {xlsx_path}")

    print("\n" + "="*60)
    print("✅ LINEAR REGRESSION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else '')
    else:
        config_file = sys.argv[1] if len(sys.argv) == 2 else "config_Kim.xlsx"
        script_dir  = Path(__file__).resolve().parent
        df          = pd.read_excel(script_dir / config_file, sheet_name='paths')
        cfg         = dict(zip(df['Setting'].astype(str).str.strip(), df['Value'].astype(str).str.strip()))
        main(cfg.get('cleaned_data',''), cfg.get('combined_data',''), cfg.get('shop_mapping',''))