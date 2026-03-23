"""
linear_regression.py
--------------------
Linear Regression analysis for GTO Profitability & Marketing ROI.
 
Targets (Y):
  - gto_amount  : GTO Revenue
  - gto_rent    : GTO Rent
 
Features (X candidates, auto-selected based on availability):
  - points_earned / points_issued   : loyalty points awarded
  - amount / txn_amount             : transaction spend
  - redemptions                     : unique receipt count
  - campaign_cost                   : points × POINTS_COST_SGD
  - nla_sqft                        : net lettable area (size control)
  - campaign_source_encoded         : brand vs mall (dummy)
 
Usage:
  python linear_regression.py <cleaned_data_folder> <combined_data_folder> [shop_mapping]
  python linear_regression.py                        # uses config_Kim.xlsx
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
 
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
 
warnings.filterwarnings('ignore')
 
# ─── Constants ────────────────────────────────────────────────────────────────
POINTS_COST_SGD = 0.20   # SGD cost per loyalty point issued
TARGETS         = ['gto_amount', 'gto_rent']
MIN_ROWS        = 10     # minimum rows needed to fit a model
 
# ─── Utility helpers ──────────────────────────────────────────────────────────
def safe_float(val):
    """Convert to float; return None on NaN/Inf."""
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None
 
def df_to_records_safe(df: pd.DataFrame) -> list:
    """DataFrame → JSON-safe records (NaN/Inf replaced with None)."""
    if df is None or df.empty:
        return []
    return json.loads(
        df.replace([np.inf, -np.inf], np.nan).to_json(orient='records')
    )
 
def find_col(df: pd.DataFrame, candidates: list) -> str | None:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
 
# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data(data_folder: str, mapping_file: str = '') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load campaign_all and GTO rent data from the cleaned data folder.
    Returns (campaign_df, gto_df).
    """
    # ── Campaign ──
    camp_path = os.path.join(data_folder, 'campaign_all.csv')
    if not os.path.exists(camp_path):
        raise FileNotFoundError(f"campaign_all.csv not found in {data_folder}")
    campaign = pd.read_csv(camp_path)
 
    # Parse date columns
    for col in campaign.columns:
        if 'date' in col.lower():
            campaign[col] = pd.to_datetime(campaign[col], errors='coerce')
 
    # Build month_year
    if 'month_year' not in campaign.columns:
        if 'month' in campaign.columns and 'year' in campaign.columns:
            campaign['month_year'] = (
                campaign['month'].astype(str) + '-' + campaign['year'].astype(str)
            )
 

 
    # ── GTO rent ──
    gto_path = ''
    for f in os.listdir(data_folder):
        if 'gto_monthly_rent' in f.lower() and f.endswith(('.xlsx', '.csv')):
            gto_path = os.path.join(data_folder, f)
            break
    if not gto_path:
        raise FileNotFoundError("No gto_monthly_rent file found in " + data_folder)
 
    gto = (pd.read_excel(gto_path) if gto_path.endswith('.xlsx')
           else pd.read_csv(gto_path))
    # ── Filter to matched shops ─
    VALID_METHODS = {
    'exact', 'fuzzy', 'confirmed',
    'code_match', 'combined_exact', 'combined_fuzzy'
    }

    if mapping_file and os.path.exists(mapping_file):
        try:
            mdf = pd.read_excel(mapping_file, sheet_name='mapping')

            mdf['campaign_name'] = mdf['campaign_name'].astype(str).str.strip().str.lower()
            mdf['gto_name']      = mdf['gto_name'].astype(str).str.strip().str.lower()
            mdf['method']        = mdf['method'].astype(str).str.strip().str.lower()

            valid_map = mdf[mdf['method'].isin(VALID_METHODS)]

            valid_campaign = set(valid_map['campaign_name'])
            valid_gto      = set(valid_map['gto_name'])

            # campaign filter
            if 'outlet_name' in campaign.columns:
                campaign = campaign[
                    campaign['outlet_name'].str.strip().str.lower().isin(valid_campaign)
                ].copy()

            # ── GTO filter  ──
            if 'shop_name' in gto.columns:
                gto = gto[
                    gto['shop_name'].str.strip().str.lower().isin(valid_gto)
                ].copy()

        except Exception as e:
            print(f"⚠️ mapping filter failed: {e}")

    # Build month_year for GTO
    if 'month_year' not in gto.columns and 'gto_reporting_month' in gto.columns:
        gto['month_year'] = pd.to_datetime(
            gto['gto_reporting_month'], errors='coerce'
        ).dt.strftime('%b-%Y')
 
    return campaign, gto
 
 
# ─── Feature Engineering ──────────────────────────────────────────────────────
def build_regression_dataset(campaign: pd.DataFrame,
                              gto: pd.DataFrame,
                              level: str = 'monthly') -> pd.DataFrame:
    """
    Merge campaign activity with GTO financials.
 
    level : 'monthly'  → aggregate by month_year
            'outlet'   → aggregate by outlet/shop
            'panel'    → outlet × month_year panel
    """
    print(f"\n  Building regression dataset at level='{level}' ...")
 
    # ── Standardise column names ──
    pts_col    = find_col(campaign, ['points_earned', 'points_issued'])
    amt_col    = find_col(campaign, ['amount', 'txn_amount'])
    rcpt_col   = find_col(campaign, ['receipt_no'])
    outlet_col = find_col(campaign, ['outlet_name', 'final_gto_name'])
    shop_col   = find_col(gto, ['shop_name'])
    gto_a_col  = find_col(gto, ['gto_amount'])
    gto_r_col  = find_col(gto, ['gto_rent'])
 
    # ── Aggregate campaign ──
    agg_dict = {}
    if pts_col:    agg_dict['points_issued']  = (pts_col,  'sum')
    if amt_col:    agg_dict['txn_amount']      = (amt_col,  'sum')
    if rcpt_col:   agg_dict['redemptions']     = (rcpt_col, 'nunique')
 
    if level == 'monthly':
        if 'month_year' not in campaign.columns:
            print("  ⚠️  month_year missing — cannot build monthly dataset.")
            return pd.DataFrame()
        camp_grp = campaign.groupby('month_year').agg(**agg_dict).reset_index()
        gto_grp  = gto.groupby('month_year').agg(
            gto_amount=(gto_a_col, 'sum') if gto_a_col else ('month_year', 'count'),
            gto_rent  =(gto_r_col, 'sum') if gto_r_col else ('month_year', 'count'),
        ).reset_index()
        df = pd.merge(camp_grp, gto_grp, on='month_year', how='inner')
 
    elif level == 'outlet':
        if not outlet_col or not shop_col:
            print("  ⚠️  outlet_name / shop_name missing — cannot build outlet dataset.")
            return pd.DataFrame()
        camp_grp = campaign.groupby(outlet_col).agg(**agg_dict).reset_index()
        camp_grp = camp_grp.rename(columns={outlet_col: 'shop_name'})
        gto_grp  = gto.groupby(shop_col).agg(
            gto_amount=(gto_a_col, 'sum') if gto_a_col else (shop_col, 'count'),
            gto_rent  =(gto_r_col, 'sum') if gto_r_col else (shop_col, 'count'),
        ).reset_index().rename(columns={shop_col: 'shop_name'})
        df = pd.merge(camp_grp, gto_grp, on='shop_name', how='inner')
 
    elif level == 'panel':
        if not outlet_col or not shop_col or 'month_year' not in campaign.columns:
            print("  ⚠️  Missing columns for panel dataset.")
            return pd.DataFrame()
        camp_grp = campaign.groupby([outlet_col, 'month_year']).agg(**agg_dict).reset_index()
        camp_grp = camp_grp.rename(columns={outlet_col: 'shop_name'})
        gto_grp  = gto[[shop_col, 'month_year'] +
                        ([gto_a_col] if gto_a_col else []) +
                        ([gto_r_col] if gto_r_col else [])].rename(columns={shop_col: 'shop_name'})
        df = pd.merge(camp_grp, gto_grp, on=['shop_name', 'month_year'], how='inner')
        # Size control
        nla_col = find_col(gto, ['nla_sqft'])
        if nla_col:
            nla = gto[[shop_col, nla_col]].drop_duplicates().rename(
                columns={shop_col: 'shop_name', nla_col: 'nla_sqft'})
            df = df.merge(nla, on='shop_name', how='left')
    else:
        raise ValueError(f"Unknown level: {level}")
 
    # ── Derived features ──
    if 'points_issued' in df.columns:
        df['campaign_cost'] = df['points_issued'] * POINTS_COST_SGD
 
    # Campaign source dummy (brand vs mall)
    src_col = find_col(campaign, ['campaign_source', 'campaign_type'])
    if src_col and level in ('monthly', 'panel'):
        type_map = campaign.groupby(
            'month_year' if level == 'monthly' else [outlet_col, 'month_year']
        )[src_col].agg(lambda x: x.mode()[0] if not x.empty else '').reset_index()
        if level == 'monthly':
            df = df.merge(type_map, on='month_year', how='left')
        else:
            type_map = type_map.rename(columns={outlet_col: 'shop_name'})
            df = df.merge(type_map, on=['shop_name', 'month_year'], how='left')
        if src_col in df.columns:
            df['is_brand_campaign'] = (
                df[src_col].astype(str).str.lower().str.contains('brand')
            ).astype(int)
 
    # Ensure target columns exist
    for t in TARGETS:
        if t not in df.columns:
            df[t] = np.nan
 
    df = df.dropna(subset=TARGETS, how='all').reset_index(drop=True)
    print("Final dataset rows:", len(df))
    print(f"    → {len(df)} rows, {df.shape[1]} columns")
    return df
 
 
# ─── Regression Engine ────────────────────────────────────────────────────────
FEATURE_CANDIDATES = [
    'points_issued', 'txn_amount', 'redemptions',
    'campaign_cost', 'nla_sqft', 'is_brand_campaign',
]
 
def select_features(df: pd.DataFrame, target: str) -> list[str]:
    """Auto-select available numeric feature columns (exclude target & identifiers)."""
    exclude = {'month_year', 'shop_name', 'gto_amount', 'gto_rent',
               'gto_reporting_month', 'lease_status'}
    feats = [c for c in FEATURE_CANDIDATES
             if c in df.columns and c != target and c not in exclude]
    # Drop if collinear with campaign_cost (which already encodes points_issued)
    if 'campaign_cost' in feats and 'points_issued' in feats:
        feats.remove('points_issued')   # campaign_cost is the monetised version
    return feats
 
 
def run_ols(df: pd.DataFrame, target: str, features: list[str]) -> dict:
    """
    Fit OLS regression and return a rich result dict.
    Includes: coefficients, p-values, confidence intervals,
              VIF, 5-fold cross-validated RMSE.
    """
    result = {'target': target, 'features': features}
 
    subset = df[features + [target]].dropna()
    if len(subset) < MIN_ROWS:
        result['error'] = f"Too few rows ({len(subset)}) after dropping NaN"
        return result
 
    X_raw = subset[features].astype(float)
    y     = subset[target].astype(float)
 
    # ── Standardise X for comparable coefficients ──
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw),
        columns=features, index=X_raw.index
    )
    X_const  = sm.add_constant(X_scaled)
 
    # ── OLS fit ──
    model = sm.OLS(y, X_const).fit()
 
    # ── Coefficient table ──
    coef_df = pd.DataFrame({
        'feature':    ['const'] + features,
        'coef':       model.params.values,
        'std_err':    model.bse.values,
        't_stat':     model.tvalues.values,
        'p_value':    model.pvalues.values,
        'ci_lower':   model.conf_int()[0].values,
        'ci_upper':   model.conf_int()[1].values,
        'significant': (model.pvalues.values < 0.05),
    })
    result['coef_table'] = df_to_records_safe(coef_df)
 
    # ── Model fit ──
    result['model_fit'] = {
        'r_squared':     safe_float(model.rsquared),
        'adj_r_squared': safe_float(model.rsquared_adj),
        'f_statistic':   safe_float(model.fvalue),
        'f_pvalue':      safe_float(model.f_pvalue),
        'aic':           safe_float(model.aic),
        'bic':           safe_float(model.bic),
        'n_obs':         int(model.nobs),
        'dw_stat':       safe_float(sm.stats.stattools.durbin_watson(model.resid)),
    }
 
    # ── VIF (multicollinearity check) ──
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    try:
        vif_data = pd.DataFrame({
            'feature': features,
            'vif':     [safe_float(variance_inflation_factor(X_const.values, i + 1))
                        for i in range(len(features))]
        })
        result['vif'] = df_to_records_safe(vif_data)
    except Exception:
        result['vif'] = []
 
    # ── 5-fold cross-validated RMSE ──
    try:
        kf        = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_rmse   = []
        cv_mae    = []
        cv_r2     = []
        X_np, y_np = X_scaled.values, y.values
        for train_idx, test_idx in kf.split(X_np):
            X_tr, X_te = X_np[train_idx], X_np[test_idx]
            y_tr, y_te = y_np[train_idx], y_np[test_idx]
            m = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
            pred = m.predict(sm.add_constant(X_te))
            cv_rmse.append(mean_squared_error(y_te, pred, squared=False))
            cv_mae.append(mean_absolute_error(y_te, pred))
            cv_r2.append(r2_score(y_te, pred))
        result['cross_validation'] = {
            'cv_rmse_mean': safe_float(np.mean(cv_rmse)),
            'cv_rmse_std':  safe_float(np.std(cv_rmse)),
            'cv_mae_mean':  safe_float(np.mean(cv_mae)),
            'cv_r2_mean':   safe_float(np.mean(cv_r2)),
        }
    except Exception as e:
        result['cross_validation'] = {'error': str(e)}
 
    # ── Residuals ──
    resid_df = pd.DataFrame({
        'fitted':   model.fittedvalues.values,
        'residual': model.resid.values,
        'std_resid': (model.resid / model.resid.std()).values,
    })
    result['residuals'] = df_to_records_safe(resid_df)
 
    # ── Business insight ──
    sig_feats = coef_df[coef_df['significant'] & (coef_df['feature'] != 'const')]
    insight_parts = []
    for _, row in sig_feats.iterrows():
        direction = "increases" if row['coef'] > 0 else "decreases"
        insight_parts.append(
            f"{row['feature']} {direction} {target} "
            f"(β={row['coef']:.3f}, p={row['p_value']:.3f})"
        )
    result['insight'] = (
        f"Model explains {model.rsquared*100:.1f}% of variance in {target} "
        f"(Adj-R²={model.rsquared_adj:.3f}, n={int(model.nobs)}). "
        + ("Significant predictors: " + "; ".join(insight_parts) + "."
           if insight_parts else "No significant predictors at p<0.05.")
    )
 
    return result
 
 
def run_stepwise_ols(df: pd.DataFrame, target: str, features: list[str]) -> dict:
    """
    Backward stepwise OLS: repeatedly drop the highest-p-value feature
    until all remaining features have p < 0.10.
    Returns the final reduced model result.
    """
    remaining = features.copy()
    iteration = 0
    dropped   = []
 
    while True:
        iteration += 1
        subset = df[remaining + [target]].dropna()
        if len(subset) < MIN_ROWS or not remaining:
            break
 
        X = sm.add_constant(
            StandardScaler().fit_transform(subset[remaining].astype(float))
        )
        model = sm.OLS(subset[target].astype(float), X).fit()
        pvals = pd.Series(model.pvalues[1:].values, index=remaining)  # skip const
 
        worst_p = pvals.max()
        if worst_p < 0.10:
            break   # all features significant
 
        worst_feat = pvals.idxmax()
        if worst_feat in remaining:
            dropped.append({'feature': worst_feat, 'p_value': safe_float(worst_p)})
            remaining.remove(worst_feat)
            print(f"    Stepwise iter {iteration}: dropped '{worst_feat}' (p={worst_p:.3f})")
        else:
            print(f"    ⚠️ Tried to drop '{worst_feat}' but not in remaining — skipping")
            break
 
    result = run_ols(df, target, remaining) if remaining else {'target': target, 'error': 'No features survived stepwise selection'}
    result['stepwise_dropped'] = dropped
    result['stepwise_final_features'] = remaining
    return result
 
 
# ─── Main Analysis Entry Point ────────────────────────────────────────────────
def run_linear_regression(campaign: pd.DataFrame,
                          gto: pd.DataFrame) -> dict:
    """
    Run all regression analyses. Returns a nested dict keyed by:
      linear_regression
        ├── monthly_gto_amount   : monthly-level model for GTO revenue
        ├── monthly_gto_rent     : monthly-level model for GTO rent
        ├── outlet_gto_amount    : outlet-level model for GTO revenue
        ├── outlet_gto_rent      : outlet-level model for GTO rent
        ├── panel_gto_amount     : panel (outlet×month) model for GTO revenue
        ├── panel_gto_rent       : panel (outlet×month) model for GTO rent
        └── summary              : comparison table across models
    """
    results = {}
 
    for level in ['monthly', 'outlet', 'panel']:
        df = build_regression_dataset(campaign, gto, level=level)
        if df.empty:
            print(f"  ⚠️  Skipping {level} level — empty dataset.")
            continue
 
        for target in TARGETS:
            if target not in df.columns or df[target].dropna().empty:
                print(f"  ⚠️  Skipping {level}/{target} — no data.")
                continue
 
            features = select_features(df, target)
            if not features:
                print(f"  ⚠️  No features available for {level}/{target}.")
                continue
 
            key = f"{level}_{target}"
            print(f"\n  ── {key} ──")
            print(f"     Features: {features}")
 
            full_model      = run_ols(df, target, features)
            stepwise_model  = run_stepwise_ols(df, target, features)
 
            results[key] = {
                'full_model':      full_model,
                'stepwise_model':  stepwise_model,
                'level':           level,
                'target':          target,
                'n_rows':          len(df.dropna(subset=[target])),
                'features_used':   features,
            }
            print(f"     R²={full_model.get('model_fit', {}).get('r_squared', 'N/A')}, "
                  f"Adj-R²={full_model.get('model_fit', {}).get('adj_r_squared', 'N/A')}")
 
    # ── Summary comparison table ──
    summary_rows = []
    for key, res in results.items():
        for model_type in ['full_model', 'stepwise_model']:
            m = res.get(model_type, {})
            fit = m.get('model_fit', {})
            cv  = m.get('cross_validation', {})
            summary_rows.append({
                'model_key':     key,
                'model_type':    model_type,
                'level':         res['level'],
                'target':        res['target'],
                'r_squared':     fit.get('r_squared'),
                'adj_r_squared': fit.get('adj_r_squared'),
                'f_pvalue':      fit.get('f_pvalue'),
                'n_obs':         fit.get('n_obs'),
                'cv_rmse':       cv.get('cv_rmse_mean'),
                'cv_r2':         cv.get('cv_r2_mean'),
                'insight':       m.get('insight', ''),
            })
    results['summary'] = summary_rows
 
    return results
 
 
# ─── CLI Entry Point ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) >= 3:
        DATA_FOLDER     = sys.argv[1]
        COMBINED_FOLDER = sys.argv[2]
        MAPPING_FILE    = sys.argv[3] if len(sys.argv) > 3 else ''
    else:
        _cfg_file   = sys.argv[1] if len(sys.argv) == 2 else 'config_Kim.xlsx'
        _script_dir = Path(__file__).resolve().parent
        _paths_df   = pd.read_excel(_script_dir / _cfg_file, sheet_name='paths')
        _cfg        = dict(zip(_paths_df['Setting'].astype(str).str.strip(), _paths_df['Value']))
        DATA_FOLDER     = str(_cfg.get('cleaned_data',  '')).strip()
        COMBINED_FOLDER = str(_cfg.get('combined_data', '')).strip()
        MAPPING_FILE    = str(_cfg.get('shop_mapping',  '')).strip()
        print(f"📖 Loaded config from {_cfg_file}")
 
    print('\n' + '='*60)
    print('LINEAR REGRESSION ANALYSIS STARTING')
    print('='*60)
 
    campaign, gto = load_data(DATA_FOLDER, MAPPING_FILE)
    print("After filtering:")
    print("campaign rows:", len(campaign))
    print("gto rows:", len(gto))
    print(f"  Loaded campaign: {len(campaign):,} rows | GTO: {len(gto):,} rows")
 
    results = run_linear_regression(campaign, gto)
 
    # ── Save outputs ──
    os.makedirs(COMBINED_FOLDER, exist_ok=True)
 
    json_path = os.path.join(COMBINED_FOLDER, 'linear_regression_results.json')
    with open(json_path, 'w') as f:
        json.dump({'linear_regression': results}, f, indent=2, default=str)
    print(f'\n  ✅ Saved: {json_path}')
 
    # Save summary Excel
    summary_df = pd.DataFrame(results.get('summary', []))
    if not summary_df.empty:
        xlsx_path = os.path.join(COMBINED_FOLDER, 'linear_regression_summary.xlsx')
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            # Per-model coefficient sheets
            for key, res in results.items():
                if key == 'summary':
                    continue
                for mtype in ['full_model', 'stepwise_model']:
                    coefs = res.get(mtype, {}).get('coef_table', [])
                    if coefs:
                        sheet = f"{key[:18]}_{mtype[:4]}"   # Excel 31-char limit
                        pd.DataFrame(coefs).to_excel(writer, sheet_name=sheet, index=False)
        print(f'  ✅ Saved: {xlsx_path}')
 
    print('\n' + '='*60)
    print('✅ LINEAR REGRESSION COMPLETED')
    print('='*60)