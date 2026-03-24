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

  ── Ratio features (VIF reduction) ──
  - spend_per_redemption            : avg spend per redeeming customer
  - points_per_redemption           : avg points per redeeming customer
  - cost_efficiency                 : campaign_cost / txn_amount

  ── Additional models ──
  - Ridge / Lasso   : regularised OLS, handles multicollinearity
  - PCA regression  : orthogonal components, VIF = 1 by construction
  - Binary Logit    : GTO above/below median → odds ratios, AUC, marginal effects

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
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LassoCV, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              roc_auc_score, roc_curve, classification_report,
                              confusion_matrix)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────
POINTS_COST_SGD    = 0.20
TARGETS            = ['gto_amount', 'gto_rent']
MIN_ROWS           = 10
PCA_VARIANCE_THRESHOLD = 0.90   # keep components explaining ≥90% variance
RIDGE_ALPHAS       = np.logspace(-3, 5, 50)
LASSO_ALPHAS       = np.logspace(-4, 4, 50)


# ─── Utility helpers ──────────────────────────────────────────────────────────
def safe_float(val):
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None

def df_to_records_safe(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    return json.loads(
        df.replace([np.inf, -np.inf], np.nan).to_json(orient='records')
    )

def find_col(df: pd.DataFrame, candidates: list):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data(data_folder: str, mapping_file: str = ''):
    camp_path = os.path.join(data_folder, 'campaign_all.csv')
    if not os.path.exists(camp_path):
        raise FileNotFoundError(f"campaign_all.csv not found in {data_folder}")
    campaign = pd.read_csv(camp_path)

    for col in campaign.columns:
        if 'date' in col.lower():
            campaign[col] = pd.to_datetime(campaign[col], errors='coerce')

    if 'month_year' not in campaign.columns:
        if 'month' in campaign.columns and 'year' in campaign.columns:
            campaign['month_year'] = (
                campaign['month'].astype(str) + '-' + campaign['year'].astype(str)
            )

    gto_path = ''
    for f in os.listdir(data_folder):
        if 'gto_monthly_rent' in f.lower() and f.endswith(('.xlsx', '.csv')):
            gto_path = os.path.join(data_folder, f)
            break
    if not gto_path:
        raise FileNotFoundError("No gto_monthly_rent file found in " + data_folder)

    gto = (pd.read_excel(gto_path) if gto_path.endswith('.xlsx')
           else pd.read_csv(gto_path))

    VALID_METHODS = {'exact', 'fuzzy', 'confirmed', 'code_match', 'combined_exact', 'combined_fuzzy'}
    if mapping_file and os.path.exists(mapping_file):
        try:
            mdf = pd.read_excel(mapping_file, sheet_name='mapping')
            mdf['campaign_name'] = mdf['campaign_name'].astype(str).str.strip().str.lower()
            mdf['gto_name']      = mdf['gto_name'].astype(str).str.strip().str.lower()
            mdf['method']        = mdf['method'].astype(str).str.strip().str.lower()
            valid_map      = mdf[mdf['method'].isin(VALID_METHODS)]
            valid_campaign = set(valid_map['campaign_name'])
            valid_gto      = set(valid_map['gto_name'])
            if 'outlet_name' in campaign.columns:
                campaign = campaign[
                    campaign['outlet_name'].str.strip().str.lower().isin(valid_campaign)
                ].copy()
            if 'shop_name' in gto.columns:
                gto = gto[
                    gto['shop_name'].str.strip().str.lower().isin(valid_gto)
                ].copy()
        except Exception as e:
            print(f"⚠️ mapping filter failed: {e}")

    if 'month_year' not in gto.columns and 'gto_reporting_month' in gto.columns:
        gto['month_year'] = pd.to_datetime(
            gto['gto_reporting_month'], errors='coerce'
        ).dt.strftime('%b-%Y')

    return campaign, gto


# ─── Ratio Features (VIF Reduction) ──────────────────────────────────────────
def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace or supplement raw correlated features with ratios.
    These are orthogonalised by construction and have much lower VIF.

      spend_per_redemption  = txn_amount / redemptions
      points_per_redemption = points_issued / redemptions
      cost_efficiency       = campaign_cost / txn_amount  (cost per $ revenue)

    Returns df with new columns added (originals kept for reference).
    """
    df = df.copy()

    if 'txn_amount' in df.columns and 'redemptions' in df.columns:
        df['spend_per_redemption'] = np.where(
            df['redemptions'] > 0,
            df['txn_amount'] / df['redemptions'],
            np.nan
        )

    if 'points_issued' in df.columns and 'redemptions' in df.columns:
        df['points_per_redemption'] = np.where(
            df['redemptions'] > 0,
            df['points_issued'] / df['redemptions'],
            np.nan
        )

    if 'campaign_cost' in df.columns and 'txn_amount' in df.columns:
        df['cost_efficiency'] = np.where(
            df['txn_amount'] > 0,
            df['campaign_cost'] / df['txn_amount'],
            np.nan
        )

    return df


# ─── Feature Engineering ──────────────────────────────────────────────────────
def build_regression_dataset(campaign: pd.DataFrame,
                              gto: pd.DataFrame,
                              level: str = 'monthly') -> pd.DataFrame:
    print(f"\n  Building regression dataset at level='{level}' ...")

    pts_col    = find_col(campaign, ['points_earned', 'points_issued'])
    amt_col    = find_col(campaign, ['amount', 'txn_amount'])
    rcpt_col   = find_col(campaign, ['receipt_no'])
    outlet_col = find_col(campaign, ['outlet_name', 'final_gto_name'])
    shop_col   = find_col(gto, ['shop_name'])
    gto_a_col  = find_col(gto, ['gto_amount'])
    gto_r_col  = find_col(gto, ['gto_rent'])

    agg_dict = {}
    if pts_col:  agg_dict['points_issued'] = (pts_col,  'sum')
    if amt_col:  agg_dict['txn_amount']    = (amt_col,  'sum')
    if rcpt_col: agg_dict['redemptions']   = (rcpt_col, 'nunique')

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
        nla_col = find_col(gto, ['nla_sqft'])
        if nla_col:
            nla = gto[[shop_col, nla_col]].drop_duplicates().rename(
                columns={shop_col: 'shop_name', nla_col: 'nla_sqft'})
            df = df.merge(nla, on='shop_name', how='left')
    else:
        raise ValueError(f"Unknown level: {level}")

    if 'points_issued' in df.columns:
        df['campaign_cost'] = df['points_issued'] * POINTS_COST_SGD

    # ── Add campaign dummies ──
    voucher_col = find_col(campaign, ['voucher_code', 'campaign_source'])
    if voucher_col and level in ('monthly', 'panel'):
        top_campaigns = (
            campaign[voucher_col].value_counts().head(10).index.tolist()
        )
        campaign_filtered = campaign[campaign[voucher_col].isin(top_campaigns)].copy()
        campaign_filtered[voucher_col] = campaign_filtered[voucher_col].str[:20]

        if level == 'monthly':
            dummies = pd.get_dummies(
                campaign_filtered.groupby('month_year')[voucher_col]
                .agg(lambda x: x.mode()[0] if not x.empty else 'none'),
                prefix='camp'
            )
            df = df.merge(dummies, on='month_year', how='left').fillna(0)
        elif level == 'panel':
            dummies = pd.get_dummies(
                campaign_filtered.groupby([outlet_col, 'month_year'])[voucher_col]
                .agg(lambda x: x.mode()[0] if not x.empty else 'none')
                .reset_index()
                .rename(columns={outlet_col: 'shop_name'}),
                columns=[voucher_col], prefix='camp'
            )
            df = df.merge(dummies, on=['shop_name', 'month_year'], how='left').fillna(0)

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

    for t in TARGETS:
        if t not in df.columns:
            df[t] = np.nan

    df = df.dropna(subset=TARGETS, how='all').reset_index(drop=True)

    # ── Add ratio features AFTER base features are built ──
    df = add_ratio_features(df)

    print(f"    → {len(df)} rows, {df.shape[1]} columns")
    return df


# ─── Feature Selection ────────────────────────────────────────────────────────
BASE_FEATURE_CANDIDATES = [
    'points_issued', 'txn_amount', 'redemptions',
    'campaign_cost', 'nla_sqft', 'is_brand_campaign',
]
RATIO_FEATURE_CANDIDATES = [
    'spend_per_redemption', 'points_per_redemption', 'cost_efficiency',
]


def select_features(df: pd.DataFrame, target: str,
                    use_ratio_features: bool = False) -> list:
    exclude = {'month_year', 'shop_name', 'gto_amount', 'gto_rent',
               'gto_reporting_month', 'lease_status'}

    if use_ratio_features:
        # Use ratio features instead of raw correlated ones — lower VIF
        ratio_feats = [c for c in RATIO_FEATURE_CANDIDATES
                       if c in df.columns and c != target and c not in exclude]
        # Only keep nla_sqft and is_brand_campaign from base (not collinear)
        extra_feats = [c for c in ['nla_sqft', 'is_brand_campaign']
                       if c in df.columns and c != target and c not in exclude]
        camp_dummies = [c for c in df.columns if c.startswith('camp_') and c != target]
        return ratio_feats + extra_feats + camp_dummies
    else:
        base_feats   = [c for c in BASE_FEATURE_CANDIDATES
                        if c in df.columns and c != target and c not in exclude]
        camp_dummies = [c for c in df.columns if c.startswith('camp_') and c != target]
        if camp_dummies and 'is_brand_campaign' in base_feats:
            base_feats.remove('is_brand_campaign')
        if 'campaign_cost' in base_feats and 'points_issued' in base_feats:
            base_feats.remove('points_issued')
        return base_feats + camp_dummies


# ─── OLS ─────────────────────────────────────────────────────────────────────
def run_ols(df: pd.DataFrame, target: str, features: list) -> dict:
    result = {'target': target, 'features': features, 'model_name': 'OLS'}

    subset = df[features + [target]].dropna()
    if len(subset) < MIN_ROWS:
        result['error'] = f"Too few rows ({len(subset)}) after dropping NaN"
        return result

    X_raw = subset[features].astype(float)
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
    result['coef_table'] = df_to_records_safe(coef_df)

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

    try:
        kf = KFold(n_splits=min(5, len(subset)), shuffle=True, random_state=42)
        cv_rmse, cv_r2 = [], []

        X_np = X_pca
        y_np = y.values

        for tr, te in kf.split(X_np):
            m    = sm.OLS(y_np[tr], sm.add_constant(X_np[tr])).fit()
            pred = m.predict(sm.add_constant(X_np[te]))
            cv_rmse.append(np.sqrt(mean_squared_error(y_np[te], pred)))
            cv_r2.append(r2_score(y_np[te], pred))
        result['cross_validation'] = {
            'cv_rmse_mean': safe_float(np.mean(cv_rmse)),
            'cv_rmse_std':  safe_float(np.std(cv_rmse)),
            'cv_mae_mean':  safe_float(np.mean(cv_mae)),
            'cv_r2_mean':   safe_float(np.mean(cv_r2)),
        }
    except Exception as e:
        result['cross_validation'] = {'error': str(e)}

    resid_df = pd.DataFrame({
        'fitted':    model.fittedvalues.values,
        'residual':  model.resid.values,
        'std_resid': (model.resid / model.resid.std()).values,
    })
    result['residuals'] = df_to_records_safe(resid_df)

    sig_feats = coef_df[coef_df['significant'] & (coef_df['feature'] != 'const')]
    insight_parts = [
        f"{r['feature']} {'increases' if r['coef'] > 0 else 'decreases'} {target} "
        f"(β={r['coef']:.3f}, p={r['p_value']:.3f})"
        for _, r in sig_feats.iterrows()
    ]
    result['insight'] = (
        f"OLS — R²={model.rsquared*100:.1f}%, Adj-R²={model.rsquared_adj:.3f}, n={int(model.nobs)}. "
        + ("Significant: " + "; ".join(insight_parts) + "."
           if insight_parts else "No significant predictors at p<0.05.")
    )
    return result


def run_stepwise_ols(df: pd.DataFrame, target: str, features: list) -> dict:
    remaining = features.copy()
    dropped   = []
    for iteration in range(1, 50):
        subset = df[remaining + [target]].dropna()
        if len(subset) < MIN_ROWS or not remaining:
            break
        X     = sm.add_constant(StandardScaler().fit_transform(subset[remaining].astype(float)))
        model = sm.OLS(subset[target].astype(float), X).fit()
        pvals = pd.Series(model.pvalues[1:].values, index=remaining)
        worst_p = pvals.max()
        if worst_p < 0.10:
            break
        worst_feat = pvals.idxmax()
        if worst_feat in remaining:
            dropped.append({'feature': worst_feat, 'p_value': safe_float(worst_p)})
            remaining.remove(worst_feat)
            print(f"    Stepwise iter {iteration}: dropped '{worst_feat}' (p={worst_p:.3f})")
        else:
            break
    result = (run_ols(df, target, remaining) if remaining
              else {'target': target, 'error': 'No features survived stepwise'})
    result['stepwise_dropped']         = dropped
    result['stepwise_final_features']  = remaining
    return result


# ─── Ridge Regression ────────────────────────────────────────────────────────
def run_ridge(df: pd.DataFrame, target: str, features: list) -> dict:
    """
    Ridge (L2) regression with cross-validated alpha.
    Shrinks all coefficients — never zeros them out.
    Best for: many correlated features (VIF reduction without dropping).
    """
    result = {'target': target, 'features': features, 'model_name': 'Ridge'}

    subset = df[features + [target]].dropna()
    if len(subset) < MIN_ROWS:
        result['error'] = f"Too few rows ({len(subset)})"
        return result

    X_raw = subset[features].astype(float)
    y     = subset[target].astype(float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Cross-validated alpha selection
    ridge_cv = RidgeCV(alphas=RIDGE_ALPHAS, cv=min(5, len(subset)),
                       scoring='r2')
    ridge_cv.fit(X_scaled, y)

    best_alpha = float(ridge_cv.alpha_)
    coefs      = ridge_cv.coef_

    # Cross-validation metrics
    kf = KFold(n_splits=min(5, len(subset)), shuffle=True, random_state=42)
    cv_rmse, cv_r2 = [], []
    for tr, te in kf.split(X_scaled):
        from sklearn.linear_model import Ridge as RidgeSK
        m    = RidgeSK(alpha=best_alpha).fit(X_scaled[tr], y.values[tr])
        pred = m.predict(X_scaled[te])
        cv_rmse.append(np.sqrt(mean_squared_error(y.values[te], pred)))
        cv_r2.append(r2_score(y.values[te], pred))

    pred_all = ridge_cv.predict(X_scaled)
    r2       = r2_score(y, pred_all)
    residuals = y.values - pred_all

    coef_df = pd.DataFrame({
        'feature': features,
        'coef':    [safe_float(c) for c in coefs],
        'abs_coef': [safe_float(abs(c)) for c in coefs],
    }).sort_values('abs_coef', ascending=False)

    result['coef_table'] = df_to_records_safe(coef_df)
    result['model_fit']  = {
        'r_squared':     safe_float(r2),
        'adj_r_squared': safe_float(1 - (1 - r2) * (len(y) - 1) / max(len(y) - len(features) - 1, 1)),
        'best_alpha':    safe_float(best_alpha),
        'n_obs':         int(len(subset)),
        'rmse':          safe_float(np.sqrt(np.mean(residuals**2))),
    }
    result['cross_validation'] = {
        'cv_rmse_mean': safe_float(np.mean(cv_rmse)),
        'cv_rmse_std':  safe_float(np.std(cv_rmse)),
        'cv_r2_mean':   safe_float(np.mean(cv_r2)),
    }
    result['residuals'] = df_to_records_safe(pd.DataFrame({
        'fitted':    pred_all,
        'residual':  residuals,
        'std_resid': residuals / (residuals.std() or 1),
    }))
    result['vif'] = []  # Ridge inherently handles VIF — not applicable
    result['alpha_search'] = {
        'best_alpha':  safe_float(best_alpha),
        'alpha_range': [safe_float(RIDGE_ALPHAS[0]), safe_float(RIDGE_ALPHAS[-1])],
        'note': 'Higher alpha = more regularisation = smaller coefficients',
    }
    top = coef_df.head(3)['feature'].tolist()
    result['insight'] = (
        f"Ridge (α={best_alpha:.3f}) — R²={r2*100:.1f}%, CV-R²={np.mean(cv_r2):.3f}, n={len(subset)}. "
        f"Top predictors by magnitude: {', '.join(top)}. "
        f"Ridge shrinks all coefficients without zeroing them — good for multicollinear data."
    )
    return result


# ─── Lasso Regression ────────────────────────────────────────────────────────
def run_lasso(df: pd.DataFrame, target: str, features: list) -> dict:
    """
    Lasso (L1) regression with cross-validated alpha.
    Zeros out irrelevant features — automatic feature selection.
    Best for: sparse solutions, identifying the key drivers.
    """
    result = {'target': target, 'features': features, 'model_name': 'Lasso'}

    subset = df[features + [target]].dropna()
    if len(subset) < MIN_ROWS:
        result['error'] = f"Too few rows ({len(subset)})"
        return result

    X_raw = subset[features].astype(float)
    y     = subset[target].astype(float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    lasso_cv = LassoCV(alphas=LASSO_ALPHAS, cv=min(5, len(subset)),
                       max_iter=10000, random_state=42)
    lasso_cv.fit(X_scaled, y)

    best_alpha    = float(lasso_cv.alpha_)
    coefs         = lasso_cv.coef_
    selected_mask = coefs != 0
    selected_feats = [f for f, s in zip(features, selected_mask) if s]
    zeroed_feats   = [f for f, s in zip(features, selected_mask) if not s]

    kf = KFold(n_splits=min(5, len(subset)), shuffle=True, random_state=42)
    cv_rmse, cv_r2 = [], []
    for tr, te in kf.split(X_scaled):
        from sklearn.linear_model import Lasso as LassoSK
        m    = LassoSK(alpha=best_alpha, max_iter=10000).fit(X_scaled[tr], y.values[tr])
        pred = m.predict(X_scaled[te])
        cv_rmse.append(np.sqrt(mean_squared_error(y.values[te], pred)))
        cv_r2.append(r2_score(y.values[te], pred))

    pred_all  = lasso_cv.predict(X_scaled)
    r2        = r2_score(y, pred_all)
    residuals = y.values - pred_all

    coef_df = pd.DataFrame({
        'feature':  features,
        'coef':     [safe_float(c) for c in coefs],
        'selected': [bool(s) for s in selected_mask],
        'abs_coef': [safe_float(abs(c)) for c in coefs],
    }).sort_values('abs_coef', ascending=False)

    result['coef_table'] = df_to_records_safe(coef_df)
    result['model_fit']  = {
        'r_squared':        safe_float(r2),
        'adj_r_squared':    safe_float(1 - (1 - r2) * (len(y) - 1) / max(len(y) - len(features) - 1, 1)),
        'best_alpha':       safe_float(best_alpha),
        'n_obs':            int(len(subset)),
        'n_features_in':    len(features),
        'n_features_kept':  len(selected_feats),
        'n_features_zeroed': len(zeroed_feats),
        'rmse':             safe_float(np.sqrt(np.mean(residuals**2))),
    }
    result['cross_validation'] = {
        'cv_rmse_mean': safe_float(np.mean(cv_rmse)),
        'cv_rmse_std':  safe_float(np.std(cv_rmse)),
        'cv_r2_mean':   safe_float(np.mean(cv_r2)),
    }
    result['residuals'] = df_to_records_safe(pd.DataFrame({
        'fitted':    pred_all,
        'residual':  residuals,
        'std_resid': residuals / (residuals.std() or 1),
    }))
    result['vif']            = []
    result['selected_features'] = selected_feats
    result['zeroed_features']   = zeroed_feats
    result['alpha_search']   = {
        'best_alpha':  safe_float(best_alpha),
        'note': 'Higher alpha = more features zeroed out',
    }
    result['insight'] = (
        f"Lasso (α={best_alpha:.4f}) — R²={r2*100:.1f}%, CV-R²={np.mean(cv_r2):.3f}, n={len(subset)}. "
        f"Kept {len(selected_feats)}/{len(features)} features: {', '.join(selected_feats) or 'none'}. "
        f"Zeroed: {', '.join(zeroed_feats) or 'none'}."
    )
    return result


# ─── PCA Regression ───────────────────────────────────────────────────────────
def run_pca_regression(df: pd.DataFrame, target: str, features: list) -> dict:
    """
    PCA + OLS on principal components.
    Components are orthogonal → VIF = 1 by construction.
    Loadings map each PC back to original features for interpretability.
    """
    result = {'target': target, 'features': features, 'model_name': 'PCA_OLS'}

    subset = df[features + [target]].dropna()
    if len(subset) < MIN_ROWS:
        result['error'] = f"Too few rows ({len(subset)})"
        return result

    X_raw = subset[features].astype(float)
    y     = subset[target].astype(float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ── PCA: keep components until ≥ THRESHOLD cumulative variance ──
    pca_full     = PCA(n_components=min(len(features), len(subset) - 1))
    pca_full.fit(X_scaled)
    cum_var      = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = max(1, int(np.searchsorted(cum_var, PCA_VARIANCE_THRESHOLD) + 1))
    n_components = min(n_components, len(features), len(subset) - 1)

    print(f"    PCA: {n_components} components explain {cum_var[n_components-1]*100:.1f}% variance")

    pca      = PCA(n_components=n_components)
    X_pca    = pca.fit_transform(X_scaled)
    pc_names = [f"PC{i+1}" for i in range(n_components)]

    # ── OLS on PCs ──
    X_const = sm.add_constant(X_pca)
    model   = sm.OLS(y, X_const).fit()

    coef_df = pd.DataFrame({
        'component': pc_names,
        'coef':      model.params.values[1:],
        'p_value':   model.pvalues.values[1:],
        'significant': model.pvalues.values[1:] < 0.05,
        'variance_explained_pct': pca.explained_variance_ratio_ * 100,
    })

    # ── Loadings: which original features drive each PC ──
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=pc_names,
    ).round(4)

    # ── Map PC coefficients back to original feature space ──
    # β_original ≈ loadings @ β_pc (approximate, ignores scaling)
    beta_pc      = model.params.values[1:]
    beta_original = pca.components_.T @ beta_pc  # shape (n_features,)
    original_coef_df = pd.DataFrame({
        'feature':             features,
        'implied_coefficient': [safe_float(b) for b in beta_original],
    }).sort_values('implied_coefficient', key=abs, ascending=False)

    result['coef_table']        = df_to_records_safe(coef_df)
    result['loadings']          = df_to_records_safe(loadings_df.reset_index().rename(columns={'index': 'feature'}))
    result['implied_coefs']     = df_to_records_safe(original_coef_df)
    result['model_fit'] = {
        'r_squared':         safe_float(model.rsquared),
        'adj_r_squared':     safe_float(model.rsquared_adj),
        'f_pvalue':          safe_float(model.f_pvalue),
        'n_obs':             int(model.nobs),
        'n_components_used': n_components,
        'variance_explained_pct': safe_float(float(cum_var[n_components - 1]) * 100),
        'vif_note':          'VIF = 1.0 for all PCs by construction',
        'dw_stat':           safe_float(sm.stats.stattools.durbin_watson(model.resid)),
    }

    # VIF = 1 for all components
    result['vif'] = df_to_records_safe(pd.DataFrame({
        'feature': pc_names,
        'vif':     [1.0] * n_components,
    }))

    try:
        kf = KFold(n_splits=min(5, len(subset)), shuffle=True, random_state=42)
        cv_rmse, cv_r2 = [], []
        for tr, te in kf.split(X_np):
            m    = sm.OLS(y_np[tr], sm.add_constant(X_np[tr])).fit()
            pred = m.predict(sm.add_constant(X_np[te]))
            cv_rmse.append(np.sqrt(mean_squared_error(y_np[te], pred)))
            cv_r2.append(r2_score(y.values[te], pred))
        result['cross_validation'] = {
            'cv_rmse_mean': safe_float(np.mean(cv_rmse)),
            'cv_rmse_std':  safe_float(np.std(cv_rmse)),
            'cv_r2_mean':   safe_float(np.mean(cv_r2)),
        }
    except Exception as e:
        result['cross_validation'] = {'error': str(e)}

    resid      = model.resid.values
    result['residuals'] = df_to_records_safe(pd.DataFrame({
        'fitted':    model.fittedvalues.values,
        'residual':  resid,
        'std_resid': resid / (resid.std() or 1),
    }))

    top_pc = coef_df[coef_df['significant']]['component'].tolist() if any(coef_df['significant']) else pc_names[:2]
    result['insight'] = (
        f"PCA-OLS — {n_components} components explain {cum_var[n_components-1]*100:.1f}% of feature variance. "
        f"R²={model.rsquared*100:.1f}%, Adj-R²={model.rsquared_adj:.3f}, n={int(model.nobs)}. "
        f"VIF=1 for all components. "
        f"Significant PCs: {', '.join(top_pc) or 'none'}. "
        f"Highest-loading original feature on PC1: "
        f"{loadings_df['PC1'].abs().idxmax() if 'PC1' in loadings_df.columns else '—'}."
    )
    return result


# ─── Binary Logit ─────────────────────────────────────────────────────────────
def run_binary_logit(df: pd.DataFrame, target: str, features: list,
                     threshold_type: str = 'median') -> dict:
    """
    Binary logistic regression: recode target as 1 = above median, 0 = below.
    Returns: odds ratios, marginal effects, AUC, ROC curve points,
             classification report, confusion matrix.
    """
    result = {'target': target, 'features': features,
              'model_name': 'Binary_Logit', 'threshold_type': threshold_type}

    subset = df[features + [target]].dropna()
    if len(subset) < MIN_ROWS:
        result['error'] = f"Too few rows ({len(subset)})"
        return result

    y_cont = subset[target].astype(float)
    threshold = float(y_cont.median())
    y_binary  = (y_cont > threshold).astype(int)

    if y_binary.nunique() < 2:
        result['error'] = "Target has only one class after binarisation — try a different threshold."
        return result

    class_counts = y_binary.value_counts().to_dict()
    result['binarisation'] = {
        'threshold':        safe_float(threshold),
        'threshold_type':   threshold_type,
        'class_0_count':    int(class_counts.get(0, 0)),
        'class_1_count':    int(class_counts.get(1, 0)),
        'label_0':          f"{target} ≤ {threshold:,.0f} (below median)",
        'label_1':          f"{target} > {threshold:,.0f} (above median)",
    }

    X_raw = subset[features].astype(float)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ── Statsmodels Logit for full inference ──
    X_const = sm.add_constant(X_scaled)
    try:
        sm_logit = sm.Logit(y_binary, X_const).fit(disp=False, maxiter=200)

        # Odds ratios
        or_df = pd.DataFrame({
            'feature':      ['const'] + features,
            'log_odds':     [safe_float(v) for v in sm_logit.params.values],
            'odds_ratio':   [safe_float(np.exp(v)) for v in sm_logit.params.values],
            'p_value':      [safe_float(v) for v in sm_logit.pvalues.values],
            'ci_lower_or':  [safe_float(np.exp(v)) for v in sm_logit.conf_int()[0].values],
            'ci_upper_or':  [safe_float(np.exp(v)) for v in sm_logit.conf_int()[1].values],
            'significant':  [bool(p < 0.05) for p in sm_logit.pvalues.values],
        })
        result['coef_table'] = df_to_records_safe(or_df)

        # Marginal effects (at means)
        try:
            mfx    = sm_logit.get_margeff()
            mfx_df = pd.DataFrame({
                'feature':         features,
                'marginal_effect': [safe_float(v) for v in mfx.margeff],
                'std_err':         [safe_float(v) for v in mfx.margeff_se],
                'p_value':         [safe_float(v) for v in mfx.pvalues],
            })
            result['marginal_effects'] = df_to_records_safe(mfx_df)
        except Exception as e:
            result['marginal_effects'] = []
            print(f"    ⚠️ Marginal effects failed: {e}")

        result['model_fit'] = {
            'pseudo_r2':       safe_float(sm_logit.prsquared),
            'log_likelihood':  safe_float(sm_logit.llf),
            'aic':             safe_float(sm_logit.aic),
            'bic':             safe_float(sm_logit.bic),
            'n_obs':           int(sm_logit.nobs),
            'converged':       bool(sm_logit.mle_retvals.get('converged', True)
                                    if hasattr(sm_logit, 'mle_retvals') else True),
        }

    except Exception as e:
        print(f"    ⚠️ Statsmodels logit failed ({e}) — falling back to sklearn.")
        result['model_fit'] = {}
        result['coef_table'] = []

    # ── sklearn Logit for AUC / ROC / CV ──
    sk_logit = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    sk_logit.fit(X_scaled, y_binary)

    y_prob = sk_logit.predict_proba(X_scaled)[:, 1]
    y_pred = sk_logit.predict(X_scaled)

    auc = safe_float(roc_auc_score(y_binary, y_prob))
    fpr, tpr, thresholds = roc_curve(y_binary, y_prob)

    result['roc'] = {
        'auc': auc,
        'fpr': [safe_float(v) for v in fpr.tolist()],
        'tpr': [safe_float(v) for v in tpr.tolist()],
        'roc_points': [{'fpr': safe_float(f), 'tpr': safe_float(t)}
                       for f, t in zip(fpr.tolist()[::max(1, len(fpr)//50)],
                                        tpr.tolist()[::max(1, len(tpr)//50)])],
    }

    cr         = classification_report(y_binary, y_pred, output_dict=True)
    cm         = confusion_matrix(y_binary, y_pred).tolist()
    result['classification'] = {
        'accuracy':   safe_float(cr.get('accuracy')),
        'precision':  safe_float(cr.get('1', {}).get('precision')),
        'recall':     safe_float(cr.get('1', {}).get('recall')),
        'f1_score':   safe_float(cr.get('1', {}).get('f1-score')),
        'auc':        auc,
        'confusion_matrix': cm,
    }

    # Cross-validated AUC
    try:
        skf    = StratifiedKFold(n_splits=min(5, int(y_binary.sum()), len(y_binary) - int(y_binary.sum())),
                                 shuffle=True, random_state=42)
        cv_auc = cross_val_score(sk_logit, X_scaled, y_binary, cv=skf,
                                 scoring='roc_auc')
        result['cross_validation'] = {
            'cv_auc_mean': safe_float(float(np.mean(cv_auc))),
            'cv_auc_std':  safe_float(float(np.std(cv_auc))),
        }
    except Exception as e:
        result['cross_validation'] = {'error': str(e)}

    result['vif'] = []

    pseudo_r2 = result['model_fit'].get('pseudo_r2') or 0
    result['insight'] = (
        f"Binary Logit — target recoded as above/below median ({threshold:,.0f}). "
        f"AUC={auc or 0:.3f}, McFadden Pseudo-R²={pseudo_r2:.3f}, n={len(subset)}. "
        f"{'Good discrimination (AUC > 0.7).' if (auc or 0) > 0.7 else 'Moderate discrimination.'} "
        f"Odds ratio > 1 means the feature increases probability of above-median {target}."
    )
    return result


# ─── Main Analysis Entry Point ────────────────────────────────────────────────
def run_linear_regression(campaign: pd.DataFrame, gto: pd.DataFrame) -> dict:
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

            # ── Base features (original OLS) ──
            features_base  = select_features(df, target, use_ratio_features=False)
            # ── Ratio features (reduced VIF) ──
            features_ratio = select_features(df, target, use_ratio_features=True)

            if not features_base:
                print(f"  ⚠️  No features available for {level}/{target}.")
                continue

            key = f"{level}_{target}"
            print(f"\n  ── {key} ──")

            # OLS (original)
            print(f"     [OLS] Features: {features_base}")
            full_model     = run_ols(df, target, features_base)
            stepwise_model = run_stepwise_ols(df, target, features_base)

            # OLS with ratio features
            print(f"     [OLS-Ratio] Features: {features_ratio}")
            ratio_model = run_ols(df, target, features_ratio) if features_ratio else \
                          {'error': 'No ratio features available'}
            ratio_model['model_name'] = 'OLS_Ratio'

            # Ridge
            print(f"     [Ridge]")
            ridge_model = run_ridge(df, target, features_base)

            # Lasso
            print(f"     [Lasso]")
            lasso_model = run_lasso(df, target, features_base)

            # PCA regression
            print(f"     [PCA-OLS]")
            pca_model = run_pca_regression(df, target, features_base)

            # Binary Logit
            print(f"     [Binary Logit]")
            logit_model = run_binary_logit(df, target, features_base)

            results[key] = {
                'full_model':      full_model,
                'stepwise_model':  stepwise_model,
                'ratio_model':     ratio_model,
                'ridge_model':     ridge_model,
                'lasso_model':     lasso_model,
                'pca_model':       pca_model,
                'logit_model':     logit_model,
                'level':           level,
                'target':          target,
                'n_rows':          len(df.dropna(subset=[target])),
                'features_used':   features_base,
                'ratio_features':  features_ratio,
            }

            r2_ols = full_model.get('model_fit', {}).get('r_squared', 'N/A')
            r2_rid = ridge_model.get('model_fit', {}).get('r_squared', 'N/A')
            print(f"     OLS R²={r2_ols}  Ridge R²={r2_rid}")

    # ── Extended summary including all model types ──
    summary_rows = []
    for key, res in results.items():
        model_types = {
            'full_model':     'OLS (full)',
            'stepwise_model': 'OLS (stepwise)',
            'ratio_model':    'OLS (ratio feats)',
            'ridge_model':    'Ridge',
            'lasso_model':    'Lasso',
            'pca_model':      'PCA-OLS',
            'logit_model':    'Binary Logit',
        }
        for mkey, mlabel in model_types.items():
            m   = res.get(mkey, {})
            fit = m.get('model_fit', {})
            cv  = m.get('cross_validation', {})
            cls = m.get('classification', {})
            summary_rows.append({
                'model_key':     key,
                'model_type':    mlabel,
                'level':         res['level'],
                'target':        res['target'],
                'r_squared':     fit.get('r_squared') or fit.get('pseudo_r2'),
                'adj_r_squared': fit.get('adj_r_squared'),
                'f_pvalue':      fit.get('f_pvalue'),
                'n_obs':         fit.get('n_obs'),
                'cv_rmse':       cv.get('cv_rmse_mean'),
                'cv_r2':         cv.get('cv_r2_mean') or cv.get('cv_auc_mean'),
                'auc':           cls.get('auc'),
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
    print(f"  Loaded campaign: {len(campaign):,} rows | GTO: {len(gto):,} rows")

    results = run_linear_regression(campaign, gto)

    os.makedirs(COMBINED_FOLDER, exist_ok=True)

    json_path = os.path.join(COMBINED_FOLDER, 'linear_regression_results.json')
    with open(json_path, 'w') as f:
        json.dump({'linear_regression': results}, f, indent=2, default=str)
    print(f'\n  ✅ Saved: {json_path}')

    summary_df = pd.DataFrame(results.get('summary', []))
    if not summary_df.empty:
        xlsx_path = os.path.join(COMBINED_FOLDER, 'linear_regression_summary.xlsx')
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            for key, res in results.items():
                if key == 'summary':
                    continue
                for mkey in ['full_model', 'stepwise_model', 'ratio_model',
                             'ridge_model', 'lasso_model', 'pca_model', 'logit_model']:
                    coefs = res.get(mkey, {}).get('coef_table', [])
                    if coefs:
                        sheet = f"{key[:14]}_{mkey[:5]}"
                        pd.DataFrame(coefs).to_excel(writer, sheet_name=sheet, index=False)
        print(f'  ✅ Saved: {xlsx_path}')

    print('\n' + '='*60)
    print('✅ LINEAR REGRESSION COMPLETED')
    print('='*60)