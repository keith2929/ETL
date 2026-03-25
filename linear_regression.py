import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG
# =========================
TOP_N_CAMPAIGNS = 10
COMBINED_FOLDER = "outputs"

BASE_FEATURE_CANDIDATES = [
    'txn_count', 'txn_amount', 'customer_count',
    'campaign_cost', 'points_issued', 'nla_sqft'
]

RATIO_FEATURE_CANDIDATES = [
    'avg_basket_size', 'txn_per_customer'
]

# =========================
# HELPERS
# =========================
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# =========================
# BUILD DATASET
# =========================
def build_regression_dataset(gto, campaign, level='monthly'):

    df = gto.copy()

    # ── CAMPAIGN FEATURES (FIXED) ──
    voucher_col = find_col(campaign, ['voucher_code', 'campaign_source'])

    if voucher_col:
        campaign[voucher_col] = campaign[voucher_col].astype(str)

        if level == 'monthly':
            camp_agg = (
                campaign
                .groupby(['month_year', voucher_col])['txn_amount']
                .sum()
                .unstack(fill_value=0)
            )

            # Keep top campaigns
            top_campaigns = (
                camp_agg.sum()
                .sort_values(ascending=False)
                .head(TOP_N_CAMPAIGNS)
                .index
            )

            camp_agg = camp_agg[top_campaigns]
            camp_agg.columns = [f"camp_{c}" for c in camp_agg.columns]

            df = df.merge(camp_agg, on='month_year', how='left').fillna(0)

    # ── MONTH DUMMIES ──
    df['month'] = pd.to_datetime(df['month_year'], errors='coerce').dt.month
    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)

    return df

# =========================
# FEATURE SELECTION
# =========================
def select_features(df, target, use_ratio_features=False):

    exclude = {
        'month_year', 'shop_name', 'gto_amount',
        'gto_rent', 'month'
    }

    camp_dummies = [c for c in df.columns if c.startswith('camp_')]
    month_dummies = [c for c in df.columns if c.startswith('month_')]

    if use_ratio_features:
        ratio_feats = [c for c in RATIO_FEATURE_CANDIDATES if c in df.columns]
        return ratio_feats + camp_dummies + month_dummies

    else:
        base_feats = [c for c in BASE_FEATURE_CANDIDATES if c in df.columns]

        if 'campaign_cost' in base_feats and 'points_issued' in base_feats:
            base_feats.remove('points_issued')

        return base_feats + camp_dummies + month_dummies

# =========================
# MODEL TRAINING
# =========================
def run_models(df, target):

    X_cols = select_features(df, target)
    X = df[X_cols].fillna(0)
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # OLS
    ols = LinearRegression()
    ols.fit(X_scaled, y)

    # LASSO (for selection)
    lasso = LassoCV(cv=5)
    lasso.fit(X_scaled, y)

    results = {
        'ols': {
            'features': X_cols,
            'coef': ols.coef_
        },
        'lasso_model': {
            'features': X_cols,
            'coef': lasso.coef_
        }
    }

    return results

# =========================
# CAMPAIGN RANKING
# =========================
def extract_campaign_ranking(results):

    ranking = []

    coefs = results['lasso_model']['coef']
    features = results['lasso_model']['features']

    for feat, coef in zip(features, coefs):
        if feat.startswith('camp_'):
            ranking.append({
                'campaign': feat.replace('camp_', ''),
                'impact_on_rent': coef
            })

    ranking_df = pd.DataFrame(ranking)

    if ranking_df.empty:
        print("No campaign results.")
        return ranking_df

    ranking_df = ranking_df.sort_values('impact_on_rent', ascending=False)

    print("\n🔥 TOP CAMPAIGNS:")
    print(ranking_df.head(10))

    print("\n⚠️ WORST CAMPAIGNS:")
    print(ranking_df.tail(10))

    return ranking_df

# =========================
# MAIN PIPELINE
# =========================
def main(gto_path, campaign_path):

    gto = pd.read_csv(gto_path)
    campaign = pd.read_csv(campaign_path)

    df = build_regression_dataset(gto, campaign)

    results = run_models(df, target='gto_rent')

    ranking_df = extract_campaign_ranking(results)

    if not os.path.exists(COMBINED_FOLDER):
        os.makedirs(COMBINED_FOLDER)

    ranking_path = os.path.join(COMBINED_FOLDER, 'campaign_ranking.xlsx')
    ranking_df.to_excel(ranking_path, index=False)

    print(f"\n✅ Saved ranking to {ranking_path}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main("gto.csv", "campaign.csv")