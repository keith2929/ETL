"""
ttest_FINAL.py
--------------
Per-campaign statistical analysis:
  1. Normality test     — Shapiro-Wilk per campaign
  2. One-sample t-test  — H0: campaign mean revenue = overall mean revenue
  3. ROI                — total_amount / total_voucher_value per campaign

Usage:
  python3 ttest_FINAL.py <cleaned_data> <combined_data>
  python3 ttest_FINAL.py        # uses config_Kim.xlsx
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. Normality test — Shapiro-Wilk per campaign
# ─────────────────────────────────────────────────────────────────────────────
def normality_test(campaign: pd.DataFrame) -> list:
    """
    Shapiro-Wilk normality test on amount distribution per voucher_code.
    Returns list of dicts with test results per campaign.
    Note: Shapiro-Wilk requires n >= 3 and works best with n <= 5000.
    """
    results = []
    overall_mean = campaign['amount'].mean()

    for code, grp in campaign.groupby('voucher_code'):
        amounts = grp['amount'].dropna().values

        if len(amounts) < 3:
            results.append({
                'voucher_code':  code,
                'campaign_source': grp['campaign_source'].iloc[0] if 'campaign_source' in grp.columns else '',
                'n':             len(amounts),
                'mean':          safe_float(np.mean(amounts)) if len(amounts) > 0 else None,
                'std':           None,
                'shapiro_stat':  None,
                'shapiro_p':     None,
                'normal':        None,
                'note':          'Too few observations (n<3)',
            })
            continue

        if len(amounts) < 3:
            results.append({
                'voucher_code':    code,
                'campaign_source': grp['campaign_source'].iloc[0] if 'campaign_source' in grp.columns else '',
                'n':               len(amounts),
                'mean':            safe_float(np.mean(amounts)) if len(amounts) > 0 else None,
                'std':             None,
                'shapiro_stat':    None,
                'shapiro_p':       None,
                'normal':          None,
                'note':            'Too few observations (n<3)',
            })
            continue

        # n > 30: Normal by Central Limit Theorem
        if len(amounts) > 30:
            results.append({
                'voucher_code':    code,
                'campaign_source': grp['campaign_source'].iloc[0] if 'campaign_source' in grp.columns else '',
                'n':               len(amounts),
                'mean':            safe_float(np.mean(amounts)),
                'std':             safe_float(np.std(amounts)),
                'shapiro_stat':    None,
                'shapiro_p':       None,
                'normal':          True,
                'note':            'Normal by CLT (n>30)',
            })
            continue

        # n <= 30: Shapiro-Wilk test
        try:
            stat, p = stats.shapiro(amounts)
            results.append({
                'voucher_code':    code,
                'campaign_source': grp['campaign_source'].iloc[0] if 'campaign_source' in grp.columns else '',
                'n':               len(amounts),
                'mean':            safe_float(np.mean(amounts)),
                'std':             safe_float(np.std(amounts)),
                'shapiro_stat':    safe_float(stat),
                'shapiro_p':       safe_float(p),
                'normal':          bool(p >= 0.05),
                'note':            'Normal (p≥0.05)' if p >= 0.05 else 'Not normal (p<0.05)',
            })
        except Exception as e:
            results.append({
                'voucher_code':    code,
                'campaign_source': grp['campaign_source'].iloc[0] if 'campaign_source' in grp.columns else '',
                'n':               len(amounts),
                'mean':            safe_float(np.mean(amounts)),
                'std':             safe_float(np.std(amounts)),
                'shapiro_stat':    None,
                'shapiro_p':       None,
                'normal':          None,
                'note':            f'Error: {e}',
            })
    return sorted(results, key=lambda x: x.get('n', 0), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. One-sample t-test per campaign
# H0: campaign mean = overall mean revenue
# H1: campaign mean ≠ overall mean (two-tailed)
# ─────────────────────────────────────────────────────────────────────────────
def one_sample_ttest(campaign: pd.DataFrame) -> list:
    """
    Tests whether each campaign's average revenue is significantly
    different from the overall average revenue (population mean).
    """
    overall_mean = safe_float(campaign['amount'].mean())
    results = []

    for code, grp in campaign.groupby('voucher_code'):
        amounts = grp['amount'].dropna().values
        source  = grp['campaign_source'].iloc[0] if 'campaign_source' in grp.columns else ''

        if len(amounts) < 2:
            results.append({
                'voucher_code':    code,
                'campaign_source': source,
                'n':               len(amounts),
                'campaign_mean':   safe_float(np.mean(amounts)) if len(amounts) > 0 else None,
                'overall_mean':    overall_mean,
                'diff_from_mean':  None,
                't_stat':          None,
                'p_value':         None,
                'significant':     None,
                'direction':       None,
                'note':            'Too few observations (n<2)',
            })
            continue

        try:
            t_stat, p_val = stats.ttest_1samp(amounts, overall_mean)
            camp_mean     = np.mean(amounts)
            diff          = camp_mean - overall_mean

            results.append({
                'voucher_code':    code,
                'campaign_source': source,
                'n':               len(amounts),
                'campaign_mean':   safe_float(camp_mean),
                'overall_mean':    overall_mean,
                'diff_from_mean':  safe_float(diff),
                't_stat':          safe_float(t_stat),
                'p_value':         safe_float(p_val),
                'significant':     bool(p_val < 0.05),
                'direction':       'above' if diff > 0 else 'below',
                'note':            (
                    f"Significantly {'above' if diff > 0 else 'below'} overall mean (p={p_val:.4f})"
                    if p_val < 0.05 else f"Not significant (p={p_val:.4f})"
                ),
            })
        except Exception as e:
            results.append({
                'voucher_code':    code,
                'campaign_source': source,
                'n':               len(amounts),
                'campaign_mean':   safe_float(np.mean(amounts)),
                'overall_mean':    overall_mean,
                'diff_from_mean':  None,
                't_stat':          None,
                'p_value':         None,
                'significant':     None,
                'direction':       None,
                'note':            f'Error: {e}',
            })

    # Sort: significant first, then by diff_from_mean descending
    results.sort(key=lambda x: (
        not bool(x.get('significant')),
        -(x.get('diff_from_mean') or 0)
    ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROI per campaign
# ROI = total_amount / total_voucher_value
# ─────────────────────────────────────────────────────────────────────────────
def roi_analysis(campaign: pd.DataFrame) -> list:
    """
    ROI = total transaction amount generated / total voucher value redeemed.
    ROI > 1 means the campaign generated more revenue than its cost.
    """
    results = []

    for code, grp in campaign.groupby('voucher_code'):
        source      = grp['campaign_source'].iloc[0] if 'campaign_source' in grp.columns else ''
        n           = grp['amount'].notna().sum()
        total_amt   = grp['amount'].sum()
        avg_amt     = grp['amount'].mean()

        # Voucher value
        total_voucher = None
        if 'voucher_value' in grp.columns:
            vv = pd.to_numeric(grp['voucher_value'], errors='coerce')
            total_voucher = vv.sum() if vv.notna().any() else None

        roi = safe_float(total_amt / total_voucher) if total_voucher and total_voucher > 0 else None

        results.append({
            'voucher_code':    code,
            'campaign_source': source,
            'n_redemptions':   int(n),
            'total_revenue':   safe_float(total_amt),
            'avg_revenue':     safe_float(avg_amt),
            'total_voucher_cost': safe_float(total_voucher),
            'roi':             roi,
            'roi_label':       (
                '✅ Positive ROI' if roi and roi > 1
                else ('⚠️ Break-even' if roi and roi == 1
                      else ('❌ Negative ROI' if roi and roi < 1 else '—'))
            ),
        })

    # Sort by ROI descending
    results.sort(key=lambda x: x.get('roi') or 0, reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary stats
# ─────────────────────────────────────────────────────────────────────────────
def summary_stats(campaign: pd.DataFrame) -> dict:
    overall_mean = campaign['amount'].mean()
    overall_std  = campaign['amount'].std()
    n_campaigns  = campaign['voucher_code'].nunique()
    n_receipts   = len(campaign.dropna(subset=['amount']))

    return {
        'overall_mean_revenue': safe_float(overall_mean),
        'overall_std_revenue':  safe_float(overall_std),
        'n_campaigns':          int(n_campaigns),
        'n_receipts':           int(n_receipts),
        'n_mall_campaigns':     int(campaign[campaign['campaign_source']=='mall']['voucher_code'].nunique()),
        'n_brand_campaigns':    int(campaign[campaign['campaign_source']=='brand']['voucher_code'].nunique()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(cleaned_folder: str, combined_folder: str):
    print("\n" + "="*60)
    print("T-TEST & ROI ANALYSIS STARTING")
    print("="*60)

    camp_path = os.path.join(cleaned_folder, 'campaign_all.csv')
    if not os.path.exists(camp_path):
        print(f"❌ campaign_all.csv not found in {cleaned_folder}")
        sys.exit(1)

    campaign = pd.read_csv(camp_path)
    campaign['amount'] = pd.to_numeric(campaign.get('amount', pd.Series()), errors='coerce')
    campaign = campaign.dropna(subset=['amount']).reset_index(drop=True)
    print(f"  Campaign loaded: {len(campaign):,} rows (with amount)")

    # Run analyses
    print("\n  Running normality tests (Shapiro-Wilk)...")
    normality  = normality_test(campaign)

    print("  Running one-sample t-tests...")
    ttest      = one_sample_ttest(campaign)

    print("  Calculating ROI per campaign...")
    roi        = roi_analysis(campaign)

    summ       = summary_stats(campaign)

    print(f"\n  Overall mean revenue: ${summ['overall_mean_revenue']:,.2f}")
    print(f"  Total campaigns: {summ['n_campaigns']}")

    sig_above = [r for r in ttest if r.get('significant') and r.get('direction') == 'above']
    sig_below = [r for r in ttest if r.get('significant') and r.get('direction') == 'below']
    print(f"\n  Significantly ABOVE mean: {len(sig_above)} campaigns")
    for r in sig_above[:5]:
        print(f"    {r['voucher_code']}: mean=${r['campaign_mean']:,.0f} (p={r['p_value']})")
    print(f"  Significantly BELOW mean: {len(sig_below)} campaigns")

    top_roi = [r for r in roi if r.get('roi')][:5]
    print(f"\n  Top 5 ROI campaigns:")
    for r in top_roi:
        print(f"    {r['voucher_code']}: ROI={r['roi']:.2f}x")

    # Save
    output = {
        'ttest_analysis': {
            'summary':   summ,
            'normality': normality,
            'ttest':     ttest,
            'roi':       roi,
        }
    }

    os.makedirs(combined_folder, exist_ok=True)
    json_path = os.path.join(combined_folder, 'ttest_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✅ Saved: {json_path}")

    # Save Excel
    xlsx_path = os.path.join(combined_folder, 'ttest_results.xlsx')
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        pd.DataFrame([summ]).to_excel(writer, sheet_name='Summary',   index=False)
        pd.DataFrame(ttest).to_excel(writer, sheet_name='T-Test',     index=False)
        pd.DataFrame(roi).to_excel(  writer, sheet_name='ROI',        index=False)
        pd.DataFrame(normality).to_excel(writer, sheet_name='Normality', index=False)
    print(f"✅ Saved: {xlsx_path}")

    print("\n" + "="*60)
    print("✅ T-TEST & ROI ANALYSIS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2])
    else:
        config_file = sys.argv[1] if len(sys.argv) == 2 else "config_Kim.xlsx"
        script_dir  = Path(__file__).resolve().parent
        df          = pd.read_excel(script_dir / config_file, sheet_name='paths')
        cfg         = dict(zip(df['Setting'].astype(str).str.strip(), df['Value'].astype(str).str.strip()))
        main(cfg.get('cleaned_data', ''), cfg.get('combined_data', ''))