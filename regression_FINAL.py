import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf, q_stat
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def analyse_time_series(campaign: pd.DataFrame, forecast_horizon: int = 24) -> dict:
    """
    Enhanced time series analysis on monthly Amount (member spend).
    
    Returns:
        dict with keys:
            - monthly_amount, mom_trends, moving_average, actual, anomalies, amount_by_source (existing)
            - forecast (list of dicts with forecast, lower, upper)
            - seasonally_adjusted (list of dicts)
            - diagnostics (dict with ADF, Ljung-Box, etc.)
            - decomposition (trend, seasonal, residual as records)
            - model_info (which model used, parameters)
            - month_dummies_regression (existing or enhanced)
            - trend (existing)
    """
    result = {}
    MIN_OBS = 6
    SEASONAL_PERIOD = 12

    if 'amount' not in campaign.columns or 'month_year' not in campaign.columns:
        return {'error': 'amount or month_year column missing'}

    # --- data preparation ---
    camp = campaign.copy()
    camp['amount'] = pd.to_numeric(camp['amount'], errors='coerce')
    monthly = (camp.groupby('month_year')
                   .agg(total_amount=('amount', 'sum'),
                        avg_amount=('amount', 'mean'),
                        txn_count=('receipt_no', 'nunique') if 'receipt_no' in camp.columns else ('amount', 'count'))
                   .reset_index())
    monthly['sort_key'] = pd.to_datetime(monthly['month_year'], format='%b-%Y', errors='coerce')
    monthly = monthly.sort_values('sort_key').reset_index(drop=True)
    if len(monthly) < MIN_OBS:
        return {'error': f"Only {len(monthly)} months of data — need ≥{MIN_OBS}"}

    result['monthly_amount'] = to_records(monthly.drop(columns='sort_key'))
    # MoM % change etc. as before...
    monthly['mom_pct_change'] = monthly['total_amount'].pct_change().mul(100).round(2)
    result['mom_trends'] = to_records(monthly[['month_year','total_amount','avg_amount','txn_count','mom_pct_change']])
    # linear trend for basic summary
    x_idx = np.arange(len(monthly))
    slope, intercept, r, p, _ = stats.linregress(x_idx, monthly['total_amount'].values)
    result['trend'] = {
        'slope': safe_float(slope),
        'r_squared': safe_float(r**2),
        'p_value': safe_float(p),
        'direction': 'upward' if slope > 0 else 'downward',
        'strength': 'strong' if abs(r) > 0.7 else ('moderate' if abs(r) > 0.4 else 'weak'),
        'significant': bool(p < 0.05),
    }
    # moving average (existing)
    monthly['ma3'] = monthly['total_amount'].rolling(3, min_periods=1).mean().round(2)
    result['moving_average'] = to_records(monthly[['month_year', 'ma3']])
    result['actual'] = [{'month_year': r['month_year'], 'value': safe_float(r['total_amount'])} for _, r in monthly.iterrows()]

    # --- seasonal decomposition (STL) ---
    # Use a time series with DatetimeIndex
    series = monthly.set_index('sort_key')['total_amount'].astype(float)
    series = series.asfreq('MS')  # month start frequency
    # STL requires at least 2 full seasonal cycles if seasonal is True, else raise
    if len(series) >= 2 * SEASONAL_PERIOD:
        try:
            stl = STL(series, period=SEASONAL_PERIOD, robust=True).fit()
            seasonal = stl.seasonal
            trend = stl.trend
            resid = stl.resid
            result['decomposition'] = {
                'trend': to_records(pd.DataFrame({'month_year': series.index.strftime('%b-%Y'), 'value': trend.values})),
                'seasonal': to_records(pd.DataFrame({'month_year': series.index.strftime('%b-%Y'), 'value': seasonal.values})),
                'residual': to_records(pd.DataFrame({'month_year': series.index.strftime('%b-%Y'), 'value': resid.values})),
            }
            # seasonally adjusted
            seasonally_adjusted = series - seasonal
            result['seasonally_adjusted'] = to_records(pd.DataFrame({
                'month_year': series.index.strftime('%b-%Y'),
                'value': seasonally_adjusted.values
            }))
        except Exception as e:
            result['decomposition'] = {'error': str(e)}
    else:
        result['decomposition'] = {'note': f'Not enough data for STL (need {2*SEASONAL_PERIOD} months)'}

    # --- Month dummies regression (seasonality significance) ---
    # Same as before, but store for later use if fallback needed
    month_dummies_result = {}
    try:
        monthly['_month_num'] = monthly['sort_key'].dt.month
        month_dummies = pd.get_dummies(monthly['_month_num'], prefix='month', drop_first=True)
        X = sm.add_constant(month_dummies.astype(float))
        y = monthly['total_amount'].astype(float)
        ols = sm.OLS(y, X).fit()
        coef_df = pd.DataFrame({
            'month': ['const'] + [c for c in month_dummies.columns],
            'coef': ols.params.values,
            'p_value': ols.pvalues.values,
            'significant': (ols.pvalues.values < 0.05),
        })
        month_dummies_result = {
            'coef_table': to_records(coef_df),
            'r_squared': safe_float(ols.rsquared),
            'adj_r_squared': safe_float(ols.rsquared_adj),
            'f_pvalue': safe_float(ols.f_pvalue),
            'n_obs': int(ols.nobs),
            'insight': f"Month dummies explain {ols.rsquared*100:.1f}% of variance...",
            'base_month': 'January (month=1)',
        }
    except Exception as e:
        month_dummies_result = {'error': str(e)}
    result['month_dummies_regression'] = month_dummies_result

    # --- Model selection & forecast ---
    n_months = len(series)
    use_seasonal_model = n_months >= 12

    # We'll store forecast outputs
    forecasts = []  # list of dicts: month_year, forecast, lower, upper

    if use_seasonal_model:
        # Use ExponentialSmoothing (Holt-Winters) with seasonal additive
        try:
            # Fit model (trend additive, seasonal additive, period=12)
            model = ExponentialSmoothing(series, trend='add', seasonal='add',
                                         seasonal_periods=SEASONAL_PERIOD,
                                         initialization_method='estimated')
            fitted = model.fit()
            # Forecast horizon steps
            pred = fitted.forecast(forecast_horizon)
            # Get prediction intervals: use fitted.get_prediction if available (statsmodels >= 0.12)
            # For older versions, we compute intervals via simulation or using standard errors
            # Here we use the approach of prediction intervals from the model's residual variance
            residuals = fitted.resid
            sigma = np.std(residuals)  # approximate
            # Confidence level 95%
            z = stats.norm.ppf(0.975)
            lower = pred - z * sigma
            upper = pred + z * sigma
            # Build forecast list
            last_date = series.index[-1]
            for i in range(forecast_horizon):
                fdt = last_date + pd.DateOffset(months=i+1)
                forecasts.append({
                    'month_year': fdt.strftime('%b-%Y'),
                    'forecast': safe_float(pred.iloc[i]),
                    'lower_bound': safe_float(lower.iloc[i]),
                    'upper_bound': safe_float(upper.iloc[i]),
                    'type': 'ets_forecast'
                })
            result['model_info'] = {
                'model': 'ExponentialSmoothing (additive trend + additive seasonality)',
                'params': {k: safe_float(v) for k, v in fitted.params.items() if isinstance(v, (int, float))},
                'aic': safe_float(fitted.aic),
                'bic': safe_float(fitted.bic),
            }
        except Exception as e:
            # Fallback to regression
            use_seasonal_model = False
            result['model_info'] = {'error': str(e), 'fallback': 'regression'}

    if not use_seasonal_model:
        # Use linear trend + month dummies regression for forecasting
        # Build design matrix for all months (including future)
        # We'll use the same month dummies as before
        monthly['time'] = np.arange(len(monthly))
        # Fit OLS: total_amount ~ time + month_dummies
        X_train = pd.concat([monthly[['time']], month_dummies], axis=1)
        X_train = sm.add_constant(X_train)
        y_train = monthly['total_amount'].astype(float)
        try:
            reg = sm.OLS(y_train, X_train).fit()
            # Future time steps (next forecast_horizon months)
            future_time = np.arange(len(monthly), len(monthly) + forecast_horizon)
            # Future month dummies: we need month number for each future month
            last_month_num = monthly['_month_num'].iloc[-1]
            future_months = []
            for i in range(forecast_horizon):
                next_month = (last_month_num + i - 1) % 12 + 1  # wrap around
                future_months.append(next_month)
            # Create dummy columns for future
            month_dummy_cols = month_dummies.columns
            X_future = pd.DataFrame(index=range(forecast_horizon))
            X_future['time'] = future_time
            for col in month_dummy_cols:
                # col is like "month_2", "month_3", ... ; month number is after underscore
                month_num = int(col.split('_')[1])
                X_future[col] = (np.array(future_months) == month_num).astype(int)
            X_future = sm.add_constant(X_future)
            # Predict and get prediction intervals (mean + variance)
            pred_obj = reg.get_prediction(X_future)
            pred_mean = pred_obj.predicted_mean
            pred_ci = pred_obj.conf_int(alpha=0.05)  # returns lower, upper
            # Build forecast list
            last_date = monthly['sort_key'].iloc[-1]
            for i in range(forecast_horizon):
                fdt = last_date + pd.DateOffset(months=i+1)
                forecasts.append({
                    'month_year': fdt.strftime('%b-%Y'),
                    'forecast': safe_float(pred_mean.iloc[i]),
                    'lower_bound': safe_float(pred_ci.iloc[i, 0]),
                    'upper_bound': safe_float(pred_ci.iloc[i, 1]),
                    'type': 'regression_trend_seasonal'
                })
            result['model_info'] = {
                'model': 'Linear trend + month dummies',
                'r_squared': safe_float(reg.rsquared),
                'adj_r_squared': safe_float(reg.rsquared_adj),
                'f_pvalue': safe_float(reg.f_pvalue),
            }
        except Exception as e:
            # Last resort: simple linear extrapolation
            result['model_info'] = {'error': str(e), 'fallback': 'linear_extrapolation'}
            last_val = series.iloc[-1]
            last_date = series.index[-1]
            for i in range(forecast_horizon):
                fdt = last_date + pd.DateOffset(months=i+1)
                forecasts.append({
                    'month_year': fdt.strftime('%b-%Y'),
                    'forecast': safe_float(last_val + slope * (i+1)),
                    'lower_bound': None,
                    'upper_bound': None,
                    'type': 'linear_extrapolation'
                })

    result['forecast'] = forecasts

    # --- Diagnostics ---
    # ADF test for stationarity
    adf_result = adfuller(series.dropna(), autolag='AIC')
    result['diagnostics'] = {
        'adf_statistic': safe_float(adf_result[0]),
        'adf_pvalue': safe_float(adf_result[1]),
        'adf_is_stationary': bool(adf_result[1] < 0.05),
        'adf_used_lag': adf_result[2],
    }
    # Residuals from best model (if we have a model)
    if 'model_info' in result and 'model' in result['model_info']:
        if use_seasonal_model and 'fitted' in locals():
            res = fitted.resid
            # Ljung-Box test
            ljung_box = acf(res, nlags=min(20, len(res)-1), qstat=True)
            lb_stat = ljung_box[1]
            lb_pvalue = ljung_box[2]
            result['diagnostics']['ljung_box_stat'] = safe_float(lb_stat[-1])
            result['diagnostics']['ljung_box_pvalue'] = safe_float(lb_pvalue[-1])
            result['diagnostics']['residuals_autocorrelation'] = 'significant' if lb_pvalue[-1] < 0.05 else 'not significant'
        elif 'reg' in locals():
            res = reg.resid
            ljung_box = acf(res, nlags=min(20, len(res)-1), qstat=True)
            lb_stat = ljung_box[1]
            lb_pvalue = ljung_box[2]
            result['diagnostics']['ljung_box_stat'] = safe_float(lb_stat[-1])
            result['diagnostics']['ljung_box_pvalue'] = safe_float(lb_pvalue[-1])
            result['diagnostics']['residuals_autocorrelation'] = 'significant' if lb_pvalue[-1] < 0.05 else 'not significant'
    # Durbin-Watson from existing OLS if available
    if 'month_dummies_regression' in result and 'coef_table' in result['month_dummies_regression']:
        # compute DW from the OLS model we already ran earlier
        try:
            dw = sm.stats.stattools.durbin_watson(ols.resid)
            result['diagnostics']['durbin_watson'] = safe_float(dw)
        except:
            pass

    # --- Anomaly detection (improved using residuals from best model) ---
    # Use residuals from decomposition if available, else use trend residuals
    if 'decomposition' in result and 'residual' in result['decomposition']:
        # residuals from STL
        resid_series = pd.Series([r['value'] for r in result['decomposition']['residual']])
        mean_resid = resid_series.mean()
        std_resid = resid_series.std()
        anomalies = []
        for i, row in monthly.iterrows():
            res = resid_series.iloc[i]
            if abs(res - mean_resid) > 2 * std_resid:
                anomalies.append({
                    'month_year': row['month_year'],
                    'actual': safe_float(row['total_amount']),
                    'residual': safe_float(res),
                    'direction': 'above' if res > mean_resid else 'below',
                })
        result['anomalies'] = anomalies
    else:
        # fallback: linear trend residuals
        residuals = monthly['total_amount'] - (intercept + slope * x_idx)
        std_resid = residuals.std()
        mean_resid = residuals.mean()
        anomalies = []
        for i, row in monthly.iterrows():
            res = residuals.iloc[i]
            if abs(res - mean_resid) > 2 * std_resid:
                anomalies.append({
                    'month_year': row['month_year'],
                    'actual': safe_float(row['total_amount']),
                    'residual': safe_float(res),
                    'direction': 'above' if res > mean_resid else 'below',
                })
        result['anomalies'] = anomalies

    # --- Campaign source breakdown (existing) ---
    if 'campaign_source' in campaign.columns:
        by_source = (camp.groupby(['month_year', 'campaign_source'])['amount']
                         .sum().reset_index())
        by_source['sort_key'] = pd.to_datetime(by_source['month_year'], format='%b-%Y', errors='coerce')
        by_source = by_source.sort_values('sort_key').drop(columns='sort_key')
        result['amount_by_source'] = to_records(by_source)

    # ============================================================================
    # Forecast by campaign source (Mall vs Brand)
    # ============================================================================
    if 'campaign_source' in campaign.columns:
        source_forecasts = {}
        # Prepare monthly totals per source
        source_monthly = (campaign.groupby(['month_year', 'campaign_source'])['amount']
                          .sum().reset_index())
        source_monthly['sort_key'] = pd.to_datetime(source_monthly['month_year'],
                                                     format='%b-%Y', errors='coerce')
        source_monthly = source_monthly.sort_values('sort_key').dropna(subset=['sort_key'])

        for source in ['mall', 'brand']:
            df_src = source_monthly[source_monthly['campaign_source'] == source].copy()
            if len(df_src) < MIN_OBS:
                source_forecasts[source] = {'error': f'Only {len(df_src)} months of data, need \u2265{MIN_OBS}'}
                continue

            # Create time series
            series_src = df_src.set_index('sort_key')['amount'].astype(float).asfreq('MS')
            n_months_src = len(series_src)
            use_seasonal_src = n_months_src >= 12

            forecasts_src = []
            if use_seasonal_src:
                try:
                    model_src = ExponentialSmoothing(series_src, trend='add', seasonal='add',
                                                     seasonal_periods=12,
                                                     initialization_method='estimated').fit()
                    pred_src   = model_src.forecast(forecast_horizon)
                    resid_src  = model_src.resid
                    fitted_src = model_src.fittedvalues

                    # CV-based sigma: std(resid) / mean(|fitted|)
                    # Keeps the band proportional to forecast magnitude regardless of
                    # the raw $ scale of the series (avoids massive bands on high-value sources).
                    mean_fitted = np.abs(fitted_src).mean()
                    sigma_abs   = np.std(resid_src, ddof=1)
                    cv_src      = sigma_abs / mean_fitted if mean_fitted > 0 else 0.15
                    cv_src      = min(cv_src, 0.40)   # cap at 40% so outliers don't explode band
                    z = stats.norm.ppf(0.975)

                    last_date = series_src.index[-1]
                    for i in range(forecast_horizon):
                        fdt    = last_date + pd.DateOffset(months=i + 1)
                        h      = i + 1
                        y_pred = pred_src.iloc[i]
                        # Proportional margin fanning out with horizon
                        margin = z * cv_src * abs(y_pred) * np.sqrt(h)
                        forecasts_src.append({
                            'month_year':  fdt.strftime('%b-%Y'),
                            'forecast':    safe_float(y_pred),
                            'lower_bound': safe_float(max(y_pred - margin, 0)),
                            'upper_bound': safe_float(y_pred + margin),
                            'type': 'ets_forecast'
                        })
                    source_forecasts[source] = {
                        'actual': to_records(df_src[['month_year', 'amount']].rename(columns={'amount': 'value'})),
                        'forecast': forecasts_src,
                        'model': 'ETS (additive)'
                    }
                except Exception as e:
                    use_seasonal_src = False
                    source_forecasts[source] = {'error': str(e), 'fallback': 'regression'}

            if not use_seasonal_src:
                # Fallback: linear trend + month dummies
                try:
                    df_src = df_src.copy()
                    df_src['time'] = np.arange(len(df_src))
                    df_src['_month_num'] = df_src['sort_key'].dt.month
                    month_dummies_src = pd.get_dummies(df_src['_month_num'], prefix='month', drop_first=True)
                    X_train_src = pd.concat([df_src[['time']], month_dummies_src], axis=1)
                    X_train_src = sm.add_constant(X_train_src)
                    y_train_src = df_src['amount'].astype(float)
                    reg_src = sm.OLS(y_train_src, X_train_src).fit()

                    # Future data
                    future_time_src = np.arange(len(df_src), len(df_src) + forecast_horizon)
                    last_month_num_src = df_src['_month_num'].iloc[-1]
                    future_months_src = [(last_month_num_src + i - 1) % 12 + 1 for i in range(forecast_horizon)]
                    X_future_src = pd.DataFrame(index=range(forecast_horizon))
                    X_future_src['time'] = future_time_src
                    for col in month_dummies_src.columns:
                        month_num = int(col.split('_')[1])
                        X_future_src[col] = (np.array(future_months_src) == month_num).astype(int)
                    X_future_src = sm.add_constant(X_future_src)

                    pred_obj_src = reg_src.get_prediction(X_future_src)
                    pred_mean_src = pred_obj_src.predicted_mean
                    pred_ci_src = pred_obj_src.conf_int(alpha=0.05)

                    last_date = df_src['sort_key'].iloc[-1]
                    for i in range(forecast_horizon):
                        fdt = last_date + pd.DateOffset(months=i+1)
                        forecasts_src.append({
                            'month_year': fdt.strftime('%b-%Y'),
                            'forecast': safe_float(pred_mean_src.iloc[i]),
                            'lower_bound': safe_float(pred_ci_src.iloc[i, 0]),
                            'upper_bound': safe_float(pred_ci_src.iloc[i, 1]),
                            'type': 'regression_trend_seasonal'
                        })
                    source_forecasts[source] = {
                        'actual': to_records(df_src[['month_year', 'amount']].rename(columns={'amount': 'value'})),
                        'forecast': forecasts_src,
                        'model': 'Linear trend + month dummies'
                    }
                except Exception as e:
                    # Last resort: linear extrapolation with prediction interval
                    # PI formula: ŷ ± t * s * sqrt(1 + 1/n + (x_new - x̄)² / Σ(xᵢ - x̄)²)
                    x_train = np.arange(len(df_src), dtype=float)
                    y_train = df_src['amount'].values.astype(float)
                    slope_src, intercept_src, _, _, _ = stats.linregress(x_train, y_train)
                    y_fitted = intercept_src + slope_src * x_train
                    n_src = len(x_train)
                    # Residual standard error (df = n - 2 for simple linear regression)
                    rse_src = np.sqrt(np.sum((y_train - y_fitted) ** 2) / max(n_src - 2, 1))
                    x_mean_src = x_train.mean()
                    ss_xx_src  = np.sum((x_train - x_mean_src) ** 2) or 1.0
                    # t critical value at 95% (two-tailed), df = n-2; fall back to z if too few
                    t_crit_src = stats.t.ppf(0.975, df=max(n_src - 2, 1))
                    last_val_src = df_src['amount'].iloc[-1]
                    last_date = df_src['sort_key'].iloc[-1]
                    forecasts_src = []
                    for i in range(forecast_horizon):
                        fdt = last_date + pd.DateOffset(months=i+1)
                        x_new  = n_src + i       # next time index beyond training
                        y_pred = intercept_src + slope_src * x_new
                        # Prediction SE grows with distance from training data
                        se_pred = rse_src * np.sqrt(1 + 1/n_src + (x_new - x_mean_src)**2 / ss_xx_src)
                        margin  = t_crit_src * se_pred
                        forecasts_src.append({
                            'month_year':  fdt.strftime('%b-%Y'),
                            'forecast':    safe_float(y_pred),
                            'lower_bound': safe_float(y_pred - margin),
                            'upper_bound': safe_float(y_pred + margin),
                            'type': 'linear_extrapolation'
                        })
                    source_forecasts[source] = {
                        'actual': to_records(df_src[['month_year', 'amount']].rename(columns={'amount': 'value'})),
                        'forecast': forecasts_src,
                        'model': 'Linear extrapolation (fallback)'
                    }

        result['source_forecasts'] = source_forecasts

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def safe_float(val):
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except:
        return None

def to_records(df):
    if df is None or df.empty: return []
    import json
    return json.loads(df.replace([np.inf, -np.inf], np.nan).to_json(orient='records'))

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(cleaned_folder: str, combined_folder: str, raw_folder: str = ''):
    """
    GTO loading is now done in data_load_FINAL.py, which writes:
      gto_member_sales.csv
      gto_nonmember_sales.csv
      gto_member_sales_forecast.csv
      gto_nonmember_sales_forecast.csv
    into the cleaned_data folder.  This function reads those CSVs directly.
    The raw_folder argument is accepted for backwards-compatibility but ignored.
    """
    import os, sys, json
    from pathlib import Path

    print("\n" + "="*60)
    print("TIME SERIES & MEMBER/NON-MEMBER ANALYSIS STARTING")
    print("="*60)

    camp_path = os.path.join(cleaned_folder, 'campaign_all.csv')
    if not os.path.exists(camp_path):
        print(f"ERROR campaign_all.csv not found: {camp_path}")
        sys.exit(1)

    campaign = pd.read_csv(camp_path)
    campaign['amount'] = pd.to_numeric(campaign.get('amount', pd.Series()), errors='coerce')
    print(f"  Campaign: {len(campaign):,} rows")

    # ── Time Series ───────────────────────────────────────────────────────
    ts_result = analyse_time_series(campaign)

    # ── Member / Non-Member Sales — read CSVs produced by data_load_FINAL ─
    member_nonmember = {}

    ms_path   = os.path.join(cleaned_folder, 'gto_member_sales.csv')
    nms_path  = os.path.join(cleaned_folder, 'gto_nonmember_sales.csv')
    msf_path  = os.path.join(cleaned_folder, 'gto_member_sales_forecast.csv')
    nmsf_path = os.path.join(cleaned_folder, 'gto_nonmember_sales_forecast.csv')

    if os.path.exists(ms_path) and os.path.exists(nms_path):
        member_monthly    = pd.read_csv(ms_path)
        nonmember_monthly = pd.read_csv(nms_path)
        member_fore       = pd.read_csv(msf_path).to_dict('records')    if os.path.exists(msf_path)  else []
        nonmember_fore    = pd.read_csv(nmsf_path).to_dict('records')   if os.path.exists(nmsf_path) else []

        member_nonmember = {
            'member_sales': {
                'actual':   [{'month_year': r['month_year'], 'value': safe_float(r['member_sales'])}
                             for _, r in member_monthly.iterrows()],
                'forecast': member_fore,
            },
            'non_member_sales': {
                'actual':   [{'month_year': r['month_year'], 'value': safe_float(r['non_member_sales'])}
                             for _, r in nonmember_monthly.iterrows()],
                'forecast': nonmember_fore,
            },
        }
        print(f"  Member Sales months loaded:     {len(member_monthly)}")
        print(f"  Non-Member Sales months loaded: {len(nonmember_monthly)}")
    else:
        print("  INFO: gto_member_sales.csv not found — member/non-member section will be empty.")
        print("        Re-run data_load_FINAL.py with a GTO file present to generate it.")

    # ── Save ─────────────────────────────────────────────────────────────
    insights = {
        'time_series':      ts_result,
        'member_nonmember': member_nonmember,
    }

    os.makedirs(combined_folder, exist_ok=True)
    json_path = os.path.join(combined_folder, 'insights.json')
    with open(json_path, 'w') as f:
        json.dump(insights, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    # ── Excel ─────────────────────────────────────────────────────────────
    report_path = os.path.join(combined_folder, 'insights_report.xlsx')
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        for sheet, records in [
            ('Monthly_Amount',   ts_result.get('monthly_amount')),
            ('Month_Regression', ts_result.get('month_dummies_regression', {}).get('coef_table')),
            ('MoM_Trends',       ts_result.get('mom_trends')),
            ('Forecast',         ts_result.get('forecast')),
            ('Anomalies',        ts_result.get('anomalies')),
            ('Amount_by_Source', ts_result.get('amount_by_source')),
        ]:
            if records:
                pd.DataFrame(records).to_excel(writer, sheet_name=sheet, index=False)
        # Source forecast sheets (Mall_Forecast, Brand_Forecast)
        if 'source_forecasts' in ts_result:
            for src, data in ts_result['source_forecasts'].items():
                if 'forecast' in data and data['forecast']:
                    sheet_name = f'{src.title()}_Forecast'
                    pd.DataFrame(data['forecast']).to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"  Saved: {report_path}")

    trend = ts_result.get('trend', {})
    print(f"\n  Trend: {trend.get('direction','—')} ({trend.get('strength','—')}), "
          f"R²={trend.get('r_squared','—')}, p={trend.get('p_value','—')}")

    print("\n" + "="*60)
    print("TIME SERIES COMPLETED")
    print("="*60)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else '')
    else:
        import pandas as pd
        config_file = sys.argv[1] if len(sys.argv) == 2 else "config_Keith.xlsx"
        script_dir  = Path(__file__).resolve().parent
        df          = pd.read_excel(script_dir / config_file, sheet_name='paths')
        cfg         = dict(zip(df['Setting'].astype(str).str.strip(), df['Value'].astype(str).str.strip()))
        main(cfg.get('cleaned_data', ''), cfg.get('combined_data', ''), cfg.get('raw_data', ''))