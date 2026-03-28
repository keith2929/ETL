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

    return result