import streamlit as st
import pandas as pd
import subprocess, sys, os, threading, queue, time, json
from pathlib import Path

st.set_page_config(page_title="Capstone Pipeline", page_icon="⚡",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');
:root{--bg:#0e0f11;--surface:#16181c;--border:#2a2d35;--accent:#c8f135;--accent2:#4af0c4;--text:#e8eaf0;--muted:#6b7280;--danger:#ff5c5c;--warn:#fbbf24;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Mono',monospace!important;}
[data-testid="stAppViewContainer"]{padding:0!important;}
[data-testid="stHeader"]{display:none;}[data-testid="stSidebar"]{display:none;}
[data-testid="block-container"]{padding:2rem 3rem!important;max-width:1400px!important;}
.app-header{display:flex;align-items:baseline;gap:1rem;margin-bottom:2.5rem;border-bottom:1px solid var(--border);padding-bottom:1.5rem;}
.app-title{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:var(--accent);letter-spacing:-.03em;margin:0;}
.app-sub{font-size:.75rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;}
[data-testid="stTabs"] [role="tablist"]{display:flex!important;border-bottom:1px solid var(--border)!important;margin-bottom:2rem!important;}
[data-testid="stTabs"] [role="tab"]{font-family:'DM Mono',monospace!important;font-size:.75rem!important;letter-spacing:.1em!important;text-transform:uppercase!important;color:var(--muted)!important;border:none!important;background:transparent!important;padding:.5rem 1.2rem!important;border-radius:0!important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;}
.log-box{background:#0a0b0d;border:1px solid var(--border);border-radius:6px;padding:1rem 1.2rem;font-family:'DM Mono',monospace;font-size:.78rem;line-height:1.7;max-height:480px;overflow-y:auto;white-space:pre-wrap;color:#b0b8c8;}
[data-testid="stTextInput"] input{background:var(--surface)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:6px!important;font-family:'DM Mono',monospace!important;}
label{color:var(--muted)!important;font-size:.72rem!important;letter-spacing:.08em!important;text-transform:uppercase!important;}
[data-testid="baseButton-primary"]{background:var(--accent)!important;color:#0e0f11!important;border:none!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:.82rem!important;border-radius:6px!important;padding:.6rem 1.5rem!important;}
[data-testid="stSuccess"]{background:#1a2e1a!important;border:1px solid #2a4a2a!important;color:var(--accent2)!important;border-radius:6px!important;}
[data-testid="stError"]{background:#2e1a1a!important;border:1px solid #4a1a1a!important;color:var(--danger)!important;border-radius:6px!important;}
[data-testid="stInfo"]{background:#1a222e!important;border:1px solid #1a3a4a!important;color:#60d0ff!important;border-radius:6px!important;}
[data-testid="stWarning"]{background:#2e2a1a!important;border:1px solid #4a3a1a!important;color:var(--warn)!important;border-radius:6px!important;}
hr{border-color:var(--border)!important;margin:1.5rem 0!important;}
.insight-box{background:#1a1f2e;border-left:3px solid var(--accent);border-radius:0 6px 6px 0;padding:.8rem 1.2rem;font-size:.82rem;color:var(--text);margin:.8rem 0;line-height:1.6;}
</style>
""", unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parent

def get_configs():
    return sorted([f.name for f in APP_DIR.glob("config_*.xlsx")])

def load_config(config_file):
    path = APP_DIR / config_file
    if not path.exists(): return {}
    df = pd.read_excel(path, sheet_name='paths')
    return dict(zip(df['Setting'].astype(str).str.strip(), df['Value'].astype(str).str.strip()))

def run_pipeline(config_file, log_queue):
    process = subprocess.Popen(
        [sys.executable, str(APP_DIR / 'main_FINAL.py'), config_file],
        cwd=str(APP_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        log_queue.put(line)
    process.wait()
    log_queue.put(f"__EXIT__{process.returncode}")

for k, v in [('log_lines',[]),('last_exit',None),('running',False)]:
    if k not in st.session_state: st.session_state[k] = v

st.markdown('<div class="app-header"><div class="app-title">⚡ PIPELINE</div><div class="app-sub">ETL · Regression · Time Series · Capstone 2025</div></div>', unsafe_allow_html=True)

configs = get_configs()
if not configs:
    st.error("No config_*.xlsx file found.")
    st.stop()

_NEW = "＋  Add new config..."
c1, c2 = st.columns([3,2])
with c1:
    choice = st.selectbox("Config", configs + [_NEW], key="global_config", label_visibility="collapsed")
with c2:
    if choice == _NEW:
        st.text_input("New config name", placeholder="e.g. Alice", key="new_config_name", label_visibility="collapsed")
    else:
        st.empty()

if choice == _NEW:
    raw = st.session_state.get("new_config_name","").strip().removeprefix("config_").removesuffix(".xlsx")
    selected_config = f"config_{raw}.xlsx" if raw else None
    is_new = True
else:
    selected_config = choice
    is_new = False

if not selected_config:
    st.info("Enter a config name above.")
    st.stop()

st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

tab_run, tab_config, tab_ts, tab_reg = st.tabs([
    "▶  RUN", "⚙  CONFIG", "📈  TIME SERIES", "📉  REGRESSION"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run:
    run_clicked = st.button("▶  RUN PIPELINE", type="primary", use_container_width=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if run_clicked and not st.session_state.running:
        st.session_state.log_lines = []
        st.session_state.last_exit = None
        st.session_state.running   = True
        q = queue.Queue()
        st.session_state.log_queue = q
        threading.Thread(target=run_pipeline, args=(selected_config, q), daemon=True).start()

    status_ph = st.empty()
    log_ph    = st.empty()

    if st.session_state.running:
        q = st.session_state.log_queue
        while True:
            try:
                line = q.get_nowait()
                if line.startswith("__EXIT__"):
                    st.session_state.last_exit = int(line.replace("__EXIT__",""))
                    st.session_state.running   = False
                    break
                st.session_state.log_lines.append(line.rstrip())
            except queue.Empty:
                break

    if st.session_state.running or st.session_state.log_lines or st.session_state.last_exit is not None:
        all_log  = "\n".join(st.session_state.log_lines)
        etl_done = "ETL COMPLETED"               in all_log
        ts_done  = "TIME SERIES" in all_log and "COMPLETED" in all_log
        reg_done = "LINEAR REGRESSION COMPLETED" in all_log

        if st.session_state.running:
            if reg_done:   status_ph.info("⏳ Finalising...")
            elif ts_done:  status_ph.info("✅ Time Series done — running Linear Regression...")
            elif etl_done: status_ph.info("✅ ETL done — running Time Series...")
            else:          status_ph.info("⏳ Running ETL...")
        elif st.session_state.last_exit == 0:
            status_ph.success("✅ ETL  ·  ✅ Time Series  ·  ✅ Linear Regression — all complete!")
        elif st.session_state.last_exit is not None:
            errs = [l for l in st.session_state.log_lines if 'error' in l.lower()]
            status_ph.error(f"❌ Pipeline failed — {errs[-1] if errs else 'Check logs'}")

        if st.session_state.log_lines:
            _hide = ('DEBUG ','Loading configuration','raw_data from','cleaned_data from',
                     'schemas from','FutureWarning','DeprecationWarning','UserWarning','warnings.warn')
            visible = [l for l in st.session_state.log_lines[-300:] if not any(p in l for p in _hide)]
            log_ph.markdown(f'<div class="log-box">{chr(10).join(visible)}</div>', unsafe_allow_html=True)

        if st.session_state.running:
            time.sleep(0.8)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
with tab_config:
    paths = load_config(selected_config) if not is_new else {}
    if is_new:
        st.info(f"New config: **{selected_config}** — fill in paths and click Save.")

    path_labels = {
        'raw_data':      'Raw Data Folder',
        'cleaned_data':  'Cleaned Data Folder',
        'combined_data': 'Combined Data Folder',
    }
    new_paths = {}
    for key, label in path_labels.items():
        c1, c2 = st.columns([5,1])
        with c1:
            new_paths[key] = st.text_input(label, value=paths.get(key,''), key=f"path_{key}")
        with c2:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            if st.button("📂", key=f"open_{key}"):
                p = Path(new_paths[key].strip())
                walk = p
                while walk != walk.parent and not walk.exists(): walk = walk.parent
                if sys.platform == 'win32':    subprocess.Popen(f'explorer "{walk}"')
                elif sys.platform == 'darwin': subprocess.Popen(['open', str(walk)])
                else:                          subprocess.Popen(['xdg-open', str(walk)])

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    if st.button("💾  Save Config", type="primary"):
        with pd.ExcelWriter(APP_DIR / selected_config, engine='openpyxl') as writer:
            pd.DataFrame([{'Setting':k,'Value':v} for k,v in new_paths.items()]).to_excel(
                writer, sheet_name='paths', index=False)
        st.success(f"✅ Saved {selected_config}")
        if is_new: st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ts:
    paths_ts    = load_config(selected_config)
    combined_ts = paths_ts.get('combined_data','')
    json_ts     = os.path.join(combined_ts, 'insights.json') if combined_ts else ''

    if not json_ts or not Path(json_ts).exists():
        st.info("No time series results yet — run the pipeline first.")
    else:
        with open(json_ts) as f:
            ins = json.load(f)

        ts = ins.get('time_series', {})
        if ts.get('error'):
            st.warning(ts['error'])
        else:
            # ── Monthly Amount trend ──────────────────────────────────────
            st.markdown("### 📈 Monthly Member Spend (Amount)")

            actual   = ts.get('actual', [])
            forecast = ts.get('forecast', [])

            if actual:
                df_act  = pd.DataFrame(actual).rename(columns={'value':'Actual Amount'})
                df_fore = pd.DataFrame(forecast).rename(columns={'forecast':'Forecast'}) if forecast else pd.DataFrame()
                df_chart = df_act.set_index('month_year')[['Actual Amount']]
                if not df_fore.empty:
                    df_chart = df_chart.join(df_fore.set_index('month_year')[['Forecast']], how='outer')
                st.line_chart(df_chart)

            trend = ts.get('trend', {})
            if trend:
                c1, c2, c3 = st.columns(3)
                c1.metric("Trend Direction", trend.get('direction','—').title())
                c2.metric("Trend Strength",  trend.get('strength','—').title())
                c3.metric("R²",              f"{trend.get('r_squared') or 0:.3f}")
                st.caption("✅ Significant (p<0.05)" if trend.get('significant')
                           else "⚠️ Not significant (p≥0.05)")

            st.markdown("---")

            # ── Month dummies regression ──────────────────────────────────
            st.markdown("### 📅 Seasonality — Month Dummies Regression")
            st.caption("Which months have significantly higher/lower spend? Base = January")

            mdr = ts.get('month_dummies_regression', {})
            if mdr.get('error'):
                st.warning(mdr['error'])
            elif mdr.get('coef_table'):
                if mdr.get('insight'):
                    st.markdown(f'<div class="insight-box">💡 {mdr["insight"]}</div>', unsafe_allow_html=True)

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("R²",       f"{mdr.get('r_squared') or 0:.3f}")
                c2.metric("Adj R²",   f"{mdr.get('adj_r_squared') or 0:.3f}")
                c3.metric("F p-value",f"{mdr.get('f_pvalue') or 0:.4f}")
                c4.metric("N obs",    mdr.get('n_obs','—'))

                df_coef      = pd.DataFrame(mdr['coef_table'])
                df_coef_plot = df_coef[df_coef['month'] != 'const'].copy()

                col_chart, col_table = st.columns([2,3])
                with col_chart:
                    if not df_coef_plot.empty:
                        st.markdown("**Coefficient by Month**")
                        st.bar_chart(df_coef_plot.set_index('month')['coef'])
                with col_table:
                    def hl_month(row):
                        return ['background-color:#3d3800; color:#fbbf24']*len(row) if row.get('significant') else ['']*len(row)
                    st.dataframe(
                        df_coef[['month','coef','p_value','significant']].style.apply(hl_month, axis=1),
                        use_container_width=True, hide_index=True
                    )
                st.caption("✅ Green = significant at p<0.05. Positive = higher spend than January.")

            st.markdown("---")

            # ── MoM trends ────────────────────────────────────────────────
            st.markdown("### 📊 Month-on-Month Trends")
            mom = ts.get('mom_trends', [])
            if mom:
                st.dataframe(pd.DataFrame(mom), use_container_width=True, hide_index=True)

            st.markdown("---")

            # ── Amount by source ──────────────────────────────────────────
            by_source = ts.get('amount_by_source', [])
            if by_source:
                st.markdown("### 🏷️ Spend by Campaign Source (Mall vs Brand)")
                df_src = pd.DataFrame(by_source)
                if 'campaign_source' in df_src.columns and 'month_year' in df_src.columns:
                    try:
                        pivoted = df_src.pivot(index='month_year', columns='campaign_source',
                                               values='amount').fillna(0)
                        st.bar_chart(pivoted)
                    except:
                        st.dataframe(df_src, use_container_width=True, hide_index=True)

            st.markdown("---")

            # ── Anomalies ─────────────────────────────────────────────────
            anomalies = ts.get('anomalies', [])
            if anomalies:
                st.markdown("### ⚠️ Anomalous Months")
                st.dataframe(pd.DataFrame(anomalies), use_container_width=True, hide_index=True)
            else:
                st.caption("No anomalous months detected.")

            st.markdown("---")
            report = os.path.join(combined_ts, 'insights_report.xlsx')
            if Path(report).exists():
                with open(report,'rb') as fh:
                    st.download_button("⬇️  Download Time Series Report (.xlsx)",
                                       data=fh.read(), file_name='insights_report.xlsx',
                                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                       type='primary')


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reg:
    paths_reg    = load_config(selected_config)
    combined_reg = paths_reg.get('combined_data','')
    reg_json     = os.path.join(combined_reg, 'linear_regression_results.json') if combined_reg else ''
    reg_xlsx     = os.path.join(combined_reg, 'linear_regression_summary.xlsx') if combined_reg else ''

    if not reg_json or not Path(reg_json).exists():
        st.info("No regression results yet — run the pipeline first.")
        st.stop()

    with open(reg_json) as f:
        reg_data = json.load(f)

    lr = reg_data.get('linear_regression', {})

    # ── Summary ───────────────────────────────────────────────────────────
    st.markdown("### 📊 Model Comparison Summary")
    summary = lr.get('summary', [])
    if summary:
        df_sum = pd.DataFrame(summary)
        disp   = [c for c in ['model_key','label','n_obs','r_squared',
                               'adj_r_squared','f_pvalue','cv_r2','cv_rmse'] if c in df_sum.columns]
        st.dataframe(
            df_sum[disp], use_container_width=True, hide_index=True,
            column_config={
                'r_squared':     st.column_config.ProgressColumn('R²',     min_value=0, max_value=1, format='%.3f'),
                'adj_r_squared': st.column_config.ProgressColumn('Adj R²', min_value=0, max_value=1, format='%.3f'),
                'cv_r2':         st.column_config.ProgressColumn('CV R²',  min_value=0, max_value=1, format='%.3f'),
            }
        )
        valid = df_sum.dropna(subset=['adj_r_squared'])
        if not valid.empty:
            best = valid.loc[valid['adj_r_squared'].idxmax()]
            st.markdown(
                f'<div class="insight-box">🏆 Best model: <strong>{best["model_key"]}</strong> — '
                f'Adj-R² = <strong>{best["adj_r_squared"]:.3f}</strong></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Model selector ────────────────────────────────────────────────────
    model_options = {
        'Regression 1 — Y = Amount (outlet×month)': 'regression_1_amount',
        'Regression 2 — Y = Amount (per receipt)':  'regression_2_amount',
    }
    selected_label = st.radio("Select model", list(model_options.keys()), horizontal=True)
    model = lr.get(model_options[selected_label], {})

    if model.get('error'):
        st.error(f"Model error: {model['error']}")
        st.stop()

    if model.get('insight'):
        st.markdown(f'<div class="insight-box">💡 {model["insight"]}</div>', unsafe_allow_html=True)

    # ── Model fit ─────────────────────────────────────────────────────────
    st.markdown("#### Model Fit")
    fit = model.get('model_fit', {})
    cv  = model.get('cross_validation', {})

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("R²",       f"{fit.get('r_squared') or 0:.3f}")
    c2.metric("Adj R²",   f"{fit.get('adj_r_squared') or 0:.3f}")
    c3.metric("F p-value",f"{fit.get('f_pvalue') or 0:.4f}")
    c4.metric("N obs",    fit.get('n_obs','—'))
    c5.metric("CV RMSE",  f"{cv.get('cv_rmse_mean') or 0:,.1f}" if cv.get('cv_rmse_mean') else "—")
    c6.metric("CV R²",    f"{cv.get('cv_r2_mean') or 0:.3f}"    if cv.get('cv_r2_mean')   else "—")

    dw = fit.get('dw_stat')
    if dw:
        st.caption(f"Durbin-Watson: {dw:.3f} — "
                   + ("✅ No autocorrelation" if 1.5 < dw < 2.5 else "⚠️ Possible autocorrelation"))

    st.markdown("---")

    # ── Feature groups ────────────────────────────────────────────────────
    fg = model.get('feature_groups', {})
    if fg:
        cols = st.columns(len(fg))
        for col, (group, feats) in zip(cols, fg.items()):
            with col:
                st.markdown(f"**{group.title()} ({len(feats)})**")
                st.caption(", ".join(feats[:10]) + ("..." if len(feats) > 10 else ""))

    st.markdown("---")

    # ── Coefficients ──────────────────────────────────────────────────────
    st.markdown("#### Coefficients (standardised β)")
    coef_table = model.get('coef_table', [])
    if coef_table:
        df_coef      = pd.DataFrame(coef_table)
        df_coef_plot = df_coef[df_coef['feature'] != 'const'].copy()

        show_sig = st.checkbox("Show significant predictors only (p<0.05)", value=False)
        if show_sig:
            df_coef_plot = df_coef_plot[df_coef_plot['significant'] == True]

        col_chart, col_table = st.columns([2,3])
        with col_chart:
            if not df_coef_plot.empty:
                top20 = df_coef_plot.nlargest(20, 'coef', keep='all')
                st.markdown("**Top coefficients**")
                st.bar_chart(top20.set_index('feature')['coef'])
        with col_table:
            disp_cols = [c for c in ['feature','coef','std_err','t_stat',
                                      'p_value','ci_lower','ci_upper','significant']
                         if c in df_coef.columns]
            def hl(row):
                return ['background-color:#3d3800; color:#fbbf24']*len(row) if row.get('significant') else ['']*len(row)
            st.dataframe(
                df_coef[disp_cols].style.apply(hl, axis=1),
                use_container_width=True, hide_index=True
            )
        st.caption("✅ Green = significant at p<0.05. Standardised β — comparable in magnitude.")

    st.markdown("---")

    # ── VIF ───────────────────────────────────────────────────────────────
    vif = model.get('vif', [])
    if vif:
        st.markdown("#### Multicollinearity (VIF)")
        df_vif = pd.DataFrame(vif)
        c1, c2 = st.columns([1,2])
        with c1: st.dataframe(df_vif, use_container_width=True, hide_index=True)
        with c2:
            st.caption("VIF < 5 → ✅ OK  |  5–10 → ⚠️ Moderate  |  >10 → ❌ High")
            if not df_vif.empty and 'vif' in df_vif.columns:
                mv = df_vif['vif'].dropna().max()
                if mv and mv > 10:  st.warning(f"⚠️ Max VIF={mv:.1f} — high multicollinearity")
                elif mv and mv > 5: st.warning(f"⚠️ Max VIF={mv:.1f} — moderate multicollinearity")
                else:               st.success("✅ All VIF < 5")

    st.markdown("---")

    # ── Residuals ─────────────────────────────────────────────────────────
    residuals = model.get('residuals', [])
    if residuals:
        st.markdown("#### Residual Diagnostics")
        df_resid = pd.DataFrame(residuals)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Fitted vs Residuals**")
            st.scatter_chart(df_resid, x='fitted', y='residual', height=280)
            st.caption("Ideal: randomly scattered around 0")
        with c2:
            st.markdown("**Standardised Residuals Distribution**")
            st.bar_chart(df_resid['std_resid'].value_counts(bins=15).sort_index(), height=280)
            st.caption("Ideal: roughly bell-shaped around 0")

    st.markdown("---")

    # ── Download ──────────────────────────────────────────────────────────
    if reg_xlsx and Path(reg_xlsx).exists():
        with open(reg_xlsx,'rb') as fh:
            st.download_button("⬇️  Download Regression Summary (.xlsx)",
                               data=fh.read(), file_name='linear_regression_summary.xlsx',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                               type='primary')