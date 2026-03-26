"""
app.py — Streamlit UI for Capstone Pipeline
---------------------------------------------
Fixes vs previous version:
  • st.rerun() moved to END of script (after all tab blocks) — prevents duplicate UI render.
  • running state auto-resets if thread dies without sending __EXIT__ (stuck-state fix).
  • Reset button clears stuck running state.
  • run_config snapshot always happens at click-time before thread starts.
  • Global filters: campaign_source, outlet_code, month_year above tab bar.
  • No filters active  → reads pre-computed JSON files (fast, original behaviour).
  • Filters active     → re-runs analysis with filtered data (lazy imports, disk-cached).
  • Analysis imports are LAZY — nothing heavy runs on startup.
"""

import streamlit as st
import pandas as pd
import subprocess, sys, os, threading, queue, time, json, io
from pathlib import Path

# ── App dir ────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

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


# ═══════════════════════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════════════════════
def get_configs():
    return sorted([f.name for f in APP_DIR.glob("config_*.xlsx")])

def load_config(config_file: str) -> dict:
    path = APP_DIR / config_file
    if not path.exists():
        return {}
    df = pd.read_excel(path, sheet_name='paths')
    return dict(zip(df['Setting'].astype(str).str.strip(),
                    df['Value'].astype(str).str.strip()))


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline runner  (runs in a background thread)
# ═══════════════════════════════════════════════════════════════════════════════
def run_pipeline(config_file: str, log_queue: queue.Queue):
    process = subprocess.Popen(
        [sys.executable, str(APP_DIR / 'main_FINAL.py'), config_file],
        cwd=str(APP_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace'
    )
    for line in process.stdout:
        log_queue.put(line)
    process.wait()
    log_queue.put(f"__EXIT__{process.returncode}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fast JSON readers  (original behaviour — used when no filters are active)
# ═══════════════════════════════════════════════════════════════════════════════
def read_ts_json(combined_path: str):
    if not combined_path:
        return None
    p = Path(combined_path) / 'insights.json'
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f).get('time_series', {})
    except Exception:
        return None

def read_reg_json(combined_path: str):
    if not combined_path:
        return None
    p = Path(combined_path) / 'linear_regression_results.json'
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f).get('linear_regression', {})
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Cached filtered analysis  (lazy imports — nothing heavy runs on startup)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_campaign_csv(cleaned_data_path: str) -> pd.DataFrame:
    if not cleaned_data_path:
        return pd.DataFrame()
    camp_csv = Path(cleaned_data_path) / 'campaign_all.csv'
    if not camp_csv.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(camp_csv)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()


def _apply_filters(df: pd.DataFrame,
                   filt_source: tuple,
                   filt_outlet: tuple,
                   filt_months: tuple) -> pd.DataFrame:
    if filt_source:
        df = df[df['campaign_source'].isin(filt_source)]
    if filt_outlet:
        df = df[df['outlet_name'].astype(str).isin([str(x) for x in filt_outlet])]
    if filt_months:
        df = df[df['month_year'].isin(filt_months)]
    return df


@st.cache_data(show_spinner=False)
def get_ts_filtered(cleaned_path: str,
                    filt_source: tuple,
                    filt_outlet: tuple,
                    filt_months: tuple) -> dict:
    try:
        from regression_FINAL import analyse_time_series
    except Exception as e:
        return {'error': f'Cannot import regression_FINAL: {e}'}
    df = load_campaign_csv(cleaned_path)
    if df.empty:
        return {'error': 'campaign_all.csv not found or empty — run the pipeline first.'}
    df = _apply_filters(df, filt_source, filt_outlet, filt_months)
    if df.empty:
        return {'error': 'No rows remain after applying the selected filters.'}
    try:
        return analyse_time_series(df)
    except Exception as e:
        return {'error': str(e)}


@st.cache_data(show_spinner=False)
def get_reg_filtered(cleaned_path: str,
                     filt_source: tuple,
                     filt_outlet: tuple,
                     filt_months: tuple) -> dict:
    try:
        from linear_regression_FINAL import regression_1, regression_2, build_summary
    except Exception as e:
        return {'__error': f'Cannot import linear_regression_FINAL: {e}'}
    df = load_campaign_csv(cleaned_path)
    if df.empty:
        return {'__error': 'campaign_all.csv not found or empty — run the pipeline first.'}
    df = _apply_filters(df, filt_source, filt_outlet, filt_months)
    if df.empty:
        return {'__error': 'No rows remain after applying the selected filters.'}
    try:
        r1 = regression_1(df)
        r2 = regression_2(df)
        return {
            'regression_1_amount': r1,
            'regression_2_amount': r2,
            'summary':             build_summary(r1, r2),
        }
    except Exception as e:
        return {'__error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# Excel builders (in-memory)
# ═══════════════════════════════════════════════════════════════════════════════
def _ts_to_excel(ts: dict) -> bytes:
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            for sheet, key in [
                ('Monthly_Amount',   'monthly_amount'),
                ('Month_Regression', None),
                ('MoM_Trends',       'mom_trends'),
                ('Forecast',         'forecast'),
                ('Anomalies',        'anomalies'),
                ('Amount_by_Source', 'amount_by_source'),
            ]:
                records = (ts.get('month_dummies_regression', {}).get('coef_table')
                           if key is None else ts.get(key))
                if records:
                    pd.DataFrame(records).to_excel(writer, sheet_name=sheet, index=False)
        return buf.getvalue()
    except Exception:
        return b''


def _reg_to_excel(lr: dict) -> bytes:
    buf  = io.BytesIO()
    summ = pd.DataFrame(lr.get('summary', []))
    if summ.empty:
        return b''
    try:
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            summ.to_excel(writer, sheet_name='Summary', index=False)
        return buf.getvalue()
    except Exception:
        return b''


# ═══════════════════════════════════════════════════════════════════════════════
# Session state init  (one-time defaults)
# ═══════════════════════════════════════════════════════════════════════════════
for _k, _v in [('log_lines', []), ('last_exit', None), ('running', False),
               ('run_config', None), ('log_queue', None), ('last_queue_check', 0.0)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════════
# Dead-thread watchdog — auto-reset if running but queue has been dry for >30 s
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.running and st.session_state.log_queue is not None:
    try:
        _peek = st.session_state.log_queue.get_nowait()
        if _peek.startswith("__EXIT__"):
            st.session_state.last_exit = int(_peek.replace("__EXIT__", ""))
            st.session_state.running   = False
            load_campaign_csv.clear()
            get_ts_filtered.clear()
            get_reg_filtered.clear()
        else:
            st.session_state.log_lines.append(_peek.rstrip())
            st.session_state.last_queue_check = time.time()
    except queue.Empty:
        # If no data for > 60 s, assume the thread died silently
        if time.time() - st.session_state.last_queue_check > 60:
            st.session_state.running = False


# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="app-header">'
    '<div class="app-title">⚡ PIPELINE</div>'
    '<div class="app-sub">ETL · Regression · Time Series · Capstone 2025</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Config selector
# ═══════════════════════════════════════════════════════════════════════════════
configs = get_configs()
if not configs:
    st.error("No config_*.xlsx file found in the app directory.")
    st.stop()

_NEW = "＋  Add new config..."

# Initialise only when missing or stale (pointing at a file that no longer exists)
if ('_config_choice' not in st.session_state
        or (st.session_state._config_choice not in configs
            and st.session_state._config_choice != _NEW)):
    st.session_state._config_choice = configs[0]

c1, c2 = st.columns([3, 2])
with c1:
    # Use index= so the widget reflects the stored choice without key= side-effects
    _opts = configs + [_NEW]
    _idx  = _opts.index(st.session_state._config_choice) if st.session_state._config_choice in _opts else 0
    choice = st.selectbox(
        "Config", _opts,
        index=_idx,
        label_visibility="collapsed",
    )
    st.session_state._config_choice = choice   # persist the user's selection

with c2:
    if choice == _NEW:
        st.text_input("New config name", placeholder="e.g. Alice",
                      key="new_config_name", label_visibility="collapsed")
    else:
        st.empty()

if choice == _NEW:
    raw = (st.session_state.get("new_config_name", "").strip()
           .removeprefix("config_").removesuffix(".xlsx"))
    selected_config = f"config_{raw}.xlsx" if raw else None
    is_new = True
else:
    selected_config = choice
    is_new = False

if not selected_config:
    st.info("Enter a config name above.")
    st.stop()

st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

# ── Resolve paths from selected config ────────────────────────────────────────
_cfg_paths     = load_config(selected_config) if not is_new else {}
_cleaned_path  = _cfg_paths.get('cleaned_data', '')
_combined_path = _cfg_paths.get('combined_data', '')

# ── Check if campaign_all.csv exists ──────────────────────────────────────────
_camp_csv  = Path(_cleaned_path) / 'campaign_all.csv' if _cleaned_path else Path('')
_data_ready = _camp_csv.exists()

# ═══════════════════════════════════════════════════════════════════════════════
# Global Filters  (shown only when campaign_all.csv exists)
# ═══════════════════════════════════════════════════════════════════════════════
filt_source: tuple = ()
filt_outlet: tuple = ()
filt_months: tuple = ()

if _data_ready:
    _filter_df = load_campaign_csv(_cleaned_path)

    with st.expander(
        "🔽  Global Filters  —  applied to Time Series & Regression tabs",
        expanded=False,
    ):
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            src_opts = (
                sorted(_filter_df['campaign_source'].dropna().astype(str).unique().tolist())
                if 'campaign_source' in _filter_df.columns else []
            )
            filt_source = tuple(
                st.multiselect("Campaign Source", src_opts,
                               key="filt_source", placeholder="All sources")
            )

        with fc2:
            oc_opts = (
                sorted(_filter_df['outlet_name'].dropna().astype(str).unique().tolist())
                if 'outlet_name' in _filter_df.columns else []
            )
            filt_outlet = tuple(
                st.multiselect("Outlet Name", oc_opts,
                               key="filt_outlet", placeholder="All outlets")
            )

        with fc3:
            try:
                my_opts = sorted(
                    _filter_df['month_year'].dropna().astype(str).unique().tolist(),
                    key=lambda x: pd.to_datetime(x, format='%b-%Y', errors='coerce'),
                ) if 'month_year' in _filter_df.columns else []
            except Exception:
                my_opts = sorted(
                    _filter_df['month_year'].dropna().astype(str).unique().tolist()
                ) if 'month_year' in _filter_df.columns else []

            filt_months = tuple(
                st.multiselect("Month", my_opts,
                               key="filt_months", placeholder="All months")
            )

        active_parts = []
        if filt_source: active_parts.append(f"Source: {', '.join(filt_source)}")
        if filt_outlet: active_parts.append(f"Outlet: {', '.join(str(x) for x in filt_outlet)}")
        if filt_months: active_parts.append(f"Months: {', '.join(filt_months)}")

        if active_parts:
            st.caption(f"🔍 Active filters: {' | '.join(active_parts)}")
        else:
            st.caption("No filters active — showing pre-computed results from last pipeline run.")

elif _cleaned_path and not is_new:
    st.caption("ℹ️ Run the pipeline first to enable global filters.")

_filters_active = bool(filt_source or filt_outlet or filt_months)
active_parts    = []   # ensure defined even if _data_ready is False
if filt_source: active_parts.append(f"Source: {', '.join(filt_source)}")
if filt_outlet: active_parts.append(f"Outlet: {', '.join(str(x) for x in filt_outlet)}")
if filt_months: active_parts.append(f"Months: {', '.join(filt_months)}")

# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_run, tab_config, tab_ts, tab_reg = st.tabs([
    "▶  RUN", "⚙  CONFIG", "📈  TIME SERIES", "📉  REGRESSION"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run:
    # ── Controls row ──────────────────────────────────────────────────────────
    btn_col, reset_col = st.columns([6, 1])
    with btn_col:
        run_clicked = st.button(
            "▶  RUN PIPELINE", type="primary",
            use_container_width=True,
            disabled=st.session_state.running,
        )
    with reset_col:
        if st.button("🔄 Reset", help="Clear stuck running state", use_container_width=True):
            st.session_state.running    = False
            st.session_state.log_lines  = []
            st.session_state.last_exit  = None
            st.session_state.log_queue  = None
            st.session_state.run_config = None
            st.rerun()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Start pipeline ─────────────────────────────────────────────────────────
    if run_clicked and not st.session_state.running:
        # Snapshot the config at click-time — stored separately from the dropdown
        st.session_state.run_config        = selected_config
        st.session_state.log_lines         = []
        st.session_state.last_exit         = None
        st.session_state.running           = True
        st.session_state.last_queue_check  = time.time()
        q = queue.Queue()
        st.session_state.log_queue = q
        # Clear analysis caches so tabs pick up fresh data after the run
        load_campaign_csv.clear()
        get_ts_filtered.clear()
        get_reg_filtered.clear()
        threading.Thread(
            target=run_pipeline,
            args=(st.session_state.run_config, q),
            daemon=True,
        ).start()

    # ── Drain the queue (non-blocking) ────────────────────────────────────────
    if st.session_state.running and st.session_state.log_queue is not None:
        q = st.session_state.log_queue
        drained = 0
        while drained < 200:          # cap per-render drain to avoid blocking
            try:
                line = q.get_nowait()
                if line.startswith("__EXIT__"):
                    st.session_state.last_exit = int(line.replace("__EXIT__", ""))
                    st.session_state.running   = False
                    load_campaign_csv.clear()
                    get_ts_filtered.clear()
                    get_reg_filtered.clear()
                    break
                st.session_state.log_lines.append(line.rstrip())
                st.session_state.last_queue_check = time.time()
                drained += 1
            except queue.Empty:
                break

    # ── Status + log display ──────────────────────────────────────────────────
    if (st.session_state.running
            or st.session_state.log_lines
            or st.session_state.last_exit is not None):

        _rc = st.session_state.run_config or selected_config
        st.caption(f"Config used: **{_rc}**")

        all_log  = "\n".join(st.session_state.log_lines)
        etl_done = "ETL COMPLETED"               in all_log
        ts_done  = "TIME SERIES"  in all_log and "COMPLETED" in all_log
        reg_done = "LINEAR REGRESSION COMPLETED" in all_log

        if st.session_state.running:
            if reg_done:   st.info("⏳ Finalising…")
            elif ts_done:  st.info("✅ Time Series done — running Linear Regression…")
            elif etl_done: st.info("✅ ETL done — running Time Series…")
            else:          st.info("⏳ Running ETL…")
        elif st.session_state.last_exit == 0:
            st.success("✅ ETL  ·  ✅ Time Series  ·  ✅ Linear Regression — all complete!")
        elif st.session_state.last_exit is not None:
            errs = [l for l in st.session_state.log_lines if 'error' in l.lower()]
            st.error(f"❌ Pipeline failed — {errs[-1] if errs else 'Check logs below'}")

        if st.session_state.log_lines:
            _hide = ('DEBUG ', 'Loading configuration', 'raw_data from',
                     'cleaned_data from', 'schemas from',
                     'FutureWarning', 'DeprecationWarning', 'UserWarning', 'warnings.warn')
            visible = [l for l in st.session_state.log_lines[-300:]
                       if not any(p in l for p in _hide)]
            st.markdown(
                f'<div class="log-box">{chr(10).join(visible)}</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
with tab_config:
    paths_cfg = load_config(selected_config) if not is_new else {}
    if is_new:
        st.info(f"New config: **{selected_config}** — fill in paths and click Save.")

    path_labels = {
        'raw_data':      'Raw Data Folder',
        'cleaned_data':  'Cleaned Data Folder',
        'combined_data': 'Combined Data Folder',
    }
    new_paths = {}
    for key, label in path_labels.items():
        c1, c2 = st.columns([5, 1])
        with c1:
            new_paths[key] = st.text_input(
                label, value=paths_cfg.get(key, ''), key=f"path_{key}"
            )
        with c2:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            if st.button("📂", key=f"open_{key}"):
                p    = Path(new_paths[key].strip())
                walk = p
                while walk != walk.parent and not walk.exists():
                    walk = walk.parent
                if sys.platform == 'win32':
                    subprocess.Popen(f'explorer "{walk}"')
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', str(walk)])
                else:
                    subprocess.Popen(['xdg-open', str(walk)])

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    if st.button("💾  Save Config", type="primary"):
        with pd.ExcelWriter(APP_DIR / selected_config, engine='openpyxl') as writer:
            pd.DataFrame(
                [{'Setting': k, 'Value': v} for k, v in new_paths.items()]
            ).to_excel(writer, sheet_name='paths', index=False)
        st.success(f"✅ Saved {selected_config}")
        load_campaign_csv.clear()
        get_ts_filtered.clear()
        get_reg_filtered.clear()
        if is_new:
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ts:
    if _filters_active:
        with st.spinner("Running time series on filtered data…"):
            ts = get_ts_filtered(_cleaned_path, filt_source, filt_outlet, filt_months)
    else:
        ts = read_ts_json(_combined_path)

    if ts is None:
        st.info("No time series results yet — run the pipeline first.")
    elif ts.get('error'):
        st.warning(ts['error'])
    else:
        if _filters_active and active_parts:
            st.caption(f"🔍 Showing filtered results — {' | '.join(active_parts)}")

        # ── Monthly Amount trend ──────────────────────────────────────────────
        st.markdown("### 📈 Monthly Member Spend (Amount)")

        actual   = ts.get('actual', [])
        forecast = ts.get('forecast', [])

        if actual:
            df_act  = pd.DataFrame(actual).rename(columns={'value': 'Actual Amount'})
            df_fore = (pd.DataFrame(forecast).rename(columns={'forecast': 'Forecast'})
                       if forecast else pd.DataFrame())
            df_chart = df_act.set_index('month_year')[['Actual Amount']]
            if not df_fore.empty:
                df_chart = df_chart.join(
                    df_fore.set_index('month_year')[['Forecast']], how='outer'
                )
            st.line_chart(df_chart)

        trend = ts.get('trend', {})
        if trend:
            c1, c2, c3 = st.columns(3)
            c1.metric("Trend Direction", trend.get('direction', '—').title())
            c2.metric("Trend Strength",  trend.get('strength',  '—').title())
            c3.metric("R²",              f"{trend.get('r_squared') or 0:.3f}")
            st.caption(
                "✅ Significant (p<0.05)" if trend.get('significant')
                else "⚠️ Not significant (p≥0.05)"
            )

        st.markdown("---")

        # ── Month dummies regression ──────────────────────────────────────────
        st.markdown("### 📅 Seasonality — Month Dummies Regression")
        st.caption("Which months have significantly higher/lower spend? Base = January")

        mdr = ts.get('month_dummies_regression', {})
        if mdr.get('error'):
            st.warning(mdr['error'])
        elif mdr.get('coef_table'):
            if mdr.get('insight'):
                st.markdown(
                    f'<div class="insight-box">💡 {mdr["insight"]}</div>',
                    unsafe_allow_html=True,
                )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²",       f"{mdr.get('r_squared')     or 0:.3f}")
            c2.metric("Adj R²",   f"{mdr.get('adj_r_squared') or 0:.3f}")
            c3.metric("F p-value",f"{mdr.get('f_pvalue')      or 0:.4f}")
            c4.metric("N obs",     mdr.get('n_obs', '—'))

            df_coef      = pd.DataFrame(mdr['coef_table'])
            df_coef_plot = df_coef[df_coef['month'] != 'const'].copy()

            col_chart, col_table = st.columns([2, 3])
            with col_chart:
                if not df_coef_plot.empty:
                    st.markdown("**Coefficient by Month**")
                    st.bar_chart(df_coef_plot.set_index('month')['coef'])
            with col_table:
                def _hl_month(row):
                    return (['background-color:#3d3800; color:#fbbf24'] * len(row)
                            if row.get('significant') else [''] * len(row))
                st.dataframe(
                    df_coef[['month', 'coef', 'p_value', 'significant']]
                    .style.apply(_hl_month, axis=1),
                    use_container_width=True, hide_index=True,
                )
            st.caption("✅ Highlighted = significant at p<0.05. Positive = higher spend than January.")

        st.markdown("---")

        # ── MoM trends ────────────────────────────────────────────────────────
        st.markdown("### 📊 Month-on-Month Trends")
        mom = ts.get('mom_trends', [])
        if mom:
            st.dataframe(pd.DataFrame(mom), use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Amount by source ──────────────────────────────────────────────────
        by_source = ts.get('amount_by_source', [])
        if by_source:
            st.markdown("### 🏷️ Spend by Campaign Source (Mall vs Brand)")
            df_src = pd.DataFrame(by_source)
            if 'campaign_source' in df_src.columns and 'month_year' in df_src.columns:
                try:
                    pivoted = df_src.pivot(
                        index='month_year', columns='campaign_source', values='amount'
                    ).fillna(0)
                    st.bar_chart(pivoted)
                except Exception:
                    st.dataframe(df_src, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Anomalies ─────────────────────────────────────────────────────────
        anomalies = ts.get('anomalies', [])
        if anomalies:
            st.markdown("### ⚠️ Anomalous Months")
            st.dataframe(
                pd.DataFrame(anomalies), use_container_width=True, hide_index=True
            )
        else:
            st.caption("No anomalous months detected.")

        st.markdown("---")

        # ── Download ──────────────────────────────────────────────────────────
        if _filters_active:
            xlsx = _ts_to_excel(ts)
            if xlsx:
                st.download_button(
                    "⬇️  Download Filtered Time Series Report (.xlsx)",
                    data=xlsx,
                    file_name='insights_report_filtered.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    type='primary',
                )
        else:
            report = Path(_combined_path) / 'insights_report.xlsx' if _combined_path else Path('')
            if report.exists():
                with open(report, 'rb') as fh:
                    st.download_button(
                        "⬇️  Download Time Series Report (.xlsx)",
                        data=fh.read(),
                        file_name='insights_report.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        type='primary',
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reg:
    if _filters_active:
        with st.spinner("Running regression on filtered data…"):
            lr_raw = get_reg_filtered(_cleaned_path, filt_source, filt_outlet, filt_months)
        if lr_raw.get('__error'):
            st.warning(lr_raw['__error'])
            lr = {}
        else:
            lr = lr_raw
    else:
        lr = read_reg_json(_combined_path)

    if lr is None:
        st.info("No regression results yet — run the pipeline first.")
        lr = {}

    if lr:
        if _filters_active and active_parts:
            st.caption(f"🔍 Showing filtered results — {' | '.join(active_parts)}")

        # ── Summary ───────────────────────────────────────────────────────────
        st.markdown("### 📊 Model Comparison Summary")
        summary = lr.get('summary', [])
        if summary:
            df_sum = pd.DataFrame(summary)
            disp   = [c for c in ['model_key', 'label', 'n_obs', 'r_squared',
                                   'adj_r_squared', 'f_pvalue', 'cv_r2', 'cv_rmse']
                      if c in df_sum.columns]
            st.dataframe(
                df_sum[disp], use_container_width=True, hide_index=True,
                column_config={
                    'r_squared':     st.column_config.ProgressColumn('R²',     min_value=0, max_value=1, format='%.3f'),
                    'adj_r_squared': st.column_config.ProgressColumn('Adj R²', min_value=0, max_value=1, format='%.3f'),
                    'cv_r2':         st.column_config.ProgressColumn('CV R²',  min_value=0, max_value=1, format='%.3f'),
                },
            )
            valid = df_sum.dropna(subset=['adj_r_squared'])
            if not valid.empty:
                best = valid.loc[valid['adj_r_squared'].idxmax()]
                st.markdown(
                    f'<div class="insight-box">🏆 Best model: <strong>{best["model_key"]}</strong>'
                    f' — Adj-R² = <strong>{best["adj_r_squared"]:.3f}</strong></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Model selector ────────────────────────────────────────────────────
        model_options = {
            'Regression 1 — Y = Amount (outlet×month)': 'regression_1_amount',
            'Regression 2 — Y = Amount (per receipt)':  'regression_2_amount',
        }
        selected_label = st.radio(
            "Select model", list(model_options.keys()), horizontal=True
        )
        model = lr.get(model_options[selected_label], {})

        if model.get('error'):
            st.error(f"Model error: {model['error']}")
        elif model:
            if model.get('insight'):
                st.markdown(
                    f'<div class="insight-box">💡 {model["insight"]}</div>',
                    unsafe_allow_html=True,
                )

            # ── Model fit ─────────────────────────────────────────────────────
            st.markdown("#### Model Fit")
            fit = model.get('model_fit', {})
            cv  = model.get('cross_validation', {})

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("R²",        f"{fit.get('r_squared')     or 0:.3f}")
            c2.metric("Adj R²",    f"{fit.get('adj_r_squared') or 0:.3f}")
            c3.metric("F p-value", f"{fit.get('f_pvalue')      or 0:.4f}")
            c4.metric("N obs",      fit.get('n_obs', '—'))
            c5.metric("CV RMSE",   f"{cv.get('cv_rmse_mean') or 0:,.1f}"
                                   if cv.get('cv_rmse_mean') else "—")
            c6.metric("CV R²",     f"{cv.get('cv_r2_mean') or 0:.3f}"
                                   if cv.get('cv_r2_mean')   else "—")

            dw = fit.get('dw_stat')
            if dw:
                st.caption(
                    f"Durbin-Watson: {dw:.3f} — "
                    + ("✅ No autocorrelation" if 1.5 < dw < 2.5 else "⚠️ Possible autocorrelation")
                )

            st.markdown("---")

            # ── Feature groups ────────────────────────────────────────────────
            fg = model.get('feature_groups', {})
            if fg:
                cols = st.columns(len(fg))
                for col, (group, feats) in zip(cols, fg.items()):
                    with col:
                        st.markdown(f"**{group.title()} ({len(feats)})**")
                        st.caption(", ".join(feats[:10]) + ("…" if len(feats) > 10 else ""))

            st.markdown("---")

            # ── Coefficients ──────────────────────────────────────────────────
            st.markdown("#### Coefficients (standardised β)")
            coef_table = model.get('coef_table', [])
            if coef_table:
                df_coef      = pd.DataFrame(coef_table)
                df_coef_plot = df_coef[df_coef['feature'] != 'const'].copy()

                show_sig = st.checkbox("Show significant predictors only (p<0.05)", value=False)
                if show_sig:
                    df_coef_plot = df_coef_plot[df_coef_plot['significant'] == True]

                col_chart, col_table = st.columns([2, 3])
                with col_chart:
                    if not df_coef_plot.empty:
                        top20 = df_coef_plot.nlargest(20, 'coef', keep='all')
                        st.markdown("**Top coefficients**")
                        st.bar_chart(top20.set_index('feature')['coef'])
                with col_table:
                    disp_cols = [c for c in ['feature', 'coef', 'std_err', 't_stat',
                                              'p_value', 'ci_lower', 'ci_upper', 'significant']
                                 if c in df_coef.columns]
                    def _hl(row):
                        return (['background-color:#3d3800; color:#fbbf24'] * len(row)
                                if row.get('significant') else [''] * len(row))
                    st.dataframe(
                        df_coef[disp_cols].style.apply(_hl, axis=1),
                        use_container_width=True, hide_index=True,
                    )
                st.caption("✅ Highlighted = significant at p<0.05. Standardised β — comparable in magnitude.")

            st.markdown("---")

            # ── VIF ───────────────────────────────────────────────────────────
            vif = model.get('vif', [])
            if vif:
                st.markdown("#### Multicollinearity (VIF)")
                df_vif = pd.DataFrame(vif)
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.dataframe(df_vif, use_container_width=True, hide_index=True)
                with c2:
                    st.caption("VIF < 5 → ✅ OK  |  5–10 → ⚠️ Moderate  |  >10 → ❌ High")
                    if not df_vif.empty and 'vif' in df_vif.columns:
                        mv = df_vif['vif'].dropna().max()
                        if mv and mv > 10:
                            st.warning(f"⚠️ Max VIF={mv:.1f} — high multicollinearity")
                        elif mv and mv > 5:
                            st.warning(f"⚠️ Max VIF={mv:.1f} — moderate multicollinearity")
                        else:
                            st.success("✅ All VIF < 5")

            st.markdown("---")

            # ── Residuals ─────────────────────────────────────────────────────
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
                    st.bar_chart(
                        df_resid['std_resid'].value_counts(bins=15).sort_index(),
                        height=280,
                    )
                    st.caption("Ideal: roughly bell-shaped around 0")

            st.markdown("---")

        # ── Download ──────────────────────────────────────────────────────────
        if _filters_active:
            xlsx = _reg_to_excel(lr)
            if xlsx:
                st.download_button(
                    "⬇️  Download Filtered Regression Summary (.xlsx)",
                    data=xlsx,
                    file_name='linear_regression_summary_filtered.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    type='primary',
                )
        else:
            reg_xlsx = Path(_combined_path) / 'linear_regression_summary.xlsx' if _combined_path else Path('')
            if reg_xlsx.exists():
                with open(reg_xlsx, 'rb') as fh:
                    st.download_button(
                        "⬇️  Download Regression Summary (.xlsx)",
                        data=fh.read(),
                        file_name='linear_regression_summary.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        type='primary',
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# Polling rerun — MUST be at the very end, outside all tab blocks.
# This prevents the "duplicate UI" artifact caused by rerunning mid-render.
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.running:
    time.sleep(0.8)
    st.rerun()
