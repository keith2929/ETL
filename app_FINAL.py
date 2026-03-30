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
import plotly.graph_objects as go
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
            return json.load(f)
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
                   filt_months: tuple,
                   filt_campaign: tuple = ()) -> pd.DataFrame:
    if filt_source:
        df = df[df['campaign_source'].isin(filt_source)]
    if filt_outlet:
        df = df[df['outlet_name'].astype(str).isin([str(x) for x in filt_outlet])]
    if filt_months:
        df = df[df['month_year'].isin(filt_months)]
    if filt_campaign:
        df = df[df['voucher_code'].astype(str).isin([str(x) for x in filt_campaign])]
    return df


@st.cache_data(show_spinner=False)
def get_ts_filtered(cleaned_path: str,
                    filt_source: tuple,
                    filt_outlet: tuple,
                    filt_months: tuple,
                    filt_campaign: tuple = ()) -> dict:
    try:
        from regression_FINAL import analyse_time_series
    except Exception as e:
        return {'error': f'Cannot import regression_FINAL: {e}'}
    df = load_campaign_csv(cleaned_path)
    if df.empty:
        return {'error': 'campaign_all.csv not found or empty — run the pipeline first.'}
    df = _apply_filters(df, filt_source, filt_outlet, filt_months, filt_campaign)
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
                     filt_months: tuple,
                     filt_campaign: tuple = ()) -> dict:
    try:
        from linear_regression_FINAL import regression_1, build_summary
    except Exception as e:
        return {'__error': f'Cannot import linear_regression_FINAL: {e}'}
    df = load_campaign_csv(cleaned_path)
    if df.empty:
        return {'__error': 'campaign_all.csv not found or empty — run the pipeline first.'}
    df = _apply_filters(df, filt_source, filt_outlet, filt_months, filt_campaign)
    if df.empty:
        return {'__error': 'No rows remain after applying the selected filters.'}
    try:
        r1 = regression_1(df)
        return {
            'regression_1_amount': r1,
            'summary':             build_summary(r1),
        }
    except Exception as e:
        return {'__error': str(e)}


@st.cache_data(show_spinner=False)
def get_tt_filtered(cleaned_path: str,
                    filt_source: tuple,
                    filt_outlet: tuple,
                    filt_months: tuple,
                    filt_campaign: tuple) -> dict:
    try:
        from ttest_FINAL import normality_test, one_sample_ttest, roi_analysis, summary_stats
    except Exception as e:
        return {'__error': f'Cannot import ttest_FINAL: {e}'}
    df = load_campaign_csv(cleaned_path)
    if df.empty:
        return {'__error': 'campaign_all.csv not found or empty — run the pipeline first.'}
    df = _apply_filters(df, filt_source, filt_outlet, filt_months, filt_campaign)
    if df.empty:
        return {'__error': 'No rows remain after applying the selected filters.'}
    df['amount'] = pd.to_numeric(df.get('amount', pd.Series(dtype=float)), errors='coerce')
    df = df.dropna(subset=['amount']).reset_index(drop=True)
    if df.empty:
        return {'__error': 'No rows with valid amount after filtering.'}
    try:
        return {
            'summary':   summary_stats(df),
            'normality': normality_test(df),
            'ttest':     one_sample_ttest(df),
            'roi':       roi_analysis(df),
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
            get_tt_filtered.clear()
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
# Global filter state — each tab manages its own expander, inheriting these defaults
# ═══════════════════════════════════════════════════════════════════════════════
_filters_active = False   # placeholder; each tab sets its own flag
active_parts: list = []

# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_run, tab_config, tab_ts, tab_reg, tab_tt = st.tabs([
    "▶  RUN", "⚙  CONFIG", "📈  TIME SERIES", "📉  REGRESSION", "🧪  T-TEST & ROI"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run:
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

    if run_clicked and not st.session_state.running:
        st.session_state.run_config        = selected_config
        st.session_state.log_lines         = []
        st.session_state.last_exit         = None
        st.session_state.running           = True
        st.session_state.last_queue_check  = time.time()
        q = queue.Queue()
        st.session_state.log_queue = q
        load_campaign_csv.clear()
        get_ts_filtered.clear()
        get_reg_filtered.clear()
        get_tt_filtered.clear()
        threading.Thread(
            target=run_pipeline,
            args=(st.session_state.run_config, q),
            daemon=True,
        ).start()

    # ── Drain the queue ───────────────────────────────────────────────────────
    if st.session_state.running and st.session_state.log_queue is not None:
        q = st.session_state.log_queue
        drained = 0
        while drained < 200:
            try:
                line = q.get_nowait()
                if line.startswith("__EXIT__"):
                    st.session_state.last_exit = int(line.replace("__EXIT__", ""))
                    st.session_state.running   = False
                    load_campaign_csv.clear()
                    get_ts_filtered.clear()
                    get_reg_filtered.clear()
                    get_tt_filtered.clear()
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
        tt_done  = "T-TEST & ROI ANALYSIS COMPLETED" in all_log

        if st.session_state.running:
            if tt_done:    st.info("⏳ Finalising…")
            elif reg_done: st.info("✅ Linear Regression done — running T-Test & ROI…")
            elif ts_done:  st.info("✅ Time Series done — running Linear Regression…")
            elif etl_done: st.info("✅ ETL done — running Time Series…")
            else:          st.info("⏳ Running ETL…")
        elif st.session_state.last_exit == 0:
            st.success("✅ ETL  ·  ✅ Time Series  ·  ✅ Linear Regression  ·  ✅ T-Test & ROI — all complete!")
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
# TAB 5 — T-TEST & ROI
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tt:
    # ── T-Test tab-local filters ──────────────────────────────────────────────
    tt_filters_active = False
    tt_active_parts   = []

    if _data_ready:
        _tt_df = load_campaign_csv(_cleaned_path)

        with st.expander("🔽  T-Test Filters  —  refine which data this tab analyses", expanded=False):
            tf1, tf2, tf3, tf4 = st.columns(4)

            with tf1:
                tt_src_opts = (
                    sorted(_tt_df['campaign_source'].dropna().astype(str).unique().tolist())
                    if 'campaign_source' in _tt_df.columns else []
                )
                tt_filt_source = tuple(
                    st.multiselect("Campaign Source", tt_src_opts,
                                   key="tt_filt_source", placeholder="All sources")
                )

            with tf2:
                tt_oc_opts = (
                    sorted(_tt_df['outlet_name'].dropna().astype(str).unique().tolist())
                    if 'outlet_name' in _tt_df.columns else []
                )
                tt_filt_outlet = tuple(
                    st.multiselect("Outlet Name", tt_oc_opts,
                                   key="tt_filt_outlet", placeholder="All outlets")
                )

            with tf3:
                try:
                    tt_my_opts = sorted(
                        _tt_df['month_year'].dropna().astype(str).unique().tolist(),
                        key=lambda x: pd.to_datetime(x, format='%b-%Y', errors='coerce'),
                    ) if 'month_year' in _tt_df.columns else []
                except Exception:
                    tt_my_opts = sorted(
                        _tt_df['month_year'].dropna().astype(str).unique().tolist()
                    ) if 'month_year' in _tt_df.columns else []
                tt_filt_months = tuple(
                    st.multiselect("Month", tt_my_opts,
                                   key="tt_filt_months", placeholder="All months")
                )

            with tf4:
                # Pre-filter by source/outlet/month to get relevant campaigns
                _tt_df_pre = _apply_filters(_tt_df, tt_filt_source, tt_filt_outlet, tt_filt_months)
                tt_camp_opts = (
                    sorted(_tt_df_pre['voucher_code'].dropna().astype(str).unique().tolist())
                    if 'voucher_code' in _tt_df_pre.columns else []
                )
                tt_filt_campaign = tuple(
                    st.multiselect("Campaign (Voucher Code)", tt_camp_opts,
                                   key="tt_filt_campaign", placeholder="All campaigns")
                )

            tt_active_parts = []
            if tt_filt_source:   tt_active_parts.append(f"Source: {', '.join(tt_filt_source)}")
            if tt_filt_outlet:   tt_active_parts.append(f"Outlet: {', '.join(str(x) for x in tt_filt_outlet)}")
            if tt_filt_months:   tt_active_parts.append(f"Months: {', '.join(tt_filt_months)}")
            if tt_filt_campaign: tt_active_parts.append(f"Campaign: {', '.join(tt_filt_campaign)}")

            if tt_active_parts:
                st.caption(f"🔍 Active filters: {' | '.join(tt_active_parts)}")
            else:
                st.caption("No filters active — showing pre-computed results from last pipeline run.")

        tt_filters_active = bool(tt_filt_source or tt_filt_outlet or tt_filt_months or tt_filt_campaign)
    else:
        tt_filt_source   = ()
        tt_filt_outlet   = ()
        tt_filt_months   = ()
        tt_filt_campaign = ()

    # ── Load data: filtered (live) or pre-computed JSON ───────────────────
    if tt_filters_active:
        with st.spinner("Running T-Test & ROI on filtered data…"):
            tt_raw = get_tt_filtered(_cleaned_path, tt_filt_source, tt_filt_outlet,
                                     tt_filt_months, tt_filt_campaign)
        if tt_raw.get('__error'):
            st.warning(tt_raw['__error'])
            tt_raw = {}
        tt_data = tt_raw
    else:
        tt_path = Path(_combined_path) / 'ttest_results.json' if _combined_path else Path('')
        if not tt_path.exists():
            st.info("No T-Test results yet — run the pipeline first.")
            tt_data = {}
        else:
            try:
                with open(tt_path) as f:
                    content = f.read().strip()
                    tt_data = json.loads(content).get('ttest_analysis', {}) if content else {}
            except (json.JSONDecodeError, Exception):
                st.warning("T-Test results file is corrupted — please re-run the pipeline.")
                tt_data = {}

    if tt_data:
        if tt_filters_active and tt_active_parts:
            st.caption(f"🔍 Showing filtered results — {' | '.join(tt_active_parts)}")

        summ      = tt_data.get('summary', {})
        ttest     = tt_data.get('ttest', [])
        roi       = tt_data.get('roi', [])
        normality = tt_data.get('normality', [])

        # ── Summary metrics ───────────────────────────────────────────────
        st.markdown("### 📊 Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Mean Revenue",  f"${summ.get('overall_mean_revenue') or 0:,.0f}")
        c2.metric("Total Campaigns",        summ.get('n_campaigns', '—'))
        c3.metric("Mall Campaigns",         summ.get('n_mall_campaigns', '—'))
        c4.metric("Brand Campaigns",        summ.get('n_brand_campaigns', '—'))

        st.markdown("---")

        # ── One-sample T-test ─────────────────────────────────────────────
        st.markdown("### 🧪 One-Sample T-Test")
        st.caption(f"H₀: Campaign mean revenue = Overall mean (${summ.get('overall_mean_revenue') or 0:,.0f}). Highlighted = significant at p<0.05.")

        if ttest:
            df_tt = pd.DataFrame(ttest)

            sig_above = df_tt[(df_tt['significant']==True) & (df_tt['direction']=='above')]
            sig_below = df_tt[(df_tt['significant']==True) & (df_tt['direction']=='below')]

            ca, cb = st.columns(2)
            ca.metric("Significantly Above Mean", len(sig_above), delta="↑ higher spend")
            cb.metric("Significantly Below Mean", len(sig_below), delta="↓ lower spend", delta_color="inverse")

            disp_cols = [c for c in ['voucher_code','campaign_source','n',
                                     'campaign_mean','overall_mean','diff_from_mean',
                                     't_stat','p_value','significant','direction']
                         if c in df_tt.columns]

            def _hl_tt(row):
                if row.get('significant') and row.get('direction') == 'above':
                    return ['background-color:#1a2e1a; color:#4af0c4'] * len(row)
                elif row.get('significant') and row.get('direction') == 'below':
                    return ['background-color:#2e1a1a; color:#ff5c5c'] * len(row)
                return [''] * len(row)

            st.dataframe(
                df_tt[disp_cols].style.apply(_hl_tt, axis=1),
                use_container_width=True, hide_index=True,
                column_config={
                    'voucher_code':    st.column_config.TextColumn('Campaign'),
                    'campaign_source': st.column_config.TextColumn('Source'),
                    'n':               st.column_config.NumberColumn('N', format='%d'),
                    'campaign_mean':   st.column_config.NumberColumn('Campaign Mean ($)', format='$%.0f'),
                    'overall_mean':    st.column_config.NumberColumn('Overall Mean ($)', format='$%.0f'),
                    'diff_from_mean':  st.column_config.NumberColumn('Diff ($)', format='$%.0f'),
                    't_stat':          st.column_config.NumberColumn('T-stat', format='%.3f'),
                    'p_value':         st.column_config.NumberColumn('P-value', format='%.4f'),
                }
            )
            st.caption("🟢 Green = significantly above mean | 🔴 Red = significantly below mean")

        st.markdown("---")

        # ── ROI ───────────────────────────────────────────────────────────
        st.markdown("### 💰 ROI per Campaign")
        st.caption("ROI = Total Revenue / Total Voucher Cost. ROI > 1 = positive return.")

        if roi:
            df_roi = pd.DataFrame(roi)

            df_roi_valid = df_roi[df_roi['roi'].notna()].copy()
            if not df_roi_valid.empty:
                top10_roi = df_roi_valid.head(10).copy()
                st.markdown("**Top 10 Campaigns by ROI**")
                top10_roi_chart = pd.DataFrame(index=top10_roi['voucher_code'])
                top10_roi_chart['Mall']  = top10_roi.set_index('voucher_code')['roi'].where(
                    top10_roi.set_index('voucher_code')['campaign_source'] == 'mall')
                top10_roi_chart['Brand'] = top10_roi.set_index('voucher_code')['roi'].where(
                    top10_roi.set_index('voucher_code')['campaign_source'] == 'brand')
                st.bar_chart(top10_roi_chart, color=['#ff5c5c', '#4af0c4'])

            def _hl_roi(row):
                r = row.get('roi')
                if r and r > 1:   return ['background-color:#1a2e1a; color:#4af0c4'] * len(row)
                if r and r < 1:   return ['background-color:#2e1a1a; color:#ff5c5c'] * len(row)
                return [''] * len(row)

            disp_roi = [c for c in ['voucher_code','campaign_source','n_redemptions',
                         'total_revenue','avg_revenue',
                         'voucher_value','roi','roi_label']
                        if c in df_roi.columns]
            st.dataframe(
                df_roi[disp_roi].style.apply(_hl_roi, axis=1),
                use_container_width=True, hide_index=True,
                column_config={
                    'voucher_code':       st.column_config.TextColumn('Campaign'),
                    'campaign_source':    st.column_config.TextColumn('Source'),
                    'n_redemptions':      st.column_config.NumberColumn('Redemptions', format='%d'),
                    'total_revenue':      st.column_config.NumberColumn('Total Revenue ($)', format='$%.0f'),
                    'avg_revenue':        st.column_config.NumberColumn('Avg Revenue ($)', format='$%.0f'),
                    'voucher_value': st.column_config.NumberColumn('Voucher Value ($)', format='$%.0f'),
                    'roi':                st.column_config.NumberColumn('ROI', format='%.2fx'),
                }
            )
            st.caption("🟢 Green = ROI > 1 | 🔴 Red = ROI < 1")

        st.markdown("---")

        # ── Normality ─────────────────────────────────────────────────────
        st.markdown("### 📐 Normality Test (Shapiro-Wilk)")
        st.caption("n>30: Normal by Central Limit Theorem. n≤30: Shapiro-Wilk test (p≥0.05 = normal).")

        if normality:
            df_norm = pd.DataFrame(normality)
            n_normal     = df_norm['normal'].sum() if 'normal' in df_norm.columns else 0
            n_not_normal = (~df_norm['normal'].fillna(False)).sum()

            cn1, cn2 = st.columns(2)
            cn1.metric("Normal Distribution",     int(n_normal))
            cn2.metric("Not Normal Distribution", int(n_not_normal))

            def _hl_norm(row):
                if row.get('normal') == True:  return ['background-color:#1a2e1a; color:#4af0c4'] * len(row)
                if row.get('normal') == False: return ['background-color:#2e1a1a; color:#ff5c5c'] * len(row)
                return [''] * len(row)

            disp_norm = [c for c in ['voucher_code','campaign_source','n',
                                      'mean','std','shapiro_stat','shapiro_p','normal','note']
                         if c in df_norm.columns]
            st.dataframe(
                df_norm[disp_norm].style.apply(_hl_norm, axis=1),
                use_container_width=True, hide_index=True,
                column_config={
                    'voucher_code':    st.column_config.TextColumn('Campaign'),
                    'campaign_source': st.column_config.TextColumn('Source'),
                    'n':               st.column_config.NumberColumn('N', format='%d'),
                    'mean':            st.column_config.NumberColumn('Mean ($)', format='$%.0f'),
                    'std':             st.column_config.NumberColumn('Std ($)', format='$%.0f'),
                    'shapiro_stat':    st.column_config.NumberColumn('Shapiro W', format='%.4f'),
                    'shapiro_p':       st.column_config.NumberColumn('P-value', format='%.4f'),
                }
            )
            st.caption("🟢 Green = normal | 🔴 Red = not normal")

            # ── Significant + Normal ──────────────────────────────────────────
            st.markdown("---")
            st.markdown("### ✅ Most Reliable Results — Significant & Normally Distributed")
            st.caption("Campaigns that are both statistically significant (t-test p<0.05) AND normally distributed. These are the most trustworthy findings.")

            if ttest and normality:
                ttest_dict     = {r['voucher_code']: r for r in ttest}
                normality_dict = {r['voucher_code']: r for r in normality}

                reliable = []
                for code, t in ttest_dict.items():
                    n = normality_dict.get(code, {})
                    if t.get('significant') and n.get('normal'):
                        reliable.append({
                            'voucher_code':    code,
                            'campaign_source': t.get('campaign_source'),
                            'n':               t.get('n'),
                            'campaign_mean':   t.get('campaign_mean'),
                            'overall_mean':    t.get('overall_mean'),
                            'diff_from_mean':  t.get('diff_from_mean'),
                            'direction':       t.get('direction'),
                            'p_ttest':         t.get('p_value'),
                            'p_shapiro':       n.get('shapiro_p'),
                        })

                if reliable:
                    df_rel = pd.DataFrame(reliable)
                    above  = [r for r in reliable if r['direction'] == 'above']
                    below  = [r for r in reliable if r['direction'] == 'below']
                    cr1, cr2, cr3 = st.columns(3)
                    cr1.metric("Total Reliable", len(reliable))
                    cr2.metric("Above Mean ↑",   len(above))
                    cr3.metric("Below Mean ↓",   len(below))

                    def _hl_rel(row):
                        if row.get('direction') == 'above':
                            return ['background-color:#1a2e1a; color:#4af0c4'] * len(row)
                        return ['background-color:#2e1a1a; color:#ff5c5c'] * len(row)

                    st.dataframe(
                        df_rel.style.apply(_hl_rel, axis=1),
                        use_container_width=True, hide_index=True,
                        column_config={
                            'voucher_code':    st.column_config.TextColumn('Campaign'),
                            'campaign_source': st.column_config.TextColumn('Source'),
                            'n':               st.column_config.NumberColumn('N', format='%d'),
                            'campaign_mean':   st.column_config.NumberColumn('Campaign Mean ($)', format='$%.0f'),
                            'overall_mean':    st.column_config.NumberColumn('Overall Mean ($)', format='$%.0f'),
                            'diff_from_mean':  st.column_config.NumberColumn('Diff ($)', format='$%.0f'),
                            'direction':       st.column_config.TextColumn('Direction'),
                            'p_ttest':         st.column_config.NumberColumn('T-test p', format='%.4f'),
                            'p_shapiro':       st.column_config.NumberColumn('Shapiro p', format='%.4f'),
                        }
                    )
                    st.caption("🟢 Green = above overall mean | 🔴 Red = below overall mean")

                    above_names = ", ".join([r['voucher_code'] for r in above])
                    below_names = ", ".join([r['voucher_code'] for r in below])
                    insight = ""
                    if above:
                        insight += f"Campaigns significantly above average: {above_names}. "
                    if below:
                        insight += f"Campaigns significantly below average: {below_names}."
                    if insight:
                        st.markdown(
                            f'<div class="insight-box">💡 {insight}</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No campaigns found that are both significant and normally distributed.")

        # ── Download ──────────────────────────────────────────────────────
        st.markdown("---")
        if tt_filters_active:
            # Build in-memory Excel for filtered results
            _tt_buf = __import__('io').BytesIO()
            try:
                with pd.ExcelWriter(_tt_buf, engine='openpyxl') as _w:
                    pd.DataFrame([summ]).to_excel(_w, sheet_name='Summary',   index=False)
                    pd.DataFrame(ttest).to_excel(_w, sheet_name='T-Test',     index=False)
                    pd.DataFrame(roi).to_excel(  _w, sheet_name='ROI',        index=False)
                    pd.DataFrame(normality).to_excel(_w, sheet_name='Normality', index=False)
                st.download_button(
                    "⬇️  Download Filtered T-Test & ROI Report (.xlsx)",
                    data=_tt_buf.getvalue(),
                    file_name='ttest_results_filtered.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    type='primary',
                )
            except Exception:
                pass
        else:
            tt_xlsx = Path(_combined_path) / 'ttest_results.xlsx' if _combined_path else Path('')
            if tt_xlsx.exists():
                with open(tt_xlsx, 'rb') as fh:
                    st.download_button(
                        "⬇️  Download T-Test & ROI Report (.xlsx)",
                        data=fh.read(),
                        file_name='ttest_results.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        type='primary',
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
        get_tt_filtered.clear()
        if is_new:
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ts:
    # ── Filters ───────────────────────────────────────────────────────────────
    ts_filters_active = False
    ts_active_parts: list = []

    if _data_ready:
        _ts_df = load_campaign_csv(_cleaned_path)
        with st.expander("🔽  Filters", expanded=False):
            _c1, _c2, _c3, _c4 = st.columns(4)
            with _c1:
                _opts = sorted(_ts_df['campaign_source'].dropna().astype(str).unique().tolist()) if 'campaign_source' in _ts_df.columns else []
                ts_filt_source = tuple(st.multiselect("Campaign Source", _opts, key="ts_filt_source", placeholder="All sources"))
            with _c2:
                _opts = sorted(_ts_df['outlet_name'].dropna().astype(str).unique().tolist()) if 'outlet_name' in _ts_df.columns else []
                ts_filt_outlet = tuple(st.multiselect("Outlet Name", _opts, key="ts_filt_outlet", placeholder="All outlets"))
            with _c3:
                try:
                    _opts = sorted(_ts_df['month_year'].dropna().astype(str).unique().tolist(), key=lambda x: pd.to_datetime(x, format='%b-%Y', errors='coerce')) if 'month_year' in _ts_df.columns else []
                except Exception:
                    _opts = sorted(_ts_df['month_year'].dropna().astype(str).unique().tolist()) if 'month_year' in _ts_df.columns else []
                ts_filt_months = tuple(st.multiselect("Month", _opts, key="ts_filt_months", placeholder="All months"))
            with _c4:
                _pre = _apply_filters(_ts_df, ts_filt_source, ts_filt_outlet, ts_filt_months)
                _opts = sorted(_pre['voucher_code'].dropna().astype(str).unique().tolist()) if 'voucher_code' in _pre.columns else []
                ts_filt_campaign = tuple(st.multiselect("Campaign", _opts, key="ts_filt_campaign", placeholder="All campaigns"))
            if ts_filt_source:   ts_active_parts.append(f"Source: {', '.join(ts_filt_source)}")
            if ts_filt_outlet:   ts_active_parts.append(f"Outlet: {', '.join(str(x) for x in ts_filt_outlet)}")
            if ts_filt_months:   ts_active_parts.append(f"Months: {', '.join(ts_filt_months)}")
            if ts_filt_campaign: ts_active_parts.append(f"Campaign: {', '.join(ts_filt_campaign)}")
            st.caption(f"🔍 {' | '.join(ts_active_parts)}" if ts_active_parts else "No filters active — showing pre-computed results.")
        ts_filters_active = bool(ts_filt_source or ts_filt_outlet or ts_filt_months or ts_filt_campaign)
    else:
        ts_filt_source = ts_filt_outlet = ts_filt_months = ts_filt_campaign = ()

    if ts_filters_active:
        with st.spinner("Running time series on filtered data…"):
            ts  = get_ts_filtered(_cleaned_path, ts_filt_source, ts_filt_outlet, ts_filt_months, ts_filt_campaign)
        mnm = {}
    else:
        ins = read_ts_json(_combined_path)
        if not ins:
            st.info("No time series results yet — run the pipeline first.")
            st.stop()
        ts  = ins.get('time_series', {})
        mnm = ins.get('member_nonmember', {})

    if not ts:
        st.info("No time series results yet — run the pipeline first.")
    elif ts.get('error'):
        st.warning(ts['error'])
    else:
        if ts_filters_active and ts_active_parts:
            st.caption(f"🔍 Showing filtered results — {' | '.join(ts_active_parts)}")

        # ── Monthly Amount trend ──────────────────────────────────────────────
        st.markdown("### 📈 Monthly Member Spend (Amount)")

        actual     = ts.get('actual', [])
        forecast   = ts.get('forecast', [])
        decomp     = ts.get('decomposition', {})
        adjusted   = ts.get('seasonally_adjusted', [])
        diag       = ts.get('diagnostics', {})
        model_info = ts.get('model_info', {})

        if actual:
            df_act  = pd.DataFrame(actual)
            df_act['month_year'] = pd.to_datetime(df_act['month_year'], format='%b-%Y')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_act['month_year'], y=df_act['value'],
                mode='lines+markers', name='Actual Amount',
                line=dict(color='#4af0c4')
            ))

            if forecast:
                df_fore = pd.DataFrame(forecast)
                df_fore['month_year'] = pd.to_datetime(df_fore['month_year'], format='%b-%Y')
                fig.add_trace(go.Scatter(
                    x=df_fore['month_year'], y=df_fore['forecast'],
                    mode='lines+markers', name='Forecast',
                    line=dict(color='#c8f135', dash='dash')
                ))
                if 'upper_bound' in df_fore.columns and 'lower_bound' in df_fore.columns:
                    fig.add_trace(go.Scatter(
                        x=df_fore['month_year'].tolist() + df_fore['month_year'].tolist()[::-1],
                        y=df_fore['upper_bound'].tolist() + df_fore['lower_bound'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(200,241,53,0.15)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Prediction Interval'
                    ))

            fig.update_layout(
                xaxis_title='Month', yaxis_title='Amount',
                legend=dict(
                    orientation='h',
                    yanchor='bottom', y=1.02,
                    xanchor='right', x=1,
                    font=dict(color='#e8eaf0', size=12),  # bright white text
                ),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,24,28,1)',
                font=dict(color='#e8eaf0'),
                xaxis=dict(gridcolor='#2a2d35'), yaxis=dict(gridcolor='#2a2d35')
            )
            st.plotly_chart(fig, use_container_width=True)
            if forecast and 'upper_bound' in pd.DataFrame(forecast).columns:
                st.caption("Shaded region indicates 95% prediction interval.")

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

        # ── Seasonal Decomposition ────────────────────────────────────────────
        if decomp and decomp.get('trend') and decomp.get('seasonal'):
            st.markdown("### 📊 Seasonal Decomposition (STL)")
            st.caption("Trend, seasonal, and residual components extracted from the series.")

            def _decomp_df(key):
                rows = decomp.get(key, [])
                if not rows:
                    return pd.DataFrame()
                d = pd.DataFrame(rows)
                d['month_year'] = pd.to_datetime(d['month_year'], format='%b-%Y')
                return d.set_index('month_year')

            trend_df    = _decomp_df('trend')
            seasonal_df = _decomp_df('seasonal')
            resid_df    = _decomp_df('residual')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Trend**")
                if not trend_df.empty:
                    st.line_chart(trend_df, use_container_width=True)
            with col2:
                st.markdown("**Seasonal**")
                if not seasonal_df.empty:
                    st.line_chart(seasonal_df, use_container_width=True)
            with col3:
                st.markdown("**Residual**")
                if not resid_df.empty:
                    st.line_chart(resid_df, use_container_width=True)

            st.markdown("---")

        # ── Seasonally Adjusted ───────────────────────────────────────────────
        if adjusted:
            st.markdown("### 📉 Seasonally Adjusted Spend")
            adj_df = pd.DataFrame(adjusted)
            adj_df['month_year'] = pd.to_datetime(adj_df['month_year'], format='%b-%Y')
            adj_df = adj_df.set_index('month_year')
            st.line_chart(adj_df, use_container_width=True)
            st.markdown("---")

        # ── Diagnostics & Model Info ──────────────────────────────────────────
        if diag:
            st.markdown("### 🔬 Model Diagnostics")
            col1, col2, col3 = st.columns(3)
            with col1:
                adf_p = diag.get('adf_pvalue')
                if adf_p is not None:
                    is_stat = diag.get('adf_is_stationary', adf_p < 0.05)
                    st.metric("ADF p-value", f"{adf_p:.4f}",
                              delta="Stationary" if is_stat else "Non-stationary",
                              delta_color="normal" if is_stat else "inverse")
            with col2:
                lb_p = diag.get('ljung_box_pvalue')
                if lb_p is not None:
                    ok = lb_p > 0.05
                    st.metric("Ljung-Box p-value", f"{lb_p:.4f}",
                              delta="Residuals OK" if ok else "Autocorrelation present",
                              delta_color="normal" if ok else "inverse")
            with col3:
                dw = diag.get('durbin_watson')
                if dw is not None:
                    dw_ok = 1.5 < dw < 2.5
                    st.metric("Durbin-Watson", f"{dw:.3f}",
                              delta="No autocorr" if dw_ok else "Possible issue",
                              delta_color="normal" if dw_ok else "inverse")

        if model_info:
            with st.expander("🤖 Model Details", expanded=False):
                st.json(model_info)

        if diag or model_info:
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

        # ── Amount by source (with forecast) ────────────────────────────────
        st.markdown("### 🏷️ Spend by Campaign Source (Mall vs Brand)")
        st.caption("Actual (solid) and forecast (dashed) with 95% prediction intervals.")

        source_forecasts = ts.get('source_forecasts', {})
        if not source_forecasts:
            # Fallback to old static bar chart if forecasts not available
            by_source = ts.get('amount_by_source', [])
            if by_source:
                df_src = pd.DataFrame(by_source)
                if 'campaign_source' in df_src.columns and 'month_year' in df_src.columns:
                    try:
                        pivoted = df_src.pivot(
                            index='month_year', columns='campaign_source', values='amount'
                        ).fillna(0)
                        st.bar_chart(pivoted)
                    except Exception:
                        st.dataframe(df_src, use_container_width=True, hide_index=True)
            else:
                st.info("No source breakdown available.")
        else:
            fig_src = go.Figure()
            colors    = {'mall': '#ff5c5c', 'brand': '#4af0c4'}
            # Keep opacity low (0.12) so the two bands don't swallow each other
            pi_colors = {'mall': 'rgba(255,92,92,0.12)', 'brand': 'rgba(74,240,196,0.12)'}

            # Draw PI bands FIRST so forecast lines render on top
            for source, data in source_forecasts.items():
                if 'error' in data and 'actual' not in data:
                    continue
                forecast_df = pd.DataFrame(data.get('forecast', []))
                if forecast_df.empty:
                    continue
                forecast_df['month_year'] = pd.to_datetime(
                    forecast_df['month_year'], format='%b-%Y')
                # Only rows where both bounds are present and valid
                has_bounds = (
                    'lower_bound' in forecast_df.columns and
                    'upper_bound' in forecast_df.columns and
                    forecast_df['lower_bound'].notna().any() and
                    forecast_df['upper_bound'].notna().any()
                )
                if has_bounds:
                    band = forecast_df.dropna(subset=['lower_bound', 'upper_bound'])
                    x_fill = band['month_year'].tolist() + band['month_year'].tolist()[::-1]
                    y_fill = band['upper_bound'].tolist() + band['lower_bound'].tolist()[::-1]
                    fig_src.add_trace(go.Scatter(
                        x=x_fill, y=y_fill,
                        fill='toself',
                        fillcolor=pi_colors.get(source, 'rgba(255,255,255,0.12)'),
                        line=dict(color='rgba(0,0,0,0)'),
                        showlegend=True,
                        name=f'{source.title()} 95% PI',
                        legendgroup=source,
                    ))

            # Draw actual + forecast lines on top of bands
            for source, data in source_forecasts.items():
                if 'error' in data and 'actual' not in data:
                    st.warning(f"{source.title()}: {data['error']}")
                    continue

                actual_df = pd.DataFrame(data.get('actual', []))
                if not actual_df.empty:
                    actual_df['month_year'] = pd.to_datetime(
                        actual_df['month_year'], format='%b-%Y')
                    fig_src.add_trace(go.Scatter(
                        x=actual_df['month_year'], y=actual_df['value'],
                        mode='lines+markers', name=f'{source.title()} Actual',
                        line=dict(color=colors.get(source, '#ffffff'), width=2),
                        legendgroup=source,
                    ))

                forecast_df = pd.DataFrame(data.get('forecast', []))
                if not forecast_df.empty:
                    forecast_df['month_year'] = pd.to_datetime(
                        forecast_df['month_year'], format='%b-%Y')
                    fig_src.add_trace(go.Scatter(
                        x=forecast_df['month_year'], y=forecast_df['forecast'],
                        mode='lines+markers', name=f'{source.title()} Forecast',
                        line=dict(color=colors.get(source, '#ffffff'), dash='dash', width=2),
                        legendgroup=source,
                    ))

            fig_src.update_layout(
                xaxis_title='Month', yaxis_title='Amount ($)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1, font=dict(color='#e8eaf0', size=12)),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,24,28,1)',
                font=dict(color='#e8eaf0'),
                xaxis=dict(gridcolor='#2a2d35'), yaxis=dict(gridcolor='#2a2d35')
            )
            st.plotly_chart(fig_src, use_container_width=True)
            st.caption("🟢 Brand | 🔴 Mall — Dashed lines = forecast, shaded = 95% prediction interval.")

        st.markdown("---")
        # ── Member vs Non-Member Sales ────────────────────────────────────
        st.markdown("---")
        st.markdown("### 👥 Member vs Non-Member Sales")
        st.caption("Member Sales = campaign redemption revenue. "
                   "Non-Member Sales = GTO Amount − Member Sales.")

        if not mnm:
            st.info("No Member/Non-Member data — re-run the pipeline with GTO file.")
        else:
            ms  = mnm.get('member_sales', {})
            nms = mnm.get('non_member_sales', {})

            if ms.get('actual') or nms.get('actual'):
                # ── Build combined actual dataframe ───────────────────────
                frames = []
                if ms.get('actual'):
                    df_ms = pd.DataFrame(ms['actual']).rename(
                        columns={'value': 'Member Sales (Actual)'})
                    frames.append(df_ms.set_index('month_year'))

                if nms.get('actual'):
                    df_nms = pd.DataFrame(nms['actual']).rename(
                        columns={'value': 'Non-Member Sales (Actual)'})
                    frames.append(df_nms.set_index('month_year'))

                # ── Build combined forecast dataframe ─────────────────────
                fore_frames = []
                if ms.get('forecast'):
                    df_ms_fore = pd.DataFrame(ms['forecast']).rename(
                        columns={'forecast': 'Member Sales (Forecast)'})
                    if 'month_year' in df_ms_fore.columns:
                        fore_frames.append(df_ms_fore.set_index('month_year')[['Member Sales (Forecast)']])

                if nms.get('forecast'):
                    df_nms_fore = pd.DataFrame(nms['forecast']).rename(
                        columns={'forecast': 'Non-Member Sales (Forecast)'})
                    if 'month_year' in df_nms_fore.columns:
                        fore_frames.append(df_nms_fore.set_index('month_year')[['Non-Member Sales (Forecast)']])

                # ── Combine all into one chart ────────────────────────────
                all_frames = frames + fore_frames
                if all_frames:
                    df_combined = pd.concat(all_frames, axis=1)

                    # Sort index chronologically
                    df_combined.index = pd.to_datetime(
                        df_combined.index, format='%b-%Y', errors='coerce')
                    df_combined = df_combined[df_combined.index.year > 2000]  # remove invalid dates
                    df_combined = df_combined.sort_index()
                    df_combined.index = df_combined.index.strftime('%b-%Y')

                    fig_mnm = go.Figure()

                    colors = {
                        'Member Sales (Actual)':          '#4af0c4',
                        'Non-Member Sales (Actual)':      '#ff5c5c',
                        'Member Sales (Forecast)':        '#c8f135',
                        'Non-Member Sales (Forecast)':    '#fbbf24',
                    }

                    for col in df_combined.columns:
                        is_forecast = 'Forecast' in col
                        fig_mnm.add_trace(go.Scatter(
                            x=df_combined.index,
                            y=df_combined[col],
                            mode='lines+markers',
                            name=col,
                            line=dict(
                                color=colors.get(col, '#ffffff'),
                                dash='dash' if is_forecast else 'solid',
                                width=2,
                            ),
                            marker=dict(size=5),
                        ))

                    fig_mnm.update_layout(
                        xaxis_title='Month',
                        yaxis_title='Amount ($)',
                        legend=dict(
                            orientation='h',
                            yanchor='bottom', y=1.02,
                            xanchor='right', x=1,
                            font=dict(color='#e8eaf0', size=12),  # bright legend text
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(22,24,28,1)',
                        font=dict(color='#e8eaf0'),
                        xaxis=dict(gridcolor='#2a2d35'),
                        yaxis=dict(gridcolor='#2a2d35'),
                    )
                    st.plotly_chart(fig_mnm, use_container_width=True)
                    st.caption("Solid lines = actual data | Dashed lines = forecast. "
                               "🟢 Member Sales | 🔴 Non-Member Sales")
                    st.markdown("**Monthly Breakdown**")
                    df_table = df_combined.copy()
                    df_table.index.name = 'Month'

                    # Format numbers
                    st.dataframe(
                        df_table.style.format("${:,.0f}", na_rep="—"),
                        use_container_width=True,
                        column_config={
                            'Member Sales (Actual)':       st.column_config.NumberColumn('Member Sales Actual ($)',       format='$%.0f'),
                            'Non-Member Sales (Actual)':   st.column_config.NumberColumn('Non-Member Sales Actual ($)',   format='$%.0f'),
                            'Member Sales (Forecast)':     st.column_config.NumberColumn('Member Sales Forecast ($)',     format='$%.0f'),
                            'Non-Member Sales (Forecast)': st.column_config.NumberColumn('Non-Member Sales Forecast ($)', format='$%.0f'),
                        }
                    )


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
        if ts_filters_active:
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
    # ── Filters ───────────────────────────────────────────────────────────────
    reg_filters_active = False
    reg_active_parts: list = []

    if _data_ready:
        _reg_df = load_campaign_csv(_cleaned_path)
        with st.expander("🔽  Filters", expanded=False):
            _c1, _c2, _c3, _c4 = st.columns(4)
            with _c1:
                _opts = sorted(_reg_df['campaign_source'].dropna().astype(str).unique().tolist()) if 'campaign_source' in _reg_df.columns else []
                reg_filt_source = tuple(st.multiselect("Campaign Source", _opts, key="reg_filt_source", placeholder="All sources"))
            with _c2:
                _opts = sorted(_reg_df['outlet_name'].dropna().astype(str).unique().tolist()) if 'outlet_name' in _reg_df.columns else []
                reg_filt_outlet = tuple(st.multiselect("Outlet Name", _opts, key="reg_filt_outlet", placeholder="All outlets"))
            with _c3:
                try:
                    _opts = sorted(_reg_df['month_year'].dropna().astype(str).unique().tolist(), key=lambda x: pd.to_datetime(x, format='%b-%Y', errors='coerce')) if 'month_year' in _reg_df.columns else []
                except Exception:
                    _opts = sorted(_reg_df['month_year'].dropna().astype(str).unique().tolist()) if 'month_year' in _reg_df.columns else []
                reg_filt_months = tuple(st.multiselect("Month", _opts, key="reg_filt_months", placeholder="All months"))
            with _c4:
                _pre = _apply_filters(_reg_df, reg_filt_source, reg_filt_outlet, reg_filt_months)
                _opts = sorted(_pre['voucher_code'].dropna().astype(str).unique().tolist()) if 'voucher_code' in _pre.columns else []
                reg_filt_campaign = tuple(st.multiselect("Campaign", _opts, key="reg_filt_campaign", placeholder="All campaigns"))
            if reg_filt_source:   reg_active_parts.append(f"Source: {', '.join(reg_filt_source)}")
            if reg_filt_outlet:   reg_active_parts.append(f"Outlet: {', '.join(str(x) for x in reg_filt_outlet)}")
            if reg_filt_months:   reg_active_parts.append(f"Months: {', '.join(reg_filt_months)}")
            if reg_filt_campaign: reg_active_parts.append(f"Campaign: {', '.join(reg_filt_campaign)}")
            st.caption(f"🔍 {' | '.join(reg_active_parts)}" if reg_active_parts else "No filters active — showing pre-computed results.")
        reg_filters_active = bool(reg_filt_source or reg_filt_outlet or reg_filt_months or reg_filt_campaign)
    else:
        reg_filt_source = reg_filt_outlet = reg_filt_months = reg_filt_campaign = ()

    if reg_filters_active:
        with st.spinner("Running regression on filtered data…"):
            lr_raw = get_reg_filtered(_cleaned_path, reg_filt_source, reg_filt_outlet, reg_filt_months, reg_filt_campaign)
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
        if reg_filters_active and reg_active_parts:
            st.caption(f"🔍 Showing filtered results — {' | '.join(reg_active_parts)}")

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
            'Regression 1 — Y = Amount (per receipt, by campaign)': 'regression_1_amount'
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
            # ── Campaign Summary Table ─────────────────────────────────────────────
            camp_summary = model.get('campaign_summary', [])
            if camp_summary and model_options[selected_label] == 'regression_1_amount':
                st.markdown("---")
                st.markdown("#### 🏆 Campaign Revenue Summary")
                st.caption("Total revenue, average spend per receipt, and redemption count per campaign. Sorted by total revenue. 🟡 = statistically significant in regression (p<0.05)")

                df_camp = pd.DataFrame(camp_summary)

                # is_brand → Source
                if 'is_brand' in df_camp.columns:
                    df_camp['source'] = df_camp['is_brand'].map({0: 'Mall', 1: 'Brand'})
                    df_camp = df_camp.drop(columns=['is_brand'])

                # significant campaigns (from coef_table)
                sig_camps = set()
                if coef_table:
                    df_coef_sig = pd.DataFrame(coef_table)
                    sig_rows = df_coef_sig[
                        (df_coef_sig['significant'] == True) &
                        (df_coef_sig['feature'].str.startswith('camp_'))
                    ]
                    # remove camp_ prefix and match with voucher_code
                    for feat in sig_rows['feature'].tolist():
                        sig_camps.add(feat.replace('camp_', '', 1))

                col_order = [c for c in ['voucher_code', 'source', 'total_amount',
                                        'avg_amount', 'n_receipts'] if c in df_camp.columns]
                df_camp = df_camp[col_order]

                # highlight significant rows
                def hl_camp(row):
                    # if voucher_code is in significant camp list -> yellow
                    code = str(row.get('voucher_code', ''))
                    for s in sig_camps:
                        if s.lower() in code.lower() or code.lower() in s.lower():
                            return ['background-color:#3d3800; color:#fbbf24'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    df_camp.style.apply(hl_camp, axis=1),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'voucher_code':  st.column_config.TextColumn('Campaign'),
                        'source':        st.column_config.TextColumn('Source'),
                        'total_amount':  st.column_config.NumberColumn('Total Revenue ($)', format='$%.0f'),
                        'avg_amount':    st.column_config.NumberColumn('Avg per Receipt ($)', format='$%.0f'),
                        'n_receipts':    st.column_config.NumberColumn('Redemptions', format='%d'),
                    }
                )

                # Top 10 bar chart
                st.markdown("**Top 10 Campaigns by Total Revenue**")
                top10 = df_camp.head(10).copy()
                top10_mall  = top10[top10['source'] == 'Mall'].set_index('voucher_code')[['total_amount']].rename(columns={'total_amount': 'Mall'})
                top10_brand = top10[top10['source'] == 'Brand'].set_index('voucher_code')[['total_amount']].rename(columns={'total_amount': 'Brand'})

                # distinguish mall/brand
                top10_chart = pd.DataFrame(index=top10['voucher_code'])
                top10_chart['Mall']  = top10.set_index('voucher_code')['total_amount'].where(top10.set_index('voucher_code')['source'] == 'Mall')
                top10_chart['Brand'] = top10.set_index('voucher_code')['total_amount'].where(top10.set_index('voucher_code')['source'] == 'Brand')
                st.bar_chart(top10_chart, color=['#ff5c5c', '#4af0c4'])

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

        # ── Download ──────────────────────────────────────────────────────────
        if reg_filters_active:
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