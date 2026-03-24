import streamlit as st
import pandas as pd
import subprocess
import sys
import os
import threading
import queue
import time
import json
from pathlib import Path

st.set_page_config(
    page_title="Capstone Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:      #0e0f11;
    --surface: #16181c;
    --border:  #2a2d35;
    --accent:  #c8f135;
    --accent2: #4af0c4;
    --text:    #e8eaf0;
    --muted:   #6b7280;
    --danger:  #ff5c5c;
    --warn:    #fbbf24;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stAppViewContainer"] { padding: 0 !important; }
[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] { display: none; }
[data-testid="block-container"] { padding: 2rem 3rem !important; max-width: 1400px !important; }

.app-header {
    display: flex; align-items: baseline; gap: 1rem;
    margin-bottom: 2.5rem; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem;
}
.app-title { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: var(--accent); letter-spacing: -0.03em; margin: 0; }
.app-sub   { font-size: 0.75rem; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; }

[data-testid="stTabs"] { width: 100% !important; }
[data-testid="stTabs"] > div:first-child { width: 100% !important; }
[data-testid="stTabs"] [role="tablist"] {
    display: flex !important; flex-direction: row !important;
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 2rem !important; gap: 0 !important; overflow: visible !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    color: var(--muted) !important; border: none !important;
    background: transparent !important; padding: 0.5rem 1.2rem !important;
    border-radius: 0 !important; display: block !important;
    visibility: visible !important; opacity: 1 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important;
}

.log-box {
    background: #0a0b0d; border: 1px solid var(--border); border-radius: 6px;
    padding: 1rem 1.2rem; font-family: 'DM Mono', monospace; font-size: 0.78rem;
    line-height: 1.7; max-height: 480px; overflow-y: auto;
    white-space: pre-wrap; color: #b0b8c8;
}
[data-testid="stTextInput"] input {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.82rem !important;
}
label { color: var(--muted) !important; font-size: 0.72rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
[data-testid="baseButton-primary"] {
    background: var(--accent) !important; color: #0e0f11 !important; border: none !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.82rem !important; letter-spacing: 0.08em !important;
    border-radius: 6px !important; padding: 0.6rem 1.5rem !important;
}
[data-testid="baseButton-secondary"] {
    background: transparent !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important; border-radius: 6px !important;
}
[data-testid="stSuccess"] { background: #1a2e1a !important; border: 1px solid #2a4a2a !important; color: var(--accent2) !important; border-radius: 6px !important; }
[data-testid="stError"]   { background: #2e1a1a !important; border: 1px solid #4a1a1a !important; color: var(--danger)  !important; border-radius: 6px !important; }
[data-testid="stWarning"] { background: #2e2a1a !important; border: 1px solid #4a3a1a !important; color: var(--warn)    !important; border-radius: 6px !important; }
[data-testid="stInfo"]    { background: #1a222e !important; border: 1px solid #1a3a4a !important; color: #60d0ff       !important; border-radius: 6px !important; }
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

.metric-box { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem 1.5rem; text-align: center; }
.metric-num { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; color: var(--accent); line-height: 1; }
.metric-label { font-size: 0.65rem; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.3rem; }

.reg-insight {
    background: #1a1f2e; border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0; padding: 0.8rem 1.2rem;
    font-size: 0.82rem; color: var(--text); margin: 0.8rem 0; line-height: 1.6;
}

/* ── Final matches summary table ── */
.final-match-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 0.5rem; }
.final-match-table th {
    background: var(--surface); color: var(--muted);
    font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.6rem 1rem; border-bottom: 1px solid var(--border); text-align: left;
}
.final-match-table td { padding: 0.5rem 1rem; border-bottom: 1px solid #1e2025; }
.final-match-table tr:hover td { background: #1a1c20; }
.match-val       { color: #4af0c4; font-weight: 500; }
.match-unmatched { color: #ff5c5c; font-style: italic; }
.method-pill {
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 3px;
    font-size: 0.65rem; letter-spacing: 0.08em; text-transform: uppercase;
}
.pill-exact, .pill-combined_exact { background: #1a2e1a; color: #4af0c4; }
.pill-fuzzy, .pill-combined_fuzzy { background: #2e2a1a; color: #fbbf24; }
.pill-confirmed, .pill-code_match { background: #1a2a2e; color: #60d0ff; }
.pill-unmatched { background: #2e1a1a; color: #ff5c5c; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent

def get_config_files():
    return sorted([f.name for f in APP_DIR.glob("config_*.xlsx")])

def load_config(config_file: str) -> dict:
    path = APP_DIR / config_file
    if not path.exists():
        return {}
    df = pd.read_excel(path, sheet_name='paths')
    return dict(zip(df['Setting'].astype(str).str.strip(), df['Value'].astype(str).str.strip()))

def load_shop_mapping(mapping_path: str) -> pd.DataFrame:
    p = Path(mapping_path)
    if not p.exists():
        return pd.DataFrame(columns=['campaign_name','gto_name','suggested_gto_name','confirmed_gto_name','method'])
    return pd.read_excel(p, sheet_name='mapping').fillna('')

def save_shop_mapping(mapping_path: str, df: pd.DataFrame):
    instr = pd.DataFrame([
        {"Instructions": "campaign_name: outlet name from campaign data — DO NOT edit."},
        {"Instructions": "gto_name: GTO name used for matching (confirmed > suggested)."},
        {"Instructions": "suggested_gto_name: best automatic match — DO NOT edit."},
        {"Instructions": "confirmed_gto_name: fill this to override suggestion or fix unmatched."},
        {"Instructions": "method: exact/fuzzy/confirmed = matched; unmatched = no GTO; gto_only = GTO with no campaign."},
    ])
    with pd.ExcelWriter(mapping_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='mapping', index=False)
        instr.to_excel(writer, sheet_name='instructions', index=False)

def load_schemas(schema_path: str) -> dict:
    p = Path(schema_path)
    if not p.exists():
        return {}
    xl = pd.ExcelFile(p)
    return {sheet: xl.parse(sheet) for sheet in xl.sheet_names}

def save_schemas(schema_path: str, sheets: dict):
    with pd.ExcelWriter(schema_path, engine='openpyxl') as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet, index=False)

def run_pipeline(config_file: str, log_queue: queue.Queue):
    process = subprocess.Popen(
        [sys.executable, str(APP_DIR / 'main.py'), config_file],
        cwd=str(APP_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in process.stdout:
        log_queue.put(line)
    process.wait()
    log_queue.put(f"__EXIT__{process.returncode}")


# ── FIX 1: compute the final match the pipeline will use — mirrors pipeline logic exactly ──
def compute_final_match(row) -> str:
    confirmed = str(row.get('confirmed_gto_name', '')).strip()
    suggested = str(row.get('suggested_gto_name', '')).strip()
    if confirmed and confirmed.lower() != 'nan':
        return confirmed
    if suggested and suggested.lower() != 'nan':
        return suggested
    return ''


def prepare_mapping_display(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a display-ready copy of the mapping df:
      - Auto-fills confirmed_gto_name for auto-matched rows that have a blank confirmed
        so exact/fuzzy rows never show a blank confirmed field.
      - Adds final_match column showing exactly what the pipeline will use.
    Does NOT write to disk.
    """
    df = df_raw.copy()
    AUTO_MATCHED = {'exact', 'fuzzy', 'code_match', 'combined_exact', 'combined_fuzzy'}

    fill_mask = (
        df['method'].isin(AUTO_MATCHED) &
        (df['confirmed_gto_name'].astype(str).str.strip().isin(['', 'nan'])) &
        (~df['suggested_gto_name'].astype(str).str.strip().isin(['', 'nan']))
    )
    df.loc[fill_mask, 'confirmed_gto_name'] = df.loc[fill_mask, 'suggested_gto_name']
    df['final_match'] = df.apply(compute_final_match, axis=1)
    return df


# ── Session state ─────────────────────────────────────────────────────────────
if 'log_lines'          not in st.session_state: st.session_state.log_lines          = []
if 'last_exit'          not in st.session_state: st.session_state.last_exit          = None
if 'running'            not in st.session_state: st.session_state.running            = False
if 'drag_matches'       not in st.session_state: st.session_state.drag_matches       = {}
if 'mapping_editor_v'   not in st.session_state: st.session_state.mapping_editor_v   = 0
if 'mapping_just_saved' not in st.session_state: st.session_state.mapping_just_saved = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">⚡ PIPELINE</div>
  <div class="app-sub">ETL → Power BI · Capstone 2025</div>
</div>
""", unsafe_allow_html=True)

configs = get_config_files()
if not configs:
    st.error("No config_*.xlsx file found in the app folder.")
    st.stop()

_NEW_LABEL = "＋  Add new config..."
_col_sel, _col_new = st.columns([3, 2])
with _col_sel:
    _options = configs + [_NEW_LABEL]
    _choice  = st.selectbox("Config", _options, key="global_config", label_visibility="collapsed")
with _col_new:
    if _choice == _NEW_LABEL:
        _new_name = st.text_input("New config name", placeholder="e.g. Alice",
                                  key="new_config_name", label_visibility="collapsed")
    else:
        st.empty()

if _choice == _NEW_LABEL:
    _raw = st.session_state.get("new_config_name", "").strip()
    _raw = _raw.removeprefix("config_").removesuffix(".xlsx")
    selected_config = f"config_{_raw}.xlsx" if _raw else None
    _is_new = True
else:
    selected_config = _choice
    _is_new = False

if selected_config is None:
    st.info("Enter a name for your new config above, then fill in the paths in the ⚙ Config tab and click Save.")
    st.stop()

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

tab_run, tab_config, tab_mapping, tab_schema, tab_ts, tab_reg = st.tabs([
    "▶  RUN", "⚙  CONFIG", "🔗  SHOP MAPPING", "📋  SCHEMA",
    "📈  TIME SERIES", "📉  REGRESSION"
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

    status_placeholder = st.empty()
    log_placeholder    = st.empty()

    if st.session_state.running:
        q = st.session_state.log_queue
        while True:
            try:
                line = q.get_nowait()
                if line.startswith("__EXIT__"):
                    st.session_state.last_exit = int(line.replace("__EXIT__", ""))
                    st.session_state.running   = False
                    break
                st.session_state.log_lines.append(line.rstrip())
            except queue.Empty:
                break

    if st.session_state.running or st.session_state.log_lines or st.session_state.last_exit is not None:
        all_log         = "\n".join(st.session_state.log_lines)
        loader_done     = "OK Data Loader completed" in all_log or "COMPLETED" in all_log
        regression_done = "OK Regression completed"  in all_log or "REGRESSION COMPLETED" in all_log
        linreg_done     = "OK Linear Regression completed" in all_log or "LINEAR REGRESSION COMPLETED" in all_log

        if st.session_state.running:
            if linreg_done:       status_placeholder.info("⏳ Finalising...")
            elif regression_done: status_placeholder.info("✅ Regression complete — running Linear Regression...")
            elif loader_done:     status_placeholder.info("✅ Data Loader complete — running Regression...")
            else:                 status_placeholder.info("⏳ Running Data Loader...")
        elif st.session_state.last_exit == 0:
            status_placeholder.success("✅ Data Loader complete  ·  ✅ Regression complete  ·  ✅ Linear Regression complete")
        elif st.session_state.last_exit is not None:
            error_lines = [l for l in st.session_state.log_lines if 'error' in l.lower()]
            hint = error_lines[-1] if error_lines else "Check logs for details."
            status_placeholder.error(f"❌ Pipeline failed — {hint}")

        if st.session_state.log_lines:
            _hide = ('DEBUG ','Loading configuration from','raw_data from config',
                     'cleaned_data from config','schemas from config','FutureWarning',
                     'DeprecationWarning','UserWarning','warnings.warn','✓ Loaded:',
                     '🗑 Deleted','✅ Exported','✅ Loaded shop_mapping','✅ Loaded ',
                     '✅ Successfully loaded paths','GTO header configurations',
                     '📁 Raw data path','📁 Output path','📁 Schema file',
                     '📁 Cleaned data','📁 Combined data','📖 Loaded config',
                     '   - gto/','   - cat/')
            visible_lines = [l for l in st.session_state.log_lines[-300:]
                             if not any(pat in l for pat in _hide)]
            log_placeholder.markdown(
                f'<div class="log-box">{chr(10).join(visible_lines)}</div>',
                unsafe_allow_html=True)

        if st.session_state.running:
            time.sleep(0.8)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
with tab_config:
    paths = load_config(selected_config) if not _is_new else {}
    if _is_new:
        st.info(f"Creating new config: **{selected_config}** — fill in the paths below and click Save.")

    path_labels = {
        'raw_data':      'Raw Data Folder',
        'cleaned_data':  'Cleaned Data Folder',
        'combined_data': 'Combined Data Folder',
        'schemas':       'Schemas File (.xlsx)',
        'shop_mapping':  'Shop Mapping File (.xlsx)',
    }

    def open_in_explorer(path_str: str, select_file: bool = False):
        p = Path(path_str.strip()) if path_str.strip() else None
        if not p:
            st.warning("No path entered.")
            return
        target = p; folder = p.parent if select_file else p
        walk = folder
        while walk != walk.parent and not walk.exists():
            walk = walk.parent
        if sys.platform == 'win32':
            subprocess.Popen(f'explorer /select,"{target}"') if select_file and target.exists() else subprocess.Popen(f'explorer "{walk}"')
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', '-R', str(target)]) if select_file and target.exists() else subprocess.Popen(['open', str(walk)])
        else:
            subprocess.Popen(['xdg-open', str(walk)])

    new_paths = {}
    for key, label in path_labels.items():
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            new_paths[key] = st.text_input(label, value=paths.get(key, ''), key=f"path_{key}")
        with col_btn:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            is_file = key in ('schemas', 'shop_mapping')
            if st.button("📄" if is_file else "📂", key=f"open_{key}", help=f"Open {label} in file explorer"):
                if new_paths[key].strip():
                    open_in_explorer(new_paths[key], select_file=is_file)
                else:
                    st.warning("Enter a path first.")

    st.markdown("---")
    st.markdown("#### GTO Header Rows")
    st.caption("Row number where column headers start in each GTO file (1-indexed)")

    try:
        gto_df = pd.read_excel(APP_DIR / selected_config, sheet_name='gto_headers')
    except:
        gto_df = pd.DataFrame([
            {'category': 'gto', 'dataset': 'monthly_sales',   'header_row': 7},
            {'category': 'gto', 'dataset': 'monthly_rent',    'header_row': 8},
            {'category': 'gto', 'dataset': 'tenant_turnover', 'header_row': 7},
        ])

    gto_vals = {}
    g_cols = st.columns(3)
    for col, (_, row) in zip(g_cols, gto_df.iterrows()):
        with col:
            k = f"{row['category']}_{row['dataset']}"
            gto_vals[k] = st.number_input(
                row['dataset'].replace('_', ' ').title(),
                min_value=1, max_value=50, value=int(row['header_row']), key=f"gto_{k}")

    st.markdown("---")
    st.markdown("#### 🧹 Data Cleaning")
    st.caption("Controls how blanks and outliers are handled when the pipeline runs")

    _BLANK_NUM_OPTS    = ['zero', 'mean', 'median', 'drop_row']
    _BLANK_STR_OPTS    = ['empty', 'drop_row']
    _OUTLIER_METH_OPTS = ['none', 'iqr', 'zscore', 'winsorise']
    _OUTLIER_ACT_OPTS  = ['cap', 'drop_row']

    try:
        _clean_df  = pd.read_excel(APP_DIR / selected_config, sheet_name='data_cleaning')
        _clean_cfg = dict(zip(_clean_df['Setting'].astype(str).str.strip().str.lower().str.replace(' ','_'),
                              _clean_df['Value'].astype(str).str.strip().str.lower()))
    except:
        _clean_cfg = {}

    def _idx(opts, key, default):
        val = _clean_cfg.get(key, default)
        return opts.index(val) if val in opts else opts.index(default)

    cl1, cl2 = st.columns(2)
    with cl1:
        blank_numeric  = st.selectbox("Blank numeric cells", _BLANK_NUM_OPTS,
                                      index=_idx(_BLANK_NUM_OPTS,  'blank_numeric',  'zero'),  key='cl_blank_num')
        blank_string   = st.selectbox("Blank text cells",    _BLANK_STR_OPTS,
                                      index=_idx(_BLANK_STR_OPTS,  'blank_string',   'empty'), key='cl_blank_str')
    with cl2:
        outlier_method = st.selectbox("Outlier detection",   _OUTLIER_METH_OPTS,
                                      index=_idx(_OUTLIER_METH_OPTS,'outlier_method', 'none'), key='cl_out_meth')
        outlier_action = st.selectbox("Outlier action",      _OUTLIER_ACT_OPTS,
                                      index=_idx(_OUTLIER_ACT_OPTS, 'outlier_action', 'cap'),  key='cl_out_act')

    _thresh_default   = float(_clean_cfg.get('outlier_threshold', '1.5') or '1.5')
    outlier_threshold = st.number_input(
        "Outlier threshold  (IQR multiplier · Z-score cutoff · Winsorise percentile)",
        min_value=0.1, max_value=10.0, value=_thresh_default, step=0.1, key='cl_threshold')

    _desc = {'zero':'Fill blank numbers with 0','mean':'Fill blank numbers with the column mean',
             'median':'Fill blank numbers with the column median','drop_row':'Drop any row that contains a blank',
             'empty':'Fill blank text with an empty string','none':'No outlier detection — keep all values',
             'iqr':'Flag values outside Q1 − k×IQR … Q3 + k×IQR',
             'zscore':'Flag values where |z-score| > threshold',
             'winsorise':'Cap at the threshold-th and (100−threshold)-th percentile',
             'cap':'Replace outliers with the boundary value'}
    st.caption(f"ℹ️  Blanks: {_desc.get(blank_numeric,'')}. Outliers: {_desc.get(outlier_method,'')} → {_desc.get(outlier_action,'')}.")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("💾  Save Config", type="primary"):
        new_gto_rows  = [{'category': row['category'], 'dataset': row['dataset'],
                          'header_row': gto_vals[f"{row['category']}_{row['dataset']}"]}
                         for _, row in gto_df.iterrows()]
        cleaning_rows = [{'Setting': 'blank_numeric',     'Value': blank_numeric},
                         {'Setting': 'blank_string',      'Value': blank_string},
                         {'Setting': 'outlier_method',    'Value': outlier_method},
                         {'Setting': 'outlier_action',    'Value': outlier_action},
                         {'Setting': 'outlier_threshold', 'Value': outlier_threshold}]
        paths_df = pd.DataFrame([{'Setting': k, 'Value': v} for k, v in new_paths.items()])
        with pd.ExcelWriter(APP_DIR / selected_config, engine='openpyxl') as writer:
            paths_df.to_excel(writer, sheet_name='paths', index=False)
            pd.DataFrame(new_gto_rows).to_excel(writer, sheet_name='gto_headers', index=False)
            pd.DataFrame(cleaning_rows).to_excel(writer, sheet_name='data_cleaning', index=False)
        st.success(f"✅ Saved {selected_config}")
        if _is_new:
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHOP MAPPING
# FIXES:
#   1. Blanks shown as exact — auto-fill confirmed_gto_name on load & display
#   2. After save, table reloads cleanly from disk (mapping_just_saved flag)
#   3. New "Final Matches" summary showing exactly what the pipeline will use
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mapping:
    import streamlit.components.v1 as components

    paths        = load_config(selected_config)
    mapping_path = paths.get('shop_mapping', '')

    if not mapping_path:
        st.warning("shop_mapping path not set in config.")
    else:
        # FIX 2: clear the just-saved flag on this render pass (drag_matches already
        # cleared on save, so df_map_raw below reads fresh from disk)
        if st.session_state.mapping_just_saved:
            st.session_state.mapping_just_saved = False

        # Always read fresh from disk
        df_map_raw = load_shop_mapping(mapping_path)

        if df_map_raw.empty:
            st.info("No shop mapping file yet — run the pipeline first to generate it.")
        else:
            # Apply any in-session drag-drop changes before display prep
            for camp_name, gto_name in st.session_state.drag_matches.items():
                mask = df_map_raw['campaign_name'].str.strip().str.lower() == camp_name.strip().lower()
                df_map_raw.loc[mask, 'confirmed_gto_name'] = gto_name
                df_map_raw.loc[mask, 'method']             = 'confirmed'
                df_map_raw.loc[mask, 'gto_name']           = gto_name

            # FIX 1: auto-fill blanks and compute final_match for display
            df_map = prepare_mapping_display(df_map_raw)

            # ── Metrics ───────────────────────────────────────────────────
            method_counts = df_map['method'].value_counts().to_dict()
            all_methods   = ['confirmed','code_match','combined_exact','combined_fuzzy',
                             'exact','fuzzy','unmatched','gto_only']
            m_cols = st.columns(len(all_methods))
            for col, method in zip(m_cols, all_methods):
                with col:
                    st.markdown(f"""<div class="metric-box">
                        <div class="metric-num">{method_counts.get(method, 0)}</div>
                        <div class="metric-label">{method}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

            # ══════════════════════════════════════════════════════════════
            # FIX 3: FINAL MATCHES SUMMARY
            # ══════════════════════════════════════════════════════════════
            campaign_only = df_map[df_map['method'] != 'gto_only'].copy()
            n_unmatched   = int((campaign_only['final_match'] == '').sum())
            n_resolved    = int((campaign_only['final_match'] != '').sum())

            summary_label = (
                f"✅ Final Matches — what the pipeline will use  "
                f"({n_resolved} resolved · "
                f"{'⚠ ' if n_unmatched else ''}{n_unmatched} unmatched)"
            )
            with st.expander(summary_label, expanded=(n_unmatched > 0)):
                st.caption(
                    "This shows the exact GTO name the pipeline will use for each campaign outlet. "
                    "Unmatched rows (⚠) are excluded from all analysis. "
                    "Fix them in the All Mappings table below, then Save."
                )

                summary_df = campaign_only[
                    campaign_only['campaign_name'].astype(str).str.strip() != ''
                ][['campaign_name', 'final_match', 'method']].copy()
                summary_df['_sort'] = summary_df['final_match'].apply(lambda x: 0 if x == '' else 1)
                summary_df = (summary_df.sort_values(['_sort', 'campaign_name'])
                                        .drop(columns='_sort')
                                        .reset_index(drop=True))

                rows_html = ''
                for _, row in summary_df.iterrows():
                    camp   = str(row['campaign_name'])
                    match  = str(row['final_match'])
                    method = str(row['method'])
                    match_html = (
                        '<span class="match-unmatched">⚠ no match — will be excluded from analysis</span>'
                        if match == '' else
                        f'<span class="match-val">{match}</span>'
                    )
                    pill_cls = (f'pill-{method}' if method in
                                ('exact','fuzzy','confirmed','code_match',
                                 'combined_exact','combined_fuzzy','unmatched')
                                else 'pill-unmatched')
                    rows_html += f"""<tr>
                        <td>{camp}</td>
                        <td>{match_html}</td>
                        <td><span class="method-pill {pill_cls}">{method}</span></td>
                    </tr>"""

                st.markdown(f"""
                <table class="final-match-table">
                  <thead><tr>
                    <th>Campaign Outlet Name</th>
                    <th>Final GTO Match (used by pipeline)</th>
                    <th>Method</th>
                  </tr></thead>
                  <tbody>{rows_html}</tbody>
                </table>""", unsafe_allow_html=True)

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

            # ── Drag-and-drop widget ───────────────────────────────────────
            gto_only_rows    = df_map[df_map['method'] == 'gto_only'].copy()
            campaign_rows_df = df_map[df_map['method'] != 'gto_only'].copy()

            st.markdown("#### 🎯 Shop Matching")
            st.caption("Pick a GTO match for each unmatched campaign. Pending matches appear below — click Save when done.")

            def _existing(row):
                c = str(row.get('confirmed_gto_name', '')).strip()
                g = str(row.get('gto_name', '')).strip()
                return c if c and c != 'nan' else (g if g and g != 'nan' else '')

            all_campaign_list_raw = [
                {'name': str(r['campaign_name']).strip(), 'method': str(r['method']).strip(), 'gto': _existing(r)}
                for _, r in campaign_rows_df.iterrows()
                if str(r.get('campaign_name', '')).strip() not in ('', 'nan')
            ]
            all_campaign_list = (
                sorted([x for x in all_campaign_list_raw if x['method'] == 'unmatched'], key=lambda x: x['name'].lower()) +
                sorted([x for x in all_campaign_list_raw if x['method'] != 'unmatched'], key=lambda x: x['name'].lower())
            )
            gto_only_list = sorted([
                r for r in gto_only_rows['gto_name'].tolist()
                if r and str(r).strip() not in ('', 'nan')
            ])

            # ── Pending session matches ────────────────────────────────────
            if st.session_state.drag_matches:
                st.markdown("**Pending matches (unsaved):**")
                for camp_key, gto_val in list(st.session_state.drag_matches.items()):
                    _c1, _c2, _c3 = st.columns([4, 4, 1])
                    _c1.write(camp_key)
                    _c2.write(f"→ **{gto_val}**")
                    if _c3.button("✕", key=f"rm_match_{camp_key}", help="Remove this match"):
                        del st.session_state.drag_matches[camp_key]
                        st.rerun()
                st.markdown("---")

            # ── Unmatched campaigns ────────────────────────────────────────
            _matched_keys = {k.strip().lower() for k in st.session_state.drag_matches}
            _unmatched = [
                x for x in all_campaign_list
                if x['method'] == 'unmatched' and x['name'].strip().lower() not in _matched_keys
            ]

            if _unmatched:
                if gto_only_list:
                    st.markdown("**Unmatched campaigns — select a GTO name:**")
                    _gto_options = ['— select —'] + gto_only_list
                    for _camp in _unmatched:
                        _col1, _col2 = st.columns([4, 5])
                        _col1.markdown(f"⚠ `{_camp['name']}`")
                        _chosen = _col2.selectbox(
                            f"gto_for_{_camp['name']}",
                            _gto_options,
                            key=f"sel_{_camp['name']}",
                            label_visibility="collapsed",
                        )
                        if _chosen != '— select —':
                            st.session_state.drag_matches[_camp['name'].lower()] = _chosen
                            st.rerun()
                else:
                    st.info("No GTO-only shops available to match against.")
            else:
                st.success("✅ All campaigns are matched!")


            # ── All Mappings editable table ────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📋 All Mappings")
            st.caption(
                "The **✅ Final Match** column shows exactly what the pipeline will use. "
                "Edit **✏ Confirmed** to override any automatic match, then click Save."
            )

            editor_cols = [c for c in ['campaign_name','final_match','confirmed_gto_name',
                                        'suggested_gto_name','gto_name','method']
                           if c in df_map.columns]
            edited = st.data_editor(
                df_map[editor_cols].copy(),
                use_container_width=True,
                column_config={
                    "campaign_name":      st.column_config.TextColumn("Campaign Name",                disabled=True),
                    "final_match":        st.column_config.TextColumn("✅ Final Match (will be used)", disabled=True),
                    "confirmed_gto_name": st.column_config.TextColumn("✏ Confirmed (override)",       disabled=False),
                    "suggested_gto_name": st.column_config.TextColumn("Suggested",                    disabled=True),
                    "gto_name":           st.column_config.TextColumn("GTO Name",                     disabled=True),
                    "method":             st.column_config.TextColumn("Method",                       disabled=True),
                },
                hide_index=True,
                key=f"mapping_editor_{st.session_state.mapping_editor_v}"
            )

            st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
            if st.button("💾  Save Shop Mapping", type="primary"):
                if isinstance(edited, dict):
                    edited = pd.DataFrame(edited)

                # Merge edits back onto the full raw df
                save_df = df_map_raw.copy()
                for _, erow in edited.iterrows():
                    camp = str(erow.get('campaign_name', '')).strip().lower()
                    conf = str(erow.get('confirmed_gto_name', '')).strip()
                    if not camp or camp == 'nan':
                        continue
                    mask = save_df['campaign_name'].astype(str).str.strip().str.lower() == camp
                    if conf and conf.lower() != 'nan':
                        save_df.loc[mask, 'confirmed_gto_name'] = conf
                        save_df.loc[mask, 'method']             = 'confirmed'
                        save_df.loc[mask, 'gto_name']           = conf

                # Apply drag matches on top
                for camp_name, gto_name in st.session_state.drag_matches.items():
                    mask = save_df['campaign_name'].str.strip().str.lower() == camp_name.strip().lower()
                    save_df.loc[mask, 'confirmed_gto_name'] = gto_name
                    save_df.loc[mask, 'method']             = 'confirmed'
                    save_df.loc[mask, 'gto_name']           = gto_name

                # Auto-fill confirmed_gto_name for auto-matched rows before writing to disk
                # so the saved file is also clean — no blank confirmed on exact/fuzzy rows
                AUTO_MATCHED = {'exact', 'fuzzy', 'code_match', 'combined_exact', 'combined_fuzzy'}
                fill_mask = (
                    save_df['method'].isin(AUTO_MATCHED) &
                    (save_df['confirmed_gto_name'].astype(str).str.strip().isin(['', 'nan'])) &
                    (~save_df['suggested_gto_name'].astype(str).str.strip().isin(['', 'nan']))
                )
                save_df.loc[fill_mask, 'confirmed_gto_name'] = save_df.loc[fill_mask, 'suggested_gto_name']

                # Remove gto_only rows whose name is now used as a confirmed match
                confirmed_set = set(
                    save_df.loc[~save_df['confirmed_gto_name'].astype(str).str.strip().isin(['', 'nan']),
                                'confirmed_gto_name'].astype(str).str.strip().str.lower()
                )
                before   = len(save_df)
                save_df  = save_df[~(
                    (save_df['method'] == 'gto_only') &
                    (save_df['gto_name'].astype(str).str.strip().str.lower().isin(confirmed_set))
                )].reset_index(drop=True)
                removed  = before - len(save_df)

                # Drop computed column before writing
                save_df  = save_df.drop(columns=['final_match'], errors='ignore')
                save_shop_mapping(mapping_path, save_df)

                # FIX 2: signal clean reload; clear in-session state
                st.session_state.drag_matches       = {}
                st.session_state.mapping_editor_v  += 1
                st.session_state.mapping_just_saved = True

                st.success(f"✅ Saved! {removed} gto_only row(s) removed. Reloading table…")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_schema:
    paths       = load_config(selected_config)
    schema_path = paths.get('schemas', '')

    if not schema_path:
        st.warning("Schema path not set in config.")
    elif not Path(schema_path).exists():
        st.error(f"Schema file not found: {schema_path}")
    else:
        sheets = load_schemas(schema_path)
        selected_sheet = st.selectbox("Dataset", list(sheets.keys()))
        df_schema = sheets[selected_sheet].copy()

        edited_schema = st.data_editor(
            df_schema, use_container_width=True, num_rows="dynamic",
            column_config={
                "original_column":  st.column_config.TextColumn("Original Column"),
                "canonical_column": st.column_config.TextColumn("Canonical Column"),
            },
            hide_index=True, key=f"schema_editor_{selected_sheet}")

        if st.button("💾  Save Schema", type="primary"):
            sheets[selected_sheet] = edited_schema
            save_schemas(schema_path, sheets)
            st.success(f"✅ Saved schema for '{selected_sheet}'")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ts:
    paths_ts           = load_config(selected_config)
    combined_folder_ts = paths_ts.get('combined_data', '')
    json_path_ts       = os.path.join(combined_folder_ts, 'insights.json') if combined_folder_ts else ''

    if not json_path_ts or not Path(json_path_ts).exists():
        st.info("No time series data yet — run the pipeline first.")
    else:
        with open(json_path_ts) as f:
            ts_ins = json.load(f)
        ts = ts_ins.get('time_series', {})
        if not ts:
            st.info("No time series results found — re-run the pipeline to generate them.")
        else:
            if ts.get('summary'):
                st.info(f"🔍 {ts['summary'].get('interpretation', '')}")

            st.markdown("### 📉 GTO Revenue — Trend & Forecast")
            gto_ts = ts.get('gto_trend', {})
            if gto_ts.get('error'):
                st.warning(gto_ts['error'])
            elif gto_ts.get('actual'):
                df_actual   = pd.DataFrame(gto_ts['actual']).rename(columns={'value': 'Actual GTO'})
                df_forecast = pd.DataFrame(gto_ts.get('forecast', [])).rename(columns={'forecast': 'Forecast'})
                df_chart    = df_actual.set_index('month_year')[['Actual GTO']]
                if not df_forecast.empty:
                    df_chart = df_chart.join(df_forecast.set_index('month_year')[['Forecast']], how='outer')
                st.line_chart(df_chart)
                trend = gto_ts.get('trend', {})
                if trend:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Trend Direction", trend.get('direction', '—').title())
                    c2.metric("Trend Strength",  trend.get('strength',  '—').title())
                    c3.metric("R²",              f"{trend.get('r_squared') or 0:.3f}")
                    st.caption("✅ Statistically significant (p < 0.05)" if trend.get('significant')
                               else "⚠️ Not statistically significant (p ≥ 0.05)")
                if gto_ts.get('moving_average'):
                    st.markdown("**3-Month Moving Average**")
                    st.line_chart(pd.DataFrame(gto_ts['moving_average']).rename(columns={'value':'3-month MA'}).set_index('month_year'))
                if gto_ts.get('decomposition'):
                    d = gto_ts['decomposition']
                    st.markdown("**Seasonal Decomposition**")
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        if d.get('trend'):
                            st.markdown("Trend Component")
                            st.line_chart(pd.DataFrame(d['trend']).rename(columns={'value':'Trend'}).set_index('month_year'))
                    with dc2:
                        if d.get('seasonal'):
                            st.markdown("Seasonal Component")
                            st.line_chart(pd.DataFrame(d['seasonal']).rename(columns={'value':'Seasonal'}).set_index('month_year'))
                anomalies = gto_ts.get('anomalies', [])
                if anomalies:
                    st.markdown("**⚠️ Anomalous Months (GTO)**")
                    st.dataframe(pd.DataFrame(anomalies), use_container_width=True, hide_index=True)
                else:
                    st.caption("No anomalous months detected in GTO revenue.")

            st.markdown("---")
            st.markdown("### 🎯 Campaign Activity — Trend & Forecast")
            camp_ts = ts.get('campaign_trend', {})
            if camp_ts.get('error'):
                st.warning(camp_ts['error'])
            elif camp_ts.get('actual'):
                df_actual_c   = pd.DataFrame(camp_ts['actual']).rename(columns={'value': 'Actual Activity'})
                df_forecast_c = pd.DataFrame(camp_ts.get('forecast', [])).rename(columns={'forecast': 'Forecast'})
                df_chart_c    = df_actual_c.set_index('month_year')[['Actual Activity']]
                if not df_forecast_c.empty:
                    df_chart_c = df_chart_c.join(df_forecast_c.set_index('month_year')[['Forecast']], how='outer')
                st.line_chart(df_chart_c)
                trend_c = camp_ts.get('trend', {})
                if trend_c:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Trend Direction", trend_c.get('direction', '—').title())
                    c2.metric("Trend Strength",  trend_c.get('strength',  '—').title())
                    c3.metric("R²",              f"{trend_c.get('r_squared') or 0:.3f}")
                anomalies_c = camp_ts.get('anomalies', [])
                if anomalies_c:
                    st.markdown("**⚠️ Anomalous Months (Campaign)**")
                    st.dataframe(pd.DataFrame(anomalies_c), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### 🔗 Lead-Lag: Do Campaigns Drive Future GTO?")
            st.caption("Tests whether campaign activity today predicts GTO revenue 0–3 months later.")
            lead_lag = ts.get('lead_lag', {})
            if lead_lag:
                df_ll = pd.DataFrame([
                    {'Lag (months)': int(k.replace('lag_','')), 'Correlation': v.get('correlation'),
                     'P-value': v.get('p_value'), 'Significant': '✅' if v.get('significant') else '',
                     'Description': v.get('label', '')}
                    for k, v in lead_lag.items()
                ])
                st.dataframe(df_ll, use_container_width=True, hide_index=True)
                st.bar_chart(df_ll.set_index('Lag (months)')[['Correlation']])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reg:
    paths_reg    = load_config(selected_config)
    combined_reg = paths_reg.get('combined_data', '')
    reg_json     = os.path.join(combined_reg, 'linear_regression_results.json') if combined_reg else ''
    reg_xlsx     = os.path.join(combined_reg, 'linear_regression_summary.xlsx') if combined_reg else ''

    if not reg_json or not Path(reg_json).exists():
        st.info("No regression results yet — run the pipeline first to generate linear regression analysis.")
        st.stop()

    with open(reg_json) as f:
        reg_data = json.load(f)
    lr = reg_data.get('linear_regression', {})
    if not lr:
        st.info("No regression results found in file.")
        st.stop()

    st.markdown("### 📊 Model Comparison Summary")
    st.caption("All regression models across monthly / outlet / panel levels, full and stepwise.")
    summary_rows = lr.get('summary', [])
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        display_cols = [c for c in ['model_key','model_type','level','target',
                                    'r_squared','adj_r_squared','f_pvalue','n_obs','cv_rmse','cv_r2']
                        if c in df_sum.columns]
        st.dataframe(df_sum[display_cols], use_container_width=True, hide_index=True,
                     column_config={
                         'r_squared':     st.column_config.ProgressColumn('R²',     min_value=0, max_value=1, format='%.3f'),
                         'adj_r_squared': st.column_config.ProgressColumn('Adj R²', min_value=0, max_value=1, format='%.3f'),
                         'cv_r2':         st.column_config.ProgressColumn('CV R²',  min_value=0, max_value=1, format='%.3f'),
                     })
        valid = df_sum.dropna(subset=['adj_r_squared'])
        if not valid.empty:
            best = valid.loc[valid['adj_r_squared'].idxmax()]
            st.markdown(
                f'<div class="reg-insight">🏆 Best model: <strong>{best["model_key"]}</strong> '
                f'({best["model_type"]}) — Adj-R² = <strong>{best["adj_r_squared"]:.3f}</strong>, '
                f'CV-R² = {best.get("cv_r2") or "—"}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Explore Individual Models")
    model_keys = [k for k in lr.keys() if k != 'summary']
    if not model_keys:
        st.warning("No individual model results found.")
        st.stop()

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_key = st.selectbox("Select model", model_keys,
                                    format_func=lambda k: k.replace('_',' ').title(), key="reg_model_key")
    with col_sel2:
        model_type = st.radio("Model type", ["full_model","stepwise_model"],
                              format_func=lambda x: "Full (all features)" if x=="full_model" else "Stepwise (significant only)",
                              horizontal=True, key="reg_model_type")

    model = lr.get(selected_key, {}).get(model_type, {})
    if model.get('error'):
        st.error(f"Model error: {model['error']}")
        st.stop()
    if model.get('insight'):
        st.markdown(f'<div class="reg-insight">💡 {model["insight"]}</div>', unsafe_allow_html=True)

    st.markdown("#### Model Fit")
    fit = model.get('model_fit', {}); cv = model.get('cross_validation', {})
    mc1,mc2,mc3,mc4,mc5,mc6 = st.columns(6)
    mc1.metric("R²",        f"{fit.get('r_squared') or 0:.3f}")
    mc2.metric("Adj R²",    f"{fit.get('adj_r_squared') or 0:.3f}")
    mc3.metric("F p-value", f"{fit.get('f_pvalue') or 0:.4f}")
    mc4.metric("N obs",     fit.get('n_obs', '—'))
    mc5.metric("CV RMSE",   f"{cv.get('cv_rmse_mean') or 0:,.1f}" if cv.get('cv_rmse_mean') else "—")
    mc6.metric("CV R²",     f"{cv.get('cv_r2_mean') or 0:.3f}"    if cv.get('cv_r2_mean')  else "—")
    dw = fit.get('dw_stat')
    if dw is not None:
        st.caption(f"Durbin-Watson: {dw:.3f}  —  {'✅ No autocorrelation' if 1.5 < dw < 2.5 else '⚠️ Possible autocorrelation'}")

    st.markdown("---")
    st.markdown("#### Coefficients (standardised)")
    coef_table = model.get('coef_table', [])
    if coef_table:
        df_coef      = pd.DataFrame(coef_table)
        df_coef_plot = df_coef[df_coef['feature'] != 'const'].copy()
        col_chart, col_table = st.columns([2,3])
        with col_chart:
            if not df_coef_plot.empty and 'coef' in df_coef_plot.columns:
                st.markdown("**Coefficient magnitudes**")
                st.bar_chart(df_coef_plot.set_index('feature')['coef'])
        with col_table:
            dcols = [c for c in ['feature','coef','std_err','t_stat','p_value','ci_lower','ci_upper','significant']
                     if c in df_coef.columns]
            def highlight_sig(row):
                return ['background-color: #1a2e1a']*len(row) if row.get('significant') is True else ['']*len(row)
            st.dataframe(df_coef[dcols].style.apply(highlight_sig, axis=1), use_container_width=True, hide_index=True)
        st.caption("✅ Green rows = significant at p < 0.05. Standardised β — comparable in magnitude.")

    st.markdown("---")
    vif_data = model.get('vif', [])
    if vif_data:
        st.markdown("#### Multicollinearity Check (VIF)")
        df_vif = pd.DataFrame(vif_data)
        cv1, cv2 = st.columns([1,2])
        with cv1:
            st.dataframe(df_vif, use_container_width=True, hide_index=True)
        with cv2:
            st.caption("VIF < 5 → ✅  |  5–10 → ⚠️ Moderate  |  > 10 → ❌ High")
            if not df_vif.empty and 'vif' in df_vif.columns:
                max_vif = df_vif['vif'].dropna().max()
                if max_vif and max_vif > 10:   st.warning(f"⚠️ Max VIF = {max_vif:.1f} — multicollinearity detected.")
                elif max_vif and max_vif > 5:  st.warning(f"⚠️ Max VIF = {max_vif:.1f} — moderate multicollinearity.")
                else:                          st.success("✅ All VIF values look good (< 5).")

    st.markdown("---")
    if model_type == 'stepwise_model':
        dropped = model.get('stepwise_dropped', []); final = model.get('stepwise_final_features', [])
        if dropped:
            st.markdown("#### Stepwise Feature Selection")
            sd1, sd2 = st.columns(2)
            with sd1:
                st.markdown("**Dropped features** (p ≥ 0.10)")
                st.dataframe(pd.DataFrame(dropped), use_container_width=True, hide_index=True)
            with sd2:
                st.markdown("**Retained features**")
                st.dataframe(pd.DataFrame({'feature': final}), use_container_width=True, hide_index=True)
        st.markdown("---")

    residuals = model.get('residuals', [])
    if residuals:
        st.markdown("#### Residual Diagnostics")
        df_resid = pd.DataFrame(residuals)
        cr1, cr2 = st.columns(2)
        with cr1:
            st.markdown("**Fitted vs Residuals**")
            st.scatter_chart(df_resid, x='fitted', y='residual', height=280)
            st.caption("Ideal: randomly scattered around 0 — no funnel shape.")
        with cr2:
            st.markdown("**Standardised Residuals Distribution**")
            st.bar_chart(df_resid['std_resid'].value_counts(bins=15).sort_index(), height=280)
            st.caption("Ideal: roughly bell-shaped around 0.")

    st.markdown("---")
    if reg_xlsx and Path(reg_xlsx).exists():
        with open(reg_xlsx, 'rb') as fh:
            st.download_button("⬇️  Download Regression Summary (.xlsx)", data=fh.read(),
                               file_name='linear_regression_summary.xlsx',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                               type='primary')