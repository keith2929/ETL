import streamlit as st
import pandas as pd
import subprocess
import sys
import os
import threading
import queue
import time
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
    display: flex !important;
    flex-direction: row !important;
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 2rem !important;
    gap: 0 !important;
    overflow: visible !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    border: none !important;
    background: transparent !important;
    padding: 0.5rem 1.2rem !important;
    border-radius: 0 !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
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

.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 500; }
.badge-exact     { background: #1a2e1a; color: var(--accent2); border: 1px solid #2a4a2a; }
.badge-fuzzy     { background: #2e2a1a; color: var(--warn);    border: 1px solid #4a3a1a; }
.badge-confirmed { background: #1a2a2e; color: #60d0ff;        border: 1px solid #1a3a4a; }
.badge-unmatched { background: #2e1a1a; color: var(--danger);  border: 1px solid #4a1a1a; }
.badge-gto_only  { background: #2a1a2e; color: #c084fc;        border: 1px solid #3a1a4a; }
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

# ── Session state ─────────────────────────────────────────────────────────────
if 'log_lines' not in st.session_state: st.session_state.log_lines = []
if 'running'   not in st.session_state: st.session_state.running   = False
if 'last_exit' not in st.session_state: st.session_state.last_exit = None
if 'log_queue' not in st.session_state: st.session_state.log_queue = None

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

tab_run, tab_config, tab_mapping, tab_schema = st.tabs([
    "▶  RUN", "⚙  CONFIG", "🔗  SHOP MAPPING", "📋  SCHEMA"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run:
    selected_config = st.selectbox("Config", configs, label_visibility="collapsed")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    run_clicked = st.button(
        "▶  RUN PIPELINE",
        type="primary",
        disabled=st.session_state.running,
        use_container_width=True
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if run_clicked and not st.session_state.running:
        st.session_state.log_lines = []
        st.session_state.last_exit = None
        st.session_state.running   = True
        q = queue.Queue()
        st.session_state.log_queue = q
        threading.Thread(target=run_pipeline, args=(selected_config, q), daemon=True).start()

    if st.session_state.running or st.session_state.log_lines:
        log_placeholder    = st.empty()
        status_placeholder = st.empty()

        if st.session_state.running:
            q = st.session_state.log_queue
            while True:
                try:
                    line = q.get(timeout=0.05)
                    if line.startswith("__EXIT__"):
                        st.session_state.last_exit = int(line.replace("__EXIT__", ""))
                        st.session_state.running   = False
                        break
                    st.session_state.log_lines.append(line.rstrip())
                except queue.Empty:
                    break

        log_placeholder.markdown(
            f'<div class="log-box">{"<br>".join(st.session_state.log_lines)}</div>',
            unsafe_allow_html=True
        )

        if st.session_state.running:
            status_placeholder.info("⏳ Pipeline running...")
            time.sleep(0.3)
            st.rerun()
        elif st.session_state.last_exit == 0:
            status_placeholder.success("✅ Pipeline completed successfully.")
        elif st.session_state.last_exit is not None:
            status_placeholder.error(f"❌ Pipeline failed (exit code {st.session_state.last_exit}). See log above.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
with tab_config:
    selected_config_edit = st.selectbox("Config file", configs, key="config_edit_sel")
    paths = load_config(selected_config_edit)

    path_labels = {
        'raw_data':      'Raw Data Folder',
        'cleaned_data':  'Cleaned Data Folder',
        'combined_data': 'Combined Data Folder',
        'schemas':       'Schemas File (.xlsx)',
        'shop_mapping':  'Shop Mapping File (.xlsx)',
    }

    new_paths = {}
    for key, label in path_labels.items():
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            new_paths[key] = st.text_input(label, value=paths.get(key, ''), key=f"path_{key}")
        with col_btn:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            folder = new_paths[key]
            # For folder paths, open in Explorer / Finder
            if key in ('raw_data', 'cleaned_data', 'combined_data'):
                if st.button("📂", key=f"open_{key}", help=f"Open {label} in file explorer"):
                    if folder and Path(folder).exists():
                        if sys.platform == 'win32':
                            os.startfile(folder)
                        else:
                            subprocess.Popen(['open', folder])
                    else:
                        st.warning(f"Folder not found: {folder}")
            # For file paths, open the file directly
            else:
                if st.button("📄", key=f"open_{key}", help=f"Open {label}"):
                    if folder and Path(folder).exists():
                        if sys.platform == 'win32':
                            os.startfile(folder)
                        else:
                            subprocess.Popen(['open', folder])
                    else:
                        st.warning(f"File not found: {folder}")

    st.markdown("---")
    st.markdown("#### GTO Header Rows")
    st.caption("Row number where column headers start in each GTO file (1-indexed)")

    try:
        gto_df = pd.read_excel(APP_DIR / selected_config_edit, sheet_name='gto_headers')
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
                min_value=1, max_value=50,
                value=int(row['header_row']),
                key=f"gto_{k}"
            )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("💾  Save Config", type="primary"):
        new_gto_rows = [
            {'category': row['category'], 'dataset': row['dataset'],
             'header_row': gto_vals[f"{row['category']}_{row['dataset']}"]}
            for _, row in gto_df.iterrows()
        ]
        paths_df = pd.DataFrame([{'Setting': k, 'Value': v} for k, v in new_paths.items()])
        with pd.ExcelWriter(APP_DIR / selected_config_edit, engine='openpyxl') as writer:
            paths_df.to_excel(writer, sheet_name='paths', index=False)
            pd.DataFrame(new_gto_rows).to_excel(writer, sheet_name='gto_headers', index=False)
        st.success(f"✅ Saved {selected_config_edit}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHOP MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mapping:
    sel_cfg      = st.selectbox("Config file", configs, key="mapping_cfg")
    paths        = load_config(sel_cfg)
    mapping_path = paths.get('shop_mapping', '')

    if not mapping_path:
        st.warning("shop_mapping path not set in config.")
    else:
        df_map = load_shop_mapping(mapping_path)

        if df_map.empty:
            st.info("No shop mapping file yet — run the pipeline first to generate it.")
        else:
            # Metrics
            method_counts = df_map['method'].value_counts().to_dict()
            m_cols = st.columns(5)
            for col, method in zip(m_cols, ['exact','fuzzy','confirmed','unmatched','gto_only']):
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                      <div class="metric-num">{method_counts.get(method, 0)}</div>
                      <div class="metric-label">{method}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            edited = st.data_editor(
                df_map,
                use_container_width=True,
                column_config={
                    "campaign_name":      st.column_config.TextColumn("Campaign Name",  disabled=True),
                    "gto_name":           st.column_config.TextColumn("GTO Name",       disabled=True),
                    "suggested_gto_name": st.column_config.TextColumn("Suggested",      disabled=True),
                    "confirmed_gto_name": st.column_config.TextColumn("✏ Confirmed",   disabled=False),
                    "method":             st.column_config.TextColumn("Method",         disabled=True),
                },
                hide_index=True,
                key="mapping_editor"
            )

            if st.button("💾  Save Shop Mapping", type="primary"):
                save_shop_mapping(mapping_path, edited)
                st.success("✅ Saved. Re-run the pipeline to apply changes.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_schema:
    sel_cfg     = st.selectbox("Config file", configs, key="schema_cfg")
    paths       = load_config(sel_cfg)
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
            df_schema,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "original_column":  st.column_config.TextColumn("Original Column"),
                "canonical_column": st.column_config.TextColumn("Canonical Column"),
            },
            hide_index=True,
            key=f"schema_editor_{selected_sheet}"
        )

        if st.button("💾  Save Schema", type="primary"):
            sheets[selected_sheet] = edited_schema
            save_schemas(schema_path, sheets)
            st.success(f"✅ Saved schema for '{selected_sheet}'")