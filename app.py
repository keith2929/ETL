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
if 'last_exit' not in st.session_state: st.session_state.last_exit = None
if 'running'   not in st.session_state: st.session_state.running   = False

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

# ── Config selector + new config inline ───────────────────────────────────────
_NEW_LABEL = "＋  Add new config..."
_col_sel, _col_new = st.columns([3, 2])

with _col_sel:
    _options = configs + [_NEW_LABEL]
    _choice  = st.selectbox("Config", _options, key="global_config", label_visibility="collapsed")

with _col_new:
    if _choice == _NEW_LABEL:
        _new_name = st.text_input(
            "New config name", placeholder="e.g. Alice",
            key="new_config_name", label_visibility="collapsed"
        )
    else:
        st.empty()

# Resolve which config is actually selected
if _choice == _NEW_LABEL:
    _raw = st.session_state.get("new_config_name", "").strip()
    # Strip any accidental "config_" prefix and ".xlsx" suffix the user might type
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

tab_run, tab_config, tab_mapping, tab_schema, tab_insights = st.tabs([
    "▶  RUN", "⚙  CONFIG", "🔗  SHOP MAPPING", "📋  SCHEMA", "📊  INSIGHTS"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run:
    run_clicked = st.button(
        "▶  RUN PIPELINE",
        type="primary",
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

    # Persistent placeholders — created once, updated in-place so only their
    # content changes, not the whole page. This eliminates the flicker.
    status_placeholder = st.empty()
    log_placeholder    = st.empty()

    if st.session_state.running:
        # Drain ALL available lines in one go before rendering
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

        # Update status badge in-place (no full page re-render)
        if st.session_state.running:
            if regression_done:
                status_placeholder.info("⏳ Finalising...")
            elif loader_done:
                status_placeholder.info("✅ Data Loader complete — running Regression...")
            else:
                status_placeholder.info("⏳ Running Data Loader...")
        elif st.session_state.last_exit == 0:
            status_placeholder.success("✅ Data Loader complete  ·  ✅ Regression complete")
        elif st.session_state.last_exit is not None:
            error_lines = [l for l in st.session_state.log_lines if 'error' in l.lower()]
            hint = error_lines[-1] if error_lines else "Check logs for details."
            status_placeholder.error(f"❌ Pipeline failed — {hint}")

        # Update log box in-place — no flicker since placeholder already exists
        if st.session_state.log_lines:
            # Patterns to hide from the client-facing log — still printed to console
            _hide = (
                'DEBUG ',
                'Loading configuration from',
                'raw_data from config',
                'cleaned_data from config',
                'schemas from config',
                'FutureWarning',
                'DeprecationWarning',
                'UserWarning',
                'warnings.warn',
                '✓ Loaded:',
                '🗑 Deleted',
                '✅ Exported',
                '✅ Loaded shop_mapping',
                '✅ Loaded ', '✅ Successfully loaded paths',
                'GTO header configurations',
                '📁 Raw data path',
                '📁 Output path',
                '📁 Schema file',
                '📁 Cleaned data',
                '📁 Combined data',
                '📖 Loaded config',
                '   - gto/',
                '   - cat/',
            )
            visible_lines = [
                l for l in st.session_state.log_lines[-300:]
                if not any(pat in l for pat in _hide)
            ]
            log_text = "\n".join(visible_lines)
            log_placeholder.markdown(
                f'<div class="log-box">{log_text}</div>',
                unsafe_allow_html=True
            )

        # Only rerun while actively running; longer interval = less flicker
        if st.session_state.running:
            time.sleep(0.8)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
with tab_config:
    # For a new config, start with empty paths; for existing, load from file
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
        """Open file explorer at the given path.
        - Folders: open the folder directly (create if needed).
        - Files: open the parent folder with the file selected (Windows) or just the folder (Mac).
        - If path doesn't exist, walk up to the nearest existing ancestor.
        """
        p = Path(path_str.strip()) if path_str.strip() else None
        if not p:
            st.warning("No path entered.")
            return
        # For a file path, the target to highlight is the file; the folder to open is its parent
        if select_file:
            target = p
            folder = p.parent
        else:
            target = p
            folder = p
        # Walk up to nearest existing ancestor
        walk = folder
        while walk != walk.parent and not walk.exists():
            walk = walk.parent
        if sys.platform == 'win32':
            if select_file and target.exists():
                # /select highlights the specific file in Explorer
                subprocess.Popen(f'explorer /select,"{target}"')
            else:
                subprocess.Popen(f'explorer "{walk}"')
        elif sys.platform == 'darwin':
            if select_file and target.exists():
                subprocess.Popen(['open', '-R', str(target)])
            else:
                subprocess.Popen(['open', str(walk)])
        else:
            subprocess.Popen(['xdg-open', str(walk)])

    new_paths = {}
    for key, label in path_labels.items():
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            new_paths[key] = st.text_input(label, value=paths.get(key, ''), key=f"path_{key}")
        with col_btn:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            target_path = new_paths[key]
            is_file = key in ('schemas', 'shop_mapping')
            icon    = "📄" if is_file else "📂"
            tip     = f"Open {label} in file explorer"
            if st.button(icon, key=f"open_{key}", help=tip):
                if target_path.strip():
                    open_in_explorer(target_path, select_file=is_file)
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
        with pd.ExcelWriter(APP_DIR / selected_config, engine='openpyxl') as writer:
            paths_df.to_excel(writer, sheet_name='paths', index=False)
            pd.DataFrame(new_gto_rows).to_excel(writer, sheet_name='gto_headers', index=False)
        st.success(f"✅ Saved {selected_config}")
        if _is_new:
            st.rerun()  # refresh so new config appears in the dropdown


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHOP MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mapping:
    sel_cfg      = selected_config
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
    sel_cfg     = selected_config
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    import json

    paths        = load_config(selected_config)
    combined_folder = paths.get('combined_data', '')
    json_path    = os.path.join(combined_folder, 'insights.json') if combined_folder else ''

    if not json_path or not Path(json_path).exists():
        st.info("No insights found yet — run the pipeline first to generate analysis.")
    else:
        with open(json_path) as f:
            ins = json.load(f)

        # ── Month-on-month trends ─────────────────────────────────────────
        st.markdown("### 📈 Month-on-Month Trends")
        mom = ins.get('mom_trends', {})

        col1, col2 = st.columns(2)
        with col1:
            if mom.get('redemptions_by_month'):
                df_r = pd.DataFrame(mom['redemptions_by_month'])
                st.markdown("**Campaign Redemptions**")
                st.bar_chart(df_r.set_index('month_year')['redemptions'])
        with col2:
            if mom.get('gto_by_month'):
                df_g = pd.DataFrame(mom['gto_by_month'])
                st.markdown("**GTO Revenue ($)**")
                st.bar_chart(df_g.set_index('month_year')['total_gto_amount'])

        if mom.get('redemptions_by_month'):
            st.dataframe(pd.DataFrame(mom['redemptions_by_month']), use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Campaign ROI ──────────────────────────────────────────────────
        st.markdown("### 💰 Campaign ROI")
        roi = ins.get('campaign_roi', {})

        if roi.get('summary'):
            s = roi['summary']
            avg_roi    = s.get('avg_roi_ratio')
            median_roi = s.get('median_roi_ratio')
            total_v    = s.get('total_voucher_redeemed') or 0
            total_gto  = s.get('total_gto_revenue')      or 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg ROI Ratio",         f"{avg_roi}x"    if avg_roi    is not None else "—")
            c2.metric("Median ROI Ratio",      f"{median_roi}x" if median_roi is not None else "—")
            c3.metric("Total Txn Revenue ($)", f"${total_v:,.0f}")
            c4.metric("Total GTO Revenue ($)", f"${total_gto:,.0f}")

        if roi.get('monthly_roi'):
            df_roi = pd.DataFrame(roi['monthly_roi'])
            st.markdown("**ROI Ratio by Month**")
            st.line_chart(df_roi.set_index('month_year')['roi_ratio'])

        if roi.get('top_outlets_by_roi'):
            st.markdown("**Top 10 Outlets by ROI**")
            st.dataframe(pd.DataFrame(roi['top_outlets_by_roi']), use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Campaign type ─────────────────────────────────────────────────
        st.markdown("### 🏷️ Brand-Funded vs Mall-Funded")
        ct = ins.get('campaign_type', {})

        if ct.get('by_funding_type'):
            st.dataframe(pd.DataFrame(ct['by_funding_type']), use_container_width=True, hide_index=True)

        if ct.get('gto_by_funding_type'):
            df_ct = pd.DataFrame(ct['gto_by_funding_type'])
            col1, col2 = st.columns(2)
            with col1:
                if 'funding_type' in df_ct.columns and 'total_gto_revenue' in df_ct.columns:
                    st.markdown("**GTO Revenue by Funding Type**")
                    st.bar_chart(df_ct.set_index('funding_type')['total_gto_revenue'])
            with col2:
                if 'funding_type' in df_ct.columns and 'roi_ratio' in df_ct.columns:
                    st.markdown("**ROI Ratio by Funding Type**")
                    st.bar_chart(df_ct.set_index('funding_type')['roi_ratio'])

        if ct.get('monthly_by_type'):
            df_monthly = pd.DataFrame(ct['monthly_by_type'])
            if 'funding_type' in df_monthly.columns and 'month_year' in df_monthly.columns:
                try:
                    pivoted = df_monthly.pivot(index='month_year', columns='funding_type', values='redemptions').fillna(0)
                    st.markdown("**Monthly Redemptions by Funding Type**")
                    st.bar_chart(pivoted)
                except Exception:
                    st.dataframe(df_monthly, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Loyalty points ────────────────────────────────────────────────
        st.markdown("### 💳 Loyalty Points Effectiveness")
        loy = ins.get('loyalty_points', {})

        if loy.get('correlations'):
            c = loy['correlations']
            col1, col2 = st.columns(2)
            col1.metric("Points Earned vs GTO Revenue (correlation)", c.get('points_vs_gto_revenue', '—'))
            col2.metric("Member Spend vs GTO Revenue (correlation)",  c.get('spend_vs_gto_revenue',  '—'))
            st.caption("Correlation of 1.0 = perfect positive relationship, 0 = no relationship, -1 = inverse relationship")

        if loy.get('top_outlets_by_points'):
            st.markdown("**Top 10 Outlets by Points Earned**")
            df_pts = pd.DataFrame(loy['top_outlets_by_points'])
            name_col = 'final_gto_name' if 'final_gto_name' in df_pts.columns else df_pts.columns[0]
            if 'total_points' in df_pts.columns:
                st.bar_chart(df_pts.set_index(name_col)['total_points'])
            st.dataframe(df_pts, use_container_width=True, hide_index=True)

        if loy.get('regression'):
            reg = loy['regression']
            if 'error' not in reg:
                r2      = reg.get('r_squared', '—')
                insight = reg.get('insight', '')
                st.markdown(f"**Regression: what drives GTO revenue?** &nbsp; R² = `{r2}`")
                if insight:
                    st.caption(insight)
                coef_df = pd.DataFrame([
                    {
                        'variable':    k,
                        'coefficient': v,
                        'p_value':     reg.get('pvalues', {}).get(k, '—'),
                        'significant': '✅' if k in reg.get('significant', []) else '',
                    }
                    for k, v in reg.get('coefs', {}).items() if k != 'const'
                ])
                st.dataframe(coef_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Tenant turnover ───────────────────────────────────────────────
        st.markdown("### 🔄 Tenant Turnover")
        tt = ins.get('tenant_turnover', {})

        if tt.get('turnover_by_trade_type'):
            df_tt = pd.DataFrame(tt['turnover_by_trade_type'])
            trade_name_col = 'ion_trade_type_name' if 'ion_trade_type_name' in df_tt.columns else df_tt.columns[0]
            st.markdown("**Turnover Rate by Trade Type**")
            if 'turnover_rate_pct' in df_tt.columns:
                st.bar_chart(df_tt.set_index(trade_name_col)['turnover_rate_pct'])
            st.dataframe(df_tt, use_container_width=True, hide_index=True)

        if tt.get('gto_stayed_vs_exited'):
            sv = tt['gto_stayed_vs_exited']
            st.markdown("**GTO Performance: Stayed vs Exited Tenants**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Stayed**")
                st.metric("Count",          sv['stayed'].get('count','—'))
                stayed_a = sv['stayed'].get('avg_gto_amount')
                stayed_r = sv['stayed'].get('avg_gto_rent')
                st.metric("Avg GTO ($)",      f"${stayed_a:,.0f}" if stayed_a is not None else '—')
                st.metric("Avg GTO Rent ($)", f"${stayed_r:,.0f}" if stayed_r is not None else '—')
            with col2:
                st.markdown("**Exited**")
                st.metric("Count",          sv['exited'].get('count','—'))
                exited_a = sv['exited'].get('avg_gto_amount')
                exited_r = sv['exited'].get('avg_gto_rent')
                st.metric("Avg GTO ($)",      f"${exited_a:,.0f}" if exited_a is not None else '—')
                st.metric("Avg GTO Rent ($)", f"${exited_r:,.0f}" if exited_r is not None else '—')

        # Download report
        st.markdown("---")
        report_path = os.path.join(combined_folder, 'insights_report.xlsx')
        if Path(report_path).exists():
            with open(report_path, 'rb') as fh:
                st.download_button(
                    "⬇️  Download Full Insights Report (.xlsx)",
                    data=fh.read(),
                    file_name='insights_report.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    type='primary'
                )