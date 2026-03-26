import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import glob
import os
import re
from collections import defaultdict
from difflib import get_close_matches
import sys
from pathlib import Path


# -----------------------------
# Configuration Loader
# -----------------------------
def load_configuration(config_file="config_Kim.xlsx"):
    script_dir  = Path(__file__).resolve().parent
    config_file = str((script_dir / config_file).resolve())

    default_file_path   = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\raw data\raw data"
    default_output_path = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\cleaned data"
    default_schema_file = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\schemas.xlsx"

    default_header_rows = {
        ("gto", "monthly_sales"):   7,
        ("gto", "monthly_rent"):    8,
        ("gto", "tenant_turnover"): 7
    }

    if os.path.exists(config_file):
        try:
            print(f"📖 Loading configuration from {config_file}")
            file_path = output_path = schema_file = ""
            header_rows   = {}
            config_loaded = False

            try:
                paths_df    = pd.read_excel(config_file, sheet_name='paths')
                if 'Setting' in paths_df.columns and 'Value' in paths_df.columns:
                    config_dict = dict(zip(paths_df['Setting'].astype(str).str.strip(), paths_df['Value']))
                    file_path   = str(config_dict.get('raw_data',     '')).strip()
                    output_path = str(config_dict.get('cleaned_data', '')).strip()
                    schema_file = str(config_dict.get('schemas',      '')).strip()
                    print("DEBUG raw_data from config =",     repr(file_path))
                    print("DEBUG cleaned_data from config =", repr(output_path))
                    print("DEBUG schemas from config =",      repr(schema_file))
                    if all([file_path, output_path, schema_file]):
                        config_loaded = True
                        print("✅ Successfully loaded paths from config.xlsx")
                    else:
                        print("⚠️ Config file has empty paths, using defaults")
                        file_path, output_path, schema_file = default_file_path, default_output_path, default_schema_file
                else:
                    print("⚠️ Config missing required columns, using defaults")
                    file_path, output_path, schema_file = default_file_path, default_output_path, default_schema_file
            except Exception as e:
                print(f"⚠️ Error reading 'paths' sheet: {e}, using defaults")
                file_path, output_path, schema_file = default_file_path, default_output_path, default_schema_file

            try:
                headers_df = pd.read_excel(config_file, sheet_name='gto_headers')
                if all(col in headers_df.columns for col in ['category', 'dataset', 'header_row']):
                    header_rows = {}
                    for _, row in headers_df.iterrows():
                        key = (str(row['category']).strip().lower(), str(row['dataset']).strip().lower())
                        try:
                            header_rows[key] = int(row['header_row']) - 1
                        except ValueError:
                            print(f"⚠️ Invalid header_row for {key}: {row['header_row']}")
                    if header_rows:
                        print(f"✅ Loaded {len(header_rows)} GTO header configurations")
                    else:
                        header_rows = default_header_rows
                        print("⚠️ No valid GTO headers, using defaults")
                else:
                    print("⚠️ 'gto_headers' missing required columns, using defaults")
                    header_rows = default_header_rows
            except Exception as e:
                print(f"⚠️ Error reading 'gto_headers': {e}, using defaults")
                header_rows = default_header_rows

            return file_path, output_path, schema_file, header_rows, config_loaded

        except Exception as e:
            print(f"❌ Error processing config: {e}, using all defaults")
            return default_file_path, default_output_path, default_schema_file, default_header_rows, False
    else:
        print(f"ℹ️ Config '{config_file}' not found, using defaults")
        return default_file_path, default_output_path, default_schema_file, default_header_rows, False


# -----------------------------
# Helper Functions
# -----------------------------
def extract_year(filename: str):
    name  = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"20\d{2}", name)
    return int(match.group(0)) if match else None

def detect_dayfirst(series, sample_size=10):
    sample = series.dropna().astype(str).head(sample_size)
    day_first_count = 0
    for val in sample:
        parts = val.split('/')
        if len(parts) != 3:
            continue
        try:
            if int(parts[0]) > 12:
                day_first_count += 1
        except:
            continue
    return day_first_count >= 1

def add_month_year_columns(df, date_cols=None):
    df         = df.copy()
    df.columns = df.columns.map(str)
    if date_cols is None:
        date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        if col in df.columns:
            dayfirst    = detect_dayfirst(df[col])
            parsed      = pd.to_datetime(df[col], errors='coerce', dayfirst=dayfirst)
            df[col]     = parsed.apply(lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else "")
            df['month'] = parsed.dt.month_name().str[:3].fillna('')
            df['year']  = parsed.dt.year.fillna('')
    return df

def extract_voucher_value(voucher_code):
    if pd.isna(voucher_code) or not isinstance(voucher_code, str):
        return ""
    voucher_code = str(voucher_code).strip()
    if "2024-" in voucher_code:
        after = voucher_code.split("2024-", 1)[1]
        return after.split("_", 1)[0] if "_" in after else after
    elif "EYR-35000"  in voucher_code: return "2500"
    elif "EYR-75000"  in voucher_code: return "5625"
    elif "EYR-100000" in voucher_code: return "7500"
    elif "EYR-150000" in voucher_code: return "11250"
    return ""

def standardise_schema(df, schema_map):
    df          = df.copy()
    rename_dict = {c: schema_map[c] for c in df.columns if c in schema_map}
    df          = df.rename(columns=rename_dict)
    for col in schema_map.values():
        if col not in df.columns:
            df[col] = pd.NA
    return df

def classify_file(filename: str):
    name = filename.lower()
    if name.startswith('combined_'):           return ("skip", "skip")
    if "mall"  in name and "campaign" in name: return ("mall",  "campaign")
    if "mall"  in name and "member"   in name: return ("mall",  "member")
    if "brand" in name and "reward"   in name: return ("brand", "rewards")
    if "gto"   in name:
        if "sales"                           in name: return ("gto", "monthly_sales")
        if "rent"                            in name: return ("gto", "monthly_rent")
        if "turnover" in name or "occupancy" in name: return ("gto", "tenant_turnover")
    return ("unknown", "unclassified")


# -----------------------------
# Load schemas from Excel
# -----------------------------
def load_schemas_from_excel(schema_file):
    xl      = pd.ExcelFile(schema_file)
    schemas = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        if 'original_column' in df.columns and 'canonical_column' in df.columns:
            schemas[sheet] = dict(zip(df['original_column'], df['canonical_column']))
        else:
            print(f"⚠️ Sheet {sheet} missing required columns")
    return schemas


# -----------------------------
# Load Excel Files
# -----------------------------
def load_excel_files(folder_path, header_rows_config, default_sheet="Sheet1"):
    files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    data  = defaultdict(lambda: defaultdict(dict))

    for file in files:
        workbook_name = os.path.splitext(os.path.basename(file))[0]

        # Skip outlet-code-mapping
        if 'outlet' in workbook_name.lower() and 'code' in workbook_name.lower():
            print(f"⏭ Skipped (outlet-code-mapping): {workbook_name}")
            continue
        if 'outlet-code' in workbook_name.lower() or 'outletcode' in workbook_name.lower():
            print(f"⏭ Skipped (outlet-code-mapping): {workbook_name}")
            continue

        # ★ Skip combined outlet files
        if workbook_name.lower().startswith('combined_'):
            print(f"⏭ Skipped (combined outlet file): {workbook_name}")
            continue

        try:
            category, dataset = classify_file(workbook_name)
            if category == "skip":
                continue

            header_row = header_rows_config.get((category, dataset), 0)

            try:
                df = pd.read_excel(file, sheet_name=default_sheet, header=header_row)
            except:
                xl = pd.ExcelFile(file)
                df = xl.parse(xl.sheet_names[0], header=header_row)

            df   = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
            year = extract_year(file)
            df["year"] = year

            data[category][dataset][year] = df
            print(f"✓ Loaded: {workbook_name} → {category}/{dataset}/{year} (header row: {header_row})")

        except Exception as e:
            print(f"✗ Error loading {workbook_name}: {e}")

    return data


# -----------------------------
# Cleaning Configuration
# -----------------------------
DEFAULT_CLEANING_CONFIG = {
    'blank_numeric':     'zero',
    'blank_string':      'empty',
    'outlier_method':    'none',
    'outlier_action':    'cap',
    'outlier_threshold': 1.5,
    'apply_to':          'numeric',
}

def load_cleaning_config(config_file: str) -> dict:
    cfg = DEFAULT_CLEANING_CONFIG.copy()
    try:
        df = pd.read_excel(config_file, sheet_name='data_cleaning')
        if 'Setting' in df.columns and 'Value' in df.columns:
            for _, row in df.iterrows():
                key = str(row['Setting']).strip().lower().replace(' ', '_')
                val = str(row['Value']).strip().lower()
                if key in cfg:
                    if key == 'outlier_threshold':
                        try:    cfg[key] = float(val)
                        except: pass
                    else:
                        cfg[key] = val
    except Exception:
        pass
    return cfg

def apply_cleaning(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df       = df.copy()
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    str_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    blank_num = cfg.get('blank_numeric', 'zero')
    if   blank_num == 'zero':     df[num_cols] = df[num_cols].fillna(0)
    elif blank_num == 'mean':
        for c in num_cols: df[c] = df[c].fillna(df[c].mean())
    elif blank_num == 'median':
        for c in num_cols: df[c] = df[c].fillna(df[c].median())
    elif blank_num == 'drop_row': df = df.dropna(subset=num_cols)

    blank_str = cfg.get('blank_string', 'empty')
    if blank_str == 'drop_row':
        df = df.dropna(subset=str_cols)
    else:
        for c in str_cols:
            df[c] = df[c].fillna('').astype(str).str.strip()

    method    = cfg.get('outlier_method',    'none')
    action    = cfg.get('outlier_action',    'cap')
    threshold = float(cfg.get('outlier_threshold', 1.5))

    if method == 'none' or not num_cols:
        return df

    outlier_mask = pd.DataFrame(False, index=df.index, columns=num_cols)

    if method == 'iqr':
        for c in num_cols:
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr    = q3 - q1
            lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
            outlier_mask[c] = (df[c] < lo) | (df[c] > hi)
            if action == 'cap': df[c] = df[c].clip(lower=lo, upper=hi)
    elif method == 'zscore':
        for c in num_cols:
            z      = (df[c] - df[c].mean()) / df[c].std(ddof=0).replace(0, 1)
            lo_val = df[c].mean() - threshold * df[c].std(ddof=0)
            hi_val = df[c].mean() + threshold * df[c].std(ddof=0)
            outlier_mask[c] = z.abs() > threshold
            if action == 'cap': df[c] = df[c].clip(lower=lo_val, upper=hi_val)
    elif method == 'winsorise':
        for c in num_cols:
            lo = df[c].quantile(threshold / 100)
            hi = df[c].quantile(1 - threshold / 100)
            outlier_mask[c] = (df[c] < lo) | (df[c] > hi)
            df[c] = df[c].clip(lower=lo, upper=hi)

    if action == 'drop_row':
        df = df[~outlier_mask.any(axis=1)]

    return df

def merge_dataset_dynamic(dataset_dict, schema_map, dataset_base_name="Dataset",
                          add_month_year=True, extra_columns=None, cleaning_config=None):
    dfs = []
    for year in sorted(dataset_dict.keys()):
        df = dataset_dict[year].copy()
        df = standardise_schema(df, schema_map)
        if extra_columns:
            for col, value in extra_columns.items():
                df[col] = value
        if add_month_year:
            df = add_month_year_columns(df)
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = apply_cleaning(merged_df, cleaning_config or DEFAULT_CLEANING_CONFIG)
    return dataset_base_name, merged_df


# -----------------------------
# Export to Excel
# -----------------------------
def export_to_excel(merged_data, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file in glob.glob(os.path.join(output_folder, "*.xlsx")) + \
                glob.glob(os.path.join(output_folder, "*.csv")):
        try:
            os.remove(file)
            print(f"🗑 Deleted old file: {file}")
        except Exception as e:
            print(f"⚠️ Could not delete {file}: {e}")
    for name, df in merged_data.items():
        xlsx_path = os.path.join(output_folder, f"{name}.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"✅ Exported Excel {name} → {xlsx_path}")
        csv_path = os.path.join(output_folder, f"{name}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ Exported CSV {name} → {csv_path}")


# -----------------------------
# Outlet-Code-Mapping Loader
# -----------------------------
def load_outlet_code_mapping(raw_data_folder: str) -> pd.DataFrame:
    candidates = []
    for f in os.listdir(raw_data_folder):
        fl = f.lower()
        if f.endswith('.xlsx') and (
            ('outlet' in fl and 'code' in fl) or
            ('outlet' in fl and 'mapping' in fl)
        ):
            candidates.append(os.path.join(raw_data_folder, f))

    if not candidates:
        print("ℹ️  No outlet-code-mapping file found — skipping bridge.")
        return pd.DataFrame()

    path = candidates[0]
    print(f"📂 Loading outlet-code-mapping: {os.path.basename(path)}")

    try:
        raw = pd.read_excel(path, sheet_name='Sheet1', header=None)
        header_row = None
        for i, row in raw.iterrows():
            if any(str(v).strip() in ('Shop Name', 'Brand') for v in row.values):
                header_row = i
                break

        if header_row is None:
            print("   ⚠️ Could not find header row — skipping bridge.")
            return pd.DataFrame()

        df = pd.read_excel(path, sheet_name='Sheet1', header=header_row)
        df.columns = df.columns.str.strip()

        if 'Shop Name' in df.columns:
            df['shop_name_norm'] = df['Shop Name'].astype(str).str.strip().str.lower()
        if 'Brand' in df.columns:
            df['brand_norm'] = df['Brand'].astype(str).str.strip().str.lower()

        print(f"   ✅ Loaded {len(df)} rows (header at row {header_row}), columns: {list(df.columns)}")
        return df

    except Exception as e:
        print(f"   ⚠️ Could not load outlet-code-mapping: {e}")
        return pd.DataFrame()


# -----------------------------
# ★ NEW — Combined Outlet Files Loader
# -----------------------------
def load_combined_outlet_names(raw_data_folder: str) -> set:
    """
    Load outlet names from:
      combined_Mall_Trans     → 'Outlet Name'
      combined_Mall_Campaign  → 'Redeem Outlet Name'
      combined_Brand_Campaign → 'Redeem Outlet Name'

    Returns a set of normalised (lowercase, stripped) outlet names.
    """
    file_col_map = {
        'combined_mall_trans':     'Outlet Name',
        'combined_mall_campaign':  'Redeem Outlet Name',
        'combined_brand_campaign': 'Redeem Outlet Name',
    }

    combined_names = set()

    for f in os.listdir(raw_data_folder):
        if not f.endswith('.xlsx'):
            continue

        # Normalise filename for matching
        fl_stem = f.lower().replace(' ', '_').replace('-', '_').replace('.xlsx', '')

        matched_col = None
        for key, col in file_col_map.items():
            if key in fl_stem:
                matched_col = col
                break

        if matched_col is None:
            continue

        path = os.path.join(raw_data_folder, f)
        try:
            df = pd.read_excel(path)
            df.columns = df.columns.str.strip()

            # Case-insensitive column search
            if matched_col not in df.columns:
                col_match = [c for c in df.columns
                             if c.strip().lower() == matched_col.lower()]
                if col_match:
                    matched_col = col_match[0]
                else:
                    print(f"   ⚠️ '{matched_col}' not found in {f} — columns: {list(df.columns)}")
                    continue

            names = (df[matched_col].dropna()
                                    .astype(str)
                                    .str.strip()
                                    .str.lower())
            names = names[names != ''].unique()
            combined_names.update(names)
            print(f"   ✅ {f} → '{matched_col}': {len(names)} unique outlet names")

        except Exception as e:
            print(f"   ⚠️ Could not load {f}: {e}")

    print(f"   🔗 Total unique outlet names from combined files: {len(combined_names)}")
    return combined_names


# -----------------------------
# Shop Name Resolution
# -----------------------------
def resolve_shop_name(name: str, gto_names: list, fuzzy_threshold: float = 0.6) -> tuple:
    name_lower = str(name).strip().lower()
    if name_lower in gto_names:
        return name_lower, 'exact'
    matches = get_close_matches(name_lower, gto_names, n=1, cutoff=fuzzy_threshold)
    if matches:
        return matches[0], 'fuzzy'
    return name_lower, 'unmatched'


def resolve_shop_names(campaign_names: pd.Series,
                       gto_names: pd.Series,
                       mapping_file: str,
                       outlet_code_df: pd.DataFrame = None,
                       combined_outlet_names: set = None) -> pd.DataFrame:
    """
    Resolve campaign outlet names → GTO shop names.

    Priority:
      1. confirmed       — user manually confirmed
      2. code_match      — outlet-code-mapping Brand → Shop Name
      3. combined_exact  — ★ name in combined files, exact GTO match
      4. combined_fuzzy  — ★ name in combined files, fuzzy GTO match
      5. exact           — direct string match to GTO
      6. fuzzy           — difflib similarity to GTO
      7. unmatched
    """
    gto_names_lower = gto_names.str.strip().str.lower().dropna().unique().tolist()
    unique_campaign = campaign_names.str.strip().str.lower().dropna().unique()

    # ── Brand→GTO bridge (outlet-code-mapping) ────────────────────────────
    brand_to_shop = {}
    if outlet_code_df is not None and not outlet_code_df.empty:
        if 'brand_norm' in outlet_code_df.columns and 'shop_name_norm' in outlet_code_df.columns:
            for _, row in outlet_code_df.iterrows():
                b = str(row['brand_norm']).strip().lower()
                s = str(row['shop_name_norm']).strip().lower()
                if b and s and b != 'nan' and s != 'nan':
                    gto_match = get_close_matches(s, gto_names_lower, n=1, cutoff=0.85)
                    if gto_match:
                        brand_to_shop[b] = gto_match[0]
                    elif s in gto_names_lower:
                        brand_to_shop[b] = s
            print(f"   🔗 Brand→GTO bridge: {len(brand_to_shop)} entries")

    # ── ★ Combined files → GTO bridge ─────────────────────────────────────
    combined_to_gto = {}
    if combined_outlet_names:
        for cname in combined_outlet_names:
            if cname in gto_names_lower:
                combined_to_gto[cname] = (cname, 'exact')
            else:
                matches = get_close_matches(cname, gto_names_lower, n=1, cutoff=0.7)
                if matches:
                    combined_to_gto[cname] = (matches[0], 'fuzzy')
        exact_c = sum(1 for v in combined_to_gto.values() if v[1] == 'exact')
        fuzzy_c = sum(1 for v in combined_to_gto.values() if v[1] == 'fuzzy')
        print(f"   🔗 Combined→GTO bridge: {len(combined_to_gto)} entries "
              f"({exact_c} exact, {fuzzy_c} fuzzy)")

    # ── Load existing confirmed mappings ───────────────────────────────────
    existing = {}
    if os.path.exists(mapping_file):
        try:
            df_ex = pd.read_excel(mapping_file, sheet_name='mapping')
            for _, row in df_ex.iterrows():
                c         = str(row.get('campaign_name',      '')).strip().lower()
                confirmed = str(row.get('confirmed_gto_name', '')).strip().lower()
                suggested = str(row.get('suggested_gto_name', '')).strip().lower()
                if c:
                    existing[c] = {
                        'confirmed': confirmed if confirmed and confirmed != 'nan' else '',
                        'suggested': suggested,
                        'method':    str(row.get('method', '')).strip()
                    }
            print(f"✅ Loaded shop_mapping.xlsx ({len(existing)} entries)")
        except Exception as e:
            print(f"⚠️ Could not read shop_mapping.xlsx: {e} — rebuilding.")

    # ── Resolve ────────────────────────────────────────────────────────────
    rows = []
    for name in unique_campaign:

        base = {
            'campaign_name':      name,
            'gto_name':           '',
            'suggested_gto_name': '',
            'confirmed_gto_name': existing.get(name, {}).get('confirmed', ''),
            'method':             '',
            'bridge_brand':       '',
            'bridge_shop_name':   '',
            'customer_group':     '',
            'lease_no':           '',
            'combined_source':    '',
        }

        # 1. Confirmed
        if name in existing and existing[name]['confirmed']:
            rows.append({**base,
                'gto_name':           existing[name]['confirmed'],
                'suggested_gto_name': existing[name]['suggested'],
                'confirmed_gto_name': existing[name]['confirmed'],
                'method':             'confirmed',
            })
            continue

        # 2. Code-bridge
        if name in brand_to_shop:
            resolved = brand_to_shop[name]
            br = pd.Series(dtype=str)
            if outlet_code_df is not None and not outlet_code_df.empty \
                    and 'brand_norm' in outlet_code_df.columns:
                mr = outlet_code_df[outlet_code_df['brand_norm'] == name]
                if not mr.empty:
                    br = mr.iloc[0]
            rows.append({**base,
                'gto_name':           resolved,
                'suggested_gto_name': resolved,
                'method':             'code_match',
                'bridge_brand':       str(br.get('Brand',          '')),
                'bridge_shop_name':   str(br.get('Shop Name',      '')),
                'customer_group':     str(br.get('Customer Group', '')),
                'lease_no':           str(br.get('Lease No.',      '')),
            })
            continue

        # 3 & 4. ★ Combined-file match
        if name in combined_to_gto:
            resolved, sub = combined_to_gto[name]
            rows.append({**base,
                'gto_name':           resolved,
                'suggested_gto_name': resolved,
                'method':             f'combined_{sub}',
                'combined_source':    'combined_files',
            })
            continue

        # 5 & 6. Exact / fuzzy fallback
        resolved, method = resolve_shop_name(name, gto_names_lower)

        br = pd.Series(dtype=str)
        if outlet_code_df is not None and not outlet_code_df.empty \
                and 'brand_norm' in outlet_code_df.columns:
            close = get_close_matches(name, outlet_code_df['brand_norm'].tolist(),
                                      n=1, cutoff=0.8)
            if close:
                mr = outlet_code_df[outlet_code_df['brand_norm'] == close[0]]
                if not mr.empty:
                    br = mr.iloc[0]

        rows.append({**base,
            'gto_name':           resolved if method != 'unmatched' else '',
            'suggested_gto_name': resolved,
            'method':             method,
            'bridge_brand':       str(br.get('Brand',          '')),
            'bridge_shop_name':   str(br.get('Shop Name',      '')),
            'customer_group':     str(br.get('Customer Group', '')),
            'lease_no':           str(br.get('Lease No.',      '')),
        })

    # ── Sort: matched first ────────────────────────────────────────────────
    matched   = [r for r in rows if r['method'] != 'unmatched']
    unmatched = [r for r in rows if r['method'] == 'unmatched']

    blank_rows = [{**r,
        'gto_name': '', 'suggested_gto_name': '',
        'confirmed_gto_name': '', 'method': 'unmatched',
        'bridge_brand': '', 'bridge_shop_name': '',
        'customer_group': '', 'lease_no': '', 'combined_source': '',
    } for r in unmatched]

    resolution_df = pd.DataFrame(matched + blank_rows)

    # ── GTO-only rows ──────────────────────────────────────────────────────
    resolved_gto = set(resolution_df['suggested_gto_name'].astype(str).str.strip().str.lower())
    resolved_gto |= (set(resolution_df['confirmed_gto_name'].astype(str).str.strip().str.lower())
                     - {'', 'nan'})
    gto_only = sorted([n for n in gto_names_lower if n not in resolved_gto])

    gto_only_rows = pd.DataFrame([{
        'campaign_name': '', 'gto_name': n, 'suggested_gto_name': '',
        'confirmed_gto_name': '', 'method': 'gto_only',
        'bridge_brand': '', 'bridge_shop_name': '',
        'customer_group': '', 'lease_no': '', 'combined_source': '',
    } for n in gto_only])

    full_mapping_df = pd.concat([resolution_df, gto_only_rows], ignore_index=True)

    # ── Save ───────────────────────────────────────────────────────────────
    instructions = pd.DataFrame([
        {"Instructions": "campaign_name: outlet name from campaign data — DO NOT edit."},
        {"Instructions": "gto_name: GTO name used for matching (confirmed > code_match > combined > suggested)."},
        {"Instructions": "suggested_gto_name: best automatic match — DO NOT edit."},
        {"Instructions": "confirmed_gto_name: fill this to override any automatic match."},
        {"Instructions": "method: confirmed > code_match > combined_exact > combined_fuzzy > exact > fuzzy > unmatched > gto_only."},
        {"Instructions": "bridge_brand / bridge_shop_name: from outlet-code-mapping file."},
        {"Instructions": "customer_group / lease_no: from outlet-code-mapping file."},
        {"Instructions": "combined_source: match came from combined_Mall_Trans / Campaign / Brand files."},
        {"Instructions": "Re-run the pipeline after editing confirmed_gto_name."},
    ])
    with pd.ExcelWriter(mapping_file, engine='openpyxl') as writer:
        full_mapping_df.drop(columns=['final_gto_name'], errors='ignore').to_excel(
            writer, sheet_name='mapping', index=False)
        instructions.to_excel(writer, sheet_name='instructions', index=False)

    print(f"📝 shop_mapping.xlsx updated.")
    if gto_only:
        print(f"   ℹ️  {len(gto_only)} GTO shops have no campaign match (gto_only).")

    counts = pd.Series([r['method'] for r in rows]).value_counts()
    print(f"\n📊 Shop name resolution summary:")
    for method, count in counts.items():
        print(f"   {method}: {count}")

    if unmatched:
        print(f"\n⚠️  {len(unmatched)} unmatched:")
        for r in unmatched:
            print(f"   - {r['campaign_name']}")

    resolution_df['final_gto_name'] = resolution_df['confirmed_gto_name'].where(
        resolution_df['confirmed_gto_name'].astype(str).str.strip().str.len() > 0,
        resolution_df['suggested_gto_name']
    )

    return resolution_df


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) >= 5:
        file_path    = sys.argv[1]
        output_path  = sys.argv[2]
        schema_file  = sys.argv[3]
        mapping_file = sys.argv[4]
        config_file  = sys.argv[5] if len(sys.argv) > 5 else "config_Kim.xlsx"
        _, _, _, header_rows_config, _ = load_configuration(config_file)
    else:
        config_file  = sys.argv[1] if len(sys.argv) == 2 else "config_Kim.xlsx"
        file_path, output_path, schema_file, header_rows_config, _ = load_configuration(config_file)
        _paths_df    = pd.read_excel(Path(__file__).resolve().parent / config_file, sheet_name='paths')
        _config      = dict(zip(_paths_df['Setting'].astype(str).str.strip(), _paths_df['Value']))
        mapping_file = str(_config.get('shop_mapping', '')).strip()

    print("\n" + "="*60)
    print("ETL PROCESS STARTING")
    print("="*60)
    print(f"📁 Raw data path: {file_path}")
    print(f"📁 Output path:   {output_path}")
    print(f"📁 Schema file:   {schema_file}")
    print(f"📋 GTO header configurations: {len(header_rows_config)}")
    for (cat, ds), hr in header_rows_config.items():
        print(f"   - {cat}/{ds}: header row {hr}")
    print("="*60 + "\n")

    if not os.path.exists(file_path):
        print(f"❌ ERROR: Raw data folder not found: {file_path}")
        exit(1)
    if not os.path.exists(schema_file):
        print(f"❌ ERROR: Schema file not found: {schema_file}")
        exit(1)

    SCHEMAS  = load_schemas_from_excel(schema_file)
    print(f"✅ Loaded {len(SCHEMAS)} schema definitions")

    CLEANING = load_cleaning_config(
        config_file if len(sys.argv) >= 5
        else str(Path(__file__).resolve().parent / config_file)
    )
    print(f"🧹 Cleaning — blanks(numeric): {CLEANING['blank_numeric']}  "
          f"blanks(text): {CLEANING['blank_string']}  "
          f"outliers: {CLEANING['outlier_method']} / {CLEANING['outlier_action']} "
          f"(threshold: {CLEANING['outlier_threshold']})")

    # ── Load bridge files ──────────────────────────────────────────────────
    outlet_code_df = load_outlet_code_mapping(file_path)

    print("\n📂 Loading combined outlet files...")
    combined_outlet_names = load_combined_outlet_names(file_path)  # ★ NEW

    data = load_excel_files(file_path, header_rows_config)

    merged_data  = {}
    campaign_dfs = []

    for category in data:
        for dataset in data[category]:
            dataset_name = f"{category}_{dataset}"
            schema       = SCHEMAS.get(dataset_name, {})

            if (category, dataset) in [("mall", "campaign"), ("brand", "rewards")]:
                _, df = merge_dataset_dynamic(
                    data[category][dataset],
                    schema_map=schema,
                    dataset_base_name="campaigns",
                    extra_columns={
                        "campaign_source": dataset_name,
                        "campaign_type":   category
                    },
                    cleaning_config=CLEANING,
                )
                campaign_dfs.append(df)
            else:
                df_name, df = merge_dataset_dynamic(
                    data[category][dataset],
                    schema_map=schema,
                    dataset_base_name=dataset_name,
                    cleaning_config=CLEANING,
                )
                merged_data[df_name] = df

    if campaign_dfs:
        merged_data["campaign_all"] = pd.concat(campaign_dfs, ignore_index=True)

    export_to_excel(merged_data, output_path)

    # ── Resolve shop names ─────────────────────────────────────────────────
    if 'campaign_all' in merged_data:
        print("\n" + "="*60)
        print("RESOLVING SHOP NAMES")
        print("="*60)

        gto_shop_names = pd.Series(dtype=str)
        for key, df in merged_data.items():
            if 'gto' in key and 'shop_name' in df.columns:
                gto_shop_names = df['shop_name']
                break

        if gto_shop_names.empty:
            print("⚠️ No GTO data found — skipping shop name resolution.")
        else:
            campaign_df = merged_data['campaign_all'].copy()

            resolution = resolve_shop_names(
                campaign_df['outlet_name'],
                gto_shop_names,
                mapping_file,
                outlet_code_df=outlet_code_df,
                combined_outlet_names=combined_outlet_names,  # ★ NEW
            )

            campaign_df['outlet_name_lower'] = campaign_df['outlet_name'].str.strip().str.lower()
            campaign_df = campaign_df.merge(
                resolution[['campaign_name', 'final_gto_name']],
                left_on='outlet_name_lower',
                right_on='campaign_name',
                how='left'
            ).drop(columns=['outlet_name_lower', 'campaign_name'], errors='ignore')

            extra_cols = [c for c in ['customer_group', 'lease_no']
                          if c in resolution.columns]
            if extra_cols:
                campaign_df = campaign_df.merge(
                    resolution[['campaign_name'] + extra_cols].rename(
                        columns={'campaign_name': '_merge_key'}
                    ),
                    left_on='outlet_name',
                    right_on='_merge_key',
                    how='left'
                ).drop(columns=['_merge_key'], errors='ignore')

            merged_data['campaign_all'] = campaign_df

            camp_xlsx = os.path.join(output_path, 'campaign_all.xlsx')
            camp_csv  = os.path.join(output_path, 'campaign_all.csv')
            campaign_df.to_excel(camp_xlsx, index=False)
            campaign_df.to_csv(camp_csv, index=False, encoding='utf-8-sig')
            print(f"✅ Re-exported campaign_all with resolved shop names")

    print("\n" + "="*60)
    print(f"✅ ETL PROCESS COMPLETED SUCCESSFULLY")
    print(f"📊 Created {len(merged_data)} cleaned datasets")
    print(f"💾 Output saved to: {output_path}")
    print("="*60)