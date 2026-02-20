import pandas as pd
import glob
import os
import re
from collections import defaultdict
from pathlib import Path


# -----------------------------
# Configuration Loader
# -----------------------------
def load_configuration(config_file="config.xlsx"):
    """
    Load configuration from config.xlsx
    Returns: (file_path, output_path, schema_file, header_rows, using_config)
    """
    script_dir = Path(__file__).resolve().parent
    config_file = str((script_dir / config_file).resolve())
    # Default paths (fallback if config doesn't exist)
    default_file_path = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\raw data\raw data"
    default_output_path = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\cleaned data"
    default_schema_file = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\schemas.xlsx"
    
    # Default GTO header rows (fallback)
    default_header_rows = {
        ("gto", "monthly_sales"): 7,
        ("gto", "monthly_rent"): 8,
        ("gto", "tenant_turnover"): 7
    }
    
    # Try to load from config file
    if os.path.exists(config_file):
        try:
            print(f"📖 Loading configuration from {config_file}")
            
            # Initialize
            file_path = ""
            output_path = ""
            schema_file = ""
            header_rows = {}
            config_loaded = False
            
            # Try to load paths sheet
            try:
                paths_df = pd.read_excel(config_file, sheet_name='paths')
                if 'Setting' in paths_df.columns and 'Value' in paths_df.columns:
                    config_dict = dict(zip(paths_df['Setting'].astype(str).str.strip(),
                       paths_df['Value']))
                    
                    file_path = str(config_dict.get('raw_data', '')).strip()
                    output_path = str(config_dict.get('cleaned_data', '')).strip()
                    schema_file = str(config_dict.get('schemas', '')).strip()
                    
                    print("DEBUG raw_data from config =", repr(file_path))
                    print("DEBUG cleaned_data from config =", repr(output_path))
                    print("DEBUG schemas from config =", repr(schema_file))


                    if all([file_path, output_path, schema_file]):
                        config_loaded = True
                        print("✅ Successfully loaded paths from config.xlsx")
                    else:
                        print("⚠️ Config file exists but has empty paths, using defaults")
                        file_path = default_file_path
                        output_path = default_output_path
                        schema_file = default_schema_file
                else:
                    print("⚠️ Config file missing required columns, using defaults")
                    file_path = default_file_path
                    output_path = default_output_path
                    schema_file = default_schema_file
            except Exception as e:
                print(f"⚠️ Error reading 'paths' sheet: {e}, using defaults")
                file_path = default_file_path
                output_path = default_output_path
                schema_file = default_schema_file
            
            # Try to load GTO headers sheet
            try:
                headers_df = pd.read_excel(config_file, sheet_name='gto_headers')
                if all(col in headers_df.columns for col in ['category', 'dataset', 'header_row']):
                    header_rows = {}
                    for _, row in headers_df.iterrows():
                        key = (str(row['category']).strip().lower(), 
                               str(row['dataset']).strip().lower())
                        try:
                            header_rows[key] = int(row['header_row'])-1
                        except ValueError:
                            print(f"⚠️ Invalid header_row value for {key}: {row['header_row']}")
                    
                    if header_rows:
                        print(f"✅ Loaded {len(header_rows)} GTO header configurations")
                    else:
                        header_rows = default_header_rows
                        print("⚠️ No valid GTO headers found, using defaults")
                else:
                    print("⚠️ 'gto_headers' sheet missing required columns, using defaults")
                    header_rows = default_header_rows
            except Exception as e:
                print(f"⚠️ Error reading 'gto_headers' sheet: {e}, using defaults")
                header_rows = default_header_rows
            
            return file_path, output_path, schema_file, header_rows, config_loaded
            
        except Exception as e:
            print(f"❌ Error processing config file: {e}, using all defaults")
            return default_file_path, default_output_path, default_schema_file, default_header_rows, False
    else:
        print(f"ℹ️ Config file '{config_file}' not found, using all defaults")
        return default_file_path, default_output_path, default_schema_file, default_header_rows, False

# -----------------------------
# Helper Functions (unchanged)
# -----------------------------
def extract_year(filename: str):
    """Extract first 4-digit year starting with 20 from the filename"""
    name = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"20\d{2}", name)
    return int(match.group(0)) if match else None

def detect_dayfirst(series, sample_size=10):
    """Detect if a date series is day-first"""
    sample = series.dropna().astype(str).head(sample_size)
    day_first_count = 0
    for val in sample:
        parts = val.split('/')
        if len(parts) != 3:
            continue
        try:
            day = int(parts[0])
            if day > 12:
                day_first_count += 1
        except:
            continue
    return day_first_count >= 1

def add_month_year_columns(df, date_cols=None):
    """Convert date columns to proper date strings and add 'month' and 'year' columns"""
    df = df.copy()
    df.columns = df.columns.map(str)
    
    if date_cols is None:
        date_cols = [col for col in df.columns if 'date' in col.lower()]
    
    for col in date_cols:
        if col in df.columns:
            # Detect dayfirst format for parsing
            dayfirst = detect_dayfirst(df[col])
            
            # Parse to datetime first for month/year extraction
            parsed_dates = pd.to_datetime(df[col], errors='coerce', dayfirst=dayfirst)
            
            # Convert to string format 'dd/mm/yyyy'
            df[col] = parsed_dates.apply(lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else "")
            
            # Add month and year columns
            df['month'] = parsed_dates.dt.month_name().str[:3]
            df['year'] = parsed_dates.dt.year
            
            # Fill NaN values in month and year with appropriate defaults
            df['month'] = df['month'].fillna('')
            df['year'] = df['year'].fillna('')
    
    return df

def extract_voucher_value(voucher_code):
    """
    Extract voucher value from voucher code based on the Power Query rules.
    
    Rules:
    1. If contains "2024-", extract text after "2024-" and before "_"
    2. EYR specific mappings:
       - "EYR-35000" → "2500"
       - "EYR-75000" → "5625"
       - "EYR-100000" → "7500"
       - "EYR-150000" → "11250"
    3. Otherwise return empty string
    """
    if pd.isna(voucher_code) or not isinstance(voucher_code, str):
        return ""
    
    voucher_code = str(voucher_code).strip()
    
    # Rule 1: Contains "2024-"
    if "2024-" in voucher_code:
        # Extract text after "2024-" and before "_"
        after_2024 = voucher_code.split("2024-", 1)[1]
        if "_" in after_2024:
            return after_2024.split("_", 1)[0]
        else:
            return after_2024
    
    # Rule 2: EYR specific mappings
    elif "EYR-35000" in voucher_code:
        return "2500"
    elif "EYR-75000" in voucher_code:
        return "5625"
    elif "EYR-100000" in voucher_code:
        return "7500"
    elif "EYR-150000" in voucher_code:
        return "11250"
    
    # Rule 3: No match
    return ""

def standardise_schema(df, schema_map):
    """Rename columns according to schema and ensure all canonical columns exist"""
    df = df.copy()
    rename_dict = {c: schema_map[c] for c in df.columns if c in schema_map}
    df = df.rename(columns=rename_dict)
    for col in schema_map.values():
        if col not in df.columns:
            df[col] = pd.NA
    return df

def classify_file(filename: str):
    """Classify file type based on filename"""
    name = filename.lower()
    if "mall" in name and "campaign" in name: return ("mall", "campaign")
    if "mall" in name and "member" in name: return ("mall", "member")
    if "brand" in name and "reward" in name: return ("brand", "rewards")
    if "gto" in name:
        if "sales" in name: return ("gto", "monthly_sales")
        if "rent" in name: return ("gto", "monthly_rent")
        if "turnover" in name or "occupancy" in name: return ("gto", "tenant_turnover")
    return ("unknown", "unclassified")

# -----------------------------
# Load schemas from Excel
# -----------------------------
def load_schemas_from_excel(schema_file):
    """Load schemas from an Excel file with one sheet per dataset"""
    xl = pd.ExcelFile(schema_file)
    schemas = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        if 'original_column' in df.columns and 'canonical_column' in df.columns:
            mapping = dict(zip(df['original_column'], df['canonical_column']))
            schemas[sheet] = mapping
        else:
            print(f"⚠️ Sheet {sheet} missing required columns 'original_column' and 'canonical_column'")
    return schemas

# -----------------------------
# Load Excel Files with configurable header rows
# -----------------------------
def load_excel_files(folder_path, header_rows_config, default_sheet="Sheet1"):
    files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    data = defaultdict(lambda: defaultdict(dict))

    for file in files:
        workbook_name = os.path.splitext(os.path.basename(file))[0]
        try:
            category, dataset = classify_file(workbook_name)
            header_row = header_rows_config.get((category, dataset), 0)

            try:
                df = pd.read_excel(file, sheet_name=default_sheet, header=header_row)
            except:
                xl = pd.ExcelFile(file)
                df = xl.parse(xl.sheet_names[0], header=header_row)

            df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)

            year = extract_year(file)
            df["year"] = year

            data[category][dataset][year] = df
            print(f"✓ Loaded: {workbook_name} → {category}/{dataset}/{year} (header row: {header_row})")

        except Exception as e:
            print(f"✗ Error loading {workbook_name}: {e}")

    return data

# -----------------------------
# Merge Datasets (optimized for redemptions)
# -----------------------------
def merge_dataset_dynamic(dataset_dict, schema_map, dataset_base_name="Dataset", add_month_year=True, extra_columns=None):
    dfs = []
    years = sorted(dataset_dict.keys())

    for year in years:
        df = dataset_dict[year].copy()
        df = standardise_schema(df, schema_map)

        if extra_columns:
            for col, value in extra_columns.items():
                df[col] = value

        if add_month_year:
            df = add_month_year_columns(df)

        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    df_name = f"{dataset_base_name}_{years[0]}_to_{years[-1]}"
    return df_name, merged_df

# -----------------------------
# Export to Excel
# -----------------------------
def export_to_excel(merged_data, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Delete existing Excel files in the folder
    for file in glob.glob(os.path.join(output_folder, "*.xlsx")) + glob.glob(os.path.join(output_folder, "*.csv")):
        try:
            os.remove(file)
            print(f"🗑 Deleted old file: {file}")
        except Exception as e:
            print(f"⚠️ Could not delete {file}: {e}")

    # Export new Excel files
    for name, df in merged_data.items():
        # Excel
        xlsx_path = os.path.join(output_folder, f"{name}.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"✅ Exported Excel {name} → {xlsx_path}")

        # CSV
        csv_path = os.path.join(output_folder, f"{name}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ Exported CSV {name} → {csv_path}")
        


# -----------------------------
# Main Execution with Full Config Support
# -----------------------------
if __name__ == "__main__":
    # Load configuration
    file_path, output_path, schema_file, header_rows_config, using_config = load_configuration()
    
    print("\n" + "="*60)
    print("ETL PROCESS STARTING")
    print("="*60)
    print(f"📁 Raw data path: {file_path}")
    print(f"📁 Output path: {output_path}")
    print(f"📁 Schema file: {schema_file}")
    print(f"⚙️ Using config file: {using_config}")
    print(f"📋 GTO header configurations: {len(header_rows_config)}")
    for (cat, ds), hr in header_rows_config.items():
        print(f"   - {cat}/{ds}: header row {hr}")
    print("="*60 + "\n")
    
    # Verify paths exist
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Raw data folder not found at: {file_path}")
        if not using_config:
            print("\n💡 CONFIGURATION HELP:")
            print("1. Create a 'config.xlsx' file with two sheets:")
            print("   - Sheet 'paths' with columns: Setting, Value")
            print("   - Sheet 'gto_headers' with columns: category, dataset, header_row")
            print("2. Required 'paths' settings: raw_data, cleaned_data, schemas")
            print("3. Example 'gto_headers' rows:")
            print("   - gto, monthly_sales, 7")
            print("   - gto, monthly_rent, 8")
            print("   - gto, tenant_turnover, 7")
        exit(1)
        
    if not os.path.exists(schema_file):
        print(f"❌ ERROR: Schema file not found at: {schema_file}")
        exit(1)
    
    # Load schemas from Excel
    SCHEMAS = load_schemas_from_excel(schema_file)
    print(f"✅ Loaded {len(SCHEMAS)} schema definitions")
    
    # Load all Excel datasets using configurable header rows
    data = load_excel_files(file_path, header_rows_config)
    
    merged_data = {}
    campaign_dfs = []

    for category in data:
        for dataset in data[category]:
            dataset_name = f"{category}_{dataset}"  # match your Excel sheet names
            schema = SCHEMAS.get(dataset_name, {})

            if (category, dataset) in [("mall", "campaign"), ("brand", "rewards")]:
                _, df = merge_dataset_dynamic(
                    data[category][dataset],
                    schema_map=schema,
                    dataset_base_name="campaigns",
                    extra_columns={
                        "campaign_source": dataset_name,
                        "campaign_type": category
                    }
                )
                campaign_dfs.append(df)
            else:
                df_name, df = merge_dataset_dynamic(
                    data[category][dataset],
                    schema_map=schema,
                    dataset_base_name=dataset_name
                )
                merged_data[df_name] = df

    # Final combined redemptions fact table
    if campaign_dfs:
        merged_data["campaign_all"] = pd.concat(campaign_dfs, ignore_index=True)

    # Export all datasets
    export_to_excel(merged_data, output_path)
    
    print("\n" + "="*60)
    print(f"✅ ETL PROCESS COMPLETED SUCCESSFULLY")
    print(f"📊 Created {len(merged_data)} cleaned datasets")
    print(f"💾 Output saved to: {output_path}")
    print("="*60)