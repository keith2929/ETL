import pandas as pd
import glob
import os
import re
from collections import defaultdict

# -----------------------------
# Helper Functions
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
    """Convert date columns to datetime and add 'month' and 'year' columns"""
    df = df.copy()
    df.columns = df.columns.map(str)
    if date_cols is None:
        date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        if col in df.columns:
            dayfirst = detect_dayfirst(df[col])
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=dayfirst)
            df['month'] = df[col].dt.month_name().str[:3]
            df['year'] = df[col].dt.year
    return df

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
# User-Defined Header Rows for GTO Files
# -----------------------------
HEADER_ROWS = {
    ("gto","monthly_sales"): 7,
    ("gto","monthly_rent"): 8,
    ("gto","tenant_turnover"): 7
}

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
# Load Excel Files
# -----------------------------
def load_excel_files(folder_path, default_sheet="Sheet1"):
    files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    data = defaultdict(lambda: defaultdict(dict))

    for file in files:
        workbook_name = os.path.splitext(os.path.basename(file))[0]
        try:
            category, dataset = classify_file(workbook_name)
            header_row = HEADER_ROWS.get((category,dataset), 0)

            try:
                df = pd.read_excel(file, sheet_name=default_sheet, header=header_row)
            except:
                xl = pd.ExcelFile(file)
                df = xl.parse(xl.sheet_names[0], header=header_row)

            df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)

            year = extract_year(file)
            df["year"] = year

            data[category][dataset][year] = df
            print(f"✓ Loaded: {workbook_name} → {category}/{dataset}/{year}")

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
    for file in glob.glob(os.path.join(output_folder, "*.xlsx")):
        try:
            os.remove(file)
            print(f"🗑 Deleted old file: {file}")
        except Exception as e:
            print(f"⚠️ Could not delete {file}: {e}")

    # Export new Excel files
    for name, df in merged_data.items():
        file_path = os.path.join(output_folder, f"{name}.xlsx")
        df.to_excel(file_path, index=False)
        print(f"✅ Exported {name} → {file_path}")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    file_path = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\raw data\raw data"
    output_path = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\cleaned data"
    schema_file = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\schemas.xlsx"

    # Load schemas from Excel
    SCHEMAS = load_schemas_from_excel(schema_file)

    # Load all Excel datasets
    data = load_excel_files(file_path)

    merged_data = {}
    redemption_dfs = []

    for category in data:
        for dataset in data[category]:
            dataset_name = f"{category}_{dataset}"  # match your Excel sheet names
            schema = SCHEMAS.get(dataset_name, {})

            if (category, dataset) in [("mall", "campaign"), ("brand", "rewards")]:
                _, df = merge_dataset_dynamic(
                    data[category][dataset],
                    schema_map=schema,
                    dataset_base_name="redemptions",
                    extra_columns={
                        "redemption_source": dataset_name,
                        "funding_type": category
                    }
                )
                redemption_dfs.append(df)
            else:
                df_name, df = merge_dataset_dynamic(
                    data[category][dataset],
                    schema_map=schema,
                    dataset_base_name=dataset_name
                )
                merged_data[df_name] = df

    # Final combined redemptions fact table
    if redemption_dfs:
        merged_data["redemptions_all"] = pd.concat(redemption_dfs, ignore_index=True)

    # Export all datasets
    export_to_excel(merged_data, output_path)
