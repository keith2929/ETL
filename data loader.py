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
# Schema Maps (fill in for your needs)
# -----------------------------
SCHEMAS = {
    ("mall","campaign"): {
        "SrNo.": "sr_no",
        "Receipt No": "receipt_no",
        "Voucher Type Code": "voucher_code",
        "Campaign Code": "voucher_code",
        "Voucher Value": "voucher_value",
        "Redeem Outlet Code": "outlet_code",
        "Outlet Code": "outlet_code",
        "Redeem Outlet Name": "outlet_name",
        "Outlet Name": "outlet_name",
        "Redeem Date": "transaction_date",
        "Transact Date": "transaction_date"
    },
    ("mall","member"): {
        "SrNo.":"sr_no",
        "Trans Date":"transaction_date",
        "Outlet Code":"outlet_code",
        "Outlet Name":"outlet_name",
        "Type":"transaction_type",
        "TransactRef5":"points_earned",
        "TransactRef6":"points_formula",
        "Amount Spent":"amount"
    },
    ("brand","rewards"): {
        "SrNo.":"sr_no",
        "Voucher Type Code":"voucher_code",
        "Voucher Value": "voucher_value",
        "Redeem Outlet Code": "outlet_code",
        "Redeem Outlet Name": "outlet_name",
        "Redeem Date": "transaction_date",
        "Receipt No": "receipt_no"        
        },
    
    ("gto","monthly_sales"): {
        "Unit No.":"unit_no",
        "Contract No":"contract_no",
        "Contract Name":"contract_name",
        "Lease Start":"lease_start",
        "Lease End":"lease_end",
        "Online Sales Jan 2025":"online_sales",
        "Offline Sales Jan 2025":"offline_sales",
        "Estimation Jan 2025 $":"estimated_sales",
        "Total GTO Sales Jan 2025 $":"total_GTO"
        },
    ("gto","monthly_rent"): {
        "Lease Number":"lease_no",
        "Customer Group":"customer_group",
        "Customer":"customer",
        "Shop Name":"shop_name",
        "Business Entity":"business_entity",
        "Cost center":"cost_center",
        "Site(s)":"site",
        "Building":"building",
        "Level(s)":"level",
        "Unit(s)": "unit_no",
        "Lease Start Date":"lease_start_date",
        "Lease Expiry Date":"lease_expiry_date",
        "Lease Terimination Date":"lease_termination_date",
        "Biz Commencement Date":"biz_commencement_date",
        "Lease Status":"lease_status",
        "Lease Type":"lease_type",
        "Trade Type":"trade_type",
        "Sub Trade Type": "sub_trade_type",
        "Usage Type": "usage_type",
        "Space Type": "space_type",
        "Space Design Type": "space_design_type",
        "NLA(sq ft)": "nla_sqft",
        "GTO Reporting Month": "gto_reporting_month",
        "GTO Period From": "gto_period_from",
        "GTO Period To": "gto_period_to",
        "Product Type": "product_type",
        "GTO Amount ($)": "gto_amount",
        "Sale GTO?": "sale_gto",
        "GTO Rent ($)": "gto_rent",
        "GTO Type": "gto_type",
        "GTO Source": "gto_source",
        "Ver. No.": "ver_no",
        "Active Record?": "active_record",
        "GTO Updated Date": "gto_updated_date",
        "Reported GTO No.": "reported_gto_no",
        "GTO Adjustment No.": "gto_adjustment_no",
        "Lease GTO ObjectID": "lease_gto_objectid",
        "Count": "count"
        },
    ("gto","tenant_turnover"): {
        "Business Entity": "business_entity",
        "Building": "building",
        "Level(s)": "level",
        "Unit(s)": "unit_no",
        "Lease Number": "lease_no",
        "Shop Name": "shop_name",
        "Customer Name": "customer_name",
        "NLA": "nla",
        "ION Trade Type Code": "ion_trade_type_code",
        "ION Trade Type Name": "ion_trade_type_name",
        "JV(SG) Trade Type": "jv_sg_trade_type",
        "JV(SG) Sub Trade": "jv_sg_sub_trade",
        "JV(HK) Trade Type Code": "jv_hk_trade_type_code",
        "JV(HK) Trade Type Name": "jv_hk_trade_type_name",
        "Lease Commencement Date": "lease_commencement_date",
        "Lease Expiry Date": "lease_expiry_date",
        "Lease Status": "lease_status",
        "GTO Description": "gto_description",
        "Min GTO Rate$psf": "min_gto_rate_psf",
        "Matrix": "matrix",
        "Total": "total",
        "Average": "average",
        "Count": "count"
    # Month columns like 2024-1, 2024-2, … are handled dynamically in your loader
    }

}

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
            schema = SCHEMAS.get((category, dataset), {})
            header_row = HEADER_ROWS.get((category,dataset), 0)

            # Read Excel
            try:
                df = pd.read_excel(file, sheet_name=default_sheet, header=header_row)
            except:
                xl = pd.ExcelFile(file)
                df = xl.parse(xl.sheet_names[0], header=header_row)

            # Clean blank rows/columns
            df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)

            # Add year column from filename
            year = extract_year(file)
            df["year"] = year

            # Store
            data[category][dataset][year] = df
            print(f"✓ Loaded: {workbook_name} → {category}/{dataset}/{year}")

        except Exception as e:
            print(f"✗ Error loading {workbook_name}: {e}")

    return data

# -----------------------------
# Merge Datasets
# -----------------------------
def merge_dataset_dynamic(dataset_dict, schema_map, dataset_base_name="Dataset", add_month_year=True):
    dfs = []
    years = sorted(dataset_dict.keys())
    for year in years:
        df = dataset_dict[year].copy()
        df = standardise_schema(df, schema_map)
        if add_month_year:
            df = add_month_year_columns(df)
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    start_year = years[0]
    end_year = years[-1]
    df_name = f"{dataset_base_name}_{start_year}_to_{end_year}"
    return df_name, merged_df

# -----------------------------
# Export to Excel
# -----------------------------
def export_to_excel(merged_data, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for name, df in merged_data.items():
        file_path = os.path.join(output_folder, f"{name}.xlsx")
        df.to_excel(file_path, index=False)
        print(f"✅ Exported {name} → {file_path}")

# -----------------------------
# Usage
# -----------------------------
if __name__ == "__main__":
    # User just changes this path
    file_path = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\raw data\raw data"
    output_path = r"C:\Users\user\OneDrive - Singapore Management University\Desktop\smu\subject\y4s2\capstone\cleaned data"

    data = load_excel_files(file_path)

    # Merge all datasets
    merged_data = {}
    for category in data:
        for dataset in data[category]:
            schema = SCHEMAS.get((category,dataset), {})
            df_name, df = merge_dataset_dynamic(
                data[category][dataset],
                schema_map=schema,
                dataset_base_name=f"{category}_{dataset}"
            )
            merged_data[df_name] = df
