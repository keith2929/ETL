"""
convert_campaign_format.py
--------------------------
Converts new-format campaign files to the standard 7-column format
expected by the ETL pipeline.

New format (38 columns):
  SrNo., Campaign Code, Member ID, ..., Outlet Code, Outlet Name, ..., Transact Date, Receipt No

Standard format (7 columns):
  SrNo., Voucher Type Code, Voucher Value, Redeem Outlet Code, Redeem Outlet Name, Redeem Date, Receipt No

Usage:
  python3 convert_campaign_format.py                  # uses config_Kim.xlsx
  python3 convert_campaign_format.py config_Kim.xlsx
  python3 convert_campaign_format.py config_Keith.xlsx
"""

import pandas as pd
import os
import sys
from pathlib import Path

# ── Column mapping: new format → standard format ──────────────────────────
COL_MAP = {
    'SrNo.':         'SrNo.',
    'Campaign Code': 'Voucher Type Code',
    'Outlet Code':   'Redeem Outlet Code',
    'Outlet Name':   'Redeem Outlet Name',
    'Transact Date': 'Redeem Date',
    'Receipt No':    'Receipt No',
}

STANDARD_COLS = [
    'SrNo.',
    'Voucher Type Code',
    'Voucher Value',
    'Redeem Outlet Code',
    'Redeem Outlet Name',
    'Redeem Date',
    'Receipt No',
]

# Standard format header signature — used to detect if a file needs conversion
STANDARD_HEADER = {'SrNo.', 'Voucher Type Code', 'Redeem Outlet Name'}
NEW_FORMAT_HEADER = {'Campaign Code', 'Outlet Name', 'Transact Date'}


# ── Load config ────────────────────────────────────────────────────────────
def load_raw_data_path(config_file: str) -> str:
    script_dir  = Path(__file__).resolve().parent
    config_path = script_dir / config_file

    if not config_path.exists():
        print(f"ERROR Config file not found: {config_path}")
        sys.exit(1)

    df = pd.read_excel(config_path, sheet_name='paths')
    config = dict(zip(df['Setting'].astype(str).str.strip(), df['Value'].astype(str).str.strip()))
    raw_data = config.get('raw_data', '').strip()

    if not raw_data:
        print("ERROR 'raw_data' path not found in config.")
        sys.exit(1)

    return raw_data


# ── Detect header row ──────────────────────────────────────────────────────
def find_header_row(path: str) -> int:
    df_raw = pd.read_excel(path, header=None)
    for i, row in df_raw.iterrows():
        if any(str(v) == 'SrNo.' for v in row.values):
            return i
    return 0


# ── Check if file needs conversion ────────────────────────────────────────
def needs_conversion(path: str) -> bool:
    """Returns True if the file is in new format and needs conversion."""
    header_row = find_header_row(path)
    df = pd.read_excel(path, header=header_row, nrows=0)
    cols = set(df.columns.str.strip())

    if NEW_FORMAT_HEADER.issubset(cols):
        return True   # new format — needs conversion
    if STANDARD_HEADER.issubset(cols):
        return False  # already standard format
    return False      # unknown — skip


# ── Convert one file ───────────────────────────────────────────────────────
def convert_file(input_path: str) -> str:
    """
    Convert a new-format campaign file to standard format.
    Saves output as <original_stem>_converted.xlsx in the same folder.
    Returns the output path.
    """
    print(f"\n  Converting: {os.path.basename(input_path)}")

    header_row = find_header_row(input_path)
    print(f"  Header row: {header_row}")

    df = pd.read_excel(input_path, header=header_row)
    df.columns = df.columns.str.strip()
    df = df.dropna(how='all').reset_index(drop=True)
    print(f"  Rows after cleaning: {len(df)}")

    # Select and rename columns
    missing = [c for c in COL_MAP if c not in df.columns]
    if missing:
        print(f"  WARNING Missing columns: {missing}")

    available = {k: v for k, v in COL_MAP.items() if k in df.columns}
    df_out = df[list(available.keys())].rename(columns=available)

    # Add Voucher Value as empty (not in new format)
    df_out.insert(2, 'Voucher Value', '')

    # Ensure all standard columns exist
    for col in STANDARD_COLS:
        if col not in df_out.columns:
            df_out[col] = ''
    df_out = df_out[STANDARD_COLS]

    # Build output path — replace original file
    stem   = Path(input_path).stem
    folder = Path(input_path).parent

    # Remove old suffixes like _2Mar2026, _2, etc. and add _converted
    output_path = str(folder / f"{stem}_converted.xlsx")

    df_out.to_excel(output_path, index=False)
    print(f"  Saved: {os.path.basename(output_path)}")
    print(f"  Sample:")
    print(df_out.head(3).to_string(index=False))

    return output_path


# ── Scan raw data folder ───────────────────────────────────────────────────
def scan_and_convert(raw_data_folder: str):
    """
    Scan raw data folder for campaign/brand files that need conversion.
    Skips combined_ files and outlet-code-mapping files.
    """
    converted = []
    skipped   = []
    already_ok = []

    print(f"\nScanning: {raw_data_folder}")

    for f in sorted(os.listdir(raw_data_folder)):
        if not f.endswith('.xlsx'):
            continue

        fl = f.lower()

        # Skip non-campaign files
        if fl.startswith('combined_'):
            continue
        if 'outlet' in fl and 'code' in fl:
            continue
        if 'gto' in fl:
            continue
        if 'member' in fl:
            continue

        # Only process campaign / brand reward files
        is_campaign = ('campaign' in fl or 'brand' in fl or 'reward' in fl)
        if not is_campaign:
            continue

        path = os.path.join(raw_data_folder, f)

        try:
            if needs_conversion(path):
                out = convert_file(path)
                converted.append((f, os.path.basename(out)))
            else:
                already_ok.append(f)
                print(f"  OK (already standard format): {f}")
        except Exception as e:
            skipped.append((f, str(e)))
            print(f"  ERROR {f}: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)

    if converted:
        print(f"\n✅ Converted ({len(converted)}):")
        for orig, out in converted:
            print(f"   {orig}")
            print(f"   → {out}")
            print(f"\n   ACTION REQUIRED: Delete the original file and")
            print(f"   rename the _converted file to replace it.")

    if already_ok:
        print(f"\n✅ Already standard format ({len(already_ok)}):")
        for f in already_ok:
            print(f"   {f}")

    if skipped:
        print(f"\n⚠️  Skipped due to errors ({len(skipped)}):")
        for f, err in skipped:
            print(f"   {f}: {err}")

    if converted:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Check the _converted files look correct")
        print("2. Delete the original files")
        print("3. Rename _converted files (remove '_converted' suffix)")
        print("4. Re-run the pipeline")
        print("="*60)
    return converted

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config_Kim.xlsx"

    print("="*60)
    print("CAMPAIGN FORMAT CONVERTER")
    print("="*60)
    print(f"Config: {config_file}")

    raw_data_folder = load_raw_data_path(config_file)
    print(f"Raw data folder: {raw_data_folder}")

    if not os.path.exists(raw_data_folder):
        print(f"ERROR Raw data folder not found: {raw_data_folder}")
        sys.exit(1)

    converted = scan_and_convert(raw_data_folder)

    # ── Auto rename: delete original, rename _converted ───────────────────
    if converted:
        print("\n" + "="*60)
        print("AUTO RENAMING FILES")
        print("="*60)
        for orig, out in converted:
            orig_path = os.path.join(raw_data_folder, orig)
            out_path  = os.path.join(raw_data_folder, out)

            # Build final name: remove '_converted' suffix
            final_name = out.replace('_converted', '')
            final_path = os.path.join(raw_data_folder, final_name)

            try:
                # Delete original
                if os.path.exists(orig_path):
                    os.remove(orig_path)
                    print(f"  🗑  Deleted original: {orig}")

                # Rename _converted → final name
                os.rename(out_path, final_path)
                print(f"  ✅ Renamed to: {final_name}")

            except Exception as e:
                print(f"  ⚠️  Could not rename {out}: {e}")

        print("\n✅ All done! You can now re-run the pipeline.")