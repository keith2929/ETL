"""
main.py
-------
Entry point for the full ETL + analysis pipeline.
Loads config once and passes paths to all scripts.

Usage (developers):
    python main.py                       # uses default config
    python main.py config_Keith.xlsx
    python main.py config_Kim.xlsx

Usage (client):
    Double-click run_pipeline.bat (Windows) or run_pipeline.command (Mac)
"""

import subprocess
import sys
from pathlib import Path

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

SCRIPTS = [
    ("Data Loader", SCRIPT_DIR / "data_loader.py"),
    ("Regression",  SCRIPT_DIR / "regression.py"),
]

DEFAULT_CONFIG = "config_Keith.xlsx"


def load_config_paths(config_file: str) -> dict:
    """Read all paths from config once. All scripts receive paths as args."""
    import pandas as pd

    config_path = SCRIPT_DIR / config_file
    if not config_path.exists():
        print(f"⚠️  Config file '{config_file}' not found — scripts will use their hardcoded defaults.")
        return {}

    paths_df = pd.read_excel(config_path, sheet_name='paths')
    config = dict(zip(paths_df['Setting'].astype(str).str.strip(), paths_df['Value']))
    return {k: str(v).strip() for k, v in config.items()}


def run_script(label: str, path: Path, args: list) -> bool:
    """Run a script as a subprocess with given args. Returns True if successful."""
    print("\n" + "="*60)
    print(f"▶  Running: {label}  ({path.name})")
    print("="*60)

    result = subprocess.run(
        [sys.executable, str(path)] + args,
        cwd=str(SCRIPT_DIR)
    )

    if result.returncode == 0:
        print(f"\n✅ {label} completed successfully.")
        return True
    else:
        print(f"\n❌ {label} failed with exit code {result.returncode}.")
        return False


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG

    print("="*60)
    print("  PIPELINE STARTING")
    print("="*60)
    print(f"  Config:     {config_file}")
    print(f"  Script dir: {SCRIPT_DIR}")

    # Load config once here — pass paths as arguments to each script
    paths = load_config_paths(config_file)

    raw_data     = paths.get('raw_data',      '')
    cleaned_data = paths.get('cleaned_data',  '')
    combined_data= paths.get('combined_data', '')
    schemas      = paths.get('schemas',       '')
    shop_mapping  = paths.get('shop_mapping',   '')

    if not all([raw_data, cleaned_data, combined_data, schemas]):
        print("\n❌ One or more required paths are missing from config. Check your config file.")
        print("   Required settings: raw_data, cleaned_data, combined_data, schemas")
        sys.exit(1)

    print(f"\n  raw_data:      {raw_data}")
    print(f"  cleaned_data:  {cleaned_data}")
    print(f"  combined_data: {combined_data}")
    print(f"  schemas:       {schemas}")
    print(f"  shop_mapping:  {shop_mapping}")

    # Script definitions with the args each needs
    scripts_with_args = [
        ("Data Loader", SCRIPT_DIR / "data_loader.py", [raw_data, cleaned_data, schemas, config_file]),
        ("Regression",  SCRIPT_DIR / "regression.py",  [cleaned_data, combined_data]),
    ]

    for label, script_path, args in scripts_with_args:
        if not script_path.exists():
            print(f"\n❌ Script not found: {script_path}")
            print("   Pipeline aborted.")
            sys.exit(1)

        success = run_script(label, script_path, args)

        if not success:
            print(f"\n⛔ Pipeline stopped after '{label}' failed.")
            sys.exit(1)

    print("\n" + "="*60)
    print("  ✅ FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)