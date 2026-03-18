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

import os
import sys
import subprocess
import io

# Force UTF-8 output so emoji from subprocesses display correctly on Windows
# Guarded: Spyder's TTYOutStream has no .buffer attribute so we skip it there
if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

try:
    # Running as a PyInstaller .exe — files are extracted to a temp folder
    SCRIPT_DIR = Path(sys._MEIPASS)
except AttributeError:
    try:
        SCRIPT_DIR = Path(__file__).resolve().parent
    except NameError:
        SCRIPT_DIR = Path.cwd()

# Config, schemas, shop_mapping sit next to the .exe (or script), not in the bundle
try:
    CONFIG_DIR = Path(sys.executable).resolve().parent
except Exception:
    CONFIG_DIR = SCRIPT_DIR

SCRIPTS = [
    ("Data Loader", SCRIPT_DIR / "data_loader.py"),
    ("Regression",  SCRIPT_DIR / "regression.py"),
]

DEFAULT_CONFIG = "config_Kim.xlsx"


def load_config_paths(config_file: str) -> dict:
    """Read all paths from config once. All scripts receive paths as args."""
    import pandas as pd

    # Config lives next to the .exe, not inside the bundle
    config_path = CONFIG_DIR / config_file
    if not config_path.exists():
        config_path = SCRIPT_DIR / config_file  # fallback for dev mode
    if not config_path.exists():
        print(f"WARNING  Config file '{config_file}' not found in {CONFIG_DIR}")
        return {}

    paths_df = pd.read_excel(config_path, sheet_name='paths')
    config = dict(zip(paths_df['Setting'].astype(str).str.strip(), paths_df['Value']))
    return {k: str(v).strip() for k, v in config.items()}


def run_script(label: str, path: Path, args: list) -> bool:
    """Run a script as a subprocess with given args. Returns True if successful."""
    print("\n" + "="*60)
    print(f"Running: {label}  ({path.name})")
    print("="*60)

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    process = subprocess.Popen(
        [sys.executable, str(path)] + args,
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        env=env
    )
    for line in process.stdout:
        # sys.stdout.buffer is unavailable in Spyder (TTYOutStream) — use print as fallback
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout.buffer.write(line.encode('utf-8', errors='replace'))
            sys.stdout.buffer.flush()
        else:
            print(line, end='', flush=True)
    process.wait()

    if process.returncode == 0:
        print(f"\nOK {label} completed successfully.")
        return True
    else:
        print(f"\nERROR {label} failed with exit code {process.returncode}.")
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

    raw_data      = paths.get('raw_data',      '')
    cleaned_data  = paths.get('cleaned_data',  '')
    combined_data = paths.get('combined_data', '')
    schemas       = paths.get('schemas',       '')
    shop_mapping  = paths.get('shop_mapping',  '')

    if not all([raw_data, cleaned_data, combined_data, schemas, shop_mapping]):
        print("\nERROR One or more required paths are missing from config. Check your config file.")
        print("   Required settings: raw_data, cleaned_data, combined_data, schemas, shop_mapping")
        sys.exit(1)

    print(f"\n  raw_data:      {raw_data}")
    print(f"  cleaned_data:  {cleaned_data}")
    print(f"  combined_data: {combined_data}")
    print(f"  schemas:       {schemas}")
    print(f"  shop_mapping:  {shop_mapping}")

    # Script definitions with the args each needs
    scripts_with_args = [
        ("Data Loader", SCRIPT_DIR / "data_loader.py", [raw_data, cleaned_data, schemas, shop_mapping, config_file]),
        ("Regression",  SCRIPT_DIR / "regression.py",  [cleaned_data, combined_data, shop_mapping]),
        ("Linear Regression", SCRIPT_DIR / "linear_regression.py", [cleaned_data, combined_data, shop_mapping]),
    ]

    for label, script_path, args in scripts_with_args:
        if not script_path.exists():
            print(f"\nERROR Script not found: {script_path}")
            print("   Pipeline aborted.")
            sys.exit(1)

        success = run_script(label, script_path, args)

        if not success:
            print(f"\nSTOPPED Pipeline stopped after '{label}' failed.")
            sys.exit(1)

    print("\n" + "="*60)
    print("  OK FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)