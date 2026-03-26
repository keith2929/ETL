"""
main.py
-------
Pipeline entry point. Runs in order:
  1. data_load_FINAL.py        — load 3 combined files → campaign_all
  2. regression_FINAL.py         — time series (Y=Amount, month dummies)
  3. linear_regression_FINAL.py  — Regression 1 + 2 (Y=Amount)
"""

import os, sys, subprocess, io
from pathlib import Path

if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    SCRIPT_DIR = Path(sys._MEIPASS)
except AttributeError:
    try:
        SCRIPT_DIR = Path(__file__).resolve().parent
    except NameError:
        SCRIPT_DIR = Path.cwd()

try:
    CONFIG_DIR = Path(sys.executable).resolve().parent
except:
    CONFIG_DIR = SCRIPT_DIR

DEFAULT_CONFIG = "config_Kim.xlsx"


def load_config_paths(config_file: str) -> dict:
    import pandas as pd
    config_path = CONFIG_DIR / config_file
    if not config_path.exists():
        config_path = SCRIPT_DIR / config_file
    if not config_path.exists():
        print(f"WARNING Config file '{config_file}' not found")
        return {}
    df  = pd.read_excel(config_path, sheet_name='paths')
    cfg = dict(zip(df['Setting'].astype(str).str.strip(), df['Value']))
    return {k: str(v).strip() for k, v in cfg.items()}


def run_script(label: str, path: Path, args: list) -> bool:
    print("\n" + "="*60)
    print(f"Running: {label}  ({path.name})")
    print("="*60)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    process = subprocess.Popen(
        [sys.executable, str(path)] + args,
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', env=env
    )
    for line in process.stdout:
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

    paths = load_config_paths(config_file)

    raw_data      = paths.get('raw_data',      '')
    cleaned_data  = paths.get('cleaned_data',  '')
    combined_data = paths.get('combined_data', '')

    if not all([raw_data, cleaned_data, combined_data]):
        print("\nERROR Missing required paths (raw_data, cleaned_data, combined_data)")
        sys.exit(1)

    print(f"\n  raw_data:      {raw_data}")
    print(f"  cleaned_data:  {cleaned_data}")
    print(f"  combined_data: {combined_data}")

    scripts = [
        ("Data Loader",
         SCRIPT_DIR / "data_load_FINAL.py",
         [raw_data, cleaned_data]),
        ("Time Series",
         SCRIPT_DIR / "regression_FINAL.py",
         [cleaned_data, combined_data]),
        ("Linear Regression",
         SCRIPT_DIR / "linear_regression_FINAL.py",
         [cleaned_data, combined_data]),
    ]

    for label, script_path, args in scripts:
        if not script_path.exists():
            print(f"\nERROR Script not found: {script_path}")
            sys.exit(1)
        if not run_script(label, script_path, args):
            print(f"\nSTOPPED after '{label}' failed.")
            sys.exit(1)

    print("\n" + "="*60)
    print("  OK FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)