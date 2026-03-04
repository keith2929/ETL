# Capstone ETL Pipeline

Processes mall, brand, and GTO Excel files into clean, Power BI–ready datasets.

---

## What it does

1. Reads raw Excel files from your data folder
2. Standardises column names using your schema definitions
3. Cleans and exports datasets to your cleaned data folder
4. Resolves shop name mismatches between campaign and GTO data
5. Merges campaign, transaction, and GTO data into combined outputs

---

## Folder structure

```
ETL/
├── app.py                  ← Streamlit web app (the UI)
├── main.py                 ← Pipeline orchestrator
├── data_loader.py          ← ETL script
├── regression.py           ← Merging and analysis script
├── schemas.xlsx            ← Column name mappings (shared, edit via app)
├── shop_mapping.xlsx       ← Shop name mappings (auto-generated, edit via app)
├── config_Keith.xlsx       ← Keith's personal config (gitignored)
├── config_Kim.xlsx         ← Kim's personal config (gitignored)
├── launch_app.bat          ← Windows launcher
└── launch_app.command      ← Mac launcher
```

Your data folders (raw, cleaned, combined) live **outside** the repo and are referenced via your personal config file.

---

## Getting started

### 1. Install dependencies

```
pip install streamlit pandas openpyxl statsmodels scipy
```

Or just launch the app — the launcher scripts install everything automatically.

### 2. Set up your config file

Copy `config.xlsx` and rename it `config_YourName.xlsx`. Open it and update the paths sheet with your own folder locations:

| Setting | Description |
|---|---|
| raw_data | Folder containing your raw Excel files |
| cleaned_data | Where cleaned outputs will be saved |
| combined_data | Where merged outputs will be saved |
| schemas | Full path to `schemas.xlsx` |
| shop_mapping | Full path to `shop_mapping.xlsx` |

Keep your config file in the ETL folder. It is gitignored so your personal paths are never committed.

### 3. Launch the app

**Windows** — double-click `launch_app.bat`

**Mac** — double-click `launch_app.command`
> First time only: open Terminal, type `chmod +x ` then drag the `.command` file in and press Enter. Then double-click works normally.

Your browser will open automatically at `http://localhost:8501`.

---

## Using the app

### ▶ Run tab

Select your config file from the dropdown and click **RUN PIPELINE**. The log streams live so you can see exactly what's happening and where it fails if something goes wrong.

### ⚙ Config tab

Edit your folder paths and GTO header row settings directly in the app. Click the 📂 button next to any path to open that folder in your file explorer. Click **Save Config** when done.

### 🔗 Shop Mapping tab

Campaign data and GTO data often record shop names differently (e.g. `dunkin'donuts` vs `DUNKIN'`). The pipeline handles this automatically:

- **First run** — the pipeline auto-matches names using fuzzy matching and saves a `shop_mapping.xlsx` file. This tab shows the results.
- **Review** — check the table. The `method` column tells you how each match was found:
  - `exact` — perfect match after normalising case/spaces ✅
  - `fuzzy` — close but not exact, check it looks right 🟡
  - `unmatched` — campaign outlet with no GTO match found ❌
  - `gto_only` — GTO shop with no campaign activity ℹ️
  - `confirmed` — manually confirmed by you ✅
- **Fix** — type the correct GTO name into the **✏ Confirmed** column for any row that looks wrong or is unmatched.
- **Save** — click **Save Shop Mapping**, then re-run the pipeline. Confirmed entries are preserved across all future runs.

### 📋 Schema tab

View and edit the column name mappings for each dataset. `original_column` is the name in your raw files, `canonical_column` is the standardised name the pipeline uses. You can add or delete rows. Click **Save Schema** when done.

---

## Running without the app (developers)

```bash
# Default config
python main.py

# Specific config
python main.py config_Keith.xlsx
```

Individual scripts can also be run directly in Spyder or VS Code — they will read the config file automatically when no arguments are passed.

```bash
python data_loader.py
python regression.py
```

---

## Config file reference

Each person maintains their own `config_<name>.xlsx` with two sheets:

**paths sheet**

| Setting | Example value |
|---|---|
| raw_data | C:\...\capstone\raw data\raw data |
| cleaned_data | C:\...\capstone\cleaned data |
| combined_data | C:\...\capstone\combined data |
| schemas | C:\...\ETL\schemas.xlsx |
| shop_mapping | C:\...\ETL\shop_mapping.xlsx |

**gto_headers sheet**

| category | dataset | header_row |
|---|---|---|
| gto | monthly_sales | 7 |
| gto | monthly_rent | 8 |
| gto | tenant_turnover | 7 |

`header_row` is the row number (1-indexed) where column headers start in each GTO file.

---

## Gitignore

The following are excluded from version control:

```
config_*.xlsx       # personal configs with local paths
raw data/           # sensitive raw data
cleaned data/       # generated outputs
combined data/      # generated outputs
```

`schemas.xlsx` and `shop_mapping.xlsx` are committed — they are shared reference files.

---

## Tech stack

- Python 3
- pandas, openpyxl, statsmodels, scipy
- Streamlit (UI)