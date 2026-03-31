# Capstone ETL Pipeline

Processes mall, brand, and GTO Excel files into clean analysis.

---

## Requirements

### Python version

Python **3.9 or later** is required. The pipeline has been tested on Python 3.10 and 3.11.

Recommended distribution: [Anaconda](https://www.anaconda.com) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### Python packages

Install all dependencies with:

```bash
pip install streamlit pandas openpyxl statsmodels scipy scikit-learn plotly
```

Or use the provided launcher scripts — they install everything automatically on first run.

| Package | Version (min) | Purpose |
|---|---|---|
| `streamlit` | 1.30+ | Web UI |
| `pandas` | 2.0+ | Data loading, cleaning, merging |
| `openpyxl` | 3.1+ | Reading / writing `.xlsx` files |
| `statsmodels` | 0.14+ | OLS regression, time series (STL, ETS, ADF) |
| `scipy` | 1.11+ | T-tests, normality tests, Pearson correlation |
| `scikit-learn` | 1.3+ | StandardScaler, KFold cross-validation, LassoCV |
| `plotly` | 5.18+ | Interactive charts in the Streamlit app |
| `numpy` | 1.24+ | Numerical operations (installed automatically with pandas) |

### Node.js (optional)

Only required if you want to generate `.docx` report files outside the pipeline. Not needed to run the app.

```bash
npm install -g docx
```

### Operating system

The pipeline runs on **Windows, macOS, and Linux**. Launcher scripts are provided for Windows (`.bat`) and macOS (`.command`).

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
├── app_FINAL.py              ← Streamlit web app (the UI)
├── main_FINAL.py             ← Pipeline orchestrator
├── data_load_FINAL.py        ← ETL script (loads combined campaign + GTO files)
├── regression_FINAL.py       ← Time series analysis (ETS, STL, month dummies)
├── linear_regression_FINAL.py← Regression 1: Y = Amount by campaign
├── ttest_FINAL.py            ← T-tests, normality tests, ROI per campaign
├── schemas.xlsx              ← Column name mappings (shared, edit via app)
├── shop_mapping.xlsx         ← Shop name mappings (auto-generated, edit via app)
├── config_Keith.xlsx         ← Keith's personal config (gitignored)
├── config_Kim.xlsx           ← Kim's personal config (gitignored)
├── windows_run.bat           ← Windows launcher
└── ios_run.command           ← Mac launcher
```

Your data folders (raw, cleaned, combined) live **outside** the repo and are referenced via your personal config file.

---

## Getting started

### 1. Install dependencies

```bash
pip install streamlit pandas openpyxl statsmodels scipy scikit-learn plotly
```

Or just launch the app — the launcher scripts install everything automatically.

### 2. Set up your config file

Copy `config.xlsx` and rename it `config_YourName.xlsx`. Open it and update the paths sheet with your own folder locations:

| Setting | Description |
|---|---|
| raw_data | Folder containing your raw Excel files |
| cleaned_data | Where cleaned outputs will be saved |
| combined_data | Where merged outputs will be saved |

Keep your config file in the ETL folder. It is gitignored so your personal paths are never committed.

### 3. Prepare raw data files

Place the following files in your `raw_data` folder:

| File (name must contain) | Description |
|---|---|
| `combined_Mall_Campaign.xlsx` | Mall-funded voucher redemptions |
| `combined_Brand_Campaign.xlsx` | Brand-funded voucher redemptions |
| `combined_Mall_Trans.xlsx` | Member transactions (contains Amount) |
| `*gto*08*.xlsx` | GTO monthly sales file (header row 7) |

> **Note:** The GTO file is optional. If absent, the Member / Non-Member Sales charts in the Time Series tab will be empty but the rest of the pipeline will run normally.

### 4. Launch the app

**Windows** — double-click `windows_run.bat`

**Mac** — double-click `ios_run.command`
> First time only: open Terminal, type `chmod +x ` then drag the `.command` file in and press Enter. Then double-click works normally.

Your browser will open automatically at `http://localhost:8501`.

---

## Using the app

### ▶ Run tab

Select your config file from the dropdown and click **RUN PIPELINE**. The log streams live so you can see exactly what's happening and where it fails if something goes wrong.

A **Reset** button is available if the pipeline gets stuck.

### ⚙ Config tab

Edit your folder paths directly in the app. Click the 📂 button next to any path to open that folder in your file explorer. Click **Save Config** when done.

### 📈 Time Series tab

Shows monthly member spend trends, seasonal decomposition (STL), Holt-Winters forecasts with prediction intervals, month-dummies regression (which months are significantly different), and Member vs Non-Member Sales split.

Use the **Prediction Interval** selector (80% / 90% / 95%) to control band width. 80% is recommended for retail settings.

### 📉 Regression tab

Runs OLS regression: **Y = transaction Amount**, **X = campaign (voucher code) dummies**. Significant campaigns are highlighted. Includes a campaign revenue summary table and residual diagnostics.

### 🧪 T-Test & ROI tab

Per-campaign statistical analysis:
- **Normality test** — Shapiro-Wilk (n ≤ 30) or CLT (n > 30)
- **One-sample t-test** — tests whether each campaign's mean revenue differs from the overall mean
- **ROI** — total revenue / total voucher cost per campaign
- **Reliable results** — campaigns that are both significant and normally distributed

All tabs support live **filters** (by campaign source, outlet, month, voucher code) that re-run the analysis on the filtered subset without needing to re-run the pipeline.

---

## Running without the app (developers)

```bash
# Run full pipeline with default config
python main_FINAL.py

# Run with a specific config
python main_FINAL.py config_Keith.xlsx
```

Individual scripts can also be run directly:

```bash
python data_load_FINAL.py config_Keith.xlsx
python regression_FINAL.py <cleaned_data> <combined_data>
python linear_regression_FINAL.py <cleaned_data> <combined_data>
python ttest_FINAL.py <cleaned_data> <combined_data>
```

---

## Pipeline outputs

After a successful run the following files are saved:

**Cleaned data folder**

| File | Description |
|---|---|
| `campaign_all.xlsx / .csv` | Merged campaign + transaction data |
| `gto_member_sales.csv` | Monthly member sales from campaign amounts |
| `gto_nonmember_sales.csv` | Monthly non-member sales (GTO − member) |
| `gto_member_sales_forecast.csv` | 24-month member sales forecast |
| `gto_nonmember_sales_forecast.csv` | 24-month non-member sales forecast |

**Combined data folder**

| File | Description |
|---|---|
| `insights.json` | Time series results (trends, forecasts, decomposition) |
| `insights_report.xlsx` | Time series results as Excel sheets |
| `linear_regression_results.json` | Regression coefficients and model fit |
| `linear_regression_summary.xlsx` | Regression summary table |
| `ttest_results.json` | T-test, normality, and ROI results |
| `ttest_results.xlsx` | T-test results as Excel sheets |

---

## Config file reference

Each person maintains their own `config_<name>.xlsx` with one sheet:

**paths sheet**

| Setting | Example value |
|---|---|
| raw_data | `C:\...\capstone\raw data\raw data` |
| cleaned_data | `C:\...\capstone\cleaned data` |
| combined_data | `C:\...\capstone\combined data` |

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

- Python 3.9+
- pandas, openpyxl, numpy
- statsmodels (OLS, STL, ETS, ADF, Ljung-Box)
- scipy (t-tests, Shapiro-Wilk, Pearson)
- scikit-learn (StandardScaler, KFold CV, LassoCV)
- plotly (interactive charts)
- Streamlit (UI)
