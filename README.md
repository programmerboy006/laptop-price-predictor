# 💻 Laptop Price Predictor

An end-to-end machine learning regression project that predicts laptop prices (in Euros) based on hardware specifications. The pipeline covers data loading, exploratory data analysis, feature engineering on messy text columns, model comparison via cross-validation, and rich visualizations — all inside a Jupyter notebook.

---

## 📂 Project Structure

```
ml projects/
├── laptop.ipynb                   # Main notebook — full ML pipeline
├── laptop_backup.ipynb            # Backup of original notebook (before fixes)
├── laptop_price - dataset.csv     # Source dataset (1,275 laptops × 15 columns)
├── laptop_price_predictor.png     # Generated dashboard visualization
├── data                           # Auxiliary data file
└── README.md                      # This file
```

---

## 📊 Dataset

| Detail | Value |
|--------|-------|
| **Source** | `laptop_price - dataset.csv` |
| **Rows** | 1,275 laptops |
| **Columns** | 15 raw features |
| **Target** | `Price (Euro)` |
| **Missing values** | None |
| **Encoding** | Latin-1 |

### Raw Features

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `Company` | str | Apple, HP, Dell | Manufacturer |
| `Product` | str | MacBook Pro, 250 G6 | Model name |
| `TypeName` | str | Ultrabook, Notebook, Gaming | Laptop category |
| `Inches` | float | 13.3, 15.6, 17.3 | Screen size in inches |
| `ScreenResolution` | str | "IPS Panel Retina Display 2560x1600" | Resolution string with display features |
| `CPU_Company` | str | Intel, AMD | CPU manufacturer |
| `CPU_Type` | str | "Core i5 7200U" | CPU model string |
| `CPU_Frequency (GHz)` | float | 2.3, 1.8, 2.5 | Clock speed |
| `RAM (GB)` | int | 4, 8, 16, 32 | Installed RAM |
| `Memory` | str | "256GB SSD + 1TB HDD" | Storage string (may include dual drives) |
| `GPU_Company` | str | Intel, Nvidia, AMD | GPU manufacturer |
| `GPU_Type` | str | "Iris Plus Graphics 640" | GPU model (106 unique values) |
| `OpSys` | str | macOS, Windows 10, Linux | Operating system |
| `Weight (kg)` | float | 1.37, 2.04 | Laptop weight |
| `Price (Euro)` | float | 575.00 – 6099.00 | **Target variable** |

---

## 🔍 Exploratory Data Analysis

The notebook performs EDA to understand data characteristics:

- **Data shape & types** — `df.info()` confirms 1,275 rows, 15 columns, zero null values
- **Descriptive statistics** — `df.describe()` for numeric column distributions
- **GPU variety** — 106 unique GPU types identified across Intel, Nvidia, and AMD
- **Price distribution** — right-skewed, motivating log-transform for modeling

### Key Data Insights

| Observation | Detail |
|-------------|--------|
| **Most common CPU** | Intel Core i7 (515 laptops, 40%) |
| **Most common storage** | SSD (837 laptops, 66%) |
| **Median price** | ~€977 |
| **Price range** | €174 – €6,099 |
| **Dual-storage laptops** | Entries with "+" separator (e.g., "256GB SSD + 1TB HDD") |

---

## ⚙️ Feature Engineering

Raw text columns are parsed into model-friendly numeric/categorical features using three custom functions:

### `parse_screen()` — Screen Resolution → Numeric Features
| Feature | Type | Logic |
|---------|------|-------|
| `Is_Touchscreen` | binary | Keyword search for "Touchscreen" |
| `Is_IPS` | binary | Keyword search for "IPS" |
| `Res_Width` | float | Regex extraction from "2560x1600" (default: 1366) |
| `Res_Height` | float | Regex extraction from "2560x1600" (default: 768) |
| `PPI` | float | `√(width² + height²) / Inches` |

### `parse_storage()` — Memory → Storage Features
| Feature | Type | Logic |
|---------|------|-------|
| `Primary_GB` | int | Parse primary drive size (handles TB → GB) |
| `Secondary_GB` | int | Parse secondary drive if "+" separator exists |
| `Has_Secondary` | binary | Dual-storage flag |
| `Total_Storage_GB` | int | Sum of primary + secondary |
| `Primary_Storage_Type` | str | Categorized as SSD / HDD / Flash / Other |

### `parse_cpu()` — CPU Type → CPU Series
Maps raw CPU strings to standardized tiers using keyword matching:

| CPU Series | Count | Example Raw String |
|------------|-------|--------------------|
| Core i7 | 515 | "Core i7 7700HQ" |
| Core i5 | 423 | "Core i5 7200U" |
| Core i3 | 134 | "Core i3 6006U" |
| Celeron | 78 | "Celeron N3350" |
| AMD APU / Other | 78 | "E2-9000e" |
| Pentium | 30 | "Pentium N4200" |
| Atom | 13 | "Atom x5-Z8350" |
| Xeon | 4 | "Xeon E3-1505M V6" |

---

## 🧩 Pipeline Architecture

```
                    laptop_price - dataset.csv
                              │
                    ┌─────────┼─────────┐
                    │         │         │
              parse_screen  parse_storage  parse_cpu
                    │         │         │
                    └─────────┼─────────┘
                              │
                     17 model-ready features
                      (10 numeric + 7 categorical)
                              │
                    ┌─────────┴─────────┐
                    │                   │
              StandardScaler      OneHotEncoder
              (numeric cols)     (categorical cols)
                    │                   │
                    └─────────┬─────────┘
                              │
                      ColumnTransformer
                              │
                    ┌─────────┼─────────┐
                    │         │         │
              LinearReg     Ridge    RandomForest
                              │
                    5-Fold Cross-Validation
                              │
                    Best → RandomForest
                              │
                    Train/Test Split (80/20)
                              │
                    Predicted Price (€)
```

---

## 🤖 Models & Results

### 5-Fold Cross-Validation (on log-transformed prices)

| Model | R² Mean | R² ± Std | Log-RMSE |
|-------|---------|----------|----------|
| Linear Regression | 0.8265 | 0.0371 | 0.2531 |
| Ridge (α=10) | 0.8239 | 0.0374 | 0.2551 |
| **Random Forest** | **0.8794** | **0.0205** | **0.2116** |

### Best Model — Held-Out Test Set (20%)

| Metric | Value |
|--------|-------|
| **R²** | 0.8412 |
| **RMSE** | €280.77 |
| **Model** | RandomForestRegressor |
| **n_estimators** | 200 |
| **max_features** | sqrt |
| **random_state** | 42 |

> **Why log-transform?** The price distribution is right-skewed — a few expensive gaming/workstation laptops dominate the upper tail. Training on `log1p(price)` and converting predictions back via `expm1` reduces the influence of these outliers and improves model stability.

### Top Feature Importances (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | RAM (GB) | High |
| 2 | TypeName: Notebook | High |
| 3 | CPU Series: Core i7 | High |
| 4 | PPI | Medium |
| 5 | Primary Storage Type: SSD | Medium |
| 6 | Primary Storage Type: HDD | Medium |
| 7 | Weight (kg) | Medium |
| 8 | CPU Frequency (GHz) | Medium |

---

## 📈 Visualizations

The notebook generates an 8-panel dashboard with a dark theme (`#0f1117` background):

![Laptop Price Predictor Dashboard](laptop_price_predictor.png)

| Panel | Chart Type | Description |
|-------|-----------|-------------|
| **(1,1)** Price Distribution | Histogram | Price distribution with median line |
| **(1,2)** Median Price by Brand | Horizontal bar | Top 8 manufacturers by median price |
| **(1,3)** RAM vs Median Price | Line + area fill | How RAM affects pricing |
| **(2,1)** CV R² Comparison | Horizontal bar + error bars | Side-by-side model comparison |
| **(2,2)** Actual vs Predicted | Scatter | With perfect-fit diagonal reference line |
| **(2,3)** Residuals | Scatter | Checks for systematic prediction bias |
| **(3,1-2)** Feature Importances | Horizontal bar | Top 15 Random Forest importances |
| **(3,3)** Price by CPU Series | Horizontal bar | Median price for each CPU tier |

Output saved to: `laptop_price_predictor.png`

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10+
- **Jupyter Notebook** or JupyterLab

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Running the Notebook

```bash
jupyter notebook laptop.ipynb
```

Then **Run All Cells** sequentially. The notebook will:

1. Install/verify dependencies (`scikit-learn`, `matplotlib`)
2. Load the CSV dataset and perform EDA
3. Engineer features from text columns (screen, memory, CPU)
4. Define numeric/categorical feature splits
5. Build a `ColumnTransformer` preprocessing pipeline
6. Compare 3 models via 5-fold cross-validation
7. Train the best model (Random Forest) on 80% data
8. Evaluate on held-out 20% test set
9. Compute and display feature importances
10. Generate and save an 8-panel visualization dashboard

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.13 | Runtime |
| **pandas** | — | Data loading, manipulation, & analysis |
| **NumPy** | 2.4+ | Numerical operations & array math |
| **scikit-learn** | 1.9+ | Preprocessing, pipelines, models, metrics, cross-validation |
| **matplotlib** | 3.11+ | Multi-panel visualization dashboard |
| **re** (stdlib) | — | Regex parsing for feature engineering |

---

## 🐛 Known Issues & Fixes Applied

The following bugs were identified and corrected (originals preserved in `laptop_backup.ipynb`):

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Feature importance showed `NaN` labels | `_short_name()` had `return name` unreachable inside an `if` block (after another `return`) | Moved `return name` to the `for`-loop level as a fallback |
| Dead code in `parse_cpu()` | `print(df)` placed after `return df` — never executes | Removed the unreachable `print(df)` statement |
| `FileNotFoundError` on savefig | Path `/mnt/user-data/outputs/` doesn't exist on Windows | Changed to local path `laptop_price_predictor.png` |

---

## 💡 Key Takeaways

- **Random Forest** outperforms linear models on this dataset (R² ~0.88 CV, ~0.84 test) — likely because it captures non-linear interactions between specs and price
- **RAM** is the single strongest price predictor, followed by laptop type and CPU series
- **Storage type** (SSD vs HDD) significantly impacts pricing — SSDs command a premium
- **Feature engineering** is critical — raw text columns like `ScreenResolution` and `Memory` must be parsed before any model can learn from them
- **Log-transforming** the right-skewed price distribution improves model stability and reduces outlier influence
- **PPI** (computed from resolution and screen size) is more informative than raw resolution alone

---

## 🔮 Possible Future Improvements

- [ ] Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
- [ ] Try gradient boosting models (XGBoost, LightGBM)
- [ ] Engineer GPU tier features similar to CPU series
- [ ] Add brand × category interaction features
- [ ] Deploy as a web app with Flask/Streamlit for live predictions
- [ ] Perform SHAP analysis for more interpretable feature importance

---

## 📄 License

This project is for educational and personal use.
