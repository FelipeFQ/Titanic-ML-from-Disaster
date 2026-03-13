# 🚢 Titanic — Survival Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.3.2-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

The Titanic dataset is one of the most analyzed in data science — which made me want to approach it differently. Rather than just optimizing accuracy, I framed it as a sociological investigation: **when the ship sank, what structural factors determined who got a lifeboat?**

The answer is surprisingly consistent: gender, passenger class, and family structure explain most of the variance. The ML model ends up being a formalization of patterns already visible in the raw data — and that tension between "simple rules" and "complex model" is one of the more interesting things to unpack here.

---

## 🎯 Objective

Predict binary survival (`Survived = 1 / 0`) for the 418 test passengers. The goal isn't just to maximize Kaggle accuracy — it's to build a model that is interpretable, leakage-free, and backed by domain knowledge from the historical event.

---

## ⚙️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.13 |
| Data manipulation | pandas 2.3, NumPy 2.1 |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Notebooks | Jupyter |

---

## 🔄 Pipeline Overview

```
data/
├── train.csv  (891 rows)
└── test.csv   (418 rows)
        │
        ▼
┌─────────────────────────────────────────────┐
│  01_data_overview.ipynb                     │
│  Data types, missing values, distributions  │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  02_eda.ipynb                               │
│  4 hypotheses tested: gender, class,        │
│  family structure, fare as proxy            │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  03_feature_engineering.ipynb               │
│  Title, FamilySize, WomanOrChild,           │
│  FarePerPerson, TicketGroupSize, AgeGroup   │
├─────────────────────────────────────────────┤
│  → train_engineered.csv    891 × 19 cols    │
│  → test_engineered.csv     418 × 18 cols    │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  04_modeling.ipynb                          │
│  6-model benchmark, XGBoost tuning,         │
│  SHAP interpretation, Kaggle submission     │
├─────────────────────────────────────────────┤
│  → submission_best_cv.csv                   │
│  → submission_tuned_xgb.csv                 │
└─────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
Titanic_Project/
├── data/
│   ├── train.csv                     # Raw training data (891 passengers)
│   ├── test.csv                      # Raw test data (418 passengers)
│   ├── gender_submission.csv         # Kaggle baseline
│   ├── train_engineered.csv          # Output of 03_feature_engineering.ipynb
│   └── test_engineered.csv           # Output of 03_feature_engineering.ipynb
│
├── notebooks/
│   ├── 01_data_overview.ipynb        ✅ Complete
│   ├── 02_eda.ipynb                  ✅ Complete
│   ├── 03_feature_engineering.ipynb  ✅ Complete
│   └── 04_modeling.ipynb             ✅ Complete
│
├── outputs/
│   ├── submission_best_cv.csv        # Best model by CV ROC-AUC
│   └── submission_tuned_xgb.csv      # Tuned XGBoost
│
├── requirements.txt
├── titanic_environment.yml           # Conda environment
└── .gitignore
```

---

## 📓 Notebooks

### 🔍 01 — Data Overview

**Goal:** Understand the raw data before touching anything — types, missing values, class distribution, and initial anomalies.

- **Input:** `train.csv`, `test.csv`
- **Output:** observations only — no transformations

| Column | Missing (train) | Note |
|--------|-----------------|------|
| `Age` | 19.9% | Missing non-randomly — more frequent in 3rd class |
| `Cabin` | 77.1% | Missingness is itself a class signal |
| `Embarked` | 0.2% | Two rows only |

**Key observation:** Cabin's missingness rate (77%) is not random — passengers without a cabin record were predominantly 3rd class. The missing data encodes socioeconomic status before we engineer a single feature.

---

### 📊 02 — EDA (Hypothesis-driven)

**Goal:** Test four sociological hypotheses against the data before building any features.

- **Input:** `train.csv`
- **Output:** observations + visualizations (no file export)

| Hypothesis | Claim | Result | Evidence |
|------------|-------|--------|----------|
| H1 — Women and children first | Gender drives survival | ✅ Confirmed | Female: 74.2% vs Male: 18.9% |
| H2 — Class determines access | Pclass determines lifeboat access | ✅ Confirmed | 1st: 62.9%, 2nd: 47.3%, 3rd: 24.2% |
| H3 — Family structure matters | Traveling alone or in large groups hurts | ✅ Confirmed | Solo: 30%, small family: 55–61%, large: 16% |
| H4 — Fare proxies class | Higher fare = better odds | ✅ Confirmed | Survivors paid ~2.5× more at median |

**Interaction plots included:** Sex × Pclass heatmap, IsAlone × Sex, AgeGroup × Pclass.

---

### ⚙️ 03 — Feature Engineering

**Goal:** Create all features in a single, reproducible function applied identically to train and test. Export the engineered datasets for the modeling notebook.

- **Input:** `train.csv`, `test.csv`
- **Output:** `train_engineered.csv`, `test_engineered.csv`

**Standard features** (EDA-motivated):

| Feature | Source | Reasoning |
|---------|--------|-----------|
| `Title` | `Name` regex | Encodes gender + social status + life stage in one signal |
| `FamilySize` | `SibSp + Parch + 1` | U-shaped survival pattern — more useful than two separate columns |
| `IsAlone` | `FamilySize == 1` | Solo travelers had a distinct 30% survival rate |
| `Deck` | First letter of `Cabin` | Proxy for cabin location; 'U' for unknown |
| `Fare_log1p` | `log(1 + Fare)` | Compresses right skew; handles zero fares |

**Personal additions** (domain knowledge):

| Feature | Logic | Why it matters |
|---------|-------|----------------|
| `WomanOrChild` | `Sex == female OR Age < 15` | Explicit encoding of the "women and children first" rescue protocol |
| `FarePerPerson` | `Fare / FamilySize` | Group tickets split the cost — raw Fare overstates individual wealth |
| `TicketGroupSize` | `groupby(Ticket).transform('count')` | Captures travel companions beyond declared family |
| `AgeGroup` | `pd.cut()` into 5 life stages | Age-survival relationship is non-linear; bins make it explicit |

---

### 🤖 04 — Modeling

**Goal:** Benchmark multiple models on the engineered features, tune the best one, interpret predictions with SHAP, and generate Kaggle submissions.

- **Input:** `train_engineered.csv`, `test_engineered.csv`
- **Output:** `submission_best_cv.csv`, `submission_tuned_xgb.csv`

**Design decisions:**
- scikit-learn `Pipeline` + `ColumnTransformer` for leakage-safe preprocessing
- `StratifiedKFold(n_splits=5)` to preserve the 62/38 class ratio in every fold
- `RandomizedSearchCV(n_iter=30, scoring='roc_auc')` for XGBoost tuning
- Two submission files — best CV model and tuned XGBoost — to compare leaderboard vs CV performance

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Best CV Accuracy | ~0.848 (XGBoost tuned) |
| Evaluation approach | Accuracy + ROC-AUC + F1 (accuracy alone is misleading on imbalanced classes) |

**🏆 Top features by SHAP importance:**

| Rank | Feature | Why |
|------|---------|-----|
| 1 | `Title_Mr` | Packs sex + adulthood + social status into one signal |
| 2 | `Sex_female` | Direct gender effect |
| 3 | `Pclass` | Socioeconomic gradient |
| 4 | `WomanOrChild` | Explicit rescue protocol encoding |

> Something that surprised me: `Title_Mr` consistently outranks raw `Sex_female` in SHAP importance. It makes sense in retrospect — it encodes being male AND adult AND not having a special honorific, all in a single value. A model using raw `Sex` and `Age` separately needs multiple splits to reconstruct this combined signal.

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/FelipeFQ/Titanic-ML-from-Disaster.git
cd Titanic-ML-from-Disaster
```

**2. Create the environment**
```bash
conda env create -f titanic_environment.yml
conda activate titanic
```

Or with pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap
```

**3. Run notebooks in order**
```bash
jupyter lab
```

Each notebook exports its output before the next one reads it. Start with `01_data_overview.ipynb` and follow the sequence through `04_modeling.ipynb`.

---

## 🏅 Kaggle Competition

[Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
