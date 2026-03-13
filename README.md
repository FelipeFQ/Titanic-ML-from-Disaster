# Titanic — Survival Prediction

The Titanic dataset is one of the most analyzed datasets in data science — which made me want to approach it differently. Rather than just optimizing accuracy, I used it to investigate a historical question: **when the ship sank, what structural factors determined who got a lifeboat?**

The answer is surprisingly consistent: gender, passenger class, and family structure explain most of the variance. The ML model ends up being a formalization of patterns that are already visible in the raw data. That tension between "simple rules" and "complex model" is one of the more interesting things to unpack here.

---

## Project Structure

```
├── data/
│   ├── train.csv                  # Raw training data (891 passengers)
│   ├── test.csv                   # Raw test data (418 passengers)
│   ├── gender_submission.csv      # Kaggle baseline
│   ├── train_engineered.csv       # Output of 03_feature_engineering.ipynb
│   └── test_engineered.csv        # Output of 03_feature_engineering.ipynb
│
├── notebooks/
│   ├── 01_data_overview.ipynb     # Data types, missing values, initial observations
│   ├── 02_eda.ipynb               # Hypothesis-driven exploration of survival patterns
│   ├── 03_feature_engineering.ipynb  # All feature creation (separated from modeling)
│   └── 04_modeling.ipynb          # Model benchmark, tuning, SHAP interpretation
│
├── outputs/                       # Submission files
├── requirements.txt
├── titanic_environment.yml        # Conda environment for reproducibility
└── .gitignore
```

---

## Workflow

### 01 — Data Overview
Load train/test, inspect types, quantify missing values, identify anomalies.
Key finding: Cabin is missing for 77% of rows — not randomly, but because passengers with no cabin record were predominantly 3rd class. The missingness is itself a signal.

### 02 — EDA (Hypothesis-driven)
I started with four hypotheses and tested each against the data:
- **H1 — Women and children first** ✅ Female survival: 74.2% vs male: 18.9%
- **H2 — Class determines access** ✅ 1st: 62.9%, 2nd: 47.3%, 3rd: 24.2%
- **H3 — Family structure matters** ✅ U-shaped: solo 30%, small families 55–61%, large families 16%
- **H4 — Fare proxies class** ✅ Survivors paid ~2.5× more at median; right-skewed, log transform helps

Includes interaction plots: Sex × Pclass heatmap, IsAlone × Sex, AgeGroup × Pclass.

### 03 — Feature Engineering
All feature creation in one place, exported to `train_engineered.csv` and `test_engineered.csv`.

Standard features (EDA-motivated):
- `Title` — extracted from Name; encodes gender + social status in one signal
- `FamilySize`, `IsAlone` — family size and solo-traveler flag
- `Deck` — first letter of Cabin; missing → 'U' (Unknown)
- `Fare_log1p` — log transform of Fare to reduce skew

Personal additions:
- `WomanOrChild` — explicitly encodes the "women and children first" protocol rather than letting the model infer it from Sex + Age
- `FarePerPerson` — normalizes fare by family size (group tickets split the cost)
- `TicketGroupSize` — counts passengers sharing the same ticket number (reveals travel companions beyond declared family)
- `AgeGroup` — binned Age into life stages (Child, Teen, YoungAdult, Adult, Senior)

### 04 — Modeling
- 6 models benchmarked: Logistic Regression, Random Forest, Gradient Boosting, HistGB, XGBoost, LightGBM
- 3 evaluation metrics: Accuracy, ROC-AUC, F1 (accuracy alone is misleading on imbalanced classes)
- XGBoost hyperparameter tuning with RandomizedSearchCV (30 iterations)
- **SHAP analysis** for model interpretation — explains individual predictions and global feature importance

---

## Results

Best CV accuracy: ~0.848 (XGBoost tuned)
Best model by ROC-AUC: see comparison table in `04_modeling.ipynb`

Top features by SHAP:
1. `Title_Mr` — single strongest predictor (encodes sex + adulthood + social status)
2. `Sex_female` — direct gender signal
3. `Pclass` — socioeconomic gradient
4. `WomanOrChild` — the explicit protocol encoding

Something that surprised me: `Title_Mr` consistently outranks raw `Sex` in importance. It makes sense in retrospect — it packs more information into a single feature.

---

## Setup

```bash
conda env create -f titanic_environment.yml
conda activate titanic
jupyter lab
```

Or with pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap
```

---

## Skills demonstrated

- Hypothesis-driven EDA (not just plotting everything)
- Feature engineering motivated by domain knowledge (historical rescue protocol)
- Leakage-safe preprocessing pipelines with scikit-learn
- Multi-metric model evaluation (Accuracy + ROC-AUC + F1)
- Model interpretation with SHAP

---

## Kaggle competition

[Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
