# 🚢 Titanic - Machine Learning from Disaster

A complete end-to-end **data science project** solving the classic Kaggle Titanic survival prediction challenge.  
This repository demonstrates **data exploration, feature engineering, and machine learning modeling**, wrapped in a reproducible environment for transparency and reusability.

---

## 📂 Project Structure

```
├── data/
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   ├── gender_submission.csv  # Sample submission file
│
├── notebooks/
│   ├── data_overview.ipynb    # Dataset understanding and inspection
│   ├── eda.ipynb              # Exploratory data analysis and feature insights
│   ├── modeling.ipynb         # Model building, evaluation, and predictions
│
├── outputs/                   #  Saved models, submission files
│
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python package dependencies
├── titanic_environment.yml   # Conda environment for reproducibility
└── README.md                 # Project documentation
```

---

## 🎯 Objective

The goal of this project is to predict **which passengers survived** the Titanic shipwreck based on features such as age, sex, socio-economic class, and family relations.  

This problem is a great introduction to:

- Data cleaning and preprocessing  
- Feature engineering and selection  
- Model comparison and evaluation  
- Reproducible ML workflows  

---

## 🔍 Workflow

### 1. Data Overview (`data_overview.ipynb`)
- Import and inspect datasets (`train.csv`, `test.csv`, `gender_submission.csv`).  
- Check missing values, data types, and basic distributions.  

### 2. Exploratory Data Analysis (`eda.ipynb`)
- Visual exploration of key survival patterns.  
- Feature relationships with target variable (`Survived`).  
- Handling missing values (e.g., `Age`, `Cabin`, `Embarked`).  
- Transformations and encoding (categorical → numeric).  

### 3. Modeling (`modeling.ipynb`)
- Train/test split and preprocessing pipelines.  
- Baseline models (Logistic Regression, Decision Trees).  
- Advanced models (Random Forest, Gradient Boosting, XGBoost, LightGBM).  
- Hyperparameter tuning and cross-validation.  
- Feature importance visualization.  
- Submission file creation for Kaggle.  

---

## ⚙️ Environment & Dependencies

All dependencies are captured in [`titanic_environment.yml`](./titanic_environment.yml).  

### Key Libraries
- **Data analysis**: `pandas`, `numpy`  
- **Visualization**: `matplotlib`, `seaborn`  
- **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`, `shap`  
- **Notebooks**: `jupyterlab`, `ipykernel`

### 🔧 Setup

Create the conda environment:

```bash
conda env create -f titanic_environment.yml
conda activate titanic
```

Or install requirements manually (example):

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap
```

---

## 📊 Results

- Models achieve competitive accuracy scores (~0.77) on validation.  
- Ensemble approaches (LightGBM/XGBoost) outperform simpler baselines.  
- Feature importance shows **Sex, Pclass, Age, Fare** as key survival indicators.  
- Final predictions are exported in Kaggle-compatible format (`submission.csv`).  

---

## ✨ Skills Demonstrated

- **Data Wrangling**: Handling missing data, feature engineering.  
- **Visualization**: Identifying insights with plots and distributions.  
- **ML Modeling**: Building, tuning, and comparing models.  
- **Reproducibility**: Using `conda` environments and structured notebooks.  
- **Communication**: Documenting workflow and presenting results clearly.  

---

## 📌 Next Steps

- Try deep learning approaches (e.g., neural networks).  
- Explore ensemble stacking for improved performance.  
- Automate pipeline with `scikit-learn`’s `Pipeline` or `mlflow`.  

---

## 🏆 Kaggle Competition Link
👉 [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
