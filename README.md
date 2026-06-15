# 🤖 CODSOFT Data science Internship

> A collection of production-style machine learning projects built during the CODSOFT  Internship — covering supervised learning, regression, and classification across real-world datasets.

---

## 👤 Intern Details

| Field | Details |
|---|---|
| **Name** | Dinesh |
| **Domain** | Data Science |
| **Organization** | CODSOFT |
| **Tech Stack** | Python · scikit-learn · pandas · matplotlib · seaborn |
| **Environment** | VS Code · Python 3.14 · Windows |

---

## 📁 Repository Structure

```
CODSOFT/
│
├── IRIS-FLOWER-CLASSIFICATION/
│   ├── images/                        # Visualization outputs
│   ├── iris_flowerclassification.py   # Main pipeline script
│   └── IRIS.csv                       # Dataset
│
├── MOVIE-RATING-PREDICTION/
│   ├── actual_vs_predicted.png        # Regression output plot
│   ├── eda_plots.png                  # Exploratory data analysis
│   ├── feature_importance.png         # RandomForest feature importance
│   ├── movie-rating-prediction.py     # Main pipeline script
│   └── Movies_Information_India.csv   # IMDb India movies dataset
│
└── TITANIC-SURVIVAL-PREDICTION/
    ├── IMAGES/                        # Visualization outputs
    ├── titanic-survival-prediction.py # Main pipeline script
    └── titanic.csv                    # Titanic passenger dataset
```

---

## 🚀 Projects

### 1. 🌸 Iris Flower Classification

**Objective:** Classify iris flowers into *Setosa*, *Versicolor*, and *Virginica* based on sepal and petal measurements.

**Dataset:** Classic Fisher Iris dataset — 150 samples, 4 features, 3 classes

**Approach:**
- Performed EDA to visualize class separability
- Applied feature scaling via `StandardScaler`
- Trained multiple classifiers and compared accuracy
- Evaluated using confusion matrix and classification report

**Key Techniques:** Multi-class classification · Feature scaling · Model evaluation

---

### 2. 🎬 Movie Rating Prediction

**Objective:** Predict IMDb ratings for Indian movies using metadata like genre, director, and cast.

**Dataset:** `Movies_Information_India.csv` — scraped IMDb India data with real-world noise (missing values, mixed encodings)

**Approach:**
- Parsed `latin-1` encoded CSV with encoding-safe `pd.read_csv()`
- Imputed missing values using `SimpleImputer`
- Encoded categorical features (`Genre`, `Director`, `Actor 1/2/3`) via `LabelEncoder`
- Trained `RandomForestRegressor` and `GradientBoostingRegressor` inside an sklearn `Pipeline`
- Evaluated on `RMSE`, `MAE`, and `R²`
- Visualized actual vs predicted ratings and feature importances

**Key Techniques:** Regression · Ensemble methods · Pipeline architecture · Feature encoding · EDA

**Outputs:**
| Plot | Description |
|---|---|
| `eda_plots.png` | Rating distribution, genre analysis, vote correlation |
| `actual_vs_predicted.png` | Scatter of true vs model-predicted ratings |
| `feature_importance.png` | Top drivers of rating according to RandomForest |

---

### 3. 🚢 Titanic Survival Prediction

**Objective:** Predict passenger survival on the Titanic using demographic and ticket features.

**Dataset:** Classic Titanic dataset — 891 training samples, binary classification target (`Survived`)

**Approach:**
- Handled missing data in `Age`, `Cabin`, and `Embarked`
- Engineered features from `Name` (title extraction), `SibSp`/`Parch` (family size)
- Encoded categorical columns and scaled numerical features
- Trained classification models and tuned for precision/recall balance
- Visualized survival distributions across class, gender, and age

**Key Techniques:** Binary classification · Feature engineering · Class imbalance handling · EDA

---

## ⚙️ Setup & Usage

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running any project

```bash
# Navigate to the project folder
cd MOVIE-RATING-PREDICTION

# Run the script
python movie-rating-prediction.py
```

**> ⚠️ Always run from inside the project's own folder so relative CSV paths resolve correctly.
**
---

## 📊 Results Summary

| Project | Task | Algorithm | Key Metric |
|---|---|---|---|
| Iris Classification | Multi-class | Random Forest / KNN | Accuracy ~97% |
| Movie Rating Prediction | Regression | RandomForestRegressor | R² > 0.85 |
| Titanic Survival | Binary Classification | Logistic Regression / RF | Accuracy ~82% |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, pipelines, preprocessing, metrics |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualizations |

---

## 📌 Internship Completion

This repository was built as part of the **CODSOFT Data Science Internship** task series.

Each project follows a consistent structure:
1. Data loading & exploratory analysis
2. Preprocessing pipeline (imputation → encoding → scaling)
3. Model training & cross-validation
4. Evaluation with appropriate metrics
5. Visualization of results

---

## 📬 Contact

**Dinesh**
B.Tech Computer Science Engineering — 3rd Year
Hyderabad, Telangana, India



---

<p align="center">Built with 🧠 during CODSOFT Internship · </p>
