# 📚 Predicting Author Ratings from Book Sales Data

This project uses machine learning to predict the **Author Rating** based on features related to books and their commercial performance. It leverages advanced preprocessing, feature engineering, and classification models to build a reliable predictive pipeline.

---

## 🔍 Project Overview

The dataset includes the following features for each book:

- **Book Name**
- **Author**
- **Language Code**
- **Genre**
- **Book Average Rating**
- **Book Ratings Count**
- **Gross Sales**
- **Publisher Revenue**
- **Sale Price**
- **Sales Rank**
- **Units Sold**
- **Author Rating** *(Target Variable)*

The goal of this project is to predict the **Author Rating**, which serves as a proxy for an author's success, using available book data.

---

## 🛠️ Tools & Technologies

- **Python 3**
- **Jupyter Notebook**
- **Pandas / NumPy** for data manipulation
- **Scikit-learn** for machine learning
- **Matplotlib / Seaborn** for visualization
- **MLxtend** for additional ML utilities

---

## 🧪 Methodology

The pipeline involves:

1. **Data Preprocessing**
   - Handling missing values with `SimpleImputer` and `KNNImputer`
   - Encoding categorical features with `OneHotEncoder` and `OrdinalEncoder`
   - Feature scaling with `StandardScaler`

2. **Feature Engineering**
   - PCA and LDA for dimensionality reduction
   - Custom feature transformations

3. **Model Selection**
   - Evaluated models include:
     - Logistic Regression
     - K-Nearest Neighbors
     - Random Forest
     - Perceptron
   - Hyperparameter tuning via `GridSearchCV` and `RandomizedSearchCV`

4. **Evaluation Metrics**
   - Accuracy
   - F1 Score
   - ROC AUC

---

## 📊 Results

Model performance is assessed using cross-validation, with metrics tracked and compared to select the most robust model for deployment.
