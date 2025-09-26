# 🫀 Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/Pandas-1.5.3-lightblue?logo=pandas)
![numpy](https://img.shields.io/badge/NumPy-1.24.0-yellow?logo=numpy)
![matplotlib](https://img.shields.io/badge/Matplotlib-3.6.2-green?logo=plotly)
![seaborn](https://img.shields.io/badge/Seaborn-0.12.2-teal)


This project builds a heart disease prediction system using the UCI Heart Disease dataset. It includes data preprocessing, exploratory analysis, PCA for dimensionality reduction, supervised learning (Logistic Regression, KNN, SVM, Decision Trees, Random Forest, Gradient Boosting) with hyperparameter tuning, and clustering (K-Means & Hierarchical). The best-performing model (SVM) is selected via GridSearchCV and deployed as a FastAPI service, allowing real-time predictions.

# 🚀 Project Workflow

### 1️⃣ Data Preprocessing

• Handled missing values and categorical encodings

• Standardized features (important for ML models and PCA)

• Performed dimensionality reduction with PCA to retain variance while reducing noise

### 2️⃣ Exploratory Data Analysis (EDA)

• Visualized feature distributions and correlations

• Analyzed important medical features such as chest pain type, blood pressure, cholesterol, and ST segment slope

• Compared class distributions (healthy vs. heart disease)

### 3️⃣ Supervised Learning (Classification Models)

• Trained and compared multiple models:

      • Logistic Regression
      
      • K-Nearest Neighbors (KNN)
      
      • Support Vector Machine (SVM)
      
      • Decision Tree
      
      • Random Forest
      
      • Gradient Boosting

###### Each model was evaluated on training vs testing accuracy to check for overfitting or underfitting.

### 4️⃣ Hyperparameter Tuning

• Applied GridSearchCV with cross-validation for each model

• Identified the best parameters and highest accuracy model

• Compared tuned model performances

### 5️⃣ Clustering (Unsupervised Learning)

• Applied K-Means Clustering (with elbow method)

• Applied Hierarchical Clustering (with dendrogram analysis)

• Compared clustering results against true disease labels using Adjusted Rand Index and Silhouette Score

### 6️⃣ Deployment

• Exported the best tuned model using joblib (.pkl format)

• Built a FastAPI service for real-time heart disease prediction

• API returns:

      • Predicted label (0 = No Disease, 1 = Disease)
      
      • Probability of heart disease
      
      • Best model name & accuracy score
