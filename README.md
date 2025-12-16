# Animal Shelter Adoption Prediction – Machine Learning Project
This project applies multiple machine learning models to predict dog adoption outcomes at an animal shelter. Using historical shelter intake and outcome data, the pipeline explores how animal characteristics, intake conditions, and temporal features influence the likelihood of adoption.
The project is motivated by real-world challenges faced by shelters, including overcrowding, limited resources, and the need to improve adoption rates.

# Objectives

* Predict whether a dog will be adopted (binary classification)
* Compare traditional and advanced machine learning models
* Analyze feature importance and adoption drivers
* Balance predictive performance with interpretability

# Models Implemented

# 1. Logistic Regression

* Baseline logistic regression (no regularization)
* Lasso (L1) Logistic Regression** with hyperparameter tuning
* 10-fold cross-validation
* Feature selection and coefficient interpretation

# 2. Support Vector Machines (SVM)

* Linear kernel
* Radial Basis Function (RBF) kernel
* Polynomial kernel
* Performance comparison across kernels
* Training time analysis

# 3. Neural Networks (PyTorch)

* Simple Neural Network: 1 hidden layer (Sigmoid activation)
* Deeper Neural Network: 3 hidden layers (ReLU activation)
* Binary cross-entropy loss
* Adam optimizer
* Training curves and confusion matrices

# Dataset

* Source: Multiple CSV files stored in Google Drive, from San Jose Animal Care Center
* Filtered to include dogs only
* Records ineligible for adoption removed (e.g., wildlife, deceased animals)

# Key Features

* Age (parsed into years)
* Length of stay (days)
* Breed (top breeds grouped, rare breeds merged)
* Color (top colors grouped)
* Sex
* Intake type and condition
* Season of intake
* HasName (binary feature)

Categorical variables are one-hot encoded.

# Data Processing Pipeline

1. Load and concatenate multiple CSV files
2. Drop irrelevant or administrative columns
3. Handle missing values
4. Parse age strings into numeric values
5. Feature engineering (season, has-name indicator)
6. One-hot encoding
7. Train-test split (70/30)
8. Feature scaling where appropriate



# Exploratory Data Analysis (EDA)

* Adoption rate distribution
* Length of stay vs. adoption outcome
* Age vs. adoption outcome
* Breed-level adoption rates
* Intake type adoption analysis
* Correlation heatmap of numeric features

# Evaluation Metrics

All models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

Additional evaluations include:

* Confusion matrices (raw and normalized)
* Cross-validation mean ± standard error
* Training time (for SVMs)


# Key Results

* Logistic regression provides strong interpretability and stable performance
* Lasso regularization reduces feature dimensionality while maintaining F1-score
* RBF SVM generally outperforms linear and polynomial kernels
* Neural networks achieve high training accuracy but show signs of overfitting
* ReLU-based deep networks outperform sigmoid-based shallow networks

# Requirements

* Python 3.9+
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* torch
* tqdm

This project was developed and ran in Google Colab

# How to Run

1. Upload dataset CSV files to Google Drive
2. Update the dataset folder path if necessary
3. Open the notebook/script in Google Colab
4. Run all cells sequentially

# Author

Raymond Rico

For academic and educational use.
