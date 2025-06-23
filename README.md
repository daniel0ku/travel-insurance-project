# Travel Insurance Prediction
This repository contains a machine learning project focused on predicting whether customers will purchase travel insurance based on various demographic and travel-related features. The analysis includes data preparation, exploratory data analysis (EDA), hypothesis testing, model fitting, and evaluation using machine learning techniques.

## Project Overview
The project aims to analyze the factors influencing the purchase of travel insurance. By building and evaluating predictive models, the goal is to accurately classify customers into those who are likely to buy travel insurance and those who are not.

## Goal
The primary objective of this project is to develop a predictive model that can accurately determine whether a customer will purchase travel insurance based on their demographic and travel history data.

## Hypotheses

H0: There is no significant difference in the likelihood of purchasing travel insurance across different income groups.

H1: Individuals in higher income groups are significantly more likely to purchase travel insurance compared to those in lower income groups.

H0: Age group has no significant impact on the likelihood of purchasing travel insurance.

H1: Older age groups are significantly more likely to purchase travel insurance compared to younger age groups.

H0: Family size has no significant effect on the likelihood of purchasing travel insurance.

H1: Larger family sizes are significantly more likely to purchase travel insurance compared to smaller family sizes.

## Insights

Features such as ever_travelled_abroad, income_group, family_size, and age_group were identified as significant predictors of travel insurance purchase.

The Chi-square test indicated that features like chronic_diseases and graduated_or_not are insignificant, contributing minimally to the model’s performance.

The tuned Random Forest model achieved a test accuracy of 75.87%, slightly higher than the base model, demonstrating the importance of feature selection and tuning.

## Findings

The decision tree visualization provided insights into how different features influence the prediction, with "ever_travelled_abroad" being a key decision node.
Cross-validation results showed consistent model performance, with a mean accuracy of approximately 77.50%, indicating that the model generalizes well across different subsets of the data.
The Random Forest and Decision Tree models performed similarly, suggesting that while the Random Forest model offers more stability, the Decision Tree model is competitive in terms of accuracy.

## Suggestions for Improvement

Feature Engineering: Introduce interaction terms and derived features to capture more complex relationships in the data.

Model Comparison: Experiment with additional algorithms, such as Gradient Boosting Machines (GBM) or XGBoost, to explore potential performance gains.

Data Enrichment: Consider augmenting the dataset with additional demographic or behavioral data to enhance model accuracy.

## Conclusion

The project successfully built and evaluated models to predict travel insurance purchases. The findings support the hypotheses that travel history is a strong predictor of insurance purchase and that hyperparameter tuning can enhance model performance. While the models performed well, further improvements can be made by exploring additional features, algorithms, and explainability techniques.

## Needed Libraries:
* import inflection
* import matplotlib.pyplot as plt
* import numpy as np
* import pandas as pd
* import seaborn as sns
* from scipy.stats import ttest_ind
* from scipy.stats import chi2_contingency
* from sklearn.ensemble import RandomForestClassifier
* from sklearn.metrics import accuracy_score, classification_report
* from sklearn.model_selection import KFold, cross_val_score
* from sklearn.model_selection import train_test_split
* from sklearn.model_selection import GridSearchCV
* from sklearn.tree import DecisionTreeClassifier
## Author Information
Name: Daniel Kurmel

Email: danielkurmel@gmail.com

Project Link: GitHub Repository

## Files Included in this project:
* Travel Insurance Prediction.ipynb
* README.md
* Data/ TravelInsurancePrediction.csv
* venv
## Dataset Source:
https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data