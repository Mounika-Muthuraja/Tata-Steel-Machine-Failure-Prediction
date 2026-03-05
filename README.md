# TATA Steel Predictive Maintenance

## Overview

Developed a machine learning model to predict equipment failures using historical manufacturing data.
The goal was to detect breakdown signals early and support preventive maintenance decisions.

The final model achieved **91.4% recall**, identifying 393 out of 430 actual failures during validation.

## Dataset

* 136K+ training records
* 90K+ test records
* 10 operational variables
* 1.57% failure rate (high class imbalance)

Key inputs: temperature, torque, rotational speed, tool wear, product type.

## Approach

* Performed exploratory data analysis to examine failure distribution and operational thresholds
* Engineered domain features such as temperature difference, power, and stress interaction
* Applied SMOTE to handle imbalance
* Compared Logistic Regression, Random Forest, and XGBoost
* Selected XGBoost based on highest failure detection rate

## Results

* 91.4% failure detection rate
* 37 missed failures during validation
* Tool wear above 200 minutes linked to ~6× higher breakdown probability
* Rotational speed and torque identified as primary predictive factors

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib

This project demonstrates practical experience in:

* Handling imbalanced datasets
* Feature engineering
* Model evaluation using recall-focused metrics
* Statistical validation of operational drivers
