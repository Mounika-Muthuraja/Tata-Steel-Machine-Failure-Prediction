# TATA Steel Predictive Maintenance Project

## 📋 Overview

This project builds a machine learning–based predictive maintenance model to estimate machine failure probability using historical operational data.

The model analyzes temperature, torque, rotational speed, and tool wear to identify breakdown patterns and support preventive maintenance planning.

The final XGBoost model achieved **91.4% recall**, detecting 393 out of 430 actual failures during validation.

## 🎯 Objective

Manufacturing environments face operational disruption due to unexpected equipment breakdowns.

This project aims to:

* Identify failure-prone operating conditions
* Detect breakdown signals early
* Evaluate model performance under severe class imbalance
* Provide data-backed maintenance thresholds

## 📊 Dataset

**Training Data:** 136,429 records
**Test Data:** 90,954 records
**Features:** 10 core operational variables
**Failure Rate:** ~1.57% (highly imbalanced)

### Key Variables

* Air temperature [K]
* Process temperature [K]
* Rotational speed [rpm]
* Torque [Nm]
* Tool wear [min]
* Product Type (L, M, H)
* Machine Failure (Target)

## 🧠 Methodology

### 1️⃣ Exploratory Analysis

* Measured overall failure rate (1.57%)
* Identified tool wear above 200 minutes as a high-risk zone
* Observed strong negative correlation (-0.779) between torque and rotational speed
* Found Heat Dissipation Failure as the most frequent failure category

### 2️⃣ Feature Engineering

Created domain-based features:

| Feature             | Formula            | Purpose             |
| ------------------- | ------------------ | ------------------- |
| Temp_Difference     | Process − Air Temp | Heat imbalance      |
| Power               | Torque × Speed     | Mechanical stress   |
| Stress_Product      | Tool wear × Torque | Overstrain signal   |
| Rotational_speed_sq | Speed²             | Non-linear modeling |

### 3️⃣ Data Preparation

* Removed identifier columns
* Encoded product category
* Applied StandardScaler
* Used SMOTE to address 62:1 class imbalance
* Performed 80–20 stratified split

### 4️⃣ Model Comparison

| Model               | Recall     | Precision | F1     |
| ------------------- | ---------- | --------- | ------ |
| **XGBoost**         | **91.40%** | 2.68%     | 5.21%  |
| Random Forest       | 76.51%     | 12.20%    | 21.04% |
| Logistic Regression | 78.37%     | 7.00%     | 12.85% |

XGBoost was selected due to highest failure detection rate.

## 📈 Results

### Validation Performance

* **393 failures detected**
* **37 failures missed**
* **91.4% recall**
* 8.6% miss rate

## 🔍 Key Findings

* Tool wear >200 minutes increases breakdown probability ~6×
* Lower temperature difference contributes to heat-related failure
* Rotational speed and torque are strongest predictive variables
* Failure occurrence differs across product categories

All major relationships were validated using:

* Chi-square test
* Mann-Whitney U test
  (p < 0.01)


## 🛠️ Technologies Used

| Category           | Tools                 |
| ------------------ | --------------------- |
| Programming        | Python                |
| Data Handling      | Pandas, NumPy         |
| Visualization      | Matplotlib, Seaborn   |
| Modeling           | Scikit-learn, XGBoost |
| Imbalance Handling | SMOTE                 |
| Model Saving       | Joblib                |


## 📌 Future Improvements

* Multi-class prediction for specific failure types
* Real-time monitoring dashboard
* Cost-based evaluation metrics
* Automated retraining pipeline

⭐ This project demonstrates end-to-end machine learning workflow including data preprocessing, feature engineering, model selection, statistical testing, and validation under class imbalance conditions.


* Or convert this into 3 strong resume bullet points aligned with this README 🚀
