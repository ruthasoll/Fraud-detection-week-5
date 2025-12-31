# Fraud-detection-week-5
# Fraud Detection System for E-Commerce and Banking Transactions

**10 Academy - Artificial Intelligence Mastery | Week 5 & 6 Challenge**  
**Date:** December 31, 2025  

## Project Overview

This repository contains a complete, end-to-end fraud detection system developed for **Adey Innovations Inc.**, a fintech company providing security solutions for e-commerce platforms and bank credit card transactions.

The goal is to accurately detect fraudulent transactions while minimizing false positives that could harm legitimate user experience. The project addresses the classic fraud detection challenges: severe class imbalance, subtle behavioral patterns, and the critical trade-off between security and customer friction.

The system was built across three main tasks:

### Task 1: Data Analysis and Preprocessing
- Loaded and cleaned two primary datasets:
  - `Fraud_Data.csv`: E-commerce transactions with user, device, and behavioral features.
  - `creditcard.csv`: Anonymized bank credit card transactions (PCA-transformed features).
- Performed thorough Exploratory Data Analysis (EDA) to uncover fraud patterns:
  - Higher purchase values, younger users, night-time activity, rapid signups.
  - Severe class imbalance (~9% fraud in e-commerce, ~0.17% in banking).
- Engineered high-signal features:
  - `time_since_signup_hours`: Captures "quick-strike" fraud from new accounts.
  - `hour_of_day` and `day_of_week`: Identifies off-hour and weekend anomalies.
  - `tx_count_last_24h`: Transaction velocity per user (rolling 24-hour window).
  - `country`: Mapped from IP address using range-based lookup.
- Applied data transformations (scaling, one-hot encoding) and prepared model-ready datasets.
- Demonstrated SMOTE for handling class imbalance (applied only on training data in later tasks).

### Task 2: Model Building and Training
- Implemented stratified train-test split to preserve real-world fraud ratios.
- Trained a **Logistic Regression** baseline for interpretability.
- Developed and tuned an **XGBoost** ensemble model using:
  - Memory-efficient SMOTE (sampling_strategy=0.5)
  - Grid search with stratified cross-validation
  - AUC-PR as primary metric (suitable for imbalanced data)
- Achieved strong performance:
  - XGBoost: **AUC-PR 0.78**, **F1-Score 0.74**, high precision with solid recall
  - Significantly outperformed baseline
- Saved best model and performance visualizations.

### Task 3: Model Explainability and Business Insights
- Applied **SHAP (SHapley Additive exPlanations)** to the best XGBoost model.
- Generated:
  - Global summary plot: Ranked feature importance across all predictions.
  - Individual force plots: Explained true positives, false positives, and false negatives.
- Identified top fraud drivers:
  1. High `purchase_value`
  2. Low `time_since_signup_hours` (new accounts)
  3. High transaction velocity
  4. Suspicious countries and timing
- Translated insights into **actionable business recommendations**:
  - Add friction (SMS/3D-Secure) for new accounts or high-velocity users.
  - Tiered verification based on risk score.

## Repository Structure
