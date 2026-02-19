# Fraud Guard: Production-Grade Fraud Detection System

**10 Academy - Artificial Intelligence Mastery | Week 12 Capstone Project**

[![Fraud Detection CI](https://github.com/ruthasoll/Fraud-detection-week-5/actions/workflows/ci.yml/badge.svg)](https://github.com/ruthasoll/Fraud-detection-week-5/actions/workflows/ci.yml)

## Business Problem
Fraudulent transactions cause billions in losses annually for the finance and e-commerce sectors. Traditional rule-based systems often struggle with evolving fraud patterns and high false-positive rates. This project provides a robust, ML-driven solution designed to detect fraudulent behavior in real-time, reducing financial loss while maintaining a smooth user experience for legitimate customers.

## Solution Overview
Our approach leverages an **XGBoost Classifier** trained on engineered features derived from transaction patterns. Key innovations include:
- **Dynamic Velocity Tracking**: Monitors transaction frequency in rolling windows to detect "burst" attacks.
- **IP-to-Country Mapping**: Geospatial analysis to identify high-risk locations.
- **Explainable AI (SHAP)**: Provides transparency into model decisions for stakeholders.
- **Real-Time Monitoring**: Integrated drift detection to ensure model reliability in production.

## Key Results
- **PR-AUC**: 0.82 (High precision for fraud flagging).
- **F1-Score**: 0.78 (Balanced performance on imbalanced data).
- **Inference Latency**: <150ms (Scalable for real-time traffic).
- **Business Impact**: Potential to reduce fraud losses by an estimated 70% based on historical validation.

## Project Structure
```text
/
├── .github/workflows/    # CI/CD pipelines
├── api/                  # FastAPI inference service
├── src/                  # Modular source code
│   ├── data/            # Data loaders
│   ├── features/        # Feature engineering & stateful Feature Store
│   ├── models/          # Training pipelines & drift detection
│   └── utils/           # Configuration & simulations
├── dashboard/            # Streamlit real-time dashboard
├── tests/                # Automated unit tests
├── Dockerfile            # Container configuration
└── docker-compose.yml    # Service orchestration
```

## Quick Start
```bash
# 1. Clone & Install
git clone https://github.com/ruthasoll/Fraud-detection-week-5
cd Fraud-detection-week-5
pip install -r requirements.txt

# 2. Run Local Services
docker-compose up --build
```
*Access the API at http://localhost:8000 and the Dashboard at http://localhost:8501.*

## Technical Details
- **Data**: Historical fraud data enriched with location and time-based features.
- **Model**: Tuned XGBoost with SMOTE handling for class imbalance (1:11 ratio).
- **Evaluation**: Logged via MLflow, including Precision-Recall curves and SHAP importance.

## Future Improvements
- Implement **Redis** for distributed state management in the Feature Store.
- Add **Kafka** integration for asynchronous transaction processing.
- Expand monitoring to include **Model Performance Drift** with ground truth feedback loops.

## Author
**Ruth A.**
[LinkedIn](https://www.linkedin.com/in/yourprofile) | [GitHub](https://github.com/ruthasoll)
