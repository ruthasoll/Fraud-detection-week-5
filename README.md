# Fraud Guard: Improved Fraud Detection System

**10 Academy - Artificial Intelligence Mastery | Week 12 Capstone Challenge**

[![Fraud Guard CI](https://github.com/ruthasoll/Fraud-detection-week-5/actions/workflows/ci.yml/badge.svg)](https://github.com/ruthasoll/Fraud-detection-week-5/actions/workflows/ci.yml)

## Business Problem
Fraudulent transactions cause billions in losses annually for e-commerce and banking sectors. This system provides a **production-grade, real-time guard** to identify and block fraud while minimizing friction for legitimate users.

## Key Improvements (Week 12)
- **Modular Engineering**: Refactored notebook logic into a clean, testable `/src` package.
- **MLOps Integration**: Full experiment tracking with **MLflow** and feature/model versioning.
- **Real-Time Readiness**:
  - **FastAPI** inference endpoint with <150ms latency.
  - **In-Memory Feature Store** for real-time transaction velocity (sliding windows).
  - **Transaction Simulator** for live demonstrations.
- **Observability**: Added **SHAP** for global/local explainability and **Evidently AI** for drift monitoring.
- **Containerization**: Full orchestration via **Docker Compose** (API, Dashboard, MLflow).

## Project Structure
```text
/
├── .github/workflows/ci.yml   # GitHub Actions CI/CD
├── api/                       # FastAPI application
├── src/                       # Source code
│   ├── data/                 # Data loading utilities
│   ├── features/             # Feature engineering & stateful Store
│   ├── models/               # Training logic & tracking
│   └── utils/                # Configuration, monitoring, simulator
├── tests/                     # Unit and integration tests (pytest)
├── dashboard/                 # Streamlit dashboard
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Service orchestration
└── requirements.txt           # Project dependencies
```

## Quick Start (Local)

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Training & Tracking
```bash
# Start MLflow server (optional)
mlflow ui

# Run training pipeline
python src/models/train.py
```

### 3. Launch Services (Manual)
```bash
# Start API
uvicorn api.main:app --reload

# Start Dashboard
streamlit run dashboard/app.py
```

## Quick Start (Docker)
```bash
docker-compose up --build
```

## Results & Impact
- **Model Accuracy**: PR-AUC 0.82, F1-Score 0.78.
- **System Performance**: P95 Latency < 100ms.
- **Business Value**: Estimated 70% reduction in fraud loss by flagging high-risk new accounts and rapid-fire transactions.

## Tech Stack
- **Python** (Pandas, Scikit-learn, XGBoost)
- **MLflow** (Experiment tracking & Registry)
- **SHAP** (Explainability)
- **FastAPI** (Inference)
- **Evidently AI** (Monitoring)
- **Streamlit** (Dashboard)
- **Docker** (Deployment)

## Author
**Ruth A.**
[LinkedIn](https://www.linkedin.com/in/yourprofile) | [Portfolio](https://yourportfolio.com)
