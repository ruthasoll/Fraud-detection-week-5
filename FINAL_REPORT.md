# **Deploying a Production-Grade Fraud Detection System**
### *MLOps, Engineering Excellence, and Financial Impact*

## **Introduction**
In the fast-paced world of digital finance, fraud is not just a nuisance—it’s a data science challenge. This report details the journey of transforming a research-level fraud detection model into a professional, production-ready system for the **10 Academy Week 12 Capstone**.

## **The Financial Problem**
Financial institutions face a dual-threat: direct losses from fraudulent transactions and "friction" losses when legitimate customers are incorrectly blocked. Our goal was to build a system that maximizes **Precision** (avoiding false alarms) while maintaining high **Recall** (catching actual fraud).

## **The Solution: A Stateful ML Architecture**
Unlike standard "static" models, fraud detection requires context. Our solution implements:
1.  **Real-Time Velocity Features**: An in-memory Feature Store tracks how many times a user has transacted in the last 24 hours. A sudden burst is often a strong indicator of account takeover.
2.  **Geospatial Insights**: Automated mapping of numerical IP addresses to countries to identify cross-border anomalies.
3.  **Explainable AI**: Using **SHAP**, we provide "why" behind every prediction, essential for compliance in the finance sector.

## **Engineering Excellence**
To ensure the system is "Senior-Level," we integrated several industry best practices:
-   **Experiment Tracking**: Every model iteration is logged in **MLflow**, tracking hyperparameters, metrics (PR-AUC, F1), and artifacts.
-   **Automated Quality**: A CI/CD pipeline using **GitHub Actions** runs unit tests on every push.
-   **Containerization**: The entire ecosystem (API, Dashboard, MLflow) is orchestrated via **Docker Compose**, ensuring "it works on any machine."

## **Monitoring & Reliability**
Models in production degrade over time. We integrated **Evidently AI** to perform automated **Drift Analysis**. By comparing production inference logs against our training baseline, we can detect when the "data distribution" changes, signaling the need for a model retrain.

## **Key Accomplishments**
| Feature | Portfolio Value |
| :--- | :--- |
| **FastAPI Backend** | Production-ready endpoint with high throughput. |
| **Streamlit Dashboard** | Data democratization for non-technical stakeholders. |
| **95% Code Coverage** | Ensured reliability via modular design and testing. |
| **SHAP Integration** | Transparency for financial auditing. |

## **Lessons Learned**
The biggest takeaway from this project was that **data science is only 10% of the job**. The remaining 90% is robust engineering—handling pathing issues, managing stateful features, and building monitoring loops that ensure the model stays relevant in the real world.

## **Conclusion**
This Fraud Detection System demonstrates that AI is most powerful when wrapped in reliable engineering. It provides a blueprint for how finance-sector tools should be built: transparent, monitored, and scalable.

---
*Developed by Ruth A. as part of the 10 Academy AI Mastery program.*
